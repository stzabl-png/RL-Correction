#!/usr/bin/env python
"""Custom Stage-A grasp-trajectory generator for an egodex take.

Scene matches the retarget sim (retarget_isaacsim.py): the object keeps its
RECONSTRUCTED orientation (global scene-rot only, identity for z-up egodex) and
is dropped onto the table; the hand starts from the reconstructed initial pose
(ref_qpos if present, else derived from replay_world.npz MANO joints) and is
placed by the SAME transform, so the reconstructed hand-object geometry -- and
the hand's above-the-table approach -- is preserved. cuRobo then drives the open
hand to the grasp, fingers close, and the wrist lifts +Z.

NB: this deliberately does NOT apply a per-object stable-pose projection -- that
rotated the object per-mesh and dragged the hand under the table. The
stable_pose_projection helper below is kept only for reference.

Everything is authored in the z-up WORLD frame; Stage B gets an identity DexYCB
manifest so its "camera frame" == this world frame.
"""
import argparse, json
import sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import glob as _glob

from ocir.grasp_synthesis.assets import load_sharpa_wave_right
from ocir.grasp_traj.planner import TransitPlanner
from ocir.grasp_traj.trajectory_schema import (
    GraspTrajectory, SEGMENT_RETARGET, SEGMENT_APPROACH, SEGMENT_CLOSE,
    SEGMENT_SQUEEZE, SEGMENT_CARRY,
)
from curobo._src.types.device_cfg import DeviceCfg
import torch


# ---------------- quaternion helpers (wxyz, Hamilton) ----------------
def quat_to_rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0); return q / np.linalg.norm(q)
    th = np.arccos(d); s = np.sin(th)
    return (np.sin((1 - t) * th) / s) * q0 + (np.sin(t * th) / s) * q1


# ---------------- stable-pose projection (RL_Correction frames.py) ----------------
def support_faces(verts, min_area=0.002):
    hull = ConvexHull(verts)
    buckets = {}
    for tri in hull.simplices:
        v0, v1, v2 = verts[tri]
        n = np.cross(v1 - v0, v2 - v0); a = 0.5 * np.linalg.norm(n)
        if a < 1e-12: continue
        n = n / np.linalg.norm(n)
        key = tuple(np.round(n, 1))
        if key not in buckets: buckets[key] = [0.0, np.zeros(3)]
        buckets[key][0] += a; buckets[key][1] += a * n
    out = []
    for _, (area, acc) in buckets.items():
        if area > min_area:
            out.append((area, acc / np.linalg.norm(acc)))
    out.sort(key=lambda x: -x[0])
    return out


def stable_pose_projection(verts, q0):
    faces = support_faces(verts)
    if not faces: return q0.copy(), 0.0
    R = quat_to_rotmat(q0); down = np.array([0, 0, -1.0])
    best, bestang = None, None
    for _, n in faces:
        ang = np.arccos(np.clip(np.dot(R @ n, down), -1, 1))
        if bestang is None or ang < bestang: bestang, best = ang, n
    v = R @ best; axis = np.cross(v, down); s = np.linalg.norm(axis); c = np.dot(v, down)
    if s < 1e-8: return q0.copy(), 0.0
    angle = np.arctan2(s, c); axis = axis / s
    qfix = np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])
    return quat_mul(qfix, q0), np.degrees(angle)


def longest_true_run_start(mask):
    best_len = 0; best_start = 0; cur = 0; start = 0
    for i, v in enumerate(mask):
        if v:
            if cur == 0: start = i
            cur += 1
            if cur > best_len: best_len, best_start = cur, start
        else:
            cur = 0
    return best_start


def reorder(values, from_names, to_names):
    idx = {n: i for i, n in enumerate(from_names)}
    return np.array([values[idx[n]] for n in to_names], dtype=float)


def is_overclose(name):
    """Flexion joints driven past the grasp pose to grip tighter: MCP-FE (incl.
    thumb CMC/MCP-FE), PIP, thumb IP. NOT the fingertip DIP (driving the distal-
    most joint deeper rolls the fingertip off a convex surface for ~0 force)."""
    return ("_FE" in name) or ("_PIP" in name) or name.endswith("_IP")


def is_flexion(name):
    return any(tok in name for tok in ("_FE", "_PIP", "_DIP")) or name.endswith("_IP")


def select_table_clearing_grasp(synth_dir, joint_order, up_obj, table_level, asset, device_cfg, margin=0.005):
    """Pick the best force-closure grasp whose hand CLEARS the table. FK all 37
    hand spheres for each candidate (object frame), measure the lowest sphere's
    height above the object-frame tabletop plane; prefer grasps that clear it
    (>= -margin), lowest grasp_error among them. If none clear, take the least-
    dipping one. Returns (path, grasp_error, clearance_m)."""
    import torch
    from ocir.grasp_synthesis.clearance import ClearanceChecker
    cc = ClearanceChecker(asset, device_cfg)
    radii = cc.radii.detach().cpu().numpy()
    up = np.asarray(up_obj, dtype=float) / (np.linalg.norm(up_obj) + 1e-12)
    cands = []
    files = sorted(_glob.glob(str(Path(synth_dir) / "grasp_*.json"))) + \
        sorted(_glob.glob(str(Path(synth_dir) / "failed_grasp_*.json")))
    for p in files:
        g = json.loads(Path(p).read_text())
        a = np.asarray(g["action"], dtype=float)
        jn = g["joint_names"]; idx = {n: i for i, n in enumerate(jn)}
        full = np.zeros(7 + len(joint_order))
        full[:7] = a[:7]
        for i, n in enumerate(joint_order):
            full[7 + i] = a[7 + idx[n]]
        ft = torch.tensor(full[None], device="cuda", dtype=torch.float32)
        c = cc.sphere_world_positions(ft)[0].detach().cpu().numpy()
        clr = float(((c @ up - radii) - table_level).min())
        cands.append((p, float(g.get("grasp_error_max", 1e9)), clr))
    clearing = [c for c in cands if c[2] >= -margin]
    if clearing:
        best = min(clearing, key=lambda c: c[1])
    else:
        best = max(cands, key=lambda c: c[2])
    return best, cands


def sharpa_base_quat_from_joints(joints21):
    """MANO 21 joints (21,3) -> SharpaWave floating-base quat (wxyz), matching
    RL_Correction frames.sharpa_base_quat_from_joints / retarget base convention:
    +z = wrist->four-finger-MCP centroid, +y = index-MCP->pinky-MCP, +x = y x z."""
    j = np.asarray(joints21, dtype=float)
    w = j[0]
    z = j[[5, 9, 13, 17]].mean(0) - w
    z /= np.linalg.norm(z) + 1e-9
    radial = j[5] - j[17]
    y = radial - np.dot(radial, z) * z
    y /= np.linalg.norm(y) + 1e-9
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=-1)  # columns are basis vectors
    return rotmat_to_quat(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence-dir", required=True)
    ap.add_argument("--grasp-json", default=None,
                    help="A single grasp record. Ignored if --synthesis-dir is given.")
    ap.add_argument("--synthesis-dir", default=None,
                    help="Directory of grasp candidates; the best force-closure grasp that CLEARS the "
                         "table (given the object's on-table placement) is auto-selected.")
    ap.add_argument("--replay-npz", required=True)
    ap.add_argument("--refqpos-npz", default=None,
                    help="Optional retargeted SharpaWave qpos. If absent, the init hand pose "
                         "is derived from replay_world.npz MANO joints.")
    ap.add_argument("--recon-mesh", default=None, help="full mesh for stable-pose support faces (default: sequence object.obj)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--asset-config", default="assets/robots/hands/sharpa_wave/sharpa_wave_right.yml")
    ap.add_argument("--table-z", type=float, default=0.85)
    ap.add_argument("--gap", type=float, default=0.01)
    ap.add_argument("--scene-rot", default="1,0,0,0",
                    help="Global scene rotation (quat wxyz), matching retarget_isaacsim --scene-rot. "
                         "egodex replay_world is already z-up so the default identity is correct.")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--open-seconds", type=float, default=0.5)
    ap.add_argument("--approach-seconds", type=float, default=2.0)
    ap.add_argument("--preapproach-backoff", type=float, default=0.10)
    ap.add_argument("--to-grasp-seconds", type=float, default=0.7)
    ap.add_argument("--close-seconds", type=float, default=1.0)
    ap.add_argument("--settle-seconds", type=float, default=1.0)
    ap.add_argument("--squeeze-overclose", type=float, default=0.0,
                    help="Radians the flexion fingers are driven PAST the grasp pose after reaching it, "
                         "held through squeeze+carry, so the soft drives keep a firm grip instead of "
                         "relaxing (reduces slip). 0 = hold the grasp pose exactly. ~0.12 is a firm grip.")
    ap.add_argument("--overclose-seconds", type=float, default=0.3,
                    help="Time to blend from the grasp pose to the overclosed grip.")
    ap.add_argument("--carry-lift-height", type=float, default=0.10)
    ap.add_argument("--carry-lift-seconds", type=float, default=2.0)
    ap.add_argument("--carry-hold-seconds", type=float, default=1.0)
    ap.add_argument("--max-speed", type=float, default=0.25)
    args = ap.parse_args()

    fps = args.fps; dt = 1.0 / fps
    seq_dir = Path(args.sequence_dir)
    obj_mesh_path = seq_dir / "object.obj"

    asset = load_sharpa_wave_right(args.asset_config)
    joint_order = list(asset.config["joint_order"])
    limits = asset.config["joint_limits"]

    replay = np.load(args.replay_npz, allow_pickle=True)
    # ref_qpos.npz (retargeted SharpaWave wrist+finger) is OPTIONAL. When absent
    # we derive the reconstructed initial hand pose directly from replay_world's
    # MANO joints -- so the hand-object relative start ALWAYS comes from the
    # reconstruction, whether or not the retarget export was run.
    refq = None
    if args.refqpos_npz and Path(args.refqpos_npz).exists():
        refq = np.load(args.refqpos_npz, allow_pickle=True)
        valid = refq["valid"].astype(bool)
    else:
        valid = replay["valid_right"].astype(bool)
    t0 = longest_true_run_start(valid)
    print(f"[gen] init-pose source = {'ref_qpos' if refq is not None else 'replay_world MANO (derived)'}; t_init = {t0}")

    obj_pose = replay["obj_pose"]  # (T,7) [x,y,z, qw,qx,qy,qz]
    q0 = np.asarray(obj_pose[t0, 3:7], dtype=float)
    pos0 = np.asarray(obj_pose[t0, 0:3], dtype=float)

    mesh_for_support = trimesh.load(args.recon_mesh or str(obj_mesh_path), force="mesh", process=False)
    verts = np.asarray(mesh_for_support.vertices, dtype=float)
    # Match the retarget scene (retarget_isaacsim.py): keep the object's RECONSTRUCTED
    # orientation (apply only the global scene-rot, identity for z-up egodex) and drop it
    # onto the table -- NO per-object stable-pose projection. The hand and object share the
    # same transform so the reconstructed hand-object geometry (and the hand's above-the-table
    # position) is preserved. (Stable-pose rotated the object per-mesh, which dragged the hand
    # under the table.)
    scene_q = np.array([float(x) for x in args.scene_rot.split(",")], dtype=float)
    scene_q = scene_q / (np.linalg.norm(scene_q) + 1e-12)
    R_recon = quat_to_rotmat(q0)                 # reconstructed object mesh->world at t_init
    q_init = quat_mul(scene_q, q0)               # object world orientation (scene-rotated recon)
    R_init = quat_to_rotmat(q_init)
    v_rot = verts @ R_init.T
    z_init = args.table_z + args.gap - float(v_rot[:, 2].min())
    t_wo = np.array([0.0, 0.0, z_init])          # recenter object frame-0 to the table center
    print(f"[gen] scene-rot={args.scene_rot} (retarget-matched; no stable-pose); "
          f"object world quat (wxyz) = {np.round(q_init,4).tolist()}")

    # reconstructed hand init wrist pose (world) -> object frame
    if refq is not None:
        wrist_pos0 = np.asarray(refq["wrist_pos"][t0], dtype=float)
        wrist_quat0 = np.asarray(refq["wrist_quat_wxyz"][t0], dtype=float)
        ref_jn = [str(n) for n in refq["joint_names"]]
        init_finger_ref = reorder(np.asarray(refq["finger_qpos"][t0], dtype=float), ref_jn, joint_order)
        init_finger_ref = np.clip(init_finger_ref, [limits[n][0] for n in joint_order], [limits[n][1] for n in joint_order])
    else:
        joints_right = np.asarray(replay["joints_right"], dtype=float)  # (T,21,3) MANO, world
        wrist_pos0 = joints_right[t0, 0]
        wrist_quat0 = sharpa_base_quat_from_joints(joints_right[t0])
        init_finger_ref = None  # no retargeted fingers -> start from the open pregrasp pose
    # Hand init RELATIVE to the RECONSTRUCTED object (mesh frame), so the hand-object geometry
    # is exactly as reconstructed. Using the reconstructed pose (q0,pos0) -- not the table
    # placement -- keeps the vertical hand->object offset from the demo (hand above the object).
    init_wrist_pos_obj = R_recon.T @ (wrist_pos0 - pos0)
    init_wrist_quat_obj = quat_mul(quat_conj(q0), wrist_quat0)

    # grasp (object frame): auto-select the best table-clearing candidate, or use --grasp-json
    if args.synthesis_dir:
        up_obj = R_recon.T @ np.array([0.0, 0.0, 1.0])          # table up in object frame
        table_level = float((verts @ up_obj).min())             # object rests on the table here
        device_cfg_sel = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        (best_path, best_ge, best_clr), cands = select_table_clearing_grasp(
            args.synthesis_dir, joint_order, up_obj, table_level, asset, device_cfg_sel)
        n_clear = sum(1 for c in cands if c[2] >= -0.005)
        print(f"[gen] table-clearing grasp selection: {n_clear}/{len(cands)} candidates clear the table; "
              f"picked {Path(best_path).name} (grasp_err={best_ge:.4f}, clearance={best_clr*1000:.1f}mm)")
        args.grasp_json = best_path
    if not args.grasp_json:
        print("error: provide --grasp-json or --synthesis-dir", file=sys.stderr)
        return 2
    grasp = json.loads(Path(args.grasp_json).read_text())
    action = np.asarray(grasp["action"], dtype=float)
    grasp_wrist_pos_obj = action[:3]
    grasp_wrist_quat_obj = action[3:7]
    grasp_jn = grasp["joint_names"]
    grasp_finger = reorder(action[7:], grasp_jn, joint_order)
    grasp_finger = np.clip(grasp_finger, [limits[n][0] for n in joint_order], [limits[n][1] for n in joint_order])

    # open (pregrasp) pose: zero the flexion channels, keep spread/thumb-orientation from the grasp
    open_finger = grasp_finger.copy()
    for i, n in enumerate(joint_order):
        if is_flexion(n):
            open_finger[i] = max(0.0, limits[n][0])
    open_by_name = {n: float(open_finger[i]) for i, n in enumerate(joint_order)}

    # squeeze (grip) pose: grasp pose driven deeper on the flexion joints so the soft
    # drives keep pressing after contact (held through squeeze + carry). Grip force is
    # bounded by the finger effort caps, so a deeper target just holds firmer, not crush.
    upper = np.asarray([limits[n][1] for n in joint_order], dtype=float)
    lower = np.asarray([limits[n][0] for n in joint_order], dtype=float)
    squeeze_finger = grasp_finger.copy()
    if args.squeeze_overclose > 0.0:
        for i, n in enumerate(joint_order):
            if is_overclose(n):
                squeeze_finger[i] = grasp_finger[i] + args.squeeze_overclose
        squeeze_finger = np.clip(squeeze_finger, lower, upper)

    # initial fingers: reconstructed retarget pose if available, else the open pregrasp
    init_finger = init_finger_ref if init_finger_ref is not None else open_finger.copy()

    # preapproach in object frame: back the grasp wrist off radially from the object center
    obj_center = verts.mean(axis=0)
    approach_dir = grasp_wrist_pos_obj - obj_center
    approach_dir = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)
    preapproach_pos_obj = grasp_wrist_pos_obj + approach_dir * args.preapproach_backoff
    preapproach_quat_obj = grasp_wrist_quat_obj

    # --- cuRobo approach: init_wrist -> preapproach (object frame, fingers locked open) ---
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    planner = TransitPlanner(asset, open_by_name, str(obj_mesh_path.resolve()), device_cfg)
    print("[gen] planning cuRobo approach init -> preapproach ...")
    plan = planner.plan(
        init_wrist_pos_obj, init_wrist_quat_obj,
        preapproach_pos_obj, preapproach_quat_obj,
        seconds=args.approach_seconds, fps=fps, max_speed_mps=args.max_speed,
    )
    if plan is None:
        print("[gen] cuRobo FAILED -> linear fallback for approach")
        n_ap = max(2, int(args.approach_seconds * fps))
        ts = np.linspace(0, 1, n_ap + 1)[1:]
        ap_pos = np.array([init_wrist_pos_obj + t * (preapproach_pos_obj - init_wrist_pos_obj) for t in ts])
        ap_quat = np.array([slerp(init_wrist_quat_obj, preapproach_quat_obj, t) for t in ts])
        planner_used = "linear"
    else:
        ap_pos, ap_quat = plan
        planner_used = "curobo"
    print(f"[gen] approach planned ({planner_used}), {len(ap_pos)} steps")

    # ---------------- assemble object-frame segments ----------------
    seg_pos, seg_quat, seg_fing, seg_lbl = [], [], [], []

    def push(p, q, f, lbl):
        seg_pos.append(np.asarray(p)); seg_quat.append(np.asarray(q))
        seg_fing.append(np.asarray(f)); seg_lbl.append(lbl)

    # 1) open at the reconstructed init wrist (fingers reconstructed -> open)
    n_open = max(2, int(args.open_seconds * fps))
    for k in range(n_open):
        s = (k + 1) / n_open
        push(init_wrist_pos_obj, init_wrist_quat_obj, (1 - s) * init_finger + s * open_finger, SEGMENT_RETARGET)
    switch_idx = len(seg_pos)

    # 2) cuRobo approach init -> preapproach (fingers open)
    for p, q in zip(ap_pos, ap_quat):
        push(p, q, open_finger, SEGMENT_APPROACH)

    # 3) preapproach -> grasp wrist (fingers still open)
    n_g = max(2, int(args.to_grasp_seconds * fps))
    for k in range(n_g):
        s = (k + 1) / n_g
        push(preapproach_pos_obj + s * (grasp_wrist_pos_obj - preapproach_pos_obj),
             slerp(preapproach_quat_obj, grasp_wrist_quat_obj, s), open_finger, SEGMENT_CLOSE)

    # 4) close fingers open -> grasp at the grasp wrist
    n_c = max(2, int(args.close_seconds * fps))
    for k in range(n_c):
        s = (k + 1) / n_c
        push(grasp_wrist_pos_obj, grasp_wrist_quat_obj, (1 - s) * open_finger + s * grasp_finger, SEGMENT_CLOSE)

    # 4b) overclose: drive the flexion fingers past the grasp pose to grip tighter
    if args.squeeze_overclose > 0.0:
        n_oc = max(2, int(args.overclose_seconds * fps))
        for k in range(n_oc):
            s = (k + 1) / n_oc
            push(grasp_wrist_pos_obj, grasp_wrist_quat_obj, (1 - s) * grasp_finger + s * squeeze_finger, SEGMENT_CLOSE)

    # 5) squeeze/settle hold at the (over)closed grip
    n_s = max(2, int(args.settle_seconds * fps))
    for _ in range(n_s):
        push(grasp_wrist_pos_obj, grasp_wrist_quat_obj, squeeze_finger, SEGMENT_SQUEEZE)

    seg_pos = np.array(seg_pos); seg_quat = np.array(seg_quat)
    seg_fing = np.array(seg_fing); seg_lbl = np.array(seg_lbl, dtype=np.int8)

    # ---------------- transform object-frame wrist poses to world ----------------
    hand_pos_w = (R_init @ seg_pos.T).T + t_wo
    hand_quat_w = np.array([quat_mul(q_init, q) for q in seg_quat])

    # ---------------- carry: lift +Z in world (cosine ease) then hold ----------------
    grasp_pos_w = hand_pos_w[-1].copy()
    grasp_quat_w = hand_quat_w[-1].copy()
    n_lift = max(2, int(args.carry_lift_seconds * fps))
    lift_pos, lift_quat, lift_fing, lift_lbl = [], [], [], []
    for k in range(n_lift):
        s = (k + 1) / n_lift
        ease = 0.5 * (1 - np.cos(np.pi * s))  # 0->1 cosine, zero boundary velocity
        p = grasp_pos_w.copy(); p[2] += args.carry_lift_height * ease
        lift_pos.append(p); lift_quat.append(grasp_quat_w.copy())
        lift_fing.append(squeeze_finger.copy()); lift_lbl.append(SEGMENT_CARRY)
    n_hold = max(1, int(args.carry_hold_seconds * fps))
    top = grasp_pos_w.copy(); top[2] += args.carry_lift_height
    for _ in range(n_hold):
        lift_pos.append(top.copy()); lift_quat.append(grasp_quat_w.copy())
        lift_fing.append(squeeze_finger.copy()); lift_lbl.append(SEGMENT_CARRY)

    hand_pos_w = np.concatenate([hand_pos_w, np.array(lift_pos)], axis=0)
    hand_quat_w = np.concatenate([hand_quat_w, np.array(lift_quat)], axis=0)
    fing_all = np.concatenate([seg_fing, np.array(lift_fing)], axis=0)
    lbl_all = np.concatenate([seg_lbl, np.array(lift_lbl, dtype=np.int8)], axis=0)

    T = hand_pos_w.shape[0]
    # object reference pose (world) held at the stable pose for every frame
    obj_pos_w = np.tile(t_wo, (T, 1))
    obj_quat_w = np.tile(q_init, (T, 1))

    # grasp_root_tf: world grasp wrist 4x4
    grasp_tf = np.eye(4); grasp_tf[:3, :3] = quat_to_rotmat(grasp_quat_w); grasp_tf[:3, 3] = grasp_pos_w

    traj = GraspTrajectory(
        hand_pos_camera=hand_pos_w.astype(np.float64),
        hand_quat_camera=hand_quat_w.astype(np.float64),
        finger_targets=fing_all.astype(np.float64),
        object_pos_camera=obj_pos_w.astype(np.float64),
        object_quat_camera=obj_quat_w.astype(np.float64),
        segment=lbl_all.astype(np.int8),
        dt=dt,
        joint_order=tuple(joint_order),
        grasp_json=str(args.grasp_json),
        sequence_dir=str(seq_dir),
        switch_frame_index=int(switch_idx),
        grasp_root_tf=grasp_tf.tolist(),
        clearance_report={"scene_rot": args.scene_rot, "approach_planner": planner_used},
        extra_metadata={
            "object_mesh": str(obj_mesh_path.resolve()),
            "object_name": "object",
            "world_frame": "rl_correction_z_up",
            "table_z": float(args.table_z),
        },
    )
    traj.save(args.out_dir)
    print(f"[gen] wrote trajectory: {T} steps -> {args.out_dir}")
    print(f"[gen] segments: retarget={int((lbl_all==SEGMENT_RETARGET).sum())} approach={int((lbl_all==SEGMENT_APPROACH).sum())} "
          f"close={int((lbl_all==SEGMENT_CLOSE).sum())} squeeze={int((lbl_all==SEGMENT_SQUEEZE).sum())} carry={int((lbl_all==SEGMENT_CARRY).sum())}")


if __name__ == "__main__":
    main()
