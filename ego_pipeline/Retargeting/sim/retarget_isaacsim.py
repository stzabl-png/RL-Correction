#!/usr/bin/env python
"""Bare Isaac Sim (NO Isaac Lab) replay of a reconstructed hand+object sequence
onto the SharpaWave hand.

Input is one trajectory .npz (e.g. from scripts/hawor_to_joints.py, extended with
an object pose stream). EITHER single-hand or two-hand keys:
    hand_joints : (T,21,3)  single-hand, world meters, OpenPose/MediaPipe order
    joints_left / joints_right : (T,21,3)  two-hand (loads whichever are present)
    obj_pose    : (T,7)    float, world: [x,y,z, qw,qx,qy,qz]
    fps         : scalar                                   (optional, default 30)
    wrist_pos[_left|_right]  : (T,3)    float   (optional; else from joints[:,0])
    wrist_quat[_left|_right] : (T,4)    float wxyz (optional; else operator frame)

Hand fingers are driven by magicdexmate retarget (MANO->21 joints already done
upstream; here joints -> SharpaWave 22-DOF qpos). Each hand BASE (floating) is
placed each frame at its wrist pose, so the hand travels through space with the
reconstructed trajectory (this is why it is a new script, not teleop_isaac_single
which has a fixed base).

Playback is INTERACTIVE by default: the scene opens with hands + object at frame
0; press ENTER to play one pass, then ENTER again to replay (q+ENTER quits). Use
--auto for non-interactive (one pass, or continuous with --loop). The given hand
and object tracks are drawn as 3D lines (disable with --no-traj).

Two modes:
  --mode render   kinematic: fingers set_joint_positions, base + OBJECT both
                  posed from their trajectories every frame. Gravity off. The
                  object follows its own trajectory regardless of contact. For
                  RGB/seg/pose data generation.
  --mode physics  dynamic object: fingers driven as PD targets, base teleported
                  along the wrist trajectory, the object is a rigid body placed
                  at obj_pose[0] then left to physics. Whether SharpaWave actually
                  grasps/lifts it is decided by contact. obj_pose[1:] is unused.

Run (first launch accepts the EULA):
  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/retarget_isaacsim.py \
      --traj replay.npz --object-usd chair_027.usd --hand right \
      --mode render --headless --duration 0
"""
import argparse
import os
import sys

# ---- args ------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--traj", required=True, help="trajectory .npz")
parser.add_argument("--object-usd", default=None, nargs="+",
                    help="object USD(s) (see scripts/obj_to_usd.py). Several may be given "
                         "for a paired scene (bottle + cap): they are matched in order to "
                         "obj_pose_all in the trajectory npz. All objects share one scene "
                         "transform, so their relative geometry is preserved.")
parser.add_argument("--no-object", action="store_true",
                    help="replay hands + hand tracks only (no object mesh, no amber track); "
                         "the object pose is still used for gravity/recenter")
parser.add_argument("--hand", choices=["right", "left"], default="right")
parser.add_argument("--hand-usd", default=None, help="hand USD (default: repo converted USD for --hand)")
parser.add_argument("--mode", choices=["render", "physics"], default="render")
parser.add_argument("--retarget-mode", choices=["vector", "dexpilot"], default="dexpilot")
parser.add_argument("--fps", type=float, default=30.0, help="used if traj has no fps")
parser.add_argument("--loop", action="store_true")
parser.add_argument("--duration", type=float, default=0.0, help="extra physics seconds after replay (physics mode)")
parser.add_argument("--collision", choices=["convexHull", "convexDecomposition", "none"],
                    default="convexHull", help="object collision approximation (physics mode)")
parser.add_argument("--obj-mass", type=float, default=0.2, help="object mass kg (physics mode)")
parser.add_argument("--log-contact", action="store_true",
                    help="(physics) log per-frame hand<->object contact impulse + object speed to "
                         "tell compression (impulse>0) from pass-through (overlap, impulse~0) from "
                         "blow-up (speed spike); also writes a CSV next to --traj")
parser.add_argument("--max-depen-vel", type=float, default=0.5,
                    help="(physics) object max depenetration velocity m/s; low value stops a deep "
                         "teleport-penetration from launching the object")
parser.add_argument("--solver-pos-iter", type=int, default=32,
                    help="(physics) object solver position iterations (higher = more stable contact)")
# scene / placement
parser.add_argument("--table", action=argparse.BooleanOptionalAction, default=True,
                    help="add a square table (默认开: 下游要放机器人, 没有桌子无从摆放; --no-table 关闭)")
parser.add_argument("--table-size", type=float, default=4.0, help="table side length (m)")
parser.add_argument("--table-height", type=float, default=0.85, help="table top surface height (m)")
parser.add_argument("--table-thickness", type=float, default=0.04, help="tabletop slab thickness (m)")
parser.add_argument("--scene-rot", default="1,0,0,0",
                    help="global scene rotation applied to hand+object: quat wxyz, 'cv2zup' "
                         "(level camera -> z-up), or 'hoi4d' (estimate gravity from the object "
                         "lift; correct for HOI4D's downward-looking camera)")
parser.add_argument("--smooth", type=int, default=1,
                    help="moving-average window (frames) on hand/object positions; 1=off")
parser.add_argument("--recenter", choices=["none", "table"], default="table",
                    help="translate so the object's frame-0 lands at the table center "
                         "(默认 table: 与 rl_rebuild/correction/frames.py align_replay 同一约定, "
                         "手和物体施加**同一个**变换, 相对几何不变)")
parser.add_argument("--obj-place", choices=["traj", "grip"], default="traj",
                    help="traj = object follows obj_pose; grip = ignore object trajectory and "
                         "place the object statically at the two-hand grip midpoint (physics decides)")
parser.add_argument("--obj-lift", type=float, default=0.01,
                    help="gap between the object's bottom and the table top at frame 0 (m); a "
                         "1cm default lets it settle cleanly (0 = exactly coplanar can tunnel the "
                         "thin tabletop; large values drop it from a height and topple tall objects).")
parser.add_argument("--finger-kp", type=float, default=20.0)
parser.add_argument("--finger-kd", type=float, default=2.0)
# wrist -> robot-base calibration (tune from snapshots). Offset is applied in the
# wrist frame: base_quat = wrist_quat * rot_offset; base_pos = wrist_pos + pos_offset.
parser.add_argument("--wrist-pos-offset", default="0,0,0", help="x,y,z meters")
parser.add_argument("--wrist-rot-offset", default="1,0,0,0", help="quat wxyz")
parser.add_argument("--base", choices=["joints", "operator"], default="joints",
                    help="floating-hand base orientation: 'joints' = build from joints in the "
                         "SharpaWave convention (+z fingers,+y thumb,+x palm); 'operator' = legacy")
parser.add_argument("--palm-flip", default="left",
                    help="comma sides to rotate the base 180 deg about the fingers axis "
                         "(SharpaWave left model is mirrored, so its palm needs this; "
                         "default 'left'. Use '' to disable, or 'left,right' for both)")
parser.add_argument("--no-base-motion", action="store_true", help="freeze base at frame-0 wrist pose")
# ★ --record 已于 2026-08-14 删除: 无头录像的 /World/snap_cam 位姿设不上去(set_world_pose
#   与 set_camera_view 都试过), 录出来是只有网格地板的空场景 —— 一个看起来能用、实际
#   永远产出废片的功能。GUI 走的是**视口相机**(第 711 行 set_camera_view), 不受影响。
#   要录像请用屏幕录制, 或先修好 snap_cam 再重新引入。
parser.add_argument("--snap-focal", "--record-focal", dest="record_focal", type=float, default=2.4,
                    help="录像相机焦距(Isaac 单位)。默认 2.4 是很广的视角; 想拉近放大到 4~6")
parser.add_argument("--snap-dir", default=None, help="save RGB snapshots here")
parser.add_argument("--snap-times", default="", help="comma sim-seconds to snap, e.g. 0.0,1.5,3.0")
parser.add_argument("--cam-view", choices=["topdown", "iso", "manual"], default="topdown",
                    help="topdown=egocentric look-down over the hands; iso=3/4 view; "
                         "manual=use --cam-eye/--cam-target")
parser.add_argument("--cam-eye", default="1.8,-1.8,1.8", help="manual camera eye x,y,z (m)")
parser.add_argument("--cam-target", default="0.0,0.0,1.0", help="manual camera look-at x,y,z (m)")
parser.add_argument("--cam-dist", type=float, default=1.5, help="topdown camera height above the workspace (m)")
parser.add_argument("--floor-usd",
                    default=os.path.expanduser(
                        "~/Project/Reconstruct_and_Retarget/third_party/mano2gripper/Assets/Scene/"
                        "Collected_default_environment/default_environment.usd"),
                    help="grid floor USD (mano2gripper default_environment); '' = plain ground plane")
parser.add_argument("--hand-color", default="0.5,0.5,0.5", help="hand RGB 0..1 (gray)")
parser.add_argument("--auto", action="store_true",
                    help="play automatically without waiting for ENTER (default: interactive)")
parser.add_argument("--no-traj", action="store_true", help="do not draw hand/object trajectory lines")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_THIS)
sys.path.insert(0, REPO)

# ---- retarget + trajectory: BEFORE Kit (pinocchio/boost vs Kit/USD clash) ---
import numpy as np  # noqa: E402

from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value  # noqa: E402
from magicdexmate.retarget.frames import estimate_frame_from_hand_points, to_mano  # noqa: E402
from magicdexmate.retarget.mapping import JointMapper  # noqa: E402


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def estimate_up_rotation(obj_xyz):
    """Camera->world rotation for HOI4D-style downward cameras. The object's lift is a
    reversible 'up then down' motion; removing the start->end linear drift (horizontal
    carry) and taking the top PCA axis of the residual isolates the gravity/up axis in
    the camera frame, independent of how the camera is tilted."""
    a = np.asarray(obj_xyz, np.float64)
    n = len(a)
    tt = np.linspace(0.0, 1.0, n)[:, None]
    resid = a - (a[0] * (1 - tt) + a[-1] * tt)
    _, _, Vt = np.linalg.svd(resid - resid.mean(0), full_matrices=False)
    up = Vt[0]
    proj = resid @ up
    if proj[np.argmax(np.abs(proj))] < 0:
        up = -up
    u = up / np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0])               # camera optical axis -> world +Y (into scene)
    wy = fwd - (fwd @ u) * u
    wy = wy / np.linalg.norm(wy)
    wx = np.cross(wy, u)
    return np.stack([wx, wy, u])                   # rows = world x,y,z expressed in cam coords


def moving_average(a, w):
    """Centered moving average along axis 0 (edge-padded). w<=1 is a no-op."""
    if w <= 1:
        return a
    a = np.asarray(a, np.float64)
    pad = w // 2
    ap = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    ker = np.ones(w) / w
    flat = ap.reshape(ap.shape[0], -1)
    out = np.empty((a.shape[0], flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(flat[:, c], ker, mode="valid")[: a.shape[0]]
    return out.reshape(a.shape)


data = np.load(args.traj)
fps = float(data["fps"]) if "fps" in data else args.fps

# ---- which hands? data-driven: two-hand (joints_left/right) or legacy single --
hand_joints_raw = {}
if "joints_left" in data or "joints_right" in data:
    for side in ("left", "right"):
        if f"joints_{side}" in data:
            arr = np.asarray(data[f"joints_{side}"], np.float64)
            assert arr.ndim == 3 and arr.shape[1:] == (21, 3), arr.shape
            hand_joints_raw[side] = arr
else:
    arr = np.asarray(data["hand_joints"], np.float64)
    assert arr.ndim == 3 and arr.shape[1:] == (21, 3), arr.shape
    hand_joints_raw[args.hand] = arr

# clean HaWoR data quality: a hand's NaN frames mean "not tracked there" (validity is
# evidence-gated upstream). Keep any hand that has real frames — even a partially tracked
# one — and nearest-fill its NaN gaps so retarget SVD stays stable; only drop a hand that
# is genuinely absent (fewer than 2 tracked frames, e.g. a one-handed take).
for side in list(hand_joints_raw):
    j = hand_joints_raw[side]
    bad = np.isnan(j).any(axis=(1, 2))
    n_good = int((~bad).sum())
    if n_good < 2:
        print(f"[setup] dropping {side} hand: only {n_good} tracked frame(s) (absent)")
        del hand_joints_raw[side]
        continue
    if bad.any():
        good = np.where(~bad)[0]
        for i in np.where(bad)[0]:
            j[i] = j[good[np.argmin(np.abs(good - i))]]   # nearest valid frame
        print(f"[setup] {side} hand: filled {int(bad.sum())} NaN frame(s)")
        hand_joints_raw[side] = j
if not hand_joints_raw:
    raise SystemExit("all hands are NaN/invalid in this trajectory")
n_hands = len(hand_joints_raw)

T = max(len(a) for a in hand_joints_raw.values())
# obj_pose_all (n,T,7) when the reconstruction tracked several objects in one pass;
# obj_pose (T,7) is object 0 and stays the reference for gravity/recenter so results are
# comparable with single-object runs.
if "obj_pose_all" in data:
    obj_poses = [np.asarray(p, np.float64) for p in np.asarray(data["obj_pose_all"], np.float64)]
elif "obj_pose" in data:
    obj_poses = [np.asarray(data["obj_pose"], np.float64)]
else:
    obj_poses = [np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(np.float64)]
obj_pose = obj_poses[0]

# ---- global scene placement: rotate camera-frame data into a z-up world, then
# recenter the object onto the table. Applied to every hand + the object so their
# GT relative geometry is preserved. Finger retarget is invariant to this
# (to_mano re-centers), so only each hand BASE pose and object pose move. --------
if args.smooth > 1:
    for side in hand_joints_raw:
        hand_joints_raw[side] = moving_average(hand_joints_raw[side], args.smooth)
    obj_pose[:, :3] = moving_average(obj_pose[:, :3], args.smooth)
    print(f"[setup] smoothed hand+object positions, window={args.smooth} frames")

if args.scene_rot == "cv2zup":
    Rs = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # level cam(+X right,+Y down,+Z fwd) -> z-up
elif args.scene_rot == "hoi4d":
    Rs = estimate_up_rotation(obj_pose[:, :3])           # gravity from object lift (downward cam)
    print(f"[setup] scene_rot=hoi4d  estimated up(cam)={Rs[2].round(3).tolist()}")
elif args.scene_rot == "hands":
    wr = np.mean([hand_joints_raw[s][:, 0] for s in hand_joints_raw], axis=0)  # mean wrist track
    Rs = estimate_up_rotation(wr)                        # gravity from the hands' lift (no object)
    print(f"[setup] scene_rot=hands  estimated up(in)={Rs[2].round(3).tolist()}")
elif args.scene_rot == "obj0":
    # anchor on the OBJECT: rotate so the object's frame-0 orientation becomes canonical
    # (upright). Applied to object AND hands, so the hand-object relative geometry is kept
    # exactly; the absolute (noisy SLAM) scene is discarded. Robust when the object barely
    # moves (gravity-from-motion fails) and lets us just stand the object on the table.
    Rs = quat_to_rotmat(obj_pose[0, 3:7]).T
    print("[setup] scene_rot=obj0  object frame-0 anchored upright; hands kept relative to it")
elif args.scene_rot.startswith("up:"):
    # explicit gravity-up vector (in the npz world frame) -> world +Z. Use when the
    # reconstruction provides a gravity vector (e.g. from ViPE) or you measured one.
    u = np.array([float(x) for x in args.scene_rot[3:].split(",")], float)
    u = u / np.linalg.norm(u)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0.0, 1, 0])
    wx = a - (a @ u) * u; wx /= np.linalg.norm(wx)
    Rs = np.stack([wx, np.cross(u, wx), u])
    print(f"[setup] scene_rot=up  gravity-up(in)={u.round(3).tolist()}")
else:
    Rs = quat_to_rotmat(np.array([float(x) for x in args.scene_rot.split(",")]))
qs = rotmat_to_quat(Rs)
obj_pos_w = np.einsum("ij,tj->ti", Rs, obj_pose[:, :3])
obj_quat_w = np.stack([quat_mul(qs, obj_pose[i, 3:7]) for i in range(len(obj_pose))])
scene_shift = np.zeros(3)
if args.recenter == "table":
    # x,y: center the object's frame-0 on the table; z: rest the LOWEST point of the whole
    # object trajectory on the table, so the object sits on the table at its resting moment
    # and lifts ABOVE it (anchoring frame 0 sinks the rest below the table if frame 0 is high)
    # ★ 落桌高度必须用**真实顶点**最低点。原来用 obj_pos_w[:,2].min() —— 那是物体
    #   **原点**的高度, 与网格底面差多少完全取决于建模时原点放哪, 可以差十几厘米。
    #   (rl_rebuild align_replay 的注释只警告了 AABB 角点的 4.6cm 悬空; 用原点更糟。)
    _bottom = None
    if "obj_verts_local" in data:
        Vl = np.asarray(data["obj_verts_local"], np.float64)      # (n_obj,N,3)
        zs = []
        for oi in range(len(Vl)):
            q = obj_quat_w if oi == 0 else None
            P = obj_pos_w if oi == 0 else None
            if oi > 0 and oi < len(obj_poses):
                P = np.einsum("ij,tj->ti", Rs, obj_poses[oi][:, :3])
                q = np.stack([quat_mul(qs, obj_poses[oi][i, 3:7])
                              for i in range(len(obj_poses[oi]))])
            if q is None:
                continue
            for t in range(0, len(q), max(1, len(q) // 40)):
                zs.append((quat_to_rotmat(q[t]) @ Vl[oi].T).T[:, 2].min() + P[t, 2])
        if zs:
            _bottom = float(np.min(zs))
    if _bottom is None:
        _bottom = float(obj_pos_w[:, 2].min())
        print("[setup] ⚠ replay 里没有 obj_verts_local, 落桌退回物体原点高度 —— 可能悬空/陷入桌面; "
              "重跑 retarget 可修")
    scene_shift = np.array([-obj_pos_w[0, 0], -obj_pos_w[0, 1],
                            args.table_height + args.obj_lift - _bottom])
    print(f"[setup] 落桌: 物体轨迹真实最低点 z={_bottom:.3f} -> 桌面 {args.table_height:.2f}"
          f"+{args.obj_lift:.2f}m, 场景平移 {scene_shift.round(3).tolist()}")
for side in hand_joints_raw:
    hand_joints_raw[side] = np.einsum("ij,tkj->tki", Rs, hand_joints_raw[side]) + scene_shift
obj_pos_w = obj_pos_w + scene_shift
obj_pose = np.concatenate([obj_pos_w, obj_quat_w], axis=1)
# every other object rides the identical Rs + scene_shift; using a per-object transform
# would destroy the relative placement that made a single-pass reconstruction necessary
_others = []
for _p in obj_poses[1:]:
    _pos = np.einsum("ij,tj->ti", Rs, _p[:, :3]) + scene_shift
    _quat = np.stack([quat_mul(qs, _p[i, 3:7]) for i in range(len(_p))])
    _others.append(np.concatenate([_pos, _quat], axis=1))
obj_poses = [obj_pose] + _others

pos_off = np.array([float(x) for x in args.wrist_pos_offset.split(",")])
rot_off = np.array([float(x) for x in args.wrist_rot_offset.split(",")])
rot_off = rot_off / np.linalg.norm(rot_off)

print(f"[setup] retarget sharpa_wave {sorted(hand_joints_raw)} {args.retarget_mode}; "
      f"T={T} frames @ {fps:g}fps, mode={args.mode}")


def base_rot_from_joints(kp, side):
    """SharpaWave base orientation from world-frame joints (columns = base x/y/z in world).
    Convention (retarget README): +z = fingers, +y = thumb side, +x = palm normal."""
    w = kp[0]
    z = kp[[5, 9, 13, 17]].mean(0) - w            # wrist -> MCP centroid = fingers (+z)
    z = z / (np.linalg.norm(z) + 1e-9)
    radial = kp[5] - kp[17]                        # index_mcp - pinky_mcp -> thumb side (+y)
    y = radial - (radial @ z) * z
    y = y / (np.linalg.norm(y) + 1e-9)
    x = np.cross(y, z)                             # palm normal (+x), right-handed
    R = np.stack([x, y, z], axis=1)
    if side in [s for s in args.palm_flip.split(",") if s]:
        R[:, 0] *= -1; R[:, 1] *= -1               # 180 deg about fingers: flip palm + thumb
    return R


def precompute_hand(side, joints_h):
    """MANO joints -> per-frame 22-DOF finger qpos (SDK order) + floating base pose."""
    retargeting = build_sharpa_retargeting(side, args.retarget_mode)
    mapper = JointMapper(retargeting, side)
    Th = len(joints_h)
    fq = np.zeros((Th, 22)); bp = np.zeros((Th, 3)); bq = np.zeros((Th, 4))
    wp_key, wq_key = f"wrist_pos_{side}", f"wrist_quat_{side}"
    for i in range(Th):
        kp = joints_h[i]
        fq[i] = mapper.to_sdk(retargeting.retarget(
            compute_ref_value(retargeting, to_mano(kp, side))))
        if wp_key in data and wq_key in data:                       # per-hand wrist
            wp = Rs @ np.asarray(data[wp_key][i], np.float64) + scene_shift
            wq = quat_mul(qs, np.asarray(data[wq_key][i], np.float64))
        elif n_hands == 1 and "wrist_pos" in data and "wrist_quat" in data:
            wp = Rs @ np.asarray(data["wrist_pos"][i], np.float64) + scene_shift
            wq = quat_mul(qs, np.asarray(data["wrist_quat"][i], np.float64))
        else:                                                       # derive from joints
            wp = kp[0]
            if args.base == "joints":
                wq = rotmat_to_quat(base_rot_from_joints(kp, side))
            else:
                wq = rotmat_to_quat(estimate_frame_from_hand_points(kp - kp[0]))
        bp[i] = wp + pos_off
        bq[i] = quat_mul(wq, rot_off)
    if args.no_base_motion:
        bp[:] = bp[0]; bq[:] = bq[0]
    return dict(side=side, retargeting=retargeting, mapper=mapper,
                finger_qpos=fq, base_pos=bp, base_quat=bq)


HANDS = [precompute_hand(side, hand_joints_raw[side]) for side in sorted(hand_joints_raw)]

# --obj-place grip: static object at the two-hand grip point (frame of closest wrists),
# decoupled from the (possibly misaligned) object trajectory.
obj_place = None
if args.obj_place == "grip":
    if len(HANDS) == 2:
        mid = 0.5 * (HANDS[0]["base_pos"] + HANDS[1]["base_pos"])   # (T,3) hand midpoint
    else:
        mid = HANDS[0]["base_pos"]
    g = int(np.argmin(mid[:, 2]))                      # lowest hands = the pick-up moment
    obj_place = mid[g].copy()
    obj_place[2] = args.table_height + args.obj_lift   # rest on the table at the grip XY
    print(f"[setup] obj-place=grip at lowest-hands frame {g} -> world {obj_place.round(3).tolist()}")

for h in HANDS:
    print(f"[setup] {h['side']}: {len(h['finger_qpos'])} finger qpos + base poses; "
          f"wrist travel = {np.linalg.norm(h['base_pos'][-1] - h['base_pos'][0]) * 100:.1f} cm")

# ---- boot Kit --------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import omni.usd  # noqa: E402
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane  # noqa: E402
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim, SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

def hand_usd_path(side):
    p = (args.hand_usd if (args.hand_usd and n_hands == 1) else os.path.join(
        REPO, "assets", "robots", "hands", "sharpa_wave", side, f"{side}_sharpa_wave.usd"))
    if not os.path.exists(p):
        raise SystemExit(f"hand USD missing: {p} (scripts/convert_sharpa_urdf.py --side {side})")
    return p


def count_colliders(prim_path):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    rng = Usd.PrimRange(root, Usd.TraverseInstanceProxies())  # hand meshes are instanced
    return sum(1 for p in rng if p.HasAPI(UsdPhysics.CollisionAPI))


def add_object_physics(prim_path, mass, approx):
    """Make the referenced object a dynamic rigid body with mesh collision."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    # stability: more solver iterations + cap depenetration velocity so a deep
    # teleport-penetration (kinematic hand base) cannot launch the object.
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateSolverPositionIterationCountAttr(int(args.solver_pos_iter))
    rb.CreateSolverVelocityIterationCountAttr(4)
    rb.CreateMaxDepenetrationVelocityAttr(float(args.max_depen_vel))
    if args.log_contact:
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0.0)
    n_mesh = 0
    for prim in Usd.PrimRange(root):  # root + all descendants
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
            n_mesh += 1
    print(f"[physics] object rigid body + {n_mesh} mesh collider(s) approx={approx} mass={mass}kg")


def add_hand_colliders(prim_path, approx="convexHull"):
    """Apply convex collision to every hand mesh that lacks one (the converted SharpaWave
    USD ships no active colliders, so fingers pass through without this)."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    n = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)
            n += 1
    return n


# ---- world + scene ---------------------------------------------------------
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / max(fps, 1.0))
stage = omni.usd.get_context().get_stage()
if args.floor_usd and os.path.exists(args.floor_usd):
    add_reference_to_stage(os.path.abspath(args.floor_usd), "/World/ground")  # light-blue grid floor
    print(f"[scene] grid floor: {os.path.basename(args.floor_usd)}")
else:
    GroundPlane("/World/ground", z_position=0.0)
    if args.floor_usd:
        print(f"[warn] floor USD missing ({args.floor_usd}); using plain ground plane")
light = UsdLux.DomeLight.Define(stage, "/World/light")
light.CreateIntensityAttr(2500.0)


def bind_color(prim_path, rgb):
    """Bind a gray UsdPreviewSurface to a prim subtree (overrides authored materials)."""
    from pxr import Gf, Sdf, UsdShade
    mat_path = f"/World/Looks/mat{prim_path.replace('/', '_')}"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
if args.table:
    th = args.table_thickness
    white = np.array([1.0, 1.0, 1.0])
    FixedCuboid(                                  # thin tabletop, top surface at table_height
        "/World/table_top",
        position=np.array([0.0, 0.0, args.table_height - th / 2.0]),
        scale=np.array([args.table_size, args.table_size, th]),
        color=white,
    )
    leg = 0.08
    leg_h = max(args.table_height - th, 0.01)
    inset = args.table_size / 2.0 - 0.3
    for li, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        FixedCuboid(
            f"/World/table_leg_{li}",
            position=np.array([sx * inset, sy * inset, leg_h / 2.0]),
            scale=np.array([leg, leg, leg_h]),
            color=white,
        )
    print(f"[scene] white table {args.table_size:g}x{args.table_size:g} m, "
          f"top slab {th * 100:g} cm @ z={args.table_height:g} m")

obj_present = (args.object_usd is not None) and not args.no_object
OBJ_PATHS = []
if obj_present:
    if len(args.object_usd) > len(obj_poses):
        raise SystemExit(f"{len(args.object_usd)} object USDs given but the trajectory has "
                         f"{len(obj_poses)} object pose stream(s)")
    if len(args.object_usd) < len(obj_poses):
        print(f"[setup] {len(obj_poses)} object trajectories, {len(args.object_usd)} USD(s) "
              f"given -- replaying only the first {len(args.object_usd)}")
        obj_poses = obj_poses[: len(args.object_usd)]
    for i, u in enumerate(args.object_usd):
        path = f"/World/Object_{i}" if len(args.object_usd) > 1 else "/World/Object"
        add_reference_to_stage(os.path.abspath(u), path)
        OBJ_PATHS.append(path)
elif args.object_usd is None and not args.no_object:
    raise SystemExit("need --object-usd, or pass --no-object to replay hands only")

# obj0 anchor: rest the (now upright) object's BOTTOM on the table, and shift the hands
# by the same amount so the hand-object relative geometry is preserved exactly.
if obj_present and args.scene_rot == "obj0":
    try:
        prim = stage.GetPrimAtPath(OBJ_PATHS[0])
        rng = UsdGeom.Imageable(prim).ComputeLocalBound(
            Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedRange()
        rest = -float(rng.GetMin()[2])                 # lift so canonical bottom hits the origin z
        for _p in obj_poses:                           # same lift for all, keeps them paired
            _p[:, 2] += rest
        obj_pose = obj_poses[0]
        for h in HANDS:
            h["base_pos"][:, 2] += rest
        if obj_place is not None:
            obj_place[2] += rest
        print(f"[setup] rest object bottom on table: +{rest * 100:.1f} cm")
    except Exception as e:
        print(f"[warn] could not compute object bbox for table-resting: {e!r}")

hand_rgb = [float(x) for x in args.hand_color.split(",")]
for h in HANDS:
    add_reference_to_stage(hand_usd_path(h["side"]), f"/World/Hand_{h['side']}")
    bind_color(f"/World/Hand_{h['side']}", hand_rgb)

physics = args.mode == "physics"
if physics and len(OBJ_PATHS) > 1:
    raise SystemExit("--mode physics supports a single object; a multi-object scene needs "
                     "per-object mass/collision tuning that is not modelled here. Use "
                     "--mode render to inspect the trajectories.")
OBJS = []
for i, path in enumerate(OBJ_PATHS):
    if physics and args.collision != "none":
        add_object_physics(path, args.obj_mass, args.collision)
    OBJS.append(SingleRigidPrim(path, name=f"obj{i}") if physics
                else SingleXFormPrim(path, name=f"obj{i}"))
obj = OBJS[0] if OBJS else None
for h in HANDS:
    h["art"] = SingleArticulation(f"/World/Hand_{h['side']}", name=f"hand_{h['side']}")

if physics:
    for h in HANDS:
        nhc = add_hand_colliders(f"/World/Hand_{h['side']}", "convexHull")
        print(f"[physics] hand {h['side']}: added {nhc} finger colliders (convexHull)")

if not physics:
    # kinematic replay: kill gravity so nothing drifts between explicit sets
    world.get_physics_context().set_gravity(0.0)

world.reset()

# per-hand joint-order map: SDK order (mapper.sdk_names) -> isaac dof order
for h in HANDS:
    art = h["art"]
    art.initialize()
    isaac_names = list(art.dof_names)
    sdk_names = h["mapper"].sdk_names
    missing = [n for n in isaac_names if n not in sdk_names]
    if missing:
        raise SystemExit(f"{h['side']} hand USD dofs not in SDK list: {missing}")
    h["sdk2isaac"] = np.array([sdk_names.index(n) for n in isaac_names], dtype=int)
    if physics:
        ndof = art.num_dof
        art.get_articulation_controller().set_gains(
            kps=np.full(ndof, args.finger_kp), kds=np.full(ndof, args.finger_kd))

if physics:
    for h in HANDS:
        nc = count_colliders(f"/World/Hand_{h['side']}")
        print(f"[physics] hand {h['side']}: {nc} colliders, {h['art'].num_dof} dofs, "
              f"gains kp={args.finger_kp} kd={args.finger_kd}"
              + ("  <-- NO HAND COLLIDERS (fingers will pass through!)" if nc == 0 else ""))
    if obj is not None:
        print(f"[physics] object: {count_colliders('/World/Object')} colliders, "
              f"solverPosIter={args.solver_pos_iter} maxDepenVel={args.max_depen_vel}")


def place_at(idx):
    """Pose every hand (base + fingers) and, in render mode, the object at frame idx."""
    for h in HANDS:
        hi = min(idx, len(h["finger_qpos"]) - 1)
        q = h["finger_qpos"][hi][h["sdk2isaac"]]
        if not args.no_base_motion:
            h["art"].set_world_pose(position=h["base_pos"][hi], orientation=h["base_quat"][hi])
        if physics:
            h["art"].get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=q))
        else:
            h["art"].set_joint_positions(q)
    if not physics and obj_place is None:
        for o, pose in zip(OBJS, obj_poses):
            oi = min(idx, len(pose) - 1)
            o.set_world_pose(pose[oi, :3], pose[oi, 3:7])


def obj_init_pose(k: int = 0):
    if obj_place is not None:
        return obj_place, np.array([1.0, 0.0, 0.0, 0.0])
    return obj_poses[k][0, :3], obj_poses[k][0, 3:7]


def place_objects_at_start():
    for k, o in enumerate(OBJS):
        o.set_world_pose(*obj_init_pose(k))


# place hands + object at frame 0
for h in HANDS:
    h["art"].set_world_pose(position=h["base_pos"][0], orientation=h["base_quat"][0])
    h["art"].set_joint_positions(h["finger_qpos"][0][h["sdk2isaac"]])
if obj is not None:
    place_objects_at_start()
    if physics:
        obj.set_linear_velocity(np.zeros(3)); obj.set_angular_velocity(np.zeros(3))

# ---- camera viewpoint (live GUI viewport + snapshot sensor) ----------------
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

# aim at the hand cloud center (incl. height) so the moving/lifted hands stay framed
allbp = np.concatenate([h["base_pos"] for h in HANDS], axis=0)
cx, cy, cz = float(allbp[:, 0].mean()), float(allbp[:, 1].mean()), float(allbp[:, 2].mean())
if args.cam_view == "topdown":            # egocentric: above the hands, looking down
    dz = args.cam_dist
    cam_eye = np.array([cx, cy - 0.13 * dz, cz + dz])
    cam_target = np.array([cx, cy, cz])
elif args.cam_view == "iso":              # 3/4 view
    d = args.cam_dist
    cam_eye = np.array([cx + d, cy - d, cz + 0.7 * d])
    cam_target = np.array([cx, cy, cz])
else:                                     # manual
    cam_eye = np.array([float(x) for x in args.cam_eye.split(",")])
    cam_target = np.array([float(x) for x in args.cam_target.split(",")])
try:
    set_camera_view(eye=cam_eye, target=cam_target)   # the perspective the GUI shows
except Exception as e:
    print(f"[warn] viewport camera set failed: {e!r}")
print(f"[cam] view={args.cam_view} eye={cam_eye.round(2).tolist()} target={cam_target.round(2).tolist()}")

cam = None
snap_times = [float(x) for x in args.snap_times.split(",") if x.strip()] if args.snap_times else []
if args.snap_dir:
    if args.snap_dir:
        os.makedirs(args.snap_dir, exist_ok=True)
    try:
        from isaacsim.sensors.camera import Camera
        cam = Camera(prim_path="/World/snap_cam",
                     resolution=(1280, 800),
                     position=cam_eye)
        cam.initialize()
        cam.set_focal_length(args.record_focal)
        set_camera_view(eye=cam_eye, target=cam_target, camera_prim_path="/World/snap_cam")
        # ★ 不依赖 set_camera_view 对**非视口**相机是否生效 —— 显式算 look-at 位姿。
        #   实测只调 set_camera_view 时录出来的是白桌面+网格地板, 手和物体不在画面里。
        #   USD 相机约定: 局部 -Z 为视线方向, +Y 为上。
        _f = cam_target - cam_eye
        _f = _f / (np.linalg.norm(_f) + 1e-9)
        _up0 = np.array([0.0, 0.0, 1.0])
        if abs(float(_f @ _up0)) > 0.95:                  # 近乎垂直俯视时换个参考上方向
            _up0 = np.array([0.0, 1.0, 0.0])
        _r = np.cross(_f, _up0); _r /= (np.linalg.norm(_r) + 1e-9)
        _u = np.cross(_r, _f)
        _R = np.stack([_r, _u, -_f], axis=1)              # 列 = 右/上/-视线
        _t = np.trace(_R)
        if _t > 0:
            _s = np.sqrt(_t + 1.0) * 2
            _q = np.array([0.25 * _s, (_R[2, 1] - _R[1, 2]) / _s,
                           (_R[0, 2] - _R[2, 0]) / _s, (_R[1, 0] - _R[0, 1]) / _s])
        else:
            _i = int(np.argmax(np.diag(_R)))
            if _i == 0:
                _s = np.sqrt(1.0 + _R[0, 0] - _R[1, 1] - _R[2, 2]) * 2
                _q = np.array([(_R[2, 1] - _R[1, 2]) / _s, 0.25 * _s,
                               (_R[0, 1] + _R[1, 0]) / _s, (_R[0, 2] + _R[2, 0]) / _s])
            elif _i == 1:
                _s = np.sqrt(1.0 + _R[1, 1] - _R[0, 0] - _R[2, 2]) * 2
                _q = np.array([(_R[0, 2] - _R[2, 0]) / _s, (_R[0, 1] + _R[1, 0]) / _s,
                               0.25 * _s, (_R[1, 2] + _R[2, 1]) / _s])
            else:
                _s = np.sqrt(1.0 + _R[2, 2] - _R[0, 0] - _R[1, 1]) * 2
                _q = np.array([(_R[1, 0] - _R[0, 1]) / _s, (_R[0, 2] + _R[2, 0]) / _s,
                               (_R[1, 2] + _R[2, 1]) / _s, 0.25 * _s])
        _q = _q / (np.linalg.norm(_q) + 1e-9)
        try:
            cam.set_world_pose(position=cam_eye, orientation=_q)
            print(f"[cam] snap_cam 显式位姿 eye={cam_eye.round(3).tolist()} "
                  f"look={cam_target.round(3).tolist()}")
        except Exception as e:
            print(f"[warn] snap_cam set_world_pose 失败: {e!r}")
        print(f"[snap] camera -> {args.snap_dir}")
    except Exception as e:  # snapshots are best-effort
        print(f"[warn] camera init failed, skipping snapshots: {e!r}")
        cam = None


def maybe_snap(t):
    if cam is None:
        return
    while snap_times and t >= snap_times[0]:
        st = snap_times.pop(0)
        try:
            import imageio
            imageio.imwrite(os.path.join(args.snap_dir, f"snap_{st:06.2f}.png"), cam.get_rgba()[:, :, :3])
            print(f"[snap] t={st:.2f}s -> {args.snap_dir}")
        except Exception as e:
            print(f"[warn] snap t={st} failed: {e!r}")


# ---- trajectory visualization (lines in space) -----------------------------
def draw_trajectories():
    """Draw the given hand-base and object position tracks as persistent 3D lines."""
    if args.no_traj:
        return None
    try:
        from isaacsim.util.debug_draw import _debug_draw
        dd = _debug_draw.acquire_debug_draw_interface()
    except Exception as e:  # best-effort overlay
        print(f"[warn] debug_draw unavailable, no trajectory lines: {e!r}")
        return None
    dd.clear_lines()
    obj_color = (1.0, 0.85, 0.1, 1.0)                       # amber
    hand_color = {"right": (0.1, 0.9, 0.25, 1.0),           # green
                  "left": (0.2, 0.55, 1.0, 1.0)}            # blue

    def polyline(points, color, width):
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
        if len(pts) < 2:
            return
        dd.draw_lines(pts[:-1], pts[1:], [color] * (len(pts) - 1), [width] * (len(pts) - 1))

    show_obj_track = obj_present and obj_place is None and not physics  # no track if static/dynamic
    if show_obj_track:
        obj_palette = [obj_color, (0.20, 0.95, 0.55, 1.0), (0.95, 0.55, 0.20, 1.0)]
        for k, pose in enumerate(obj_poses):
            polyline(pose[:, :3], obj_palette[k % len(obj_palette)], 5.0)
    for h in HANDS:
        polyline(h["base_pos"], hand_color.get(h["side"], (1, 1, 1, 1)), 3.0)
    print("[viz] trajectory lines: "
          + ("object=amber, " if show_obj_track else "")
          + ", ".join(f"{h['side']}={'green' if h['side'] == 'right' else 'blue'}" for h in HANDS))
    return dd


def reset_to_start():
    place_at(0)
    if obj is not None and (physics or obj_place is not None):
        place_objects_at_start()
        if physics:
            obj.set_linear_velocity(np.zeros(3)); obj.set_angular_velocity(np.zeros(3))


# ---- one replay pass -------------------------------------------------------
steps_per_frame = max(1, round((1.0 / fps) / world.get_physics_dt()))
_render = (not args.headless)

# contact logging: accumulate hand<->object impulse per physics step
_contact = {"imp": 0.0, "hands": set()}
contact_sub = None
contact_rows = []
if physics and args.log_contact:
    def _on_contact(headers, data):
        for ch in headers:
            a0 = PhysicsSchemaTools.intToSdfPath(ch.actor0).pathString
            a1 = PhysicsSchemaTools.intToSdfPath(ch.actor1).pathString
            if "Object" not in a0 and "Object" not in a1:
                continue
            other = a1 if "Object" in a0 else a0
            if "Hand_" not in other:        # ignore object<->table/ground contacts
                continue
            _contact["hands"].add(other.split("Hand_")[1].split("/")[0])
            for i in range(ch.contact_data_offset, ch.contact_data_offset + ch.num_contact_data):
                imp = getattr(data[i], "impulse", (0, 0, 0))
                _contact["imp"] += float((imp[0] ** 2 + imp[1] ** 2 + imp[2] ** 2) ** 0.5)
    try:
        from omni.physx import get_physx_simulation_interface
        contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)
        print("[physics] contact logging ON (PRESS=fingers squeeze, PASS=overlap but no force, FLY=blow-up)")
    except Exception as e:
        print(f"[warn] contact logging unavailable: {e!r}")


def play_one_pass():
    obj_log = []
    t = 0.0
    for frame in range(T):
        place_at(frame)
        _contact["imp"] = 0.0; _contact["hands"] = set()
        for _ in range(steps_per_frame):
            world.step(render=_render or cam is not None)
        t += steps_per_frame * world.get_physics_dt()
        maybe_snap(t)
        if physics and obj is not None:
            p, _ = obj.get_world_pose()
            obj_log.append((round(t, 3), float(p[2])))
            if args.log_contact:
                spd = float(np.linalg.norm(obj.get_linear_velocity()))
                # nearest hand: wrist-base to object distance (proximity = should be touching)
                dist = min(float(np.linalg.norm(h["base_pos"][min(frame, len(h["base_pos"]) - 1)] - p))
                           for h in HANDS)
                imp = _contact["imp"]; nh = len(_contact["hands"])
                if spd > 1.0:
                    flag = "FLY"
                elif imp > 1e-4:
                    flag = "PRESS"
                elif dist < 0.15:
                    flag = "PASS-THROUGH?"   # hand at object but no contact force
                else:
                    flag = ""
                contact_rows.append((round(t, 3), round(float(p[2]), 4), round(spd, 3),
                                     round(imp, 4), nh, round(dist, 3), flag))
                if flag or frame % 10 == 0:
                    print(f"[contact] t={t:5.2f} z={p[2]:.3f} spd={spd:5.2f} imp={imp:7.4f} "
                          f"hands_touch={nh} hand_dist={dist:.3f}m {flag}", flush=True)
        if not app.is_running():
            break
    if physics and obj_log:
        z0, z1 = obj_log[0][1], obj_log[-1][1]
        print(f"[physics] object z: start={z0:.3f}m end={z1:.3f}m "
              f"(lifted {(z1 - z0) * 100:+.1f} cm)  samples={len(obj_log)}")
    if physics and args.log_contact and contact_rows:
        import csv
        path = os.path.splitext(args.traj)[0] + "_contact.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "obj_z", "obj_speed", "contact_impulse", "n_hands_touching", "hand_dist", "flag"])
            w.writerows(contact_rows)
        npress = sum(1 for r in contact_rows if r[6] == "PRESS")
        npass = sum(1 for r in contact_rows if r[6] == "PASS-THROUGH?")
        nfly = sum(1 for r in contact_rows if r[6] == "FLY")
        print(f"[contact] wrote {path}: {len(contact_rows)} frames  "
              f"PRESS={npress}  PASS-THROUGH?={npass}  FLY={nfly}")
    print(f"[done] replayed {T} frames")


# ---- run -------------------------------------------------------------------
# --object-usd is nargs="+", so it is always a list -- basename() on it raised TypeError
# and killed every multi-object run before the first frame.
print(f"[run] mode={args.mode} hands={[h['side'] for h in HANDS]} "
      f"object={[os.path.basename(u) for u in args.object_usd] if obj_present else '(hidden)'}")
dd = draw_trajectories()
try:
    if args.headless or args.auto:
        # non-interactive: one pass, or continuous with --loop
        while app.is_running():
            play_one_pass()
            if not args.loop:
                break
            reset_to_start()
    else:
        import queue
        import threading
        cmd_q = queue.Queue()

        def _stdin_reader():
            for line in sys.stdin:
                cmd_q.put(line.strip().lower())

        threading.Thread(target=_stdin_reader, daemon=True).start()
        print("\n[ready] hands + object placed at frame 0. "
              "Press ENTER to play one pass; type q then ENTER to quit.")
        while app.is_running():
            world.step(render=_render)          # keep the viewport live while waiting
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                continue
            if cmd == "q":
                break
            play_one_pass()
            reset_to_start()
            draw_trajectories()                 # redraw in case the overlay was cleared
            print("\n[ready] press ENTER to replay, or q then ENTER to quit.")
finally:
    import threading
    sys.stdout.flush()  # os._exit skips buffer flush; keep redirected logs intact
    threading.Timer(20.0, lambda: os._exit(0)).start()  # known shutdown hang
    app.close()

