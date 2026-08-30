"""Build the 20 Hz Sweep2 P-OBJ reference from reconstructed tool trajectories.

Each tool is placed with the same canon-rotation, pinned yaw, and affordance-centred
translation used by ``GraspTaskEnv``. Reconstructed world-space position and rotation
increments are then applied to that reachable pose. Each tool pose is converted to
a hand pose with its GraspPose ``T_object_hand`` and solved by continuous-seed arm
IK. Fingers are fixed throughout; reconstructed human finger motion is not P-OBJ.
"""
from __future__ import annotations

import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--output", default="tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz")
p.add_argument("--source_fps", type=float, default=30.0)
p.add_argument("--control_hz", type=float, default=20.0)
p.add_argument("--broom_prior", default="tasks/pregrasp/priors/Sweep2_broom_v2.npz")
p.add_argument("--pan_prior", default="tasks/pregrasp/priors/Sweep2_dustpan.npz")
p.add_argument("--anchor", default="tasks/pregrasp/priors/anchor_T_right.json")
p.add_argument("--broom_yaw_deg", type=float, default=90.0)
p.add_argument("--pan_yaw_deg", type=float, default=64.2)
p.add_argument("--world_yaw_deg", type=float, default=120.0,
               help="Rigid recon-world to sim-world yaw applied to all SE(3) increments")
p.add_argument("--rot_scale", type=float, default=0.08,
               help="Retained fraction of reconstructed object rotation increments")
p.add_argument("--geometry_only", action="store_true",
               help="Audit cube/brush geometry without running IK or writing output")
p.add_argument("--scan_world_yaw", action="store_true",
               help="Print the geometry audit for -180..170 degrees in one load")
p.add_argument("--audit_side", choices=("right", "left"), default=None,
               help="Run IK diagnostics for one side and exit without writing")
p.add_argument("--scan_ik_yaw", action="store_true",
               help="Multi-start audit the main broom-action row over world yaw")
args = p.parse_args()

import numpy as np

from rl_rebuild.correction.kinematics import ArmIK
from rl_rebuild.correction.ref_builders.replay_grasp import (GENERIC_JOINT_ORDER,
                                                             load_replay_grasp)
from rl_rebuild.correction.schema import ObjectSemantics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
DATA = os.path.join(ROOT, "datasets", "sweep_2_better")
TABLE_Z = 0.87


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def qconj(q):
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def q_to_R(q):
    q = np.asarray(q, np.float64); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def R_to_q(R):
    # Stable scalar-first matrix conversion.
    m = np.asarray(R, np.float64)
    vals = np.array([1+m.trace(), 1+m[0, 0]-m[1, 1]-m[2, 2],
                     1-m[0, 0]+m[1, 1]-m[2, 2], 1-m[0, 0]-m[1, 1]+m[2, 2]])
    i = int(np.argmax(vals)); q = np.zeros(4)
    if i == 0:
        q[0] = 0.5*np.sqrt(max(vals[0], 0)); d = max(4*q[0], 1e-12)
        q[1:] = [(m[2, 1]-m[1, 2])/d, (m[0, 2]-m[2, 0])/d,
                 (m[1, 0]-m[0, 1])/d]
    else:
        j, k = (i % 3) + 1, ((i + 1) % 3) + 1
        q[i] = 0.5*np.sqrt(max(vals[i], 0)); d = max(4*q[i], 1e-12)
        q[0] = (m[k-1, j-1]-m[j-1, k-1])/d
        q[j] = (m[i-1, j-1]+m[j-1, i-1])/d
        q[k] = (m[i-1, k-1]+m[k-1, i-1])/d
    return q / np.linalg.norm(q)


def pose_T(pos, quat):
    T = np.eye(4); T[:3, :3] = q_to_R(quat); T[:3, 3] = pos; return T


def interp_T(track, times):
    """Linear position + shortest-arc quaternion interpolation."""
    n = len(track); out = np.zeros((len(times), 4, 4))
    for k, t in enumerate(times):
        i = min(int(np.floor(t)), n - 2); a = float(t - i)
        p = (1-a)*track[i, :3, 3] + a*track[i+1, :3, 3]
        q0, q1 = R_to_q(track[i, :3, :3]), R_to_q(track[i+1, :3, :3])
        if np.dot(q0, q1) < 0: q1 = -q1
        dot = float(np.clip(np.dot(q0, q1), -1, 1))
        if dot > 0.9995:
            q = q0 + a*(q1-q0); q /= np.linalg.norm(q)
        else:
            th = np.arccos(dot); q = (np.sin((1-a)*th)*q0 + np.sin(a*th)*q1)/np.sin(th)
        out[k] = pose_T(p, q)
    return out


def load_rts(oi):
    return np.load(os.path.join(DATA, "poseqa", f"rts_sweep_dustpan_2_object_{oi}.npz"))


import trimesh  # noqa: E402


def scene_pose(oi, hand, prior_path, yaw_deg):
    """Reproduce GraspTaskEnv's canon-rotation/yaw/affordance placement on CPU."""
    mesh_path = os.path.join(DATA, "objects", f"object_{oi}",
                             "object_mesh_scaled_final.obj")
    du = load_replay_grasp(
        os.path.join(DATA, "retarget", "replay_world.npz"), mesh_path,
        usd_path=os.path.join(DATA, "retarget", f"object_{oi}.usd"), hand=hand,
        scene_layout_json=os.path.join(DATA, "scene_layout.json"),
        clip_id=f"Sweep2_{'broom' if oi == 1 else 'dustpan'}",
        target_hz=args.control_hz, table_height=TABLE_Z,
        semantics=ObjectSemantics(label=hand, mass_kg=0.1, friction=0.6), verbose=False)
    pose = np.asarray(du.object_init_pose, np.float64).copy()
    rest_R = q_to_R(pose[3:7])
    prior = np.load(prior_path)
    canon_q = np.asarray(prior["canon_rot"], np.float64)
    yaw = np.radians(yaw_deg)
    yaw_q = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])
    final_q = qmul(yaw_q, canon_q); final_R = q_to_R(final_q)
    verts = np.asarray(trimesh.load(mesh_path, force="mesh").vertices, np.float64)
    pose[2] = TABLE_Z + 0.002 - (verts @ q_to_R(canon_q).T)[:, 2].min()
    a_obj = verts.mean(0)
    pose[:2] += (rest_R @ a_obj)[:2] - (final_R @ a_obj)[:2]
    pose[3:7] = final_q
    return pose


scene = {0: scene_pose(0, "left", args.pan_prior, args.pan_yaw_deg),
         1: scene_pose(1, "right", args.broom_prior, args.broom_yaw_deg)}
with open(args.anchor) as f:
    anchor_T = np.asarray(json.load(f)["anchor_T"], np.float64)
zr = {i: load_rts(i) for i in (0, 1)}
raw = {i: np.asarray(zr[i]["object_ob_in_world_smooth"], np.float64) for i in (0, 1)}

duration = (len(raw[0]) - 1) / args.source_fps
times_s = np.arange(0.0, duration + 1e-9, 1.0 / args.control_hz)
source_times = times_s * args.source_fps
resampled = {i: interp_T(raw[i], source_times) for i in (0, 1)}


def registered_tools(world_yaw_deg):
    tool_T = {}
    wy = np.radians(world_yaw_deg)
    R_world = np.array([[np.cos(wy), -np.sin(wy), 0.0],
                        [np.sin(wy), np.cos(wy), 0.0], [0.0, 0.0, 1.0]])
    for i in (0, 1):
        rs = resampled[i]
        Ts = pose_T(scene[i][:3], scene[i][3:7])
        out_i = np.zeros_like(rs)
        for t, Tr in enumerate(rs):
            out_i[t] = np.eye(4)
            out_i[t, :3, 3] = scene[i][:3] + R_world @ (
                Tr[:3, 3] - raw[i][0, :3, 3])
            dR = Tr[:3, :3] @ raw[i][0, :3, :3].T
            qd = R_to_q(dR)
            if qd[0] < 0.0:
                qd = -qd
            half = np.arccos(np.clip(qd[0], -1.0, 1.0))
            if half > 1e-9:
                axis = qd[1:] / np.sin(half)
                hs = args.rot_scale * half
                dR = q_to_R(np.r_[np.cos(hs), axis*np.sin(hs)])
            out_i[t, :3, :3] = R_world @ dR @ R_world.T @ Ts[:3, :3]
        tool_T[i] = out_i
    return tool_T

# Fixed easy cube placement is selected from a conservative grid just outside the
# initial pan lip.  It is still fixed across all episodes; this search merely makes
# the reconstructed broom pass through the cube instead of repeating the old 2.4 cm
# miss.  Only the broad brush-head half of the mesh participates.
bm = trimesh.load(os.path.join(DATA, "objects", "object_1",
                               "object_mesh_scaled_final.obj"), force="mesh")
bv = np.asarray(bm.vertices, np.float64)
bv = bv[bv[:, 2] > 0.04]
bv = bv[np.random.RandomState(0).choice(len(bv), min(600, len(bv)), replace=False)]

if args.scan_ik_yaw:
    prior = np.load(args.broom_prior)
    T_oh = pose_T(prior["grasp"][:3], prior["grasp"][3:7])
    ik = ArmIK("right", anchor_link="arm_center", anchor_T=anchor_T)
    rng = np.random.default_rng(20260830)
    seeds = [ik.q_default] + [rng.uniform(ik.lower, ik.upper) for _ in range(16)]
    key_rows = np.unique(np.r_[np.arange(0, len(times_s), 10), len(times_s)-1]).astype(int)
    for angle in np.arange(-180.0, 180.0, 20.0):
        tools = registered_tools(angle)
        bad = None
        for row in key_rows:
            Twh = tools[1][row] @ T_oh
            ans = ik.solve_best(Twh[:3, 3], Twh[:3, :3], seeds,
                                iters=300, pos_tol=0.005, rot_tol=0.05)
            if not ans["ok"]:
                bad = (row, 100*ans["pos_err"], np.degrees(ans["rot_err"]))
                break
        print(f"[ik-yaw-scan] world_yaw={angle:6.1f} "
              f"all_key_rows={bad is None} first_bad={bad}")
    raise SystemExit(0)
def brush_geometry(tool_T):
    pan0 = tool_T[0][0]
    candidates = []
    for x in np.linspace(-0.04, 0.04, 9):
        for zloc in np.linspace(0.110, 0.140, 7):
            p = (pan0 @ np.array([x, 0.0, zloc, 1.0]))[:3]
            p[2] = TABLE_Z + 0.0055
            candidates.append(p)
    candidates = np.asarray(candidates)
    best = (float("inf"), None, None, None)
    for ti in range(10, len(times_s) - 10):
        world = bv @ tool_T[1][ti, :3, :3].T + tool_T[1][ti, :3, 3]
        all_d = np.linalg.norm(world[:, None, :] - candidates[None, :, :], axis=2)
        d = all_d.min(axis=0)
        ci = int(np.argmin(d))
        if float(d[ci]) < best[0]:
            best = (float(d[ci]), ti, ci, int(np.argmin(all_d[:, ci])))
    return candidates, best


if args.scan_world_yaw:
    for angle in np.arange(-180.0, 180.0, 10.0):
        _, scan_best = brush_geometry(registered_tools(angle))
        print(f"[geometry-scan] world_yaw={angle:6.1f} "
              f"brush_distance_cm={100*scan_best[0]:.3f} row={scan_best[1]}")
    raise SystemExit(0)

tool_T = registered_tools(args.world_yaw_deg)
candidates, best = brush_geometry(tool_T)
cube_start = candidates[best[2]]
contact_row = int(best[1])
brush_contact_local = bv[best[3]]
brush_contact_world = (tool_T[1][contact_row, :3, :3] @ brush_contact_local +
                       tool_T[1][contact_row, :3, 3])
print(f"[reference] world_yaw={args.world_yaw_deg:.1f} easy cube={np.round(cube_start, 4).tolist()} | "
      f"nominal brush distance={best[0]*100:.2f}cm @ row {contact_row}")
print(f"[reference] brush_contact_world={np.round(brush_contact_world, 4).tolist()} "
      f"delta_to_cube_cm={np.round((brush_contact_world-cube_start)*100, 3).tolist()}")
if args.geometry_only:
    raise SystemExit(0)

priors = {"right": np.load(args.broom_prior), "left": np.load(args.pan_prior)}
oid = {"right": 1, "left": 0}
arm_q, ik_report = {}, {}


def solve_continuous(ik, P, Q, seed):
    """Sparse multi-solution IK + global branch DP + dense local refinement."""
    from scipy.interpolate import PchipInterpolator

    rng = np.random.default_rng(seed)
    key_rows = np.unique(np.r_[np.arange(0, len(P), 10), len(P)-1]).astype(int)
    pools = []
    seed_bank = [ik.q_default] + [rng.uniform(ik.lower, ik.upper) for _ in range(24)]
    for row in key_rows:
        trials = [ik.solve(P[row], q_to_R(Q[row]), q0=q0, iters=300,
                           pos_tol=0.005, rot_tol=0.05) for q0 in seed_bank]
        pool = []
        for ans in trials:
            if ans["ok"] and all(np.linalg.norm(ans["q"] - q) > 0.08 for q in pool):
                pool.append(np.asarray(ans["q"], np.float64))
        if not pool:
            best = min(trials, key=lambda r: r["pos_err"]**2 +
                       (0.35*r["rot_err"])**2)
            raise RuntimeError(f"IK key row {row} has no solution: "
                               f"{100*best['pos_err']:.2f}cm/"
                               f"{np.degrees(best['rot_err']):.2f}deg")
        pools.append(pool)
        print(f"[reference] IK key row {row}: {len(pool)} branches")

    costs = [np.array([np.linalg.norm(q - ik.q_default)**2 for q in pools[0]])]
    parents = []
    for prev, cur, prev_cost in zip(pools[:-1], pools[1:], costs):
        edge = np.array([[np.linalg.norm(q1-q0)**2 for q1 in cur] for q0 in prev])
        total = prev_cost[:, None] + edge
        parents.append(np.argmin(total, axis=0))
        costs.append(np.min(total, axis=0))
    j = int(np.argmin(costs[-1]))
    path = [pools[-1][j]]
    for k in range(len(parents)-1, -1, -1):
        j = int(parents[k][j]); path.append(pools[k][j])
    path = np.asarray(path[::-1])
    q_seed = PchipInterpolator(key_rows, path, axis=0)(np.arange(len(P)))

    reports = []
    for row, (p_tgt, quat_tgt, q0) in enumerate(zip(P, Q, q_seed)):
        R_tgt = q_to_R(quat_tgt)
        ans = ik.solve(p_tgt, R_tgt, q0=q0, iters=250,
                       pos_tol=0.005, rot_tol=0.05)
        if not ans["ok"]:
            trials = [ik.solve(p_tgt, R_tgt, q0=s, iters=300,
                               pos_tol=0.005, rot_tol=0.05)
                      for s in [q0] + seed_bank]
            good = [r for r in trials if r["ok"]]
            if good:
                ans = min(good, key=lambda r: np.linalg.norm(r["q"] - q0))
        reports.append(ans)
    return reports


for side in ((args.audit_side,) if args.audit_side else ("right", "left")):
    T_oh = pose_T(priors[side]["grasp"][:3], priors[side]["grasp"][3:7])
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=anchor_T)
    hand_T = [Twt @ T_oh for Twt in tool_T[oid[side]]]
    P = np.asarray([T[:3, 3] for T in hand_T])
    Q = np.asarray([R_to_q(T[:3, :3]) for T in hand_T])
    reports = solve_continuous(ik, P, Q, 20260830 + int(side == "left"))
    qs = [np.asarray(ans["q"], np.float64) for ans in reports]
    pe = [float(ans["pos_err"]) for ans in reports]
    re = [float(ans["rot_err"]) for ans in reports]
    ok = [bool(ans["ok"]) for ans in reports]
    arm_q[side] = np.asarray(qs, np.float32)
    ik_report[side] = dict(ok_ratio=float(np.mean(ok)), pos_max_cm=100*max(pe),
                           rot_max_deg=float(np.degrees(max(re))),
                           joint_step_max_deg=float(np.degrees(np.abs(np.diff(qs, axis=0)).max())))
    print(f"[reference] {side}: {ik_report[side]}")
    bad = np.flatnonzero(~np.asarray(ok))
    step = np.max(np.abs(np.diff(qs, axis=0)), axis=1)
    print(f"[reference] {side} bad_rows={bad.tolist()} "
          f"bad_source_frames={np.round(source_times[bad], 1).tolist()} "
          f"max_pos_row={int(np.argmax(pe))} max_rot_row={int(np.argmax(re))} "
          f"max_step_row={int(np.argmax(step)+1)}")

if args.audit_side:
    raise SystemExit(0)

out = {
    "right_q": arm_q["right"], "left_q": arm_q["left"],
    "right_f": np.repeat(priors["right"]["grasp"][None, 7:29], len(times_s), axis=0).astype(np.float32),
    "left_f": np.repeat(priors["left"]["grasp"][None, 7:29], len(times_s), axis=0).astype(np.float32),
    "fin_names": np.array(GENERIC_JOINT_ORDER, dtype=object),
    "source_frame": source_times.astype(np.float32),
    "control_hz": np.float32(args.control_hz),
    "cube_start_w": cube_start.astype(np.float32),
    "contact_row": np.int32(contact_row),
    "nominal_brush_cube_distance_m": np.float32(best[0]),
    "world_yaw_deg": np.float32(args.world_yaw_deg),
    "rotation_increment_scale": np.float32(args.rot_scale),
    "brush_contact_local": brush_contact_local.astype(np.float32),
    "scene_pose_0": scene[0].astype(np.float32),
    "scene_pose_1": scene[1].astype(np.float32),
    "meta": ("Sweep2 P-OBJ v1; RTS tools 30Hz->20Hz; full reconstructed position "
             f"increments; {args.rot_scale:.3f} reconstructed rotation increments; "
             f"recon-to-sim world yaw {args.world_yaw_deg:.1f}deg; GraspTask canon+yaw "
             "placement; GraspPose-locked hands; fingers fixed"),
}
for i in (0, 1):
    out[f"obj_pos_{i}"] = tool_T[i][:, :3, 3].astype(np.float32)
    out[f"obj_quat_{i}"] = np.asarray([R_to_q(T[:3, :3]) for T in tool_T[i]], np.float32)
    cf = np.minimum(np.asarray(zr[i]["conf_pos"]), np.asarray(zr[i]["conf_rot"]))
    out[f"confidence_{i}"] = np.interp(source_times, np.arange(len(cf)), cf).astype(np.float32)
out["ik_report"] = np.array(str(ik_report))

assert min(v["ok_ratio"] for v in ik_report.values()) >= 0.99, ik_report
assert max(v["pos_max_cm"] for v in ik_report.values()) <= 0.5, ik_report
assert best[0] <= 0.020, f"retargeted brush misses every easy cube candidate: {best[0]:.4f}m"
assert max(v["joint_step_max_deg"] for v in ik_report.values()) <= 8.0, ik_report
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
np.savez(args.output, **out)
print(f"[reference] wrote validated {args.output}: rows={len(times_s)}, "
      f"duration={times_s[-1]:.3f}s")
