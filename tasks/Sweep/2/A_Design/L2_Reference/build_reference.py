"""Build the source-faithful 20 Hz Sweep2 P-OBJ/HYB reference.

The RTS tool trajectories keep their full bimanual 6DoF path under one shared scene
registration. GraspPose supplies the frame-zero object-to-hand transform and fixed
finger posture. Object-driven wrist targets are solved by continuous arm IK. Human
wrist tracks are welded to the same GraspPose at frame zero and stored only as motion
shape guidance for medium/low-confidence rows, matching Pour17's HYB contract.
"""
from __future__ import annotations

import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--output", default="tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz")
p.add_argument("--control_hz", type=float, default=20.0)
p.add_argument("--broom_prior", default="tasks/pregrasp/priors/Sweep2_broom.npz")
p.add_argument("--pan_prior", default="tasks/pregrasp/priors/Sweep2_dustpan.npz")
p.add_argument("--anchor", default="tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json")
p.add_argument("--scene_yaw_deg", type=float, default=-14.0,
               help="Single shared frame-zero yaw registration; never a per-tool rewrite")
p.add_argument("--broom_start_x", type=float, default=-0.116)
p.add_argument("--broom_start_y", type=float, default=-0.177)
p.add_argument("--max_pos_step_m", type=float, default=0.008,
               help="Time-expand, without changing the path, above this tool translation step")
p.add_argument("--max_rot_step_deg", type=float, default=3.5,
               help="Time-expand, without changing the path, above this tool rotation step")
p.add_argument("--geometry_only", action="store_true",
               help="Audit cube/brush geometry without running IK or writing output")
p.add_argument("--audit_side", choices=("right", "left"), default=None,
               help="Run IK diagnostics for one side and exit without writing")
args = p.parse_args()

import numpy as np

from rl_rebuild.correction.kinematics import ArmIK
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER

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


with open(args.anchor) as f:
    anchor_T = np.asarray(json.load(f)["anchor_T"], np.float64)
zr = {i: load_rts(i) for i in (0, 1)}
raw = {i: np.asarray(zr[i]["object_ob_in_world_smooth"], np.float64) for i in (0, 1)}
replay = np.load(os.path.join(DATA, "retarget", "replay_world.npz"), allow_pickle=True)
source_fps = float(np.asarray(replay["fps"]))
assert source_fps == 15.0, f"unexpected Sweep trajectory clock: {source_fps}"


def rotation_angle(R0, R1):
    return float(np.arccos(np.clip((np.trace(R0.T @ R1) - 1.0) / 2.0, -1.0, 1.0)))


duration = (len(raw[0]) - 1) / source_fps
base_times = np.arange(0.0, duration + 1e-9, 1.0 / args.control_hz) * source_fps
if base_times[-1] < len(raw[0]) - 1:
    base_times = np.r_[base_times, len(raw[0]) - 1]
# A regular 20 Hz sample can straddle a 15 Hz reconstruction knot. Preserve the
# exact knot whenever either tool has a large source-frame jump; otherwise a jump
# can be hidden between two regular samples and reappear in only one output step.
critical = {0, len(raw[0]) - 1}
for row in range(len(raw[0]) - 1):
    for i in (0, 1):
        a, b = raw[i][row], raw[i][row + 1]
        if (np.linalg.norm(b[:3, 3] - a[:3, 3]) > args.max_pos_step_m or
                rotation_angle(a[:3, :3], b[:3, :3]) >
                np.radians(args.max_rot_step_deg)):
            critical.update((row, row + 1))
base_times = np.unique(np.r_[base_times, sorted(critical)])
base_tracks = {i: interp_T(raw[i], base_times) for i in (0, 1)}

# Preserve every pose while slowing only jumps that exceed the physical reference
# contract. This is Pour's time-expansion idea, not trajectory attenuation.
source_times = [float(base_times[0])]
for row in range(1, len(base_times)):
    ratio = 1.0
    for i in (0, 1):
        a, b = base_tracks[i][row - 1], base_tracks[i][row]
        ratio = max(ratio,
                    np.linalg.norm(b[:3, 3] - a[:3, 3]) / args.max_pos_step_m,
                    rotation_angle(a[:3, :3], b[:3, :3]) /
                    np.radians(args.max_rot_step_deg))
    count = int(np.ceil(ratio))
    source_times.extend(np.linspace(base_times[row - 1], base_times[row],
                                    count + 1)[1:].tolist())
source_times = np.asarray(source_times, np.float64)
times_s = np.arange(len(source_times), dtype=np.float64) / args.control_hz
resampled = {i: interp_T(raw[i], source_times) for i in (0, 1)}
print(f"[reference] source clock={source_fps:.1f}Hz; regular rows={len(base_times)}; "
      f"time-expanded rows={len(source_times)} ({times_s[-1]:.2f}s)")
for i in (0, 1):
    pos_step = np.linalg.norm(np.diff(resampled[i][:, :3, 3], axis=0), axis=1)
    rot_step = np.asarray([rotation_angle(a[:3, :3], b[:3, :3])
                           for a, b in zip(resampled[i][:-1], resampled[i][1:])])
    print(f"[reference] object_{i} path-preserving step audit: "
          f"pos_max={1000*pos_step.max():.2f}mm "
          f"rot_max={np.degrees(rot_step.max()):.2f}deg")
    assert pos_step.max() <= args.max_pos_step_m + 1e-9
    assert rot_step.max() <= np.radians(args.max_rot_step_deg) + 1e-9

with open(os.path.join(DATA, "scene_layout.json")) as f:
    layout = json.load(f)


def registered_tools(scene_yaw_deg):
    tool_T = {}
    wy = np.radians(scene_yaw_deg)
    R_world = np.array([[np.cos(wy), -np.sin(wy), 0.0],
                        [np.sin(wy), np.cos(wy), 0.0], [0.0, 0.0, 1.0]])
    source_origin = raw[1][0, :3, 3]
    scene_origin = np.array([
        args.broom_start_x,
        args.broom_start_y,
        source_origin[2] + TABLE_Z - float(layout["scene_table_z"]),
    ])
    for i in (0, 1):
        rs = resampled[i]
        out_i = np.zeros_like(rs)
        for t, Tr in enumerate(rs):
            out_i[t] = np.eye(4)
            out_i[t, :3, 3] = scene_origin + R_world @ (Tr[:3, 3] - source_origin)
            out_i[t, :3, :3] = R_world @ Tr[:3, :3]
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

def brush_geometry(tool_T):
    pan0 = tool_T[0][0]
    candidates = []
    # Keep the full cube footprint inside the pan's lateral mouth corridor while
    # allowing the fixed start to align with the reconstructed brush edge.
    for x in np.linspace(-0.05, 0.05, 11):
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


tool_T = registered_tools(args.scene_yaw_deg)
scene = {i: np.r_[tool_T[i][0, :3, 3], R_to_q(tool_T[i][0, :3, :3])]
         for i in (0, 1)}
candidates, best = brush_geometry(tool_T)
cube_start = candidates[best[2]]
contact_row = int(best[1])
brush_contact_local = bv[best[3]]
brush_contact_world = (tool_T[1][contact_row, :3, :3] @ brush_contact_local +
                       tool_T[1][contact_row, :3, 3])
print(f"[reference] shared scene_yaw={args.scene_yaw_deg:.1f}deg "
      f"easy cube={np.round(cube_start, 4).tolist()} | "
      f"nominal brush distance={best[0]*100:.2f}cm @ row {contact_row}")
print(f"[reference] brush_contact_world={np.round(brush_contact_world, 4).tolist()} "
      f"delta_to_cube_cm={np.round((brush_contact_world-cube_start)*100, 3).tolist()}")
if args.geometry_only:
    raise SystemExit(0)

priors = {"right": np.load(args.broom_prior), "left": np.load(args.pan_prior)}
oid = {"right": 1, "left": 0}
arm_q, ik_report, object_hand0 = {}, {}, {}


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
    seed_pe, seed_re = [], []
    for q, p_tgt, quat_tgt in zip(q_seed, P, Q):
        p_fk, R_fk = ik.fk(q)
        seed_pe.append(np.linalg.norm(p_fk - p_tgt))
        seed_re.append(np.linalg.norm(ik._cost(q, p_tgt, q_to_R(quat_tgt), 0.35)[3]))
    print("[reference] sparse-DP spline audit: "
          f"pos_max={100*max(seed_pe):.3f}cm "
          f"rot_max={np.degrees(max(seed_re)):.3f}deg "
          f"joint_step_max={np.degrees(np.abs(np.diff(q_seed, axis=0)).max()):.3f}deg")

    reports = []
    for row, (p_tgt, quat_tgt, q0) in enumerate(zip(P, Q, q_seed)):
        R_tgt = q_to_R(quat_tgt)
        # Track the preceding dense solution as well as the sparse DP spline.
        # Solving every row from the spline alone can cross to another redundant
        # IK branch between key rows even though both neighbouring poses are close.
        local_seeds = [q0]
        if reports:
            local_seeds.append(np.asarray(reports[-1]["q"], np.float64))
        trials = [ik.solve(p_tgt, R_tgt, q0=s, iters=300,
                           pos_tol=0.002, rot_tol=0.02) for s in local_seeds]
        good = [r for r in trials if r["ok"]]
        target = local_seeds[-1]
        if not good:
            trials = [ik.solve(p_tgt, R_tgt, q0=s, iters=350,
                               pos_tol=0.002, rot_tol=0.02)
                      for s in seed_bank]
            good = [r for r in trials if r["ok"]]
        ans = (min(good, key=lambda r: np.linalg.norm(r["q"] - target))
               if good else min(trials, key=lambda r: r["pos_err"]**2
                                  + (0.35*r["rot_err"])**2))
        reports.append(ans)
    return reports


for side in ((args.audit_side,) if args.audit_side else ("right", "left")):
    T_oh = pose_T(priors[side]["grasp"][:3], priors[side]["grasp"][3:7])
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=anchor_T)
    hand_T = [Twt @ T_oh for Twt in tool_T[oid[side]]]
    object_hand0[side] = hand_T[0]
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


def aligned_human_targets(side):
    """Pour first-frame weld: borrow reconstructed motion, never absolute wrist pose."""
    path = os.path.join(DATA, "retarget", f"ref_qpos_{side}.npz")
    zh = np.load(path, allow_pickle=True)
    assert float(np.asarray(zh["fps"])) == source_fps
    wrist = np.asarray([pose_T(p, q) for p, q in
                        zip(zh["wrist_pos"], zh["wrist_quat_wxyz"])])
    wrist = interp_T(wrist, source_times)
    yaw = np.radians(args.scene_yaw_deg)
    R_world = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    anchor = object_hand0[side]
    P, Q = [], []
    for T in wrist:
        out = np.eye(4)
        out[:3, 3] = anchor[:3, 3] + R_world @ (T[:3, 3] - wrist[0, :3, 3])
        dR = R_world @ (T[:3, :3] @ wrist[0, :3, :3].T) @ R_world.T
        out[:3, :3] = dR @ anchor[:3, :3]
        P.append(out[:3, 3])
        Q.append(R_to_q(out[:3, :3]))
    return np.asarray(P), np.asarray(Q)


human_q, human_ik_report = {}, {}
for side in ("right", "left"):
    P, Q = aligned_human_targets(side)
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=anchor_T)
    reports = solve_continuous(ik, P, Q, 20260840 + int(side == "left"))
    qs = np.asarray([answer["q"] for answer in reports], np.float64)
    pe = np.asarray([answer["pos_err"] for answer in reports])
    re = np.asarray([answer["rot_err"] for answer in reports])
    ok = np.asarray([answer["ok"] for answer in reports])
    human_q[side] = qs.astype(np.float32)
    human_ik_report[side] = {
        "ok_ratio": float(ok.mean()),
        "pos_max_cm": float(100 * pe.max()),
        "rot_max_deg": float(np.degrees(re.max())),
        "joint_step_max_deg": float(np.degrees(np.abs(np.diff(qs, axis=0)).max())),
    }
    print(f"[reference] human-shape {side}: {human_ik_report[side]}")

out = {
    "right_q": arm_q["right"], "left_q": arm_q["left"],
    "human_right_q": human_q["right"], "human_left_q": human_q["left"],
    "right_f": np.repeat(priors["right"]["grasp"][None, 7:29], len(times_s), axis=0).astype(np.float32),
    "left_f": np.repeat(priors["left"]["grasp"][None, 7:29], len(times_s), axis=0).astype(np.float32),
    "fin_names": np.array(GENERIC_JOINT_ORDER, dtype=object),
    "source_frame": source_times.astype(np.float32),
    "control_hz": np.float32(args.control_hz),
    "cube_start_w": cube_start.astype(np.float32),
    "contact_row": np.int32(contact_row),
    "nominal_brush_cube_distance_m": np.float32(best[0]),
    "scene_yaw_deg": np.float32(args.scene_yaw_deg),
    "source_fps": np.float32(source_fps),
    "brush_contact_local": brush_contact_local.astype(np.float32),
    "scene_pose_0": scene[0].astype(np.float32),
    "scene_pose_1": scene[1].astype(np.float32),
    "meta": ("Sweep2 P-OBJ/HYB v2; authoritative 15Hz trajectory clock -> 20Hz; "
             "full RTS bimanual 6DoF path; local time expansion only; shared "
             f"scene registration yaw {args.scene_yaw_deg:.1f}deg; GraspPose-locked "
             "frame-zero hands; human wrists first-frame welded for motion-shape only; "
             "fingers fixed"),
}
for i in (0, 1):
    out[f"obj_pos_{i}"] = tool_T[i][:, :3, 3].astype(np.float32)
    out[f"obj_quat_{i}"] = np.asarray([R_to_q(T[:3, :3]) for T in tool_T[i]], np.float32)
    cf = np.minimum(np.asarray(zr[i]["conf_pos"]), np.asarray(zr[i]["conf_rot"]))
    out[f"confidence_{i}"] = np.interp(source_times, np.arange(len(cf)), cf).astype(np.float32)
out["ik_report"] = np.array(str(ik_report))
out["human_ik_report"] = np.array(str(human_ik_report))

assert min(v["ok_ratio"] for v in ik_report.values()) >= 0.99, ik_report
assert max(v["pos_max_cm"] for v in ik_report.values()) <= 0.5, ik_report
assert best[0] <= 0.020, f"retargeted brush misses every easy cube candidate: {best[0]:.4f}m"
assert max(v["joint_step_max_deg"] for v in ik_report.values()) <= 8.0, ik_report
assert min(v["ok_ratio"] for v in human_ik_report.values()) >= 0.99, human_ik_report
assert max(v["pos_max_cm"] for v in human_ik_report.values()) <= 1.0, human_ik_report
assert max(v["joint_step_max_deg"] for v in human_ik_report.values()) <= 12.0, human_ik_report
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
np.savez(args.output, **out)
print(f"[reference] wrote validated {args.output}: rows={len(times_s)}, "
      f"duration={times_s[-1]:.3f}s")
