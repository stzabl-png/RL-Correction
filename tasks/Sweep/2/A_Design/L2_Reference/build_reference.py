"""Build the 20 Hz Sweep2 P-OBJ reference from reconstructed tool trajectories.

The dustpan's first RTS pose is mapped to the verified robot scene pose with one
shared SE(3) transform.  The same transform is applied to the broom, preserving the
reconstructed two-tool relationship.  Each tool pose is converted to a hand pose
with its GraspPose ``T_object_hand`` and solved by continuous-seed arm IK.  Fingers
are fixed at the selected grasp throughout; reconstructed human finger motion is
not part of P-OBJ.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--output", default="tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz")
p.add_argument("--source_fps", type=float, default=30.0)
p.add_argument("--control_hz", type=float, default=20.0)
p.add_argument("--broom_prior", default="tasks/pregrasp/priors/Sweep2_broom.npz")
p.add_argument("--pan_prior", default="tasks/pregrasp/priors/Sweep2_dustpan.npz")
p.add_argument("--broom_yaw", type=float, default=30.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_build_reference")
app = AppLauncher(args).app

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
DATA = os.path.join(ROOT, "datasets", "sweep_2_better")


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


cfg = GraspTaskCfg()
clips.configure_cfg(cfg, "Sweep2_broom")
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.broom_prior, args.broom_yaw, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()

# Scene pose used only as a reachable global anchor.  All relative two-tool motion
# remains the reconstruction's because one transform is shared by both tracks.
scene_pan = E.aux.data.root_state_w[0, :7].detach().cpu().numpy().astype(np.float64)
T_scene_pan = pose_T(scene_pan[:3], scene_pan[3:7])
zr = {i: load_rts(i) for i in (0, 1)}
raw = {i: np.asarray(zr[i]["object_ob_in_world_smooth"], np.float64) for i in (0, 1)}
A = T_scene_pan @ np.linalg.inv(raw[0][0])

duration = (len(raw[0]) - 1) / args.source_fps
times_s = np.arange(0.0, duration + 1e-9, 1.0 / args.control_hz)
source_times = times_s * args.source_fps
tool_T = {i: np.einsum("ab,tbc->tac", A, interp_T(raw[i], source_times)) for i in (0, 1)}

priors = {"right": np.load(args.broom_prior), "left": np.load(args.pan_prior)}
oid = {"right": 1, "left": 0}
arm_q, ik_report = {}, {}
for side in ("right", "left"):
    T_oh = pose_T(priors[side]["grasp"][:3], priors[side]["grasp"][3:7])
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=E._anchor_T)
    seed = None; qs = []; pe = []; re = []; ok = []
    for Twt in tool_T[oid[side]]:
        Twh = Twt @ T_oh
        ans = ik.solve(Twh[:3, 3], Twh[:3, :3], q0=seed, iters=250)
        seed = np.asarray(ans["q"], np.float64)
        qs.append(seed); pe.append(float(ans["pos_err"])); re.append(float(ans["rot_err"])); ok.append(bool(ans["ok"]))
    arm_q[side] = np.asarray(qs, np.float32)
    ik_report[side] = dict(ok_ratio=float(np.mean(ok)), pos_max_cm=100*max(pe),
                           rot_max_deg=float(np.degrees(max(re))),
                           joint_step_max_deg=float(np.degrees(np.abs(np.diff(qs, axis=0)).max())))
    print(f"[reference] {side}: {ik_report[side]}")

out = {
    "right_q": arm_q["right"], "left_q": arm_q["left"],
    "right_f": np.repeat(priors["right"]["grasp"][None, 7:29], len(times_s), axis=0).astype(np.float32),
    "left_f": np.repeat(priors["left"]["grasp"][None, 7:29], len(times_s), axis=0).astype(np.float32),
    "fin_names": np.array(GENERIC_JOINT_ORDER, dtype=object),
    "source_frame": source_times.astype(np.float32),
    "control_hz": np.float32(args.control_hz),
    "shared_world_transform": A,
    "meta": "Sweep2 P-OBJ v1; RTS tools at source 30Hz -> explicit 20Hz; shared SE3 scene map; GraspPose-locked tool-to-hand; fingers fixed",
}
for i in (0, 1):
    out[f"obj_pos_{i}"] = tool_T[i][:, :3, 3].astype(np.float32)
    out[f"obj_quat_{i}"] = np.asarray([R_to_q(T[:3, :3]) for T in tool_T[i]], np.float32)
    cf = np.minimum(np.asarray(zr[i]["conf_pos"]), np.asarray(zr[i]["conf_rot"]))
    out[f"confidence_{i}"] = np.interp(source_times, np.arange(len(cf)), cf).astype(np.float32)
out["ik_report"] = np.array(str(ik_report))

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
np.savez(args.output, **out)
print(f"[reference] wrote {args.output}: rows={len(times_s)}, duration={times_s[-1]:.3f}s")
assert min(v["ok_ratio"] for v in ik_report.values()) >= 0.99, ik_report
assert max(v["pos_max_cm"] for v in ik_report.values()) <= 0.5, ik_report
try: _slot.release()
except Exception: pass
app.close()
