"""Plan and physically validate a successful residual expert for actor BC.

The reconstructed reference remains unchanged.  This script constructs a separate
right-arm residual around nominal brush contact: align the selected brush-head point
behind the fixed cube, then push it across the dustpan lip while the left arm follows
the reconstructed pan reference.  Only a simulator rollout that reaches the exact
stable-containment terminal is saved as expert data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--out", default="logs/expert/sweep2_success_v1.npz")
p.add_argument("--video", default="outputs_video/sweep2_expert_v1.mp4")
p.add_argument("--steps", type=int, default=360)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_make_expert")
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
out = os.path.abspath(args.out); video = os.path.abspath(args.video)
assert os.path.commonpath([os.path.join(ROOT, "logs"), out]) == os.path.join(ROOT, "logs")
assert os.path.commonpath([os.path.join(ROOT, "outputs_video"), video]) == os.path.join(ROOT, "outputs_video")

raw = SE.SweepEnv(SE.build_cfg(1)); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
z = raw._z


def T_pose(pos, quat):
    w, x, y, zz = quat / np.linalg.norm(quat)
    R = np.array([[1-2*(y*y+zz*zz), 2*(x*y-w*zz), 2*(x*zz+w*y)],
                  [2*(x*y+w*zz), 1-2*(x*x+zz*zz), 2*(y*zz-w*x)],
                  [2*(x*zz-w*y), 2*(y*zz+w*x), 1-2*(x*x+y*y)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos; return T


def smooth(x): return x*x*(3-2*x)


def plan(depth_z, lateral_x):
    Tn = raw.T; cr = raw.contact_row
    pan0 = T_pose(np.asarray(z["obj_pos_0"])[0], np.asarray(z["obj_quat_0"])[0])
    broom = [T_pose(p, q) for p, q in zip(z["obj_pos_1"], z["obj_quat_1"])]
    cube = np.asarray(z["cube_start_w"], np.float64)
    brush = np.asarray(z["brush_contact_local"], np.float64)
    Rb = broom[cr][:3, :3]
    outward = pan0[:3, :3] @ np.array([0., 0., 1.])
    p_out = cube + 0.012 * outward
    c_local = pan0[:3, :3].T @ (cube - pan0[:3, 3])
    goal_local = np.array([c_local[0] + lateral_x, c_local[1], depth_z])
    p_goal = (pan0 @ np.r_[goal_local, 1.])[:3]
    p_goal[2] = cube[2]
    a0, a1, a2 = max(0, cr - 24), cr, min(Tn - 1, cr + 55)
    desired = [x.copy() for x in broom]
    root_out = p_out - Rb @ brush
    root_goal = p_goal - Rb @ brush
    for t in range(a0, a1 + 1):
        s = smooth((t-a0)/max(a1-a0, 1))
        desired[t][:3, :3] = Rb
        desired[t][:3, 3] = (1-s)*broom[a0][:3, 3] + s*root_out
    for t in range(a1, a2 + 1):
        s = smooth((t-a1)/max(a2-a1, 1))
        desired[t][:3, :3] = Rb
        desired[t][:3, 3] = (1-s)*root_out + s*root_goal
    for t in range(a2 + 1, Tn): desired[t] = desired[a2].copy()
    prior = np.load(SE.PRIOR_BROOM)
    Toh = T_pose(prior["grasp"][:3], prior["grasp"][3:7])
    ik = ArmIK("right", anchor_link="arm_center", anchor_T=raw._anchor_T)
    seed = np.asarray(z["right_q"])[0]; qr = []
    report = {"pos_max_cm": 0., "rot_max_deg": 0., "ok": 0}
    for Tb in desired:
        Th = Tb @ Toh
        ans = ik.solve(Th[:3, 3], Th[:3, :3], q0=seed, iters=250)
        seed = np.asarray(ans["q"], np.float64); qr.append(seed)
        report["ok"] += int(ans["ok"])
        report["pos_max_cm"] = max(report["pos_max_cm"], 100*float(ans["pos_err"]))
        report["rot_max_deg"] = max(report["rot_max_deg"], float(np.degrees(ans["rot_err"])))
    q = np.c_[np.asarray(qr), np.asarray(z["left_q"])]
    ref = np.c_[np.asarray(z["right_q"]), np.asarray(z["left_q"])]
    residual = q - ref
    report["ok_ratio"] = report.pop("ok") / Tn
    report["residual_abs_max"] = float(np.abs(residual).max())
    return residual.astype(np.float32), report


def rollout(residual, annot=None):
    obs = env.reset(); O = []; P = []; A = []; R = []; rows = []; frames = []
    for t in range(args.steps):
        row = int(raw.row[0]); conf = raw.conf[row]
        c14 = torch.cat([conf[:1].expand(7), conf[1:].expand(7)])
        step = raw.step_hi + c14 * (raw.step_lo - raw.step_hi)
        goal = torch.tensor(residual[row], device=raw.device)
        action = ((goal - raw.cum_res[0]) / step).clamp(-1, 1).unsqueeze(0)
        O.append(obs["obs"][0].detach().cpu().numpy())
        P.append(obs["priv_info"][0].detach().cpu().numpy())
        A.append(action[0].detach().cpu().numpy()); rows.append(row)
        obs, reward, done, info = env.step(action); R.append(float(reward[0]))
        if annot is not None:
            raw.sim.render(); image = annot.get_data()
            if image is not None and getattr(image, "size", 0):
                frames.append(np.asarray(image)[..., :3].astype(np.uint8))
        if bool(done[0]):
            success = bool(raw._tick_out["success"][0])
            return success, dict(obs=np.asarray(O), priv_info=np.asarray(P),
                                 actions=np.asarray(A), rewards=np.asarray(R),
                                 rows=np.asarray(rows)), frames
    return False, dict(obs=np.asarray(O), priv_info=np.asarray(P), actions=np.asarray(A),
                       rewards=np.asarray(R), rows=np.asarray(rows)), frames


# Render-product setup is delayed until after planning so failed static plans do not
# create project outputs. Frames are captured only for the accepted rollout below.
attempts = []
winner = None
winner_residual = None
for depth in (0.075, 0.065, 0.055):
    for lateral in (0.0, -0.01, 0.01):
        residual, report = plan(depth, lateral)
        ok, data, _ = rollout(residual)
        attempts.append({"depth_z": depth, "lateral_x": lateral,
                         "success": ok, "plan": report,
                         "max_gate": raw.progress.gates[0].int().tolist()})
        print(f"[expert] depth={depth:.3f} lateral={lateral:+.3f} success={ok} {report}")
        if ok:
            winner = data; winner_residual = residual; break
    if winner is not None: break

os.makedirs(os.path.join(ROOT, "logs", "expert"), exist_ok=True)
with open(os.path.join(ROOT, "logs", "expert", "planner_attempts.json"), "w") as f:
    json.dump(attempts, f, indent=2)
if winner is None:
    np.savez(os.path.join(ROOT, "logs", "expert", "last_failed_trace.npz"), **data)
    raise RuntimeError("no physically successful expert; see logs/expert/planner_attempts.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, **winner, meta="physically validated stable-containment Sweep2 expert for one-shot actor BC")
print(f"[expert] wrote {out}: transitions={len(winner['actions'])}")

# Re-run the accepted plan once for its required project-root video artifact.
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402
stage = omni.usd.get_context().get_stage()
cam = UsdGeom.Camera.Define(stage, "/World/SweepExpertCam")
cam.CreateFocalLengthAttr().Set(18.0)
m = Gf.Matrix4d(); m.SetLookAt(Gf.Vec3d(0.95, -1.15, 1.45),
                               Gf.Vec3d(0.25, 0.0, 0.90), Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
rp = rep.create.render_product("/World/SweepExpertCam", (1280, 720))
annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach(rp)
ok_video, _, frames = rollout(winner_residual, annot=annot)
assert ok_video, "accepted expert failed deterministic video replay"
os.makedirs(os.path.dirname(video), exist_ok=True)
imageio.mimsave(video, frames, fps=20)
print(f"[expert] wrote {video}: frames={len(frames)}")
try: _slot.release()
except Exception: pass
app.close()
