"""A/B the immutable v1 replay with one right-hand depth extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--baseline", required=True)
p.add_argument("--trace", required=True)
p.add_argument("--metrics", required=True)
p.add_argument("--video", required=True)
p.add_argument("--depth_mm", type=float, required=True)
p.add_argument("--max_bound_excess_deg", type=float, default=3.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_v1_smooth_entry_ab")
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
LOGS = os.path.join(ROOT, "logs")
VIDEOS = os.path.join(ROOT, "outputs_video")
DEPTH = float(args.depth_mm) / 1000.0
assert DEPTH > 0.0


def under(root: str, path: str) -> str:
    path = os.path.abspath(path)
    assert os.path.commonpath([root, path]) == root, (root, path)
    return path


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def containment_deficit(c: np.ndarray, g: SE.SweepGeometry) -> np.ndarray:
    dx = np.maximum(np.abs(c[:, 0]) + g.cube_half - g.pan_half_width, 0.0)
    dz0 = np.maximum(g.pan_inside_z_min - (c[:, 2] - g.cube_half), 0.0)
    dz1 = np.maximum((c[:, 2] + g.cube_half) - g.pan_mouth_z, 0.0)
    dy0 = np.maximum(g.pan_center_y_min - c[:, 1], 0.0)
    dy1 = np.maximum(c[:, 1] - g.pan_center_y_max, 0.0)
    return np.sqrt(dx*dx + dz0*dz0 + dz1*dz1 + dy0*dy0 + dy1*dy1)


baseline_path = under(LOGS, args.baseline)
trace_path = under(LOGS, args.trace)
metrics_path = under(LOGS, args.metrics)
video_path = under(VIDEOS, args.video)
base = np.load(baseline_path)
actions = np.asarray(base["actions"], dtype=np.float32)
rows = np.asarray(base["rows"], dtype=np.int64)
assert actions.shape == (500, 14)
assert np.array_equal(rows, np.arange(500))
expected_action_sha = sha(actions)
expected_right_sha = "ddd14c4e18a0889074d88817907b2344a90bcf87da0b3c9b8ce644d7d0f0b522"
assert sha(actions[:, :7]) == expected_right_sha
assert np.count_nonzero(actions[:, 7:]) == 0

raw = SE.SweepEnv(SE.build_cfg(1))
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
assert raw.T >= len(rows)
pan_asset = clips.clip_entry("Sweep2_broom")["secondary"]["mesh"]
assert pan_asset.endswith("tasks/Sweep/2/assets/dustpan_smooth_entry/object_mesh_scaled_final.obj")
assert os.path.isfile(pan_asset)

# Freeze both reconstructed arm/tool trajectories.  The script never writes them.
arm_ref_before = raw.ref_arm.detach().clone()
obj_pos_before = {i: raw.ref_pos[i].detach().clone() for i in (0, 1)}
obj_quat_before = {i: raw.ref_quat[i].detach().clone() for i in (0, 1)}
arm_ref_sha = sha(arm_ref_before.cpu().numpy())
tool_ref_sha = {f"pos_{i}": sha(obj_pos_before[i].cpu().numpy()) for i in (0, 1)}
tool_ref_sha |= {f"quat_{i}": sha(obj_quat_before[i].cpu().numpy()) for i in (0, 1)}


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


# Reconstruct the archived v1 cumulative residual exactly, then alter only the
# right hand after row370.  The Cartesian offset is expressed in each row's pan
# frame: the requested inward depth (-z), then hold.  Retraction/lift is intentionally absent
# from this first A/B so the sole changed physical variable is insertion depth.
nominal_cum = np.zeros((len(rows), 14), dtype=np.float64)
dev_schedule = np.zeros((len(rows), 14), dtype=np.float64)
cum = np.zeros(14, dtype=np.float64)
conf_np = raw.conf.detach().cpu().numpy()
step_hi = raw.step_hi.detach().cpu().numpy()
step_lo = raw.step_lo.detach().cpu().numpy()
dev_hi = raw.dev_hi.detach().cpu().numpy()
dev_lo = raw.dev_lo.detach().cpu().numpy()
for t, row in enumerate(rows):
    c14 = np.r_[np.repeat(conf_np[row, 0], 7),
                np.repeat(conf_np[row, 1], 7)]
    step = step_hi + c14 * (step_lo - step_hi)
    dev = dev_hi + c14 * (dev_lo - dev_hi)
    cum = np.clip(cum + actions[t] * step, -dev, dev)
    nominal_cum[t] = cum
    dev_schedule[t] = dev

ik = ArmIK("right", anchor_link="arm_center", anchor_T=raw._anchor_T)
planned_residual = nominal_cum.copy()
plan_pos_err, plan_rot_err, plan_depth, plan_lift = [], [], [], []
prev_q = None
for t, row in enumerate(rows):
    depth = (0.0 if row < 370 else
             DEPTH * smoothstep((row - 370) / 20.0) if row <= 390 else
             DEPTH if row <= 420 else
             DEPTH * (1.0 - smoothstep((row - 420) / 30.0))
             if row <= 450 else 0.0)
    lift = 0.0
    plan_depth.append(depth)
    plan_lift.append(lift)
    if depth == 0.0 and lift == 0.0:
        plan_pos_err.append(0.0); plan_rot_err.append(0.0)
        continue
    q_nom = (arm_ref_before[row, :7].cpu().numpy().astype(np.float64)
             + nominal_cum[t, :7])
    p_nom, R_nom = ik.fk(q_nom)
    pan_R = quat_to_R(obj_quat_before[0][row].cpu().numpy())
    p_tgt = p_nom + pan_R @ np.array([0.0, lift, -depth])
    seeds = [q_nom] + ([] if prev_q is None else [prev_q])
    trials = [ik.solve(p_tgt, R_nom, q0=seed, iters=300,
                       pos_tol=0.002, rot_tol=0.02) for seed in seeds]
    good = [ans for ans in trials if ans["ok"]]
    assert good, (row, [(ans["pos_err"], ans["rot_err"]) for ans in trials])
    ans = min(good, key=lambda item: np.linalg.norm(item["q"] - q_nom))
    prev_q = np.asarray(ans["q"], dtype=np.float64)
    planned_residual[t, :7] = prev_q - arm_ref_before[row, :7].cpu().numpy()
    plan_pos_err.append(float(ans["pos_err"]))
    plan_rot_err.append(float(ans["rot_err"]))

assert np.count_nonzero(planned_residual[:, 7:]) == 0
assert np.array_equal(planned_residual[:370], nominal_cum[:370])
right_dev = np.maximum(
    np.abs(planned_residual[:, :7]) - dev_schedule[:, :7], 0.0)
worst = np.unravel_index(int(np.argmax(right_dev)), right_dev.shape)
print("[right-depth-bound] " + json.dumps({
    "row": int(rows[worst[0]]),
    "joint": int(worst[1]),
    "excess_deg": float(np.degrees(right_dev[worst])),
    "planned_deg": float(np.degrees(planned_residual[worst[0], worst[1]])),
    "bound_deg": float(np.degrees(dev_schedule[worst[0], worst[1]])),
    "depth_mm": float(1000.0 * plan_depth[worst[0]]),
    "lift_mm": float(1000.0 * plan_lift[worst[0]]),
}, sort_keys=True))
assert float(right_dev.max()) <= np.radians(args.max_bound_excess_deg), float(right_dev.max())
joint_step = np.abs(np.diff(planned_residual[:, :7], axis=0))
print("[right-depth-plan] " + json.dumps({
    "max_pos_err_mm": 1000.0 * max(plan_pos_err),
    "max_rot_err_deg": float(np.degrees(max(plan_rot_err))),
    "max_abs_residual_deg": float(np.degrees(np.abs(planned_residual[:, :7]).max())),
    "max_residual_step_deg": float(np.degrees(joint_step.max())),
}, sort_keys=True))

# The reference NPZ's easy-start point was edited after v1.  Restore the actual
# v1 frame-zero cube world position from the archived observation, so cube setup
# is controlled rather than silently becoming a second experimental variable.
v1_cube_start = torch.tensor(base["obs"][0, 138:141], dtype=torch.float32,
                             device=raw.device)
raw.cube_start_ref[:] = v1_cube_start
obs = env.reset()

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

stage = omni.usd.get_context().get_stage()
entry_collider = stage.GetPrimAtPath(
    "/World/envs/env_0/Aux/SweepOpenCollision/ramp")
assert entry_collider.IsValid()
assert bool(UsdPhysics.CollisionAPI(entry_collider)
            .GetCollisionEnabledAttr().Get())

# Check that selecting the repaired asset did not mutate any command trajectory.
assert torch.equal(raw.ref_arm, arm_ref_before)
for i in (0, 1):
    assert torch.equal(raw.ref_pos[i], obj_pos_before[i])
    assert torch.equal(raw.ref_quat[i], obj_quat_before[i])

# The same camera as the archived expert generator keeps visual comparison direct.
cam = UsdGeom.Camera.Define(stage, "/World/SweepV1SmoothEntryCam")
cam.CreateFocalLengthAttr().Set(18.0)
m = Gf.Matrix4d(); m.SetLookAt(Gf.Vec3d(0.95, -1.15, 1.45),
                               Gf.Vec3d(0.25, 0.0, 0.90), Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
rp = rep.create.render_product(str(cam.GetPath()), (1280, 720))
annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach(rp)

raw.force_replay = True
raw.progress.reset(torch.tensor([0], dtype=torch.long, device=raw.device))
raw.row[0] = 0
raw.cum_res.zero_()
raw.last_act.zero_()
raw.episode_length_buf[0] = 0

frames, cube_pan, cube_world = [], [], []
pan_world, broom_world, gates, stable_run = [], [], [], []
entered, fully_inside, rel_speed, rewards = [], [], [], []
executed_actions, executed_rows = [], []
success_frame = None
success_cube_pan_post = None

with torch.no_grad():
    for t, row in enumerate(rows):
        raw.row[0] = int(row)
        # Pre-step sampling matches the archived v1 trace convention.
        sig, _, _ = raw._signals()
        cube_pan.append(sig["cube_pan"][0].cpu().numpy().copy())
        cube_world.append((raw.cube.data.root_pos_w[0] - raw.scene.env_origins[0]).cpu().numpy().copy())
        pan_world.append((raw.aux.data.root_pos_w[0] - raw.scene.env_origins[0]).cpu().numpy().copy())
        broom_world.append((raw.object.data.root_pos_w[0] - raw.scene.env_origins[0]).cpu().numpy().copy())

        confidence = raw.conf[int(row)]
        c14 = torch.cat([confidence[:1].expand(7), confidence[1:].expand(7)])
        step = raw.step_hi + c14 * (raw.step_lo - raw.step_hi)
        goal = torch.tensor(planned_residual[t], dtype=torch.float32,
                            device=raw.device)
        action = ((goal - raw.cum_res[0]) / step).clamp(-1.0, 1.0).unsqueeze(0)
        executed_actions.append(action[0].cpu().numpy().copy())
        executed_rows.append(int(raw.row[0]))
        obs, reward, _, _ = env.step(action)
        tick = raw._tick_out
        gates.append(tick["gates"][0].cpu().numpy().copy())
        stable_run.append(int(tick["stable_run"][0]))
        entered.append(bool(tick["entered"][0]))
        fully_inside.append(bool(tick["fully_inside"][0]))
        rel_speed.append(float(tick["rel_speed"][0]))
        rewards.append(float(reward[0]))
        raw.sim.render()
        image = annot.get_data()
        if image is not None and getattr(image, "size", 0):
            frames.append(np.asarray(image)[..., :3].astype(np.uint8))
        if bool(tick["success"][0]):
            success_frame = t
            success_cube_pan_post = tick["cube_pan"][0].cpu().numpy().copy()
            break

executed_actions = np.asarray(executed_actions, dtype=np.float32)
executed_rows = np.asarray(executed_rows, dtype=np.int64)
assert np.array_equal(executed_rows, rows[:len(executed_rows)])
assert np.count_nonzero(executed_actions[:, 7:]) == 0
assert torch.equal(raw.ref_arm, arm_ref_before)
for i in (0, 1):
    assert torch.equal(raw.ref_pos[i], obj_pos_before[i])
    assert torch.equal(raw.ref_quat[i], obj_quat_before[i])

cube_pan = np.asarray(cube_pan, dtype=np.float32)
gates = np.asarray(gates, dtype=bool)
stable_run = np.asarray(stable_run, dtype=np.int64)
fully_inside = np.asarray(fully_inside, dtype=bool)
entered = np.asarray(entered, dtype=bool)
assert success_frame is not None, "planned expert never crossed the entry success boundary"
base_cube_pan = np.asarray(base["priv_info"][:, :3], dtype=np.float32)
a_def = containment_deficit(base_cube_pan, raw.geometry)
b_def = containment_deficit(cube_pan, raw.geometry)
a_best, b_best = int(np.argmin(a_def)), int(np.argmin(b_def))
metrics = {
    "comparison": f"A=archived v1; B=same scene/left/cube with only a right-hand pan-local {args.depth_mm:g}mm depth extension, stopping on first entry success",
    "action_sha256_A": expected_action_sha,
    "action_sha256_B": sha(executed_actions),
    "right_action_sha256_A": expected_right_sha,
    "right_action_sha256_B": sha(executed_actions[:, :7]),
    "left_actions_all_zero": bool(np.count_nonzero(executed_actions[:, 7:]) == 0),
    "right_plan": {
        "depth_ramp_rows": [370, 390],
        "depth_hold_m": DEPTH,
        "ik_solver": f"ArmIK with URDF limits; confidence residual bound may exceed by at most {args.max_bound_excess_deg:g}deg",
        "residual_bound_excess_max_deg": float(np.degrees(right_dev.max())),
        "depth_hold_rows": [390, 420],
        "depth_release_rows": [420, 450],
        "lift_m": 0.0,
        "ik_pos_max_mm": float(1000.0 * max(plan_pos_err)),
        "ik_rot_max_deg": float(np.degrees(max(plan_rot_err))),
        "residual_abs_max_deg": float(np.degrees(np.abs(planned_residual[:, :7]).max())),
        "residual_step_max_deg": float(np.degrees(joint_step.max())),
    },
    "arm_reference_sha256_before_after": arm_ref_sha,
    "tool_reference_sha256": tool_ref_sha,
    "rows_exact_prefix": bool(np.array_equal(executed_rows, rows[:len(executed_rows)])),
    "v1_cube_start_world_m": v1_cube_start.cpu().numpy().tolist(),
    "dustpan_asset": os.path.relpath(pan_asset, ROOT),
    "dustpan_asset_sha256": file_sha(pan_asset),
    "entry_collider": {
        "z_span_mm": [80.0, 108.0],
        "top_height_mm": [8.5, 6.0],
        "enabled": True,
    },
    "A_best_frame": a_best,
    "A_best_deficit_mm": float(a_def[a_best] * 1000.0),
    "A_best_cube_pan_mm": (base_cube_pan[a_best] * 1000.0).tolist(),
    "B_best_frame": b_best,
    "B_best_deficit_mm": float(b_def[b_best] * 1000.0),
    "B_best_cube_pan_mm": (cube_pan[b_best] * 1000.0).tolist(),
    "B_gate_first_frames": [int(np.flatnonzero(gates[:, i])[0]) if gates[:, i].any() else None
                            for i in range(4)],
    "B_ever_entered": bool(entered.any()),
    "B_ever_fully_inside": bool(fully_inside.any()),
    "B_max_stable_run": int(stable_run.max()),
    "B_success": bool(gates[:, 3].any()),
    "success_frame": int(success_frame),
    "success_cube_pan_post_mm": (success_cube_pan_post * 1000.0).tolist(),
    "frames": len(frames),
}

for path in (trace_path, metrics_path, video_path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
np.savez(trace_path, rows=executed_rows, actions=executed_actions,
         planned_residual=planned_residual[:len(executed_rows)].astype(np.float32),
         plan_depth=np.asarray(plan_depth[:len(executed_rows)], dtype=np.float32),
         plan_lift=np.asarray(plan_lift[:len(executed_rows)], dtype=np.float32),
         cube_pan=cube_pan, cube_world=np.asarray(cube_world),
         pan_world=np.asarray(pan_world), broom_world=np.asarray(broom_world),
         gates=gates, stable_run=stable_run, entered=entered, fully_inside=fully_inside,
         rel_speed=np.asarray(rel_speed), rewards=np.asarray(rewards),
         baseline_cube_pan=base_cube_pan)
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
imageio.mimsave(video_path, frames, fps=20)
print("[v1-smooth-entry-ab] " + json.dumps(metrics, sort_keys=True))
print(f"[v1-smooth-entry-ab] wrote {trace_path}, {metrics_path}, {video_path}")
try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
os._exit(0)
