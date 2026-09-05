#!/usr/bin/env python3
"""Physical zero-residual safety gate for the five Sweep2 cube variants."""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--variants", required=True)
p.add_argument("--out", required=True)
p.add_argument("--relative-tolerance", type=float, default=1.15)
p.add_argument("--max-single-step-mm", type=float, default=30.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.headless
os.environ["SWEEP_CUBE_VARIANTS_NPZ"] = os.path.abspath(args.variants)

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_cube_variant_validation")
app = AppLauncher(args).app

import torch  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

raw = SE.SweepEnv(SE.build_cfg(5, ablation_method="full"))
raw.force_replay = True
raw.suppress_terminal_reset = True
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
obs = env.reset()
start = raw.cube.data.root_pos_w.clone()
max_pre_disp = torch.zeros(5, device=raw.device)
max_pre_speed = torch.zeros(5, device=raw.device)
max_speed = torch.zeros(5, device=raw.device)
max_step = torch.zeros(5, device=raw.device)
prev = start.clone()
gates = torch.zeros(5, 4, dtype=torch.bool, device=raw.device)
terminated = torch.zeros(5, dtype=torch.bool, device=raw.device)

with torch.no_grad():
    for step in range(raw.T + 40):
        action = torch.zeros(5, SE.ACT_DIM, device=raw.device)
        obs, reward, done, info = env.step(action)
        pos = raw.cube.data.root_pos_w.clone()
        speed = torch.linalg.vector_norm(raw.cube.data.root_lin_vel_w, dim=1)
        delta = torch.linalg.vector_norm(pos - prev, dim=1)
        max_speed = torch.maximum(max_speed, speed)
        max_step = torch.maximum(max_step, delta)
        if step < SE.SCRIPTED_PRELUDE_STEPS:
            max_pre_disp = torch.maximum(max_pre_disp,
                                         torch.linalg.vector_norm(pos - start, dim=1))
            max_pre_speed = torch.maximum(max_pre_speed, speed)
        gates |= raw._tick_out["gates"]
        terminated |= raw._tick_out["terminated"] | raw._tick_out["timeout"]
        prev = pos

baseline = {
    "max_prelude_displacement_mm": float(max_pre_disp[0] * 1000),
    "max_prelude_speed_mps": float(max_pre_speed[0]),
    "max_trajectory_speed_mps": float(max_speed[0]),
    "max_single_step_displacement_mm": float(max_step[0] * 1000),
}
report = {
    "schema": 2,
    "criterion": "shifted variants must not launch more severely than center",
    "relative_tolerance": args.relative_tolerance,
    "max_single_step_mm": args.max_single_step_mm,
    "baseline_variant_id": raw.cube_variant_ids[0],
    "baseline_metrics": baseline,
    "variants": [],
    "all_pass": True,
}
for i, name in enumerate(raw.cube_variant_ids):
    failures = []
    pre_disp_mm = float(max_pre_disp[i] * 1000)
    pre_speed = float(max_pre_speed[i])
    trajectory_speed = float(max_speed[i])
    step_mm = float(max_step[i] * 1000)
    reset_error_mm = float(torch.linalg.vector_norm(
        start[i] - (raw.cube_start_variants[i] + raw.scene.env_origins[i])) * 1000)
    if reset_error_mm > 0.01:
        failures.append("environment did not apply configured cube start")
    if i > 0 and pre_disp_mm > baseline["max_prelude_displacement_mm"] * args.relative_tolerance:
        failures.append("prelude displacement exceeds center baseline tolerance")
    if i > 0 and pre_speed > baseline["max_prelude_speed_mps"] * args.relative_tolerance:
        failures.append("prelude speed exceeds center baseline tolerance")
    if i > 0 and trajectory_speed > baseline["max_trajectory_speed_mps"] * args.relative_tolerance:
        failures.append("trajectory speed exceeds center baseline tolerance")
    if i > 0 and step_mm > args.max_single_step_mm:
        failures.append("cube made an implausible single-step jump")
    item = {"variant_id": name,
            "configured_start_world_m": raw.cube_start_variants[i].cpu().tolist(),
            "reset_position_error_mm": reset_error_mm,
            "max_prelude_displacement_mm": pre_disp_mm,
            "max_prelude_speed_mps": pre_speed,
            "max_trajectory_speed_mps": trajectory_speed,
            "max_single_step_displacement_mm": step_mm,
            "gates": gates[i].int().cpu().tolist(),
            "terminated": bool(terminated[i]),
            "status": ("BASELINE" if i == 0 and not failures else
                       "PASS" if not failures else "FAIL"),
            "failures": failures}
    report["variants"].append(item)
    if i > 0:
        report["all_pass"] &= not failures

out = os.path.abspath(args.out)
logs = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../logs"))
assert os.path.commonpath([logs, out]) == logs
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps(report, indent=2))
if not report["all_pass"]:
    raise SystemExit(2)
try: _slot.release()
except Exception: pass
sys.stdout.flush()
os._exit(0)
