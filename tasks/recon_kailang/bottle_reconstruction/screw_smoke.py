"""Headless physical audit for both PCO-1810 screw-cap task variants."""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="water_bottle_twist_assembled")
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--torque-nm", type=float, default=1.0e-6)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("bottle_screw_smoke")
app = AppLauncher(args).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.utils.math import quat_apply as torch_quat_apply  # noqa: E402

from rl_rebuild.correction import clips, frames as F  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.env import (  # noqa: E402
    BottleReconstructionEnv,
)
from tasks.recon_kailang.bottle_reconstruction.screw_joint import (  # noqa: E402
    ScrewSpec,
    helical_travel,
)


def _relative_state(env, spec: ScrewSpec) -> tuple[float, float, float]:
    """Return raw twist angle, axial travel and radial drift in body frame."""

    body_pos = env.object.data.root_pos_w[0].detach().cpu().numpy()
    body_quat = env.object.data.root_quat_w[0].detach().cpu().numpy()
    cap_pos = env.cap.data.root_pos_w[0].detach().cpu().numpy()
    cap_quat = env.cap.data.root_quat_w[0].detach().cpu().numpy()
    conjugate = body_quat * np.array([1.0, -1.0, -1.0, -1.0])
    relative_quat = F.quat_mul(conjugate, cap_quat)
    relative_quat *= 1.0 if relative_quat[0] >= 0.0 else -1.0
    angle = 2.0 * np.arctan2(relative_quat[3], relative_quat[0])
    local_offset = F.rot_apply(conjugate, cap_pos - body_pos)
    axial = float(local_offset[2] - spec.closed_offset_m)
    radial = float(np.linalg.norm(local_offset[:2]))
    return float(angle), axial, radial


def _joint_report(env) -> dict:
    stage = omni.usd.get_context().get_stage()
    paths = env.screw_metadata_paths[0]
    prim = stage.GetPrimAtPath(paths["metadata"])
    return {
        "paths": paths,
        "constraint_type": "gpu_tensor_helical_projection",
        "usd_metadata_type": prim.GetTypeName(),
        "lead_m_per_rad": env.screw_spec.pitch_m / (2.0 * np.pi),
        "mode": env.screw_spec.mode,
    }


def _stage_capture_entry(env, spec: ScrewSpec) -> None:
    """Place every free cap at the thread entrance for capture auditing."""

    axis_local = torch.zeros(args.num_envs, 3, device=env.device)
    axis_local[:, 2] = 1.0
    axis_w = torch.nn.functional.normalize(
        torch_quat_apply(env.object.data.root_quat_w, axis_local), dim=1
    )
    cap_pos = env.object.data.root_pos_w + (
        spec.closed_offset_m + spec.travel_m
    ) * axis_w
    cap_pose = torch.cat([cap_pos, env.object.data.root_quat_w], dim=1)
    env.cap.write_root_pose_to_sim(cap_pose)
    env.cap.write_root_velocity_to_sim(
        torch.zeros(args.num_envs, 6, device=env.device)
    )
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)
    env.apply_screw_constraint(integrate_angle=False)


def _audit_free_cap_barrier(env) -> bool:
    """Return whether a misaligned free cap is blocked above the bottle."""

    axis_local = torch.zeros(args.num_envs, 3, device=env.device)
    axis_local[:, 2] = 1.0
    side_local = torch.zeros_like(axis_local)
    side_local[:, 0] = 1.0
    body_q = env.object.data.root_quat_w
    axis_w = torch_quat_apply(body_q, axis_local)
    side_w = torch_quat_apply(body_q, side_local)
    cap_pos = (
        env.object.data.root_pos_w
        + (env.body_top_offset_m - 0.005) * axis_w
        + 0.010 * side_w
    )
    cap_pose = torch.cat([cap_pos, body_q], dim=1)
    env.cap.write_root_pose_to_sim(cap_pose)
    env.cap.write_root_velocity_to_sim(
        torch.zeros(args.num_envs, 6, device=env.device)
    )
    env.apply_screw_constraint(integrate_angle=False)
    relative = env.cap.data.root_pos_w - env.object.data.root_pos_w
    axial = (relative * axis_w).sum(dim=1)
    return bool((
        (axial >= env.body_top_offset_m - 1.0e-5)
        & ~env.screw_engaged
    ).all().item())


def main() -> int:
    if args.steps < 1 or args.settle_steps < 0 or args.torque_nm <= 0.0:
        raise ValueError("steps and torque must be positive; settle steps cannot be negative")
    entry = clips.clip_entry(args.clip)
    secondary = entry.get("secondary") or {}
    spec = ScrewSpec.from_mapping(secondary.get("assembly"))
    if spec is None:
        raise ValueError(f"{args.clip!r} is not a screw-cap clip")

    cfg = DexmateCorrectionEnvCfg()
    clips.configure_cfg(cfg, args.clip)
    cfg.sim.device = args.device
    if args.num_envs < 1:
        raise ValueError("num-envs must be positive")
    cfg.scene.num_envs = args.num_envs
    cfg.rsi_prob = 0.0
    cfg.grasp_only = False
    cfg.show_dexmate = False
    cfg.show_human_traj = False

    env = BottleReconstructionEnv(cfg)
    failures: list[str] = []
    try:
        env.reset()
        free_barrier_passed = (
            _audit_free_cap_barrier(env) if spec.mode == "capture" else True
        )

        # Exercise the exact DirectRLEnv.step path used by training, including
        # all decimation substeps and the final observation-time projection.
        env.reset()
        if spec.mode == "capture":
            _stage_capture_entry(env, spec)
        step_start_angle = env.screw_angle.clone()
        axis_local = torch.zeros(args.num_envs, 3, device=env.device)
        axis_local[:, 2] = 1.0
        axis_w = torch_quat_apply(env.object.data.root_quat_w, axis_local)
        motion_sign = 1.0 if spec.mode == "preengaged" else -1.0
        step_velocity = torch.cat([
            env.object.data.root_lin_vel_w,
            env.object.data.root_ang_vel_w + motion_sign * axis_w,
        ], dim=1)
        env.cap.write_root_velocity_to_sim(step_velocity)
        zero_action = torch.zeros(
            args.num_envs, cfg.action_space, device=env.device
        )
        step_obs, _, _, _, _ = env.step(zero_action)
        step_delta = env.screw_angle - step_start_angle
        direct_step_finite = all(
            not torch.isnan(value).any().item() for value in step_obs.values()
        )
        direct_step_advanced = bool(
            (motion_sign * step_delta > 0.01).all().item()
        )

        # Reset so the full motion audit below starts from a canonical state.
        env.reset()
        initially_engaged = bool(env.screw_engaged[0].item())
        if spec.mode == "capture":
            _stage_capture_entry(env, spec)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(cfg.sim.dt)
        captured = bool(env.screw_engaged[0].item())
        env.apply_screw_constraint(integrate_angle=False)
        initial = _relative_state(env, spec)
        initial_constraint_angle = float(env.screw_angle[0].item())
        raw_angles = [initial[0]]
        axial = [initial[1]]
        radial = [initial[2]]
        engaged_samples = [bool(env.screw_engaged[0].item())]
        finite = True

        torque = torch.zeros(args.num_envs, 3, device=env.device)
        torque[..., 2] = motion_sign * args.torque_nm
        for _ in range(args.steps):
            env.apply_screw_constraint(extra_cap_torque_local=torque)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(cfg.sim.dt)
            env.apply_screw_constraint(integrate_angle=False)
            state = _relative_state(env, spec)
            raw_angles.append(state[0])
            axial.append(state[1])
            radial.append(state[2])
            engaged_samples.append(bool(env.screw_engaged[0].item()))
            finite = finite and bool(np.isfinite(state).all())
            if spec.mode == "preengaged":
                if not env.screw_engaged[0].item():
                    break
            elif env.screw_angle[0].item() <= 1.0e-3:
                break

        for _ in range(args.settle_steps):
            zero_torque = torch.zeros_like(torque)
            env.apply_screw_constraint(extra_cap_torque_local=zero_torque)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(cfg.sim.dt)
            env.apply_screw_constraint(integrate_angle=False)
            state = _relative_state(env, spec)
            raw_angles.append(state[0])
            axial.append(state[1])
            radial.append(state[2])
            engaged_samples.append(bool(env.screw_engaged[0].item()))
            finite = finite and bool(np.isfinite(state).all())

        angles = np.unwrap(np.asarray(raw_angles, dtype=np.float64))
        angles -= angles[0]
        axial_values = np.asarray(axial, dtype=np.float64) - axial[0]
        expected = helical_travel(angles, spec)
        engaged_mask = np.asarray(engaged_samples, dtype=bool)
        coupling_error = axial_values[engaged_mask] - expected[engaged_mask]
        final_angle = float(angles[-1])
        final_travel = float(axial_values[-1])
        max_error_mm = float(np.max(np.abs(coupling_error)) * 1000.0)
        max_radial_mm = float(np.max(np.asarray(radial)[engaged_mask]) * 1000.0)
        final_constraint_angle = float(env.screw_angle[0].item())
        engaged_final = bool(env.screw_engaged[0].item())
        completed = (
            not engaged_final if spec.mode == "preengaged"
            else engaged_final and final_constraint_angle <= 1.0e-3
        )

        checks = {
            "states_finite": finite,
            "direct_rl_step_observations_finite": direct_step_finite,
            "direct_rl_step_advances_screw": direct_step_advanced,
            "misaligned_free_cap_blocked": free_barrier_passed,
            "correct_initial_engagement_state": (
                initially_engaged == (spec.mode == "preengaged")
            ),
            "thread_capture_or_preengaged": captured,
            "rotation_in_commanded_direction": motion_sign * final_angle > 2.0 * np.pi,
            "axial_motion_in_commanded_direction": (
                motion_sign * final_travel > 0.8 * spec.pitch_m
            ),
            "completed_unscrew_or_screw_on": completed,
            "helical_coupling_within_0_5mm": max_error_mm <= 0.5,
            "travel_within_limits": (
                float((motion_sign * axial_values).min()) >= -0.0005
                and float((motion_sign * axial_values).max())
                <= spec.travel_m + 0.0005
            ),
            "radial_drift_within_0_5mm": max_radial_mm <= 0.5,
        }
        failures.extend(name for name, passed in checks.items() if not passed)
        report = {
            "clip": args.clip,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "settle_steps": args.settle_steps,
            "torque_nm": args.torque_nm,
            "spec": {
                "pitch_m": spec.pitch_m,
                "turns": spec.turns,
                "travel_m": spec.travel_m,
                "direction": spec.direction,
                "mode": spec.mode,
                "max_angular_velocity_rad_s": spec.max_angular_velocity_rad_s,
                "free_barrier_top_offset_m": env.body_top_offset_m,
                "free_barrier_radius_m": env.free_collision_radius_m,
            },
            "joints": _joint_report(env),
            "motion": {
                "final_rotation_rad": final_angle,
                "final_rotation_turns": final_angle / (2.0 * np.pi),
                "final_axial_travel_m": final_travel,
                "expected_axial_travel_m": float(expected[-1]),
                "initial_constraint_angle_rad": initial_constraint_angle,
                "final_constraint_angle_rad": final_constraint_angle,
                "initially_engaged": initially_engaged,
                "engaged_final": engaged_final,
                "max_coupling_error_mm": max_error_mm,
                "max_radial_drift_mm": max_radial_mm,
            },
            "checks": checks,
            "passed": not failures,
        }
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        print("\n[bottle-screw-smoke-report]\n" + payload, flush=True)
        if args.report:
            path = Path(args.report).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        if failures:
            print("[bottle-screw-smoke] FAIL: " + ", ".join(failures), file=sys.stderr)
            env.close()
            os._exit(1)
        print("[bottle-screw-smoke] PASS", flush=True)
    finally:
        env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
