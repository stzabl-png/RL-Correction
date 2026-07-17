#!/usr/bin/env python3
"""Run carry-only closed-loop object-path tracking in Isaac Sim.

All trajectory frames before the first carry frame use the existing open-loop
playback unchanged.  At carry entry, a bounded SE(3) path follower begins
reading the dynamic object pose at 60 Hz and correcting only the wrist.  The
finger targets remain exactly equal to the final grasp target.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import time

import numpy as np

from ocir.full_traj.controller import ControllerConfig, PathFollowingController
from ocir.full_traj.reference import FullTrajectoryReference
from ocir.grasp_traj.trajectory_schema import SEGMENT_CARRY, pos_quat_to_matrix
from ocir.isaac import simulate_grasp_traj as base
from ocir.isaac.sim_cli import run_sim_cli_main
from ocir.sim.control_client import request_json
from ocir.sim.isaac_server import DEFAULT_OCIR_DATA_ROOT


DEFAULT_OUT_DIR = DEFAULT_OCIR_DATA_ROOT / "testing/full_traj/isaac_sim"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_FULL_TRAJ_SIM {message}", flush=True)


@dataclass
class FullTrajectoryRuntime:
    controller: PathFollowingController
    carry_start_step: int
    fixed_finger_targets: np.ndarray


def _config_from_args(args) -> ControllerConfig:
    return ControllerConfig(
        control_hz=60.0,
        hold_alignment=bool(args.hold_alignment),
        catchup_seconds=float(args.catchup_seconds),
        position_tolerance_m=float(args.position_tolerance),
        orientation_tolerance_rad=np.deg2rad(float(args.orientation_tolerance_deg)),
        lookahead_translation_m=float(args.lookahead),
        lookahead_rotation_rad=np.deg2rad(float(args.lookahead_rotation_deg)),
        projection_window_points=int(args.projection_window),
        projection_rotation_scale_m_per_rad=float(args.projection_rotation_scale),
        projection_acceptance_translation_m=float(args.projection_acceptance),
        waypoint_timeout_seconds=float(args.waypoint_timeout),
        total_timeout_scale=float(args.total_timeout_scale),
        kp_translation=float(args.kp_translation),
        ki_translation=float(args.ki_translation),
        kp_rotation=float(args.kp_rotation),
        ki_rotation=float(args.ki_rotation),
        max_integral_translation_m_s=float(args.max_integral_translation),
        max_integral_rotation_rad_s=np.deg2rad(float(args.max_integral_rotation_deg_s)),
        max_correction_translation_m=float(args.max_correction_translation),
        max_correction_rotation_rad=np.deg2rad(float(args.max_correction_rotation_deg)),
        max_linear_speed_mps=float(args.max_wrist_speed),
        max_angular_speed_radps=np.deg2rad(float(args.max_wrist_angular_speed_deg)),
        max_linear_accel_mps2=float(args.max_wrist_accel),
        max_angular_accel_radps2=np.deg2rad(float(args.max_wrist_angular_accel_deg)),
        lost_grasp_translation_m=float(args.lost_grasp_translation),
        lost_grasp_rotation_rad=np.deg2rad(float(args.lost_grasp_rotation_deg)),
        lost_grasp_confirm_steps=int(args.lost_grasp_confirm_steps),
    )


def _runtime_factory(args):
    reference = FullTrajectoryReference.load(args.trajectory_dir)
    config = _config_from_args(args)

    def factory(*, frame_mapper, z_offset: float, trajectory) -> FullTrajectoryRuntime:
        carry_rows = np.flatnonzero(trajectory.segment == SEGMENT_CARRY)
        if carry_rows.size == 0 or int(carry_rows[0]) != int(reference.carry_start_step):
            raise ValueError(
                "full trajectory/reference carry boundary mismatch: "
                f"trajectory={int(carry_rows[0]) if carry_rows.size else None}, "
                f"reference={reference.carry_start_step}"
            )
        if tuple(reference.fixed_finger_targets.shape) != (len(trajectory.joint_order),):
            raise ValueError("full trajectory fixed finger target does not match trajectory joint order")
        obj_pos, obj_quat = base.camera_pos_quat_to_isaac(
            reference.object_pos_camera,
            reference.object_quat_camera,
            frame_mapper,
            z_offset=z_offset,
        )
        wrist_pos, wrist_quat = base.camera_pos_quat_to_isaac(
            reference.mano_wrist_pos_camera,
            reference.mano_wrist_quat_camera,
            frame_mapper,
            z_offset=z_offset,
        )
        controller = PathFollowingController(
            pos_quat_to_matrix(obj_pos, obj_quat),
            pos_quat_to_matrix(wrist_pos, wrist_quat),
            source_fps=reference.source_fps,
            source_duration_seconds=reference.metadata.get("source_duration_seconds"),
            config=config,
        )
        return FullTrajectoryRuntime(
            controller=controller,
            carry_start_step=reference.carry_start_step,
            fixed_finger_targets=reference.fixed_finger_targets.copy(),
        )

    return factory


def simulate_full_traj(app, args, progress=None) -> dict:
    if bool(args.contact_aware_finger_targets):
        raise ValueError("full trajectory requires fixed finger targets; use --no-contact-aware-finger-targets")
    if str(args.carry_mode) != base.CARRY_MODE_FRICTION:
        raise ValueError("closed-loop object tracking requires a dynamic object; use --carry-mode friction")
    report = base.simulate_grasp_traj(
        app,
        args,
        progress=progress,
        carry_controller_factory=_runtime_factory(args),
    )
    return report


def build_parser():
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(out_dir=DEFAULT_OUT_DIR, contact_aware_finger_targets=False, carry_mode=base.CARRY_MODE_FRICTION)
    for action in parser._actions:
        if action.dest in {"contact_aware_finger_targets", "contact_target_lead_rad"}:
            action.help = argparse.SUPPRESS
    group = parser.add_argument_group("closed-loop full-trajectory control")
    group.add_argument(
        "--hold-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track the reference path aligned to the object pose measured at carry entry (relative path). --no-hold-alignment restores absolute-path convergence over --catchup-seconds.",
    )
    group.add_argument("--catchup-seconds", type=float, default=1.0, help="Only used with --no-hold-alignment: seconds over which the carry-entry alignment is removed.")
    group.add_argument("--position-tolerance", type=float, default=0.008, help="Cross-track position tolerance used for coverage reporting and end-of-path completion (m).")
    group.add_argument("--orientation-tolerance-deg", type=float, default=5.0)
    group.add_argument("--lookahead", type=float, default=0.02, help="Pursuit lookahead distance along the path (m).")
    group.add_argument("--lookahead-rotation-deg", type=float, default=10.0, help="Pursuit lookahead cap on accumulated path rotation (degrees).")
    group.add_argument("--projection-window", type=int, default=40, help="Forward window (path points) searched when projecting the object onto the path.")
    group.add_argument("--projection-rotation-scale", type=float, default=0.05, help="Meters-per-radian weight mixing orientation into the projection distance.")
    group.add_argument("--projection-acceptance", type=float, default=0.05, help="Maximum object-to-path distance (m) that still counts as path progress.")
    group.add_argument("--waypoint-timeout", type=float, default=1.0, help="Stall-escape: force one path point of progress after this many seconds without projection progress.")
    group.add_argument("--total-timeout-scale", type=float, default=3.0, help="Maximum carry duration as a multiple of the source carry duration.")
    group.add_argument("--kp-translation", type=float, default=0.6)
    group.add_argument("--ki-translation", type=float, default=0.15)
    group.add_argument("--kp-rotation", type=float, default=0.6)
    group.add_argument("--ki-rotation", type=float, default=0.15)
    group.add_argument("--max-integral-translation", type=float, default=0.10, help="Translation integral norm limit (m*s).")
    group.add_argument("--max-integral-rotation-deg-s", type=float, default=45.0)
    group.add_argument("--max-correction-translation", type=float, default=0.05)
    group.add_argument("--max-correction-rotation-deg", type=float, default=20.0)
    group.add_argument("--max-wrist-speed", type=float, default=0.25)
    group.add_argument("--max-wrist-angular-speed-deg", type=float, default=90.0)
    group.add_argument("--max-wrist-accel", type=float, default=1.0)
    group.add_argument("--max-wrist-angular-accel-deg", type=float, default=360.0)
    group.add_argument("--lost-grasp-translation", type=float, default=0.05)
    group.add_argument("--lost-grasp-rotation-deg", type=float, default=30.0)
    group.add_argument("--lost-grasp-confirm-steps", type=int, default=5)
    return parser


def normalize_paths(args) -> None:
    base.normalize_paths(args)
    if args.status or args.shutdown_server:
        return
    FullTrajectoryReference.load(args.trajectory_dir)
    _config_from_args(args).validate()
    if bool(args.contact_aware_finger_targets):
        raise ValueError("full trajectory does not permit contact-aware finger target rewriting")
    if str(args.carry_mode) != base.CARRY_MODE_FRICTION:
        raise ValueError("full trajectory requires --carry-mode friction")


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    base.apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def normalize_task_args(params: dict):
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    base.apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def run_simulation_task(app, args, progress) -> dict:
    old_log = base.log
    base.log = log
    try:
        return simulate_full_traj(app, args, progress=progress)
    finally:
        base.log = old_log


def register_sim_tasks(registry) -> None:
    registry.register(
        "full_traj_simulation",
        run=run_simulation_task,
        normalize=normalize_task_args,
        description="Track a demonstrated object SE(3) carry path using carry-only closed-loop wrist control.",
    )


def main() -> int:
    args = parse_args()
    if args.status:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_FULL_TRAJ_SIM no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    if args.shutdown_server:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "POST", "/shutdown", {}, timeout=2.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_FULL_TRAJ_SIM no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    old_log = base.log
    base.log = log
    try:
        return run_sim_cli_main(
            args,
            log=log,
            run_batch=simulate_full_traj,
            task="full_traj_simulation",
            log_prefix="OCIR_FULL_TRAJ_SIM",
            restart_hint=(
                "Restart the server with this task module, for example:\n"
                "  scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py "
                "--task-module scripts/full_traj/simulate_full_traj.py"
            ),
        )
    finally:
        base.log = old_log


if __name__ == "__main__":
    raise SystemExit(main())
