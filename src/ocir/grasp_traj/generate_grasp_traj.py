"""CLI entrypoint for grasp-trajectory generation (Stage A) with optional
Isaac Sim physics-simulation submission (Stage B).

Fail-closed on CUDA availability (trajectory generation needs the same
grasp-synthesis backend as ``synthesize_sharpa_anchored_bodex.py``). Reads a
single grasp record (given directly, or resolved from a synthesis output
dir's ``summary.json``) plus its sequence directory, writes
``trajectory.npz``/``.json`` to ``--out-dir``, and -- unless ``--no-simulate``
-- submits it to Isaac Sim (persistent server or standalone) via
``ocir.isaac.simulate_grasp_traj``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[3]
VIS_TASK_NAME = "grasp_traj_simulation"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=None, help="Sequence directory (human_demo.npz, object mesh, ...). Required unless --check-only.")
    parser.add_argument("--grasp-json", type=Path, default=None, help="A specific grasp record. Mutually exclusive with --synthesis-out-dir.")
    parser.add_argument("--synthesis-out-dir", type=Path, default=None, help="A per-sequence synthesis output dir; the record is resolved from its summary.json (grasp_json, else failed_grasp_json).")
    parser.add_argument("--asset-config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="Required unless --check-only.")

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--approach-seconds", type=float, default=0.5)
    parser.add_argument("--close-seconds", type=float, default=0.3)
    parser.add_argument("--squeeze-seconds", type=float, default=0.3)
    parser.add_argument("--standoff", type=float, default=0.10)
    parser.add_argument("--pregrasp-open-fraction", type=float, default=1.0, help="How wide the fingers open before the final reach: 1.0 scales all flexion joints to 0 rad (fully open), 0.0 keeps the grasp posture.")
    parser.add_argument("--squeeze-delta", type=float, default=0.15)
    parser.add_argument("--approach-clearance", type=float, default=0.01)
    parser.add_argument("--carry-start", choices=["grasp_frame", "pickup_frame"], default="grasp_frame")
    parser.add_argument("--max-wrist-speed", type=float, default=0.25, help="Cap on wrist speed (m/s) in the synthetic approach/close segments; step counts grow beyond the seconds-based defaults when a leg would exceed it.")
    parser.add_argument("--carry-blend-seconds", type=float, default=0.3, help="Blend duration easing the hand from the squeeze-end pose onto the recorded carry trajectory.")
    parser.add_argument("--open-clearance", type=float, default=0.05, help="SDF clearance (m) the WIDE-OPEN hand must have at the switch frame (the switch-frame search walks backward through the demo until satisfied).")
    parser.add_argument("--open-horizon-seconds", type=float, default=1.0, help="Duration of the smooth finger-opening ramp blended into the tail of the retarget replay, ending fully open at the switch frame.")
    parser.add_argument("--planner", choices=["curobo", "linear"], default="curobo", help="Transit planner for retreat-pose -> standoff: cuRobo v2 MotionPlanner with the object mesh as obstacle (falls back to linear on failure), or plain straight-line + via-point.")
    parser.add_argument("--final-close-seconds", type=float, default=0.4, help="Duration of the slow final close from the near-contact posture to the contact-projected posture.")
    parser.add_argument("--near-contact-margin", type=float, default=0.003, help="Clearance (m) of the near-contact posture that the fast close stage sweeps to before the slow final close.")

    parser.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isaac-mode", choices=["server", "standalone"], default="server")
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--isaac-request-timeout", type=float, default=0.0)

    # Stage B passthrough (see ocir.isaac.simulate_grasp_traj for defaults/docs).
    parser.add_argument("--isaac-hand-usd", type=Path, default=None)
    parser.add_argument("--isaac-object-mass", type=float, default=None)
    parser.add_argument("--isaac-object-density", type=float, default=700.0)
    parser.add_argument("--isaac-friction", type=float, default=2.0)
    parser.add_argument("--isaac-joint-stiffness", type=float, default=80.0)
    parser.add_argument("--isaac-joint-damping", type=float, default=20.0)
    parser.add_argument("--isaac-joint-max-force", type=float, default=300.0)
    parser.add_argument("--isaac-joint-armature", type=float, default=0.01)
    parser.add_argument("--isaac-joint-friction", type=float, default=0.05)
    parser.add_argument("--isaac-close-joint-stiffness", type=float, default=20.0)
    parser.add_argument("--isaac-close-joint-max-force", type=float, default=60.0)
    parser.add_argument("--isaac-convex-decomp-max-hulls", type=int, default=32)
    parser.add_argument("--isaac-sim-steps-per-frame", type=int, default=2)
    parser.add_argument("--isaac-time-steps-per-second", type=float, default=120.0)
    parser.add_argument("--isaac-capture-every", type=int, default=1)
    parser.add_argument("--isaac-settle-steps", type=int, default=60)
    parser.add_argument("--isaac-video-fps", type=float, default=None)
    parser.add_argument("--isaac-carry-mode", choices=["friction", "kinematic"], default="friction")
    parser.add_argument("--isaac-lift-threshold", type=float, default=0.02)
    parser.add_argument("--isaac-drop-threshold", type=float, default=0.005)
    parser.add_argument("--isaac-tabletop-z", type=float, default=0.0)
    parser.add_argument("--isaac-table-margin", type=float, default=0.18)
    parser.add_argument("--isaac-width", type=int, default=1024)
    parser.add_argument("--isaac-height", type=int, default=768)
    parser.add_argument("--isaac-hold-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--isaac-hold-open-seconds", type=float, default=10.0)

    parser.add_argument("--check-only", action="store_true", help="Only check CUDA/backend availability.")
    return parser


def run_standalone_simulation(params: dict) -> bool:
    cmd = [
        str(REPO_ROOT / "scripts/run_isaacsim_conda.sh"),
        str(REPO_ROOT / "scripts/isaac/simulate_grasp_traj.py"),
        "--mode",
        "local",
    ]
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            cmd.append(flag if value else flag.replace("--", "--no-", 1))
        elif value is not None:
            cmd.extend([flag, str(value)])
    print(f"OCIR_GRASP_TRAJ launching standalone Isaac Sim simulation: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


def submit_simulation_job(params: dict, control_host: str, control_port: int, request_timeout: float) -> tuple[bool, dict]:
    from ocir.sim.control_client import request_json, server_is_running, submit_job

    if not server_is_running(control_host, control_port, timeout=1.0):
        print(
            f"OCIR_GRASP_TRAJ no persistent Isaac control server at {control_host}:{control_port}. "
            "Start it with scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py first, "
            "or pass --isaac-mode standalone.",
            file=sys.stderr,
            flush=True,
        )
        return False, {}
    status = request_json(control_host, control_port, "GET", "/status", timeout=1.0)
    registered = {task.get("name") for task in status.get("registered_tasks", [])}
    if VIS_TASK_NAME not in registered and not bool(status.get("hot_reload_tasks", False)):
        print(
            f"OCIR_GRASP_TRAJ control server does not know {VIS_TASK_NAME!r} and hot reload is off. "
            "Restart it with scripts/isaac/simulate_grasp_traj.py in --task-module (or via "
            "visualize_grasp.py's hot-reload chain).",
            file=sys.stderr,
            flush=True,
        )
        return False, {}
    code, response = submit_job(
        host=control_host, port=control_port, task=VIS_TASK_NAME, params=params, request_timeout=request_timeout
    )
    return code == 0, (response or {})


def _simulation_params(args: argparse.Namespace, trajectory_dir: Path, out_dir: Path) -> dict:
    params = {
        "trajectory_dir": str(trajectory_dir),
        "out_dir": str(out_dir),
        "hand_usd": str(args.isaac_hand_usd) if args.isaac_hand_usd else None,
        "object_mass": args.isaac_object_mass,
        "object_density": args.isaac_object_density,
        "friction": args.isaac_friction,
        "joint_stiffness": args.isaac_joint_stiffness,
        "joint_damping": args.isaac_joint_damping,
        "joint_max_force": args.isaac_joint_max_force,
        "joint_armature": args.isaac_joint_armature,
        "joint_friction": args.isaac_joint_friction,
        "close_joint_stiffness": args.isaac_close_joint_stiffness,
        "close_joint_max_force": args.isaac_close_joint_max_force,
        "convex_decomp_max_hulls": args.isaac_convex_decomp_max_hulls,
        "sim_steps_per_frame": args.isaac_sim_steps_per_frame,
        "time_steps_per_second": args.isaac_time_steps_per_second,
        "capture_every": args.isaac_capture_every,
        "settle_steps": args.isaac_settle_steps,
        "video_fps": args.isaac_video_fps,
        "carry_mode": args.isaac_carry_mode,
        "lift_threshold": args.isaac_lift_threshold,
        "drop_threshold": args.isaac_drop_threshold,
        "tabletop_z": args.isaac_tabletop_z,
        "table_margin": args.isaac_table_margin,
        "width": args.isaac_width,
        "height": args.isaac_height,
        "hold_open": args.isaac_hold_open,
        "hold_open_seconds": args.isaac_hold_open_seconds,
    }
    return {key: value for key, value in params.items() if value is not None}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.grasp_json is not None and args.synthesis_out_dir is not None:
        print("--grasp-json and --synthesis-out-dir are mutually exclusive", file=sys.stderr)
        return 2

    import torch

    if not torch.cuda.is_available():
        print("grasp-trajectory generation requires CUDA", file=sys.stderr)
        return 2
    if args.check_only:
        print("Grasp-trajectory generation backend is available (CUDA present).")
        return 0
    if args.sequence_dir is None:
        print("--sequence-dir is required", file=sys.stderr)
        return 2
    if args.out_dir is None:
        print("--out-dir is required", file=sys.stderr)
        return 2

    from curobo._src.types.device_cfg import DeviceCfg

    from ocir.grasp_synthesis.assets import load_sharpa_wave_right
    from ocir.grasp_traj.generator import (
        GraspTrajectoryConfig,
        GraspTrajectoryGenerator,
        load_grasp_record,
        resolve_grasp_record,
    )

    try:
        grasp_json_path = resolve_grasp_record(args.synthesis_out_dir, args.grasp_json)
        record = load_grasp_record(grasp_json_path)
    except Exception as exc:
        print(f"OCIR_GRASP_TRAJ failed to resolve/load grasp record: {exc}", file=sys.stderr, flush=True)
        return 2

    asset = load_sharpa_wave_right(args.asset_config)
    config = GraspTrajectoryConfig(
        fps=args.fps,
        approach_seconds=args.approach_seconds,
        close_seconds=args.close_seconds,
        squeeze_seconds=args.squeeze_seconds,
        standoff_m=args.standoff,
        pregrasp_open_fraction=args.pregrasp_open_fraction,
        squeeze_delta=args.squeeze_delta,
        approach_clearance_m=args.approach_clearance,
        carry_start=args.carry_start,
        max_wrist_speed_mps=args.max_wrist_speed,
        carry_blend_seconds=args.carry_blend_seconds,
        open_clearance_m=args.open_clearance,
        open_horizon_seconds=args.open_horizon_seconds,
        planner=args.planner,
        final_close_seconds=args.final_close_seconds,
        near_contact_margin_m=args.near_contact_margin,
    )
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

    print(f"OCIR_GRASP_TRAJ generating trajectory for {args.sequence_dir} from {grasp_json_path}", flush=True)
    try:
        generator = GraspTrajectoryGenerator(asset, config, device_cfg)
        trajectory = generator.generate(args.sequence_dir, record)
    except Exception as exc:
        print(f"OCIR_GRASP_TRAJ generation failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory.save(args.out_dir)
    print(
        f"OCIR_GRASP_TRAJ wrote trajectory.npz/.json to {args.out_dir} "
        f"(num_steps={trajectory.num_steps}, switch_frame_index={trajectory.switch_frame_index}, "
        f"clearance_satisfied={trajectory.clearance_report.get('clearance_satisfied')})",
        flush=True,
    )

    if not args.simulate:
        return 0

    sim_out_dir = args.out_dir / "isaac_sim"
    params = _simulation_params(args, args.out_dir, sim_out_dir)
    if args.isaac_mode == "standalone":
        ok = run_standalone_simulation(params)
    else:
        ok, response = submit_simulation_job(params, args.control_host, args.control_port, args.isaac_request_timeout)
        if ok:
            print(f"OCIR_GRASP_TRAJ simulation job response: {json.dumps(response, indent=2)}", flush=True)
    if not ok:
        print("OCIR_GRASP_TRAJ Isaac simulation failed or was skipped.", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
