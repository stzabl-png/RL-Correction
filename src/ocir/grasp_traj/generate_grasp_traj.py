"""CLI entrypoint for grasp-trajectory generation (Stage A) with optional
Isaac Sim physics-simulation submission (Stage B).

Fail-closed on CUDA availability (trajectory generation needs the same
grasp-synthesis backend as ``synthesize_sharpa_anchored_bodex.py``). With
``--synthesis-out-dir`` it runs EVERY ranked grasp pose in that object's
synthesis dir (``grasp_pose_N.json``), writing each into its own
``<out-dir>/<pose>/`` subfolder; with an explicit ``--grasp-json`` it runs
that single record into ``--out-dir``. Each writes ``trajectory.npz``/``.json``
and -- unless ``--no-simulate`` -- submits it to Isaac Sim (persistent server
or standalone) via ``ocir.isaac.simulate_grasp_traj``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[3]
VIS_TASK_NAME = "grasp_traj_simulation"


def _pose_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def discover_grasp_records(synthesis_out_dir: Path) -> list[Path]:
    """All ranked grasp-pose records in a synthesis output dir, best-first.

    Primary naming is ``grasp_pose_N.json`` (1-indexed, rank order). Falls
    back to the legacy ``grasp_NNN`` / ``failed_grasp_NNN`` names so older
    synthesis dirs still resolve."""

    directory = Path(synthesis_out_dir)
    records = sorted(directory.glob("grasp_pose_*.json"), key=_pose_index)
    if not records:
        records = sorted(
            set(directory.glob("grasp_[0-9]*.json")) | set(directory.glob("failed_grasp_*.json")),
            key=_pose_index,
        )
    if not records:
        raise FileNotFoundError(
            f"no grasp-pose records (grasp_pose_*.json) under {directory}; run grasp synthesis first"
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=None, help="Sequence directory (human_demo.npz, object mesh, ...). Required unless --check-only.")
    parser.add_argument("--grasp-json", type=Path, default=None, help="A specific grasp record. Mutually exclusive with --synthesis-out-dir.")
    parser.add_argument("--synthesis-out-dir", type=Path, default=None, help="A per-object synthesis output dir. ALL its ranked grasp poses (grasp_pose_N.json) are run, each into its own <out-dir>/grasp_pose_N/ subfolder. Mutually exclusive with --grasp-json (which runs a single record into --out-dir).")
    parser.add_argument("--asset-config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="Required unless --check-only.")

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--approach-seconds", type=float, default=1.0, help="Minimum lead time before grasp used for handoff selection and minimum duration of the direct cuRobo plan to pregrasp.")
    parser.add_argument("--close-seconds", type=float, default=0.3)
    parser.add_argument("--squeeze-seconds", type=float, default=0.3)
    parser.add_argument("--pregrasp-open-fraction", type=float, default=1.0, help="How wide the fingers open before the final reach: 1.0 scales all flexion joints to 0 rad (fully open), 0.0 keeps the grasp posture.")
    parser.add_argument("--squeeze-delta", type=float, default=0.15)
    parser.add_argument("--held-finger-pose", choices=["grasp", "squeeze"], default="grasp", help="Finger pose held through the squeeze segment, settle, and carry. grasp (default): enforce the record's GRASP joints -- no drive-through past contact. squeeze: hold the synthesized (overclosed) squeeze pose. Grasp synthesis is unaffected either way.")
    parser.add_argument("--approach-clearance", type=float, default=0.003, help="Minimum all-sphere clearance for the direct planned approach, including the open-hand pregrasp endpoint.")
    parser.add_argument("--carry-style", choices=["vertical_lift", "demo"], default="vertical_lift", help="vertical_lift: after squeeze+settle, raise the wrist straight up (world +z) by --carry-lift-height and hold. demo: follow the recorded human carry trajectory.")
    parser.add_argument("--settle-seconds", type=float, default=1.0, help="Post-squeeze hold (wrist parked, squeeze targets held) letting the contacts settle before the carry; appended to the squeeze segment.")
    parser.add_argument("--carry-lift-height", type=float, default=0.20, help="Vertical lift height (m), carry style 'vertical_lift'.")
    parser.add_argument("--carry-lift-seconds", type=float, default=2.0, help="Duration of the vertical lift (cosine-eased; steps grow if its peak speed would exceed --max-wrist-speed).")
    parser.add_argument("--carry-hold-seconds", type=float, default=1.0, help="Hold at the top of the vertical lift.")
    parser.add_argument("--dexycb-manifest", type=Path, default=None, help="DexYCB manifest for the camera->world extrinsics defining 'up' (carry style 'vertical_lift'). Default: auto-resolve next to the sequences root, then the Stage B default manifest.")
    parser.add_argument("--carry-start", choices=["grasp_frame", "pickup_frame"], default="grasp_frame")
    parser.add_argument("--max-wrist-speed", type=float, default=0.25, help="Cap on wrist speed (m/s) in the synthetic approach/close segments; step counts grow beyond the seconds-based defaults when a leg would exceed it.")
    parser.add_argument("--carry-blend-seconds", type=float, default=0.3, help="Blend duration easing the hand from the squeeze-end pose onto the recorded carry trajectory.")
    parser.add_argument("--open-clearance", type=float, default=0.05, help="SDF clearance (m) the WIDE-OPEN hand must have at the switch frame (the switch-frame search walks backward through the demo until satisfied).")
    parser.add_argument("--open-horizon-seconds", type=float, default=1.0, help="Duration of the smooth finger-opening ramp blended into the tail of the retarget replay, ending fully open at the switch frame.")
    parser.add_argument("--planner", choices=["curobo", "linear"], default="curobo", help="Planner for handoff -> repaired open-hand pregrasp. cuRobo uses the object mesh as a collision constraint and fails closed; linear is an explicit debug-only alternative.")
    parser.add_argument("--final-close-seconds", type=float, default=1.0, help="Duration of the slow final close from the near-contact posture to the contact-projected posture.")
    parser.add_argument("--near-contact-margin", type=float, default=0.003, help="Clearance (m) of the near-contact posture that the fast close stage sweeps to before the slow final close.")
    parser.add_argument("--self-clearance-buffer", type=float, default=0.0, help="Minimum sphere-metric hand self-clearance (m) enforced on the INITIAL posture (frame 0) by spreading the fingers -- the sim teleport-initializes the articulation there, and an interpenetrating start blows up under PhysX hand self-collision. 0.0 = sphere surfaces touching (~1.5-1.8mm true mesh gap at the binding finger-base pairs); positive values are anatomically unreachable at open postures. Negative disables.")
    parser.add_argument("--self-clearance-decay-seconds", type=float, default=1.0, help="Duration over which the frame-0 self-clearance correction decays back to the original trajectory (no target jump at frame 1).")

    parser.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isaac-mode", choices=["server", "standalone"], default="server")
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--isaac-request-timeout", type=float, default=0.0)

    # Stage B passthrough (see ocir.isaac.simulate_grasp_traj for defaults/docs).
    parser.add_argument("--isaac-hand-usd", type=Path, default=None)
    parser.add_argument("--isaac-object-mass", type=float, default=None)
    parser.add_argument("--isaac-object-density", type=float, default=700.0)
    parser.add_argument("--isaac-friction-target", choices=["both", "object"], default="both")
    parser.add_argument("--isaac-friction", type=float, default=3.0, help="Friction of the bound material.")
    parser.add_argument("--isaac-friction-combine-mode", choices=["max", "multiply", "average", "min"], default="multiply")
    parser.add_argument("--isaac-joint-armature", type=float, default=0.001, help="Finger joint armature (reference drive-table default); gains/effort caps come from the per-joint table in the sim module.")
    parser.add_argument("--isaac-joint-friction", type=float, default=0.0)
    parser.add_argument("--isaac-contact-aware-finger-targets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--isaac-contact-target-lead-rad", type=float, default=0.03)
    parser.add_argument("--isaac-convex-decomp-max-hulls", type=int, default=32)
    parser.add_argument("--isaac-object-collision", choices=["sdf", "convex"], default="convex")
    parser.add_argument("--isaac-sdf-resolution", type=int, default=256)
    parser.add_argument("--isaac-hand-rest-offset", type=float, default=None, help="Explicit override; default keeps the asset's baked collider offsets.")
    parser.add_argument("--isaac-hand-self-collisions", action=argparse.BooleanOptionalAction, default=True, help="PhysX self-collision between the hand's own links. Default on.")
    parser.add_argument("--isaac-hand-table-collision", action=argparse.BooleanOptionalAction, default=False, help="Hand-table contact pairs; default off (collision-group filtered, as in the reference validator).")
    parser.add_argument("--isaac-sim-steps-per-frame", type=int, default=2)
    parser.add_argument("--isaac-time-steps-per-second", type=float, default=180.0)
    parser.add_argument("--isaac-gravity", type=float, default=9.81, help="Gravity magnitude (m/s^2); default 9.81 (realistic weight), the reference uses 30 as a stress load.")
    parser.add_argument("--isaac-contact-slop", type=float, default=0.2, help="Object contactSlopCoefficient; default 0.2 matches the reference, 0 disables.")
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
        "friction_target": args.isaac_friction_target,
        "friction": args.isaac_friction,
        "friction_combine_mode": args.isaac_friction_combine_mode,
        "joint_armature": args.isaac_joint_armature,
        "joint_friction": args.isaac_joint_friction,
        "contact_aware_finger_targets": args.isaac_contact_aware_finger_targets,
        "contact_target_lead_rad": args.isaac_contact_target_lead_rad,
        "convex_decomp_max_hulls": args.isaac_convex_decomp_max_hulls,
        "object_collision": args.isaac_object_collision,
        "sdf_resolution": args.isaac_sdf_resolution,
        "hand_rest_offset": args.isaac_hand_rest_offset,
        "hand_self_collisions": args.isaac_hand_self_collisions,
        "hand_table_collision": args.isaac_hand_table_collision,
        "sim_steps_per_frame": args.isaac_sim_steps_per_frame,
        "time_steps_per_second": args.isaac_time_steps_per_second,
        "gravity": args.isaac_gravity,
        "contact_slop": args.isaac_contact_slop,
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

    # Resolve the record list. An explicit --grasp-json is a single run into
    # --out-dir; a --synthesis-out-dir runs ALL of that object's ranked grasp
    # poses, each into its own <out-dir>/<record_stem>/ subfolder (same object
    # output folder).
    try:
        if args.grasp_json is not None:
            jobs = [(resolve_grasp_record(None, args.grasp_json), args.out_dir)]
        elif args.synthesis_out_dir is not None:
            records = discover_grasp_records(args.synthesis_out_dir)
            jobs = [(rec, args.out_dir / rec.stem) for rec in records]
            print(
                f"OCIR_GRASP_TRAJ running all {len(jobs)} grasp pose(s) from {args.synthesis_out_dir} "
                f"into {args.out_dir}/<pose>",
                flush=True,
            )
        else:
            print("provide --grasp-json or --synthesis-out-dir", file=sys.stderr, flush=True)
            return 2
    except Exception as exc:
        print(f"OCIR_GRASP_TRAJ failed to resolve grasp records: {exc}", file=sys.stderr, flush=True)
        return 2

    asset = load_sharpa_wave_right(args.asset_config)
    config = GraspTrajectoryConfig(
        fps=args.fps,
        approach_seconds=args.approach_seconds,
        close_seconds=args.close_seconds,
        squeeze_seconds=args.squeeze_seconds,
        pregrasp_open_fraction=args.pregrasp_open_fraction,
        squeeze_delta=args.squeeze_delta,
        held_finger_pose=args.held_finger_pose,
        approach_clearance_m=args.approach_clearance,
        carry_style=args.carry_style,
        settle_seconds=args.settle_seconds,
        carry_lift_height_m=args.carry_lift_height,
        carry_lift_seconds=args.carry_lift_seconds,
        carry_hold_seconds=args.carry_hold_seconds,
        dexycb_manifest=str(args.dexycb_manifest) if args.dexycb_manifest else None,
        carry_start=args.carry_start,
        max_wrist_speed_mps=args.max_wrist_speed,
        carry_blend_seconds=args.carry_blend_seconds,
        open_clearance_m=args.open_clearance,
        open_horizon_seconds=args.open_horizon_seconds,
        planner=args.planner,
        final_close_seconds=args.final_close_seconds,
        near_contact_margin_m=args.near_contact_margin,
        self_clearance_buffer_m=args.self_clearance_buffer,
        self_clearance_decay_seconds=args.self_clearance_decay_seconds,
    )
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    generator = GraspTrajectoryGenerator(asset, config, device_cfg)

    def _run_one(grasp_json_path: Path, out_dir: Path) -> bool:
        print(f"OCIR_GRASP_TRAJ generating trajectory for {args.sequence_dir} from {grasp_json_path}", flush=True)
        try:
            record = load_grasp_record(grasp_json_path)
            trajectory = generator.generate(args.sequence_dir, record)
        except Exception as exc:
            print(f"OCIR_GRASP_TRAJ generation failed for {grasp_json_path.name}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            return False

        out_dir.mkdir(parents=True, exist_ok=True)
        trajectory.save(out_dir)
        print(
            f"OCIR_GRASP_TRAJ wrote trajectory.npz/.json to {out_dir} "
            f"(num_steps={trajectory.num_steps}, switch_frame_index={trajectory.switch_frame_index}, "
            f"clearance_satisfied={trajectory.clearance_report.get('clearance_satisfied')})",
            flush=True,
        )

        if not args.simulate:
            return True

        sim_out_dir = out_dir / "isaac_sim"
        params = _simulation_params(args, out_dir, sim_out_dir)
        if args.isaac_mode == "standalone":
            ok = run_standalone_simulation(params)
        else:
            ok, response = submit_simulation_job(params, args.control_host, args.control_port, args.isaac_request_timeout)
            if ok:
                # The full report (incl. per-step diagnostics) lives in
                # <sim_out_dir>/report.json; only surface the core outcome here.
                report = response.get("summary") if isinstance(response.get("summary"), dict) else response
                metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
                print(
                    f"OCIR_GRASP_TRAJ simulation done ({out_dir.name}): "
                    f"grasp_success={metrics.get('grasp_success')} lifted={metrics.get('lifted')} "
                    f"sustained={metrics.get('sustained_lift')} dropped={metrics.get('object_dropped')} "
                    f"max_lift_m={metrics.get('max_lift_m')} "
                    f"final_object_position_error_m={metrics.get('final_object_position_error_m')} "
                    f"(full report: {sim_out_dir / 'report.json'})",
                    flush=True,
                )
        if not ok:
            print(f"OCIR_GRASP_TRAJ Isaac simulation failed or was skipped for {out_dir.name}.", file=sys.stderr, flush=True)
        return ok

    n_ok = sum(1 for grasp_json_path, out_dir in jobs if _run_one(grasp_json_path, out_dir))
    print(f"OCIR_GRASP_TRAJ finished {n_ok}/{len(jobs)} grasp pose(s) for {args.sequence_dir}", flush=True)
    return 0 if n_ok == len(jobs) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
