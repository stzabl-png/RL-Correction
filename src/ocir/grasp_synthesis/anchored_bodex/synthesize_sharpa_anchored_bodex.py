"""Entrypoint for anchored (human-demo-guided) BODex grasp synthesis.

Same fail-closed backend checks and Isaac visualization plumbing as
``synthesize_sharpa_bodex_curobo_v2`` (the pure-BODex CLI, which stays
untouched), but every sequence directory must additionally provide a
``human_demo.npz`` demonstration (see
``ocir.grasp_synthesis.anchored_bodex.demo_data``); the per-sequence
``affordance.npz`` cache is created on demand.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from ocir.grasp_synthesis.bodex_curobo_v2 import ExactBodexUnavailable, check_exact_bodex_v2_backend
from ocir.sim.control_client import request_json, server_is_running, submit_job

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_NAME = "curobo_v2_anchored_bodex"
VIS_TASK_NAME = "anchored_grasp_visualization"
DEFAULT_DEMO_MANIFEST = Path(os.environ.get("OCIR_DATA_ROOT", "/data/users/hangkes2/OCIR")).expanduser() / "processed_data/dex_ycb/manifests/selected_5_sequences.json"


def _sequence_meta(sequence_dir: Path) -> dict:
    meta_path = sequence_dir / "sequence.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def _demo_artifact_available(sequence_dir: Path) -> bool:
    meta = _sequence_meta(sequence_dir)
    explicit = meta.get("human_demo")
    if explicit is not None:
        path = Path(explicit)
        path = path if path.is_absolute() else sequence_dir / path
        return path.exists()
    return (sequence_dir / "human_demo.npz").exists()


def _jobs_sequence_root(jobs: list[Path]) -> Path:
    parents = {Path(job).parent for job in jobs}
    if len(parents) != 1:
        raise ValueError(f"cannot auto-export DexYCB demos for sequence dirs from multiple roots: {sorted(map(str, parents))}")
    return next(iter(parents))


def _demo_manifest_path(args: argparse.Namespace, sequences_root: Path) -> Path:
    if args.demo_manifest is not None:
        return args.demo_manifest.expanduser()
    from ocir.dexycb.export_grasp_sequences import default_manifest_for_sequences_root

    inferred = default_manifest_for_sequences_root(sequences_root)
    if inferred.exists():
        return inferred
    return DEFAULT_DEMO_MANIFEST


def ensure_human_demos(jobs: list[Path], args: argparse.Namespace) -> None:
    if not args.auto_export_demo:
        return
    missing = [Path(job) for job in jobs if not _demo_artifact_available(Path(job))]
    if not missing:
        return

    from ocir.dexycb.export_grasp_sequences import (
        export_sequence_demo,
        infer_data_root_from_sequences_root,
        localize_manifest_paths,
    )
    from ocir.dexycb.labels import load_manifest, sequence_by_id

    sequences_root = _jobs_sequence_root(missing)
    manifest_path = _demo_manifest_path(args, sequences_root)
    manifest = load_manifest(manifest_path)
    local_data_root = infer_data_root_from_sequences_root(sequences_root)
    if local_data_root is not None:
        manifest = localize_manifest_paths(manifest, local_data_root)

    names = ", ".join(sequence_dir.name for sequence_dir in missing)
    print(
        f"OCIR_ANCHORED_BODEX exporting missing DexYCB human demos for {len(missing)} sequence(s): {names}",
        flush=True,
    )
    print(f"OCIR_ANCHORED_BODEX demo manifest={manifest_path} sequences_root={sequences_root}", flush=True)
    for sequence_dir in missing:
        sequence = sequence_by_id(manifest, sequence_dir.name)
        export_sequence_demo(manifest, sequence, sequences_root, bool(args.force_demo_export))

    still_missing = [Path(job) for job in jobs if not _demo_artifact_available(Path(job))]
    if still_missing:
        names = ", ".join(sequence_dir.name for sequence_dir in still_missing)
        raise FileNotFoundError(f"DexYCB demo export did not produce usable human_demo metadata for: {names}")


def run_standalone_visualization(params: dict) -> bool:
    cmd = [
        str(REPO_ROOT / "scripts/run_isaacsim_conda.sh"),
        str(REPO_ROOT / "scripts/isaac/visualize_anchored_grasp.py"),
        "--mode",
        "local",
    ]
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            cmd.append(flag if value else flag.replace("--", "--no-", 1))
        else:
            cmd.extend([flag, str(value)])
    print(f"OCIR_ANCHORED_BODEX launching standalone Isaac Sim visualization: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--sequence-dir", type=Path, default=None, help="Run a single sequence directory.")
    input_group.add_argument("--sequences-root", type=Path, default=None, help="Batch mode: every immediate subdirectory is a sequence.")
    parser.add_argument("--asset-config", type=Path, default=None)
    parser.add_argument("--object-mesh", type=Path, default=None, help="Override the auto-discovered object mesh (single sequence only).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Required unless --check-only is passed.")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opt-iters", type=int, default=500)
    parser.add_argument("--grasp-threshold", type=float, default=0.001)
    parser.add_argument("--distance-threshold", type=float, default=0.01)
    # Anchored-seeding / guidance knobs.
    parser.add_argument("--relax-flexion", type=float, default=0.15, help="Radians opened on flexion joints so seeds start out of contact.")
    parser.add_argument("--relax-standoff", type=float, default=0.015, help="Meters the wrist is pulled back along the approach axis.")
    parser.add_argument("--jitter-pos", type=float, default=0.01)
    parser.add_argument("--jitter-rot-deg", type=float, default=10.0)
    parser.add_argument("--jitter-joint", type=float, default=0.08)
    parser.add_argument("--affordance-weight", type=float, default=20.0, help="Weight of the affordance-attraction energy.")
    parser.add_argument("--pose-weight", type=float, default=1.0, help="Scale on the annealed human-pose prior.")
    parser.add_argument("--contact-subset", action=argparse.BooleanOptionalAction, default=True, help="Use only the contact points whose roles the human used (regenerating pressure constraints).")
    parser.add_argument("--afford-tau", type=float, default=0.3, help="Heatmap threshold defining the high-affordance region.")
    parser.add_argument("--rank-affordance-weight", type=float, default=1.0)
    parser.add_argument("--rank-pose-weight", type=float, default=0.5)
    parser.add_argument("--squeeze-min", type=float, default=0.15, help="Flexion-only floor (rad) on the squeeze stage's per-joint extrapolation beyond the contact grasp.")
    parser.add_argument("--force-affordance", action="store_true", help="Recompute the per-sequence affordance cache.")
    parser.add_argument(
        "--auto-export-demo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the DexYCB human-demo exporter for sequence dirs with no usable human_demo metadata.",
    )
    parser.add_argument(
        "--demo-manifest",
        type=Path,
        default=None,
        help="DexYCB selected-subset manifest used for automatic human_demo export. Defaults to the manifest next to --sequences-root, then OCIR_DATA_ROOT.",
    )
    parser.add_argument("--force-demo-export", action=argparse.BooleanOptionalAction, default=False)
    # Isaac visualization (same plumbing as the pure-BODex CLI).
    parser.add_argument("--isaac-visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isaac-visualize-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isaac-mode", choices=["server", "standalone"], default="server")
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--isaac-request-timeout", type=float, default=0.0)
    parser.add_argument("--isaac-width", type=int, default=1024)
    parser.add_argument("--isaac-height", type=int, default=768)
    parser.add_argument("--isaac-tabletop-z", type=float, default=0.8)
    parser.add_argument("--isaac-hold-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--isaac-hold-open-seconds",
        type=float,
        default=10.0,
        help="How long the visualization stays up before the pipeline continues. In standalone mode the "
        "one-shot Isaac window closes itself after this hold, so batch runs proceed unattended.",
    )
    parser.add_argument("--isaac-show-object-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check-only", action="store_true", help="Only check backend availability.")
    parser.add_argument("--strict-success-exit-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds <= 0:
        print("--seeds must be positive", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    if args.object_mesh is not None and args.sequence_dir is None:
        print("--object-mesh requires --sequence-dir", file=sys.stderr)
        return 2
    try:
        report = check_exact_bodex_v2_backend()
    except ExactBodexUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.check_only:
        print(f"Anchored-BODex backend is available. Active cuRobo path: {report.curobo_path}")
        return 0

    if args.sequence_dir is None and args.sequences_root is None:
        print("one of --sequence-dir or --sequences-root is required", file=sys.stderr)
        return 2
    if args.out_dir is None:
        print("--out-dir is required", file=sys.stderr)
        return 2

    from ocir.grasp_synthesis.anchored_bodex.solver import solve_sharpa_anchored_bodex

    if args.sequence_dir is not None:
        jobs = [args.sequence_dir]
    else:
        if not Path(args.sequences_root).is_dir():
            print(f"sequences root not found: {args.sequences_root}", file=sys.stderr)
            return 2
        jobs = sorted(p for p in Path(args.sequences_root).iterdir() if p.is_dir())
        if not jobs:
            print(f"no sequence directories found under {args.sequences_root}", file=sys.stderr)
            return 2

    try:
        ensure_human_demos(jobs, args)
    except Exception as exc:
        print(f"OCIR_ANCHORED_BODEX failed to auto-export DexYCB human demos: {exc}", file=sys.stderr, flush=True)
        return 2

    summaries = []
    failed_visualizations = 0
    solver_failures = 0
    strict_failures = 0
    for sequence_dir in jobs:
        sequence_name = sequence_dir.name
        out_dir = Path(args.out_dir) / sequence_name
        print(f"OCIR_ANCHORED_BODEX solving {sequence_name} sequence_dir={sequence_dir}", flush=True)
        try:
            run_summary = solve_sharpa_anchored_bodex(
                sequence_dir=sequence_dir,
                out_dir=out_dir,
                object_mesh=args.object_mesh,
                seeds=args.seeds,
                top_k=args.top_k,
                opt_iters=args.opt_iters,
                seed=args.seed,
                grasp_threshold=args.grasp_threshold,
                distance_threshold=args.distance_threshold,
                relax_flexion=args.relax_flexion,
                relax_standoff=args.relax_standoff,
                jitter_pos=args.jitter_pos,
                jitter_rot_deg=args.jitter_rot_deg,
                jitter_joint=args.jitter_joint,
                affordance_weight=args.affordance_weight,
                pose_weight=args.pose_weight,
                contact_subset=args.contact_subset,
                afford_tau=args.afford_tau,
                rank_affordance_weight=args.rank_affordance_weight,
                rank_pose_weight=args.rank_pose_weight,
                force_affordance=args.force_affordance,
                squeeze_min_rad=args.squeeze_min,
            )
        except Exception as exc:
            solver_failures += 1
            run_summary = {
                "ok": False,
                "backend": BACKEND_NAME,
                "sequence_id": sequence_name,
                "sequence_dir": str(sequence_dir),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"OCIR_ANCHORED_BODEX solver failed for {sequence_name}: {exc}", file=sys.stderr, flush=True)
            summaries.append(run_summary)
            continue

        if not run_summary.get("ok", False):
            strict_failures += 1

        if args.isaac_visualize:
            vis_out_dir = out_dir / "isaac_visualization"
            grasp_json = run_summary.get("grasp_json")
            used_failed_fallback = False
            if grasp_json is None and args.isaac_visualize_failed:
                grasp_json = run_summary.get("failed_grasp_json")
                used_failed_fallback = grasp_json is not None
            params = {
                "sequence_dir": str(sequence_dir),
                "grasp_json": grasp_json,
                "asset_config": str(args.asset_config) if args.asset_config else None,
                "object_mesh": run_summary.get("object_mesh"),
                "out_dir": str(vis_out_dir),
                "tabletop_z": float(args.isaac_tabletop_z),
                "show_object_points": bool(args.isaac_show_object_points),
                "width": int(args.isaac_width),
                "height": int(args.isaac_height),
                "hold_open": bool(args.isaac_hold_open),
                "hold_open_seconds": float(args.isaac_hold_open_seconds),
            }
            params = {key: value for key, value in params.items() if value is not None}
            if used_failed_fallback:
                print(
                    f"OCIR_ANCHORED_BODEX {sequence_name}: no strict success, visualizing best-ranked "
                    f"failed seed instead ({grasp_json}).",
                    flush=True,
                )
            if grasp_json is None:
                reason = (
                    "no grasp_json or failed_grasp_json available"
                    if args.isaac_visualize_failed
                    else "no grasp_json (pass --isaac-visualize-failed to visualize the best failed seed)"
                )
                print(f"OCIR_ANCHORED_BODEX skipping Isaac visualization for {sequence_name}: {reason}.", file=sys.stderr, flush=True)
                failed_visualizations += 1
            elif args.isaac_mode == "standalone":
                ok = run_standalone_visualization(params)
                run_summary["isaac_visualization"] = {"ok": ok, "out_dir": str(vis_out_dir), "mode": "standalone"}
                failed_visualizations += int(not ok)
            elif not server_is_running(args.control_host, int(args.control_port), timeout=1.0):
                print(
                    "OCIR_ANCHORED_BODEX Isaac visualization requested, but no persistent Isaac control "
                    f"server is running at {args.control_host}:{args.control_port}. Start it with "
                    "scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py first, or pass "
                    "--isaac-mode standalone.",
                    file=sys.stderr,
                    flush=True,
                )
                failed_visualizations += 1
            else:
                status = request_json(args.control_host, int(args.control_port), "GET", "/status", timeout=1.0)
                registered = {task.get("name") for task in status.get("registered_tasks", [])}
                # A hot-reloading server picks the anchored task up from
                # source on submission even if /status predates it; fall back
                # to the plain grasp task only when neither path can work.
                if VIS_TASK_NAME in registered or bool(status.get("hot_reload_tasks", False)):
                    task_name = VIS_TASK_NAME
                elif "grasp_pose_visualization" in registered:
                    print(
                        f"OCIR_ANCHORED_BODEX server does not know {VIS_TASK_NAME!r} and hot reload is off; "
                        "falling back to plain grasp_pose_visualization (no demo/affordance overlays).",
                        flush=True,
                    )
                    task_name = "grasp_pose_visualization"
                else:
                    task_name = None
                if task_name is None:
                    print(
                        "OCIR_ANCHORED_BODEX persistent Isaac server is running, but it has not registered "
                        "any grasp visualization task.",
                        file=sys.stderr,
                        flush=True,
                    )
                    failed_visualizations += 1
                else:
                    code, response = submit_job(
                        host=args.control_host,
                        port=int(args.control_port),
                        task=task_name,
                        params=params,
                        request_timeout=float(args.isaac_request_timeout),
                    )
                    run_summary["isaac_visualization"] = {
                        "ok": code == 0,
                        "out_dir": str(vis_out_dir),
                        "task": task_name,
                        "response": response,
                    }
                    failed_visualizations += int(code != 0)
        summaries.append(run_summary)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    overall_ok = solver_failures == 0 and failed_visualizations == 0 and (
        strict_failures == 0 or not args.strict_success_exit_code
    )
    summary = {
        "ok": overall_ok,
        "backend": BACKEND_NAME,
        "curobo_path": str(report.curobo_path),
        "solver_failures": solver_failures,
        "failed_visualizations": failed_visualizations,
        "strict_failures": strict_failures,
        "runs": summaries,
    }
    summary_path = Path(args.out_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"OCIR_ANCHORED_BODEX done: ok={overall_ok} runs={len(summaries)} "
        f"solver_failures={solver_failures} failed_visualizations={failed_visualizations} "
        f"strict_failures={strict_failures} summary={summary_path}",
        flush=True,
    )
    if solver_failures or failed_visualizations:
        return 1
    if args.strict_success_exit_code and strict_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
