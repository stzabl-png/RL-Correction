"""Fail-closed entrypoint for exact BODex grasp synthesis on official cuRobo v2.

This port keeps the active runtime on the official NVLabs cuRobo submodule and
ports BODex-specific grasp objective code under
``ocir.grasp_synthesis.bodex_curobo_v2``, never importing anything from
``third_party/BODex``. It fails closed when the exact backend (official
cuRobo v2 + standalone ``coal`` + this package's ported BODex objective) is
not available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from ocir.grasp_synthesis.bodex_curobo_v2 import ExactBodexUnavailable, check_exact_bodex_v2_backend
from ocir.grasp_synthesis.object_surface import infer_surface_artifact_path
from ocir.sim.control_client import request_json, server_is_running, submit_job

DEFAULT_DATA_ROOT = Path("/data/users/hangkes2/OCIR")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "testing/grasp_synthesis/sharpa_wave_bodex_curobo_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=None)
    parser.add_argument("--surface-artifact", type=Path, default=None)
    parser.add_argument("--asset-config", type=Path, default=None)
    parser.add_argument("--object-mesh", type=Path, default=None)
    parser.add_argument("--dexycb-manifest", type=Path, default=DEFAULT_DATA_ROOT / "processed_data/dex_ycb/manifests/selected_5_sequences.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opt-iters", type=int, default=500)
    parser.add_argument("--grasp-threshold", type=float, default=0.001)
    parser.add_argument("--distance-threshold", type=float, default=0.01)
    parser.add_argument("--isaac-visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--isaac-visualize-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If no seed reached strict success, visualize the best-ranked (lowest-score) failed seed instead of skipping.",
    )
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--isaac-request-timeout", type=float, default=0.0)
    parser.add_argument("--isaac-width", type=int, default=1024)
    parser.add_argument("--isaac-height", type=int, default=768)
    parser.add_argument("--isaac-tabletop-z", type=float, default=0.8)
    parser.add_argument("--isaac-hold-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isaac-hold-open-seconds", type=float, default=5.0)
    parser.add_argument("--isaac-show-object-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check-only", action="store_true", help="Only check exact BODex backend availability.")
    parser.add_argument(
        "--strict-success-exit-code",
        action="store_true",
        help="Exit 1 if any sequence has zero successful grasp seeds, in addition to solver/visualization failures.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds <= 0:
        print("--seeds must be positive", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    try:
        report = check_exact_bodex_v2_backend()
    except ExactBodexUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.check_only:
        print(f"Exact BODex-curobo-v2 backend is available. Active cuRobo path: {report.curobo_path}")
        return 0

    from ocir.grasp_synthesis.bodex_curobo_v2.solver import solve_sharpa_bodex

    jobs: list[tuple[Path, Path]] = []
    if args.surface_artifact is not None:
        surface = args.surface_artifact
        sequence_dir = args.sequence_dir or surface.parents[1]
        jobs.append((surface, sequence_dir))
    elif args.sequence_dir is not None:
        jobs.append((infer_surface_artifact_path(args.sequence_dir), args.sequence_dir))
    else:
        manifest = json.loads(Path(args.dexycb_manifest).read_text(encoding="utf-8"))
        for sequence in manifest.get("sequences", []):
            sequence_dir = Path(sequence.get("sequence_dir") or sequence["path"])
            jobs.append((infer_surface_artifact_path(sequence_dir), sequence_dir))

    summaries = []
    failed_visualizations = 0
    solver_failures = 0
    strict_failures = 0
    for surface_artifact, sequence_dir in jobs:
        sequence_name = sequence_dir.name
        out_dir = Path(args.out_dir) / sequence_name
        print(f"OCIR_BODEX_CUROBO_V2 solving {sequence_name} surface={surface_artifact}", flush=True)
        try:
            run_summary = solve_sharpa_bodex(
                surface_artifact=surface_artifact,
                out_dir=out_dir,
                object_mesh=args.object_mesh,
                seeds=args.seeds,
                top_k=args.top_k,
                opt_iters=args.opt_iters,
                seed=args.seed,
                grasp_threshold=args.grasp_threshold,
                distance_threshold=args.distance_threshold,
            )
        except Exception as exc:
            solver_failures += 1
            run_summary = {
                "ok": False,
                "backend": "curobo_v2_bodex_exact_single_object",
                "sequence_id": sequence_name,
                "surface_artifact": str(surface_artifact),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"OCIR_BODEX_CUROBO_V2 solver failed for {sequence_name}: {exc}", file=sys.stderr, flush=True)
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
                "manifest": str(args.dexycb_manifest),
                "sequence_id": run_summary.get("sequence_id"),
                "grasp_json": grasp_json,
                "surface_artifact": str(surface_artifact),
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
                    f"OCIR_BODEX_CUROBO_V2 {sequence_name}: no strict success, visualizing best-ranked "
                    f"failed seed instead ({grasp_json}).",
                    flush=True,
                )
            if grasp_json is None:
                reason = (
                    "no grasp_json or failed_grasp_json available"
                    if args.isaac_visualize_failed
                    else "no grasp_json (synthesis did not produce a successful grasp; "
                    "pass --isaac-visualize-failed to visualize the best failed seed instead)"
                )
                print(
                    f"OCIR_BODEX_CUROBO_V2 skipping Isaac visualization for {sequence_name}: {reason}.",
                    file=sys.stderr,
                    flush=True,
                )
                failed_visualizations += 1
            elif not server_is_running(args.control_host, int(args.control_port), timeout=1.0):
                print(
                    "OCIR_BODEX_CUROBO_V2 Isaac visualization requested, but no persistent Isaac control "
                    f"server is running at {args.control_host}:{args.control_port}. Start it with "
                    "scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py first.",
                    file=sys.stderr,
                    flush=True,
                )
                failed_visualizations += 1
            else:
                status = request_json(args.control_host, int(args.control_port), "GET", "/status", timeout=1.0)
                registered = {task.get("name") for task in status.get("registered_tasks", [])}
                if "grasp_pose_visualization" not in registered:
                    print(
                        "OCIR_BODEX_CUROBO_V2 persistent Isaac server is running, but it has not registered "
                        "'grasp_pose_visualization'. Restart the server with scripts/isaac/visualize_grasp.py "
                        "in the task module list.",
                        file=sys.stderr,
                        flush=True,
                    )
                    failed_visualizations += 1
                else:
                    code, response = submit_job(
                        host=args.control_host,
                        port=int(args.control_port),
                        task="grasp_pose_visualization",
                        params=params,
                        request_timeout=float(args.isaac_request_timeout),
                    )
                    run_summary["isaac_visualization"] = {
                        "ok": code == 0,
                        "out_dir": str(vis_out_dir),
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
        "backend": "curobo_v2_bodex_exact_single_object",
        "curobo_path": str(report.curobo_path),
        "solver_failures": solver_failures,
        "failed_visualizations": failed_visualizations,
        "strict_failures": strict_failures,
        "runs": summaries,
    }
    summary_path = Path(args.out_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"OCIR_BODEX_CUROBO_V2 done: ok={overall_ok} runs={len(summaries)} "
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
