"""Combine ready per-cycle sequences into one video-wide fixed-ID manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import write_json_atomic
from .segmentation_report import write_segmentation_report
from .video_mask_sequence import combine_mask_sequences


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_registry = json.loads(args.video_registry.read_text(encoding="utf-8"))
    manifests = []
    for path in args.cycle_sequence:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["manifest_path"] = str(path.resolve())
        manifests.append(manifest)
    combined = combine_mask_sequences(
        video_registry,
        manifests,
    )
    manifest_path = None
    report_path = None
    if combined["status"] == "ready":
        manifest_path = output_dir / "video_mask_sequence.json"
        write_json_atomic(manifest_path, combined)
        report_path = output_dir / "segmentation_report.md"
        write_segmentation_report(combined, report_path)
    failed_videos = [] if combined["status"] == "ready" else [str(args.video.resolve())]
    summary = {
        "schema_version": "persistent_video_mask_sequence_run_v1",
        "status": "success" if combined["status"] == "ready" else combined["status"],
        "video": str(args.video.resolve()),
        "object_count": len(combined["objects"]),
        "objects": combined["objects"],
        "cross_object_conflict_count": len(combined["cross_object_conflicts"]),
        "failures": combined["failures"],
        "video_mask_sequence": None if manifest_path is None else str(manifest_path.resolve()),
        "segmentation_report": None if report_path is None else str(report_path.resolve()),
        "failed_videos": failed_videos,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-registry", type=Path, required=True)
    parser.add_argument("--cycle-sequence", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "persistent_video_mask_sequence_run_v1",
            "status": "failed_fatal_error",
            "video": str(args.video.resolve()),
            "failures": [{"reason": f"{type(exc).__name__}: {exc}"}],
            "video_mask_sequence": None,
            "failed_videos": [str(args.video.resolve())],
        }
        write_json_atomic(args.output_dir / "fatal_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed_videos"]:
        print("FAILED VIDEOS:")
        for video in summary["failed_videos"]:
            print(video)
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
