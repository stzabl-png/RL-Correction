#!/usr/bin/env python3
"""Propagate SAM2 object masks after manual labeling."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parents[1]
SAM2_OBJ_DIR = Path(__file__).resolve().parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))
if str(SAM2_OBJ_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_OBJ_DIR))

from _common.dataset import VideoJob  # noqa: E402
from _common.paths import REPO_ROOT, interim_step_dir, is_step_complete, write_step_completion  # noqa: E402
from sam2_object_common import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_MODEL_CFG,
    has_label_prompt,
    label_prompt_path,
    run_object_masks,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--sam2-model-cfg", default=DEFAULT_SAM2_MODEL_CFG)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--visualize-object-id",
        default=None,
        help="Optional object id to render alone; by default visualization renders all objects with different colors",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Run SAM2 masks and visualization under sam2_object_diagnostic/ without writing the main completion marker",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    job = VideoJob(dataset=args.dataset, video_id=args.video_id, video_path=args.video.resolve())
    source_step_dir = interim_step_dir(job.dataset, job.video_id, "sam2_object")
    step_dir = source_step_dir
    source_prompt = label_prompt_path(source_step_dir)

    if args.diagnostic_only and not source_prompt.is_file():
        cached_prompt = REPO_ROOT / "data" / "object_labels" / job.dataset / job.video_id / "sam2_object" / "label_prompt.json"
        flat_cached_prompt = REPO_ROOT / "data" / "object_labels" / job.video_id / "sam2_object" / "label_prompt.json"
        legacy_cached_prompt = REPO_ROOT / "data" / "object_label_backup" / job.video_id / "sam2_object" / "label_prompt.json"
        if not cached_prompt.is_file() and flat_cached_prompt.is_file():
            cached_prompt = flat_cached_prompt
        if not cached_prompt.is_file() and legacy_cached_prompt.is_file():
            cached_prompt = legacy_cached_prompt
        if cached_prompt.is_file():
            source_prompt = cached_prompt

    if not source_prompt.is_file():
        print(
            f"Skip {job.video_id}: no label_prompt.json — run label_object.py first or restore cached labels",
            file=sys.stderr,
        )
        return 2

    if args.diagnostic_only:
        step_dir = interim_step_dir(job.dataset, job.video_id, "sam2_object_diagnostic")
        shutil.rmtree(step_dir, ignore_errors=True)
        step_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_prompt, label_prompt_path(step_dir))
        args.visualize = True
        args.force = True

    if not args.force and is_step_complete(step_dir, "sam2_object"):
        return 0

    stats = run_object_masks(
        video_path=job.video_path,
        step_dir=step_dir,
        video_id=job.video_id,
        gpu_id=args.gpu,
        visualize=args.visualize,
        visualize_object_id=args.visualize_object_id,
        checkpoint=args.sam2_checkpoint,
        model_cfg=args.sam2_model_cfg,
    )
    if args.diagnostic_only:
        print(f"Diagnostic SAM2 masks: {step_dir / 'video_segmentation' / 'masks'}", flush=True)
        if "vis_video" in stats:
            print(f"Diagnostic SAM2 visualization: {stats['vis_video']}", flush=True)
        return 0
    write_step_completion(
        step_dir,
        "sam2_object",
        dataset=job.dataset,
        video_id=job.video_id,
        extra=stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
