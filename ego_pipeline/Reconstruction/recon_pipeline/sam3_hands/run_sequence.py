#!/usr/bin/env python3
"""SAM3 left/right hand masks for one video."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.artifacts import discard_sam3_hands_nonessential  # noqa: E402
from _common.dataset import VideoJob  # noqa: E402
from _common.paths import interim_step_dir, is_step_complete, resolve_repo_path, write_step_completion  # noqa: E402
from _common.viz import encode_hands_vis_mp4, vis_path  # noqa: E402


def _load_sam3_common():
    name = "recon_sam3_common"
    if name in sys.modules:
        return sys.modules[name]
    path = RECON_ROOT / "_legacy" / "sam3" / "common.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sam3_common from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_sam3_hands(
    job: VideoJob,
    *,
    gpu: int,
    visualize: bool,
    force: bool,
    frame_idx: int,
    checkpoint: Path | None,
) -> dict:
    sam3 = _load_sam3_common()
    step_dir = interim_step_dir(job.dataset, job.video_id, "sam3_hands")
    video_path = resolve_repo_path(job.video_path)

    if not force and is_step_complete(step_dir, "sam3_hands"):
        print(f"[sam3_hands] skipped (complete): {job.video_id}", flush=True)
        return {"skipped": True}

    if not video_path.is_file():
        raise FileNotFoundError(f"RGB video not found: {video_path}")

    print(f"[sam3_hands] running {job.video_id} → {step_dir}", flush=True)
    config = sam3.Sam3RunConfig(
        output_dir=step_dir,
        frame_idx=frame_idx,
        checkpoint=resolve_repo_path(checkpoint) if checkpoint is not None else None,
        visualize=False,
        cleanup_intermediates=True,
    )
    result = sam3.run_hand_masks_sequence(
        video_path,
        job.video_id,
        config,
        gpu_id=gpu,
    )

    masks_dir = sam3.masks_root(step_dir, job.video_id)
    if not masks_dir.is_dir():
        raise RuntimeError(f"SAM3 hand masks missing under {masks_dir}")

    extra = {
        "hands": result.get("hands", {}),
        "masks_dir": str(masks_dir),
        "sam3_version": result.get("sam3_version"),
        "sam3_requested_version": result.get("sam3_requested_version"),
        "sam3_version_fallback": result.get("sam3_version_fallback"),
    }
    if visualize:
        out_vis = encode_hands_vis_mp4(
            video_path=video_path,
            masks_dir=masks_dir,
            out_path=vis_path(step_dir, job.video_id),
        )
        extra["vis_video"] = str(out_vis)

    sam3.cleanup_sam3_run_artifacts(step_dir)
    discard_sam3_hands_nonessential(step_dir, job.video_id)

    write_step_completion(
        step_dir,
        "sam3_hands",
        dataset=job.dataset,
        video_id=job.video_id,
        extra=extra,
    )
    legacy_marker = step_dir / job.video_id / "hand_masks_complete.json"
    if not legacy_marker.is_file():
        legacy_marker.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "complete",
            "sequence": job.video_id,
            "hands": result.get("hands", {}),
            "sam3_version": result.get("sam3_version"),
            "sam3_requested_version": result.get("sam3_requested_version"),
            "sam3_version_fallback": result.get("sam3_version_fallback"),
            "masks_dir": str(masks_dir),
            "vis_video": extra.get("vis_video"),
        }
        legacy_marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[sam3_hands] done {job.video_id}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional local SAM3/SAM3.1 checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Write combined L+R hand mask MP4 under vis/")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    job = VideoJob(
        dataset=args.dataset,
        video_id=args.video_id,
        video_path=resolve_repo_path(args.video),
    )
    run_sam3_hands(
        job,
        gpu=args.gpu,
        visualize=args.visualize,
        force=args.force,
        frame_idx=args.frame_idx,
        checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
