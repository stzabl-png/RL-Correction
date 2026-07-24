#!/usr/bin/env python3
"""HaWoR hand MANO in ViPE world frame with SAM3 hand filter (no infiller)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.artifacts import discard_hawor_nonessential  # noqa: E402
from _common.dataset import VideoJob  # noqa: E402
from _common.paths import interim_step_dir, is_step_complete, write_step_completion  # noqa: E402
from _common.viz import vis_path  # noqa: E402

def _patch_torch_load_weights_only() -> None:
    """torch>=2.6 defaults torch.load(weights_only=True), which rejects HaWoR's
    omegaconf-containing checkpoints. These are trusted local files -> default to
    weights_only=False so HAWOR.load_from_checkpoint works on torch 2.11."""
    import torch

    if getattr(torch.load, "_recon_compat", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._recon_compat = True
    torch.load = _load


_patch_torch_load_weights_only()

HAWOR_CAMERA_TIMELINE = "vipe_time_aligned_depth_v2"


def _ensure_sam3_hands_legacy_marker(sam3_dir: Path, video_id: str) -> None:
    recon_marker = sam3_dir / "sam3_hands_complete.json"
    legacy_dir = sam3_dir / video_id
    legacy_marker = legacy_dir / "hand_masks_complete.json"
    if legacy_marker.is_file() or not recon_marker.is_file():
        return
    payload = json.loads(recon_marker.read_text(encoding="utf-8"))
    payload.setdefault("status", "complete")
    payload.setdefault("sequence", video_id)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_hawor_common():
    name = "hawor_common"
    if name in sys.modules:
        return sys.modules[name]
    hawor_dir = RECON_ROOT / "_legacy" / "hawor"
    if str(hawor_dir) not in sys.path:
        sys.path.insert(0, str(hawor_dir))
    path = hawor_dir / "common.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def run_hawor(job: VideoJob, *, gpu: int, visualize: bool, force: bool) -> dict:
    hawor = _load_hawor_common()
    step_dir = interim_step_dir(job.dataset, job.video_id, "hawor")
    vipe_dir = interim_step_dir(job.dataset, job.video_id, "vipe")
    sam3_dir = interim_step_dir(job.dataset, job.video_id, "sam3_hands")
    _ensure_sam3_hands_legacy_marker(sam3_dir, job.video_id)

    complete_marker = step_dir / "hawor_complete.json"
    if not force and is_step_complete(step_dir, "hawor"):
        try:
            payload = json.loads(complete_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("camera_timeline") == HAWOR_CAMERA_TIMELINE:
            return {"skipped": True}
        print(f"[hawor] stale camera timeline; rerunning {job.video_id}", flush=True)

    config = hawor.HaworRunConfig(
        output_dir=step_dir,
        vipe_camera_params=vipe_dir,
        sam3_dir=sam3_dir,
        use_infiller=False,
        sam3_filter=True,
        visualize=visualize,
        sam3_filter_debug_video=visualize,
        cleanup_intermediates=True,
    )
    result = hawor.run_hawor_sequence(
        video_path=job.video_path,
        sequence_name=job.video_id,
        config=config,
        gpu_id=gpu,
    )
    hawor.cleanup_hawor_run_artifacts(step_dir)

    extra = {
        "world_result": result.get("world_result"),
        "camera_source": "vipe",
        "camera_timeline": HAWOR_CAMERA_TIMELINE,
        "depth_source": "vipe",
        "depth_representation": "inverse_depth_disparity",
        "infiller": False,
    }
    if visualize and result.get("vis_video"):
        std = vis_path(step_dir, job.video_id)
        src = Path(str(result["vis_video"]))
        if src.is_file() and src != std:
            std.parent.mkdir(parents=True, exist_ok=True)
            src.rename(std)
        if std.is_file():
            extra["vis_video"] = str(std)

    discard_hawor_nonessential(step_dir, job.video_id)

    write_step_completion(
        step_dir,
        "hawor",
        dataset=job.dataset,
        video_id=job.video_id,
        extra=extra,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    job = VideoJob(dataset=args.dataset, video_id=args.video_id, video_path=args.video.resolve())
    run_hawor(job, gpu=args.gpu, visualize=args.visualize, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
