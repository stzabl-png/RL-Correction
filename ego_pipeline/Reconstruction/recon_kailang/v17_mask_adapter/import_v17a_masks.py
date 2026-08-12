#!/usr/bin/env python3
"""Import one selected v17A object track into the reconstruction mask layout.

This adapter is intentionally selection-agnostic: an upstream selector supplies
the episode manifest, source object id, and reconstruction frame.  It does not
try to decide which episode, object, or frame is best.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[2]
RECON_ROOT = RECONSTRUCTION_ROOT / "recon_pipeline"
SAM2_OBJECT_DIR = RECON_ROOT / "sam2_object"
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))
if str(SAM2_OBJECT_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_OBJECT_DIR))

from _common.io import count_video_frames  # noqa: E402
from _common.paths import interim_step_dir, write_step_completion  # noqa: E402
from sam2_object_common import (  # noqa: E402
    OBJECT_MASK_ID,
    LabelPrompt,
    ObjectPrompt,
    object_mask_filename,
    save_label_prompt,
    validate_object_id,
)

IMPORT_META_FILENAME = "v17a_import.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (
        "persistent_video_mask_sequence_v1",  # stage B 实际输出的 schema 名
        "persistent_mask_sequence_v1",
    ):
        raise ValueError(
            f"Unsupported v17A manifest schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "ready":
        raise ValueError(f"v17A mask sequence is not ready: {manifest.get('status')!r}")
    return manifest


def _video_geometry(video_path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    num_frames = count_video_frames(video_path)
    if num_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video geometry for {video_path}: frames={num_frames}, size={width}x{height}"
        )
    return num_frames, height, width


def _manifest_observations(
    manifest: dict[str, Any], source_object_id: str
) -> dict[int, dict[str, Any]]:
    if source_object_id not in set(manifest.get("object_ids") or []):
        raise KeyError(
            f"Source object {source_object_id!r} not present; "
            f"available={manifest.get('object_ids') or []}"
        )
    observations: dict[int, dict[str, Any]] = {}
    for frame in manifest.get("frames") or []:
        frame_idx = int(frame["frame_idx"])
        obj = (frame.get("objects") or {}).get(source_object_id)
        if obj is not None:
            observations[frame_idx] = obj
    return observations


def _read_source_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read v17A mask: {path}")
    if mask.shape != expected_shape:
        raise ValueError(
            f"v17A mask has shape {mask.shape}, expected {expected_shape}: {path}"
        )
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _interior_point(mask: np.ndarray) -> tuple[float, float]:
    """Return a guaranteed foreground point nearest the mask centroid."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError("Selected reconstruction mask is empty")
    cx = float(xs.mean())
    cy = float(ys.mean())
    nearest = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return float(xs[nearest]), float(ys[nearest])


def import_v17a_sequence(
    *,
    video_path: Path,
    manifest_path: Path,
    step_dir: Path,
    dataset: str,
    video_id: str,
    source_object_id: str,
    reconstruction_frame: int,
    output_object_id: str = OBJECT_MASK_ID,
    object_name: str = "object",
    force: bool = False,
) -> dict[str, Any]:
    """Write a complete SAM2-compatible sequence for one selected v17A track."""
    video_path = video_path.resolve()
    manifest_path = manifest_path.resolve()
    output_object_id = validate_object_id(output_object_id)
    manifest = _load_manifest(manifest_path)
    observations = _manifest_observations(manifest, source_object_id)
    num_frames, height, width = _video_geometry(video_path)

    manifest_shape = tuple(int(v) for v in (manifest.get("frame_shape") or []))
    if manifest_shape and manifest_shape != (height, width):
        raise ValueError(
            f"Manifest/video shape mismatch: manifest={manifest_shape}, video={(height, width)}"
        )
    if not 0 <= reconstruction_frame < num_frames:
        raise ValueError(
            f"Reconstruction frame {reconstruction_frame} is outside [0, {num_frames})"
        )

    selected = observations.get(reconstruction_frame)
    if selected is None or selected.get("status") != "accepted" or not selected.get("mask"):
        raise ValueError(
            f"Frame {reconstruction_frame} is not an accepted mask for {source_object_id}"
        )

    if step_dir.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing step directory: {step_dir}")
        shutil.rmtree(step_dir)
    masks_root = step_dir / "video_segmentation" / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)

    detected_frames = 0
    selected_mask: np.ndarray | None = None
    zero_mask = np.zeros((height, width), dtype=np.uint8)
    target_name = object_mask_filename(output_object_id)
    for frame_idx in range(num_frames):
        target_dir = masks_root / f"frame_{frame_idx:06d}_masks"
        target_dir.mkdir(parents=True, exist_ok=True)
        observation = observations.get(frame_idx)
        mask = zero_mask
        if (
            observation is not None
            and observation.get("status") == "accepted"
            and observation.get("mask")
        ):
            mask = _read_source_mask(Path(observation["mask"]), (height, width))
            if np.any(mask):
                detected_frames += 1
        if frame_idx == reconstruction_frame:
            selected_mask = mask.copy()
        if not cv2.imwrite(str(target_dir / target_name), mask):
            raise RuntimeError(f"Failed to write mask for frame {frame_idx}")

    if selected_mask is None or not np.any(selected_mask):
        raise RuntimeError(f"Selected reconstruction mask is empty at frame {reconstruction_frame}")
    point = _interior_point(selected_mask)
    prompt = LabelPrompt(
        objects=[
            ObjectPrompt(
                object_id=output_object_id,
                frame_idx=reconstruction_frame,
                points=[point],
                labels=[1],
                name=object_name,
                locked=True,
            )
        ]
    )
    prompt_path = save_label_prompt(step_dir, prompt)

    metadata = {
        "backend": "v17a_manifest_import",
        "source_manifest": str(manifest_path),
        "source_object_id": source_object_id,
        "object_id": output_object_id,
        "object_name": object_name,
        "num_frames": num_frames,
        "detected_frames": detected_frames,
        "reconstruction_frame": reconstruction_frame,
        "prompt_point_xy": list(point),
        "masks_dir": str(masks_root),
        "label_prompt": str(prompt_path),
        "selection_policy": "upstream_explicit",
    }
    (step_dir / IMPORT_META_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_step_completion(
        step_dir,
        "sam2_object",
        dataset=dataset,
        video_id=video_id,
        extra=metadata,
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-object-id", required=True)
    parser.add_argument("--reconstruction-frame", type=int, required=True)
    parser.add_argument("--output-object-id", default=OBJECT_MASK_ID)
    parser.add_argument("--object-name", default="object")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    step_dir = interim_step_dir(args.dataset, args.video_id, "sam2_object")
    metadata = import_v17a_sequence(
        video_path=args.video,
        manifest_path=args.manifest,
        step_dir=step_dir,
        dataset=args.dataset,
        video_id=args.video_id,
        source_object_id=args.source_object_id,
        reconstruction_frame=args.reconstruction_frame,
        output_object_id=args.output_object_id,
        object_name=args.object_name,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
