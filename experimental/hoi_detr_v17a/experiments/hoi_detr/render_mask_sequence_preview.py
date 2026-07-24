"""Render an instance-mask manifest as a user-reviewable MP4 video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


_COLORS_BGR = (
    (80, 210, 255),
    (255, 130, 70),
    (120, 235, 120),
    (240, 100, 230),
    (245, 220, 70),
)


def _accepted_masks_by_frame(
    manifest: dict,
    *,
    object_id_prefix: str = "",
) -> dict[int, dict[str, Path]]:
    """Extract only accepted masks, keeping the manifest's stable object IDs."""
    result: dict[int, dict[str, Path]] = {}
    for frame in manifest.get("frames", []):
        accepted = {}
        for object_id, entry in frame.get("objects", {}).items():
            if entry.get("status") == "accepted" and entry.get("mask"):
                accepted[f"{object_id_prefix}{object_id}"] = Path(entry["mask"])
        if accepted:
            result[int(frame["frame_idx"])] = accepted
    return result


def _raw_masks_by_frame(
    root: Path,
    *,
    object_id_prefix: str = "",
) -> dict[int, dict[str, Path]]:
    """Read raw-mask artifacts from ``frame_NNNNNN/object_id.png`` folders."""

    result = {}
    for frame_dir in sorted(root.glob("frame_*")):
        if not frame_dir.is_dir():
            continue
        try:
            frame_idx = int(frame_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        masks = {
            f"{object_id_prefix}{path.stem}": path
            for path in sorted(frame_dir.glob("*.png"))
        }
        if masks:
            result[frame_idx] = masks
    return result


def render_preview(
    *,
    manifest_paths: list[Path] | None,
    output_path: Path,
    watermark: str | None = None,
    raw_mask_roots: list[Path] | None = None,
    video_path: Path | None = None,
) -> Path:
    """Overlay accepted masks from one or more manifests onto their source video."""
    manifest_paths = manifest_paths or []
    raw_mask_roots = raw_mask_roots or []
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    if not manifests and not raw_mask_roots:
        raise ValueError("at least one manifest or raw mask root is required")
    if video_path is None:
        if not manifests:
            raise ValueError("--video is required when rendering raw masks only")
        video_path = Path(manifests[0]["video"])
    if any(Path(manifest["video"]) != video_path for manifest in manifests):
        raise ValueError("all review manifests must reference the same source video")
    masks_by_frame: dict[int, dict[str, Path]] = {}
    for manifest_idx, manifest in enumerate(manifests):
        prefix = "" if len(manifests) == 1 else f"episode_{manifest_idx:02d}/"
        for frame_idx, masks in _accepted_masks_by_frame(
            manifest, object_id_prefix=prefix
        ).items():
            if frame_idx in masks_by_frame and set(masks_by_frame[frame_idx]) & set(masks):
                raise ValueError(f"duplicate object ID at frame {frame_idx} across review manifests")
            masks_by_frame.setdefault(frame_idx, {}).update(masks)
    for root_idx, root in enumerate(raw_mask_roots):
        prefix = "raw/" if len(raw_mask_roots) == 1 else f"raw_{root_idx:02d}/"
        for frame_idx, masks in _raw_masks_by_frame(
            root, object_id_prefix=prefix
        ).items():
            if frame_idx in masks_by_frame and set(masks_by_frame[frame_idx]) & set(masks):
                raise ValueError(f"duplicate raw object ID at frame {frame_idx}")
            masks_by_frame.setdefault(frame_idx, {}).update(masks)
    object_ids = sorted({object_id for masks in masks_by_frame.values() for object_id in masks})
    colors = {object_id: _COLORS_BGR[index % len(_COLORS_BGR)] for index, object_id in enumerate(object_ids)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create preview video: {output_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            overlay = frame.copy()
            for object_id, mask_path in masks_by_frame.get(frame_idx, {}).items():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != frame.shape[:2]:
                    raise RuntimeError(f"Cannot use mask for {object_id} at frame {frame_idx}: {mask_path}")
                foreground = mask > 0
                color = np.asarray(colors[object_id], dtype=np.float32)
                overlay[foreground] = (0.48 * overlay[foreground] + 0.52 * color).astype(np.uint8)
                ys, xs = np.where(foreground)
                if len(xs):
                    cv2.putText(
                        overlay,
                        object_id,
                        (int(xs.min()), max(20, int(ys.min()) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        colors[object_id],
                        2,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                overlay,
                f"frame {frame_idx}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if watermark:
                cv2.putText(
                    overlay,
                    watermark,
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(overlay)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append")
    parser.add_argument("--raw-mask-root", type=Path, action="append")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watermark")
    args = parser.parse_args(argv)
    print(
        render_preview(
            manifest_paths=args.manifest,
            output_path=args.output,
            watermark=args.watermark,
            raw_mask_roots=args.raw_mask_root,
            video_path=args.video,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
