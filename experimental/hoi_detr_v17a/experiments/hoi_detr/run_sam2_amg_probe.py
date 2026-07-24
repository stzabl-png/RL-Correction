"""Enumerate text-free SAM2 object candidates on one video frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .adapter import write_json_atomic


def _hand_region(shape: tuple[int, int], detection_frame: dict) -> np.ndarray:
    height, width = shape
    region = np.zeros(shape, dtype=bool)
    for detection in detection_frame.get("detections", []):
        if detection.get("class_name") != "hand":
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in detection["box_xyxy"]]
        region[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
    return region


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    detection_frame = detections["frames"][args.frame_idx]
    if int(detection_frame["frame_idx"]) != args.frame_idx:
        raise ValueError("detection frames must be directly indexed by frame_idx")

    cap = cv2.VideoCapture(str(args.video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {args.frame_idx}: {args.video}")
    finally:
        cap.release()

    sys.path.insert(0, str(args.sam2_root.resolve()))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(
        args.model_cfg,
        str(args.checkpoint.resolve()),
        device=f"cuda:{args.gpu}",
        apply_postprocessing=True,
    )
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_threshold,
        stability_score_thresh=args.stability_threshold,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )
    candidates = generator.generate(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand_region = _hand_region(frame.shape[:2], detection_frame)
    filtered = []
    for candidate in candidates:
        mask = np.asarray(candidate["segmentation"], dtype=bool)
        area = int(np.count_nonzero(mask))
        if area < args.min_area or area > args.max_area:
            continue
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            continue
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_fraction = float(component_areas.max() / area) if component_areas.size else 0.0
        if largest_fraction < args.min_largest_component_fraction:
            continue
        hand_overlap = float(np.count_nonzero(mask & hand_region) / area)
        if hand_overlap > args.max_hand_overlap:
            continue
        filtered.append(
            {
                "mask": mask,
                "area": area,
                "bbox_xywh": [float(value) for value in candidate["bbox"]],
                "predicted_iou": float(candidate["predicted_iou"]),
                "stability_score": float(candidate["stability_score"]),
                "largest_component_fraction": largest_fraction,
                "hand_box_overlap_fraction": hand_overlap,
            }
        )
    filtered.sort(
        key=lambda item: (
            item["predicted_iou"] * item["stability_score"],
            item["area"],
        ),
        reverse=True,
    )

    records = []
    panels = []
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for candidate_idx, candidate in enumerate(filtered):
        mask_path = mask_dir / f"candidate_{candidate_idx:03d}.png"
        if not cv2.imwrite(str(mask_path), candidate["mask"].astype(np.uint8) * 255):
            raise RuntimeError(f"failed to write mask: {mask_path}")
        record = {key: value for key, value in candidate.items() if key != "mask"}
        record["candidate_idx"] = candidate_idx
        record["mask"] = str(mask_path.resolve())
        records.append(record)
        panel = frame.copy()
        mask = candidate["mask"]
        panel[mask] = (0.4 * panel[mask] + 0.6 * np.asarray((0, 210, 255))).astype(np.uint8)
        cv2.putText(
            panel,
            f"candidate {candidate_idx} area={candidate['area']}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        panels.append(cv2.resize(panel, (480, 270)))

    if panels:
        columns = min(4, len(panels))
        rows = []
        blank = np.zeros_like(panels[0])
        for offset in range(0, len(panels), columns):
            row = panels[offset : offset + columns]
            row += [blank] * (columns - len(row))
            rows.append(np.hstack(row))
        contact_sheet = np.vstack(rows)
        vis_path = output_dir / "vis" / "candidates.jpg"
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(vis_path), contact_sheet):
            raise RuntimeError(f"failed to write visualization: {vis_path}")
    else:
        vis_path = None

    summary = {
        "schema_version": "sam2_amg_probe_v1",
        "status": "success" if records else "failed_no_candidates",
        "video": str(args.video.resolve()),
        "frame_idx": args.frame_idx,
        "raw_candidate_count": len(candidates),
        "filtered_candidate_count": len(records),
        "candidates": records,
        "visualization": None if vis_path is None else str(vis_path.resolve()),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--points-per-batch", type=int, default=32)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.8)
    parser.add_argument("--stability-threshold", type=float, default=0.92)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--max-area", type=int, default=100000)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.98)
    parser.add_argument("--max-hand-overlap", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = run(_parser().parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
