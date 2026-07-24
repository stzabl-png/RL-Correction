"""Create a relation-free multi-instance reconditioning registry at one bad frame."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from .adapter import write_json_atomic
from .instance_association import (
    InstanceCandidate,
    InstanceTrack,
    associate_tracks_to_candidates,
    derive_residual_instance_candidates,
)


def _read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    mask = image > 0
    if not np.any(mask):
        raise ValueError(f"mask is empty: {path}")
    return mask


def _read_amg_candidates(summary_path: Path) -> list[InstanceCandidate]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "success":
        raise ValueError("AMG candidate generation was not successful")
    candidates = []
    for record in summary.get("candidates", []):
        quality = float(record["predicted_iou"]) * float(record["stability_score"])
        candidates.append(
            InstanceCandidate(
                candidate_id=f"candidate_{int(record['candidate_idx']):03d}",
                mask=_read_mask(Path(record["mask"])),
                quality_score=quality,
                source="sam2_amg_recovery_candidate",
            )
        )
    if not candidates:
        raise ValueError("AMG summary contains no usable candidates")
    return candidates


def _tracks_from_registry(
    registry: dict,
    *,
    unresolved_mask_dir: Path,
) -> list[InstanceTrack]:
    tracks = []
    for object_id, entry in registry.get("objects", {}).items():
        conditions = entry.get("conditioning_masks", [])
        anchor_path = Path(conditions[0]["mask"] if conditions else entry["mask"])
        prediction_path = unresolved_mask_dir / f"{object_id}.png"
        tracks.append(
            InstanceTrack(
                object_id=object_id,
                anchor_mask=_read_mask(anchor_path),
                predicted_mask=_read_mask(prediction_path),
            )
        )
    if not tracks:
        raise ValueError("source registry contains no instance tracks")
    return tracks


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise RuntimeError(f"failed to write mask: {path}")


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_registry = json.loads(args.source_registry.read_text(encoding="utf-8"))
    if source_registry.get("status") != "ready":
        raise ValueError("source registry is not ready")
    tracks = _tracks_from_registry(
        source_registry,
        unresolved_mask_dir=args.unresolved_mask_dir.resolve(),
    )
    candidates = _read_amg_candidates(args.candidate_summary)
    candidates.extend(
        derive_residual_instance_candidates(
            candidates,
            min_child_inside_parent=args.min_child_inside_parent,
            min_residual_area=args.min_residual_area,
        )
    )
    association = associate_tracks_to_candidates(
        tracks,
        candidates,
        max_cross_instance_overlap_fraction=args.max_cross_instance_overlap_fraction,
        min_assignment_margin=args.min_assignment_margin,
    )
    if association["status"] != "success":
        summary = {
            "schema_version": "instance_recovery_v1",
            "status": association["status"],
            "recovery_frame": args.recovery_frame,
            "association": association,
            "reconstruction_registry": None,
            "failed_videos": [str(args.video.resolve())],
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return summary

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    revised = deepcopy(source_registry)
    conditioning_dir = output_dir / "conditioning_masks"
    for object_id, candidate_id in association["assignment"].items():
        entry = revised["objects"][object_id]
        if any(
            int(condition["frame_idx"]) == args.recovery_frame
            for condition in entry.get("conditioning_masks", [])
        ):
            raise ValueError(
                f"{object_id} already has a conditioning mask at recovery frame"
            )
        candidate = candidate_by_id[candidate_id]
        mask_path = conditioning_dir / f"{object_id}_{args.recovery_frame:06d}.png"
        _write_mask(mask_path, candidate.mask)
        entry.setdefault("conditioning_masks", []).append(
            {
                "frame_idx": args.recovery_frame,
                "mask": str(mask_path.resolve()),
                "source": "automatic_mutually_exclusive_instance_recovery",
                "candidate_id": candidate_id,
                "candidate_source": candidate.source,
                "candidate_quality_score": candidate.quality_score,
            }
        )
    revised["automatic_recovery"] = {
        "frame_idx": args.recovery_frame,
        "method": "candidate_to_track_one_to_one_visual_association",
        "association": association,
    }
    registry_path = output_dir / "reconstruction_registry.json"
    write_json_atomic(registry_path, revised)
    summary = {
        "schema_version": "instance_recovery_v1",
        "status": "success",
        "recovery_frame": args.recovery_frame,
        "association": association,
        "reconstruction_registry": str(registry_path.resolve()),
        "failed_videos": [],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--unresolved-mask-dir", type=Path, required=True)
    parser.add_argument("--recovery-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-child-inside-parent", type=float, default=0.9)
    parser.add_argument("--min-residual-area", type=int, default=256)
    parser.add_argument("--max-cross-instance-overlap-fraction", type=float, default=0.05)
    parser.add_argument("--min-assignment-margin", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        summary = {
            "schema_version": "instance_recovery_v1",
            "status": "failed_fatal_error",
            "recovery_frame": args.recovery_frame,
            "failures": [{"reason": f"{type(exc).__name__}: {exc}"}],
            "reconstruction_registry": None,
            "failed_videos": [str(args.video.resolve())],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output_dir / "fatal_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed_videos"]:
        print("FAILED VIDEOS:")
        for video in summary["failed_videos"]:
            print(video)
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
