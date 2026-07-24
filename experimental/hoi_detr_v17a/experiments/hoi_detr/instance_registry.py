"""Build a relation-free immutable registry of component instance seeds."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .adapter import write_json_atomic
from .instance_association import InstanceCandidate


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise RuntimeError(f"failed to write mask: {path}")


def write_instance_registry(
    candidates: Iterable[InstanceCandidate],
    *,
    source_frame: int,
    interaction_start_frame: int,
    interaction_end_frame: int,
    component_decision: dict,
    output_dir: Path,
) -> dict:
    """Persist one clean mask per fixed instance ID without semantic relations."""

    if source_frame < 0 or interaction_start_frame < 0 or interaction_end_frame < interaction_start_frame:
        raise ValueError("invalid frame indices")
    selected_candidate_ids = component_decision.get("selected_candidate_ids")
    if not isinstance(selected_candidate_ids, list) or not selected_candidate_ids:
        raise ValueError("component decision must select at least one candidate")
    candidate_items = list(candidates)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_items}
    if len(candidate_by_id) != len(candidate_items):
        raise ValueError("candidate IDs must be unique")
    if set(selected_candidate_ids).difference(candidate_by_id):
        raise ValueError("component decision references an unknown candidate")
    selected = [candidate_by_id[candidate_id] for candidate_id in selected_candidate_ids]
    shape = np.asarray(selected[0].mask, dtype=bool).shape
    if len(shape) != 2:
        raise ValueError("candidate masks must be 2D")
    for index, first in enumerate(selected):
        first_mask = np.asarray(first.mask, dtype=bool)
        if first_mask.shape != shape or not np.any(first_mask):
            raise ValueError("selected candidate masks must be non-empty and share one shape")
        for second in selected[index + 1 :]:
            if np.any(first_mask & np.asarray(second.mask, dtype=bool)):
                raise ValueError("selected component masks overlap")

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = {}
    object_order = []
    for index, candidate in enumerate(selected, start=1):
        object_id = f"instance_{index:04d}"
        mask_path = output_dir / "seeds" / f"{object_id}.png"
        _write_mask(mask_path, np.asarray(candidate.mask, dtype=bool))
        objects[object_id] = {
            "activation_frame": interaction_start_frame,
            "frame_idx": source_frame,
            "mask": str(mask_path.resolve()),
            "conditioning_masks": [
                {
                    "frame_idx": source_frame,
                    "mask": str(mask_path.resolve()),
                    "source": candidate.source,
                    "candidate_id": candidate.candidate_id,
                    "quality_score": candidate.quality_score,
                }
            ],
            "instance_source": candidate.source,
            "source_candidate_id": candidate.candidate_id,
        }
        object_order.append(object_id)
    registry = {
        "schema_version": "instance_component_registry_v1",
        "status": "ready",
        "objects": objects,
        "object_order": object_order,
        "component_decision": component_decision,
        "interaction_window": {
            "start_frame": interaction_start_frame,
            "end_frame": interaction_end_frame,
            "source_frame": source_frame,
        },
        "invariant": (
            "every output ID is a separate visible-instance mask; no semantic relation "
            "is inferred or required"
        ),
    }
    registry_path = output_dir / "reconstruction_registry.json"
    write_json_atomic(registry_path, registry)
    return {
        "status": "success",
        "reconstruction_registry": str(registry_path.resolve()),
        "object_ids": object_order,
    }
