"""Decide conservatively whether visual instances warrant separate component IDs."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ComponentMotionEvidence:
    first_candidate_id: str
    second_candidate_id: str
    frame_indices: tuple[int, ...]
    relative_displacement_diagonals: float
    jointly_visible_frame_count: int
    reliable: bool

    def to_dict(self) -> dict:
        return {
            "first_candidate_id": self.first_candidate_id,
            "second_candidate_id": self.second_candidate_id,
            "frame_indices": list(self.frame_indices),
            "relative_displacement_diagonals": self.relative_displacement_diagonals,
            "jointly_visible_frame_count": self.jointly_visible_frame_count,
            "reliable": self.reliable,
        }


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not xs.size:
        return None
    return float(xs.mean()), float(ys.mean())


def component_motion_evidence(
    first_candidate_id: str,
    second_candidate_id: str,
    tracked_masks: Mapping[int, Mapping[str, np.ndarray]],
    *,
    frame_shape: tuple[int, int],
    min_jointly_visible_frames: int = 3,
    min_relative_displacement_diagonals: float = 0.03,
) -> ComponentMotionEvidence:
    """Measure whether two proposed instances move independently over time."""

    if min_jointly_visible_frames < 2:
        raise ValueError("min_jointly_visible_frames must be at least two")
    if min_relative_displacement_diagonals < 0.0:
        raise ValueError("min_relative_displacement_diagonals must be non-negative")
    diagonal = hypot(*frame_shape)
    relative_positions = []
    frame_indices = []
    for frame_idx in sorted(tracked_masks):
        masks = tracked_masks[frame_idx]
        if first_candidate_id not in masks or second_candidate_id not in masks:
            continue
        first = _centroid(masks[first_candidate_id])
        second = _centroid(masks[second_candidate_id])
        if first is None or second is None:
            continue
        relative_positions.append((first[0] - second[0], first[1] - second[1]))
        frame_indices.append(frame_idx)
    if len(relative_positions) < min_jointly_visible_frames:
        displacement = 0.0
    else:
        origin = relative_positions[0]
        displacement = max(
            hypot(position[0] - origin[0], position[1] - origin[1]) / diagonal
            for position in relative_positions[1:]
        )
    return ComponentMotionEvidence(
        first_candidate_id=first_candidate_id,
        second_candidate_id=second_candidate_id,
        frame_indices=tuple(frame_indices),
        relative_displacement_diagonals=displacement,
        jointly_visible_frame_count=len(relative_positions),
        reliable=(
            len(relative_positions) >= min_jointly_visible_frames
            and displacement >= min_relative_displacement_diagonals
        ),
    )


def decide_component_hypothesis(
    candidate_ids: list[str],
    pair_evidence: list[ComponentMotionEvidence],
) -> dict:
    """Return multiple components only when every selected component is supported."""

    if not candidate_ids:
        raise ValueError("candidate_ids cannot be empty")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    if len(candidate_ids) == 1:
        return {
            "status": "single_object",
            "selected_candidate_ids": candidate_ids,
            "reason": "only_one_reliable_visible_instance",
            "pair_evidence": [],
        }
    reliable_pairs = {
        frozenset((evidence.first_candidate_id, evidence.second_candidate_id))
        for evidence in pair_evidence
        if evidence.reliable
    }
    unsupported = [
        candidate_id
        for candidate_id in candidate_ids
        if not any(candidate_id in pair for pair in reliable_pairs)
    ]
    if unsupported:
        return {
            "status": "single_object",
            "selected_candidate_ids": [candidate_ids[0]],
            "reason": "no_reliable_relative_motion_for_multiple_components",
            "unsupported_candidate_ids": unsupported,
            "pair_evidence": [evidence.to_dict() for evidence in pair_evidence],
        }
    return {
        "status": "multiple_components",
        "selected_candidate_ids": candidate_ids,
        "reason": "reliable_relative_motion_between_visible_instances",
        "pair_evidence": [evidence.to_dict() for evidence in pair_evidence],
    }
