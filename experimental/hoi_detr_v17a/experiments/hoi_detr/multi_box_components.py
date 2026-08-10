"""Class-free component discovery from all credible hand-linked box masks.

Each detector box is only a visual proposal.  A SAM mask becomes a new
component only when existing registered masks cannot explain it and the same
unexplained region persists across multiple frames.  This supports both
simultaneous nested boxes and a direct small-to-large box transition without
encoding any object category or assembly relation.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .component_decision import component_motion_evidence


def accepted_hand_linked_candidates(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all per-candidate accepted hand-linked observations."""

    return [
        item
        for item in frame.get("hand_linked_object_candidates", [])
        if item.get("status") == "accepted"
        and isinstance(item.get("box_xyxy"), list)
        and len(item["box_xyxy"]) == 4
    ]


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
    area = int(np.count_nonzero(mask))
    if not area:
        return np.zeros_like(mask, dtype=bool), 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool), 0.0
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = labels == label
    return largest, float(np.count_nonzero(largest) / area)


def refine_known_masks_from_candidates(
    existing_masks: dict[str, np.ndarray],
    candidate_masks: list[dict[str, Any]],
    *,
    area_references: dict[str, int],
    min_candidate_overlap_fraction: float = 0.55,
    min_existing_overlap_fraction: float = 0.20,
    min_area_scale: float = 0.45,
    max_area_scale: float = 1.80,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Use a well-sized box mask to re-anchor a known ID before subtraction.

    This is intentionally category-free.  A candidate may refine a known ID
    only when it overlaps that ID, remains close to the ID's fixed reference
    area, and wins a one-to-one assignment.  A large composite proposal cannot
    therefore replace a previously small component merely because it contains
    it.
    """

    if set(existing_masks) != set(area_references):
        raise ValueError("area references must cover every existing object ID")
    refined = {
        object_id: np.asarray(mask, dtype=bool).copy()
        for object_id, mask in existing_masks.items()
    }
    assignments: dict[str, dict[str, Any]] = {}
    used_candidates = set()
    ranked = []
    for object_id, existing in refined.items():
        existing_area = int(np.count_nonzero(existing))
        reference_area = int(area_references[object_id])
        if existing_area < 1 or reference_area < 1:
            raise ValueError("existing masks and area references must be non-empty")
        for candidate_index, candidate in enumerate(candidate_masks):
            mask = np.asarray(candidate["mask"], dtype=bool)
            if mask.shape != existing.shape:
                raise ValueError("candidate and existing masks must share one shape")
            candidate_area = int(np.count_nonzero(mask))
            if not candidate_area:
                continue
            overlap = int(np.count_nonzero(mask & existing))
            candidate_overlap = overlap / candidate_area
            existing_overlap = overlap / existing_area
            area_scale = candidate_area / reference_area
            if (
                candidate_overlap < min_candidate_overlap_fraction
                or existing_overlap < min_existing_overlap_fraction
                or not min_area_scale <= area_scale <= max_area_scale
            ):
                continue
            area_closeness = math.exp(-abs(math.log(area_scale)))
            score = (
                0.55 * candidate_overlap
                + 0.20 * existing_overlap
                + 0.20 * area_closeness
                + 0.05 * float(candidate.get("quality_score", 1.0))
            )
            ranked.append(
                (
                    score,
                    object_id,
                    candidate_index,
                    candidate_overlap,
                    existing_overlap,
                    area_scale,
                )
            )
    for score, object_id, candidate_index, candidate_overlap, existing_overlap, area_scale in sorted(
        ranked, reverse=True
    ):
        if object_id in assignments or candidate_index in used_candidates:
            continue
        candidate = candidate_masks[candidate_index]
        refined[object_id] = np.asarray(candidate["mask"], dtype=bool).copy()
        used_candidates.add(candidate_index)
        assignments[object_id] = {
            "candidate_id": str(candidate["candidate_id"]),
            "score": float(score),
            "candidate_overlap_fraction": float(candidate_overlap),
            "existing_overlap_fraction": float(existing_overlap),
            "area_scale_vs_reference": float(area_scale),
        }
    return refined, assignments


def classify_candidate_masks(
    candidate_masks: list[dict[str, Any]],
    *,
    existing_masks: dict[str, np.ndarray],
    min_area_pixels: int,
    min_candidate_explained_fraction: float = 0.85,
    min_existing_coverage_for_residual: float = 0.75,
    max_direct_overlap_fraction: float = 0.05,
    min_largest_component_fraction: float = 0.85,
    max_frame_area_fraction: float = 0.35,
) -> list[dict[str, Any]]:
    """Classify each SAM box mask as explained, residual, direct, or ambiguous."""

    if not existing_masks:
        raise ValueError("at least one existing mask is required")
    shape = np.asarray(next(iter(existing_masks.values())), dtype=bool).shape
    known_union = np.zeros(shape, dtype=bool)
    for mask in existing_masks.values():
        array = np.asarray(mask, dtype=bool)
        if array.shape != shape:
            raise ValueError("existing masks must share one shape")
        known_union |= array
    known_area = int(np.count_nonzero(known_union))
    if not known_area:
        raise ValueError("existing masks must be non-empty")
    frame_area = int(shape[0] * shape[1])
    results = []
    for candidate in candidate_masks:
        mask = np.asarray(candidate["mask"], dtype=bool)
        if mask.shape != shape:
            raise ValueError("candidate and existing masks must share one shape")
        area = int(np.count_nonzero(mask))
        overlap = int(np.count_nonzero(mask & known_union))
        explained_fraction = overlap / area if area else 0.0
        existing_coverage = overlap / known_area
        base = {
            key: value for key, value in candidate.items() if key != "mask"
        }
        base.update(
            {
                "candidate_area": area,
                "candidate_explained_fraction": explained_fraction,
                "existing_union_coverage": existing_coverage,
            }
        )
        if (
            area < min_area_pixels
            or area / frame_area > max_frame_area_fraction
            or mask[0].any()
            or mask[-1].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
        ):
            results.append({**base, "status": "rejected_geometry"})
            continue
        if explained_fraction >= min_candidate_explained_fraction:
            results.append({**base, "status": "explained_by_existing_ids"})
            continue
        if existing_coverage >= min_existing_coverage_for_residual:
            proposal = mask & ~known_union
            proposal_type = "composite_residual"
        elif explained_fraction <= max_direct_overlap_fraction:
            proposal = mask.copy()
            proposal_type = "direct_disjoint_candidate"
        else:
            results.append({**base, "status": "ambiguous_partial_overlap"})
            continue
        largest, largest_fraction = _largest_component(proposal)
        proposal_area = int(np.count_nonzero(largest))
        if proposal_area < min_area_pixels or largest_fraction < min_largest_component_fraction:
            results.append(
                {
                    **base,
                    "status": "rejected_unstable_residual_geometry",
                    "proposal_type": proposal_type,
                    "proposal_area": proposal_area,
                    "largest_component_fraction": largest_fraction,
                }
            )
            continue
        quality = float(candidate.get("quality_score", 1.0))
        results.append(
            {
                **base,
                "status": "new_component_proposal",
                "proposal_type": proposal_type,
                "proposal_area": proposal_area,
                "largest_component_fraction": largest_fraction,
                "proposal_score": quality * largest_fraction,
                "mask": largest,
            }
        )
    return results


def _mask_continuity(first: np.ndarray, second: np.ndarray) -> float:
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    denominator = min(first_area, second_area)
    return (
        int(np.count_nonzero(np.asarray(first, dtype=bool) & np.asarray(second, dtype=bool)))
        / denominator
        if denominator
        else 0.0
    )


def confirm_new_component_track(
    frame_proposals: list[dict[str, Any]],
    *,
    min_consecutive_frames: int = 3,
    max_frame_gap: int = 1,
    min_mask_continuity: float = 0.35,
    max_area_scale: float = 2.5,
) -> dict[str, Any]:
    """Confirm the earliest temporally persistent unexplained component."""

    if min_consecutive_frames < 1 or max_frame_gap < 0:
        raise ValueError("invalid temporal confirmation thresholds")
    best_by_frame: dict[int, dict[str, Any]] = {}
    for proposal in frame_proposals:
        if proposal.get("status") != "new_component_proposal":
            continue
        frame_idx = int(proposal["frame_idx"])
        current = best_by_frame.get(frame_idx)
        if current is None or float(proposal["proposal_score"]) > float(
            current["proposal_score"]
        ):
            best_by_frame[frame_idx] = proposal
    tracks: list[list[dict[str, Any]]] = []
    active: list[dict[str, Any]] = []
    for frame_idx in sorted(best_by_frame):
        proposal = best_by_frame[frame_idx]
        if active:
            previous = active[-1]
            area_scale = max(
                int(proposal["proposal_area"]) / int(previous["proposal_area"]),
                int(previous["proposal_area"]) / int(proposal["proposal_area"]),
            )
            continuous = (
                frame_idx - int(previous["frame_idx"]) <= max_frame_gap + 1
                and _mask_continuity(previous["mask"], proposal["mask"])
                >= min_mask_continuity
                and area_scale <= max_area_scale
            )
            if not continuous:
                tracks.append(active)
                active = []
        active.append(proposal)
    if active:
        tracks.append(active)
    confirmed = [track for track in tracks if len(track) >= min_consecutive_frames]
    if not confirmed:
        return {
            "status": "failed_no_temporally_confirmed_new_component",
            "track_lengths": [len(track) for track in tracks],
        }
    track = min(
        confirmed,
        key=lambda items: (
            int(items[0]["frame_idx"]),
            -len(items),
            -max(float(item["proposal_score"]) for item in items),
        ),
    )
    seed = max(
        track,
        key=lambda item: (
            float(item["proposal_score"]),
            float(item.get("quality_score", 1.0)),
            -int(item["frame_idx"]),
        ),
    )
    return {
        "status": "success",
        "activation_frame": int(track[0]["frame_idx"]),
        "seed_frame": int(seed["frame_idx"]),
        "track_frames": [int(item["frame_idx"]) for item in track],
        "selected": seed,
    }


def validate_composite_residual_motion(
    confirmation: dict[str, Any],
    *,
    frame_proposals: list[dict[str, Any]],
    known_masks_by_frame: dict[int, dict[str, np.ndarray]],
    interaction_envelopes_by_candidate_id: dict[str, np.ndarray],
    min_existing_coverage_for_residual: float,
    min_jointly_visible_frames: int,
    min_relative_displacement_diagonals: float,
) -> dict[str, Any]:
    """Require independent motion before registering a composite residual.

    A residual cut from a mask that substantially contains an existing object
    is weak evidence by itself.  It is accepted only when it moves
    independently of every substantially covered registered instance.
    """

    selected = confirmation.get("selected", {})
    if selected.get("proposal_type") != "composite_residual":
        return {"status": "not_required_for_direct_disjoint_candidate"}

    track_frames = [int(frame_idx) for frame_idx in confirmation.get("track_frames", [])]
    best_by_frame: dict[int, dict[str, Any]] = {}
    for proposal in frame_proposals:
        frame_idx = int(proposal.get("frame_idx", -1))
        if (
            frame_idx not in track_frames
            or proposal.get("status") != "new_component_proposal"
            or proposal.get("proposal_type") != "composite_residual"
        ):
            continue
        current = best_by_frame.get(frame_idx)
        if current is None or float(proposal["proposal_score"]) > float(
            current["proposal_score"]
        ):
            best_by_frame[frame_idx] = proposal
    if set(best_by_frame) != set(track_frames):
        return {"status": "failed_missing_composite_residual_track_masks"}

    seed_frame = int(confirmation["seed_frame"])
    seed_proposal = best_by_frame[seed_frame]
    seed_envelope = interaction_envelopes_by_candidate_id.get(
        str(seed_proposal["candidate_id"])
    )
    seed_known = known_masks_by_frame.get(seed_frame, {})
    if seed_envelope is None or not seed_known:
        return {"status": "failed_missing_composite_overlap_evidence"}

    related_known_ids = []
    for object_id, known_mask in seed_known.items():
        known = np.asarray(known_mask, dtype=bool)
        known_area = int(np.count_nonzero(known))
        coverage = (
            float(np.count_nonzero(np.asarray(seed_envelope, dtype=bool) & known) / known_area)
            if known_area
            else 0.0
        )
        if coverage >= min_existing_coverage_for_residual:
            related_known_ids.append(str(object_id))
    if not related_known_ids:
        return {
            "status": "failed_no_substantially_covered_known_instance",
            "related_known_ids": [],
        }

    tracked_masks: dict[int, dict[str, np.ndarray]] = {}
    for frame_idx in track_frames:
        known = known_masks_by_frame.get(frame_idx, {})
        if any(object_id not in known for object_id in related_known_ids):
            continue
        tracked_masks[frame_idx] = {
            "__new_residual__": np.asarray(best_by_frame[frame_idx]["mask"], dtype=bool),
            **{
                object_id: np.asarray(known[object_id], dtype=bool)
                for object_id in related_known_ids
            },
        }
    frame_shape = np.asarray(seed_proposal["mask"], dtype=bool).shape
    evidence = [
        component_motion_evidence(
            object_id,
            "__new_residual__",
            tracked_masks,
            frame_shape=frame_shape,
            min_jointly_visible_frames=min_jointly_visible_frames,
            min_relative_displacement_diagonals=min_relative_displacement_diagonals,
        )
        for object_id in related_known_ids
    ]
    reliable = bool(evidence) and all(item.reliable for item in evidence)
    return {
        "status": (
            "success_independent_composite_residual_motion"
            if reliable
            else "failed_no_independent_composite_residual_motion"
        ),
        "related_known_ids": related_known_ids,
        "pair_evidence": [item.to_dict() for item in evidence],
    }


def select_component_anchor_proposals(
    frame_proposals: list[dict[str, Any]],
    *,
    track_frames: list[int],
    seed_frame: int,
    max_gap_frames: int = 8,
) -> list[dict[str, Any]]:
    """Select quality anchors while bounding the gap across a confirmed track."""

    if max_gap_frames < 1 or not track_frames:
        raise ValueError("anchor selection requires a non-empty track and positive gap")
    best_by_frame = {}
    allowed = set(int(frame_idx) for frame_idx in track_frames)
    for proposal in frame_proposals:
        frame_idx = int(proposal["frame_idx"])
        if frame_idx not in allowed or proposal.get("status") != "new_component_proposal":
            continue
        current = best_by_frame.get(frame_idx)
        if current is None or float(proposal["proposal_score"]) > float(
            current["proposal_score"]
        ):
            best_by_frame[frame_idx] = proposal
    frames = sorted(best_by_frame)
    if frames != sorted(allowed):
        raise ValueError("confirmed track frames lack proposals")
    anchor_frames = [frames[0]]
    current_frame = frames[0]
    while frames[-1] - current_frame > max_gap_frames:
        window = [
            frame_idx
            for frame_idx in frames
            if current_frame < frame_idx <= current_frame + max_gap_frames
        ]
        if not window:
            raise ValueError("confirmed track contains an anchor gap")
        chosen = max(
            window,
            key=lambda frame_idx: (
                float(best_by_frame[frame_idx]["proposal_score"]),
                frame_idx,
            ),
        )
        anchor_frames.append(chosen)
        current_frame = chosen
    if frames[-1] not in anchor_frames:
        anchor_frames.append(frames[-1])
    if seed_frame not in best_by_frame:
        raise ValueError("seed frame is not part of the confirmed proposal track")
    if seed_frame not in anchor_frames:
        anchor_frames.append(seed_frame)
    return [best_by_frame[frame_idx] for frame_idx in sorted(set(anchor_frames))]
