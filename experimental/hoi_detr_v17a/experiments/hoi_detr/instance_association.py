"""Relation-free visual association for persistent component instance tracks."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot, log
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class InstanceCandidate:
    """One visible-instance proposal on a single frame."""

    candidate_id: str
    mask: np.ndarray
    quality_score: float = 1.0
    source: str = "sam2_amg"


@dataclass(frozen=True)
class InstanceTrack:
    """A fixed ID with a clean anchor and its current propagated prediction."""

    object_id: str
    anchor_mask: np.ndarray
    predicted_mask: np.ndarray


def _as_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask")
    return result


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _safe_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _validate_candidates(candidates: Iterable[InstanceCandidate]) -> list[InstanceCandidate]:
    items = list(candidates)
    if not items:
        raise ValueError("at least one instance candidate is required")
    candidate_ids = [item.candidate_id for item in items]
    if len(set(candidate_ids)) != len(candidate_ids) or any(not item for item in candidate_ids):
        raise ValueError("candidate IDs must be unique and non-empty")
    shape = _as_mask(items[0].mask, name=items[0].candidate_id).shape
    for item in items:
        if _as_mask(item.mask, name=item.candidate_id).shape != shape:
            raise ValueError("candidate masks must share one shape")
        if not 0.0 <= item.quality_score <= 1.0:
            raise ValueError("candidate quality scores must be in [0, 1]")
    return items


def derive_residual_instance_candidates(
    candidates: Iterable[InstanceCandidate],
    *,
    min_child_inside_parent: float = 0.9,
    min_residual_area: int = 256,
) -> list[InstanceCandidate]:
    """Derive disjoint residual proposals from nested visible-instance masks.

    This is geometric mask partitioning only.  It does not assign semantic
    labels or assume why the two visible instances touch.
    """

    if not 0.0 <= min_child_inside_parent <= 1.0:
        raise ValueError("min_child_inside_parent must be in [0, 1]")
    if min_residual_area < 1:
        raise ValueError("min_residual_area must be positive")
    originals = _validate_candidates(candidates)
    derived = []
    for parent in originals:
        parent_mask = _as_mask(parent.mask, name=parent.candidate_id)
        parent_area = int(np.count_nonzero(parent_mask))
        for child in originals:
            if parent.candidate_id == child.candidate_id:
                continue
            child_mask = _as_mask(child.mask, name=child.candidate_id)
            child_area = int(np.count_nonzero(child_mask))
            if child_area >= parent_area:
                continue
            contained_fraction = _safe_fraction(
                int(np.count_nonzero(parent_mask & child_mask)), child_area
            )
            if contained_fraction < min_child_inside_parent:
                continue
            residual = parent_mask & ~child_mask
            if int(np.count_nonzero(residual)) < min_residual_area:
                continue
            derived.append(
                InstanceCandidate(
                    candidate_id=(
                        f"residual__{parent.candidate_id}__without__{child.candidate_id}"
                    ),
                    mask=residual,
                    quality_score=min(parent.quality_score, child.quality_score),
                    source="geometric_residual_partition",
                )
            )
    return derived


def assess_candidate_for_track(
    track: InstanceTrack,
    candidate: InstanceCandidate,
    *,
    min_area_ratio_vs_anchor: float = 0.25,
    max_area_ratio_vs_anchor: float = 4.0,
    min_candidate_coverage_by_prediction: float = 0.5,
    max_centroid_distance_diagonals: float = 0.2,
) -> dict[str, float | int | str | bool | list[str]]:
    """Score one candidate using only per-instance visual continuity."""

    if not 0.0 < min_area_ratio_vs_anchor <= max_area_ratio_vs_anchor:
        raise ValueError("invalid anchor-area ratio range")
    if not 0.0 <= min_candidate_coverage_by_prediction <= 1.0:
        raise ValueError("min_candidate_coverage_by_prediction must be in [0, 1]")
    if max_centroid_distance_diagonals < 0.0:
        raise ValueError("max_centroid_distance_diagonals must be non-negative")
    anchor = _as_mask(track.anchor_mask, name=f"{track.object_id}.anchor")
    predicted = _as_mask(track.predicted_mask, name=f"{track.object_id}.predicted")
    proposed = _as_mask(candidate.mask, name=candidate.candidate_id)
    if anchor.shape != predicted.shape or anchor.shape != proposed.shape:
        raise ValueError("track and candidate masks must share one shape")
    anchor_area = int(np.count_nonzero(anchor))
    predicted_area = int(np.count_nonzero(predicted))
    candidate_area = int(np.count_nonzero(proposed))
    intersection = int(np.count_nonzero(predicted & proposed))
    area_ratio = _safe_fraction(candidate_area, anchor_area)
    candidate_coverage = _safe_fraction(intersection, candidate_area)
    prediction_centroid = _centroid(predicted)
    candidate_centroid = _centroid(proposed)
    diagonal = hypot(*anchor.shape)
    centroid_distance = (
        hypot(
            prediction_centroid[0] - candidate_centroid[0],
            prediction_centroid[1] - candidate_centroid[1],
        )
        / diagonal
        if prediction_centroid is not None and candidate_centroid is not None
        else float("inf")
    )
    reasons = []
    if candidate_area == 0:
        reasons.append("empty_candidate")
    if not min_area_ratio_vs_anchor <= area_ratio <= max_area_ratio_vs_anchor:
        reasons.append("area_inconsistent_with_anchor")
    if candidate_coverage < min_candidate_coverage_by_prediction:
        reasons.append("candidate_not_supported_by_track_prediction")
    if centroid_distance > max_centroid_distance_diagonals:
        reasons.append("candidate_too_far_from_track_prediction")
    area_score = exp(-abs(log(area_ratio))) if area_ratio > 0.0 else 0.0
    location_score = (
        exp(-centroid_distance / max(1e-6, max_centroid_distance_diagonals))
        if np.isfinite(centroid_distance)
        else 0.0
    )
    score = candidate.quality_score * area_score * candidate_coverage * location_score
    return {
        "object_id": track.object_id,
        "candidate_id": candidate.candidate_id,
        "anchor_area": anchor_area,
        "predicted_area": predicted_area,
        "candidate_area": candidate_area,
        "area_ratio_vs_anchor": area_ratio,
        "candidate_coverage_by_prediction": candidate_coverage,
        "centroid_distance_diagonals": centroid_distance,
        "score": score,
        "eligible": not reasons,
        "reasons": reasons,
    }


def _candidate_pair_overlap_fraction(
    first: InstanceCandidate,
    second: InstanceCandidate,
) -> float:
    first_mask = _as_mask(first.mask, name=first.candidate_id)
    second_mask = _as_mask(second.mask, name=second.candidate_id)
    overlap = int(np.count_nonzero(first_mask & second_mask))
    return _safe_fraction(overlap, min(np.count_nonzero(first_mask), np.count_nonzero(second_mask)))


def associate_tracks_to_candidates(
    tracks: Iterable[InstanceTrack],
    candidates: Iterable[InstanceCandidate],
    *,
    max_cross_instance_overlap_fraction: float = 0.05,
    min_assignment_margin: float = 0.05,
) -> dict:
    """Find one unambiguous, mutually exclusive candidate per fixed object ID."""

    if not 0.0 <= max_cross_instance_overlap_fraction <= 1.0:
        raise ValueError("max_cross_instance_overlap_fraction must be in [0, 1]")
    if min_assignment_margin < 0.0:
        raise ValueError("min_assignment_margin must be non-negative")
    track_items = list(tracks)
    if not track_items:
        raise ValueError("at least one instance track is required")
    if len({track.object_id for track in track_items}) != len(track_items):
        raise ValueError("track object IDs must be unique")
    candidate_items = _validate_candidates(candidates)
    assessments = {
        track.object_id: {
            candidate.candidate_id: assess_candidate_for_track(track, candidate)
            for candidate in candidate_items
        }
        for track in track_items
    }
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidate_items}
    eligible_by_track = {
        track.object_id: [
            candidate_id
            for candidate_id, assessment in assessments[track.object_id].items()
            if assessment["eligible"]
        ]
        for track in track_items
    }
    missing = [object_id for object_id, options in eligible_by_track.items() if not options]
    if missing:
        return {
            "status": "failed_no_candidate_for_track",
            "missing_object_ids": missing,
            "assessments": assessments,
        }

    solutions: list[tuple[float, dict[str, str]]] = []

    def visit(index: int, assignment: dict[str, str], score: float) -> None:
        if index == len(track_items):
            solutions.append((score, assignment.copy()))
            return
        track = track_items[index]
        for candidate_id in eligible_by_track[track.object_id]:
            if candidate_id in assignment.values():
                continue
            candidate = candidate_by_id[candidate_id]
            if any(
                _candidate_pair_overlap_fraction(candidate, candidate_by_id[other_id])
                > max_cross_instance_overlap_fraction
                for other_id in assignment.values()
            ):
                continue
            assignment[track.object_id] = candidate_id
            visit(
                index + 1,
                assignment,
                score + float(assessments[track.object_id][candidate_id]["score"]),
            )
            del assignment[track.object_id]

    visit(0, {}, 0.0)
    if not solutions:
        return {
            "status": "failed_no_mutually_exclusive_assignment",
            "assessments": assessments,
        }
    solutions.sort(key=lambda item: item[0], reverse=True)
    if len(solutions) > 1 and solutions[0][0] - solutions[1][0] < min_assignment_margin:
        return {
            "status": "failed_ambiguous_instance_assignment",
            "best_score": solutions[0][0],
            "second_best_score": solutions[1][0],
            "score_margin": solutions[0][0] - solutions[1][0],
            "assessments": assessments,
        }
    return {
        "status": "success",
        "assignment": solutions[0][1],
        "score": solutions[0][0],
        "assessments": assessments,
    }
