"""Choose a conservative single-object or multi-component seed hypothesis."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np

from .component_decision import ComponentMotionEvidence, decide_component_hypothesis
from .instance_association import InstanceCandidate


def _overlap_fraction(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = min(np.count_nonzero(first), np.count_nonzero(second))
    return float(np.count_nonzero(first & second) / denominator) if denominator else 0.0


def _hypothesis_score(candidates: tuple[InstanceCandidate, ...], roi: np.ndarray) -> float:
    union = np.zeros_like(roi, dtype=bool)
    for candidate in candidates:
        union |= np.asarray(candidate.mask, dtype=bool)
    roi_area = int(np.count_nonzero(roi))
    roi_coverage = int(np.count_nonzero(union & roi)) / roi_area if roi_area else 0.0
    quality = sum(candidate.quality_score for candidate in candidates) / len(candidates)
    return 0.7 * roi_coverage + 0.3 * quality


def select_initial_instance_hypothesis(
    candidates: Iterable[InstanceCandidate],
    *,
    interaction_roi: np.ndarray,
    pair_evidence: Iterable[ComponentMotionEvidence],
    min_candidate_roi_fraction: float = 0.15,
    max_candidate_overlap_fraction: float = 0.05,
    max_components: int = 4,
    min_multi_component_score_gain: float = 0.05,
) -> dict:
    """Prefer one object unless disjoint components have stronger visual evidence."""

    if not 0.0 <= min_candidate_roi_fraction <= 1.0:
        raise ValueError("min_candidate_roi_fraction must be in [0, 1]")
    if not 0.0 <= max_candidate_overlap_fraction <= 1.0:
        raise ValueError("max_candidate_overlap_fraction must be in [0, 1]")
    if max_components < 1:
        raise ValueError("max_components must be positive")
    if min_multi_component_score_gain < 0.0:
        raise ValueError("min_multi_component_score_gain must be non-negative")
    roi = np.asarray(interaction_roi, dtype=bool)
    if roi.ndim != 2 or not np.any(roi):
        raise ValueError("interaction_roi must be a non-empty 2D mask")
    items = list(candidates)
    if not items:
        return {"status": "failed_no_instance_candidates", "selected_candidate_ids": []}
    relevant = []
    for candidate in items:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != roi.shape:
            raise ValueError("candidate mask shape differs from interaction ROI")
        candidate_area = int(np.count_nonzero(mask))
        roi_fraction = int(np.count_nonzero(mask & roi)) / candidate_area if candidate_area else 0.0
        if candidate_area and roi_fraction >= min_candidate_roi_fraction:
            relevant.append(candidate)
    if not relevant:
        return {"status": "failed_no_candidates_in_interaction_roi", "selected_candidate_ids": []}
    relevant.sort(
        key=lambda candidate: (_hypothesis_score((candidate,), roi), candidate.candidate_id),
        reverse=True,
    )
    best_single = relevant[0]
    best_single_score = _hypothesis_score((best_single,), roi)
    evidence = list(pair_evidence)
    best_multi = None
    for count in range(2, min(max_components, len(relevant)) + 1):
        for subset in itertools.combinations(relevant, count):
            if any(
                _overlap_fraction(first.mask, second.mask) > max_candidate_overlap_fraction
                for first, second in itertools.combinations(subset, 2)
            ):
                continue
            decision = decide_component_hypothesis(
                [candidate.candidate_id for candidate in subset], evidence
            )
            if decision["status"] != "multiple_components":
                continue
            score = _hypothesis_score(subset, roi)
            if best_multi is None or score > best_multi[0]:
                best_multi = (score, subset, decision)
    if best_multi is None or best_multi[0] < best_single_score + min_multi_component_score_gain:
        return {
            "status": "single_object",
            "selected_candidate_ids": [best_single.candidate_id],
            "reason": "no_stronger_reliable_multi_component_hypothesis",
            "score": best_single_score,
            "single_candidate_id": best_single.candidate_id,
        }
    return {
        **best_multi[2],
        "score": best_multi[0],
        "single_object_baseline_score": best_single_score,
        "single_candidate_id": best_single.candidate_id,
    }
