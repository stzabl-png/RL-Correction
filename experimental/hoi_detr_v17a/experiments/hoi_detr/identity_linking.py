"""Conservative visual matching for reusing global instance IDs across episodes."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot, log
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeInstanceObservation:
    local_object_id: str
    mask: np.ndarray


@dataclass(frozen=True)
class GlobalInstanceObservation:
    global_object_id: str
    mask: np.ndarray


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not xs.size:
        return None
    return float(xs.mean()), float(ys.mean())


def _pair_metrics(previous: np.ndarray, current: np.ndarray) -> dict[str, float | int]:
    previous = np.asarray(previous, dtype=bool)
    current = np.asarray(current, dtype=bool)
    if previous.ndim != 2 or current.shape != previous.shape:
        raise ValueError("identity masks must be 2D and share one shape")
    previous_area = int(np.count_nonzero(previous))
    current_area = int(np.count_nonzero(current))
    intersection = int(np.count_nonzero(previous & current))
    union = int(np.count_nonzero(previous | current))
    previous_centroid = _centroid(previous)
    current_centroid = _centroid(current)
    centroid_distance = (
        hypot(
            previous_centroid[0] - current_centroid[0],
            previous_centroid[1] - current_centroid[1],
        )
        / hypot(*previous.shape)
        if previous_centroid is not None and current_centroid is not None
        else float("inf")
    )
    area_ratio = current_area / previous_area if previous_area else 0.0
    iou = intersection / union if union else 0.0
    score = (
        iou
        * (exp(-abs(log(area_ratio))) if area_ratio else 0.0)
        * (exp(-centroid_distance / 0.1) if np.isfinite(centroid_distance) else 0.0)
    )
    return {
        "previous_area": previous_area,
        "current_area": current_area,
        "area_ratio": area_ratio,
        "iou": iou,
        "centroid_distance_diagonals": centroid_distance,
        "score": score,
    }


def link_episode_instances(
    previous_instances: Iterable[GlobalInstanceObservation],
    current_instances: Iterable[EpisodeInstanceObservation],
    *,
    min_iou: float = 0.1,
    max_centroid_distance_diagonals: float = 0.15,
    min_area_ratio: float = 0.5,
    max_area_ratio: float = 2.0,
    min_score_margin: float = 0.05,
) -> dict:
    """Match known global IDs to current local IDs, leaving true new IDs unmatched."""

    if min_score_margin < 0.0:
        raise ValueError("min_score_margin must be non-negative")
    previous = list(previous_instances)
    current = list(current_instances)
    if len({item.global_object_id for item in previous}) != len(previous):
        raise ValueError("previous global IDs must be unique")
    if len({item.local_object_id for item in current}) != len(current):
        raise ValueError("current local IDs must be unique")
    metrics = {
        prior.global_object_id: {
            item.local_object_id: _pair_metrics(prior.mask, item.mask)
            for item in current
        }
        for prior in previous
    }
    eligible = {
        global_id: [
            local_id
            for local_id, item in choices.items()
            if item["iou"] >= min_iou
            and item["centroid_distance_diagonals"] <= max_centroid_distance_diagonals
            and min_area_ratio <= item["area_ratio"] <= max_area_ratio
        ]
        for global_id, choices in metrics.items()
    }
    solutions: list[tuple[float, dict[str, str]]] = []
    previous_ids = [item.global_object_id for item in previous]

    def visit(index: int, assignment: dict[str, str], score: float) -> None:
        if index == len(previous_ids):
            solutions.append((score, assignment.copy()))
            return
        global_id = previous_ids[index]
        visit(index + 1, assignment, score)
        for local_id in eligible[global_id]:
            if local_id in assignment.values():
                continue
            assignment[global_id] = local_id
            visit(index + 1, assignment, score + float(metrics[global_id][local_id]["score"]))
            del assignment[global_id]

    visit(0, {}, 0.0)
    solutions.sort(key=lambda item: item[0], reverse=True)
    best_score, best = solutions[0]
    second_score = solutions[1][0] if len(solutions) > 1 else None
    if second_score is not None and best_score - second_score < min_score_margin and best:
        return {
            "status": "failed_ambiguous_visual_identity_match",
            "best_score": best_score,
            "second_best_score": second_score,
            "score_margin": best_score - second_score,
            "metrics": metrics,
        }
    local_to_global = {local_id: global_id for global_id, local_id in best.items()}
    return {
        "status": "success",
        "local_to_existing_global": local_to_global,
        "unmatched_local_object_ids": [
            item.local_object_id for item in current if item.local_object_id not in local_to_global
        ],
        "unmatched_global_object_ids": [
            item.global_object_id for item in previous if item.global_object_id not in best
        ],
        "score": best_score,
        "metrics": metrics,
    }
