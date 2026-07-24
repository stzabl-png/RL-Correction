"""Conservative class-free visual memory for persistent component IDs."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualInstanceDescriptor:
    """Appearance and shape evidence from one verified visible mask."""

    object_id: str
    hsv_histogram: np.ndarray
    log_area_fraction: float
    aspect_ratio: float


def describe_masked_instance(
    object_id: str,
    frame_bgr: np.ndarray,
    mask: np.ndarray,
) -> VisualInstanceDescriptor:
    """Describe a visible instance without object class, text, or relations."""

    image = np.asarray(frame_bgr)
    visible = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or visible.shape != image.shape[:2]:
        raise ValueError("frame and mask shapes are incompatible")
    area = int(np.count_nonzero(visible))
    if not area:
        raise ValueError("cannot describe an empty instance mask")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], visible.astype(np.uint8), [8, 4, 4], [0, 180, 0, 256, 0, 256]
    ).reshape(-1).astype(np.float64)
    histogram /= max(float(histogram.sum()), 1.0)
    ys, xs = np.nonzero(visible)
    aspect_ratio = float(xs.max() - xs.min() + 1) / float(ys.max() - ys.min() + 1)
    return VisualInstanceDescriptor(
        object_id=object_id,
        hsv_histogram=histogram,
        log_area_fraction=log(area / float(visible.size)),
        aspect_ratio=aspect_ratio,
    )


def descriptor_similarity(first: VisualInstanceDescriptor, second: VisualInstanceDescriptor) -> float:
    """Return a bounded, conservative visual similarity score."""

    appearance = float(np.minimum(first.hsv_histogram, second.hsv_histogram).sum())
    area = exp(-abs(first.log_area_fraction - second.log_area_fraction))
    aspect = exp(-abs(log(first.aspect_ratio / second.aspect_ratio)))
    return 0.7 * appearance + 0.2 * area + 0.1 * aspect


def decide_memory_identity(
    descriptors_by_object_id: dict[str, list[VisualInstanceDescriptor]],
    current: VisualInstanceDescriptor,
    *,
    min_reuse_similarity: float = 0.80,
    max_new_similarity: float = 0.45,
    min_reuse_margin: float = 0.08,
) -> dict:
    """Classify one current mask as known, new, or ambiguous.

    A new ID is allowed only when it is visibly distinct from every remembered
    instance. Similar-but-not-decisive observations remain failures instead of
    silently changing an existing component's identity.
    """

    if not 0.0 <= max_new_similarity < min_reuse_similarity <= 1.0:
        raise ValueError("invalid visual identity thresholds")
    if min_reuse_margin < 0.0:
        raise ValueError("min_reuse_margin must be non-negative")
    scores = {
        object_id: max(descriptor_similarity(previous, current) for previous in descriptors)
        for object_id, descriptors in descriptors_by_object_id.items()
        if descriptors
    }
    if not scores:
        return {"status": "new_visual_instance", "object_id": None, "scores": {}}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_object_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else None
    if best_score >= min_reuse_similarity and (
        second_score is None or best_score - second_score >= min_reuse_margin
    ):
        return {
            "status": "reuse_visual_memory",
            "object_id": best_object_id,
            "score": best_score,
            "second_score": second_score,
            "scores": scores,
        }
    if best_score <= max_new_similarity:
        return {
            "status": "new_visual_instance",
            "object_id": None,
            "score": best_score,
            "scores": scores,
        }
    return {
        "status": "failed_ambiguous_visual_memory",
        "object_id": None,
        "score": best_score,
        "second_score": second_score,
        "scores": scores,
    }


def add_memory_observation(
    memory: dict[str, list[VisualInstanceDescriptor]],
    descriptor: VisualInstanceDescriptor,
) -> None:
    """Append one verified observation to its immutable global-ID memory."""

    memory.setdefault(descriptor.object_id, []).append(descriptor)
