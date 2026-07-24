"""Temporal quality gates for propagated per-object mask sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from statistics import median
from typing import Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class TemporalMaskGateConfig:
    min_area_pixels: int = 256
    min_largest_component_fraction: float = 0.9
    max_frame_area_fraction: float = 0.35
    max_border_area_fraction: float = 0.15
    min_area_ratio_vs_history: float = 0.2
    max_area_ratio_vs_history: float = 4.0
    max_centroid_step_diagonals: float = 2.5
    history_size: int = 7

    def validate(self) -> None:
        if self.min_area_pixels <= 0:
            raise ValueError("min_area_pixels must be positive")
        if not 0 < self.min_largest_component_fraction <= 1:
            raise ValueError("min_largest_component_fraction must be in (0, 1]")
        if not 0 < self.max_frame_area_fraction <= 1:
            raise ValueError("max_frame_area_fraction must be in (0, 1]")
        if not 0 < self.max_border_area_fraction <= self.max_frame_area_fraction:
            raise ValueError(
                "max_border_area_fraction must be in (0, max_frame_area_fraction]"
            )
        if not 0 < self.min_area_ratio_vs_history <= 1:
            raise ValueError("min_area_ratio_vs_history must be in (0, 1]")
        if self.max_area_ratio_vs_history < 1:
            raise ValueError("max_area_ratio_vs_history must be at least 1")
        if self.max_centroid_step_diagonals <= 0:
            raise ValueError("max_centroid_step_diagonals must be positive")
        if self.history_size < 1:
            raise ValueError("history_size must be positive")


@dataclass(frozen=True)
class TemporalMaskMetrics:
    frame_idx: int
    object_id: str
    area_pixels: int
    area_fraction: float
    largest_component_fraction: float
    component_count: int
    touches_image_border: bool
    centroid_xy: tuple[float, float] | None
    bbox_diagonal_pixels: float | None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.centroid_xy is not None:
            payload["centroid_xy"] = list(self.centroid_xy)
        return payload


def measure_temporal_mask(
    frame_idx: int,
    object_id: str,
    mask: np.ndarray,
    *,
    frame_shape: tuple[int, int],
) -> TemporalMaskMetrics:
    candidate = np.asarray(mask, dtype=bool)
    if candidate.shape != frame_shape:
        raise ValueError(f"mask shape mismatch: {candidate.shape} != {frame_shape}")
    height, width = frame_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid frame shape: {frame_shape}")
    area = int(np.count_nonzero(candidate))
    if not area:
        return TemporalMaskMetrics(
            frame_idx,
            object_id,
            0,
            0.0,
            0.0,
            0,
            False,
            None,
            None,
        )
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8),
        8,
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = int(np.argmax(component_areas)) + 1
    x = int(stats[largest_index, cv2.CC_STAT_LEFT])
    y = int(stats[largest_index, cv2.CC_STAT_TOP])
    bbox_width = int(stats[largest_index, cv2.CC_STAT_WIDTH])
    bbox_height = int(stats[largest_index, cv2.CC_STAT_HEIGHT])
    centroid = centroids[largest_index]
    return TemporalMaskMetrics(
        frame_idx=frame_idx,
        object_id=object_id,
        area_pixels=area,
        area_fraction=area / float(height * width),
        largest_component_fraction=float(component_areas.max() / area),
        component_count=int(component_count - 1),
        touches_image_border=bool(
            candidate[0].any()
            or candidate[-1].any()
            or candidate[:, 0].any()
            or candidate[:, -1].any()
        ),
        centroid_xy=(float(centroid[0]), float(centroid[1])),
        bbox_diagonal_pixels=hypot(bbox_width, bbox_height),
    )


def _gate_direction(
    ordered_frames: list[int],
    metrics_by_frame: Mapping[int, TemporalMaskMetrics],
    *,
    config: TemporalMaskGateConfig,
    initial_history: list[TemporalMaskMetrics],
) -> dict[int, dict]:
    history = list(initial_history)
    decisions = {}
    for frame_idx in ordered_frames:
        metric = metrics_by_frame[frame_idx]
        reasons = []
        if metric.area_pixels < config.min_area_pixels:
            reasons.append("area_too_small")
        if metric.largest_component_fraction < config.min_largest_component_fraction:
            reasons.append("fragmented_mask")
        if metric.area_fraction > config.max_frame_area_fraction:
            reasons.append("covers_too_much_of_frame")
        if (
            metric.touches_image_border
            and metric.area_fraction > config.max_border_area_fraction
        ):
            reasons.append("large_border_connected_mask")
        if history and metric.area_pixels:
            recent = history[-config.history_size :]
            reference_area = float(median(item.area_pixels for item in recent))
            area_ratio = metric.area_pixels / reference_area
            if area_ratio < config.min_area_ratio_vs_history:
                reasons.append("area_collapse_vs_history")
            if area_ratio > config.max_area_ratio_vs_history:
                reasons.append("area_spike_vs_history")
            previous = history[-1]
            if metric.centroid_xy is not None and previous.centroid_xy is not None:
                frame_gap = max(1, abs(metric.frame_idx - previous.frame_idx))
                reference_diagonal = max(
                    1.0,
                    float(median(item.bbox_diagonal_pixels for item in recent)),
                )
                step = hypot(
                    metric.centroid_xy[0] - previous.centroid_xy[0],
                    metric.centroid_xy[1] - previous.centroid_xy[1],
                ) / reference_diagonal
                if step > config.max_centroid_step_diagonals * frame_gap:
                    reasons.append("centroid_jump_vs_history")
        accepted = not reasons
        decisions[frame_idx] = {
            "status": "accepted" if accepted else "rejected_temporal_outlier",
            "reasons": reasons,
            "metrics": metric.to_dict(),
        }
        if accepted:
            history.append(metric)
    return decisions


def gate_mask_sequence(
    masks_by_frame: Mapping[int, np.ndarray],
    *,
    object_id: str,
    seed_frame: int,
    frame_shape: tuple[int, int],
    config: TemporalMaskGateConfig = TemporalMaskGateConfig(),
) -> dict[int, dict]:
    """Gate outward from the trusted seed so one bad frame cannot poison history."""

    config.validate()
    if seed_frame not in masks_by_frame:
        raise ValueError(f"seed frame {seed_frame} is absent from mask sequence")
    frame_indices = sorted(masks_by_frame)
    metrics = {
        frame_idx: measure_temporal_mask(
            frame_idx,
            object_id,
            masks_by_frame[frame_idx],
            frame_shape=frame_shape,
        )
        for frame_idx in frame_indices
    }
    seed_decision = _gate_direction(
        [seed_frame],
        metrics,
        config=config,
        initial_history=[],
    )[seed_frame]
    decisions = {seed_frame: seed_decision}
    seed_history = [metrics[seed_frame]] if seed_decision["status"] == "accepted" else []
    decisions.update(
        _gate_direction(
            [frame for frame in frame_indices if frame > seed_frame],
            metrics,
            config=config,
            initial_history=seed_history,
        )
    )
    decisions.update(
        _gate_direction(
            [frame for frame in reversed(frame_indices) if frame < seed_frame],
            metrics,
            config=config,
            initial_history=seed_history,
        )
    )
    return {frame_idx: decisions[frame_idx] for frame_idx in frame_indices}
