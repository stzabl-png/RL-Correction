"""Track hand-linked HOI-DETR boxes and confirm safe expansion events.

An HOI box describes the hand's current interaction envelope, never an object
ID.  This module detects a sustained, non-background expansion of that
envelope, which may justify looking for a *new* visible component with SAM2.
"""

from __future__ import annotations

from dataclasses import dataclass


Box = tuple[float, float, float, float]


def _box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(first: Box, second: Box) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def _coverage_of_smaller(first: Box, second: Box) -> float:
    denominator = min(_box_area(first), _box_area(second))
    return _intersection_area(first, second) / denominator if denominator else 0.0


def _iou(first: Box, second: Box) -> float:
    intersection = _intersection_area(first, second)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union else 0.0


def hand_linked_object_boxes(frame: dict) -> list[Box]:
    """Return unique non-hand endpoints of HF links, without class semantics."""

    detections = {
        str(item.get("detection_id")): item for item in frame.get("detections", [])
    }
    result: list[Box] = []
    seen_ids = set()
    for link in frame.get("links", {}).get("hf", []):
        source = detections.get(str(link.get("source_detection_id")))
        target = detections.get(str(link.get("target_detection_id")))
        if source is None or target is None:
            continue
        if source.get("class_name") == "hand":
            visual_object = target
        elif target.get("class_name") == "hand":
            visual_object = source
        else:
            continue
        detection_id = str(visual_object.get("detection_id"))
        if detection_id in seen_ids:
            continue
        seen_ids.add(detection_id)
        values = visual_object.get("box_xyxy")
        if not isinstance(values, list) or len(values) != 4:
            continue
        box = tuple(float(value) for value in values)
        if _box_area(box) > 0.0:
            result.append(box)
    return result


def _background_like(box: Box, *, frame_shape: tuple[int, int], max_area_fraction: float) -> bool:
    height, width = frame_shape
    frame_area = float(height * width)
    if not frame_area:
        return True
    return _box_area(box) / frame_area > max_area_fraction


@dataclass(frozen=True)
class HOIBoxExpansionConfig:
    """Conservative thresholds for temporal HOI interaction-envelope changes."""

    min_area_growth: float = 1.6
    min_reference_coverage: float = 0.7
    min_consecutive_frames: int = 3
    min_continuity_coverage: float = 0.45
    max_box_area_fraction: float = 0.35


def detect_confirmed_box_expansions(
    detections: dict,
    *,
    interaction_start_frame: int,
    interaction_end_frame: int,
    initial_box: Box,
    config: HOIBoxExpansionConfig = HOIBoxExpansionConfig(),
) -> dict:
    """Find sustained safe expansions of the HF box originating at ``initial_box``.

    A frame-level expansion is only an event candidate.  It becomes actionable
    after it persists for ``min_consecutive_frames`` and does not resemble a
    scene-sized detection.  Rejected observations remain in the audit trail.
    """

    if interaction_start_frame < 0 or interaction_end_frame < interaction_start_frame:
        raise ValueError("invalid interaction window")
    if config.min_area_growth <= 1.0:
        raise ValueError("min_area_growth must exceed one")
    if not 0.0 <= config.min_reference_coverage <= 1.0:
        raise ValueError("invalid min_reference_coverage")
    if config.min_consecutive_frames < 1:
        raise ValueError("min_consecutive_frames must be positive")
    if not 0.0 <= config.min_continuity_coverage <= 1.0:
        raise ValueError("invalid min_continuity_coverage")
    if not 0.0 < config.max_box_area_fraction <= 1.0:
        raise ValueError("invalid max_box_area_fraction")

    frame_shape = (int(detections["video"]["height"]), int(detections["video"]["width"]))
    frames = detections["frames"]
    if interaction_end_frame >= len(frames):
        raise ValueError("interaction window lies outside detection frames")

    stable_box = initial_box
    previous_box = initial_box
    pending: list[tuple[int, Box, float]] = []
    events = []
    audit = []
    for frame_idx in range(interaction_start_frame, interaction_end_frame + 1):
        choices = hand_linked_object_boxes(frames[frame_idx])
        if not choices:
            audit.append({"frame_idx": frame_idx, "status": "no_hand_linked_box"})
            pending.clear()
            continue
        current = max(
            choices,
            key=lambda item: (
                _coverage_of_smaller(previous_box, item),
                _iou(previous_box, item),
                -_box_area(item),
            ),
        )
        continuity = _coverage_of_smaller(previous_box, current)
        if continuity < config.min_continuity_coverage:
            audit.append(
                {
                    "frame_idx": frame_idx,
                    "status": "box_handoff_or_discontinuity",
                    "box_xyxy": list(current),
                    "continuity_coverage": continuity,
                }
            )
            pending.clear()
            previous_box = current
            stable_box = current
            continue
        previous_box = current
        stable_area = _box_area(stable_box)
        area_ratio = _box_area(current) / stable_area if stable_area else 0.0
        reference_coverage = _coverage_of_smaller(stable_box, current)
        if area_ratio < config.min_area_growth:
            audit.append(
                {
                    "frame_idx": frame_idx,
                    "status": "stable_or_small_box_change",
                    "box_xyxy": list(current),
                    "area_ratio_vs_stable": area_ratio,
                }
            )
            pending.clear()
            stable_box = current
            continue
        if reference_coverage < config.min_reference_coverage:
            audit.append(
                {
                    "frame_idx": frame_idx,
                    "status": "rejected_noncontaining_expansion",
                    "box_xyxy": list(current),
                    "area_ratio_vs_stable": area_ratio,
                    "stable_box_coverage": reference_coverage,
                }
            )
            pending.clear()
            continue
        if _background_like(
            current,
            frame_shape=frame_shape,
            max_area_fraction=config.max_box_area_fraction,
        ):
            audit.append(
                {
                    "frame_idx": frame_idx,
                    "status": "rejected_background_sized_expansion",
                    "box_xyxy": list(current),
                    "area_ratio_vs_stable": area_ratio,
                }
            )
            pending.clear()
            continue
        if pending and _coverage_of_smaller(pending[-1][1], current) < config.min_continuity_coverage:
            pending.clear()
        pending.append((frame_idx, current, area_ratio))
        audit.append(
            {
                "frame_idx": frame_idx,
                "status": "pending_expansion_confirmation",
                "box_xyxy": list(current),
                "area_ratio_vs_stable": area_ratio,
                "confirmation_count": len(pending),
            }
        )
        if len(pending) < config.min_consecutive_frames:
            continue
        onset_frame, onset_box, onset_ratio = pending[0]
        events.append(
            {
                "event_type": "confirmed_hand_linked_box_expansion",
                "activation_frame": onset_frame,
                "seed_frame": frame_idx,
                "box_xyxy": list(current),
                "initial_expansion_box_xyxy": list(onset_box),
                "area_ratio_vs_previous_stable_box": onset_ratio,
                "confirmation_frame_count": len(pending),
            }
        )
        stable_box = current
        pending.clear()
    return {
        "schema_version": "hoi_box_expansion_events_v1",
        "interaction_window": [interaction_start_frame, interaction_end_frame],
        "initial_box_xyxy": list(initial_box),
        "events": events,
        "audit": audit,
    }
