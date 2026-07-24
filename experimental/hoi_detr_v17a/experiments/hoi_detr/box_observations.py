"""Convert per-frame HOI-DETR candidates into stable object-box observations.

The detector output remains the immutable source record.  Every directly
hand-linked object box is retained for component discovery, while the legacy
single selected box remains available for interaction timing.  Obvious
screen-spanning and temporal area-spike failures are rejected per candidate so
one bad box cannot hide another valid box from the same frame.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .adapter import read_candidates_json, write_json_atomic

SCHEMA_VERSION = "hoi_detr_box_observations_v2"


@dataclass(frozen=True)
class BoxGateConfig:
    """Conservative thresholds for rejecting obvious background boxes."""

    history_size: int = 15
    min_history: int = 3
    history_spike_min_area_ratio: float = 0.10
    history_spike_area_scale: float = 8.0
    screen_spanning_min_area_ratio: float = 0.35
    screen_spanning_min_width_fraction: float = 0.90

    def validate(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if not 1 <= self.min_history <= self.history_size:
            raise ValueError("min_history must be between 1 and history_size")
        for name in (
            "history_spike_min_area_ratio",
            "screen_spanning_min_area_ratio",
            "screen_spanning_min_width_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.history_spike_area_scale <= 1.0:
            raise ValueError("history_spike_area_scale must be greater than 1")


def _box_metrics(box: Iterable[float], *, width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width = x2 - x1
    box_height = y2 - y1
    area = box_width * box_height
    return {
        "area_pixels": area,
        "area_ratio": area / float(width * height),
        "width_fraction": box_width / float(width),
        "height_fraction": box_height / float(height),
        "center_x": (x1 + x2) / 2.0,
        "center_y": (y1 + y2) / 2.0,
    }


def _select_firstobject(
    frame: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None, list[str], float | None]:
    detections = frame["detections"]
    firstobjects = [detection for detection in detections if detection["class_name"] == "firstobject"]
    if not firstobjects:
        return None, None, [], None

    by_id = {detection["detection_id"]: detection for detection in firstobjects}
    link_probability_by_target: dict[str, float] = {}
    linked_hand_ids_by_target: dict[str, list[str]] = {}
    for link in frame["links"]["hf"]:
        target_id = link["target_detection_id"]
        if target_id not in by_id:
            continue
        link_probability_by_target[target_id] = max(
            float(link["prob"]),
            link_probability_by_target.get(target_id, 0.0),
        )
        linked_hand_ids_by_target.setdefault(target_id, []).append(link["source_detection_id"])

    if link_probability_by_target:
        selected_id = max(
            link_probability_by_target,
            key=lambda detection_id: (
                link_probability_by_target[detection_id] * float(by_id[detection_id]["score"]),
                float(by_id[detection_id]["score"]),
                detection_id,
            ),
        )
        return (
            by_id[selected_id],
            "hf_link",
            sorted(set(linked_hand_ids_by_target[selected_id])),
            link_probability_by_target[selected_id],
        )

    selected = max(firstobjects, key=lambda detection: (float(detection["score"]), detection["detection_id"]))
    return selected, "unlinked_firstobject", [], None


def _hand_linked_firstobjects(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every unique first-object candidate linked to at least one hand."""

    detections = {
        str(detection["detection_id"]): detection
        for detection in frame.get("detections", [])
    }
    linked: dict[str, dict[str, Any]] = {}
    for link in frame.get("links", {}).get("hf", []):
        source = detections.get(str(link.get("source_detection_id")))
        target = detections.get(str(link.get("target_detection_id")))
        if source is None or target is None:
            continue
        if source.get("class_name") == "hand" and target.get("class_name") == "firstobject":
            hand, visual_object = source, target
        elif target.get("class_name") == "hand" and source.get("class_name") == "firstobject":
            hand, visual_object = target, source
        else:
            continue
        detection_id = str(visual_object["detection_id"])
        record = linked.setdefault(
            detection_id,
            {
                "detection": visual_object,
                "linked_hand_detection_ids": set(),
                "hand_link_probability": 0.0,
            },
        )
        record["linked_hand_detection_ids"].add(str(hand["detection_id"]))
        record["hand_link_probability"] = max(
            float(record["hand_link_probability"]),
            float(link["prob"]),
        )
    result = []
    for detection_id in sorted(linked):
        record = linked[detection_id]
        result.append(
            {
                "detection": record["detection"],
                "linked_hand_detection_ids": sorted(record["linked_hand_detection_ids"]),
                "hand_link_probability": float(record["hand_link_probability"]),
            }
        )
    return result


def _gate_metrics(
    metrics: dict[str, float],
    *,
    accepted_areas: deque[float],
    config: BoxGateConfig,
) -> dict[str, Any]:
    """Apply the same background gate independently to one box candidate."""

    history_median_area = statistics.median(accepted_areas) if accepted_areas else None
    area_scale = (
        metrics["area_pixels"] / history_median_area
        if history_median_area is not None and history_median_area > 0.0
        else None
    )
    screen_spanning = (
        metrics["area_ratio"] >= config.screen_spanning_min_area_ratio
        and metrics["width_fraction"] >= config.screen_spanning_min_width_fraction
    )
    history_spike = (
        len(accepted_areas) >= config.min_history
        and metrics["area_ratio"] >= config.history_spike_min_area_ratio
        and area_scale is not None
        and area_scale >= config.history_spike_area_scale
    )
    reasons = []
    if screen_spanning:
        reasons.append("screen_spanning_box")
    if history_spike:
        reasons.append("area_spike_vs_history")
    return {
        "status": "accepted" if not reasons else "rejected_background_spike",
        "reason": "accepted" if not reasons else "+".join(reasons),
        "history_median_area_pixels": history_median_area,
        "area_scale_vs_history_median": area_scale,
    }


def build_box_observations(
    candidates: dict[str, Any],
    *,
    config: BoxGateConfig | None = None,
) -> dict[str, Any]:
    """Gate all hand-linked boxes and retain one legacy timing observation."""

    config = config or BoxGateConfig()
    config.validate()
    width = int(candidates["video"]["width"])
    height = int(candidates["video"]["height"])
    accepted_areas: deque[float] = deque(maxlen=config.history_size)
    frames: list[dict[str, Any]] = []

    for frame in candidates["frames"]:
        candidate_records = []
        for linked in _hand_linked_firstobjects(frame):
            linked_detection = linked["detection"]
            linked_metrics = _box_metrics(
                linked_detection["box_xyxy"], width=width, height=height
            )
            gate = _gate_metrics(
                linked_metrics,
                accepted_areas=accepted_areas,
                config=config,
            )
            candidate_records.append(
                {
                    "source_detection_id": str(linked_detection["detection_id"]),
                    "linked_hand_detection_ids": linked["linked_hand_detection_ids"],
                    "hand_link_probability": linked["hand_link_probability"],
                    "box_xyxy": [float(value) for value in linked_detection["box_xyxy"]],
                    "score": float(linked_detection["score"]),
                    **linked_metrics,
                    **gate,
                }
            )

        detection, selection_source, linked_hand_ids, hand_link_probability = _select_firstobject(
            frame
        )
        if detection is None:
            frames.append(
                {
                    "frame_idx": int(frame["frame_idx"]),
                    "processed": bool(frame["processed"]),
                    "status": "missing",
                    "reason": "no_firstobject_candidate",
                    "hand_linked_object_candidates": candidate_records,
                }
            )
            continue

        accepted_linked = [
            item for item in candidate_records if item["status"] == "accepted"
        ]
        if selection_source == "hf_link" and accepted_linked:
            selected_id = max(
                accepted_linked,
                key=lambda item: (
                    item["hand_link_probability"] * item["score"],
                    item["score"],
                    item["source_detection_id"],
                ),
            )["source_detection_id"]
            if selected_id != detection["detection_id"]:
                detection = next(
                    item
                    for item in frame["detections"]
                    if item["detection_id"] == selected_id
                )
                selected_record = next(
                    item
                    for item in accepted_linked
                    if item["source_detection_id"] == selected_id
                )
                linked_hand_ids = selected_record["linked_hand_detection_ids"]
                hand_link_probability = selected_record["hand_link_probability"]

        metrics = _box_metrics(detection["box_xyxy"], width=width, height=height)
        selected_candidate = next(
            (
                item
                for item in candidate_records
                if item["source_detection_id"] == detection["detection_id"]
            ),
            None,
        )
        gate = (
            {
                key: selected_candidate[key]
                for key in (
                    "status",
                    "reason",
                    "history_median_area_pixels",
                    "area_scale_vs_history_median",
                )
            }
            if selected_candidate is not None
            else _gate_metrics(metrics, accepted_areas=accepted_areas, config=config)
        )
        accepted = gate["status"] == "accepted"
        if accepted:
            accepted_areas.append(metrics["area_pixels"])

        frames.append(
            {
                "frame_idx": int(frame["frame_idx"]),
                "processed": bool(frame["processed"]),
                "status": gate["status"],
                "reason": gate["reason"],
                "selection_source": selection_source,
                "source_detection_id": detection["detection_id"],
                "linked_hand_detection_ids": linked_hand_ids,
                "hand_link_probability": hand_link_probability,
                "box_xyxy": [float(value) for value in detection["box_xyxy"]],
                "score": float(detection["score"]),
                **metrics,
                "accepted_history_size": len(accepted_areas) - (1 if accepted else 0),
                "history_median_area_pixels": gate["history_median_area_pixels"],
                "area_scale_vs_history_median": gate["area_scale_vs_history_median"],
                "hand_linked_object_candidates": candidate_records,
            }
        )

    counts = {
        status: sum(frame["status"] == status for frame in frames)
        for status in ("accepted", "missing", "rejected_background_spike")
    }
    rejected_frames = [
        frame["frame_idx"] for frame in frames if frame["status"] == "rejected_background_spike"
    ]
    document = {
        "schema_version": SCHEMA_VERSION,
        "dataset": candidates["dataset"],
        "video_id": candidates["video_id"],
        "video": dict(candidates["video"]),
        "config": asdict(config),
        "summary": {
            "num_frames": len(frames),
            "counts_by_status": counts,
            "rejected_background_spike_frames": rejected_frames,
            "hand_linked_candidate_count": sum(
                len(frame.get("hand_linked_object_candidates", [])) for frame in frames
            ),
            "accepted_hand_linked_candidate_count": sum(
                item.get("status") == "accepted"
                for frame in frames
                for item in frame.get("hand_linked_object_candidates", [])
            ),
        },
        "frames": frames,
    }
    _validate_finite(document)
    return document


def _validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite value")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    observations = build_box_observations(read_candidates_json(args.candidates))
    write_json_atomic(args.output, observations)
    print(json.dumps(observations["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
