"""Pure-Python adapter for official HOI-DETR video prediction JSON.

The official exporter stores detections per frame and represents interactions
as indices into that frame's detection array.  This module validates that
format and converts it to a repository-owned, self-describing schema.  It has
no model or image dependencies and is therefore safe to use outside the
legacy HOI-DETR environment.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


SCHEMA_VERSION = "hoi_detr_candidates_v1"
REPORT_SCHEMA_VERSION = "hoi_detr_candidates_report_v1"
SOURCE_FORMAT = "hoi_detr_official_video_json"

CLASS_NAMES: Tuple[str, str, str] = (
    "hand",
    "firstobject",
    "secondobject",
)
CLASS_NAME_BY_ID = dict(enumerate(CLASS_NAMES))
LINK_ENDPOINT_CLASS_IDS = {
    "hf": (0, 1),
    "fs": (1, 2),
}

PathLike = Union[str, Path]


class CandidateValidationError(ValueError):
    """Raised when official or normalized candidate JSON violates its schema."""


def _error(path: str, message: str) -> CandidateValidationError:
    return CandidateValidationError("{}: {}".format(path, message))


def _require_mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise _error(path, "expected an object")
    return value


def _require_list(value: Any, path: str) -> List[Any]:
    if not isinstance(value, list):
        raise _error(path, "expected an array")
    return value


def _require_key(mapping: Mapping, key: str, path: str) -> Any:
    if key not in mapping:
        raise _error(path, "missing required key {!r}".format(key))
    return mapping[key]


def _require_exact_keys(mapping: Mapping, expected: Sequence[str], path: str) -> None:
    expected_set = set(expected)
    actual_set = set(mapping.keys())
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        raise _error(path, "missing required key(s): {}".format(", ".join(missing)))
    if extra:
        raise _error(path, "unknown key(s): {}".format(", ".join(str(key) for key in extra)))


def _require_bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise _error(path, "expected a boolean")
    return value


def _require_int(
    value: Any,
    path: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int:
        raise _error(path, "expected an integer")
    if minimum is not None and value < minimum:
        raise _error(path, "must be >= {}".format(minimum))
    if maximum is not None and value > maximum:
        raise _error(path, "must be <= {}".format(maximum))
    return value


def _require_number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    minimum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "expected a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, "expected a finite number")
    if minimum is not None:
        if minimum_exclusive and result <= minimum:
            raise _error(path, "must be > {}".format(minimum))
        if not minimum_exclusive and result < minimum:
            raise _error(path, "must be >= {}".format(minimum))
    if maximum is not None and result > maximum:
        raise _error(path, "must be <= {}".format(maximum))
    return result


def _require_string(value: Any, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise _error(path, "expected a string")
    if nonempty and not value.strip():
        raise _error(path, "must not be empty")
    return value


def _canonicalize_json_value(value: Any, path: str) -> Any:
    """Return a detached JSON value while rejecting lossy/non-finite inputs."""

    if value is None or type(value) is bool or isinstance(value, str):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _error(path, "expected a finite JSON number")
        return value
    if isinstance(value, list):
        return [
            _canonicalize_json_value(item, "{}[{}]".format(path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error(path, "JSON object keys must be strings")
            result[key] = _canonicalize_json_value(item, "{}.{}".format(path, key))
        return result
    raise _error(path, "value is not JSON-compatible")


def _canonicalize_json_mapping(value: Any, path: str) -> Dict[str, Any]:
    mapping = _require_mapping(value, path)
    result = _canonicalize_json_value(mapping, path)
    if not isinstance(result, dict):  # Defensive: _require_mapping already guarantees this.
        raise _error(path, "expected an object")
    return result


def _require_probability(value: Any, path: str) -> float:
    return _require_number(value, path, minimum=0.0, maximum=1.0)


def _require_class(
    class_id_value: Any,
    class_name_value: Any,
    path: str,
) -> Tuple[int, str]:
    class_id = _require_int(class_id_value, path + ".class_id", minimum=0, maximum=2)
    class_name = _require_string(class_name_value, path + ".class_name", nonempty=True)
    expected_name = CLASS_NAME_BY_ID[class_id]
    if class_name != expected_name:
        raise _error(
            path + ".class_name",
            "class_id {} requires {!r}, got {!r}".format(class_id, expected_name, class_name),
        )
    return class_id, class_name


def _require_box(value: Any, path: str, *, width: int, height: int) -> List[float]:
    raw_box = _require_list(value, path)
    if len(raw_box) != 4:
        raise _error(path, "expected four xyxy coordinates")
    box = [
        _require_number(coordinate, "{}[{}]".format(path, index))
        for index, coordinate in enumerate(raw_box)
    ]
    x1, y1, x2, y2 = box
    if not x1 < x2:
        raise _error(path, "requires x1 < x2")
    if not y1 < y2:
        raise _error(path, "requires y1 < y2")
    if x1 < 0.0 or x2 > float(width):
        raise _error(path, "x coordinates must be within [0, {}]".format(width))
    if y1 < 0.0 or y2 > float(height):
        raise _error(path, "y coordinates must be within [0, {}]".format(height))
    return box


def _validate_official_class_names(value: Any, path: str) -> List[str]:
    names = _require_list(value, path)
    if names != list(CLASS_NAMES):
        raise _error(path, "expected {}".format(list(CLASS_NAMES)))
    return list(names)


def _normalize_official_detection(
    raw_detection: Any,
    *,
    frame_idx: int,
    detection_index: int,
    width: int,
    height: int,
    path: str,
) -> Dict[str, Any]:
    detection = _require_mapping(raw_detection, path)
    box = _require_box(_require_key(detection, "box", path), path + ".box", width=width, height=height)
    score = _require_probability(_require_key(detection, "score", path), path + ".score")
    class_id, class_name = _require_class(
        _require_key(detection, "class_id", path),
        _require_key(detection, "class_name", path),
        path,
    )
    return {
        "detection_id": "f{:06d}_d{:04d}".format(frame_idx, detection_index),
        "detection_index": detection_index,
        "box_xyxy": box,
        "score": score,
        "class_id": class_id,
        "class_name": class_name,
    }


def _normalize_official_links(
    raw_links: Any,
    *,
    kind: str,
    frame_idx: int,
    detections: List[Dict[str, Any]],
    path: str,
) -> List[Dict[str, Any]]:
    links = _require_list(raw_links, path)
    source_class_id, target_class_id = LINK_ENDPOINT_CLASS_IDS[kind]
    normalized: List[Dict[str, Any]] = []
    for link_index, raw_link in enumerate(links):
        link_path = "{}[{}]".format(path, link_index)
        link = _require_mapping(raw_link, link_path)
        source_index = _require_int(_require_key(link, "a", link_path), link_path + ".a", minimum=0)
        target_index = _require_int(_require_key(link, "b", link_path), link_path + ".b", minimum=0)
        probability = _require_probability(_require_key(link, "prob", link_path), link_path + ".prob")
        if source_index >= len(detections):
            raise _error(link_path + ".a", "detection index is out of range")
        if target_index >= len(detections):
            raise _error(link_path + ".b", "detection index is out of range")

        source = detections[source_index]
        target = detections[target_index]
        if source["class_id"] != source_class_id:
            raise _error(
                link_path + ".a",
                "{} source must be class {} ({})".format(
                    kind,
                    source_class_id,
                    CLASS_NAME_BY_ID[source_class_id],
                ),
            )
        if target["class_id"] != target_class_id:
            raise _error(
                link_path + ".b",
                "{} target must be class {} ({})".format(
                    kind,
                    target_class_id,
                    CLASS_NAME_BY_ID[target_class_id],
                ),
            )
        normalized.append(
            {
                "link_id": "f{:06d}_{}{:04d}".format(frame_idx, kind, link_index),
                "kind": kind,
                "source_detection_index": source_index,
                "target_detection_index": target_index,
                "source_detection_id": source["detection_id"],
                "target_detection_id": target["detection_id"],
                "prob": probability,
            }
        )
    return normalized


def normalize_official_video(
    payload: Any,
    *,
    dataset: str,
    video_id: str,
    video_path: PathLike,
    source_metadata: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """Validate and normalize one official ``demo_video.py`` JSON payload.

    Detection and link ordering is preserved exactly.  In particular, multiple
    hands may link to the same first-order object and one detection may appear
    in multiple links; the adapter never selects or collapses a preferred link.
    """

    normalized_dataset = _require_string(dataset, "dataset", nonempty=True)
    normalized_video_id = _require_string(video_id, "video_id", nonempty=True)
    normalized_video_path = _require_string(str(video_path), "video_path", nonempty=True)
    if source_metadata is None:
        normalized_source_metadata: Dict[str, Any] = {}
    else:
        normalized_source_metadata = _canonicalize_json_mapping(source_metadata, "source_metadata")

    root = _require_mapping(payload, "$")
    payload_type = _require_string(_require_key(root, "type", "$"), "$.type")
    if payload_type != "video":
        raise _error("$.type", "expected 'video'")

    official_video_path = _require_string(
        _require_key(root, "video_path", "$"),
        "$.video_path",
        nonempty=True,
    )
    fps = _require_number(_require_key(root, "fps", "$"), "$.fps", minimum=0.0, minimum_exclusive=True)
    width = _require_int(_require_key(root, "width", "$"), "$.width", minimum=1)
    height = _require_int(_require_key(root, "height", "$"), "$.height", minimum=1)
    num_frames = _require_int(_require_key(root, "num_frames", "$"), "$.num_frames", minimum=1)
    score_threshold = _require_probability(_require_key(root, "score_thr", "$"), "$.score_thr")
    nms_iou = _require_probability(_require_key(root, "nms_iou", "$"), "$.nms_iou")
    frame_stride = _require_int(_require_key(root, "frame_stride", "$"), "$.frame_stride", minimum=1)
    class_names = _validate_official_class_names(_require_key(root, "class_names", "$"), "$.class_names")
    raw_frames = _require_list(_require_key(root, "frames", "$"), "$.frames")
    if len(raw_frames) != num_frames:
        raise _error(
            "$.frames",
            "length {} does not match num_frames {}".format(len(raw_frames), num_frames),
        )

    frames: List[Dict[str, Any]] = []
    for expected_frame_idx, raw_frame in enumerate(raw_frames):
        frame_path = "$.frames[{}]".format(expected_frame_idx)
        frame = _require_mapping(raw_frame, frame_path)
        frame_idx = _require_int(_require_key(frame, "frame_idx", frame_path), frame_path + ".frame_idx", minimum=0)
        if frame_idx != expected_frame_idx:
            raise _error(
                frame_path + ".frame_idx",
                "expected continuous absolute frame index {}, got {}".format(expected_frame_idx, frame_idx),
            )
        processed = _require_bool(_require_key(frame, "processed", frame_path), frame_path + ".processed")
        raw_detections = _require_list(_require_key(frame, "detections", frame_path), frame_path + ".detections")
        detections = [
            _normalize_official_detection(
                raw_detection,
                frame_idx=frame_idx,
                detection_index=detection_index,
                width=width,
                height=height,
                path="{}.detections[{}]".format(frame_path, detection_index),
            )
            for detection_index, raw_detection in enumerate(raw_detections)
        ]
        hf_links = _normalize_official_links(
            _require_key(frame, "hf", frame_path),
            kind="hf",
            frame_idx=frame_idx,
            detections=detections,
            path=frame_path + ".hf",
        )
        fs_links = _normalize_official_links(
            _require_key(frame, "fs", frame_path),
            kind="fs",
            frame_idx=frame_idx,
            detections=detections,
            path=frame_path + ".fs",
        )
        frames.append(
            {
                "frame_idx": frame_idx,
                "processed": processed,
                "detections": detections,
                "links": {
                    "hf": hf_links,
                    "fs": fs_links,
                },
            }
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "dataset": normalized_dataset,
        "video_id": normalized_video_id,
        "source": {
            "format": SOURCE_FORMAT,
            "official_video_path": official_video_path,
            "metadata": normalized_source_metadata,
        },
        "video": {
            "path": normalized_video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_stride": frame_stride,
        },
        "inference": {
            "score_threshold": score_threshold,
            "nms_iou": nms_iou,
        },
        "class_names": class_names,
        "frames": frames,
    }
    validate_candidates(document)
    return document


def normalize_official_video_json(
    payload: Any,
    *,
    dataset: str,
    video_id: str,
    video_path: PathLike,
    source_metadata: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """Compatibility alias for :func:`normalize_official_video`."""

    return normalize_official_video(
        payload,
        dataset=dataset,
        video_id=video_id,
        video_path=video_path,
        source_metadata=source_metadata,
    )


def _validate_normalized_detection(
    raw_detection: Any,
    *,
    frame_idx: int,
    detection_index: int,
    width: int,
    height: int,
    path: str,
) -> Mapping:
    detection = _require_mapping(raw_detection, path)
    _require_exact_keys(
        detection,
        ("detection_id", "detection_index", "box_xyxy", "score", "class_id", "class_name"),
        path,
    )
    expected_id = "f{:06d}_d{:04d}".format(frame_idx, detection_index)
    detection_id = _require_string(detection["detection_id"], path + ".detection_id", nonempty=True)
    if detection_id != expected_id:
        raise _error(path + ".detection_id", "expected {!r}".format(expected_id))
    actual_index = _require_int(detection["detection_index"], path + ".detection_index", minimum=0)
    if actual_index != detection_index:
        raise _error(path + ".detection_index", "expected {}".format(detection_index))
    _require_box(detection["box_xyxy"], path + ".box_xyxy", width=width, height=height)
    _require_probability(detection["score"], path + ".score")
    _require_class(detection["class_id"], detection["class_name"], path)
    return detection


def _validate_normalized_links(
    raw_links: Any,
    *,
    kind: str,
    frame_idx: int,
    detections: List[Mapping],
    path: str,
) -> None:
    links = _require_list(raw_links, path)
    source_class_id, target_class_id = LINK_ENDPOINT_CLASS_IDS[kind]
    for link_index, raw_link in enumerate(links):
        link_path = "{}[{}]".format(path, link_index)
        link = _require_mapping(raw_link, link_path)
        _require_exact_keys(
            link,
            (
                "link_id",
                "kind",
                "source_detection_index",
                "target_detection_index",
                "source_detection_id",
                "target_detection_id",
                "prob",
            ),
            link_path,
        )
        expected_link_id = "f{:06d}_{}{:04d}".format(frame_idx, kind, link_index)
        link_id = _require_string(link["link_id"], link_path + ".link_id", nonempty=True)
        if link_id != expected_link_id:
            raise _error(link_path + ".link_id", "expected {!r}".format(expected_link_id))
        actual_kind = _require_string(link["kind"], link_path + ".kind")
        if actual_kind != kind:
            raise _error(link_path + ".kind", "expected {!r}".format(kind))
        source_index = _require_int(link["source_detection_index"], link_path + ".source_detection_index", minimum=0)
        target_index = _require_int(link["target_detection_index"], link_path + ".target_detection_index", minimum=0)
        if source_index >= len(detections):
            raise _error(link_path + ".source_detection_index", "detection index is out of range")
        if target_index >= len(detections):
            raise _error(link_path + ".target_detection_index", "detection index is out of range")
        source = detections[source_index]
        target = detections[target_index]
        if source["class_id"] != source_class_id:
            raise _error(link_path + ".source_detection_index", "invalid {} source class".format(kind))
        if target["class_id"] != target_class_id:
            raise _error(link_path + ".target_detection_index", "invalid {} target class".format(kind))
        source_id = _require_string(link["source_detection_id"], link_path + ".source_detection_id")
        target_id = _require_string(link["target_detection_id"], link_path + ".target_detection_id")
        if source_id != source["detection_id"]:
            raise _error(link_path + ".source_detection_id", "does not match source_detection_index")
        if target_id != target["detection_id"]:
            raise _error(link_path + ".target_detection_id", "does not match target_detection_index")
        _require_probability(link["prob"], link_path + ".prob")


def validate_candidates(document: Any) -> None:
    """Strictly validate a repository-owned candidate document."""

    root = _require_mapping(document, "$")
    _require_exact_keys(
        root,
        (
            "schema_version",
            "dataset",
            "video_id",
            "source",
            "video",
            "inference",
            "class_names",
            "frames",
        ),
        "$",
    )
    schema_version = _require_string(root["schema_version"], "$.schema_version")
    if schema_version != SCHEMA_VERSION:
        raise _error("$.schema_version", "expected {!r}".format(SCHEMA_VERSION))
    _require_string(root["dataset"], "$.dataset", nonempty=True)
    _require_string(root["video_id"], "$.video_id", nonempty=True)
    source = _require_mapping(root["source"], "$.source")
    _require_exact_keys(source, ("format", "official_video_path", "metadata"), "$.source")
    source_format = _require_string(source["format"], "$.source.format")
    if source_format != SOURCE_FORMAT:
        raise _error("$.source.format", "expected {!r}".format(SOURCE_FORMAT))
    _require_string(source["official_video_path"], "$.source.official_video_path", nonempty=True)
    _canonicalize_json_mapping(source["metadata"], "$.source.metadata")

    video = _require_mapping(root["video"], "$.video")
    _require_exact_keys(video, ("path", "fps", "width", "height", "num_frames", "frame_stride"), "$.video")
    _require_string(video["path"], "$.video.path", nonempty=True)
    _require_number(video["fps"], "$.video.fps", minimum=0.0, minimum_exclusive=True)
    width = _require_int(video["width"], "$.video.width", minimum=1)
    height = _require_int(video["height"], "$.video.height", minimum=1)
    num_frames = _require_int(video["num_frames"], "$.video.num_frames", minimum=1)
    _require_int(video["frame_stride"], "$.video.frame_stride", minimum=1)

    inference = _require_mapping(root["inference"], "$.inference")
    _require_exact_keys(inference, ("score_threshold", "nms_iou"), "$.inference")
    _require_probability(inference["score_threshold"], "$.inference.score_threshold")
    _require_probability(inference["nms_iou"], "$.inference.nms_iou")
    _validate_official_class_names(root["class_names"], "$.class_names")

    frames = _require_list(root["frames"], "$.frames")
    if len(frames) != num_frames:
        raise _error("$.frames", "length does not match $.video.num_frames")
    for expected_frame_idx, raw_frame in enumerate(frames):
        frame_path = "$.frames[{}]".format(expected_frame_idx)
        frame = _require_mapping(raw_frame, frame_path)
        _require_exact_keys(frame, ("frame_idx", "processed", "detections", "links"), frame_path)
        frame_idx = _require_int(frame["frame_idx"], frame_path + ".frame_idx", minimum=0)
        if frame_idx != expected_frame_idx:
            raise _error(frame_path + ".frame_idx", "expected {}".format(expected_frame_idx))
        _require_bool(frame["processed"], frame_path + ".processed")
        raw_detections = _require_list(frame["detections"], frame_path + ".detections")
        detections = [
            _validate_normalized_detection(
                raw_detection,
                frame_idx=frame_idx,
                detection_index=detection_index,
                width=width,
                height=height,
                path="{}.detections[{}]".format(frame_path, detection_index),
            )
            for detection_index, raw_detection in enumerate(raw_detections)
        ]
        links = _require_mapping(frame["links"], frame_path + ".links")
        _require_exact_keys(links, ("hf", "fs"), frame_path + ".links")
        for kind in ("hf", "fs"):
            _validate_normalized_links(
                links[kind],
                kind=kind,
                frame_idx=frame_idx,
                detections=detections,
                path="{}.links.{}".format(frame_path, kind),
            )


def _frame_counts(frame: Mapping) -> Dict[str, Any]:
    by_class = dict((name, 0) for name in CLASS_NAMES)
    for detection in frame["detections"]:
        by_class[detection["class_name"]] += 1
    hf_count = len(frame["links"]["hf"])
    fs_count = len(frame["links"]["fs"])
    return {
        "frame_idx": frame["frame_idx"],
        "processed": frame["processed"],
        "num_detections": len(frame["detections"]),
        "detections_by_class": by_class,
        "num_links": hf_count + fs_count,
        "links_by_kind": {"hf": hf_count, "fs": fs_count},
    }


def _distribution(values: List[int]) -> Dict[str, Union[int, float]]:
    if not values:
        return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
    return {
        "min": min(values),
        "median": float(statistics.median(values)),
        "mean": float(sum(values)) / float(len(values)),
        "max": max(values),
    }


def _aggregate_frame_counts(frame_counts: List[Dict[str, Any]]) -> Dict[str, Any]:
    detections_by_class = dict((name, 0) for name in CLASS_NAMES)
    links_by_kind = {"hf": 0, "fs": 0}
    for counts in frame_counts:
        for class_name in CLASS_NAMES:
            detections_by_class[class_name] += counts["detections_by_class"][class_name]
        links_by_kind["hf"] += counts["links_by_kind"]["hf"]
        links_by_kind["fs"] += counts["links_by_kind"]["fs"]
    detection_counts = [counts["num_detections"] for counts in frame_counts]
    hf_counts = [counts["links_by_kind"]["hf"] for counts in frame_counts]
    fs_counts = [counts["links_by_kind"]["fs"] for counts in frame_counts]
    return {
        "num_frames": len(frame_counts),
        "num_detections": sum(detection_counts),
        "detections_by_class": detections_by_class,
        "num_links": links_by_kind["hf"] + links_by_kind["fs"],
        "links_by_kind": links_by_kind,
        "frames_with_detections": sum(1 for value in detection_counts if value > 0),
        "frames_with_hf_links": sum(1 for value in hf_counts if value > 0),
        "frames_with_fs_links": sum(1 for value in fs_counts if value > 0),
        "per_frame": {
            "detections": _distribution(detection_counts),
            "hf_links": _distribution(hf_counts),
            "fs_links": _distribution(fs_counts),
        },
    }


def summarize_candidates(document: Any) -> Dict[str, Any]:
    """Return aggregate statistics for all frames and model-processed frames."""

    validate_candidates(document)
    frame_counts = [_frame_counts(frame) for frame in document["frames"]]
    processed_counts = [counts for counts in frame_counts if counts["processed"]]
    unprocessed_counts = [counts for counts in frame_counts if not counts["processed"]]
    summary = _aggregate_frame_counts(frame_counts)
    summary["num_processed_frames"] = len(processed_counts)
    summary["num_unprocessed_frames"] = len(unprocessed_counts)
    summary["processed_frames_only"] = _aggregate_frame_counts(processed_counts)
    return summary


def build_report(document: Any) -> Dict[str, Any]:
    """Build a JSON-ready aggregate and per-frame candidate report."""

    validate_candidates(document)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "candidate_schema_version": SCHEMA_VERSION,
        "dataset": document["dataset"],
        "video_id": document["video_id"],
        "source": _canonicalize_json_mapping(document["source"], "$.source"),
        "video": dict(document["video"]),
        "inference": dict(document["inference"]),
        "summary": summarize_candidates(document),
        "frames": [_frame_counts(frame) for frame in document["frames"]],
    }


def _reject_json_constant(value: str) -> None:
    raise CandidateValidationError("$: invalid non-finite JSON constant {!r}".format(value))


def dumps_candidates(document: Any, *, indent: Optional[int] = 2) -> str:
    """Validate and serialize a candidate document as standards-compliant JSON."""

    validate_candidates(document)
    return json.dumps(document, indent=indent, ensure_ascii=False, allow_nan=False)


def loads_candidates(data: Union[str, bytes, bytearray]) -> Dict[str, Any]:
    """Deserialize and strictly validate a candidate document."""

    try:
        document = json.loads(data, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise CandidateValidationError("invalid candidate JSON: {}".format(exc)) from exc
    validate_candidates(document)
    return document


def _load_json_file(path: PathLike) -> Any:
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CandidateValidationError("cannot read {}: {}".format(source_path, exc)) from exc
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise CandidateValidationError("invalid JSON in {}: {}".format(source_path, exc)) from exc


def read_official_video_json(
    path: PathLike,
    *,
    dataset: str,
    video_id: str,
    video_path: PathLike,
    source_metadata: Optional[Mapping] = None,
) -> Dict[str, Any]:
    """Read and normalize an official HOI-DETR per-video JSON file."""

    return normalize_official_video(
        _load_json_file(path),
        dataset=dataset,
        video_id=video_id,
        video_path=video_path,
        source_metadata=source_metadata,
    )


def read_candidates_json(path: PathLike) -> Dict[str, Any]:
    """Read and validate a repository-owned candidate JSON file."""

    source_path = Path(path)
    try:
        return loads_candidates(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CandidateValidationError("cannot read {}: {}".format(source_path, exc)) from exc


def write_candidates_json(document: Any, path: PathLike) -> Path:
    """Validate and write a repository-owned candidate JSON file."""

    validate_candidates(document)
    destination = Path(path)
    write_json_atomic(destination, document)
    return destination


def write_report_json(document: Any, path: PathLike) -> Path:
    """Build and write the report associated with a candidate document."""

    report = build_report(document)
    destination = Path(path)
    write_json_atomic(destination, report)
    return destination


def write_json_atomic(path: PathLike, payload: Any) -> None:
    """Atomically write any JSON-compatible payload to ``path``.

    The temporary file is created beside the destination so ``os.replace`` is
    a same-filesystem operation.  A failed serialization or replace leaves the
    prior destination untouched and removes the temporary file.
    """

    destination = Path(path)
    canonical_payload = _canonicalize_json_value(payload, "$")
    serialized = json.dumps(
        canonical_payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(destination))
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
