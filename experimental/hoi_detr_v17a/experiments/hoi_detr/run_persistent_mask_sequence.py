"""Propagate registered object IDs and write a temporally gated all-frame mask sequence."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .adapter import write_json_atomic
from .component_expansion import lock_known_components_from_logits
from .mask_ownership import resolve_visible_mask_ownership
from .mask_sequence_quality import TemporalMaskGateConfig, gate_mask_sequence
from .sam2_multi_object import (
    ObjectMaskSeed,
    propagate_independent_object_logits,
    propagate_multi_object_logits,
)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise RuntimeError(f"failed to write mask: {path}")


def _apply_conditioning_ownership(
    masks: dict[str, np.ndarray],
    conditioning_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Keep trusted conditions intact and remove their pixels from predictions."""

    object_ids = list(conditioning_masks)
    for index, first_id in enumerate(object_ids):
        for second_id in object_ids[index + 1 :]:
            if np.any(conditioning_masks[first_id] & conditioning_masks[second_id]):
                raise ValueError(
                    f"conditioning masks overlap: {first_id} and {second_id}"
                )
    resolved = {object_id: mask.copy() for object_id, mask in masks.items()}
    conditioned_union = np.zeros_like(next(iter(conditioning_masks.values())), dtype=bool)
    for mask in conditioning_masks.values():
        conditioned_union |= mask
    for object_id in resolved:
        if object_id not in conditioning_masks:
            resolved[object_id] &= ~conditioned_union
    resolved.update(
        {object_id: mask.copy() for object_id, mask in conditioning_masks.items()}
    )
    return resolved


def _instance_identity_conflicts(
    logits: dict[str, np.ndarray],
    instance_pairs: list[tuple[str, str]],
    *,
    max_overlap_fraction: float,
) -> list[dict[str, float | int | str]]:
    """Detect two related tracks collapsing onto the same visible instance."""

    conflicts = []
    for first_id, second_id in instance_pairs:
        if first_id not in logits or second_id not in logits:
            continue
        first = logits[first_id] > 0.0
        second = logits[second_id] > 0.0
        first_area = int(np.count_nonzero(first))
        second_area = int(np.count_nonzero(second))
        intersection_area = int(np.count_nonzero(first & second))
        smaller_area = min(first_area, second_area)
        overlap_fraction = intersection_area / smaller_area if smaller_area else 0.0
        if overlap_fraction > max_overlap_fraction:
            conflicts.append(
                {
                    "first_object_id": first_id,
                    "second_object_id": second_id,
                    "first_area": first_area,
                    "second_area": second_area,
                    "intersection_area": intersection_area,
                    "overlap_fraction_of_smaller_instance": overlap_fraction,
                }
            )
    return conflicts


def _load_registry_seeds(registry_path: Path) -> tuple[dict, list[ObjectMaskSeed]]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("status") != "ready":
        raise ValueError("reconstruction registry is not ready")
    objects = registry.get("objects", {})
    if not objects:
        raise ValueError("reconstruction registry contains no objects")
    seeds = []
    for object_id, entry in objects.items():
        conditioning_masks = entry.get(
            "conditioning_masks",
            [{"frame_idx": entry["frame_idx"], "mask": entry["mask"]}],
        )
        if not isinstance(conditioning_masks, list) or not conditioning_masks:
            raise ValueError(f"{object_id} contains no conditioning masks")
        for condition in conditioning_masks:
            mask_path = Path(condition["mask"])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"cannot read registered mask: {mask_path}")
            if not np.any(mask):
                raise ValueError(f"registered conditioning mask is empty: {mask_path}")
            seeds.append(
                ObjectMaskSeed(
                    object_id=object_id,
                    frame_idx=int(condition["frame_idx"]),
                    mask=mask > 0,
                )
            )
    return registry, seeds


def _load_expansion_envelope_seeds(registry: dict) -> list[ObjectMaskSeed]:
    """Load private whole-interaction masks without promoting them to instances."""

    seeds = []
    seen_ids = set()
    for record in registry.get("confirmed_hoi_box_expansions", []):
        envelope = record.get("interaction_envelope")
        if not envelope:
            continue
        object_id = str(envelope["object_id"])
        if object_id in registry.get("objects", {}) or object_id in seen_ids:
            raise ValueError(f"interaction envelope ID is not private and unique: {object_id}")
        mask_path = Path(envelope["mask"])
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            raise FileNotFoundError(f"cannot read non-empty interaction envelope: {mask_path}")
        seeds.append(
            ObjectMaskSeed(
                object_id=object_id,
                frame_idx=int(envelope["frame_idx"]),
                mask=mask > 0,
                protected=False,
            )
        )
        seen_ids.add(object_id)
    return seeds


def _load_trusted_interaction_boxes(
    observations_path: Path | None,
    *,
    frame_start: int,
    frame_end: int,
    max_fallback_gap_frames: int = 3,
) -> dict[int, dict]:
    """Read accepted HF boxes and bridge only short rejected/missing gaps."""

    if observations_path is None:
        return {}
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    frames = observations.get("frames", [])
    if frame_end >= len(frames):
        raise ValueError("box observations do not cover the mask sequence")
    trusted = {}
    last_accepted = None
    for frame_idx in range(frame_start, frame_end + 1):
        frame = frames[frame_idx]
        if int(frame.get("frame_idx", -1)) != frame_idx:
            raise ValueError("box observation frames must be continuous")
        accepted_candidates = [
            item
            for item in frame.get("hand_linked_object_candidates", [])
            if item.get("status") == "accepted"
            and isinstance(item.get("box_xyxy"), list)
            and len(item["box_xyxy"]) == 4
        ]
        if accepted_candidates:
            boxes = [item["box_xyxy"] for item in accepted_candidates]
            union_box = [
                min(float(box[0]) for box in boxes),
                min(float(box[1]) for box in boxes),
                max(float(box[2]) for box in boxes),
                max(float(box[3]) for box in boxes),
            ]
            last_accepted = {
                "box_xyxy": union_box,
                "source_frame": frame_idx,
                "source": (
                    "accepted_hf_link_candidate_union"
                    if len(accepted_candidates) > 1
                    else "accepted_hf_link_candidate"
                ),
                "source_detection_ids": [
                    str(item["source_detection_id"])
                    for item in accepted_candidates
                ],
            }
            trusted[frame_idx] = last_accepted
            continue
        if (
            frame.get("status") == "accepted"
            and frame.get("selection_source") == "hf_link"
            and isinstance(frame.get("box_xyxy"), list)
            and len(frame["box_xyxy"]) == 4
        ):
            last_accepted = {
                "box_xyxy": [float(value) for value in frame["box_xyxy"]],
                "source_frame": frame_idx,
                "source": "accepted_hf_link",
            }
            trusted[frame_idx] = last_accepted
            continue
        if (
            last_accepted is not None
            and frame_idx - int(last_accepted["source_frame"]) <= max_fallback_gap_frames
        ):
            trusted[frame_idx] = {
                **last_accepted,
                "source": "short_gap_last_accepted_hf_link",
            }
    return trusted


def _box_region_with_margin(
    shape: tuple[int, int],
    box_xyxy: list[float],
    *,
    margin_fraction: float,
) -> np.ndarray:
    if not 0.0 <= margin_fraction <= 0.5:
        raise ValueError("interaction box margin fraction must be in [0, 0.5]")
    height, width = shape
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    if not (x1 < x2 and y1 < y2):
        raise ValueError("invalid trusted interaction box")
    margin_x = (x2 - x1) * margin_fraction
    margin_y = (y2 - y1) * margin_fraction
    left = max(0, int(np.floor(x1 - margin_x)))
    top = max(0, int(np.floor(y1 - margin_y)))
    right = min(width, int(np.ceil(x2 + margin_x)))
    bottom = min(height, int(np.ceil(y2 + margin_y)))
    region = np.zeros(shape, dtype=bool)
    region[top:bottom, left:right] = True
    return region


def _expansion_decomposition_groups(
    registry: dict,
    *,
    object_ids: list[str],
    frame_start: int,
    frame_end: int,
) -> list[dict]:
    """Validate generic known-part/new-residual groups recorded by box expansion."""

    object_order = list(registry.get("object_order", object_ids))
    known_ids = set(object_ids)
    groups = []
    for group_idx, record in enumerate(registry.get("confirmed_hoi_box_expansions", [])):
        new_object_id = str(record["new_object_id"])
        event = record["event"]
        activation_frame = int(event["activation_frame"])
        seed_frame = int(event["seed_frame"])
        recorded_known = record.get("known_object_ids")
        if recorded_known is None:
            # Backward compatibility for registries written before the group
            # membership was made explicit: every earlier registered ID was
            # already present when this residual ID was created.
            new_index = object_order.index(new_object_id)
            recorded_known = object_order[:new_index]
        group_known_ids = [str(object_id) for object_id in recorded_known]
        if (
            not group_known_ids
            or len(set(group_known_ids)) != len(group_known_ids)
            or new_object_id in group_known_ids
        ):
            raise ValueError(f"invalid expansion component membership at group {group_idx}")
        unknown = set(group_known_ids + [new_object_id]).difference(known_ids)
        if unknown:
            raise ValueError(f"expansion group references unknown object IDs: {sorted(unknown)}")
        if not frame_start <= activation_frame <= seed_frame <= frame_end:
            raise ValueError(
                "expansion activation/seed lies outside sequence window: "
                f"{activation_frame}..{seed_frame}"
            )
        area_references = record.get("known_component_area_references", {})
        if area_references and set(area_references) != set(group_known_ids):
            raise ValueError("expansion area references do not match known object IDs")
        groups.append(
            {
                "group_idx": group_idx,
                "activation_frame": activation_frame,
                "seed_frame": seed_frame,
                "known_object_ids": group_known_ids,
                "new_object_id": new_object_id,
                "interaction_envelope_object_id": str(
                    (record.get("interaction_envelope") or {}).get(
                        "object_id", new_object_id
                    )
                ),
                "known_component_area_references": {
                    object_id: int(area_references[object_id])
                    for object_id in group_known_ids
                } if area_references else {},
            }
        )
    groups.sort(key=lambda item: (item["activation_frame"], item["seed_frame"]))
    return groups


def _build_expansion_known_tracks(
    groups: list[dict],
    *,
    seeds: list[ObjectMaskSeed],
    logits_by_frame: dict[int, dict[str, np.ndarray]],
    frame_end: int,
    min_area_retention: float,
    max_area_growth: float,
) -> list[dict]:
    """Track every pre-existing component around an expansion conditioning frame.

    The previous locked mask supplies only the spatial search neighborhood.  A
    fixed pre-expansion area reference prevents a small component from growing
    cumulatively until it becomes the assembled whole.
    """

    conditions_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        conditions_by_frame.setdefault(seed.frame_idx, {})[seed.object_id] = seed.mask

    tracked_groups = []
    for group in groups:
        seed_frame = int(group["seed_frame"])
        activation_frame = int(group["activation_frame"])
        group_known_ids = list(group["known_object_ids"])
        seed_conditions = conditions_by_frame.get(seed_frame, {})
        missing_seed_ids = [
            object_id for object_id in group_known_ids if object_id not in seed_conditions
        ]
        if missing_seed_ids:
            raise ValueError(
                "expansion registry lacks known-component conditions at seed frame "
                f"{seed_frame}: {missing_seed_ids}"
            )
        seed_masks = {
            object_id: np.asarray(seed_conditions[object_id], dtype=bool).copy()
            for object_id in group_known_ids
        }
        area_references = dict(group["known_component_area_references"])
        if not area_references:
            area_references = {
                object_id: int(np.count_nonzero(mask))
                for object_id, mask in seed_masks.items()
            }
        anchor_frames = sorted(
            frame_idx
            for frame_idx, frame_conditions in conditions_by_frame.items()
            if activation_frame <= frame_idx <= frame_end
            and all(object_id in frame_conditions for object_id in group_known_ids)
        )
        anchor_masks = {
            frame_idx: {
                object_id: np.asarray(
                    conditions_by_frame[frame_idx][object_id], dtype=bool
                ).copy()
                for object_id in group_known_ids
            }
            for frame_idx in anchor_frames
        }
        locked_by_frame = {
            frame_idx: {
                object_id: mask.copy() for object_id, mask in masks.items()
            }
            for frame_idx, masks in anchor_masks.items()
        }
        metrics_by_frame = {
            frame_idx: {
                object_id: {
                    "source": "registered_expansion_conditioning_mask",
                    "locked_area": int(np.count_nonzero(mask)),
                    "area_reference_pixels": int(area_references[object_id]),
                }
                for object_id, mask in masks.items()
            }
            for frame_idx, masks in anchor_masks.items()
        }
        failures = []
        # Each conditioning frame grows a local branch in both directions.
        # A low-area partial mask stops only that branch and never replaces the
        # last reliable spatial reference or deletes the public identity.
        for anchor_frame in anchor_frames:
            directions = (
                ("forward", range(anchor_frame + 1, frame_end + 1)),
                ("reverse", range(anchor_frame - 1, activation_frame - 1, -1)),
            )
            for direction, frame_range in directions:
                spatial_reference = anchor_masks[anchor_frame]
                for frame_idx in frame_range:
                    if frame_idx in anchor_masks:
                        break
                    if frame_idx in locked_by_frame:
                        spatial_reference = locked_by_frame[frame_idx]
                        continue
                    raw_frame = logits_by_frame.get(frame_idx, {})
                    missing_logits = [
                        object_id
                        for object_id in group_known_ids
                        if object_id not in raw_frame
                    ]
                    if missing_logits:
                        failures.append(
                            {
                                "frame_idx": frame_idx,
                                "direction": direction,
                                "status": "temporarily_occluded",
                                "cause": "failed_missing_known_component_logits",
                                "missing_object_ids": missing_logits,
                            }
                        )
                        break
                    lock = lock_known_components_from_logits(
                        spatial_reference,
                        {
                            object_id: raw_frame[object_id]
                            for object_id in group_known_ids
                        },
                        area_reference_pixels=area_references,
                        min_area_retention=min_area_retention,
                        max_area_growth=max_area_growth,
                    )
                    if lock.get("status") != "success":
                        failures.append(
                            {
                                "frame_idx": frame_idx,
                                "direction": direction,
                                "status": "temporarily_occluded",
                                "cause": lock.get("status"),
                                "detail": lock,
                            }
                        )
                        break
                    spatial_reference = lock["masks"]
                    locked_by_frame[frame_idx] = {
                        object_id: mask.copy()
                        for object_id, mask in spatial_reference.items()
                    }
                    metrics_by_frame[frame_idx] = lock["metrics"]

        frozen_frames = [
            frame_idx
            for frame_idx in range(activation_frame, frame_end + 1)
            if frame_idx not in locked_by_frame
        ]
        frozen_intervals = []
        for _, interval in itertools.groupby(
            enumerate(frozen_frames), key=lambda item: item[1] - item[0]
        ):
            frames = [item[1] for item in interval]
            frozen_intervals.append(
                {
                    "start_frame": frames[0],
                    "end_frame": frames[-1],
                    "frame_count": len(frames),
                    "visible_masks": "empty",
                    "identity_state": "frozen_last_reliable",
                }
            )

        tracked_groups.append(
            {
                **group,
                "locked_known_masks_by_frame": locked_by_frame,
                "lock_metrics_by_frame": metrics_by_frame,
                "track_failures": failures,
                "conditioning_anchor_frames": anchor_frames,
                "frozen_intervals": frozen_intervals,
            }
        )
    return tracked_groups


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


def _warp_mask_with_flow(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Forward-warp a sparse binary mask with dense optical flow."""

    ys, xs = np.nonzero(mask)
    warped = np.zeros_like(mask, dtype=bool)
    if len(xs) == 0:
        return warped
    target_x = np.rint(xs + flow[ys, xs, 0]).astype(np.int32)
    target_y = np.rint(ys + flow[ys, xs, 1]).astype(np.int32)
    valid = (
        (target_x >= 0)
        & (target_x < mask.shape[1])
        & (target_y >= 0)
        & (target_y < mask.shape[0])
    )
    warped[target_y[valid], target_x[valid]] = True
    return warped


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def _mask_diagonal(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    return max(
        1.0,
        float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)),
    )


def _motion_compensated_pair_metrics(
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> dict[str, float]:
    """Measure pairwise continuity without carrying flow into mask state."""

    forward_flow = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
    )
    backward_flow = cv2.calcOpticalFlowFarneback(
        current_gray, previous_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
    )
    warped_previous = _warp_mask_with_flow(previous_mask, forward_flow)
    warped_current = _warp_mask_with_flow(current_mask, backward_flow)
    cycled_previous = _warp_mask_with_flow(warped_previous, backward_flow)
    cycled_current = _warp_mask_with_flow(warped_current, forward_flow)
    raw_intersection = int(np.count_nonzero(previous_mask & current_mask))
    warped_intersection = int(np.count_nonzero(warped_previous & current_mask))
    smaller_area = min(
        int(np.count_nonzero(previous_mask)), int(np.count_nonzero(current_mask))
    )
    continuity = max(raw_intersection, warped_intersection) / max(1, smaller_area)
    cycle_iou = min(
        _mask_iou(previous_mask, cycled_previous),
        _mask_iou(current_mask, cycled_current),
    )
    warped_center = _centroid(warped_previous)
    current_center = _centroid(current_mask)
    centroid_step = float(
        np.hypot(
            warped_center[0] - current_center[0],
            warped_center[1] - current_center[1],
        )
        / _mask_diagonal(current_mask)
    )
    return {
        "flow_warp_continuity": continuity,
        "cycle_iou": cycle_iou,
        "compensated_centroid_step_diagonals": centroid_step,
    }


def _read_gray_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read video frame {frame_idx}: {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()


def _candidate_group_masks(
    group: dict,
    *,
    frame_idx: int,
    reference_known_masks: dict[str, np.ndarray],
    logits_by_frame: dict[int, dict[str, np.ndarray]],
    trusted_interaction_boxes: dict[int, dict],
    frame_shape: tuple[int, int],
    min_area_pixels: int,
    min_area_retention: float,
    max_area_growth: float,
    interaction_box_margin_fraction: float,
) -> dict:
    """Extract one disjoint known/residual candidate without mutating state."""

    raw_frame = logits_by_frame.get(frame_idx, {})
    known_ids = list(group["known_object_ids"])
    new_object_id = str(group["new_object_id"])
    envelope_object_id = str(group["interaction_envelope_object_id"])
    required = list(dict.fromkeys(known_ids + [new_object_id, envelope_object_id]))
    missing = [object_id for object_id in required if object_id not in raw_frame]
    if missing:
        return {"status": "rejected_missing_logits", "missing_object_ids": missing}
    lock = lock_known_components_from_logits(
        reference_known_masks,
        {object_id: raw_frame[object_id] for object_id in known_ids},
        area_reference_pixels=group["known_component_area_references"],
        min_area_retention=min_area_retention,
        max_area_growth=max_area_growth,
    )
    if lock.get("status") != "success":
        return {"status": "rejected_known_component_lock", "lock": lock}
    trusted_box = trusted_interaction_boxes.get(frame_idx)
    if trusted_box is None:
        return {"status": "rejected_no_trusted_interaction_box"}
    box_region = _box_region_with_margin(
        frame_shape,
        trusted_box["box_xyxy"],
        margin_fraction=interaction_box_margin_fraction,
    )
    envelope = (np.asarray(raw_frame[envelope_object_id]) > 0.0) & box_region
    known_union = np.zeros(frame_shape, dtype=bool)
    for mask in lock["masks"].values():
        known_union |= mask
    residual = envelope & ~known_union
    residual_area = int(np.count_nonzero(residual))
    if residual_area < min_area_pixels:
        return {
            "status": "rejected_empty_or_small_residual",
            "residual_area": residual_area,
        }
    raw_new = np.asarray(raw_frame[new_object_id]) > 0.0
    raw_new_support = int(np.count_nonzero(raw_new & residual)) / residual_area
    if raw_new_support < 0.5:
        return {
            "status": "rejected_ambiguous_new_id_support",
            "raw_new_support_fraction": raw_new_support,
        }
    return {
        "status": "candidate",
        "known_masks": lock["masks"],
        "new_mask": residual,
        "envelope_mask": envelope,
        "lock_metrics": lock["metrics"],
        "raw_new_support_fraction": raw_new_support,
        "trusted_box_source": trusted_box["source"],
    }


def _find_reacquisition_anchor(
    group: dict,
    *,
    failure: dict,
    tracked_group: dict,
    video_path: Path,
    logits_by_frame: dict[int, dict[str, np.ndarray]],
    trusted_interaction_boxes: dict[int, dict],
    frame_shape: tuple[int, int],
    frame_start: int,
    frame_end: int,
    max_search_frames: int,
    min_confirmation_frames: int,
    min_flow_warp_continuity: float,
    min_cycle_iou: float,
    max_centroid_step_diagonals: float,
    min_area_pixels: int,
    min_area_retention: float,
    max_area_growth: float,
    interaction_box_margin_fraction: float,
) -> dict:
    """Find a bounded clean anchor while leaving partial masks quarantined."""

    failure_frame = int(failure["frame_idx"])
    direction = str(failure.get("direction", "forward"))
    step = 1 if direction == "forward" else -1
    locked_frames = sorted(tracked_group["locked_known_masks_by_frame"])
    reference_candidates = (
        [frame for frame in locked_frames if frame < failure_frame]
        if step > 0
        else [frame for frame in locked_frames if frame > failure_frame]
    )
    if not reference_candidates:
        return {"status": "failed_no_frozen_reference", "failure_frame": failure_frame}
    reference_frame = reference_candidates[-1] if step > 0 else reference_candidates[0]
    reference_known = tracked_group["locked_known_masks_by_frame"][reference_frame]
    reference_candidate = _candidate_group_masks(
        group,
        frame_idx=reference_frame,
        reference_known_masks=reference_known,
        logits_by_frame=logits_by_frame,
        trusted_interaction_boxes=trusted_interaction_boxes,
        frame_shape=frame_shape,
        min_area_pixels=min_area_pixels,
        min_area_retention=min_area_retention,
        max_area_growth=max_area_growth,
        interaction_box_margin_fraction=interaction_box_margin_fraction,
    )
    if reference_candidate.get("status") != "candidate":
        return {
            "status": "failed_invalid_frozen_reference_decomposition",
            "failure_frame": failure_frame,
            "detail": reference_candidate,
        }
    previous_masks = {
        **reference_candidate["known_masks"],
        str(group["new_object_id"]): reference_candidate["new_mask"],
    }
    previous_frame = reference_frame
    previous_envelope = reference_candidate["envelope_mask"]
    previous_envelope_frame = reference_frame
    run = []
    candidate_audit = []
    search_end = (
        min(frame_end, failure_frame + max_search_frames)
        if step > 0
        else max(frame_start, failure_frame - max_search_frames)
    )
    for frame_idx in range(failure_frame + step, search_end + step, step):
        raw_frame = logits_by_frame.get(frame_idx, {})
        envelope_object_id = str(group["interaction_envelope_object_id"])
        trusted_box = trusted_interaction_boxes.get(frame_idx)
        if envelope_object_id not in raw_frame or trusted_box is None:
            candidate_audit.append(
                {
                    "frame_idx": frame_idx,
                    "candidate_status": "rejected_missing_private_envelope_bridge",
                }
            )
            return {
                "status": "failed_private_envelope_track_discontinuity",
                "failure_frame": failure_frame,
                "direction": direction,
                "discontinuity_frame": frame_idx,
                "candidate_audit": candidate_audit,
            }
        box_region = _box_region_with_margin(
            frame_shape,
            trusted_box["box_xyxy"],
            margin_fraction=interaction_box_margin_fraction,
        )
        current_envelope = (
            np.asarray(raw_frame[envelope_object_id]) > 0.0
        ) & box_region
        if not np.any(current_envelope):
            candidate_audit.append(
                {
                    "frame_idx": frame_idx,
                    "candidate_status": "rejected_empty_private_envelope_bridge",
                }
            )
            return {
                "status": "failed_private_envelope_track_discontinuity",
                "failure_frame": failure_frame,
                "direction": direction,
                "discontinuity_frame": frame_idx,
                "candidate_audit": candidate_audit,
            }
        previous_envelope_gray = _read_gray_frame(
            video_path, previous_envelope_frame
        )
        current_gray = _read_gray_frame(video_path, frame_idx)
        envelope_metrics = _motion_compensated_pair_metrics(
            previous_envelope,
            current_envelope,
            previous_envelope_gray,
            current_gray,
        )
        envelope_reliable = (
            envelope_metrics["flow_warp_continuity"]
            >= min_flow_warp_continuity
            and envelope_metrics["cycle_iou"] >= min_cycle_iou
            and envelope_metrics["compensated_centroid_step_diagonals"]
            <= max_centroid_step_diagonals
        )
        if not envelope_reliable:
            candidate_audit.append(
                {
                    "frame_idx": frame_idx,
                    "candidate_status": "rejected_private_envelope_discontinuity",
                    "private_envelope_metrics": envelope_metrics,
                }
            )
            return {
                "status": "failed_private_envelope_track_discontinuity",
                "failure_frame": failure_frame,
                "direction": direction,
                "discontinuity_frame": frame_idx,
                "validation_window": [
                    max(frame_start, failure_frame - max_search_frames),
                    min(frame_end, failure_frame + max_search_frames),
                ],
                "candidate_audit": candidate_audit,
            }
        previous_envelope = current_envelope
        previous_envelope_frame = frame_idx
        candidate = _candidate_group_masks(
            group,
            frame_idx=frame_idx,
            reference_known_masks=reference_known,
            logits_by_frame=logits_by_frame,
            trusted_interaction_boxes=trusted_interaction_boxes,
            frame_shape=frame_shape,
            min_area_pixels=min_area_pixels,
            min_area_retention=min_area_retention,
            max_area_growth=max_area_growth,
            interaction_box_margin_fraction=interaction_box_margin_fraction,
        )
        audit = {"frame_idx": frame_idx, "candidate_status": candidate["status"]}
        if candidate.get("status") != "candidate":
            candidate_audit.append(
                {
                    **audit,
                    "detail": candidate,
                    "private_envelope_metrics": envelope_metrics,
                }
            )
            run = []
            continue
        current_masks = {
            **candidate["known_masks"],
            str(group["new_object_id"]): candidate["new_mask"],
        }
        if run or frame_idx == reference_frame + step:
            previous_gray = _read_gray_frame(video_path, previous_frame)
            per_object_metrics = {
                object_id: _motion_compensated_pair_metrics(
                    previous_masks[object_id],
                    current_masks[object_id],
                    previous_gray,
                    current_gray,
                )
                for object_id in current_masks
            }
            reliable = all(
                metrics["flow_warp_continuity"] >= min_flow_warp_continuity
                and metrics["cycle_iou"] >= min_cycle_iou
                and metrics["compensated_centroid_step_diagonals"]
                <= max_centroid_step_diagonals
                for metrics in per_object_metrics.values()
            )
            public_bridge_source = "adjacent_public_masks"
        else:
            # During a true occlusion there is no trustworthy public part mask
            # to flow-warp.  Identity is bridged by the private envelope's
            # adjacent-frame track; the original SAM2 logical IDs and fixed
            # area lock then restart public confirmation at this clean frame.
            per_object_metrics = {}
            reliable = True
            public_bridge_source = "private_envelope_occlusion_bridge"
        audit.update(
            {
                "reliable": reliable,
                "per_object_metrics": per_object_metrics,
                "private_envelope_metrics": envelope_metrics,
                "public_bridge_source": public_bridge_source,
                "raw_new_support_fraction": candidate["raw_new_support_fraction"],
            }
        )
        candidate_audit.append(audit)
        if reliable:
            run.append((frame_idx, candidate))
            previous_masks = current_masks
            previous_frame = frame_idx
            if len(run) >= min_confirmation_frames:
                anchor_frame, anchor = run[-1]
                return {
                    "status": "success",
                    "failure_frame": failure_frame,
                    "direction": direction,
                    "validation_window": [
                        max(frame_start, failure_frame - max_search_frames),
                        min(frame_end, failure_frame + max_search_frames),
                    ],
                    "reference_frame": reference_frame,
                    "anchor_frame": anchor_frame,
                    "confirmation_frames": [item[0] for item in run],
                    "known_masks": anchor["known_masks"],
                    "new_mask": anchor["new_mask"],
                    "envelope_mask": anchor["envelope_mask"],
                    "candidate_audit": candidate_audit,
                }
        else:
            # A full-size candidate that cannot be connected to the frozen
            # identity is a trajectory break, not an occlusion.  Rejecting the
            # search here prevents a different object that later reaches the
            # same image location from being accepted as a recovered ID.
            return {
                "status": "failed_temporal_track_discontinuity",
                "failure_frame": failure_frame,
                "direction": direction,
                "discontinuity_frame": frame_idx,
                "validation_window": [
                    max(frame_start, failure_frame - max_search_frames),
                    min(frame_end, failure_frame + max_search_frames),
                ],
                "candidate_audit": candidate_audit,
            }
    return {
        "status": "failed_no_reliable_reacquisition_anchor",
        "failure_frame": failure_frame,
        "direction": direction,
        "validation_window": [
            max(frame_start, failure_frame - max_search_frames),
            min(frame_end, failure_frame + max_search_frames),
        ],
        "candidate_audit": candidate_audit,
    }


def _masked_positive_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep a finite positive score only on a structurally assigned mask."""

    array = np.asarray(logits, dtype=np.float32)
    result = np.full(array.shape, -1_000_000.0, dtype=np.float32)
    result[mask] = np.maximum(array[mask], np.float32(1e-6))
    return result


def _apply_expansion_decompositions(
    frame_idx: int,
    raw_public_logits: dict[str, np.ndarray],
    all_raw_logits: dict[str, np.ndarray],
    tracked_groups: list[dict],
    *,
    frame_shape: tuple[int, int],
    trusted_interaction_boxes: dict[int, dict] | None = None,
    interaction_box_margin_fraction: float = 0.08,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Replace collapse-prone part tracks with disjoint known/residual masks."""

    adjusted = {
        object_id: logits.copy() for object_id, logits in raw_public_logits.items()
    }
    trusted_interaction_boxes = trusted_interaction_boxes or {}
    audits = []
    for group in tracked_groups:
        if frame_idx < int(group["activation_frame"]):
            continue
        known_ids = list(group["known_object_ids"])
        new_object_id = str(group["new_object_id"])
        envelope_object_id = str(group["interaction_envelope_object_id"])
        member_ids = known_ids + [new_object_id]
        locked = group["locked_known_masks_by_frame"].get(frame_idx)
        missing_raw_ids = [
            object_id for object_id in known_ids if object_id not in raw_public_logits
        ]
        if envelope_object_id not in all_raw_logits:
            missing_raw_ids.append(envelope_object_id)
        if locked is None or missing_raw_ids:
            for object_id in member_ids:
                adjusted[object_id] = np.full(
                    frame_shape, -1_000_000.0, dtype=np.float32
                )
            audits.append(
                {
                    "group_idx": int(group["group_idx"]),
                    "status": "temporarily_occluded_empty_visible_masks",
                    "known_object_ids": known_ids,
                    "new_object_id": new_object_id,
                    "interaction_envelope_object_id": envelope_object_id,
                    "missing_raw_object_ids": missing_raw_ids,
                    "identity_state": "frozen_last_reliable",
                }
            )
            continue

        known_union = np.zeros(frame_shape, dtype=bool)
        for object_id in known_ids:
            known_mask = np.asarray(locked[object_id], dtype=bool)
            known_union |= known_mask
            adjusted[object_id] = _masked_positive_logits(
                raw_public_logits[object_id], known_mask
            )
        envelope_raw_mask = np.asarray(all_raw_logits[envelope_object_id]) > 0.0
        trusted_box = trusted_interaction_boxes.get(frame_idx)
        if trusted_box is None:
            constrained_envelope = envelope_raw_mask
            box_source = "no_trusted_box_sam2_envelope_only"
        else:
            box_region = _box_region_with_margin(
                frame_shape,
                trusted_box["box_xyxy"],
                margin_fraction=interaction_box_margin_fraction,
            )
            constrained_envelope = envelope_raw_mask & box_region
            box_source = str(trusted_box["source"])
        new_residual = constrained_envelope & ~known_union
        adjusted[new_object_id] = _masked_positive_logits(
            all_raw_logits[envelope_object_id], new_residual
        )
        audits.append(
            {
                "group_idx": int(group["group_idx"]),
                "status": (
                    "success" if np.any(new_residual) else "empty_visible_new_component_residual"
                ),
                "known_object_ids": known_ids,
                "new_object_id": new_object_id,
                "interaction_envelope_object_id": envelope_object_id,
                "hoi_box_source": box_source,
                "hoi_box_source_frame": (
                    None if trusted_box is None else int(trusted_box["source_frame"])
                ),
                "known_union_area": int(np.count_nonzero(known_union)),
                "envelope_raw_positive_area": int(
                    np.count_nonzero(envelope_raw_mask)
                ),
                "envelope_box_constrained_area": int(
                    np.count_nonzero(constrained_envelope)
                ),
                "envelope_overlap_with_known_pixels": int(
                    np.count_nonzero(constrained_envelope & known_union)
                ),
                "new_residual_area": int(np.count_nonzero(new_residual)),
            }
        )
    return adjusted, audits


def run(args: argparse.Namespace) -> dict:
    if not 0 <= args.max_cross_instance_overlap_fraction <= 1:
        raise ValueError("max_cross_instance_overlap_fraction must be in [0, 1]")
    if not 0 < args.min_locked_component_area_retention <= 1:
        raise ValueError("min_locked_component_area_retention must be in (0, 1]")
    if args.max_locked_component_area_growth < 1:
        raise ValueError("max_locked_component_area_growth must be at least one")
    if not 0.0 <= args.interaction_box_margin_fraction <= 0.5:
        raise ValueError("interaction_box_margin_fraction must be in [0, 0.5]")
    if args.max_interaction_box_fallback_gap_frames < 0:
        raise ValueError("max_interaction_box_fallback_gap_frames must be non-negative")
    if args.max_temporal_validation_seconds <= 0:
        raise ValueError("max_temporal_validation_seconds must be positive")
    if args.min_temporal_confirmation_frames < 1:
        raise ValueError("min_temporal_confirmation_frames must be positive")
    if not 0 <= args.min_flow_warp_continuity <= 1:
        raise ValueError("min_flow_warp_continuity must be in [0, 1]")
    if not 0 <= args.min_cycle_iou <= 1:
        raise ValueError("min_cycle_iou must be in [0, 1]")
    if args.max_compensated_centroid_step_diagonals < 0:
        raise ValueError("max_compensated_centroid_step_diagonals must be non-negative")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry, seeds = _load_registry_seeds(args.registry)
    envelope_seeds = _load_expansion_envelope_seeds(registry)
    if envelope_seeds and not args.independent_object_states:
        raise ValueError("private interaction envelopes require independent SAM2 states")

    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {args.video}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    frame_end = num_frames - 1 if args.frame_end is None else args.frame_end
    if args.frame_start < 0 or frame_end < args.frame_start or frame_end >= num_frames:
        raise ValueError(f"invalid frame window: {args.frame_start}..{frame_end}")
    outside = [
        seed.object_id
        for seed in seeds + envelope_seeds
        if not args.frame_start <= seed.frame_idx <= frame_end
    ]
    if outside:
        raise ValueError(f"registered seed frames outside sequence window: {outside}")

    sys.path.insert(0, str(args.sam2_root.resolve()))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(args.checkpoint.resolve()),
        device=f"cuda:{args.gpu}",
        apply_postprocessing=True,
    )
    import torch

    object_ids = list(dict.fromkeys(seed.object_id for seed in seeds))
    activation_frames = {
        object_id: int(registry["objects"][object_id].get("activation_frame", args.frame_start))
        for object_id in object_ids
    }
    invalid_activations = {
        object_id: frame_idx
        for object_id, frame_idx in activation_frames.items()
        if not args.frame_start <= frame_idx <= frame_end
    }
    if invalid_activations:
        raise ValueError(f"object activation frames outside sequence window: {invalid_activations}")
    primary_seed_frames = {
        object_id: min(seed.frame_idx for seed in seeds if seed.object_id == object_id)
        for object_id in object_ids
    }
    expansion_groups = _expansion_decomposition_groups(
        registry,
        object_ids=object_ids,
        frame_start=args.frame_start,
        frame_end=frame_end,
    )
    trusted_interaction_boxes = _load_trusted_interaction_boxes(
        args.box_observations,
        frame_start=args.frame_start,
        frame_end=frame_end,
        max_fallback_gap_frames=args.max_interaction_box_fallback_gap_frames,
    )
    propagation_function = (
        propagate_independent_object_logits
        if args.independent_object_states
        else propagate_multi_object_logits
    )
    propagation_seeds = list(seeds + envelope_seeds)
    reacquisition_anchors = []
    temporal_candidate_tracks = []
    unresolved_reacquisitions = []
    max_search_frames = max(
        1, int(round(fps * args.max_temporal_validation_seconds))
    )
    tracked_expansion_groups = []
    # At most four clean re-anchors per expansion group.  The temporal search
    # itself is bounded by one second; this outer bound prevents a malformed
    # sequence from causing unbounded SAM2 reconditioning passes.
    max_reacquisition_passes = max(1, len(expansion_groups) * 4 + 1)
    for propagation_pass in range(max_reacquisition_passes):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            propagation = propagation_function(
                predictor,
                video_path=args.video,
                seeds=propagation_seeds,
                init_state_kwargs={"offload_video_to_cpu": True},
                frame_start=args.frame_start,
                frame_end=frame_end,
            )
        tracked_expansion_groups = _build_expansion_known_tracks(
            expansion_groups,
            seeds=[seed for seed in propagation_seeds if seed.object_id in object_ids],
            logits_by_frame=propagation.logits_by_frame,
            frame_end=frame_end,
            min_area_retention=args.min_locked_component_area_retention,
            max_area_growth=args.max_locked_component_area_growth,
        )
        pending_anchors = []
        pass_failures = []
        for group, tracked_group in zip(expansion_groups, tracked_expansion_groups):
            anchor_frames = set(tracked_group["conditioning_anchor_frames"])
            unbounded_failure = None
            for failure in sorted(
                tracked_group["track_failures"],
                key=lambda item: int(item["frame_idx"]),
            ):
                failure_frame = int(failure["frame_idx"])
                if failure.get("direction") == "reverse":
                    bounded = any(
                        failure_frame - max_search_frames <= frame < failure_frame
                        for frame in anchor_frames
                    )
                else:
                    bounded = any(
                        failure_frame < frame <= failure_frame + max_search_frames
                        for frame in anchor_frames
                    )
                if not bounded:
                    unbounded_failure = failure
                    break
            if unbounded_failure is None:
                continue
            result = _find_reacquisition_anchor(
                group,
                failure=unbounded_failure,
                tracked_group=tracked_group,
                video_path=args.video,
                logits_by_frame=propagation.logits_by_frame,
                trusted_interaction_boxes=trusted_interaction_boxes,
                frame_shape=(height, width),
                frame_start=args.frame_start,
                frame_end=frame_end,
                max_search_frames=max_search_frames,
                min_confirmation_frames=args.min_temporal_confirmation_frames,
                min_flow_warp_continuity=args.min_flow_warp_continuity,
                min_cycle_iou=args.min_cycle_iou,
                max_centroid_step_diagonals=(
                    args.max_compensated_centroid_step_diagonals
                ),
                min_area_pixels=args.min_area_pixels,
                min_area_retention=args.min_locked_component_area_retention,
                max_area_growth=args.max_locked_component_area_growth,
                interaction_box_margin_fraction=args.interaction_box_margin_fraction,
            )
            result_audit = {
                key: value
                for key, value in result.items()
                if key not in {"known_masks", "new_mask", "envelope_mask"}
            }
            result_audit["group_idx"] = int(group["group_idx"])
            temporal_candidate_tracks.append(result_audit)
            if result.get("status") != "success":
                pass_failures.append(result_audit)
                continue
            anchor_frame = int(result["anchor_frame"])
            anchor_dir = (
                output_dir
                / "reacquisition_anchors"
                / f"group_{int(group['group_idx']):02d}"
                / f"frame_{anchor_frame:06d}"
            )
            for object_id, mask in result["known_masks"].items():
                _write_mask(anchor_dir / f"{object_id}.png", mask)
                pending_anchors.append(ObjectMaskSeed(object_id, anchor_frame, mask))
            new_object_id = str(group["new_object_id"])
            envelope_object_id = str(group["interaction_envelope_object_id"])
            _write_mask(anchor_dir / f"{new_object_id}.png", result["new_mask"])
            _write_mask(
                anchor_dir / f"{envelope_object_id}.png", result["envelope_mask"]
            )
            pending_anchors.extend(
                [
                    ObjectMaskSeed(new_object_id, anchor_frame, result["new_mask"]),
                    ObjectMaskSeed(
                        envelope_object_id, anchor_frame, result["envelope_mask"]
                    ),
                ]
            )
            reacquisition_anchors.append(
                {
                    **result_audit,
                    "mask_directory": str(anchor_dir.resolve()),
                    "reconditioned_object_ids": (
                        list(result["known_masks"])
                        + [new_object_id, envelope_object_id]
                    ),
                }
            )
        if pass_failures:
            unresolved_reacquisitions = pass_failures
            break
        if not pending_anchors:
            break
        existing_keys = {
            (seed.object_id, seed.frame_idx) for seed in propagation_seeds
        }
        unique_pending = [
            seed
            for seed in pending_anchors
            if (seed.object_id, seed.frame_idx) not in existing_keys
        ]
        if not unique_pending:
            unresolved_reacquisitions = [
                {
                    "status": "failed_duplicate_reacquisition_anchor",
                    "propagation_pass": propagation_pass,
                }
            ]
            break
        propagation_seeds.extend(unique_pending)
    else:
        unresolved_reacquisitions = [
            {
                "status": "failed_reacquisition_pass_limit",
                "max_reacquisition_passes": max_reacquisition_passes,
            }
        ]

    masks_by_object = {object_id: {} for object_id in object_ids}
    unresolved_masks_by_frame: dict[int, dict[str, np.ndarray]] = {}
    conditioning_masks_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for seed in propagation_seeds:
        if seed.object_id not in object_ids:
            continue
        conditioning_masks_by_frame.setdefault(seed.frame_idx, {})[seed.object_id] = seed.mask
    overlap_before = 0
    overlap_after = 0
    identity_conflicts_by_frame: dict[int, list[dict[str, float | int | str]]] = {}
    decomposition_audit_by_frame: dict[int, list[dict]] = {}
    instance_pairs = list(itertools.combinations(object_ids, 2))
    for frame_idx in range(args.frame_start, frame_end + 1):
        all_raw_logits = propagation.logits_by_frame.get(frame_idx, {})
        raw_logits = {
            object_id: logits
            for object_id, logits in all_raw_logits.items()
            if object_id in activation_frames
            and frame_idx >= activation_frames[object_id]
        }
        logits, decomposition_audit = _apply_expansion_decompositions(
            frame_idx,
            raw_logits,
            all_raw_logits,
            tracked_expansion_groups,
            frame_shape=(height, width),
            trusted_interaction_boxes=trusted_interaction_boxes,
            interaction_box_margin_fraction=args.interaction_box_margin_fraction,
        )
        if decomposition_audit:
            decomposition_audit_by_frame[frame_idx] = decomposition_audit
        if logits:
            unresolved_masks = {
                object_id: logit > 0.0 for object_id, logit in raw_logits.items()
            }
            identity_conflicts = (
                _instance_identity_conflicts(
                    logits,
                    [
                        pair
                        for pair in instance_pairs
                        if pair[0] in logits and pair[1] in logits
                    ],
                    max_overlap_fraction=args.max_cross_instance_overlap_fraction,
                )
                if args.independent_object_states
                else []
            )
            if identity_conflicts:
                identity_conflicts_by_frame[frame_idx] = identity_conflicts
            if identity_conflicts or decomposition_audit:
                unresolved_masks_by_frame[frame_idx] = unresolved_masks
            ownership = resolve_visible_mask_ownership(
                logits,
                protected_object_ids=set(logits),
                protected_core_erosion_pixels=1,
            )
            frame_masks = ownership.masks
            for conflict in identity_conflicts:
                frame_masks[str(conflict["first_object_id"])] = np.zeros(
                    (height, width), dtype=bool
                )
                frame_masks[str(conflict["second_object_id"])] = np.zeros(
                    (height, width), dtype=bool
                )
            if frame_idx in conditioning_masks_by_frame:
                frame_masks = _apply_conditioning_ownership(
                    frame_masks,
                    conditioning_masks_by_frame[frame_idx],
                )
            overlap_before += ownership.overlap_pixels_before
            overlap_after += ownership.overlap_pixels_after
        else:
            frame_masks = {}
        for object_id in object_ids:
            masks_by_object[object_id][frame_idx] = frame_masks.get(
                object_id,
                np.zeros((height, width), dtype=bool),
            )

    config = TemporalMaskGateConfig(
        min_area_pixels=args.min_area_pixels,
        min_largest_component_fraction=args.min_largest_component_fraction,
        max_frame_area_fraction=args.max_frame_area_fraction,
        max_border_area_fraction=args.max_border_area_fraction,
        min_area_ratio_vs_history=args.min_area_ratio_vs_history,
        max_area_ratio_vs_history=args.max_area_ratio_vs_history,
        max_centroid_step_diagonals=args.max_centroid_step_diagonals,
        history_size=args.history_size,
    )
    gate_seed_frames = {
        object_id: min(
            (
                seed.frame_idx
                for seed in seeds
                if seed.object_id == object_id
                and seed.frame_idx >= activation_frames[object_id]
            ),
            default=activation_frames[object_id],
        )
        for object_id in object_ids
    }
    decisions_by_object = {
        object_id: gate_mask_sequence(
            {
                frame_idx: mask
                for frame_idx, mask in masks_by_object[object_id].items()
                if frame_idx >= activation_frames[object_id]
            },
            object_id=object_id,
            seed_frame=gate_seed_frames[object_id],
            frame_shape=(height, width),
            config=config,
        )
        for object_id in object_ids
    }

    frames = []
    accepted_overlap_pixels = 0
    for frame_idx in range(args.frame_start, frame_end + 1):
        frame_entry = {"frame_idx": frame_idx, "objects": {}}
        accepted_masks = []
        for object_id in object_ids:
            mask = masks_by_object[object_id][frame_idx]
            inactive = frame_idx < activation_frames[object_id]
            decision = (
                {
                    "frame_idx": frame_idx,
                    "object_id": object_id,
                    "status": "inactive_before_object_activation",
                    "reasons": [],
                    "metrics": None,
                }
                if inactive
                else decisions_by_object[object_id][frame_idx]
            )
            raw_path = output_dir / "raw_masks" / f"frame_{frame_idx:06d}" / f"{object_id}.png"
            _write_mask(raw_path, mask)
            unresolved_path = None
            if object_id in unresolved_masks_by_frame.get(frame_idx, {}):
                unresolved_path = (
                    output_dir
                    / "unresolved_masks"
                    / f"frame_{frame_idx:06d}"
                    / f"{object_id}.png"
                )
                _write_mask(unresolved_path, unresolved_masks_by_frame[frame_idx][object_id])
            accepted_path = None
            if not inactive and decision["status"] == "accepted":
                accepted_path = (
                    output_dir / "masks" / f"frame_{frame_idx:06d}" / f"{object_id}.png"
                )
                _write_mask(accepted_path, mask)
                accepted_masks.append(mask)
            frame_entry["objects"][object_id] = {
                **decision,
                "conditioning_frame": frame_idx
                in propagation.conditioning_frames[object_id],
                "raw_mask": str(raw_path.resolve()),
                "unresolved_mask": (
                    None if unresolved_path is None else str(unresolved_path.resolve())
                ),
                "mask": None if accepted_path is None else str(accepted_path.resolve()),
            }
        for first_idx, first in enumerate(accepted_masks):
            for second in accepted_masks[first_idx + 1 :]:
                accepted_overlap_pixels += int(np.count_nonzero(first & second))
        frames.append(frame_entry)

    objects = {}
    for object_id in object_ids:
        seed_frame = primary_seed_frames[object_id]
        readiness_anchor_frame = gate_seed_frames[object_id]
        objects[object_id] = {
            "seed_frame": seed_frame,
            "conditioning_frames": list(propagation.conditioning_frames[object_id]),
            "activation_frame": activation_frames[object_id],
            "readiness_anchor_frame": readiness_anchor_frame,
        }
    identity_conflicts_payload = {
        str(frame_idx): conflicts
        for frame_idx, conflicts in identity_conflicts_by_frame.items()
    }
    decomposition_audit_payload = {
        str(frame_idx): audit
        for frame_idx, audit in decomposition_audit_by_frame.items()
    }
    decomposition_groups_payload = [
        {
            "group_idx": int(group["group_idx"]),
            "activation_frame": int(group["activation_frame"]),
            "seed_frame": int(group["seed_frame"]),
            "known_object_ids": list(group["known_object_ids"]),
            "new_object_id": str(group["new_object_id"]),
            "interaction_envelope_object_id": str(
                group["interaction_envelope_object_id"]
            ),
            "known_component_area_references": dict(
                group["known_component_area_references"]
            ),
            "lock_metrics_by_frame": {
                str(frame_idx): metrics
                for frame_idx, metrics in group["lock_metrics_by_frame"].items()
            },
            "track_failures": group["track_failures"],
            "conditioning_anchor_frames": list(
                group["conditioning_anchor_frames"]
            ),
            "frozen_intervals": list(group["frozen_intervals"]),
        }
        for group in tracked_expansion_groups
    ]
    frozen_intervals_payload = [
        {
            "group_idx": int(group["group_idx"]),
            **interval,
        }
        for group in tracked_expansion_groups
        for interval in group["frozen_intervals"]
    ]
    private_envelope_audit = [
        {
            "object_id": seed.object_id,
            "conditioning_frame": int(seed.frame_idx),
            "public_instance": False,
            "present_in_public_object_ids": seed.object_id in object_ids,
        }
        for seed in propagation_seeds
        if seed.object_id.startswith("__interaction_envelope_")
    ]
    manifest = {
        "schema_version": "persistent_mask_sequence_v1",
        "status": "ready",
        "video": str(args.video.resolve()),
        "frame_window": [args.frame_start, frame_end],
        "frame_shape": [height, width],
        "source_registry": str(args.registry.resolve()),
        "box_observations": (
            None if args.box_observations is None else str(args.box_observations.resolve())
        ),
        "object_ids": object_ids,
        "objects": objects,
        "frames": frames,
        "overlap_pixels_before_ownership": overlap_before,
        "overlap_pixels_after_ownership": overlap_after,
        "accepted_overlap_pixels": accepted_overlap_pixels,
        "instance_identity_conflicts": identity_conflicts_payload,
        "expansion_decomposition_groups": decomposition_groups_payload,
        "expansion_decomposition_audit": decomposition_audit_payload,
        "frozen_intervals": frozen_intervals_payload,
        "reacquisition_anchors": reacquisition_anchors,
        "temporal_candidate_tracks": temporal_candidate_tracks,
        "private_interaction_envelopes": private_envelope_audit,
        "failures": [],
        "invariant": (
            "object IDs are fixed; confirmed expansion groups are decomposed as "
            "locked known components plus a disjoint new-component residual; partial "
            "occlusions freeze identity state without updating it, and visible masks "
            "remain empty until bounded temporal validation confirms the same IDs"
        ),
    }
    # Keep the frame-level diagnostic without using aggregate acceptance counts
    # to certify or suppress the generated sequence.
    diagnostic_path = output_dir / "diagnostic_mask_sequence.json"
    write_json_atomic(diagnostic_path, manifest)

    manifest_path = output_dir / "mask_sequence.json"
    write_json_atomic(manifest_path, manifest)
    summary = {
        "schema_version": "persistent_mask_sequence_run_v1",
        "status": "success",
        "video": str(args.video.resolve()),
        "frame_window": [args.frame_start, frame_end],
        "objects": objects,
        "instance_identity_conflicts": identity_conflicts_payload,
        "expansion_decomposition_groups": decomposition_groups_payload,
        "frozen_intervals": frozen_intervals_payload,
        "reacquisition_anchors": reacquisition_anchors,
        "temporal_candidate_tracks": temporal_candidate_tracks,
        "private_interaction_envelopes": private_envelope_audit,
        "failures": [],
        "mask_sequence": str(manifest_path.resolve()),
        "diagnostic_mask_sequence": str(diagnostic_path.resolve()),
        "failed_videos": [],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-area-pixels", type=int, default=256)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.9)
    parser.add_argument("--max-frame-area-fraction", type=float, default=0.35)
    parser.add_argument("--max-border-area-fraction", type=float, default=0.15)
    parser.add_argument("--min-area-ratio-vs-history", type=float, default=0.2)
    parser.add_argument("--max-area-ratio-vs-history", type=float, default=4.0)
    parser.add_argument("--max-centroid-step-diagonals", type=float, default=2.5)
    parser.add_argument("--history-size", type=int, default=7)
    parser.add_argument("--independent-object-states", action="store_true")
    parser.add_argument("--max-cross-instance-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--min-locked-component-area-retention", type=float, default=0.5)
    parser.add_argument("--max-locked-component-area-growth", type=float, default=1.35)
    parser.add_argument("--box-observations", type=Path)
    parser.add_argument("--interaction-box-margin-fraction", type=float, default=0.08)
    parser.add_argument("--max-interaction-box-fallback-gap-frames", type=int, default=3)
    parser.add_argument("--max-temporal-validation-seconds", type=float, default=1.0)
    parser.add_argument("--min-temporal-confirmation-frames", type=int, default=3)
    parser.add_argument("--min-flow-warp-continuity", type=float, default=0.50)
    parser.add_argument("--min-cycle-iou", type=float, default=0.60)
    parser.add_argument(
        "--max-compensated-centroid-step-diagonals", type=float, default=0.50
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        summary = {
            "schema_version": "persistent_mask_sequence_run_v1",
            "status": "failed_fatal_error",
            "video": str(args.video.resolve()),
            "failures": [{"reason": f"{type(exc).__name__}: {exc}"}],
            "mask_sequence": None,
            "failed_videos": [str(args.video.resolve())],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output_dir / "fatal_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed_videos"]:
        print("FAILED VIDEOS:")
        for video in summary["failed_videos"]:
            print(video)
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
