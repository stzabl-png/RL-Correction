"""Register a new visible component after a confirmed HOI box expansion.

No semantic relation is inferred here.  A new ID is permitted only for pixels
inside the expanded interaction envelope that cannot be explained by already
registered visible-instance masks.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from .adapter import write_json_atomic
from .instance_association import InstanceCandidate


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise RuntimeError(f"failed to write mask: {path}")


def _box_region(shape: tuple[int, int], box_xyxy: list[float]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = [int(round(value)) for value in box_xyxy]
    region = np.zeros(shape, dtype=bool)
    region[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
    return region


def read_amg_candidates(summary_path: Path) -> list[InstanceCandidate]:
    """Load SAM2 automatic-mask candidates while preserving their quality score."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "success":
        return []
    candidates = []
    for record in summary.get("candidates", []):
        image = cv2.imread(str(record["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"cannot read AMG candidate mask: {record['mask']}")
        candidates.append(
            InstanceCandidate(
                candidate_id=f"candidate_{int(record['candidate_idx']):03d}",
                mask=image > 0,
                quality_score=float(record["predicted_iou"]) * float(record["stability_score"]),
                source="sam2_amg_at_confirmed_hoi_box_expansion",
            )
        )
    return candidates


def _largest_component_fraction(mask: np.ndarray) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0.0
    area = int(np.count_nonzero(mask))
    return float(stats[1:, cv2.CC_STAT_AREA].max() / area) if area else 0.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def lock_known_components_from_logits(
    reference_masks: dict[str, np.ndarray],
    propagated_logits: dict[str, np.ndarray],
    *,
    area_reference_pixels: dict[str, int] | None = None,
    min_area_retention: float = 0.5,
    max_area_growth: float = 1.35,
) -> dict:
    """Prevent a known component track from absorbing newly assembled pixels.

    SAM2 can expand a previously small part into an assembled whole.  The
    short-range logits are still useful for locating the part, but their
    positive support is capped by the last reliable pre-expansion area.  The
    highest-confidence connected support becomes the locked known component.
    """

    if not 0.0 < min_area_retention <= 1.0 or max_area_growth < 1.0:
        raise ValueError("invalid known-component lock thresholds")
    if set(reference_masks) != set(propagated_logits):
        raise ValueError("reference masks and propagated logits must have identical IDs")
    if not reference_masks:
        raise ValueError("at least one known component is required")
    if area_reference_pixels is not None:
        if set(area_reference_pixels) != set(reference_masks):
            raise ValueError("area references and component masks must have identical IDs")
        if any(int(area) < 1 for area in area_reference_pixels.values()):
            raise ValueError("known-component area references must be positive")

    locked = {}
    metrics = {}
    shape = np.asarray(next(iter(reference_masks.values())), dtype=bool).shape
    for object_id, reference in reference_masks.items():
        reference = np.asarray(reference, dtype=bool)
        logits = np.asarray(propagated_logits[object_id], dtype=np.float32)
        if reference.shape != shape or logits.shape != shape:
            raise ValueError("known-component lock inputs must share one 2D shape")
        spatial_reference_area = int(np.count_nonzero(reference))
        if not spatial_reference_area:
            raise ValueError(f"empty reference mask for {object_id}")
        area_reference = (
            spatial_reference_area
            if area_reference_pixels is None
            else int(area_reference_pixels[object_id])
        )
        positive = logits > 0.0
        raw_area = int(np.count_nonzero(positive))
        min_area = max(1, int(round(area_reference * min_area_retention)))
        max_area = max(min_area, int(round(area_reference * max_area_growth)))
        if raw_area < min_area:
            return {
                "status": "failed_known_component_lost_at_expansion",
                "object_id": object_id,
                "reference_area": spatial_reference_area,
                "area_reference_pixels": area_reference,
                "raw_area": raw_area,
                "min_area": min_area,
            }
        constrained = positive
        was_area_capped = raw_area > max_area
        if was_area_capped:
            flat_logits = logits.reshape(-1)
            positive_indices = np.flatnonzero(positive.reshape(-1))
            keep_count = min(max_area, len(positive_indices))
            ranked = positive_indices[
                np.argpartition(flat_logits[positive_indices], -keep_count)[-keep_count:]
            ]
            constrained = np.zeros(logits.size, dtype=bool)
            constrained[ranked] = True
            constrained = constrained.reshape(shape)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            constrained.astype(np.uint8), 8
        )
        ys, xs = np.nonzero(reference)
        reference_diagonal = float(
            np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        )
        search_radius = max(3, int(round(reference_diagonal * 0.25)))
        search_region = cv2.dilate(
            reference.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * search_radius + 1, 2 * search_radius + 1),
            ),
        ) > 0
        components = []
        for label in range(1, count):
            component_mask = labels == label
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            overlap_pixels = int(np.count_nonzero(component_mask & search_region))
            if not overlap_pixels:
                continue
            components.append(
                (
                    overlap_pixels / component_area,
                    float(np.mean(logits[component_mask])),
                    component_area,
                    label,
                )
            )
        components.sort(reverse=True)
        component = np.zeros(shape, dtype=bool)
        for _, _, _, label in components:
            component |= labels == label
            if int(np.count_nonzero(component)) >= min_area:
                break
        locked_area = int(np.count_nonzero(component))
        if locked_area < min_area:
            return {
                "status": "failed_known_component_fragmented_at_expansion",
                "object_id": object_id,
                "reference_area": spatial_reference_area,
                "area_reference_pixels": area_reference,
                "raw_area": raw_area,
                "locked_area": locked_area,
                "min_area": min_area,
            }
        locked[object_id] = component
        metrics[object_id] = {
            "reference_area": spatial_reference_area,
            "area_reference_pixels": area_reference,
            "raw_positive_area": raw_area,
            "locked_area": locked_area,
            "locked_area_ratio_vs_reference": locked_area / spatial_reference_area,
            "locked_area_ratio_vs_area_reference": locked_area / area_reference,
            "area_cap_applied": was_area_capped,
            "retained_component_count": int(
                cv2.connectedComponents(component.astype(np.uint8), 8)[0] - 1
            ),
            "reference_search_radius_pixels": search_radius,
        }

    # Different known IDs remain exclusive even if their short-range logits
    # overlap.  Assign contested pixels to the larger logit only.
    object_ids = list(locked)
    if len(object_ids) > 1:
        score_stack = np.stack(
            [
                np.where(locked[object_id], propagated_logits[object_id], -np.inf)
                for object_id in object_ids
            ],
            axis=0,
        )
        owners = np.argmax(score_stack, axis=0)
        any_owner = np.isfinite(score_stack).any(axis=0)
        for index, object_id in enumerate(object_ids):
            locked[object_id] = any_owner & (owners == index)
            if int(np.count_nonzero(locked[object_id])) < int(
                round(metrics[object_id]["area_reference_pixels"] * min_area_retention)
            ):
                return {
                    "status": "failed_known_component_overlap_at_expansion",
                    "object_id": object_id,
                    "metrics": metrics,
                }

    return {"status": "success", "masks": locked, "metrics": metrics}


def select_unexplained_component(
    candidates: list[InstanceCandidate],
    *,
    existing_masks: dict[str, np.ndarray],
    interaction_box_xyxy: list[float],
    min_area_pixels: int,
    min_roi_fraction: float = 0.5,
    min_existing_coverage_for_residual: float = 0.85,
    max_direct_overlap_fraction: float = 0.05,
    min_largest_component_fraction: float = 0.9,
    min_score_margin: float = 0.05,
) -> dict:
    """Choose one stable, disjoint new mask from direct or residual proposals."""

    if not candidates:
        return {"status": "failed_no_expansion_candidates"}
    if not existing_masks:
        raise ValueError("at least one existing visible component is required")
    shape = np.asarray(next(iter(existing_masks.values())), dtype=bool).shape
    if len(shape) != 2:
        raise ValueError("existing masks must be 2D")
    if any(np.asarray(mask, dtype=bool).shape != shape for mask in existing_masks.values()):
        raise ValueError("existing masks must share one shape")
    if min_area_pixels < 1 or not 0.0 <= min_roi_fraction <= 1.0:
        raise ValueError("invalid expansion candidate thresholds")

    known_union = np.zeros(shape, dtype=bool)
    for mask in existing_masks.values():
        known_union |= np.asarray(mask, dtype=bool)
    known_area = int(np.count_nonzero(known_union))
    if not known_area:
        raise ValueError("existing masks must be non-empty")
    roi = _box_region(shape, interaction_box_xyxy)
    proposals = []
    for candidate in candidates:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError("candidate mask shape differs from existing masks")
        area = int(np.count_nonzero(mask))
        if not area:
            continue
        roi_fraction = int(np.count_nonzero(mask & roi)) / area
        if roi_fraction < min_roi_fraction or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            continue
        known_overlap = int(np.count_nonzero(mask & known_union))
        known_coverage = known_overlap / known_area
        direct_overlap = known_overlap / area
        if known_coverage >= min_existing_coverage_for_residual:
            new_mask = mask & ~known_union
            proposal_type = "composite_residual"
        elif direct_overlap <= max_direct_overlap_fraction:
            new_mask = mask.copy()
            proposal_type = "direct_disjoint_candidate"
        else:
            continue
        new_area = int(np.count_nonzero(new_mask))
        if new_area < min_area_pixels:
            continue
        largest_fraction = _largest_component_fraction(new_mask)
        if largest_fraction < min_largest_component_fraction:
            continue
        proposals.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_source": candidate.source,
                "proposal_type": proposal_type,
                "mask": new_mask,
                "area": new_area,
                "roi_fraction": roi_fraction,
                "known_mask_coverage": known_coverage,
                "direct_overlap_fraction": direct_overlap,
                "largest_component_fraction": largest_fraction,
                "score": candidate.quality_score * roi_fraction * largest_fraction,
                "candidate_quality_score": candidate.quality_score,
            }
        )
    if not proposals:
        return {"status": "failed_no_unexplained_visible_component"}
    proposals.sort(key=lambda item: (-item["score"], item["candidate_id"], item["proposal_type"]))
    best = proposals[0]
    second = proposals[1] if len(proposals) > 1 else None
    if second is not None and best["score"] - second["score"] < min_score_margin:
        return {
            "status": "failed_ambiguous_unexplained_component",
            "best_score": best["score"],
            "second_score": second["score"],
            "proposals": [{key: value for key, value in item.items() if key != "mask"} for item in proposals],
        }
    return {
        "status": "success",
        "selected": best,
        "proposals": [{key: value for key, value in item.items() if key != "mask"} for item in proposals],
    }


def append_expansion_component(
    source_registry_path: Path,
    *,
    existing_masks: dict[str, np.ndarray],
    event: dict,
    selection: dict,
    output_dir: Path,
    known_component_area_references: dict[str, int] | None = None,
    interaction_envelope_mask: np.ndarray | None = None,
    conditioning_anchors: list[dict] | None = None,
) -> dict:
    """Create an augmented immutable registry with one newly visible component."""

    if selection.get("status") != "success":
        raise ValueError("only a successful component selection can be registered")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    if registry.get("status") != "ready":
        raise ValueError("source registry is not ready")
    if set(existing_masks) != set(registry.get("objects", {})):
        raise ValueError("existing masks must cover exactly the registered IDs")
    if known_component_area_references is None:
        known_component_area_references = {
            object_id: int(np.count_nonzero(mask))
            for object_id, mask in existing_masks.items()
        }
    if set(known_component_area_references) != set(existing_masks):
        raise ValueError("known component area references must cover registered IDs")
    if any(int(area) < 1 for area in known_component_area_references.values()):
        raise ValueError("known component area references must be positive")
    activation_frame = int(event["activation_frame"])
    seed_frame = int(event["seed_frame"])
    if seed_frame < activation_frame:
        raise ValueError("expansion seed frame precedes activation frame")
    conditioning_anchors = conditioning_anchors or []
    anchor_by_frame = {}
    for anchor in conditioning_anchors:
        anchor_frame = int(anchor["frame_idx"])
        if anchor_frame < activation_frame:
            raise ValueError("component anchor precedes activation frame")
        if anchor_frame in anchor_by_frame:
            raise ValueError("component anchors must use unique frames")
        anchor_existing = {
            object_id: np.asarray(mask, dtype=bool)
            for object_id, mask in anchor["existing_masks"].items()
        }
        if set(anchor_existing) != set(existing_masks):
            raise ValueError("component anchor known masks must cover registered IDs")
        anchor_new = np.asarray(anchor["new_mask"], dtype=bool)
        if anchor_new.shape != np.asarray(next(iter(existing_masks.values()))).shape:
            raise ValueError("component anchor masks must share the registered shape")
        known_union = np.zeros(anchor_new.shape, dtype=bool)
        for mask in anchor_existing.values():
            if mask.shape != anchor_new.shape or not np.any(mask):
                raise ValueError("component anchor known masks must be non-empty and aligned")
            known_union |= mask
        if not np.any(anchor_new) or np.any(anchor_new & known_union):
            raise ValueError("component anchor new mask must be non-empty and disjoint")
        anchor_by_frame[anchor_frame] = {
            **anchor,
            "existing_masks": anchor_existing,
            "new_mask": anchor_new,
        }
    revised = deepcopy(registry)
    conditioning_dir = output_dir / "conditioning_masks"
    known_conditions = {
        seed_frame: {
            object_id: np.asarray(mask, dtype=bool)
            for object_id, mask in existing_masks.items()
        },
        **{
            frame_idx: anchor["existing_masks"]
            for frame_idx, anchor in anchor_by_frame.items()
        },
    }
    for condition_frame, frame_masks in sorted(known_conditions.items()):
        for object_id, mask in frame_masks.items():
            entry = revised["objects"][object_id]
            if any(
                int(item["frame_idx"]) == condition_frame
                for item in entry.get("conditioning_masks", [])
            ):
                continue
            path = conditioning_dir / f"{object_id}_{condition_frame:06d}.png"
            _write_mask(path, mask)
            entry.setdefault("conditioning_masks", []).append(
                {
                    "frame_idx": condition_frame,
                    "mask": str(path.resolve()),
                    "source": "causal_verified_existing_mask_at_all_box_anchor",
                }
            )
    next_index = max((int(object_id.rsplit("_", 1)[-1]) for object_id in revised["objects"]), default=0) + 1
    new_object_id = f"instance_{next_index:04d}"
    new_mask = np.asarray(selection["selected"]["mask"], dtype=bool)
    new_path = conditioning_dir / f"{new_object_id}_{seed_frame:06d}.png"
    _write_mask(new_path, new_mask)
    selected = selection["selected"]
    new_conditions = {
        seed_frame: {
            "mask": new_mask,
            "candidate_id": selected["candidate_id"],
            "candidate_source": selected["candidate_source"],
            "candidate_quality_score": selected["candidate_quality_score"],
            "proposal_type": selected["proposal_type"],
        },
        **{
            frame_idx: {
                "mask": anchor["new_mask"],
                "candidate_id": anchor["candidate_id"],
                "candidate_source": anchor["candidate_source"],
                "candidate_quality_score": anchor["candidate_quality_score"],
                "proposal_type": anchor["proposal_type"],
            }
            for frame_idx, anchor in anchor_by_frame.items()
        },
    }
    new_conditioning_records = []
    for condition_frame, condition in sorted(new_conditions.items()):
        path = conditioning_dir / f"{new_object_id}_{condition_frame:06d}.png"
        _write_mask(path, condition["mask"])
        new_conditioning_records.append(
            {
                "frame_idx": condition_frame,
                "mask": str(path.resolve()),
                "source": "unexplained_visible_component_at_all_box_anchor",
                "candidate_id": condition["candidate_id"],
                "candidate_source": condition["candidate_source"],
                "candidate_quality_score": condition["candidate_quality_score"],
                "proposal_type": condition["proposal_type"],
            }
        )
    revised["objects"][new_object_id] = {
        "activation_frame": activation_frame,
        "frame_idx": seed_frame,
        "mask": str(new_path.resolve()),
        "conditioning_masks": new_conditioning_records,
        "instance_source": "unexplained_visible_component_at_confirmed_hoi_box_expansion",
        "source_candidate_id": selected["candidate_id"],
    }
    revised.setdefault("object_order", list(registry["objects"])).append(new_object_id)
    expansion_index = len(revised.get("confirmed_hoi_box_expansions", []))
    envelope_record = None
    if interaction_envelope_mask is not None:
        envelope_mask = np.asarray(interaction_envelope_mask, dtype=bool)
        if envelope_mask.shape != new_mask.shape or not np.any(envelope_mask):
            raise ValueError("interaction envelope mask must be a non-empty registered-mask shape")
        envelope_object_id = f"__interaction_envelope_{expansion_index:04d}"
        envelope_path = (
            conditioning_dir / f"{envelope_object_id}_{seed_frame:06d}.png"
        )
        _write_mask(envelope_path, envelope_mask)
        envelope_record = {
            "object_id": envelope_object_id,
            "frame_idx": seed_frame,
            "mask": str(envelope_path.resolve()),
            "source": "sam2_confirmed_expanded_hoi_box_prompt",
            "public_instance": False,
        }
    revised.setdefault("confirmed_hoi_box_expansions", []).append(
        {
            "event": event,
            "known_object_ids": list(existing_masks),
            "known_component_area_references": {
                object_id: int(known_component_area_references[object_id])
                for object_id in existing_masks
            },
            "new_object_id": new_object_id,
            "interaction_envelope": envelope_record,
            "conditioning_anchor_frames": sorted(new_conditions),
            "selection": {key: value for key, value in selected.items() if key != "mask"},
        }
    )
    registry_path = output_dir / "reconstruction_registry.json"
    write_json_atomic(registry_path, revised)
    return {
        "status": "success",
        "new_object_id": new_object_id,
        "interaction_envelope_object_id": (
            None if envelope_record is None else envelope_record["object_id"]
        ),
        "conditioning_anchor_frames": sorted(new_conditions),
        "reconstruction_registry": str(registry_path.resolve()),
        "selection": {key: value for key, value in selected.items() if key != "mask"},
    }
