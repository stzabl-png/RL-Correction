"""Visible-mask ownership rules for persistent video object instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class OwnershipResult:
    """Non-overlapping visible masks and audit diagnostics for one frame."""

    masks: dict[str, np.ndarray]
    overlap_pixels_before: int
    overlap_pixels_after: int
    protected_core_pixels: dict[str, int]


def _validated_stack(mask_logits: Mapping[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    if not mask_logits:
        raise ValueError("mask_logits must contain at least one object")
    object_ids = list(mask_logits)
    arrays = [np.asarray(mask_logits[object_id], dtype=np.float32) for object_id in object_ids]
    shape = arrays[0].shape
    if len(shape) != 2:
        raise ValueError(f"mask logits must be 2D, got {shape}")
    for object_id, array in zip(object_ids, arrays):
        if array.shape != shape:
            raise ValueError(f"mask shape mismatch for {object_id}: {array.shape} != {shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"mask logits for {object_id} contain non-finite values")
    return object_ids, np.stack(arrays, axis=0)


def _eroded_core(mask: np.ndarray, erosion_pixels: int) -> np.ndarray:
    if erosion_pixels < 0:
        raise ValueError("erosion_pixels must be non-negative")
    if erosion_pixels == 0:
        return mask.astype(bool, copy=True)
    kernel_size = 2 * erosion_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0


def resolve_visible_mask_ownership(
    mask_logits: Mapping[str, np.ndarray],
    *,
    protected_object_ids: Iterable[str] = (),
    protected_core_erosion_pixels: int = 1,
) -> OwnershipResult:
    """Partition visible foreground pixels among persistent object IDs.

    All foreground pixels normally go to the object with the highest SAM2 logit.
    A confirmed object's eroded core is protected from tentative/new objects, so
    a newly proposed composite mask cannot swallow the stable interior of an
    existing small part.  Boundary pixels still use the model logits.
    """

    object_ids, logits = _validated_stack(mask_logits)
    protected_ids = set(protected_object_ids)
    unknown_ids = sorted(protected_ids.difference(object_ids))
    if unknown_ids:
        raise KeyError(f"protected object ids are missing from logits: {unknown_ids}")

    foreground = logits > 0.0
    overlap_pixels_before = int(np.count_nonzero(foreground.sum(axis=0) > 1))
    winning_index = np.argmax(logits, axis=0)
    winning_score = np.max(logits, axis=0)
    owner = np.where(winning_score > 0.0, winning_index, -1)

    protected_core_pixels: dict[str, int] = {}
    if protected_ids:
        protected_core_stack = np.zeros_like(foreground)
        for index, object_id in enumerate(object_ids):
            if object_id not in protected_ids:
                continue
            core = _eroded_core(foreground[index], protected_core_erosion_pixels)
            protected_core_stack[index] = core
            protected_core_pixels[object_id] = int(np.count_nonzero(core))

        has_protected_claim = protected_core_stack.any(axis=0)
        protected_scores = np.where(protected_core_stack, logits, -np.inf)
        protected_winner = np.argmax(protected_scores, axis=0)
        owner[has_protected_claim] = protected_winner[has_protected_claim]

    masks = {
        object_id: (owner == index)
        for index, object_id in enumerate(object_ids)
    }
    overlap_pixels_after = int(
        np.count_nonzero(np.stack(list(masks.values()), axis=0).sum(axis=0) > 1)
    )
    return OwnershipResult(
        masks=masks,
        overlap_pixels_before=overlap_pixels_before,
        overlap_pixels_after=overlap_pixels_after,
        protected_core_pixels=protected_core_pixels,
    )


def relation_instance_pairs(
    relations: Iterable[Mapping[str, Any]],
    *,
    known_object_ids: Iterable[str],
) -> list[tuple[str, str]]:
    """Return unordered instance pairs that a relation explicitly connects."""

    known = set(known_object_ids)
    pairs: set[tuple[str, str]] = set()
    for relation in relations:
        relation_ids = []
        for key, value in relation.items():
            if key.endswith("_object_id") and isinstance(value, str) and value in known:
                relation_ids.append(value)
            elif (key == "object_ids" or key.endswith("_object_ids")) and isinstance(value, list):
                relation_ids.extend(item for item in value if isinstance(item, str) and item in known)
        for index, first_id in enumerate(relation_ids):
            for second_id in relation_ids[index + 1 :]:
                if first_id != second_id:
                    pairs.add(tuple(sorted((first_id, second_id))))
    return sorted(pairs)


def directed_visible_occlusion_pairs(
    relations: Iterable[Mapping[str, Any]],
    *,
    known_object_ids: Iterable[str],
) -> list[tuple[str, str]]:
    """Read optional front/back ownership policies without assuming relation kind.

    A relation may supply ``visible_foreground_object_id`` and
    ``visible_background_object_id``.  The legacy ``supports`` shape is read
    only for backward compatibility: its part is foreground and its support is
    background.  All other relation kinds remain symmetric unless they provide
    the explicit visual policy.
    """

    known = set(known_object_ids)
    pairs: list[tuple[str, str]] = []
    for relation in relations:
        foreground_id = relation.get("visible_foreground_object_id")
        background_id = relation.get("visible_background_object_id")
        if foreground_id is None and background_id is None and relation.get("kind") == "supports":
            foreground_id = relation.get("part_object_id")
            background_id = relation.get("support_object_id")
        if foreground_id is None and background_id is None:
            continue
        if not isinstance(foreground_id, str) or not isinstance(background_id, str):
            raise ValueError("visible occlusion relation must provide both object IDs")
        if foreground_id == background_id or {foreground_id, background_id}.difference(known):
            raise ValueError("visible occlusion relation references unknown or identical objects")
        pair = (foreground_id, background_id)
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def resolve_directed_visible_ownership(
    mask_logits: Mapping[str, np.ndarray],
    *,
    foreground_background_pairs: Iterable[tuple[str, str]],
) -> OwnershipResult:
    """Resolve independent masks under optional foreground/background occlusion."""

    object_ids, logits = _validated_stack(mask_logits)
    index_by_id = {object_id: index for index, object_id in enumerate(object_ids)}
    masks = {
        object_id: logits[index] > 0.0
        for index, object_id in enumerate(object_ids)
    }
    overlap_before = int(
        np.count_nonzero(np.stack(list(masks.values()), axis=0).sum(axis=0) > 1)
    )
    relation_pairs = list(foreground_background_pairs)
    for foreground_id, background_id in relation_pairs:
        if foreground_id not in masks or background_id not in masks:
            raise KeyError(
                "relation ownership references unknown objects: "
                f"{(foreground_id, background_id)}"
            )
        masks[background_id] &= ~masks[foreground_id]

    # Any overlap not explained by an explicit occlusion relation remains a
    # normal logit competition; this keeps the output globally disjoint.
    stack = np.stack([masks[object_id] for object_id in object_ids], axis=0)
    remaining_overlap = stack.sum(axis=0) > 1
    if np.any(remaining_overlap):
        valid_logits = np.where(stack, logits, -np.inf)
        winners = np.argmax(valid_logits, axis=0)
        for index, object_id in enumerate(object_ids):
            masks[object_id][remaining_overlap] = (
                winners[remaining_overlap] == index
            )
    overlap_after = int(
        np.count_nonzero(np.stack(list(masks.values()), axis=0).sum(axis=0) > 1)
    )
    return OwnershipResult(
        masks=masks,
        overlap_pixels_before=overlap_before,
        overlap_pixels_after=overlap_after,
        protected_core_pixels={},
    )


def resolve_relation_mask_ownership(
    mask_logits: Mapping[str, np.ndarray],
    *,
    part_support_pairs: Iterable[tuple[str, str]],
) -> OwnershipResult:
    """Backward-compatible wrapper for the old part/support terminology."""

    return resolve_directed_visible_ownership(
        mask_logits,
        foreground_background_pairs=part_support_pairs,
    )


def composite_residual_mask(
    composite_mask: np.ndarray,
    confirmed_masks: Mapping[str, np.ndarray],
    *,
    exclusion_dilation_pixels: int = 0,
    min_component_area: int = 64,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Remove confirmed instances from a composite candidate mask.

    The returned residual is suitable as a *tentative* new-instance prompt.  It
    is not automatically promoted to a persistent ID; temporal confirmation is
    deliberately left to the instance registry.
    """

    composite = np.asarray(composite_mask, dtype=bool)
    if composite.ndim != 2:
        raise ValueError(f"composite_mask must be 2D, got {composite.shape}")
    if exclusion_dilation_pixels < 0:
        raise ValueError("exclusion_dilation_pixels must be non-negative")
    if min_component_area < 1:
        raise ValueError("min_component_area must be positive")

    exclusion = np.zeros_like(composite)
    for object_id, mask in confirmed_masks.items():
        array = np.asarray(mask, dtype=bool)
        if array.shape != composite.shape:
            raise ValueError(f"confirmed mask shape mismatch for {object_id}: {array.shape} != {composite.shape}")
        exclusion |= array
    if exclusion_dilation_pixels:
        kernel_size = 2 * exclusion_dilation_pixels + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        exclusion = cv2.dilate(exclusion.astype(np.uint8), kernel, iterations=1) > 0

    raw_residual = composite & ~exclusion
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_residual.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(raw_residual)
    kept_components = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            kept[labels == label] = True
            kept_components += 1

    composite_area = int(np.count_nonzero(composite))
    raw_area = int(np.count_nonzero(raw_residual))
    kept_area = int(np.count_nonzero(kept))
    diagnostics: dict[str, int | float] = {
        "composite_area": composite_area,
        "excluded_confirmed_area": int(np.count_nonzero(composite & exclusion)),
        "raw_residual_area": raw_area,
        "kept_residual_area": kept_area,
        "kept_components": kept_components,
        "kept_fraction_of_composite": kept_area / composite_area if composite_area else 0.0,
    }
    return kept, diagnostics
