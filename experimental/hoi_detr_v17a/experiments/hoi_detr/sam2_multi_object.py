"""Shared-state, bidirectional SAM2 propagation for persistent object IDs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ObjectMaskSeed:
    """One clean visible-instance mask used to initialize a persistent ID."""

    object_id: str
    frame_idx: int
    mask: np.ndarray
    protected: bool = True


@dataclass(frozen=True)
class MultiObjectPropagationResult:
    """Raw SAM2 logits indexed by absolute frame and logical object ID."""

    logits_by_frame: dict[int, dict[str, np.ndarray]]
    sam_object_ids: dict[str, int]
    seed_frames: dict[str, int]
    conditioning_frames: dict[str, tuple[int, ...]]


def _validate_seeds(seeds: Iterable[ObjectMaskSeed]) -> list[ObjectMaskSeed]:
    normalized = list(seeds)
    if not normalized:
        raise ValueError("at least one object seed is required")
    object_ids = [seed.object_id for seed in normalized]
    if any(not object_id for object_id in object_ids):
        raise ValueError("object_id must be non-empty")
    conditioning_keys = [(seed.object_id, seed.frame_idx) for seed in normalized]
    if len(set(conditioning_keys)) != len(conditioning_keys):
        raise ValueError(f"duplicate object conditioning frame: {conditioning_keys}")
    shape = np.asarray(normalized[0].mask).shape
    if len(shape) != 2:
        raise ValueError(f"seed masks must be 2D, got {shape}")
    for seed in normalized:
        mask = np.asarray(seed.mask)
        if seed.frame_idx < 0:
            raise ValueError(f"negative seed frame for {seed.object_id}: {seed.frame_idx}")
        if mask.shape != shape:
            raise ValueError(f"seed mask shape mismatch for {seed.object_id}: {mask.shape} != {shape}")
        if not np.any(mask):
            raise ValueError(f"seed mask is empty for {seed.object_id}")
    return sorted(normalized, key=lambda seed: (seed.frame_idx, seed.object_id))


def _to_numpy_2d(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"SAM2 mask logit must reduce to 2D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("SAM2 mask logit contains non-finite values")
    return array.copy()


def _collect_direction(
    predictor,
    state,
    *,
    start_frame_idx: int,
    reverse: bool,
    logical_id_by_sam_id: dict[int, str],
    max_frame_num_to_track: int | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    collected: dict[int, dict[str, np.ndarray]] = {}
    propagation_kwargs = {
        "start_frame_idx": start_frame_idx,
        "reverse": reverse,
    }
    if max_frame_num_to_track is not None:
        propagation_kwargs["max_frame_num_to_track"] = max_frame_num_to_track
    for frame_idx, sam_object_ids, mask_logits in predictor.propagate_in_video(
        state,
        **propagation_kwargs,
    ):
        frame_outputs: dict[str, np.ndarray] = {}
        for index, raw_sam_id in enumerate(sam_object_ids):
            sam_id = int(raw_sam_id)
            logical_id = logical_id_by_sam_id.get(sam_id)
            if logical_id is None:
                raise KeyError(f"SAM2 returned unknown object id {sam_id}")
            frame_outputs[logical_id] = _to_numpy_2d(mask_logits[index])
        collected[int(frame_idx)] = frame_outputs
    return collected


def propagate_multi_object_logits(
    predictor,
    *,
    video_path: Path | str,
    seeds: Iterable[ObjectMaskSeed],
    init_state_kwargs: dict[str, Any] | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> MultiObjectPropagationResult:
    """Propagate all object seeds in one SAM2 inference state.

    Forward logits are used on and after an object's seed frame; reverse logits
    are used before it.  This reproduces per-object bidirectional prompting while
    allowing SAM2 to consolidate all objects and enforce non-overlap in memory.
    """

    seeds = _validate_seeds(seeds)
    if (frame_start is None) != (frame_end is None):
        raise ValueError("frame_start and frame_end must be supplied together")
    if frame_start is not None:
        if frame_start < 0 or frame_end < frame_start:
            raise ValueError(f"invalid propagation frame range: {frame_start}..{frame_end}")
        outside = [seed.object_id for seed in seeds if not frame_start <= seed.frame_idx <= frame_end]
        if outside:
            raise ValueError(f"seed frames outside propagation range: {outside}")
    state = predictor.init_state(
        video_path=str(Path(video_path)),
        **(init_state_kwargs or {}),
    )
    logical_object_ids = list(dict.fromkeys(seed.object_id for seed in seeds))
    sam_object_ids = {object_id: index + 1 for index, object_id in enumerate(logical_object_ids)}
    logical_id_by_sam_id = {value: key for key, value in sam_object_ids.items()}
    conditioning_frames = {
        object_id: tuple(sorted(seed.frame_idx for seed in seeds if seed.object_id == object_id))
        for object_id in logical_object_ids
    }
    seed_frames = {object_id: frames[0] for object_id, frames in conditioning_frames.items()}

    if hasattr(predictor, "non_overlap_masks"):
        predictor.non_overlap_masks = True
    if hasattr(predictor, "non_overlap_masks_for_mem_enc"):
        predictor.non_overlap_masks_for_mem_enc = True

    try:
        for seed in seeds:
            predictor.add_new_mask(
                state,
                frame_idx=seed.frame_idx,
                obj_id=sam_object_ids[seed.object_id],
                mask=np.asarray(seed.mask, dtype=bool),
            )

        forward_start = min(seed_frames.values())
        reverse_start = max(seed_frames.values())
        forward = _collect_direction(
            predictor,
            state,
            start_frame_idx=forward_start,
            reverse=False,
            logical_id_by_sam_id=logical_id_by_sam_id,
            max_frame_num_to_track=None if frame_end is None else frame_end - forward_start + 1,
        )
        reverse = _collect_direction(
            predictor,
            state,
            start_frame_idx=reverse_start,
            reverse=True,
            logical_id_by_sam_id=logical_id_by_sam_id,
            max_frame_num_to_track=None if frame_start is None else reverse_start - frame_start + 1,
        )

        all_frame_indices = sorted(set(forward).union(reverse))
        if frame_start is not None:
            all_frame_indices = [index for index in all_frame_indices if frame_start <= index <= frame_end]
        logits_by_frame: dict[int, dict[str, np.ndarray]] = {}
        for frame_idx in all_frame_indices:
            frame_outputs: dict[str, np.ndarray] = {}
            for object_id, seed_frame in seed_frames.items():
                preferred = forward if frame_idx >= seed_frame else reverse
                fallback = reverse if preferred is forward else forward
                logit = preferred.get(frame_idx, {}).get(object_id)
                if logit is None:
                    logit = fallback.get(frame_idx, {}).get(object_id)
                if logit is not None:
                    frame_outputs[object_id] = logit
            logits_by_frame[frame_idx] = frame_outputs
    finally:
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)

    return MultiObjectPropagationResult(
        logits_by_frame=logits_by_frame,
        sam_object_ids=sam_object_ids,
        seed_frames=seed_frames,
        conditioning_frames=conditioning_frames,
    )


def propagate_independent_object_logits(
    predictor,
    *,
    video_path: Path | str,
    seeds: Iterable[ObjectMaskSeed],
    init_state_kwargs: dict[str, Any] | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> MultiObjectPropagationResult:
    """Propagate each logical instance in an isolated SAM2 memory state."""

    normalized = _validate_seeds(seeds)
    object_ids = list(dict.fromkeys(seed.object_id for seed in normalized))
    combined_logits: dict[int, dict[str, np.ndarray]] = {}
    seed_frames: dict[str, int] = {}
    conditioning_frames: dict[str, tuple[int, ...]] = {}
    for object_id in object_ids:
        object_seeds = [seed for seed in normalized if seed.object_id == object_id]
        result = propagate_multi_object_logits(
            predictor,
            video_path=video_path,
            seeds=object_seeds,
            init_state_kwargs=init_state_kwargs,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        seed_frames[object_id] = result.seed_frames[object_id]
        conditioning_frames[object_id] = result.conditioning_frames[object_id]
        for frame_idx, frame_logits in result.logits_by_frame.items():
            if object_id in frame_logits:
                combined_logits.setdefault(frame_idx, {})[object_id] = frame_logits[object_id]
    return MultiObjectPropagationResult(
        logits_by_frame=combined_logits,
        sam_object_ids={object_id: index + 1 for index, object_id in enumerate(object_ids)},
        seed_frames=seed_frames,
        conditioning_frames=conditioning_frames,
    )


def object_seed_from_box(
    predictor,
    *,
    video_path: Path | str,
    object_id: str,
    frame_idx: int,
    box_xyxy: Iterable[float],
    init_state_kwargs: dict[str, Any] | None = None,
) -> tuple[ObjectMaskSeed, np.ndarray]:
    """Create a visible-instance seed from one accepted detector box."""

    box = np.asarray(list(box_xyxy), dtype=np.float32)
    if box.shape != (4,):
        raise ValueError(f"box_xyxy must contain four values, got {box.shape}")
    if not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"invalid xyxy box: {box.tolist()}")
    if frame_idx < 0:
        raise ValueError("frame_idx must be non-negative")

    state = predictor.init_state(
        video_path=str(Path(video_path)),
        **(init_state_kwargs or {}),
    )
    try:
        _, sam_object_ids, mask_logits = predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=1,
            box=box,
            clear_old_points=True,
        )
        matching_indices = [index for index, sam_id in enumerate(sam_object_ids) if int(sam_id) == 1]
        if len(matching_indices) != 1:
            raise RuntimeError(f"SAM2 returned unexpected object ids: {list(sam_object_ids)}")
        logits = _to_numpy_2d(mask_logits[matching_indices[0]])
        mask = logits > 0.0
        if not np.any(mask):
            raise RuntimeError(f"SAM2 produced an empty box-prompt mask for {object_id} at frame {frame_idx}")
        return ObjectMaskSeed(object_id=object_id, frame_idx=frame_idx, mask=mask), logits
    finally:
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(state)
