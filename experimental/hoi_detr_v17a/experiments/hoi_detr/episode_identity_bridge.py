"""Bridge hidden inter-episode gaps before reusing a global instance ID."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .identity_linking import GlobalInstanceObservation
from .sam2_multi_object import ObjectMaskSeed, propagate_independent_object_logits


@dataclass(frozen=True)
class BridgeSource:
    """Last accepted visible mask for one global ID before a hidden gap."""

    global_object_id: str
    frame_idx: int
    mask: np.ndarray


def bridge_hidden_gap(
    predictor,
    *,
    video_path: Path,
    sources: Iterable[BridgeSource],
    target_frame: int,
    max_gap_frames: int,
) -> list[GlobalInstanceObservation]:
    """Propagate prior IDs internally to the next episode's first visible frame.

    Bridge masks are used only for visual identity matching.  They are never
    written to the public mask sequence, whose inter-episode frames stay empty.
    """

    items = list(sources)
    if not items:
        raise ValueError("at least one bridge source is required")
    if target_frame < 0 or max_gap_frames < 0:
        raise ValueError("target_frame and max_gap_frames must be non-negative")
    if len({item.global_object_id for item in items}) != len(items):
        raise ValueError("bridge source global IDs must be unique")
    earliest_source = min(item.frame_idx for item in items)
    if earliest_source < 0 or target_frame <= earliest_source:
        raise ValueError("bridge target must follow every source frame")
    if target_frame - earliest_source > max_gap_frames:
        raise ValueError(
            f"hidden gap {target_frame - earliest_source} exceeds {max_gap_frames} frames"
        )

    propagation = propagate_independent_object_logits(
        predictor,
        video_path=video_path,
        seeds=[
            ObjectMaskSeed(item.global_object_id, item.frame_idx, item.mask)
            for item in items
        ],
        init_state_kwargs={"offload_video_to_cpu": True},
        frame_start=earliest_source,
        frame_end=target_frame,
    )
    target_logits = propagation.logits_by_frame.get(target_frame, {})
    missing = [item.global_object_id for item in items if item.global_object_id not in target_logits]
    if missing:
        raise RuntimeError(f"bridge propagation produced no target mask for {missing}")
    observations = []
    for item in items:
        mask = np.asarray(target_logits[item.global_object_id]) > 0.0
        if not np.any(mask):
            raise RuntimeError(f"bridge propagation produced an empty mask for {item.global_object_id}")
        observations.append(GlobalInstanceObservation(item.global_object_id, mask))
    return observations
