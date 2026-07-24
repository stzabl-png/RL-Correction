"""Combine per-cycle mask sequences under immutable video-wide object IDs."""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _object_role(object_id: str) -> str:
    """Return the stable role portion used by the cycle registry renaming scheme."""
    return re.sub(r"_\d+$", "", object_id)


def _load_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    return mask > 0


def _cycle_object_map(local_ids: list[str], global_ids: list[str]) -> dict[str, str]:
    if len(local_ids) != len(global_ids):
        raise ValueError("cycle sequence and video registry object counts differ")
    remaining = set(local_ids)
    mapping: dict[str, str] = {}
    for global_id in global_ids:
        matches = [item for item in remaining if _object_role(item) == _object_role(global_id)]
        if len(matches) != 1:
            raise ValueError(
                f"cannot uniquely map video object {global_id!r} to local IDs {local_ids!r}"
            )
        local_id = matches[0]
        mapping[local_id] = global_id
        remaining.remove(local_id)
    return mapping


def _registry_cycle_object_map(cycle: dict, local_ids: list[str]) -> dict[str, str]:
    explicit = cycle.get("source_object_id_map")
    if explicit is None:
        return _cycle_object_map(local_ids, cycle["object_ids"])
    mapping = {str(local_id): str(global_id) for local_id, global_id in explicit.items()}
    if set(mapping) != set(local_ids) or set(mapping.values()) != set(cycle["object_ids"]):
        raise ValueError("cycle source object ID map does not match sequence and registry IDs")
    return mapping


def combine_mask_sequences(
    video_registry: dict,
    sequence_manifests: list[dict],
) -> dict:
    """Map cycle-local IDs to global IDs and reject cross-ID mask conflicts.

    A conditioning mask is trusted over a propagated mask. If two propagated
    masks overlap, both are rejected. Two overlapping conditioning masks are
    rejected because the registry itself is inconsistent.
    """
    if video_registry.get("status") != "ready":
        raise ValueError("video reconstruction registry is not ready")
    cycles = sorted(video_registry.get("cycles", []), key=lambda item: item["cycle_idx"])
    if len(cycles) != len(sequence_manifests):
        raise ValueError("cycle sequence count does not match video registry")
    frames: dict[int, dict[str, dict]] = defaultdict(dict)
    object_sources: dict[str, list[dict]] = defaultdict(list)
    source_sequences = []
    frame_shape = None
    for cycle, manifest in zip(cycles, sequence_manifests):
        if manifest.get("status") != "ready":
            raise ValueError(f"cycle {cycle['cycle_idx']} mask sequence is not ready")
        if frame_shape is None:
            frame_shape = manifest.get("frame_shape")
        elif manifest.get("frame_shape") != frame_shape:
            raise ValueError("cycle mask sequence frame shapes differ")
        mapping = _registry_cycle_object_map(cycle, manifest["object_ids"])
        episode_activation_frame = int(
            cycle.get("interaction_onset_frame", manifest["frame_window"][0])
        )
        deactivation_frame = int(
            cycle.get("interaction_end_frame", manifest["frame_window"][1])
        )
        if not (
            manifest["frame_window"][0]
            <= episode_activation_frame
            <= deactivation_frame
            <= manifest["frame_window"][1]
        ):
            raise ValueError(
                f"cycle {cycle['cycle_idx']} interaction window is outside its mask sequence"
            )
        source_sequences.append(
            {
                "cycle_idx": cycle["cycle_idx"],
                "manifest": manifest.get("manifest_path"),
                "source_registry": manifest.get("source_registry"),
                "frame_window": manifest["frame_window"],
                "effective_frame_window": [episode_activation_frame, deactivation_frame],
                "object_id_map": mapping,
            }
        )
        cycle_sources: dict[str, dict] = {}
        for local_id, global_id in mapping.items():
            source_object = manifest["objects"][local_id]
            registry_object = video_registry.get("objects", {}).get(global_id, {})
            registry_observations = registry_object.get("cycle_observations", [])
            matching_observations = [
                observation
                for observation in registry_observations
                if int(observation["cycle_idx"]) == int(cycle["cycle_idx"])
                and observation["source_object_id"] == local_id
            ]
            registry_activation_frame = (
                matching_observations[0].get("activation_frame")
                if len(matching_observations) == 1
                else registry_object.get("activation_frame")
            )
            object_activation_frame = max(
                episode_activation_frame,
                int(
                    registry_activation_frame
                    if registry_activation_frame is not None
                    else source_object.get("activation_frame", episode_activation_frame)
                ),
            )
            source = {
                "cycle_idx": cycle["cycle_idx"],
                "source_object_id": local_id,
                "seed_frame": source_object["seed_frame"],
                "conditioning_frames": source_object["conditioning_frames"],
                "interaction_onset_frame": episode_activation_frame,
                "activation_frame": object_activation_frame,
                "interaction_end_frame": deactivation_frame,
                "frame_window": [object_activation_frame, deactivation_frame],
            }
            object_sources[global_id].append(source)
            cycle_sources[global_id] = source
        for frame in manifest["frames"]:
            frame_idx = int(frame["frame_idx"])
            for local_id, source_entry in frame["objects"].items():
                global_id = mapping[local_id]
                object_activation_frame = cycle_sources[global_id]["activation_frame"]
                if frame_idx < object_activation_frame or frame_idx > deactivation_frame:
                    continue
                if global_id in frames[frame_idx]:
                    raise ValueError(f"duplicate mask entry for {global_id} at frame {frame_idx}")
                entry = copy.deepcopy(source_entry)
                entry["source_object_id"] = local_id
                if isinstance(entry.get("metrics"), dict):
                    entry["metrics"]["object_id"] = global_id
                frames[frame_idx][global_id] = entry

    conflicts = []
    for frame_idx, objects in sorted(frames.items()):
        accepted = {
            object_id: entry
            for object_id, entry in objects.items()
            if entry.get("status") == "accepted" and entry.get("mask")
        }
        masks = {object_id: _load_mask(entry["mask"]) for object_id, entry in accepted.items()}
        rejected_by_conflict: dict[str, list[dict]] = defaultdict(list)
        object_ids = sorted(masks)
        for first_idx, first_id in enumerate(object_ids):
            for second_id in object_ids[first_idx + 1 :]:
                overlap_pixels = int(np.count_nonzero(masks[first_id] & masks[second_id]))
                if not overlap_pixels:
                    continue
                first_conditioned = bool(accepted[first_id].get("conditioning_frame"))
                second_conditioned = bool(accepted[second_id].get("conditioning_frame"))
                if first_conditioned and not second_conditioned:
                    rejected_ids = [second_id]
                elif second_conditioned and not first_conditioned:
                    rejected_ids = [first_id]
                else:
                    rejected_ids = [first_id, second_id]
                conflict = {
                    "frame_idx": frame_idx,
                    "object_ids": [first_id, second_id],
                    "overlap_pixels": overlap_pixels,
                    "conditioning_object_ids": [
                        object_id
                        for object_id, conditioned in (
                            (first_id, first_conditioned),
                            (second_id, second_conditioned),
                        )
                        if conditioned
                    ],
                    "rejected_object_ids": rejected_ids,
                }
                conflicts.append(conflict)
                for object_id in rejected_ids:
                    rejected_by_conflict[object_id].append(conflict)
        for object_id, object_conflicts in rejected_by_conflict.items():
            entry = objects[object_id]
            entry["status"] = "rejected_cross_object_overlap"
            entry["reasons"] = list(entry.get("reasons", [])) + ["cross_object_overlap"]
            entry["mask"] = None
            entry["cross_object_conflicts"] = object_conflicts

    objects = {}
    for object_id, sources in object_sources.items():
        objects[object_id] = {
            **(sources[0] if len(sources) == 1 else {}),
            "interaction_spans": sources,
        }

    return {
        "schema_version": "persistent_video_mask_sequence_v1",
        "status": "ready",
        "video": video_registry.get("video"),
        "frame_shape": frame_shape,
        "object_ids": list(object_sources),
        "objects": objects,
        "interaction_keyframes": copy.deepcopy(
            video_registry.get("interaction_keyframes", [])
        ),
        "frames": [
            {"frame_idx": frame_idx, "objects": objects}
            for frame_idx, objects in sorted(frames.items())
        ],
        "source_sequences": source_sequences,
        "cross_object_conflicts": conflicts,
        "failures": [],
        "invariant": (
            "video-wide object IDs persist across disjoint interaction windows; masks are "
            "hidden between windows, and rejected evidence is never filled or merged"
        ),
    }
