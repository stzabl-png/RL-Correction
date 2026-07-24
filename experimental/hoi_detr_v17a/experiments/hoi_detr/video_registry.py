"""Combine successful per-cycle reconstruction registries into one video registry."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


def _ordered_local_object_ids(registry: dict[str, Any], cycle_idx: int) -> list[str]:
    cycle_objects = registry.get("objects", {})
    if not isinstance(cycle_objects, dict) or not cycle_objects:
        raise ValueError(f"cycle {cycle_idx} must contain at least one object")
    explicit_order = registry.get("object_order")
    ordered = list(cycle_objects) if explicit_order is None else list(explicit_order)
    if len(ordered) != len(set(ordered)) or set(ordered) != set(cycle_objects):
        raise ValueError(f"cycle {cycle_idx} object_order must list every object exactly once")
    return ordered


def _rewrite_relation_ids(
    relation: dict[str, Any],
    id_map: dict[str, str],
    *,
    cycle_idx: int,
) -> dict[str, Any]:
    """Rewrite role-agnostic relation endpoint fields to video-wide IDs."""

    rewritten = deepcopy(relation)
    for key, value in list(rewritten.items()):
        if key.endswith("_object_id"):
            if value not in id_map:
                raise ValueError(
                    f"cycle {cycle_idx} relation {key} references unknown object {value!r}"
                )
            rewritten[key] = id_map[value]
        elif key == "object_ids" or key.endswith("_object_ids"):
            if not isinstance(value, list) or any(item not in id_map for item in value):
                raise ValueError(
                    f"cycle {cycle_idx} relation {key} must reference known objects"
                )
            rewritten[key] = [id_map[item] for item in value]
    rewritten["cycle_idx"] = cycle_idx
    return rewritten


def combine_cycle_registries(
    registries: Iterable[dict[str, Any]],
    *,
    video: str,
    cycle_metadata: Iterable[dict[str, Any]] | None = None,
    cycle_object_id_maps: Iterable[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Assign stable video-wide IDs without altering per-cycle mask artifacts."""

    items = list(registries)
    metadata_items = list(cycle_metadata) if cycle_metadata is not None else None
    identity_maps = (
        list(cycle_object_id_maps) if cycle_object_id_maps is not None else None
    )
    if not items:
        raise ValueError("at least one cycle registry is required")
    if metadata_items is not None and len(metadata_items) != len(items):
        raise ValueError("cycle metadata count does not match registry count")
    if identity_maps is not None and len(identity_maps) != len(items):
        raise ValueError("cycle object ID map count does not match registry count")
    objects: dict[str, Any] = {}
    relations: list[dict[str, Any]] = []
    emit_relations = any(registry.get("relations") for registry in items)
    cycles: list[dict[str, Any]] = []
    next_global_index = 1
    for cycle_idx, registry in enumerate(items):
        if registry.get("status") != "ready":
            raise ValueError(f"cycle {cycle_idx} registry is not ready")
        cycle_objects = registry.get("objects", {})
        ordered_local_ids = _ordered_local_object_ids(registry, cycle_idx)
        cycle_relations = registry.get("relations", [])
        if not isinstance(cycle_relations, list):
            raise ValueError(f"cycle {cycle_idx} relations must be a list")
        if identity_maps is None:
            id_map = {
                local_id: f"object_{next_global_index + offset:04d}"
                for offset, local_id in enumerate(ordered_local_ids)
            }
            next_global_index += len(ordered_local_ids)
        else:
            id_map = {
                str(local_id): str(global_id)
                for local_id, global_id in identity_maps[cycle_idx].items()
            }
            if set(id_map) != set(ordered_local_ids):
                raise ValueError(
                    f"cycle {cycle_idx} object ID map must list every local object"
                )
            if any(not global_id for global_id in id_map.values()):
                raise ValueError(f"cycle {cycle_idx} contains an empty global object ID")
            if len(set(id_map.values())) != len(id_map):
                raise ValueError(
                    f"cycle {cycle_idx} cannot map two visible objects to one global ID"
                )
        for local_id, global_id in id_map.items():
            entry = deepcopy(cycle_objects[local_id])
            observation = {
                **entry,
                "cycle_idx": cycle_idx,
                "source_object_id": local_id,
            }
            if global_id not in objects:
                objects[global_id] = {
                    **observation,
                    "cycle_observations": [observation],
                }
            else:
                objects[global_id]["cycle_observations"].append(observation)
        rewritten_relations = (
            [
                _rewrite_relation_ids(relation, id_map, cycle_idx=cycle_idx)
                for relation in cycle_relations
            ]
            if emit_relations
            else []
        )
        relations.extend(rewritten_relations)
        cycle_entry = {
                "cycle_idx": cycle_idx,
                "object_ids": [id_map[local_id] for local_id in ordered_local_ids],
                "source_object_id_map": id_map,
            }
        evidence_frames = [
            relation["evidence_frame_idx"]
            for relation in rewritten_relations
            if "evidence_frame_idx" in relation
        ]
        if "evidence_frame_idx" in registry:
            cycle_entry["evidence_frame_idx"] = registry["evidence_frame_idx"]
        elif evidence_frames:
            cycle_entry["evidence_frame_idx"] = min(evidence_frames)
        if "event_type" in registry:
            cycle_entry["event_type"] = registry["event_type"]
        if metadata_items is not None:
            metadata = metadata_items[cycle_idx]
            cycle_entry["interaction_onset_frame"] = metadata["interaction_onset_frame"]
            cycle_entry["interaction_rising_edge_frame"] = metadata.get(
                "interaction_rising_edge_frame"
            )
            cycle_entry["interaction_keyframe_offset_frames"] = metadata.get(
                "interaction_keyframe_offset_frames", 0
            )
            cycle_entry["interaction_onset_source"] = metadata["interaction_onset_source"]
            cycle_entry["interaction_onset_confidence"] = metadata[
                "interaction_onset_confidence"
            ]
            cycle_entry["interaction_onset_evidence_frames"] = metadata[
                "interaction_onset_evidence_frames"
            ]
            cycle_entry["interaction_event_frame"] = metadata["interaction_event_frame"]
            cycle_entry["interaction_end_frame"] = metadata["interaction_end_frame"]
            cycle_entry["interaction_raw_end_frame"] = metadata["interaction_raw_end_frame"]
            cycle_entry["interaction_end_source"] = metadata["interaction_end_source"]
            cycle_entry["interaction_end_truncated"] = metadata[
                "interaction_end_truncated"
            ]
        cycles.append(cycle_entry)
    result = {
        "schema_version": "persistent_video_reconstruction_registry_v1",
        "status": "ready",
        "video": video,
        "objects": objects,
        "cycles": cycles,
        "interaction_keyframes": [
            {
                "cycle_idx": cycle["cycle_idx"],
                "frame_idx": cycle["interaction_onset_frame"],
                "start_frame": cycle["interaction_onset_frame"],
                "end_frame": cycle["interaction_end_frame"],
                "rising_edge_frame": cycle["interaction_rising_edge_frame"],
                "offset_frames": cycle["interaction_keyframe_offset_frames"],
                "source": cycle["interaction_onset_source"],
                "confidence": cycle["interaction_onset_confidence"],
                "object_ids": cycle["object_ids"],
                "end_source": cycle["interaction_end_source"],
                "end_truncated": cycle["interaction_end_truncated"],
            }
            for cycle in cycles
            if "interaction_onset_frame" in cycle
        ],
        "invariant": (
            "registered reconstruction masks are immutable per-object inputs; one "
            "physical instance keeps its global ID across disjoint interaction windows"
        ),
    }
    if emit_relations:
        result["relations"] = relations
    return result
