"""Run the relation-free, component-level automatic video segmentation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .adapter import read_candidates_json, write_json_atomic
from .box_observations import build_box_observations
from .component_expansion import (
    append_expansion_component,
    lock_known_components_from_logits,
    read_amg_candidates,
    select_unexplained_component,
)
from .episode_identity_bridge import BridgeSource, bridge_hidden_gap
from .hoi_box_events import (
    HOIBoxExpansionConfig,
    detect_confirmed_box_expansions,
    hand_linked_object_boxes,
)
from .identity_linking import (
    EpisodeInstanceObservation,
    GlobalInstanceObservation,
    link_episode_instances,
)
from .interaction_episodes import detect_interaction_episodes
from .instance_association import InstanceCandidate
from .multi_box_components import (
    accepted_hand_linked_candidates,
    classify_candidate_masks,
    confirm_new_component_track,
    refine_known_masks_from_candidates,
    validate_composite_residual_motion,
)
from .run_manifest import finish_manifest, start_manifest
from .run_combine_mask_sequences import run as run_combine_mask_sequences
from .run_combine_video_registry import run as run_combine_video_registry
from .run_instance_recovery import run as run_instance_recovery
from .run_instance_seed_discovery import (
    rank_interaction_seed_frames,
    run as run_instance_seed_discovery,
)
from .run_persistent_mask_sequence import run as run_persistent_mask_sequence
from .run_sam2_amg_probe import run as run_sam2_amg
from .sam2_multi_object import (
    ObjectMaskSeed,
    object_seed_from_box,
    propagate_multi_object_logits,
)
from .visual_instance_memory import (
    add_memory_observation,
    decide_memory_identity,
    describe_masked_instance,
)


def _episode_metadata(episode: dict) -> dict:
    return {
        "interaction_onset_frame": episode["start_frame"],
        "interaction_rising_edge_frame": episode["raw_start_frame"],
        "interaction_keyframe_offset_frames": episode["keyframe_offset_frames"],
        "interaction_onset_source": episode["start_source"],
        "interaction_onset_confidence": episode["confidence"],
        "interaction_onset_evidence_frames": [episode["raw_start_frame"]],
        "interaction_event_frame": episode["start_frame"],
        "interaction_end_frame": episode["end_frame"],
        "interaction_raw_end_frame": episode["raw_end_frame"],
        "interaction_end_source": episode["end_source"],
        "interaction_end_truncated": episode["end_truncated"],
    }


def _load_mask(path: str) -> object:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask: {path}")
    return mask > 0


def _read_video_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"cannot read video frame {frame_idx}: {video_path}")
    return frame


def _sequence_endpoint_observations(
    manifest: dict,
    *,
    at_start: bool,
) -> list[tuple[str, int, object]]:
    object_ids = manifest["object_ids"]
    entries_by_object = {object_id: [] for object_id in object_ids}
    ordered_frames = sorted(manifest["frames"], key=lambda item: item["frame_idx"], reverse=not at_start)
    for frame in ordered_frames:
        for object_id, entry in frame["objects"].items():
            if entry.get("status") == "accepted" and entry.get("mask"):
                entries_by_object[object_id].append((int(frame["frame_idx"]), entry["mask"]))
    missing = [object_id for object_id, paths in entries_by_object.items() if not paths]
    if missing:
        raise ValueError(f"sequence has no accepted mask for endpoint matching: {missing}")
    observations = []
    for object_id, paths in entries_by_object.items():
        frame_idx, mask_path = paths[0]
        observations.append((object_id, frame_idx, _load_mask(mask_path)))
    return observations


def _accepted_masks_at_frame(manifest: dict, frame_idx: int) -> tuple[dict[str, object], list[str]]:
    """Return every registered accepted mask at one event seed frame."""

    matching = [item for item in manifest["frames"] if int(item["frame_idx"]) == frame_idx]
    if len(matching) != 1:
        raise ValueError(f"sequence has no unique frame entry for {frame_idx}")
    entries = matching[0]["objects"]
    masks = {}
    missing = []
    for object_id in manifest["object_ids"]:
        entry = entries.get(object_id, {})
        if entry.get("status") != "accepted" or not entry.get("mask"):
            missing.append(object_id)
            continue
        masks[object_id] = _load_mask(entry["mask"])
    return masks, missing


def _latest_complete_masks_before(
    manifest: dict,
    frame_idx: int,
) -> tuple[int | None, dict[str, np.ndarray]]:
    """Find the last reliable all-ID state before an HOI box expansion."""

    for frame in sorted(manifest["frames"], key=lambda item: int(item["frame_idx"]), reverse=True):
        candidate_frame = int(frame["frame_idx"])
        if candidate_frame >= frame_idx:
            continue
        masks, missing = _accepted_masks_at_frame(manifest, candidate_frame)
        if not missing:
            return candidate_frame, masks
    return None, {}


def _write_evidence_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise RuntimeError(f"failed to write expansion evidence mask: {path}")
    return str(path.resolve())


def _expansion_component_evidence(
    args: argparse.Namespace,
    *,
    manifest: dict,
    event: dict,
    output_dir: Path,
) -> dict:
    """Lock known IDs, then obtain the expanded interaction-union candidate."""

    activation_frame = int(event["activation_frame"])
    seed_frame = int(event["seed_frame"])
    reference_frame, reference_masks = _latest_complete_masks_before(
        manifest, activation_frame
    )
    if reference_frame is None:
        return {"status": "failed_no_complete_pre_expansion_masks"}

    import sys

    sys.path.insert(0, str(args.sam2_root.resolve()))
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(args.checkpoint.resolve()),
        device=f"cuda:{args.gpu}",
        apply_postprocessing=True,
    )
    import torch

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        propagation = propagate_multi_object_logits(
            predictor,
            video_path=args.video,
            seeds=[
                ObjectMaskSeed(object_id, reference_frame, mask)
                for object_id, mask in reference_masks.items()
            ],
            init_state_kwargs={"offload_video_to_cpu": True},
            frame_start=reference_frame,
            frame_end=seed_frame,
        )
        event_logits = propagation.logits_by_frame.get(seed_frame, {})
        if set(event_logits) != set(reference_masks):
            return {
                "status": "failed_missing_known_component_logits_at_expansion",
                "reference_frame": reference_frame,
                "missing_object_ids": sorted(set(reference_masks) - set(event_logits)),
            }
        locked = lock_known_components_from_logits(
            reference_masks,
            event_logits,
            min_area_retention=args.min_locked_component_area_retention,
            max_area_growth=args.max_locked_component_area_growth,
        )
        if locked.get("status") != "success":
            return {**locked, "reference_frame": reference_frame}
        composite_seed, _ = object_seed_from_box(
            predictor,
            video_path=args.video,
            object_id="expanded_interaction_union",
            frame_idx=seed_frame,
            box_xyxy=event["box_xyxy"],
            init_state_kwargs={"offload_video_to_cpu": True},
        )

    output_dir = output_dir.resolve()
    evidence_paths = {
        "reference_masks": {
            object_id: _write_evidence_mask(
                output_dir / "reference" / f"{object_id}_{reference_frame:06d}.png",
                mask,
            )
            for object_id, mask in reference_masks.items()
        },
        "locked_masks": {
            object_id: _write_evidence_mask(
                output_dir / "locked" / f"{object_id}_{seed_frame:06d}.png",
                mask,
            )
            for object_id, mask in locked["masks"].items()
        },
        "expanded_interaction_union": _write_evidence_mask(
            output_dir / "expanded_interaction_union.png",
            composite_seed.mask,
        ),
    }
    return {
        "status": "success",
        "reference_frame": reference_frame,
        "seed_frame": seed_frame,
        "locked_masks": locked["masks"],
        "lock_metrics": locked["metrics"],
        "expanded_interaction_union": composite_seed.mask,
        "evidence_paths": evidence_paths,
    }


def _registry_area_references(registry: dict) -> dict[str, int]:
    """Load fixed per-instance areas, preferring recorded pre-expansion values."""

    references = {
        object_id: int(np.count_nonzero(_load_mask(entry["mask"])))
        for object_id, entry in registry["objects"].items()
    }
    for record in registry.get("confirmed_hoi_box_expansions", []):
        for object_id, area in record.get(
            "known_component_area_references", {}
        ).items():
            references[str(object_id)] = int(area)
    if set(references) != set(registry["objects"]) or any(
        area < 1 for area in references.values()
    ):
        raise ValueError("registry contains invalid component area references")
    return references


def _without_arrays(value):
    """Remove in-memory masks from a JSON audit structure."""

    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, dict):
        return {
            key: _without_arrays(item)
            for key, item in value.items()
            if not isinstance(item, np.ndarray)
        }
    if isinstance(value, list):
        return [_without_arrays(item) for item in value]
    return value


def _discover_all_box_component(
    args: argparse.Namespace,
    *,
    manifest: dict,
    registry_path: Path,
    observations: dict,
    episode: dict,
    output_dir: Path,
) -> dict:
    """Prompt every credible HOI box and confirm one unexplained mask track."""

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    area_references = _registry_area_references(registry)

    import gc
    import sys

    sys.path.insert(0, str(args.sam2_root.resolve()))
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    predictor = SAM2ImagePredictor(
        build_sam2(
            args.model_cfg,
            str(args.checkpoint.resolve()),
            device=f"cuda:{args.gpu}",
            apply_postprocessing=True,
        )
    )
    candidate_dir = output_dir / "candidate_masks"
    proposal_dir = output_dir / "proposal_masks"
    frame_proposals = []
    known_masks_by_frame = {}
    interaction_envelopes_by_candidate_id = {}
    frame_audit = []
    start_frame = int(episode["start_frame"])
    end_frame = int(episode["end_frame"])
    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {args.video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot read video frame {frame_idx}: {args.video}")
            boxes = accepted_hand_linked_candidates(observations["frames"][frame_idx])
            if not boxes:
                frame_audit.append(
                    {"frame_idx": frame_idx, "status": "no_accepted_hand_linked_boxes"}
                )
                continue
            existing_masks, missing = _accepted_masks_at_frame(manifest, frame_idx)
            if missing:
                frame_audit.append(
                    {
                        "frame_idx": frame_idx,
                        "status": "missing_existing_instance_masks",
                        "missing_object_ids": missing,
                    }
                )
                continue
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                candidate_masks = []
                prompt_rejections = []
                for box_index, box in enumerate(boxes):
                    masks, scores, _ = predictor.predict(
                        box=np.asarray(box["box_xyxy"], dtype=np.float32),
                        multimask_output=True,
                        return_logits=False,
                    )
                    best_index = int(np.argmax(scores))
                    quality = float(scores[best_index])
                    if quality < args.min_box_mask_quality:
                        prompt_rejections.append(
                            {
                                "source_detection_id": box["source_detection_id"],
                                "reason": "sam2_box_mask_quality_below_threshold",
                                "quality_score": quality,
                            }
                        )
                        continue
                    mask = np.asarray(masks[best_index], dtype=bool)
                    candidate_id = (
                        f"f{frame_idx:06d}_{box['source_detection_id']}_m{best_index}"
                    )
                    mask_path = candidate_dir / f"{candidate_id}.png"
                    _write_evidence_mask(mask_path, mask)
                    candidate_masks.append(
                        {
                            "candidate_id": candidate_id,
                            "candidate_source": "sam2_image_prompt_from_all_accepted_hoi_boxes",
                            "source_detection_id": box["source_detection_id"],
                            "box_xyxy": list(box["box_xyxy"]),
                            "quality_score": quality,
                            "candidate_quality_score": quality,
                            "mask_path": str(mask_path.resolve()),
                            "mask": mask,
                        }
                    )
                    interaction_envelopes_by_candidate_id[candidate_id] = mask.copy()
            # A credible small-box mask first restores the registered part.
            # The remaining larger masks can then expose a disjoint residual.
            refined_masks, refinements = refine_known_masks_from_candidates(
                existing_masks,
                candidate_masks,
                area_references=area_references,
                min_candidate_overlap_fraction=args.min_box_mask_known_overlap,
                min_existing_overlap_fraction=args.min_box_mask_existing_overlap,
                min_area_scale=args.min_box_mask_known_area_scale,
                max_area_scale=args.max_box_mask_known_area_scale,
            )
            classifications = classify_candidate_masks(
                candidate_masks,
                existing_masks=refined_masks,
                min_area_pixels=args.min_area_pixels,
                min_candidate_explained_fraction=(
                    args.min_box_mask_explained_fraction
                ),
                min_existing_coverage_for_residual=(
                    args.min_existing_coverage_for_residual
                ),
                max_direct_overlap_fraction=(
                    args.max_expansion_direct_overlap_fraction
                ),
                min_largest_component_fraction=(
                    args.min_largest_component_fraction
                ),
                max_frame_area_fraction=args.max_frame_area_fraction,
            )
            known_masks_by_frame[frame_idx] = {
                object_id: mask.copy() for object_id, mask in refined_masks.items()
            }
            for classification in classifications:
                classification["frame_idx"] = frame_idx
                if classification.get("status") == "new_component_proposal":
                    proposal_path = (
                        proposal_dir / f"{classification['candidate_id']}.png"
                    )
                    _write_evidence_mask(proposal_path, classification["mask"])
                    classification["proposal_mask_path"] = str(proposal_path.resolve())
                    frame_proposals.append(classification)
            frame_audit.append(
                {
                    "frame_idx": frame_idx,
                    "status": "evaluated_all_accepted_hand_linked_boxes",
                    "source_box_count": len(boxes),
                    "prompt_rejections": prompt_rejections,
                    "known_mask_refinements": refinements,
                    "candidates": _without_arrays(classifications),
                }
            )
    finally:
        cap.release()
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    confirmation = confirm_new_component_track(
        frame_proposals,
        min_consecutive_frames=args.min_hoi_box_expansion_frames,
        max_frame_gap=args.max_box_mask_confirmation_gap,
        min_mask_continuity=args.min_box_mask_temporal_continuity,
        max_area_scale=args.max_box_mask_temporal_area_scale,
    )
    if confirmation.get("status") == "success":
        motion_validation = validate_composite_residual_motion(
            confirmation,
            frame_proposals=frame_proposals,
            known_masks_by_frame=known_masks_by_frame,
            interaction_envelopes_by_candidate_id=(
                interaction_envelopes_by_candidate_id
            ),
            min_existing_coverage_for_residual=(
                args.min_existing_coverage_for_residual
            ),
            min_jointly_visible_frames=args.min_jointly_visible_frames,
            min_relative_displacement_diagonals=(
                args.min_relative_displacement_diagonals
            ),
        )
        confirmation["motion_validation"] = motion_validation
        if motion_validation["status"].startswith("failed"):
            confirmation["status"] = motion_validation["status"]
    audit = {
        "schema_version": "all_box_component_discovery_v1",
        "episode_idx": int(episode["episode_idx"]),
        "area_references": area_references,
        "confirmation": _without_arrays(confirmation),
        "frames": frame_audit,
    }
    audit_path = output_dir / "audit.json"
    write_json_atomic(audit_path, audit)
    if confirmation.get("status") != "success":
        return {
            "status": "no_confirmed_new_component",
            "audit": str(audit_path.resolve()),
            "confirmation": confirmation,
        }
    seed_frame = int(confirmation["seed_frame"])
    selected = confirmation["selected"]
    interaction_envelope_mask = interaction_envelopes_by_candidate_id.get(
        selected["candidate_id"]
    )
    if interaction_envelope_mask is None or not np.any(interaction_envelope_mask):
        return {
            "status": "failed_missing_confirmed_interaction_envelope",
            "audit": str(audit_path.resolve()),
            "confirmation": confirmation,
        }
    selected_existing = known_masks_by_frame[seed_frame]
    evidence_dir = output_dir / "confirmed_evidence"
    selected_path = _write_evidence_mask(
        evidence_dir / "new_component.png", selected["mask"]
    )
    existing_paths = {
        object_id: _write_evidence_mask(
            evidence_dir / f"known_{object_id}.png", mask
        )
        for object_id, mask in selected_existing.items()
    }
    event = {
        "activation_frame": int(confirmation["activation_frame"]),
        "seed_frame": seed_frame,
        "box_xyxy": list(selected["box_xyxy"]),
        "source_detection_id": selected["source_detection_id"],
        "evidence_frames": list(confirmation["track_frames"]),
        "source": "all_credible_hoi_box_mask_explainability",
    }
    return {
        "status": "success",
        "event": event,
        "existing_masks": selected_existing,
        "area_references": area_references,
        "selection": {"status": "success", "selected": selected},
        "interaction_envelope_mask": interaction_envelope_mask,
        "evidence_paths": {
            "new_component": selected_path,
            "known_components": existing_paths,
        },
        "audit": str(audit_path.resolve()),
    }


def _identity_maps(args: argparse.Namespace, sequence_manifests: list[dict]) -> dict:
    """Assign persistent IDs from bridge evidence, visual memory, or clear novelty.

    A short hidden-gap bridge is preferred.  If it cannot match an instance,
    class-free visual memory may reuse an ID only with a decisive score.  An ID
    is new only when it is decisively unlike every remembered instance; every
    intermediate case fails the whole video instead of guessing.
    """

    maps = []
    previous_sources: list[BridgeSource] = []
    next_global_index = 1
    failures = []
    identity_audit = []
    visual_memory = {}
    predictor = None
    for cycle_idx, manifest in enumerate(sequence_manifests):
        current_entries = _sequence_endpoint_observations(manifest, at_start=True)
        current = [
            EpisodeInstanceObservation(local_id, mask)
            for local_id, _, mask in current_entries
        ]
        current_descriptors = {
            local_id: describe_masked_instance(
                local_id,
                _read_video_frame(args.video, frame_idx),
                mask,
            )
            for local_id, frame_idx, mask in current_entries
        }
        cycle_audit = {"cycle_idx": cycle_idx, "bridge": None, "visual_memory": []}
        if not previous_sources:
            mapping = {}
            for item in current:
                mapping[item.local_object_id] = f"object_{next_global_index:04d}"
                next_global_index += 1
        else:
            target_frame = min(frame_idx for _, frame_idx, _ in current_entries)
            if predictor is None:
                import sys

                sys.path.insert(0, str(args.sam2_root.resolve()))
                from sam2.build_sam import build_sam2_video_predictor

                predictor = build_sam2_video_predictor(
                    args.model_cfg,
                    str(args.checkpoint.resolve()),
                    device=f"cuda:{args.gpu}",
                    apply_postprocessing=True,
                )
            try:
                bridged = bridge_hidden_gap(
                    predictor,
                    video_path=args.video,
                    sources=previous_sources,
                    target_frame=target_frame,
                    max_gap_frames=args.max_id_bridge_gap_frames,
                )
                linked = link_episode_instances(bridged, current)
                cycle_audit["bridge"] = linked
            except (RuntimeError, ValueError) as exc:
                linked = {"status": "failed_hidden_identity_bridge", "detail": str(exc)}
                cycle_audit["bridge"] = linked
            mapping = dict(linked.get("local_to_existing_global", {}))
            for item in current:
                local_id = item.local_object_id
                if local_id in mapping:
                    continue
                decision = decide_memory_identity(visual_memory, current_descriptors[local_id])
                cycle_audit["visual_memory"].append(
                    {"local_object_id": local_id, **decision}
                )
                if decision["status"] == "reuse_visual_memory":
                    global_id = decision["object_id"]
                    if global_id in mapping.values():
                        failures.append(
                            {
                                "cycle_idx": cycle_idx,
                                "reason": "failed_duplicate_current_global_identity",
                                "global_object_id": global_id,
                            }
                        )
                        continue
                    mapping[local_id] = global_id
                elif decision["status"] == "new_visual_instance":
                    mapping[local_id] = f"object_{next_global_index:04d}"
                    next_global_index += 1
                else:
                    failures.append({"cycle_idx": cycle_idx, "reason": decision})
            if len(mapping) != len(current) or len(set(mapping.values())) != len(mapping):
                failures.append(
                    {
                        "cycle_idx": cycle_idx,
                        "reason": "failed_incomplete_visual_identity_mapping",
                        "mapping": mapping,
                    }
                )
                continue
        maps.append(mapping)
        for local_id, descriptor in current_descriptors.items():
            add_memory_observation(
                visual_memory,
                replace(descriptor, object_id=mapping[local_id]),
            )
        end_entries = _sequence_endpoint_observations(manifest, at_start=False)
        previous_sources = [
            BridgeSource(mapping[local_id], frame_idx, mask)
            for local_id, frame_idx, mask in end_entries
        ]
        identity_audit.append(cycle_audit)
    return {
        "status": "success" if not failures else "failed",
        "cycle_object_id_maps": maps,
        "identity_audit": identity_audit,
        "failures": failures,
    }


def _run_sequence(args: argparse.Namespace, *, registry: Path, output_dir: Path, episode: dict) -> dict:
    return run_persistent_mask_sequence(
        argparse.Namespace(
            video=args.video,
            registry=registry,
            frame_start=episode["start_frame"],
            frame_end=episode["end_frame"],
            output_dir=output_dir,
            sam2_root=args.sam2_root,
            checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            gpu=args.gpu,
            min_area_pixels=args.min_area_pixels,
            min_largest_component_fraction=args.min_largest_component_fraction,
            max_frame_area_fraction=args.max_frame_area_fraction,
            max_border_area_fraction=args.max_border_area_fraction,
            min_area_ratio_vs_history=args.min_area_ratio_vs_history,
            max_area_ratio_vs_history=args.max_area_ratio_vs_history,
            max_centroid_step_diagonals=args.max_centroid_step_diagonals,
            history_size=args.history_size,
            # The locked, disjoint component conditions are propagated in
            # isolated memories so assembly cannot suppress a small part.
            # Per-pixel ownership and the raw-logit identity-conflict gate
            # still prevent two IDs from claiming the same visible instance.
            independent_object_states=True,
            max_cross_instance_overlap_fraction=args.max_cross_instance_overlap_fraction,
            min_locked_component_area_retention=args.min_locked_component_area_retention,
            max_locked_component_area_growth=args.max_locked_component_area_growth,
            box_observations=args.box_observations,
            interaction_box_margin_fraction=args.interaction_box_margin_fraction,
            max_interaction_box_fallback_gap_frames=(
                args.max_interaction_box_fallback_gap_frames
            ),
            max_temporal_validation_seconds=(
                args.max_temporal_validation_seconds
            ),
            min_temporal_confirmation_frames=(
                args.min_temporal_confirmation_frames
            ),
            min_flow_warp_continuity=args.min_flow_warp_continuity,
            min_cycle_iou=args.min_cycle_iou,
            max_compensated_centroid_step_diagonals=(
                args.max_compensated_centroid_step_diagonals
            ),
        )
    )


def _run_recovery_amg(args: argparse.Namespace, *, frame_idx: int, output_dir: Path) -> dict:
    return run_sam2_amg(
        argparse.Namespace(
            video=args.video,
            frame_idx=frame_idx,
            detections=args.detections,
            output_dir=output_dir,
            sam2_root=args.sam2_root,
            checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            gpu=args.gpu,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_threshold=args.pred_iou_threshold,
            stability_threshold=args.stability_threshold,
            min_area=args.min_area_pixels,
            max_area=args.max_seed_area_pixels,
            min_largest_component_fraction=args.min_largest_component_fraction,
            max_hand_overlap=args.max_hand_overlap,
        )
    )


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_manifest(output_dir, args=args, command=sys.argv)

    def complete(summary: dict) -> dict:
        finish_manifest(output_dir, summary)
        return summary
    detections = read_candidates_json(args.detections)
    observations = build_box_observations(detections)
    observations_path = output_dir / "box_observations.json"
    write_json_atomic(observations_path, observations)
    args.box_observations = observations_path
    episodes = detect_interaction_episodes(
        observations["frames"],
        fps=float(detections["video"]["fps"]),
    )
    write_json_atomic(output_dir / "interaction_episodes.json", {"episodes": episodes})
    if not episodes:
        summary = {
            "schema_version": "instance_video_segmentation_v1",
            "status": "failed_no_interaction_episodes",
            "video": str(args.video.resolve()),
            "failed_videos": [str(args.video.resolve())],
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return complete(summary)

    cycle_registries = []
    sequence_paths = []
    sequence_manifests = []
    failures = []
    for episode in episodes:
        cycle_idx = episode["episode_idx"]
        cycle_dir = output_dir / f"episode_{cycle_idx:02d}"
        registry_path = None
        seed_failures = []
        source_frames = rank_interaction_seed_frames(
            detections,
            interaction_start_frame=episode["start_frame"],
            interaction_end_frame=episode["end_frame"],
            max_seed_frames=args.max_seed_search_frames,
        )
        source_rank = {frame_idx: rank for rank, frame_idx in enumerate(source_frames)}
        successful_seeds = []
        for source_frame in source_frames:
            seed_dir = cycle_dir / f"seed_frame_{source_frame:06d}"
            seed = run_instance_seed_discovery(
                argparse.Namespace(
                    video=args.video,
                    detections=args.detections,
                    source_frame=source_frame,
                    interaction_start_frame=episode["start_frame"],
                    interaction_end_frame=episode["end_frame"],
                    output_dir=seed_dir,
                    sam2_root=args.sam2_root,
                    checkpoint=args.checkpoint,
                    model_cfg=args.model_cfg,
                    gpu=args.gpu,
                    points_per_side=args.points_per_side,
                    points_per_batch=args.points_per_batch,
                    pred_iou_threshold=args.pred_iou_threshold,
                    stability_threshold=args.stability_threshold,
                    min_area=args.min_area_pixels,
                    max_area=args.max_seed_area_pixels,
                    min_largest_component_fraction=args.min_largest_component_fraction,
                    max_hand_overlap=args.max_hand_overlap,
                    max_seed_candidates=args.max_seed_candidates,
                    min_candidate_roi_fraction=args.min_candidate_roi_fraction,
                    min_child_inside_parent=args.min_child_inside_parent,
                    min_residual_area=args.min_area_pixels,
                    min_jointly_visible_frames=args.min_jointly_visible_frames,
                    min_relative_displacement_diagonals=args.min_relative_displacement_diagonals,
                    max_cross_instance_overlap_fraction=args.max_cross_instance_overlap_fraction,
                    max_components=args.max_components,
                    min_multi_component_score_gain=args.min_multi_component_score_gain,
                )
            )
            if seed.get("status") == "success":
                successful_seeds.append((source_frame, seed))
                continue
            seed_failures.append({"source_frame": source_frame, "reason": seed.get("status")})
        if successful_seeds:
            def seed_rank(item):
                source_frame, seed = item
                decision = seed.get("component_decision", {})
                return (
                    decision.get("status") == "multiple_components",
                    -source_rank[source_frame],
                    float(decision.get("score", 0.0)),
                )

            selected_source_frame, selected_seed = max(successful_seeds, key=seed_rank)
            registry_path = Path(selected_seed["reconstruction_registry"])
        if registry_path is None:
            failures.append({"episode_idx": cycle_idx, "reason": "failed_seed_discovery", "attempts": seed_failures})
            continue

        sequence = _run_sequence(
            args,
            registry=registry_path,
            output_dir=cycle_dir / "sequence_attempt_00",
            episode=episode,
        )
        for recovery_idx in range(args.max_recovery_attempts):
            if sequence.get("status") == "success":
                break
            conflicts = sequence.get("instance_identity_conflicts", {})
            if not conflicts:
                break
            recovery_frame = min(int(frame_idx) for frame_idx in conflicts)
            amg_dir = cycle_dir / f"recovery_{recovery_idx:02d}" / "amg"
            amg = _run_recovery_amg(args, frame_idx=recovery_frame, output_dir=amg_dir)
            if amg.get("status") != "success":
                break
            recovery = run_instance_recovery(
                argparse.Namespace(
                    video=args.video,
                    source_registry=registry_path,
                    candidate_summary=amg_dir / "summary.json",
                    unresolved_mask_dir=(
                        cycle_dir
                        / f"sequence_attempt_{recovery_idx:02d}"
                        / "unresolved_masks"
                        / f"frame_{recovery_frame:06d}"
                    ),
                    recovery_frame=recovery_frame,
                    output_dir=cycle_dir / f"recovery_{recovery_idx:02d}" / "registry",
                    min_child_inside_parent=args.min_child_inside_parent,
                    min_residual_area=args.min_area_pixels,
                    max_cross_instance_overlap_fraction=args.max_cross_instance_overlap_fraction,
                    min_assignment_margin=args.min_assignment_margin,
                )
            )
            if recovery.get("status") != "success":
                break
            registry_path = Path(recovery["reconstruction_registry"])
            sequence = _run_sequence(
                args,
                registry=registry_path,
                output_dir=cycle_dir / f"sequence_attempt_{recovery_idx + 1:02d}",
                episode=episode,
            )
        if sequence.get("status") != "success":
            failures.append({"episode_idx": cycle_idx, "reason": sequence.get("status"), "details": sequence.get("failures", [])})
            continue

        # Preserve and prompt every credible hand-linked box.  Each resulting
        # SAM mask is compared with all currently registered IDs.  Only a
        # temporally stable unexplained residual/direct mask receives a new ID;
        # a composite whole is never registered as one extra object.
        box_events = {
            "status": "success",
            "source": "all_credible_hoi_boxes_mask_explainability",
            "component_registration": [],
        }
        expansion_failure = None
        for expansion_idx in range(args.max_components):
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            if len(registry.get("objects", {})) >= args.max_components:
                box_events["stopped_reason"] = "maximum_component_count_reached"
                break
            manifest = json.loads(
                Path(sequence["mask_sequence"]).read_text(encoding="utf-8")
            )
            event_root = cycle_dir / f"all_box_discovery_{expansion_idx:02d}"
            discovery = _discover_all_box_component(
                args,
                manifest=manifest,
                registry_path=registry_path,
                observations=observations,
                episode=episode,
                output_dir=event_root / "evidence",
            )
            event_audit = {
                key: _without_arrays(value)
                for key, value in discovery.items()
                if key not in {"existing_masks", "selection", "area_references"}
            }
            if discovery.get("status") == "no_confirmed_new_component":
                event_audit["status"] = "no_confirmed_new_component"
                box_events["component_registration"].append(event_audit)
                box_events["stopped_reason"] = "all_candidates_explained_or_unconfirmed"
                break
            if discovery.get("status") != "success":
                event_audit["status"] = discovery.get("status")
                box_events["component_registration"].append(event_audit)
                expansion_failure = event_audit
                break
            registration = append_expansion_component(
                registry_path,
                existing_masks=discovery["existing_masks"],
                event=discovery["event"],
                selection=discovery["selection"],
                output_dir=event_root / "registry",
                known_component_area_references=discovery["area_references"],
                # The full prompted HOI mask is kept as a private spatial
                # envelope.  It constrains decomposition/reacquisition but is
                # never promoted to a public instance or rendered label.
                interaction_envelope_mask=discovery[
                    "interaction_envelope_mask"
                ],
            )
            registry_path = Path(registration["reconstruction_registry"])
            sequence = _run_sequence(
                args,
                registry=registry_path,
                output_dir=cycle_dir / f"sequence_all_boxes_{expansion_idx:02d}",
                episode=episode,
            )
            event_audit["registration"] = registration
            event_audit["sequence_status"] = sequence.get("status")
            box_events["component_registration"].append(event_audit)
            if sequence.get("status") != "success":
                expansion_failure = event_audit
                break
        write_json_atomic(cycle_dir / "box_events.json", box_events)
        if expansion_failure is not None:
            failures.append(
                {
                    "episode_idx": cycle_idx,
                    "reason": "failed_confirmed_box_expansion_component_registration",
                    "details": expansion_failure,
                }
            )
            continue
        cycle_registries.append(registry_path)
        sequence_path = Path(sequence["mask_sequence"])
        sequence_paths.append(sequence_path)
        sequence_manifests.append(json.loads(sequence_path.read_text(encoding="utf-8")))

    if failures or len(cycle_registries) != len(episodes):
        summary = {
            "schema_version": "instance_video_segmentation_v1",
            "status": "failed_episode_segmentation",
            "video": str(args.video.resolve()),
            "episodes": episodes,
            "failures": failures,
            "failed_videos": [str(args.video.resolve())],
            "video_mask_sequence": None,
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return complete(summary)
    identity = _identity_maps(args, sequence_manifests)
    if identity["status"] != "success":
        summary = {
            "schema_version": "instance_video_segmentation_v1",
            "status": "failed_global_identity_linking",
            "video": str(args.video.resolve()),
            "failures": identity["failures"],
            "failed_videos": [str(args.video.resolve())],
            "video_mask_sequence": None,
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return complete(summary)
    identity_path = output_dir / "visual_identity_map.json"
    write_json_atomic(identity_path, identity)
    registry_result = run_combine_video_registry(
        argparse.Namespace(
            video=args.video,
            cycle_registry=cycle_registries,
            cycle_metadata=[_episode_metadata(episode) for episode in episodes],
            identity_map=identity["cycle_object_id_maps"],
            output_dir=output_dir / "video_registry",
        )
    )
    if registry_result.get("status") != "success":
        summary = {
            "schema_version": "instance_video_segmentation_v1",
            "status": "failed_video_registry_combine",
            "video": str(args.video.resolve()),
            "failures": registry_result.get("failures", []),
            "failed_videos": [str(args.video.resolve())],
            "video_mask_sequence": None,
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return complete(summary)
    combined = run_combine_mask_sequences(
        argparse.Namespace(
            video=args.video,
            video_registry=Path(registry_result["final_registry"]),
            cycle_sequence=sequence_paths,
            output_dir=output_dir / "video_mask_sequence",
        )
    )
    summary = {
        "schema_version": "instance_video_segmentation_v1",
        "status": "success" if combined.get("status") == "success" else combined.get("status"),
        "video": str(args.video.resolve()),
        "episodes": episodes,
        "global_identity_map": str(identity_path.resolve()),
        "final_registry": registry_result.get("final_registry"),
        "video_mask_sequence": combined.get("video_mask_sequence"),
        "segmentation_report": combined.get("segmentation_report"),
        "failures": combined.get("failures", []),
        "failed_videos": combined.get("failed_videos", []),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return complete(summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--points-per-batch", type=int, default=32)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.8)
    parser.add_argument("--stability-threshold", type=float, default=0.92)
    parser.add_argument("--max-seed-search-frames", type=int, default=2)
    parser.add_argument("--max-seed-candidates", type=int, default=6)
    parser.add_argument("--max-seed-area-pixels", type=int, default=100000)
    parser.add_argument("--min-candidate-roi-fraction", type=float, default=0.15)
    parser.add_argument("--min-child-inside-parent", type=float, default=0.9)
    parser.add_argument("--min-jointly-visible-frames", type=int, default=3)
    parser.add_argument("--min-relative-displacement-diagonals", type=float, default=0.03)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--min-multi-component-score-gain", type=float, default=0.05)
    parser.add_argument("--max-recovery-attempts", type=int, default=2)
    parser.add_argument("--min-hoi-box-area-growth", type=float, default=1.6)
    parser.add_argument("--min-hoi-box-reference-coverage", type=float, default=0.7)
    parser.add_argument("--min-hoi-box-expansion-frames", type=int, default=3)
    parser.add_argument("--min-hoi-box-continuity-coverage", type=float, default=0.45)
    parser.add_argument("--max-hoi-box-area-fraction", type=float, default=0.35)
    parser.add_argument("--min-box-mask-quality", type=float, default=0.75)
    parser.add_argument("--min-box-mask-known-overlap", type=float, default=0.55)
    parser.add_argument("--min-box-mask-existing-overlap", type=float, default=0.20)
    parser.add_argument("--min-box-mask-known-area-scale", type=float, default=0.45)
    parser.add_argument("--max-box-mask-known-area-scale", type=float, default=1.80)
    parser.add_argument("--min-box-mask-explained-fraction", type=float, default=0.85)
    parser.add_argument("--max-box-mask-confirmation-gap", type=int, default=1)
    parser.add_argument("--min-box-mask-temporal-continuity", type=float, default=0.35)
    parser.add_argument("--max-box-mask-temporal-area-scale", type=float, default=2.5)
    parser.add_argument("--min-expansion-component-roi-fraction", type=float, default=0.5)
    parser.add_argument("--min-existing-coverage-for-residual", type=float, default=0.85)
    parser.add_argument("--max-expansion-direct-overlap-fraction", type=float, default=0.05)
    parser.add_argument("--min-locked-component-area-retention", type=float, default=0.5)
    parser.add_argument("--max-locked-component-area-growth", type=float, default=1.35)
    parser.add_argument("--interaction-box-margin-fraction", type=float, default=0.08)
    parser.add_argument("--max-interaction-box-fallback-gap-frames", type=int, default=3)
    parser.add_argument("--max-temporal-validation-seconds", type=float, default=1.0)
    parser.add_argument("--min-temporal-confirmation-frames", type=int, default=3)
    parser.add_argument("--min-flow-warp-continuity", type=float, default=0.50)
    parser.add_argument("--min-cycle-iou", type=float, default=0.60)
    parser.add_argument(
        "--max-compensated-centroid-step-diagonals", type=float, default=0.50
    )
    parser.add_argument("--max-id-bridge-gap-frames", type=int, default=12)
    parser.add_argument("--min-assignment-margin", type=float, default=0.05)
    parser.add_argument("--max-cross-instance-overlap-fraction", type=float, default=0.05)
    parser.add_argument("--min-area-pixels", type=int, default=256)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.9)
    parser.add_argument("--max-frame-area-fraction", type=float, default=0.35)
    parser.add_argument("--max-border-area-fraction", type=float, default=0.15)
    parser.add_argument("--min-area-ratio-vs-history", type=float, default=0.2)
    parser.add_argument("--max-area-ratio-vs-history", type=float, default=4.0)
    parser.add_argument("--max-centroid-step-diagonals", type=float, default=2.5)
    parser.add_argument("--history-size", type=int, default=7)
    parser.add_argument("--max-hand-overlap", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # ★ 必须设默认设备。模型虽然显式放在 cuda:{args.gpu}, 但 SAM2 内部隐式创建的张量
    #   会落在**默认设备 cuda:0** 上; GPU0 被别人占满时整条传播就在那里排队 ——
    #   现象是模型所在卡 util≈0、进程 5/8 时间在 D 态、速度与是否争抢无关。
    #   2026-08-12 对照实测(同视频同配置, 传播稳态): 不设 3173 ms/帧 -> 设了 103 ms/帧, 31x。
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.set_device(int(args.gpu))
    except Exception as _e:          # 设不上不该让整步失败
        print(f'[v17a] warn: set_device({args.gpu}) 失败: {_e}')
    try:
        summary = run(args)
    except Exception as exc:
        summary = {
            "schema_version": "instance_video_segmentation_v1",
            "status": "failed_fatal_error",
            "video": str(args.video.resolve()),
            "failures": [{"reason": f"{type(exc).__name__}: {exc}"}],
            "failed_videos": [str(args.video.resolve())],
            "video_mask_sequence": None,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(args.output_dir / "fatal_summary.json", summary)
        finish_manifest(args.output_dir.resolve(), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed_videos"]:
        print("FAILED VIDEOS:")
        for video in summary["failed_videos"]:
            print(video)
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
