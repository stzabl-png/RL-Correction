"""Discover one whole object or multiple movable component instances for an episode."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from .adapter import write_json_atomic
from .component_decision import component_motion_evidence
from .instance_association import (
    InstanceCandidate,
    derive_residual_instance_candidates,
)
from .instance_hypothesis import select_initial_instance_hypothesis
from .instance_registry import write_instance_registry
from .mask_ownership import resolve_visible_mask_ownership
from .run_sam2_amg_probe import run as run_sam2_amg
from .sam2_multi_object import (
    ObjectMaskSeed,
    object_seed_from_box,
    propagate_independent_object_logits,
)


def _box_region(shape: tuple[int, int], boxes: list[list[float]]) -> np.ndarray:
    height, width = shape
    region = np.zeros(shape, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        region[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
    return region


def _linked_object_prompts(frame: dict) -> list[dict]:
    """Return visual-object prompts and the hands directly linked to them.

    Object--object links may be useful to other project stages, but they must
    never expand segmentation scope.  Segmentation is triggered solely by a
    hand--object interaction link and targets the non-hand endpoint.
    """

    detections = {
        str(detection.get("detection_id")): detection
        for detection in frame.get("detections", [])
    }
    linked_hand_ids_by_object: dict[str, set[str]] = {}
    for link in frame.get("links", {}).get("hf", []):
        source_id = str(link["source_detection_id"])
        target_id = str(link["target_detection_id"])
        source = detections.get(source_id)
        target = detections.get(target_id)
        if source is None or target is None:
            continue
        if source.get("class_name") == "hand" and target.get("class_name") != "hand":
            hand_id, object_id = source_id, target_id
        elif target.get("class_name") == "hand" and source.get("class_name") != "hand":
            hand_id, object_id = target_id, source_id
        else:
            continue
        linked_hand_ids_by_object.setdefault(object_id, set()).add(hand_id)
    return [
        {
            "source_detection_id": str(detection["detection_id"]),
            "box_xyxy": [float(value) for value in detection["box_xyxy"]],
            "linked_hand_ids": sorted(
                linked_hand_ids_by_object[str(detection["detection_id"])]
            ),
        }
        for detection in frame.get("detections", [])
        if str(detection.get("detection_id")) in linked_hand_ids_by_object
        and detection.get("class_name") != "hand"
    ]


def _linked_object_boxes(frame: dict) -> list[list[float]]:
    """Return the boxes from :func:`_linked_object_prompts`."""

    return [prompt["box_xyxy"] for prompt in _linked_object_prompts(frame)]


def _distinct_hand_box_prompt_decision(
    candidates: list[InstanceCandidate],
    *,
    prompt_by_candidate_id: dict[str, dict],
    max_candidate_overlap_fraction: float,
) -> dict | None:
    """Trust separate, non-overlapping objects linked to separate hands.

    Relative-motion evidence remains necessary when one hand has multiple
    object candidates or when candidates linked to different hands overlap.
    """

    box_candidates = [
        candidate
        for candidate in candidates
        if candidate.candidate_id in prompt_by_candidate_id
    ]
    if len(box_candidates) < 2:
        return None

    prompts = [prompt_by_candidate_id[item.candidate_id] for item in box_candidates]
    if any(len(prompt["linked_hand_ids"]) != 1 for prompt in prompts):
        return None
    hand_ids = [prompt["linked_hand_ids"][0] for prompt in prompts]
    if len(set(hand_ids)) != len(hand_ids):
        return None
    object_ids = [prompt["source_detection_id"] for prompt in prompts]
    if len(set(object_ids)) != len(object_ids):
        return None

    pair_overlaps = []
    for first_index, first in enumerate(box_candidates):
        first_mask = np.asarray(first.mask, dtype=bool)
        for second in box_candidates[first_index + 1 :]:
            second_mask = np.asarray(second.mask, dtype=bool)
            denominator = min(
                int(np.count_nonzero(first_mask)),
                int(np.count_nonzero(second_mask)),
            )
            overlap = (
                float(np.count_nonzero(first_mask & second_mask) / denominator)
                if denominator
                else 0.0
            )
            pair_overlaps.append(
                {
                    "first_candidate_id": first.candidate_id,
                    "second_candidate_id": second.candidate_id,
                    "overlap_fraction": overlap,
                }
            )
            if overlap > max_candidate_overlap_fraction:
                return None

    return {
        "status": "multiple_components",
        "selected_candidate_ids": [item.candidate_id for item in box_candidates],
        "reason": "distinct_hand_object_links_with_disjoint_masks",
        "linked_hand_ids": hand_ids,
        "source_detection_ids": object_ids,
        "pair_overlaps": pair_overlaps,
    }


def _resolve_direct_box_prompt_ownership(
    candidates: list[InstanceCandidate],
    *,
    component_decision: dict,
    logits_by_candidate_id: dict[str, np.ndarray],
) -> list[InstanceCandidate]:
    """Make trusted distinct-hand seed masks exactly disjoint using SAM logits."""

    if component_decision.get("reason") != (
        "distinct_hand_object_links_with_disjoint_masks"
    ):
        return candidates
    selected_ids = list(component_decision["selected_candidate_ids"])
    missing = sorted(set(selected_ids).difference(logits_by_candidate_id))
    if missing:
        raise ValueError(f"missing box-prompt logits for ownership: {missing}")
    ownership = resolve_visible_mask_ownership(
        {candidate_id: logits_by_candidate_id[candidate_id] for candidate_id in selected_ids}
    )
    component_decision["ownership_resolution"] = {
        "method": "highest_sam2_logit",
        "overlap_pixels_before": ownership.overlap_pixels_before,
        "overlap_pixels_after": ownership.overlap_pixels_after,
    }
    selected_set = set(selected_ids)
    return [
        InstanceCandidate(
            candidate_id=candidate.candidate_id,
            mask=ownership.masks[candidate.candidate_id],
            quality_score=candidate.quality_score,
            source=candidate.source,
        )
        if candidate.candidate_id in selected_set
        else candidate
        for candidate in candidates
    ]


def _hand_link_occlusion_fraction(frame: dict) -> float | None:
    """Return the greatest hand coverage fraction of a hand-linked object box."""

    detections = {
        str(detection.get("detection_id")): detection
        for detection in frame.get("detections", [])
    }
    fractions = []
    for link in frame.get("links", {}).get("hf", []):
        source = detections.get(str(link["source_detection_id"]))
        target = detections.get(str(link["target_detection_id"]))
        if source is None or target is None:
            continue
        if source.get("class_name") == "hand":
            hand, visual_object = source, target
        elif target.get("class_name") == "hand":
            hand, visual_object = target, source
        else:
            continue
        hx1, hy1, hx2, hy2 = (float(value) for value in hand["box_xyxy"])
        ox1, oy1, ox2, oy2 = (float(value) for value in visual_object["box_xyxy"])
        intersection = max(0.0, min(hx2, ox2) - max(hx1, ox1)) * max(
            0.0, min(hy2, oy2) - max(hy1, oy1)
        )
        object_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
        if object_area > 0.0:
            fractions.append(intersection / object_area)
    return max(fractions) if fractions else None


def rank_interaction_seed_frames(
    detections: dict,
    *,
    interaction_start_frame: int,
    interaction_end_frame: int,
    max_seed_frames: int,
) -> list[int]:
    """Choose the interaction start/end frames as SAM2 seed candidates.

    A component may be separate only at one end of an interaction: a start
    seed preserves parts that subsequently merge, while an end seed lets a
    later separation propagate backward.  The caller evaluates both and keeps
    the candidate that establishes the most reliable independent components.
    """

    if max_seed_frames < 1:
        raise ValueError("max_seed_frames must be positive")
    frames = detections.get("frames", [])
    if not 0 <= interaction_start_frame <= interaction_end_frame < len(frames):
        raise ValueError("interaction window is outside detection frames")
    linked_frames = []
    for frame_idx in range(interaction_start_frame, interaction_end_frame + 1):
        frame = frames[frame_idx]
        if int(frame.get("frame_idx", -1)) == frame_idx and _linked_object_boxes(frame):
            linked_frames.append(frame_idx)
    if not linked_frames:
        return []

    # The public end keyframe deliberately occurs just after the falling edge,
    # when the interaction mask should disappear.  It therefore often has no
    # HOI box.  Use the last still-linked frame as the reverse-propagation seed.
    selected = []
    for frame_idx in (linked_frames[0], linked_frames[-1]):
        if frame_idx not in selected:
            selected.append(frame_idx)
        if len(selected) >= max_seed_frames:
            break
    return selected


def _candidate_pool(
    amg_summary: dict,
    *,
    interaction_roi: np.ndarray,
    max_seed_candidates: int,
    min_candidate_roi_fraction: float,
) -> list[InstanceCandidate]:
    if max_seed_candidates < 1:
        raise ValueError("max_seed_candidates must be positive")
    candidates = []
    for record in amg_summary.get("candidates", []):
        import cv2

        image = cv2.imread(str(record["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"cannot read AMG mask: {record['mask']}")
        mask = image > 0
        if mask.shape != interaction_roi.shape:
            raise ValueError("AMG candidate and interaction ROI shapes differ")
        area = int(np.count_nonzero(mask))
        roi_fraction = int(np.count_nonzero(mask & interaction_roi)) / area if area else 0.0
        if roi_fraction < min_candidate_roi_fraction:
            continue
        quality = float(record["predicted_iou"]) * float(record["stability_score"])
        candidates.append(
            (
                quality * roi_fraction,
                InstanceCandidate(
                    candidate_id=f"candidate_{int(record['candidate_idx']):03d}",
                    mask=mask,
                    quality_score=quality,
                    source="sam2_amg_initial_candidate",
                ),
            )
        )
    candidates.sort(key=lambda item: (item[0], item[1].candidate_id), reverse=True)
    return [item[1] for item in candidates[:max_seed_candidates]]


def _box_prompt_candidate_pool(
    predictor,
    *,
    video_path: Path,
    source_frame: int,
    boxes: list[list[float]],
    interaction_roi: np.ndarray,
    max_seed_candidates: int,
    min_candidate_roi_fraction: float,
    logits_by_candidate_id: dict[str, np.ndarray] | None = None,
) -> list[InstanceCandidate]:
    """Create SAM2 candidates from hand-linked visual-object boxes.

    The detector box is a prompt, never an output mask. These candidates are
    the primary seed source because they preserve the interaction evidence
    supplied by HOI-DETR.
    """

    candidates = []
    for index, box in enumerate(boxes[:max_seed_candidates]):
        candidate_id = f"box_prompt_candidate_{index:03d}"
        try:
            seed, logits = object_seed_from_box(
                predictor,
                video_path=video_path,
                object_id=candidate_id,
                frame_idx=source_frame,
                box_xyxy=box,
                init_state_kwargs={"offload_video_to_cpu": True},
            )
        except RuntimeError:
            continue
        mask = np.asarray(seed.mask, dtype=bool)
        area = int(np.count_nonzero(mask))
        roi_fraction = int(np.count_nonzero(mask & interaction_roi)) / area if area else 0.0
        if not area or roi_fraction < min_candidate_roi_fraction:
            continue
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            continue
        candidates.append(
            InstanceCandidate(
                candidate_id=candidate_id,
                mask=mask,
                quality_score=roi_fraction,
                source="sam2_hand_linked_box_prompt",
            )
        )
        if logits_by_candidate_id is not None:
            logits_by_candidate_id[candidate_id] = np.asarray(logits, dtype=np.float32)
    return candidates


def run(args: argparse.Namespace) -> dict:
    if not args.interaction_start_frame <= args.source_frame <= args.interaction_end_frame:
        raise ValueError("source frame must lie inside the interaction window")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    detections = json.loads(args.detections.read_text(encoding="utf-8"))
    source_detection_frame = detections["frames"][args.source_frame]
    prompts = _linked_object_prompts(source_detection_frame)
    boxes = [prompt["box_xyxy"] for prompt in prompts]
    if not boxes:
        raise ValueError("source interaction frame contains no hand-linked visual object box")
    interaction_roi = _box_region(
        (int(detections["video"]["height"]), int(detections["video"]["width"])),
        boxes,
    )
    sys.path.insert(0, str(args.sam2_root.resolve()))
    from sam2.build_sam import build_sam2_video_predictor
    import torch

    predictor = build_sam2_video_predictor(
        args.model_cfg,
        str(args.checkpoint.resolve()),
        device=f"cuda:{args.gpu}",
        apply_postprocessing=True,
    )
    box_prompt_logits: dict[str, np.ndarray] = {}
    candidates = _box_prompt_candidate_pool(
        predictor,
        video_path=args.video,
        source_frame=args.source_frame,
        boxes=boxes,
        interaction_roi=interaction_roi,
        max_seed_candidates=args.max_seed_candidates,
        min_candidate_roi_fraction=args.min_candidate_roi_fraction,
        logits_by_candidate_id=box_prompt_logits,
    )
    prompt_by_candidate_id = {
        f"box_prompt_candidate_{index:03d}": prompt
        for index, prompt in enumerate(prompts[: args.max_seed_candidates])
    }
    candidate_source = "sam2_hand_linked_box_prompt_primary"
    if not candidates:
        # AMG loads a separate SAM2 image model. Release the video predictor
        # first so the fallback cannot retain both models on a 16 GB GPU.
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        amg_summary = run_sam2_amg(
            argparse.Namespace(
                video=args.video,
                frame_idx=args.source_frame,
                detections=args.detections,
                output_dir=output_dir / "amg",
                sam2_root=args.sam2_root,
                checkpoint=args.checkpoint,
                model_cfg=args.model_cfg,
                gpu=args.gpu,
                points_per_side=args.points_per_side,
                points_per_batch=args.points_per_batch,
                pred_iou_threshold=args.pred_iou_threshold,
                stability_threshold=args.stability_threshold,
                min_area=args.min_area,
                max_area=args.max_area,
                min_largest_component_fraction=args.min_largest_component_fraction,
                max_hand_overlap=args.max_hand_overlap,
            )
        )
        if amg_summary.get("status") != "success":
            summary = {
                "schema_version": "instance_seed_discovery_v1",
                "status": amg_summary.get("status", "failed_amg"),
                "reconstruction_registry": None,
                "failed_videos": [str(args.video.resolve())],
            }
            write_json_atomic(output_dir / "summary.json", summary)
            return summary
        candidates = _candidate_pool(
            amg_summary,
            interaction_roi=interaction_roi,
            max_seed_candidates=args.max_seed_candidates,
            min_candidate_roi_fraction=args.min_candidate_roi_fraction,
        )
        candidate_source = "sam2_automatic_mask_generator_fallback"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        predictor = build_sam2_video_predictor(
            args.model_cfg,
            str(args.checkpoint.resolve()),
            device=f"cuda:{args.gpu}",
            apply_postprocessing=True,
        )
    if not candidates:
        summary = {
            "schema_version": "instance_seed_discovery_v1",
            "status": "failed_no_candidates_in_interaction_roi",
            "reconstruction_registry": None,
            "failed_videos": [str(args.video.resolve())],
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return summary
    candidates.extend(
        derive_residual_instance_candidates(
            candidates,
            min_child_inside_parent=args.min_child_inside_parent,
            min_residual_area=args.min_residual_area,
        )
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        propagation = propagate_independent_object_logits(
            predictor,
            video_path=args.video,
            seeds=[
                ObjectMaskSeed(candidate.candidate_id, args.source_frame, candidate.mask)
                for candidate in candidates
            ],
            init_state_kwargs={"offload_video_to_cpu": True},
            frame_start=args.interaction_start_frame,
            frame_end=args.interaction_end_frame,
        )
    pair_evidence = [
        component_motion_evidence(
            first.candidate_id,
            second.candidate_id,
            propagation.logits_by_frame,
            frame_shape=interaction_roi.shape,
            min_jointly_visible_frames=args.min_jointly_visible_frames,
            min_relative_displacement_diagonals=args.min_relative_displacement_diagonals,
        )
        for index, first in enumerate(candidates)
        for second in candidates[index + 1 :]
    ]
    decision = None
    if candidate_source == "sam2_hand_linked_box_prompt_primary":
        decision = _distinct_hand_box_prompt_decision(
            candidates,
            prompt_by_candidate_id=prompt_by_candidate_id,
            max_candidate_overlap_fraction=args.max_cross_instance_overlap_fraction,
        )
    if decision is None:
        decision = select_initial_instance_hypothesis(
            candidates,
            interaction_roi=interaction_roi,
            pair_evidence=pair_evidence,
            min_candidate_roi_fraction=args.min_candidate_roi_fraction,
            max_candidate_overlap_fraction=args.max_cross_instance_overlap_fraction,
            max_components=args.max_components,
            min_multi_component_score_gain=args.min_multi_component_score_gain,
        )
    candidates = _resolve_direct_box_prompt_ownership(
        candidates,
        component_decision=decision,
        logits_by_candidate_id=box_prompt_logits,
    )
    if decision["status"].startswith("failed"):
        summary = {
            "schema_version": "instance_seed_discovery_v1",
            "status": decision["status"],
            "component_decision": decision,
            "reconstruction_registry": None,
            "failed_videos": [str(args.video.resolve())],
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return summary
    registry = write_instance_registry(
        candidates,
        source_frame=args.source_frame,
        interaction_start_frame=args.interaction_start_frame,
        interaction_end_frame=args.interaction_end_frame,
        component_decision=decision,
        output_dir=output_dir / "registry",
    )
    summary = {
        "schema_version": "instance_seed_discovery_v1",
        "status": "success",
        "source_frame": args.source_frame,
        "interaction_window": [args.interaction_start_frame, args.interaction_end_frame],
        "candidate_source": candidate_source,
        "candidate_count": len(candidates),
        "component_decision": decision,
        "reconstruction_registry": registry["reconstruction_registry"],
        "failed_videos": [],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--source-frame", type=int, required=True)
    parser.add_argument("--interaction-start-frame", type=int, required=True)
    parser.add_argument("--interaction-end-frame", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--points-per-batch", type=int, default=32)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.8)
    parser.add_argument("--stability-threshold", type=float, default=0.92)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--max-area", type=int, default=100000)
    parser.add_argument("--min-largest-component-fraction", type=float, default=0.98)
    parser.add_argument("--max-hand-overlap", type=float, default=0.1)
    parser.add_argument("--max-seed-candidates", type=int, default=6)
    parser.add_argument("--min-candidate-roi-fraction", type=float, default=0.15)
    parser.add_argument("--min-child-inside-parent", type=float, default=0.9)
    parser.add_argument("--min-residual-area", type=int, default=256)
    parser.add_argument("--min-jointly-visible-frames", type=int, default=3)
    parser.add_argument("--min-relative-displacement-diagonals", type=float, default=0.03)
    parser.add_argument("--max-cross-instance-overlap-fraction", type=float, default=0.05)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--min-multi-component-score-gain", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:
        summary = {
            "schema_version": "instance_seed_discovery_v1",
            "status": "failed_fatal_error",
            "failures": [{"reason": f"{type(exc).__name__}: {exc}"}],
            "reconstruction_registry": None,
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
