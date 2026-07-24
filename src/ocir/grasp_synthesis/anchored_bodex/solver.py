"""Anchored-BODex orchestration: demo -> guidance -> optimize -> ranked records.

Mirrors ``bodex_curobo_v2.solver.solve_sharpa_bodex`` (two rollout instances,
``BodexNewtonOpt`` inside ``MultiStageOptimizer``, identical strict-success
semantics -- success stays pure force-closure + contact distance), with the
human-guidance pipeline in front and similarity-aware ranking behind:

    sequence dir -> ObjectSurface + HumanDemo
                 -> ensure_affordance (lazy per-sequence cache)
                 -> analyze_demo (grasp window, contact roles)
                 -> HandFitter + AnchoredSeedGenerator (retargeted seeds)
                 -> AnchoredBodexRollout (contact subset + guidance energies)
                 -> optimize -> exact metrics + affordance/pose similarity
                 -> grasp_*.json / summary.json
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocir.grasp_synthesis.anchored_bodex.affordance import Affordance, ensure_affordance
from ocir.grasp_synthesis.anchored_bodex.demo_analysis import ROLE_ORDER, DemoGraspAnalysis, analyze_demo
from ocir.grasp_synthesis.anchored_bodex.demo_data import HumanDemo
from ocir.grasp_synthesis.anchored_bodex.guidance import (
    GuidanceWeights,
    build_pressure_constraints,
    select_contact_points,
)
from ocir.grasp_synthesis.anchored_bodex.grasp_stages import (
    DEFAULT_PREGRASP_CLEARANCE_M,
    SnapshotBodexNewtonOpt,
    compute_grasp_stages,
    stage_pose_dict,
)
from ocir.grasp_synthesis.anchored_bodex.retarget import HandFitter, load_mano_transfer, transfer_path
from ocir.grasp_synthesis.anchored_bodex.rollout import AnchoredBodexRollout
from ocir.grasp_synthesis.anchored_bodex.seed_generator import AnchoredSeedGenerator
from ocir.grasp_synthesis.assets import load_sharpa_wave_right
from ocir.grasp_synthesis.bodex_curobo_v2.newton_opt import BodexNewtonOptCfg
from ocir.grasp_synthesis.clearance import ClearanceChecker
from ocir.grasp_synthesis.bodex_curobo_v2.solver import (
    DEFAULT_CONTACT_STRATEGY,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_GRASP_THRESHOLD,
    build_metric_summary,
    compute_success,
    expand_active_action_to_full_joint_order,
    find_object_mesh_from_surface,
)
from ocir.grasp_synthesis.object_surface import ObjectSurface

from ocir.grasp_synthesis.bodex_curobo_v2.backend import import_official_curobo

import_official_curobo()

from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.types.device_cfg import DeviceCfg

BACKEND_NAME = "curobo_v2_anchored_bodex"
DEFAULT_AFFORD_TAU = 0.3


@dataclass(frozen=True)
class SimilarityMetrics:
    affordance_coverage: np.ndarray   # (B,) mean heatmap value at contacts
    affordance_hit_fraction: np.ndarray  # (B,) contacts on heatmap >= tau
    wrist_pos_m: np.ndarray           # (B,) |final - anchor| wrist translation
    wrist_rot_deg: np.ndarray         # (B,) final-vs-anchor wrist rotation
    joint_rms_rad: np.ndarray         # (B,)


def compute_similarity_metrics(
    final_actions: torch.Tensor,      # (B, 7+J) rollout joint order
    ref_actions: torch.Tensor,        # (B, 7+J) per-seed anchors, same order
    contact_points_world: torch.Tensor,  # (B, n_contacts, 3)
    affordance: Affordance,
    afford_tau: float = DEFAULT_AFFORD_TAU,
) -> SimilarityMetrics:
    final_np = final_actions.detach().cpu().numpy()
    ref_np = ref_actions.detach().cpu().numpy()

    wrist_pos = np.linalg.norm(final_np[:, :3] - ref_np[:, :3], axis=-1)
    qdot = np.abs(np.sum(final_np[:, 3:7] * ref_np[:, 3:7], axis=-1)).clip(max=1.0)
    wrist_rot = np.rad2deg(2.0 * np.arccos(qdot))
    joint_rms = np.sqrt(((final_np[:, 7:] - ref_np[:, 7:]) ** 2).mean(axis=-1))

    contacts = contact_points_world.detach().cpu().numpy().reshape(final_np.shape[0], -1, 3)
    points = affordance.points_object_frame
    heat = affordance.heatmap
    coverage = np.zeros((contacts.shape[0],))
    hits = np.zeros((contacts.shape[0],))
    for b in range(contacts.shape[0]):
        d2 = ((contacts[b][:, None, :] - points[None, :, :]) ** 2).sum(-1)
        nearest = d2.argmin(axis=-1)
        values = heat[nearest]
        coverage[b] = float(values.mean())
        hits[b] = float((values >= afford_tau).mean())
    return SimilarityMetrics(
        affordance_coverage=coverage,
        affordance_hit_fraction=hits,
        wrist_pos_m=wrist_pos,
        wrist_rot_deg=wrist_rot,
        joint_rms_rad=joint_rms,
    )


def rank_scores(
    costs: np.ndarray,
    similarity: SimilarityMetrics,
    *,
    rank_affordance_weight: float,
    rank_pose_weight: float,
) -> np.ndarray:
    pose_dev = (
        similarity.wrist_pos_m / 0.05 + similarity.wrist_rot_deg / 45.0 + similarity.joint_rms_rad / 0.5
    ) / 3.0
    return costs + rank_affordance_weight * (1.0 - similarity.affordance_coverage) + rank_pose_weight * pose_dev


def solve_sharpa_anchored_bodex(
    sequence_dir: Path,
    out_dir: Path,
    object_mesh: Path | None = None,
    seeds: int = 20,
    top_k: int = 5,
    opt_iters: int = 500,
    seed: int = 0,
    grasp_threshold: float = DEFAULT_GRASP_THRESHOLD,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    *,
    relax_flexion: float = 0.15,
    relax_standoff: float = 0.015,
    jitter_pos: float = 0.01,
    jitter_rot_deg: float = 10.0,
    jitter_joint: float = 0.08,
    affordance_weight: float = 20.0,
    pose_weight: float = 0.0,
    contact_subset: bool = False,
    afford_tau: float = DEFAULT_AFFORD_TAU,
    rank_affordance_weight: float = 1.0,
    rank_pose_weight: float = 0.5,
    force_affordance: bool = False,
    penetration_weight: float = 0.0,
    selfcollision_weight: float = 1000.0,
    pregrasp_clearance_m: float = DEFAULT_PREGRASP_CLEARANCE_M,
    force_closure_weight: float = 500.0,
    table_penalty_weight: float = 0.0,
    table_up_object: "np.ndarray | None" = None,
    world_up: "tuple[float, float, float]" = (0.0, 0.0, 1.0),
    approach_dir_object: "np.ndarray | None" = None,
    approach_cone_cos: float = 0.174,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("anchored BODex grasp synthesis requires CUDA")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

    surface = ObjectSurface.from_sequence_dir(sequence_dir)
    object_mesh_path = find_object_mesh_from_surface(surface, object_mesh)
    demo = HumanDemo.from_sequence_dir(sequence_dir)
    affordance = ensure_affordance(
        sequence_dir, demo, surface.points_object_frame, force=force_affordance
    )
    analysis = analyze_demo(demo, affordance)

    # Hand-table collision penalty geometry (object frame). The object rests on
    # the table at its reconstructed orientation; "up" in the object frame is the
    # world-up rotated into the object frame. Derive it from the demo's object
    # pose at the first valid frame unless the caller passes it explicitly.
    table_up_obj = None
    table_offset_val = 0.0
    if table_penalty_weight > 0.0:
        if table_up_object is not None:
            table_up_obj = np.asarray(table_up_object, dtype=np.float64).reshape(3)
        else:
            v0 = int(np.flatnonzero(demo.valid_mask)[0])
            R_wo = np.asarray(demo.object_pose_camera[v0][:3, :3], dtype=np.float64)
            table_up_obj = R_wo.T @ np.asarray(world_up, dtype=np.float64)
        table_up_obj = table_up_obj / (np.linalg.norm(table_up_obj) + 1e-12)
        # tabletop plane = the object's lowest extent along up (it rests there)
        pts = np.asarray(surface.points_object_frame, dtype=np.float64)
        table_offset_val = float((pts @ table_up_obj).min())

    asset = load_sharpa_wave_right()
    calib_path = transfer_path(asset)
    if not calib_path.exists():
        raise FileNotFoundError(
            f"missing MANO->Sharpa calibration {calib_path}; run "
            "scripts/grasp_synthesis/calibrate_mano_sharpa.py once to generate it"
        )
    calib = load_mano_transfer(calib_path)
    fitter = HandFitter(asset, calib, device_cfg)
    seed_generator = AnchoredSeedGenerator(
        fitter=fitter,
        demo=demo,
        affordance=affordance,
        analysis=analysis,
        relax_flexion=relax_flexion,
        relax_standoff=relax_standoff,
        jitter_pos=jitter_pos,
        jitter_rot_deg=jitter_rot_deg,
        jitter_joint=jitter_joint,
        seed=1312 + seed,
    )

    active_roles = analysis.active_roles if contact_subset else ROLE_ORDER
    contact_points = select_contact_points(active_roles)
    pressure_constraints = build_pressure_constraints(active_roles)

    afford_mask = affordance.heatmap >= afford_tau
    # narrow the affordance guidance region to the side the human hand approached
    # from (path-derived direction). Same wide-cone geometry as
    # affordance_seed.load_affordance_region, with a min-points fallback so a
    # noisy direction never empties the region. Orientation (thumb-down) and
    # table clearance stay handled by the pose prior and table penalty -- this
    # only trims WHERE on the object the contacts are pulled toward.
    afford_before = int(afford_mask.sum())
    if approach_dir_object is not None and afford_before > 0:
        from ocir.grasp_synthesis.affordance_seed import approach_cone_mask
        cone_keep = approach_cone_mask(
            object_mesh_path,
            affordance.points_object_frame[afford_mask],
            approach_dir_object,
            cone_cos=approach_cone_cos,
        )
        if int(cone_keep.sum()) >= 8:
            drop_idx = np.flatnonzero(afford_mask)[~cone_keep]
            afford_mask = afford_mask.copy()
            afford_mask[drop_idx] = False
            print(
                f"[anchored_bodex] approach-cone: afford region {afford_before} -> "
                f"{int(afford_mask.sum())} points "
                f"(dir={np.round(np.asarray(approach_dir_object, dtype=float), 3).tolist()})",
                flush=True,
            )
        else:
            print(
                f"[anchored_bodex] approach-cone SKIPPED (would leave "
                f"{int(cone_keep.sum())} < 8 points; kept {afford_before})",
                flush=True,
            )
    afford_points = (
        device_cfg.to_device(affordance.points_object_frame[afford_mask].astype(np.float32))
        if afford_mask.any()
        else None
    )
    weights = GuidanceWeights.from_contact_strategy(
        DEFAULT_CONTACT_STRATEGY,
        w_afford=affordance_weight,
        pose_scale=pose_weight,
        w_pene=penetration_weight,
        w_selfcol=selfcollision_weight,
    )

    pts = np.asarray(surface.points_object_frame, dtype=np.float32)
    center = pts.mean(axis=0)
    radius = float(max(np.linalg.norm(pts - center[None, :], axis=1).max() + 0.18, 0.25))
    rollout_cfg = dict(
        device_cfg=device_cfg,
        surface=surface,
        object_mesh_path=object_mesh_path,
        root_bounds_center=center,
        root_bounds_radius=radius,
        contact_points=contact_points,
        pressure_constraints=pressure_constraints,
        seed_generator=seed_generator,
        afford_points=afford_points,
        weights=weights,
        opt_iters=opt_iters,
        force_closure_weight=force_closure_weight,
        table_penalty_weight=table_penalty_weight,
        table_up_object=table_up_obj,
        table_offset=table_offset_val,
    )
    rollouts = [AnchoredBodexRollout(**rollout_cfg), AnchoredBodexRollout(**rollout_cfg)]

    opt_cfg = BodexNewtonOptCfg(
        num_iters=int(opt_iters),
        inner_iters=50,
        bodex_line_search_scale=[0.1],
        base_scale=[0.01, 0.1, 0.1],
        translation_dim=3,
        quaternion_dim=4,
        momentum=True,
        normalize_grad=True,
        momentum_decay=0.9,
        lr_decay_rate=0.95,
        fixed_iters=True,
        return_best_action=False,
        num_problems=int(seeds),
        device_cfg=device_cfg,
    )
    # Snapshot variant of the BODex optimizer: records the per-seed actions
    # the moment the staged contact cost enters its middle (1cm-standoff)
    # stage -- i.e. the end of phase 0, the pose optimized under the initial
    # ~2cm-standoff target. These become each record's "pregrasp" stage
    # (BODex's save_qpos), the loosest of the three staged targets.
    snapshot_progress = float(DEFAULT_CONTACT_STRATEGY["opt_progress"][1])
    newton_opt = SnapshotBodexNewtonOpt(
        opt_cfg, rollouts, use_cuda_graph=False, snapshot_progress=snapshot_progress
    )
    optimizer = MultiStageOptimizer([newton_opt], rollouts)
    optimizer.update_num_problems(seeds)
    for rollout in rollouts:
        rollout.batch_size = seeds
    init_action = rollouts[0].get_initial_action(use_random=True)
    optimizer.reinitialize(init_action)
    result = optimizer.optimize(init_action)

    pregrasp_result = newton_opt.pregrasp_actions
    if pregrasp_result is None:
        print(
            "[anchored_bodex] WARNING: optimizer never reached snapshot progress "
            f"{snapshot_progress}; using the final actions as pregrasp"
        )
        pregrasp_result = result
    pregrasp_cpu = pregrasp_result.detach()
    stage_clearance_checker = ClearanceChecker(asset, device_cfg)

    metrics = rollouts[0].compute_metrics_from_action(result, opt_progress=1.0)
    costs = metrics.costs_and_constraints.get_sum_cost(sum_horizon=True).detach()
    exact = rollouts[0].evaluate_exact_grasp_metrics(result)
    success, grasp_error_max = compute_success(
        exact["grasp_error"], exact["dist_error"], grasp_threshold, distance_threshold
    )
    dist_error = exact["dist_error"].detach()
    grasp_error_max = grasp_error_max.detach()

    ref_actions = rollouts[0]._remap_seed_actions(seed_generator.ref_actions)
    similarity = compute_similarity_metrics(
        final_actions=result[:, 0, :],
        ref_actions=ref_actions,
        contact_points_world=exact["contact_point"].reshape(result.shape[0], -1, 3),
        affordance=affordance,
        afford_tau=afford_tau,
    )
    costs_np = costs.cpu().numpy()
    scores = rank_scores(
        costs_np,
        similarity,
        rank_affordance_weight=rank_affordance_weight,
        rank_pose_weight=rank_pose_weight,
    )

    metric_summary = build_metric_summary(
        success=success,
        grasp_error_max=grasp_error_max,
        dist_error=dist_error,
        costs=costs,
        grasp_threshold=grasp_threshold,
        distance_threshold=distance_threshold,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("grasp_pose_*.json", "grasp_*.json", "failed_grasp_*.json"):
        for path in out_dir.glob(pattern):
            path.unlink()

    successful_seed_count = int(success.sum().item())
    result_cpu = result.detach()
    ref_np = ref_actions.detach().cpu().numpy()
    guidance_common = {
        "backend": BACKEND_NAME,
        "human_guided": True,
        "grasp_frame_id": analysis.report["grasp_frame_id"],
        "pickup_frame_id": analysis.report["pickup_frame_id"],
        "active_contact_roles": list(active_roles),
        "contact_points_used": list(contact_points),
        "pressure_constraints_used": [[list(g), c] for g, c in pressure_constraints],
        "contact_subset_enabled": bool(contact_subset),
        "guidance_weights": {
            "force_closure_weight": float(force_closure_weight),
            "affordance_weight": float(weights.w_afford),
            "pose_weights": list(weights.w_pose),
            "pose_anneal_end": float(weights.pose_anneal_end),
            "afford_decay": list(weights.afford_decay),
            "afford_tau": float(afford_tau),
            "penetration_weight": float(weights.w_pene),
            "penetration_ramp": list(weights.pene_ramp),
            "selfcollision_weight": float(weights.w_selfcol),
        },
        "seed_params": {
            "relax_flexion": float(relax_flexion),
            "relax_standoff": float(relax_standoff),
            "jitter_pos": float(jitter_pos),
            "jitter_rot_deg": float(jitter_rot_deg),
            "jitter_joint": float(jitter_joint),
        },
        "retarget_report": seed_generator.retarget_result.report,
        "demo_analysis": analysis.report,
    }

    def _expand_full(active_action: np.ndarray) -> np.ndarray:
        active_action = active_action.copy()
        active_action[3:7] = active_action[3:7] / max(np.linalg.norm(active_action[3:7]), 1e-9)
        full = expand_active_action_to_full_joint_order(
            active_action, rollouts[0].joint_names, rollouts[0].full_joint_order, rollouts[0].full_neutral_q
        )
        full[3:7] = full[3:7] / max(np.linalg.norm(full[3:7]), 1e-9)
        return full

    def _record(seed_idx: int, rank: int) -> tuple[dict, Path]:
        optimized_action = result_cpu[seed_idx, 0].cpu().numpy().copy()
        optimized_action[3:7] = optimized_action[3:7] / max(np.linalg.norm(optimized_action[3:7]), 1e-9)
        full_action = _expand_full(result_cpu[seed_idx, 0].cpu().numpy())
        pregrasp_full = _expand_full(pregrasp_cpu[seed_idx, 0].cpu().numpy())
        stages = compute_grasp_stages(
            full_action,
            pregrasp_full,
            rollouts[0].full_joint_order,
            stage_clearance_checker,
            rollouts[0].contact_world,
            pregrasp_clearance_m=pregrasp_clearance_m,
        )
        record = {
            "ok": bool(success[seed_idx].item()),
            **guidance_common,
            "hand": "sharpa_wave_right",
            "rank": int(rank),
            "seed_index": int(seed_idx),
            "anchor_frame_index": int(seed_generator.seed_frame_indices[seed_idx]),
            "anchor_frame_id": int(demo.frame_ids[seed_generator.seed_frame_indices[seed_idx]]),
            "sequence_id": str(surface.metadata.get("sequence_id")) if surface.metadata.get("sequence_id") is not None else None,
            "object_name": str(surface.metadata.get("object_name")) if surface.metadata.get("object_name") is not None else None,
            "object_mesh": str(object_mesh_path),
            "object_gravity_center": rollouts[0].object_gravity_center.detach().cpu().numpy().reshape(-1).tolist(),
            "object_obb_length": float(rollouts[0].object_obb_length.detach().cpu().item()),
            "sequence_dir": str(sequence_dir),
            "action": full_action.astype(float).tolist(),
            "joint_names": rollouts[0].full_joint_order,
            # Two-stage poses (see anchored_bodex/grasp_stages.py):
            # grasp = the fully optimized action as-is (also the pose held
            # through the whole close/settle/carry); pregrasp = stage-0
            # snapshot with the open-mask joints (flexion + both thumb-CMC
            # DoFs) opened until the hand clears the object (wrist pose
            # untouched).
            "stages": {
                "pregrasp": stage_pose_dict(stages.pregrasp, rollouts[0].full_joint_order),
                "grasp": stage_pose_dict(stages.grasp, rollouts[0].full_joint_order),
            },
            "stage_report": stages.report,
            "success": bool(success[seed_idx].item()),
            "successful_seed_count": successful_seed_count,
            "grasp_error_max": float(grasp_error_max[seed_idx].item()),
            "dist_error": float(dist_error[seed_idx].item()),
            "dist_error_final": float(dist_error[seed_idx].item()),
            "metric_summary": metric_summary,
            "score": float(costs_np[seed_idx]),
            "rank_score": float(scores[seed_idx]),
            "affordance_coverage": float(similarity.affordance_coverage[seed_idx]),
            "affordance_hit_fraction": float(similarity.affordance_hit_fraction[seed_idx]),
            "pose_similarity": {
                "wrist_pos_m": float(similarity.wrist_pos_m[seed_idx]),
                "wrist_rot_deg": float(similarity.wrist_rot_deg[seed_idx]),
                "joint_rms_rad": float(similarity.joint_rms_rad[seed_idx]),
            },
            "anchor_action": ref_np[seed_idx].astype(float).tolist(),
            "optimized_action": optimized_action.astype(float).tolist(),
            "optimized_joint_names": rollouts[0].joint_names,
            "passive_joint_names": rollouts[0].passive_joint_names,
        }
        # 1-indexed ``grasp_pose_N`` -- ranked best-first, same naming whether
        # or not any seed cleared the strict success threshold (the summary
        # still records success per record and overall).
        path = out_dir / f"grasp_pose_{rank + 1}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record, path

    def _write_records(order: list[int]) -> list[dict]:
        entries = []
        for rank, seed_idx in enumerate(order):
            _, path = _record(int(seed_idx), rank)
            entries.append(
                {
                    "rank": int(rank),
                    "seed_index": int(seed_idx),
                    "score": float(costs_np[seed_idx]),
                    "rank_score": float(scores[seed_idx]),
                    "affordance_coverage": float(similarity.affordance_coverage[seed_idx]),
                    "grasp_json": str(path),
                }
            )
        return entries

    summary_common = {
        **guidance_common,
        "hand": "sharpa_wave_right",
        "sequence_id": str(surface.metadata.get("sequence_id")) if surface.metadata.get("sequence_id") is not None else None,
        "object_name": str(surface.metadata.get("object_name")) if surface.metadata.get("object_name") is not None else None,
        "object_mesh": str(object_mesh_path),
        "sequence_dir": str(sequence_dir),
        "metric_summary": metric_summary,
        "seed_count": int(seeds),
        "opt_iters": int(opt_iters),
    }

    if successful_seed_count == 0:
        failed_order = list(np.argsort(scores)[: min(int(top_k), int(seeds))])
        top_failed_grasps = _write_records(failed_order)
        summary = {
            "ok": False,
            **summary_common,
            "success": False,
            "successful_seed_count": 0,
            "top_k": 0,
            "top_grasps": [],
            "top_failed_grasps": top_failed_grasps,
            "failed_grasp_json": str(top_failed_grasps[0]["grasp_json"]) if top_failed_grasps else None,
            "error": "No successful grasp seeds under the configured thresholds.",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    success_np = success.cpu().numpy()
    candidate_indices = np.flatnonzero(success_np)
    candidate_scores = scores[candidate_indices]
    order = list(candidate_indices[np.argsort(candidate_scores)[: min(int(top_k), len(candidate_indices))]])
    top_grasps = _write_records(order)

    best = json.loads(Path(top_grasps[0]["grasp_json"]).read_text(encoding="utf-8"))
    summary = {
        **best,
        "ok": bool(best.get("success", False)),
        "seed_count": int(seeds),
        "top_k": len(top_grasps),
        "top_grasps": top_grasps,
        "opt_iters": int(opt_iters),
        "grasp_json": str(top_grasps[0]["grasp_json"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
