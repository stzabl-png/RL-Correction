"""Stage A orchestration: human demo -> retarget -> synthetic approach/close/
squeeze -> carry -> ``GraspTrajectory``. Grasp-synthesis conda env (torch/CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from curobo._src.types.device_cfg import DeviceCfg

from ocir.grasp_synthesis.assets import SharpaWaveAsset
from ocir.grasp_synthesis.object_surface import ObjectSurface
from ocir.grasp_synthesis.anchored_bodex.affordance import ensure_affordance
from ocir.grasp_synthesis.anchored_bodex.demo_analysis import analyze_demo
from ocir.grasp_synthesis.anchored_bodex.demo_data import HumanDemo
from ocir.grasp_synthesis.anchored_bodex.retarget import HandFitter, load_mano_transfer, transfer_path
from ocir.grasp_synthesis.anchored_bodex.seed_generator import relax_joint_mask
from ocir.grasp_synthesis.bodex_curobo_v2.solver import _load_yaml

from ocir.grasp_traj.clearance import ClearanceChecker, build_contact_world
from ocir.grasp_traj.planner import TransitPlanner
from ocir.grasp_traj.segments import (
    PALM_APPROACH_AXIS_LOCAL,
    blend_into_trajectory,
    interp_trajectory,
    polyline_wrist_trajectory,
    quat_to_matrix,
    slerp_wxyz,
    standoff_pose,
    steps_for_leg,
)
from ocir.grasp_traj.switch_frame import select_switch_frame
from ocir.grasp_traj.trajectory_schema import (
    SEGMENT_APPROACH,
    SEGMENT_CARRY,
    SEGMENT_CLOSE,
    SEGMENT_RETARGET,
    SEGMENT_SQUEEZE,
    GraspTrajectory,
    compose_pos_quat,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
)

CARRY_START_GRASP_FRAME = "grasp_frame"
CARRY_START_PICKUP_FRAME = "pickup_frame"


@dataclass
class GraspTrajectoryConfig:
    fps: float = 30.0
    approach_seconds: float = 0.5
    close_seconds: float = 0.3
    squeeze_seconds: float = 0.3
    standoff_m: float = 0.10
    pregrasp_open_fraction: float = 1.0
    squeeze_delta: float = 0.15
    approach_clearance_m: float = 0.01
    carry_start: str = CARRY_START_GRASP_FRAME
    max_wrist_speed_mps: float = 0.25
    carry_blend_seconds: float = 0.3
    open_clearance_m: float = 0.05
    retreat_max_m: float = 0.25
    open_seconds: float = 0.4
    planner: str = "curobo"  # or "linear"
    final_close_seconds: float = 0.4
    near_contact_margin_m: float = 0.003

    @property
    def dt(self) -> float:
        return 1.0 / float(self.fps)

    def close_steps(self) -> int:
        return max(1, int(round(self.close_seconds * self.fps)))

    def squeeze_steps(self) -> int:
        return max(1, int(round(self.squeeze_seconds * self.fps)))


def _remap_by_name(values: np.ndarray, source_names: list[str], target_names: list[str]) -> np.ndarray:
    """(..., len(source_names)) -> (..., len(target_names)), reordering by
    joint name. Raises if ``target_names`` isn't a subset of ``source_names``."""

    if list(source_names) == list(target_names):
        return values
    index = {name: i for i, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in index]
    if missing:
        raise KeyError(f"joint names {missing} not present in source names {source_names}")
    idx = [index[name] for name in target_names]
    return values[..., idx]


class GraspTrajectoryGenerator:
    def __init__(self, asset: SharpaWaveAsset, config: GraspTrajectoryConfig, device_cfg: DeviceCfg | None = None):
        self.asset = asset
        self.config = config
        self.device_cfg = device_cfg or DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        self.joint_order = list(asset.config["joint_order"])
        limits = asset.config["joint_limits"]
        self.joint_lower = np.asarray([limits[name][0] for name in self.joint_order], dtype=np.float64)
        self.joint_upper = np.asarray([limits[name][1] for name in self.joint_order], dtype=np.float64)
        self.relax_mask = relax_joint_mask(self.joint_order).astype(np.float64)

        calib = load_mano_transfer(transfer_path(asset))
        self.fitter = HandFitter(asset, calib, self.device_cfg)
        if list(self.fitter.joint_names) != self.joint_order:
            raise ValueError(
                "HandFitter.joint_names no longer matches asset.config['joint_order']; "
                "the pos-only-remap assumption generator.py relies on needs revisiting "
                f"(fitter={list(self.fitter.joint_names)}, asset={self.joint_order})"
            )
        self.clearance_checker = ClearanceChecker(asset, self.device_cfg)

    def _clamp(self, joints: np.ndarray) -> np.ndarray:
        return np.clip(joints, self.joint_lower, self.joint_upper)

    def _min_clearance(self, world, pos: np.ndarray, quat: np.ndarray, joints: np.ndarray) -> float:
        actions = np.concatenate([pos, quat, joints], axis=-1).astype(np.float32)
        actions_t = self.device_cfg.to_device(actions)
        return float(self.clearance_checker.compute_clearances(actions_t, world).min().item())

    def _project_wrist_pose(
        self,
        world,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        pregrasp_joints: np.ndarray,
        *,
        clearance_target_m: float = 0.0,
        max_backoff_m: float = 0.04,
    ) -> tuple[np.ndarray, float]:
        """Smallest pull-back of the grasp wrist pose along its palm approach
        axis at which the WIDE-OPEN hand no longer penetrates the object.
        Returns ``(corrected_pos, backoff_m)`` (backoff 0 when the pose is
        already clear; capped with a warning when even ``max_backoff_m`` is
        not enough)."""

        grasp_pos = np.asarray(grasp_pos, dtype=np.float64)
        back_dir = -(quat_to_matrix(grasp_quat) @ PALM_APPROACH_AXIS_LOCAL)
        offsets = np.arange(0.0, max_backoff_m + 1e-9, 0.002)
        pos = grasp_pos[None] + offsets[:, None] * back_dir[None]
        quat = np.tile(np.asarray(grasp_quat, dtype=np.float64)[None], (offsets.size, 1))
        joints = np.tile(pregrasp_joints[None], (offsets.size, 1))
        actions = np.concatenate([pos, quat, joints], axis=-1).astype(np.float32)
        clearances = self.clearance_checker.compute_clearances(
            self.device_cfg.to_device(actions), world
        ).cpu().numpy()
        ok = np.where(clearances >= clearance_target_m)[0]
        if ok.size == 0:
            print(
                "[grasp_traj] WARNING: open hand still penetrates the object after backing the "
                f"wrist off {max_backoff_m:.3f} m (clearance {clearances[-1]:.4f} m); using the cap"
            )
            return pos[-1], float(offsets[-1])
        idx = int(ok[0])
        if idx > 0:
            print(f"[grasp_traj] grasp wrist pose backed off {offsets[idx]*1000:.0f} mm to clear the open hand")
        return pos[idx], float(offsets[idx])

    def _project_contact_joints(
        self,
        world,
        grasp_pos: np.ndarray,
        grasp_quat: np.ndarray,
        pregrasp_joints: np.ndarray,
        grasp_joints: np.ndarray,
        clearance_target_m: float,
    ) -> tuple[np.ndarray, float]:
        """Largest closing fraction t (joints = pregrasp + t*(grasp-pregrasp),
        wrist at the grasp pose) whose SDF clearance is still >=
        ``clearance_target_m``. The synthesized (often failed) grasp joints
        can penetrate the object by 5-15mm; commanding positions inside the
        object plows the fingers through it. Returns ``(joints, t)``; t=0
        (fully open) with a warning if even that penetrates."""

        ts = np.linspace(0.0, 1.0, 101)
        joints = pregrasp_joints[None] + ts[:, None] * (grasp_joints - pregrasp_joints)[None]
        pos = np.tile(np.asarray(grasp_pos, dtype=np.float64)[None], (ts.size, 1))
        quat = np.tile(np.asarray(grasp_quat, dtype=np.float64)[None], (ts.size, 1))
        actions = np.concatenate([pos, quat, joints], axis=-1).astype(np.float32)
        clearances = self.clearance_checker.compute_clearances(
            self.device_cfg.to_device(actions), world
        ).cpu().numpy()
        ok = np.where(clearances >= clearance_target_m)[0]
        if ok.size == 0:
            print(
                "[grasp_traj] WARNING: even the fully-open posture at the grasp wrist pose "
                f"has clearance {clearances[0]:.4f} m < {clearance_target_m:.4f} m"
            )
            return pregrasp_joints.copy(), 0.0
        t = float(ts[ok[-1]])
        return joints[ok[-1]].copy(), t

    def _plan_transit(
        self,
        surface: ObjectSurface,
        world,
        switch_pos: np.ndarray,
        switch_quat: np.ndarray,
        switch_joints: np.ndarray,
        standoff_pos: np.ndarray,
        standoff_quat: np.ndarray,
        pregrasp_joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Switch pose -> grasp standoff in three phases:

        1. retreat: pull straight back along the switch pose's palm axis
           (fingers still at the retargeted posture) until the WIDE-OPEN hand
           would clear the object by ``open_clearance_m`` (capped at
           ``retreat_max_m``) -- the hand must not open right next to the
           object or the opening fingers themselves push it away.
        2. open: hold the wrist at the retreat pose, open the fingers to
           pregrasp over ``open_seconds``.
        3. transit: cuRobo v2 MotionPlanner (floating-base hand, fingers
           locked wide open, object mesh as obstacle) from the retreat pose
           to the standoff; straight-line + radial via-point fallback if
           planning is disabled or fails.
        """

        cfg = self.config
        report: dict = {}

        # --- Phase 1: retreat until the open hand would be clear -----------
        switch_pos = np.asarray(switch_pos, dtype=np.float64)
        back_dir = -(quat_to_matrix(switch_quat) @ PALM_APPROACH_AXIS_LOCAL)
        retreat_dist = 0.0
        step = 0.01
        while retreat_dist <= cfg.retreat_max_m:
            candidate = switch_pos + retreat_dist * back_dir
            clearance = self._min_clearance(
                world, candidate[None], np.asarray(switch_quat)[None], pregrasp_joints[None]
            )
            if clearance >= cfg.open_clearance_m:
                break
            retreat_dist += step
        retreat_dist = min(retreat_dist, cfg.retreat_max_m)
        retreat_pos = switch_pos + retreat_dist * back_dir
        report["retreat_distance_m"] = retreat_dist
        report["open_pose_clearance_m"] = clearance

        retreat_steps = steps_for_leg(retreat_dist, 0.2, cfg.fps, cfg.max_wrist_speed_mps)
        retreat_p, retreat_q, retreat_j = interp_trajectory(
            switch_pos, switch_quat, switch_joints,
            retreat_pos, switch_quat, switch_joints,
            retreat_steps, include_start=True,
        )

        # --- Phase 2: open wide, in place, away from the object ------------
        open_steps = max(1, int(round(cfg.open_seconds * cfg.fps)))
        _, _, open_j = interp_trajectory(
            retreat_pos, switch_quat, switch_joints,
            retreat_pos, switch_quat, pregrasp_joints,
            open_steps, include_start=False,
        )
        open_p = np.tile(retreat_pos[None], (open_steps, 1))
        open_q = np.tile(np.asarray(switch_quat, dtype=np.float64)[None], (open_steps, 1))

        # --- Phase 3: planned transit to the standoff -----------------------
        transit = None
        if cfg.planner == "curobo":
            try:
                planner = TransitPlanner(
                    self.asset,
                    dict(zip(self.joint_order, pregrasp_joints)),
                    surface.object_mesh_path,
                    self.device_cfg,
                )
                transit = planner.plan(
                    retreat_pos, switch_quat, standoff_pos, standoff_quat,
                    seconds=cfg.approach_seconds, fps=cfg.fps,
                    max_speed_mps=cfg.max_wrist_speed_mps,
                )
            except Exception as exc:  # planner construction/planning issues
                print(f"[grasp_traj] WARNING: cuRobo transit planning raised {exc!r}; falling back to linear")
                transit = None
            report["planner"] = "curobo" if transit is not None else "linear_fallback"
            if transit is None and cfg.planner == "curobo":
                print("[grasp_traj] WARNING: cuRobo transit planning failed; falling back to straight-line + via-point")
        else:
            report["planner"] = "linear"

        if transit is None:
            transit = self._linear_transit(
                surface, world, retreat_pos, switch_quat, standoff_pos, standoff_quat, pregrasp_joints, report
            )
        transit_p, transit_q = transit
        transit_j = np.tile(pregrasp_joints[None], (transit_p.shape[0], 1))

        pos = np.concatenate([retreat_p, open_p, transit_p], axis=0)
        quat = np.concatenate([retreat_q, open_q, transit_q], axis=0)
        joints = np.concatenate([retreat_j, open_j, transit_j], axis=0)

        transit_clearance = self._min_clearance(world, transit_p, transit_q, transit_j)
        report.update(
            transit_min_clearance_m=transit_clearance,
            clearance_satisfied=transit_clearance >= cfg.approach_clearance_m,
            num_steps=int(pos.shape[0]) - 1,
        )
        if not report["clearance_satisfied"]:
            print(
                "[grasp_traj] WARNING: transit path clearance below threshold "
                f"({transit_clearance:.4f} m); proceeding anyway"
            )
        return pos, quat, joints, report

    def _linear_transit(
        self,
        surface: ObjectSurface,
        world,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        standoff_pos: np.ndarray,
        standoff_quat: np.ndarray,
        pregrasp_joints: np.ndarray,
        report: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: straight line, then one radial via-point if it collides."""

        cfg = self.config

        def build(waypoints_pos, waypoints_quat):
            wp = np.asarray(waypoints_pos, dtype=np.float64)
            wq = np.asarray(waypoints_quat, dtype=np.float64)
            dist = float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum())
            n_steps = steps_for_leg(dist, cfg.approach_seconds, cfg.fps, cfg.max_wrist_speed_mps)
            return polyline_wrist_trajectory(wp, wq, n_steps, include_start=False)

        pos, quat = build([start_pos, standoff_pos], [start_quat, standoff_quat])
        joints = np.tile(pregrasp_joints[None], (pos.shape[0], 1))
        if self._min_clearance(world, pos, quat, joints) >= cfg.approach_clearance_m:
            report["via_point"] = None
            return pos, quat

        points = np.asarray(surface.points_object_frame, dtype=np.float64)
        centroid = points.mean(axis=0)
        bbox_radius = float(np.linalg.norm(points - centroid, axis=1).max())
        midpoint = 0.5 * (np.asarray(start_pos, dtype=np.float64) + np.asarray(standoff_pos, dtype=np.float64))
        radial = midpoint - centroid
        radial_norm = float(np.linalg.norm(radial))
        direction = radial / radial_norm if radial_norm > 1e-9 else np.asarray([0.0, 0.0, 1.0])
        via_pos = centroid + direction * max(bbox_radius + cfg.standoff_m, radial_norm)
        via_quat = slerp_wxyz(start_quat, standoff_quat, 0.5)
        report["via_point"] = via_pos.tolist()
        return build([start_pos, via_pos, standoff_pos], [start_quat, via_quat, standoff_quat])

    def generate(self, sequence_dir: str | Path, grasp_record: dict) -> GraspTrajectory:
        sequence_dir = Path(sequence_dir)
        surface = ObjectSurface.from_sequence_dir(sequence_dir)
        demo = HumanDemo.from_sequence_dir(sequence_dir)
        affordance = ensure_affordance(sequence_dir, demo, surface.points_object_frame)
        analysis = analyze_demo(demo, affordance)

        record_joint_order = list(grasp_record["joint_names"])
        action = np.asarray(grasp_record["action"], dtype=np.float64)
        grasp_pos = action[:3]
        grasp_quat = action[3:7]
        grasp_quat = grasp_quat / np.linalg.norm(grasp_quat)
        grasp_joints_full = _remap_by_name(action[7:], record_joint_order, self.joint_order)
        grasp_joints_full = self._clamp(grasp_joints_full)

        # Retarget every valid demo frame up to and including the grasp frame:
        # this single batched IK call covers both the switch-frame search's
        # candidate pool and the retarget-replay segment.
        candidate_frames = np.arange(0, analysis.grasp_frame_index + 1)
        candidate_frames = candidate_frames[demo.valid_mask[candidate_frames]]
        if candidate_frames.size == 0:
            raise ValueError(f"{sequence_dir}: no valid demo frames at/before the grasp frame")
        retarget_all = self.fitter.fit_frames(
            analysis.wrist_poses_object[candidate_frames],
            demo.hand_joints_object[candidate_frames],
            candidate_frames,
        )
        frame_to_row = {int(f): i for i, f in enumerate(retarget_all.frame_indices)}

        world = build_contact_world(surface.object_mesh_path, self.asset.urdf_path, self.device_cfg)
        switch_frame, clearance_report = select_switch_frame(
            demo,
            analysis,
            retarget_all,
            self.clearance_checker,
            world,
            affordance,
            approach_seconds=self.config.approach_seconds,
            fps=self.config.fps,
            clearance_m=self.config.approach_clearance_m,
        )

        hand_pos_chunks: list[np.ndarray] = []
        hand_quat_chunks: list[np.ndarray] = []
        joints_chunks: list[np.ndarray] = []
        object_pos_chunks: list[np.ndarray] = []
        object_quat_chunks: list[np.ndarray] = []
        segment_chunks: list[np.ndarray] = []

        def _append(pos, quat, joints, obj_pos, obj_quat, label, n):
            hand_pos_chunks.append(pos)
            hand_quat_chunks.append(quat)
            joints_chunks.append(joints)
            object_pos_chunks.append(obj_pos)
            object_quat_chunks.append(obj_quat)
            segment_chunks.append(np.full((n,), label, dtype=np.int8))

        # --- Segment a: retarget replay, camera frame -----------------------
        retarget_frames = candidate_frames[candidate_frames < switch_frame]
        if retarget_frames.size:
            rows = np.asarray([frame_to_row[int(f)] for f in retarget_frames])
            hand_pos_obj = retarget_all.ref_actions[rows, :3]
            hand_quat_obj = retarget_all.ref_actions[rows, 3:7]
            hand_joints = self._clamp(retarget_all.ref_actions[rows, 7:])
            obj_pose_cam = demo.object_pose_camera[retarget_frames]
            obj_pos_cam, obj_quat_cam = matrix_to_pos_quat(obj_pose_cam)
            hand_pos_cam, hand_quat_cam = compose_pos_quat(obj_pos_cam, obj_quat_cam, hand_pos_obj, hand_quat_obj)
            _append(hand_pos_cam, hand_quat_cam, hand_joints, obj_pos_cam, obj_quat_cam, SEGMENT_RETARGET, retarget_frames.size)

        # --- Segment b: synthetic approach/close/squeeze, static object frame ----
        switch_row = frame_to_row[int(switch_frame)]
        switch_pos = retarget_all.ref_actions[switch_row, :3]
        switch_quat = retarget_all.ref_actions[switch_row, 3:7]
        switch_joints = self._clamp(retarget_all.ref_actions[switch_row, 7:])

        # Pregrasp = the grasp posture with all flexion channels scaled toward
        # 0 rad (straight/open) by pregrasp_open_fraction -- the hand must be
        # WIDE open on the way in, or closing merely pushes the object away
        # instead of wrapping around it. Non-flexion channels (AA spread,
        # thumb rotation) keep their grasp values so the hand stays oriented
        # to wrap.
        open_scale = 1.0 - float(np.clip(self.config.pregrasp_open_fraction, 0.0, 1.0))
        pregrasp_joints = self._clamp(
            np.where(self.relax_mask > 0, grasp_joints_full * open_scale, grasp_joints_full)
        )

        # Failed-grasp records can place even the PALM inside the object; no
        # finger projection can repair that. Pull the grasp wrist pose back
        # along its own approach axis until the wide-open hand clears the
        # surface, and use the corrected pose everywhere (standoff, close,
        # squeeze, and the carry's grasp_root_tf).
        grasp_pos, wrist_backoff = self._project_wrist_pose(
            world, grasp_pos, grasp_quat, pregrasp_joints,
            clearance_target_m=self.config.near_contact_margin_m,
        )
        squeeze_joints = self._clamp(grasp_joints_full + self.config.squeeze_delta * self.relax_mask)
        # The standoff is the grasp pose pulled back along ITS palm approach
        # axis, so the final reach comes straight in along the synthesized
        # approach direction instead of sweeping in from wherever the switch
        # pose happens to be.
        standoff_pos = standoff_pose(grasp_pos, grasp_quat, self.config.standoff_m)
        standoff_quat = grasp_quat

        static_object_pose = demo.object_pose_camera[switch_frame]

        approach_pos, approach_quat, approach_joints, transit_report = self._plan_transit(
            surface, world,
            switch_pos, switch_quat, switch_joints,
            standoff_pos, standoff_quat, pregrasp_joints,
        )
        clearance_report = {**clearance_report, "transit": transit_report}

        # Reach: fly the wide-open hand from the standoff to the grasp wrist
        # pose (fingers pinned open; contact with the object is not expected
        # until the fingers close, so this leg is not SDF-checked). Labeled
        # SEGMENT_APPROACH -- it is the tail of the approach.
        reach_steps = steps_for_leg(
            float(np.linalg.norm(grasp_pos - standoff_pos)),
            self.config.close_seconds, self.config.fps, self.config.max_wrist_speed_mps,
        )
        reach_pos, reach_quat, reach_joints = interp_trajectory(
            standoff_pos, standoff_quat, pregrasp_joints,
            grasp_pos, grasp_quat, pregrasp_joints,
            reach_steps, include_start=False,
        )
        reach_clearance = self._min_clearance(world, reach_pos, reach_quat, reach_joints)
        clearance_report = {**clearance_report, "reach_min_clearance_m": reach_clearance}
        if reach_clearance < 0.0:
            print(
                "[grasp_traj] WARNING: the open hand brushes the object during the reach leg "
                f"(min clearance {reach_clearance:.4f} m)"
            )
        approach_pos = np.concatenate([approach_pos, reach_pos], axis=0)
        approach_quat = np.concatenate([approach_quat, reach_quat], axis=0)
        approach_joints = np.concatenate([approach_joints, reach_joints], axis=0)

        # Close: wrist holds the grasp pose while the fingers close in two
        # stages onto a CONTACT-PROJECTED target -- the synthesized grasp
        # joints themselves often penetrate the object (all current failed_
        # grasp records do, by 5-15mm), and commanding positions inside the
        # object plows the fingers through it. Stage 1 sweeps quickly to a
        # near-contact posture (restoring near-simultaneous finger arrival),
        # stage 2 creeps the last few mm to the zero-clearance posture. The
        # original grasp joints survive only inside the squeeze target, i.e.
        # as a bounded drive-force request against real contact.
        grasp_pose_clearance = self._min_clearance(
            world, np.asarray(grasp_pos)[None], np.asarray(grasp_quat)[None], grasp_joints_full[None]
        )
        contact_joints, contact_t = self._project_contact_joints(
            world, grasp_pos, grasp_quat, pregrasp_joints, grasp_joints_full, 0.0
        )
        near_contact_joints, near_contact_t = self._project_contact_joints(
            world, grasp_pos, grasp_quat, pregrasp_joints, grasp_joints_full,
            self.config.near_contact_margin_m,
        )
        clearance_report = {
            **clearance_report,
            "grasp_pose_clearance_m": grasp_pose_clearance,
            "wrist_backoff_m": wrist_backoff,
            "contact_close_fraction": contact_t,
            "near_contact_close_fraction": near_contact_t,
        }
        close1_pos, close1_quat, close1_joints = interp_trajectory(
            grasp_pos, grasp_quat, pregrasp_joints,
            grasp_pos, grasp_quat, near_contact_joints,
            self.config.close_steps(), include_start=False,
        )
        final_close_steps = max(1, int(round(self.config.final_close_seconds * self.config.fps)))
        close2_pos, close2_quat, close2_joints = interp_trajectory(
            grasp_pos, grasp_quat, near_contact_joints,
            grasp_pos, grasp_quat, contact_joints,
            final_close_steps, include_start=False,
        )
        close_pos = np.concatenate([close1_pos, close2_pos], axis=0)
        close_quat = np.concatenate([close1_quat, close2_quat], axis=0)
        close_joints = np.concatenate([close1_joints, close2_joints], axis=0)
        squeeze_pos, squeeze_quat, squeeze_joint_traj = interp_trajectory(
            grasp_pos, grasp_quat, contact_joints,
            grasp_pos, grasp_quat, squeeze_joints,
            self.config.squeeze_steps(), include_start=False,
        )

        for pos_b, quat_b, joints_b, label in (
            (approach_pos, approach_quat, approach_joints, SEGMENT_APPROACH),
            (close_pos, close_quat, close_joints, SEGMENT_CLOSE),
            (squeeze_pos, squeeze_quat, squeeze_joint_traj, SEGMENT_SQUEEZE),
        ):
            static_obj_pose_batched = np.tile(static_object_pose[None], (pos_b.shape[0], 1, 1))
            static_obj_pos_b, static_obj_quat_b = matrix_to_pos_quat(static_obj_pose_batched)
            hand_pos_cam, hand_quat_cam = compose_pos_quat(static_obj_pos_b, static_obj_quat_b, pos_b, quat_b)
            _append(hand_pos_cam, hand_quat_cam, joints_b, static_obj_pos_b, static_obj_quat_b, label, pos_b.shape[0])

        grasp_root_tf = pos_quat_to_matrix(grasp_pos, grasp_quat)

        # --- Segment c: carry, following the recorded object trajectory ----
        carry_frame = (
            analysis.grasp_frame_index
            if self.config.carry_start == CARRY_START_GRASP_FRAME
            else analysis.pickup_frame_index
        )
        carry_indices = demo.valid_indices[demo.valid_indices >= carry_frame]
        if carry_indices.size:
            obj_pose_cam = demo.object_pose_camera[carry_indices]
            obj_pos_cam, obj_quat_cam = matrix_to_pos_quat(obj_pose_cam)
            hand_pose_cam = obj_pose_cam @ grasp_root_tf[None]
            hand_pos_cam, hand_quat_cam = matrix_to_pos_quat(hand_pose_cam)
            # Ease out of the squeeze-end pose: segment b froze the object at
            # its switch-frame pose, while carry follows the live recording,
            # so without a blend the boundary has a visible pose jump.
            n_blend = max(1, int(round(self.config.carry_blend_seconds * self.config.fps)))
            hand_pos_cam, hand_quat_cam = blend_into_trajectory(
                hand_pos_chunks[-1][-1], hand_quat_chunks[-1][-1], hand_pos_cam, hand_quat_cam, n_blend
            )
            joints_carry = np.tile(squeeze_joints[None], (carry_indices.size, 1))
            _append(hand_pos_cam, hand_quat_cam, joints_carry, obj_pos_cam, obj_quat_cam, SEGMENT_CARRY, carry_indices.size)

        traj = GraspTrajectory(
            hand_pos_camera=np.concatenate(hand_pos_chunks, axis=0),
            hand_quat_camera=np.concatenate(hand_quat_chunks, axis=0),
            finger_targets=np.concatenate(joints_chunks, axis=0),
            object_pos_camera=np.concatenate(object_pos_chunks, axis=0),
            object_quat_camera=np.concatenate(object_quat_chunks, axis=0),
            segment=np.concatenate(segment_chunks, axis=0),
            dt=self.config.dt,
            joint_order=tuple(self.joint_order),
            grasp_json=str(grasp_record.get("_grasp_json_path", "")),
            sequence_dir=str(sequence_dir),
            switch_frame_index=int(switch_frame),
            grasp_root_tf=grasp_root_tf.tolist(),
            clearance_report=clearance_report,
            extra_metadata={
                "config": {
                    "fps": self.config.fps,
                    "approach_seconds": self.config.approach_seconds,
                    "close_seconds": self.config.close_seconds,
                    "squeeze_seconds": self.config.squeeze_seconds,
                    "standoff_m": self.config.standoff_m,
                    "pregrasp_open_fraction": self.config.pregrasp_open_fraction,
                    "squeeze_delta": self.config.squeeze_delta,
                    "approach_clearance_m": self.config.approach_clearance_m,
                    "carry_start": self.config.carry_start,
                    "max_wrist_speed_mps": self.config.max_wrist_speed_mps,
                    "carry_blend_seconds": self.config.carry_blend_seconds,
                    "open_clearance_m": self.config.open_clearance_m,
                    "retreat_max_m": self.config.retreat_max_m,
                    "open_seconds": self.config.open_seconds,
                    "planner": self.config.planner,
                    "final_close_seconds": self.config.final_close_seconds,
                    "near_contact_margin_m": self.config.near_contact_margin_m,
                },
                "grasp_frame_index": int(analysis.grasp_frame_index),
                "pickup_frame_index": int(analysis.pickup_frame_index),
                "carry_start_frame": int(carry_frame),
                "object_name": grasp_record.get("object_name"),
                "object_mesh": str(surface.object_mesh_path),
                "num_retarget_frames": int(retarget_frames.size) if retarget_frames.size else 0,
                "num_carry_frames": int(carry_indices.size) if carry_indices.size else 0,
            },
        )
        return traj


def resolve_grasp_record(synthesis_out_dir: str | Path | None, grasp_json: str | Path | None) -> Path:
    """Resolve a grasp record path, either given directly or via a synthesis
    output dir's summary.json (grasp_json on success, else failed_grasp_json)
    -- the same fallback synthesize_sharpa_anchored_bodex.py's own consumers use."""

    if grasp_json is not None:
        path = Path(grasp_json)
        if not path.exists():
            raise FileNotFoundError(f"--grasp-json not found: {path}")
        return path
    if synthesis_out_dir is None:
        raise ValueError("either --grasp-json or --synthesis-out-dir is required")
    summary_path = Path(synthesis_out_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no summary.json under --synthesis-out-dir: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record_path = summary.get("grasp_json") or summary.get("failed_grasp_json")
    if not record_path:
        raise ValueError(f"{summary_path} has neither 'grasp_json' nor 'failed_grasp_json'")
    return Path(record_path)


def load_grasp_record(grasp_json: str | Path) -> dict:
    grasp_json = Path(grasp_json)
    record: dict[str, Any] = json.loads(grasp_json.read_text(encoding="utf-8"))
    record["_grasp_json_path"] = str(grasp_json)
    return record
