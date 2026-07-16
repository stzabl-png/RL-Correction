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
from ocir.grasp_traj.self_clearance import DEFAULT_SELF_CLEARANCE_BUFFER_M, SelfClearanceTuner
from ocir.grasp_traj.segments import (
    PALM_APPROACH_AXIS_LOCAL,
    blend_into_trajectory,
    interp_trajectory,
    polyline_wrist_trajectory,
    quat_to_matrix,
    slerp_wxyz,
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

CARRY_STYLE_VERTICAL_LIFT = "vertical_lift"
CARRY_STYLE_DEMO = "demo"


@dataclass
class GraspTrajectoryConfig:
    fps: float = 30.0
    approach_seconds: float = 1.0
    close_seconds: float = 0.3
    #: Duration of the close->squeeze segment (the final wrap onto the object).
    #: The segment is a static hold: the fingers are already at the grasp pose,
    #: so it lets the contacts converge at the grasp posture -- there is no
    #: separate driven-past-contact "squeeze" pose (removed 2026-07-16).
    squeeze_seconds: float = 0.3
    pregrasp_open_fraction: float = 1.0
    approach_clearance_m: float = 0.003
    #: ``vertical_lift``: after the squeeze segment (and settle hold) the wrist
    #: rises straight up (world +z) by ``carry_lift_height_m`` and holds -- the
    #: simplified pick task. ``demo``: follow the recorded human carry
    #: trajectory (the original behavior).
    carry_style: str = CARRY_STYLE_VERTICAL_LIFT
    #: Post-squeeze hold (wrist parked, grasp targets held) letting the
    #: physics contacts settle before any load transfer; appended to the
    #: squeeze segment.
    settle_seconds: float = 1.0
    carry_lift_height_m: float = 0.20
    carry_lift_seconds: float = 2.0
    #: Hold at the top of the vertical lift (sustained-lift evidence).
    carry_hold_seconds: float = 1.0
    #: DexYCB manifest for the camera->world extrinsics that define "up" in
    #: the camera frame (vertical_lift only). None: auto-resolve next to the
    #: sequences root, falling back to the Stage B default manifest.
    dexycb_manifest: str | None = None
    carry_start: str = CARRY_START_GRASP_FRAME
    max_wrist_speed_mps: float = 0.25
    carry_blend_seconds: float = 0.3
    open_clearance_m: float = 0.05
    open_horizon_seconds: float = 1.0
    planner: str = "curobo"  # or "linear"
    final_close_seconds: float = 1.0
    near_contact_margin_m: float = 0.003
    #: Minimum sphere-metric self-clearance enforced on the trajectory's
    #: INITIAL posture (frame 0, the pose the simulation teleport-initializes
    #: the articulation to -- an interpenetrating start blows up under PhysX
    #: hand self-collision). 0.0 = sphere surfaces touching (~1.5-1.8 mm true
    #: mesh gap at the binding finger-base pairs); positive values are
    #: anatomically unreachable at open postures. Negative disables the pass.
    self_clearance_buffer_m: float = DEFAULT_SELF_CLEARANCE_BUFFER_M
    #: The frame-0 correction decays linearly back to the original trajectory
    #: over this duration (no target jump at frame 1; past it the trajectory
    #: is untouched).
    self_clearance_decay_seconds: float = 1.0

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


def world_up_in_camera(sequence_dir: Path, manifest_path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """World +z (up, opposite gravity) expressed in the sequence's camera
    frame, unit norm.

    Uses the same DexYCB apriltag-extrinsics frame mapper Stage B maps the
    trajectory with (``ocir.isaac.replay_dexycb`` is import-safe without an
    Isaac runtime), so "vertical" authored here is exactly world-vertical in
    the simulation. Fails closed when the manifest or sequence entry cannot
    be found -- the cameras are tilted down at the table, so no camera-axis
    fallback is acceptably close.
    """

    from ocir.isaac.replay_dexycb import DEFAULT_MANIFEST, load_dexycb_frame_mapper

    sequence_dir = Path(sequence_dir)
    seq_meta = json.loads((sequence_dir / "sequence.json").read_text(encoding="utf-8"))
    sequence_id = str(seq_meta.get("sequence_id") or sequence_dir.name)
    candidates = (
        [Path(manifest_path)]
        if manifest_path is not None
        else [
            sequence_dir.resolve().parents[1] / "manifests" / "selected_5_sequences.json",
            Path(DEFAULT_MANIFEST),
        ]
    )
    manifest_file = next((c for c in candidates if c.is_file()), None)
    if manifest_file is None:
        raise FileNotFoundError(
            f"no DexYCB manifest found for the camera->world extrinsics (tried {[str(c) for c in candidates]}); "
            "pass --dexycb-manifest, or use --carry-style demo which needs no world frame"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    entry = next((s for s in manifest.get("sequences", []) if s.get("sequence_id") == sequence_id), None)
    if entry is None:
        raise KeyError(f"sequence_id {sequence_id!r} not found in manifest {manifest_file}")
    mapper = load_dexycb_frame_mapper(manifest, entry)
    up = np.asarray(mapper.rotation, dtype=np.float64).T @ np.array([0.0, 0.0, 1.0])
    up = up / np.linalg.norm(up)
    return up, {
        "manifest": str(manifest_file),
        "camera": mapper.camera,
        "extrinsics": mapper.extrinsics,
        "world_up_camera": up.tolist(),
    }


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
        pregrasp_pos: np.ndarray,
        pregrasp_quat: np.ndarray,
        pregrasp_joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Handoff pose (hand already fully open -- the opening ramp lives in
        the retarget segment, and the switch frame was selected for open-hand
        clearance) -> repaired pregrasp pose, via cuRobo v2 MotionPlanner
        (floating-base hand, fingers locked wide open, object mesh as
        obstacle). ``planner=linear`` is an explicit debugging alternative;
        cuRobo failures are fail-closed rather than silently replacing the
        collision-constrained plan."""

        cfg = self.config
        report: dict = {}
        switch_pos = np.asarray(switch_pos, dtype=np.float64)

        if cfg.planner == "curobo":
            try:
                planner = TransitPlanner(
                    self.asset,
                    dict(zip(self.joint_order, pregrasp_joints)),
                    surface.object_mesh_path,
                    self.device_cfg,
                )
                transit = planner.plan(
                    switch_pos, switch_quat, pregrasp_pos, pregrasp_quat,
                    seconds=cfg.approach_seconds, fps=cfg.fps,
                    max_speed_mps=cfg.max_wrist_speed_mps,
                )
            except Exception as exc:  # planner construction/planning issues
                raise RuntimeError(f"cuRobo direct-to-pregrasp planning raised {exc!r}") from exc
            if transit is None:
                raise RuntimeError("cuRobo could not find a collision-constrained direct-to-pregrasp plan")
            report["planner"] = "curobo"
        else:
            report["planner"] = "linear"
            transit = self._linear_transit(
                surface, world, switch_pos, switch_quat, pregrasp_pos, pregrasp_quat, pregrasp_joints, report
            )
        pos, quat = transit
        # plan()/_linear_transit are start-exclusive; emit the switch pose
        # itself (fully open) as the segment's first step so the retarget
        # replay hands off without a gap.
        pos = np.concatenate([switch_pos[None], pos], axis=0)
        quat = np.concatenate([np.asarray(switch_quat, dtype=np.float64)[None], quat], axis=0)
        joints = np.tile(pregrasp_joints[None], (pos.shape[0], 1))

        transit_clearance = self._min_clearance(world, pos, quat, joints)
        # The transit legitimately ENDS next to the object (its goal is the
        # synthesized pregrasp pose), so the threshold check excludes the
        # arrival tail; the full-path minimum is still reported.
        arrival_tail = min(max(3, pos.shape[0] // 10), max(pos.shape[0] - 1, 1))
        enroute_clearance = self._min_clearance(
            world, pos[:-arrival_tail], quat[:-arrival_tail], joints[:-arrival_tail]
        ) if pos.shape[0] > arrival_tail else transit_clearance
        report.update(
            transit_min_clearance_m=transit_clearance,
            enroute_min_clearance_m=enroute_clearance,
            clearance_satisfied=enroute_clearance >= cfg.approach_clearance_m,
            num_steps=int(pos.shape[0]),
        )
        if not report["clearance_satisfied"]:
            print(
                "[grasp_traj] WARNING: transit path clearance below threshold "
                f"({enroute_clearance:.4f} m en route); proceeding anyway"
            )
        return pos, quat, joints, report

    def _linear_transit(
        self,
        surface: ObjectSurface,
        world,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        pregrasp_pos: np.ndarray,
        pregrasp_quat: np.ndarray,
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

        pos, quat = build([start_pos, pregrasp_pos], [start_quat, pregrasp_quat])
        joints = np.tile(pregrasp_joints[None], (pos.shape[0], 1))
        if self._min_clearance(world, pos, quat, joints) >= cfg.approach_clearance_m:
            report["via_point"] = None
            return pos, quat

        points = np.asarray(surface.points_object_frame, dtype=np.float64)
        centroid = points.mean(axis=0)
        bbox_radius = float(np.linalg.norm(points - centroid, axis=1).max())
        midpoint = 0.5 * (np.asarray(start_pos, dtype=np.float64) + np.asarray(pregrasp_pos, dtype=np.float64))
        radial = midpoint - centroid
        radial_norm = float(np.linalg.norm(radial))
        direction = radial / radial_norm if radial_norm > 1e-9 else np.asarray([0.0, 0.0, 1.0])
        via_pos = centroid + direction * max(bbox_radius + cfg.open_clearance_m, radial_norm)
        via_quat = slerp_wxyz(start_quat, pregrasp_quat, 0.5)
        report["via_point"] = via_pos.tolist()
        return build([start_pos, via_pos, pregrasp_pos], [start_quat, via_quat, pregrasp_quat])

    def _plan_contact_leg(
        self,
        surface: ObjectSurface,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        locked_joints: np.ndarray,
        *,
        seconds: float,
    ) -> tuple[tuple[np.ndarray, np.ndarray], str]:
        """Short pregrasp -> grasp wrist leg: cuRobo when it can (fingers
        locked at the pregrasp posture), straight interpolation as the
        DESIGNED fallback -- the goal sits essentially on the contact
        boundary, where a collision-constrained plan is expected to be
        infeasible for some grasps. Returns ``((pos, quat), planner_name)``,
        start-exclusive."""

        cfg = self.config
        dist = float(np.linalg.norm(np.asarray(goal_pos) - np.asarray(start_pos)))

        def _interp(name: str) -> tuple[tuple[np.ndarray, np.ndarray], str]:
            n_steps = steps_for_leg(dist, seconds, cfg.fps, cfg.max_wrist_speed_mps)
            pos, quat = polyline_wrist_trajectory(
                np.stack([start_pos, goal_pos]), np.stack([start_quat, goal_quat]), n_steps, include_start=False
            )
            return (pos, quat), name

        # With the in-optimization penetration penalty, the record's grasp
        # stage frequently coincides with (or sits within a couple of mm of)
        # the pregrasp snapshot -- building a full cuRobo planner for a
        # sub-2mm move is pure overhead.
        if dist < 0.002:
            return _interp("interp_short")
        if cfg.planner == "curobo":
            try:
                planner = TransitPlanner(
                    self.asset,
                    dict(zip(self.joint_order, locked_joints)),
                    surface.object_mesh_path,
                    self.device_cfg,
                )
                out = planner.plan(
                    start_pos, start_quat, goal_pos, goal_quat,
                    seconds=seconds, fps=cfg.fps, max_speed_mps=cfg.max_wrist_speed_mps,
                )
                if out is not None:
                    # cuRobo can return a valid-but-pathological detour for
                    # this near-in-place leg (the pregrasp and grasp wrists
                    # coincide -- pregrasp only opens the fingers -- so the
                    # goal sits ~a cm away): a mug seed planned an 80cm wrist
                    # loop for a 1.3cm move, which reads as the hand flying
                    # away and snapping back. The straight interp with fingers
                    # locked open is collision-free here by construction, so
                    # reject any plan whose wrist path is far longer than the
                    # direct move and fall back to it.
                    plan_pos = np.asarray(out[0])
                    path_len = float(
                        np.linalg.norm(plan_pos[0] - start_pos)
                        + np.linalg.norm(np.diff(plan_pos, axis=0), axis=1).sum()
                    )
                    detour_limit = max(2.5 * dist, dist + 0.04)
                    if path_len <= detour_limit:
                        return out, "curobo"
                    print(
                        f"[grasp_traj] pregrasp->grasp leg: cuRobo path {path_len*100:.1f}cm detours far past "
                        f"the {dist*100:.1f}cm direct move (limit {detour_limit*100:.1f}cm); interpolating"
                    )
                else:
                    print("[grasp_traj] pregrasp->grasp leg: cuRobo found no plan (goal at contact boundary); interpolating")
            except Exception as exc:
                print(f"[grasp_traj] pregrasp->grasp leg: cuRobo raised {exc!r}; interpolating")
        return _interp("interp")

    def _resample_carry_object_poses(
        self,
        object_pos_camera: np.ndarray,
        object_quat_camera: np.ndarray,
        grasp_root_tf: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Insert samples into the recorded object trajectory until the
        composed hand-root translation respects ``max_wrist_speed_mps``.

        Demo frames are retained exactly; only intermediate object poses are
        added (linear translation + quaternion slerp).  This prevents a fast
        recorded carry from yanking a marginal friction grasp loose while
        preserving the demonstrated geometric path.
        """

        obj_pos = np.asarray(object_pos_camera, dtype=np.float64)
        obj_quat = np.asarray(object_quat_camera, dtype=np.float64)
        if obj_pos.shape[0] <= 1 or self.config.max_wrist_speed_mps <= 0.0:
            return obj_pos, obj_quat, {"raw_steps": int(obj_pos.shape[0]), "resampled_steps": int(obj_pos.shape[0])}

        hand_pos_raw, _ = matrix_to_pos_quat(
            pos_quat_to_matrix(obj_pos, obj_quat) @ np.asarray(grasp_root_tf, dtype=np.float64)[None]
        )
        out_pos = [obj_pos[0]]
        out_quat = [obj_quat[0]]
        subdivisions: list[int] = []
        for i in range(obj_pos.shape[0] - 1):
            hand_distance = float(np.linalg.norm(hand_pos_raw[i + 1] - hand_pos_raw[i]))
            n_steps = max(
                1,
                int(np.ceil(hand_distance * self.config.fps / self.config.max_wrist_speed_mps)),
            )
            subdivisions.append(n_steps)
            for step in range(1, n_steps + 1):
                alpha = step / float(n_steps)
                out_pos.append(obj_pos[i] * (1.0 - alpha) + obj_pos[i + 1] * alpha)
                out_quat.append(slerp_wxyz(obj_quat[i], obj_quat[i + 1], alpha))

        out_pos_np = np.asarray(out_pos, dtype=np.float64)
        out_quat_np = np.asarray(out_quat, dtype=np.float64)
        hand_pos_out, _ = matrix_to_pos_quat(
            pos_quat_to_matrix(out_pos_np, out_quat_np) @ np.asarray(grasp_root_tf, dtype=np.float64)[None]
        )
        step_dist = np.linalg.norm(np.diff(hand_pos_out, axis=0), axis=1)
        return out_pos_np, out_quat_np, {
            "raw_steps": int(obj_pos.shape[0]),
            "resampled_steps": int(out_pos_np.shape[0]),
            "max_subdivisions_per_demo_step": int(max(subdivisions, default=1)),
            "max_hand_step_m": float(step_dist.max()) if step_dist.size else 0.0,
            "max_hand_speed_mps": float(step_dist.max() * self.config.fps) if step_dist.size else 0.0,
        }

    def _cap_synchronized_pose_speed(
        self,
        hand_pos: np.ndarray,
        hand_quat: np.ndarray,
        object_pos: np.ndarray,
        object_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Densify synchronized hand/object poses to enforce the translational
        wrist speed cap, including the squeeze->carry boundary blend."""

        if hand_pos.shape[0] <= 1 or self.config.max_wrist_speed_mps <= 0.0:
            return hand_pos, hand_quat, object_pos, object_quat
        hp = [np.asarray(hand_pos[0], dtype=np.float64)]
        hq = [np.asarray(hand_quat[0], dtype=np.float64)]
        op = [np.asarray(object_pos[0], dtype=np.float64)]
        oq = [np.asarray(object_quat[0], dtype=np.float64)]
        for i in range(hand_pos.shape[0] - 1):
            distance = float(np.linalg.norm(hand_pos[i + 1] - hand_pos[i]))
            n_steps = max(1, int(np.ceil(distance * self.config.fps / self.config.max_wrist_speed_mps)))
            for step in range(1, n_steps + 1):
                alpha = step / float(n_steps)
                hp.append(hand_pos[i] * (1.0 - alpha) + hand_pos[i + 1] * alpha)
                hq.append(slerp_wxyz(hand_quat[i], hand_quat[i + 1], alpha))
                op.append(object_pos[i] * (1.0 - alpha) + object_pos[i + 1] * alpha)
                oq.append(slerp_wxyz(object_quat[i], object_quat[i + 1], alpha))
        return np.asarray(hp), np.asarray(hq), np.asarray(op), np.asarray(oq)

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

        # Stage records (anchored_bodex/grasp_stages.py) carry pregrasp
        # (opened clear of the object) and grasp (the fully optimized action,
        # possibly slightly penetrating) poses computed at synthesis time;
        # when present they replace this generator's own wrist/finger contact
        # projection entirely.
        record_stages = grasp_record.get("stages") or None

        def _stage_action(name: str) -> np.ndarray:
            entry = record_stages[name]
            joints = np.asarray([entry["joints"][j] for j in self.joint_order], dtype=np.float64)
            quat = np.asarray(entry["orientation"], dtype=np.float64)
            quat = quat / np.linalg.norm(quat)
            return np.concatenate([np.asarray(entry["position"], dtype=np.float64), quat, self._clamp(joints)])

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

        # Pregrasp = the grasp posture with all flexion channels scaled toward
        # 0 rad (straight/open) by pregrasp_open_fraction -- the hand must be
        # WIDE open on the way in, or closing merely pushes the object away
        # instead of wrapping around it. Non-flexion channels (AA spread,
        # thumb rotation) keep their grasp values so the hand stays oriented
        # to wrap. Computed before switch-frame selection: the switch frame
        # must be one where the FULLY OPEN hand already clears the object by
        # open_clearance_m, so the smooth opening ramp (below) finishes well
        # away from the object.
        open_scale = 1.0 - float(np.clip(self.config.pregrasp_open_fraction, 0.0, 1.0))
        pregrasp_joints = self._clamp(
            np.where(self.relax_mask > 0, grasp_joints_full * open_scale, grasp_joints_full)
        )

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
            clearance_m=self.config.open_clearance_m,
            override_joints=pregrasp_joints,
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
        # Fingers open SMOOTHLY over the last open_horizon_seconds of the
        # replay: each frame in the window blends the retargeted joints toward
        # the wide-open pregrasp, reaching fully open exactly at the switch
        # frame -- no separate in-place opening action right next to the
        # object (the opening fingers themselves used to nudge it).
        retarget_frames = candidate_frames[candidate_frames < switch_frame]
        if retarget_frames.size:
            rows = np.asarray([frame_to_row[int(f)] for f in retarget_frames])
            hand_pos_obj = retarget_all.ref_actions[rows, :3]
            hand_quat_obj = retarget_all.ref_actions[rows, 3:7]
            hand_joints = self._clamp(retarget_all.ref_actions[rows, 7:])
            open_horizon = max(1, int(round(self.config.open_horizon_seconds * self.config.fps)))
            m = retarget_frames.size
            # steps_before_switch for row j is m - j (the switch pose itself
            # comes one step after the last retarget row, at blend weight 1).
            steps_before_switch = m - np.arange(m)
            blend = np.clip(1.0 - steps_before_switch / float(open_horizon), 0.0, 1.0)
            hand_joints = hand_joints * (1.0 - blend[:, None]) + pregrasp_joints[None] * blend[:, None]
            obj_pose_cam = demo.object_pose_camera[retarget_frames]
            obj_pos_cam, obj_quat_cam = matrix_to_pos_quat(obj_pose_cam)
            hand_pos_cam, hand_quat_cam = compose_pos_quat(obj_pos_cam, obj_quat_cam, hand_pos_obj, hand_quat_obj)
            _append(hand_pos_cam, hand_quat_cam, hand_joints, obj_pos_cam, obj_quat_cam, SEGMENT_RETARGET, retarget_frames.size)

        # --- Segment b: synthetic approach/close/squeeze, static object frame ----
        switch_row = frame_to_row[int(switch_frame)]
        switch_pos = retarget_all.ref_actions[switch_row, :3]
        switch_quat = retarget_all.ref_actions[switch_row, 3:7]

        static_object_pose = demo.object_pose_camera[switch_frame]
        final_close_steps = max(1, int(round(self.config.final_close_seconds * self.config.fps)))

        if record_stages is not None:
            # --- Stage-driven segment b (retarget handoff -> planned transit
            # to the synthesized PREGRASP pose -> close through to the GRASP
            # pose), both poses from the record. ---
            stage_pregrasp = _stage_action("pregrasp")
            stage_grasp = _stage_action("grasp")
            pre_pos, pre_quat, pre_joints = stage_pregrasp[:3], stage_pregrasp[3:7], stage_pregrasp[7:]
            grasp_pos, grasp_quat = stage_grasp[:3], stage_grasp[3:7]
            grasp_stage_joints = stage_grasp[7:]
            # The pose held through the squeeze segment, settle, and carry is
            # the record's GRASP pose -- no drive-through past contact; the
            # grasp joints are enforced for the whole carry.
            squeeze_joints = grasp_stage_joints

            # The wide-open hand cannot necessarily BE at the pregrasp wrist
            # (that pose is only clear with its own near-closed joints), so
            # the plan goal is an adaptive PREAPPROACH point: the pregrasp
            # wrist backed off along its palm axis just far enough for the
            # open hand to clear -- the Articulation_Bodex step-back, but
            # computed from the SDF instead of a fixed distance.
            preapproach_pos, preapproach_backoff = self._project_wrist_pose(
                world, pre_pos, pre_quat, pregrasp_joints,
                clearance_target_m=self.config.approach_clearance_m,
                max_backoff_m=0.12,
            )

            # cuRobo transit: switch pose -> preapproach, fingers locked wide
            # open (fail-closed, same as the non-stage path).
            approach_pos, approach_quat, approach_joints, transit_report = self._plan_transit(
                surface, world,
                switch_pos, switch_quat,
                preapproach_pos, pre_quat, pregrasp_joints,
            )
            clearance_report = {
                **clearance_report,
                "transit": transit_report,
                "preapproach_backoff_m": preapproach_backoff,
            }

            # Close, under the simulation's softened finger gains throughout:
            # 1. step-in: preapproach -> pregrasp wrist while the fingers
            #    blend wide-open -> synthesized pregrasp posture
            #    (Articulation_Bodex's step-in);
            # 2. wrist pregrasp -> grasp pose (cuRobo with the fingers locked
            #    at the pregrasp posture when it finds a plan; the grasp pose
            #    is the raw optimized action and may slightly penetrate, so
            #    planning to it is EXPECTED to fail often -- the straight
            #    interpolation is the designed fallback, not an error);
            # 3. fingers pregrasp -> grasp posture, in place.
            close1_steps = steps_for_leg(
                float(np.linalg.norm(pre_pos - preapproach_pos)),
                self.config.close_seconds, self.config.fps, self.config.max_wrist_speed_mps,
            )
            close1_pos, close1_quat, close1_joints = interp_trajectory(
                preapproach_pos, pre_quat, pregrasp_joints,
                pre_pos, pre_quat, pre_joints,
                close1_steps, include_start=False,
            )
            (leg2_pos, leg2_quat), leg2_planner = self._plan_contact_leg(
                surface, pre_pos, pre_quat, grasp_pos, grasp_quat, pre_joints,
                seconds=self.config.final_close_seconds,
            )
            leg2_joints = np.tile(pre_joints[None], (leg2_pos.shape[0], 1))
            close3_pos, close3_quat, close3_joints = interp_trajectory(
                grasp_pos, grasp_quat, pre_joints,
                grasp_pos, grasp_quat, grasp_stage_joints,
                final_close_steps, include_start=False,
            )
            close_pos = np.concatenate([close1_pos, leg2_pos, close3_pos], axis=0)
            close_quat = np.concatenate([close1_quat, leg2_quat, close3_quat], axis=0)
            close_joints = np.concatenate([close1_joints, leg2_joints, close3_joints], axis=0)

            squeeze_pos, squeeze_quat, squeeze_joint_traj = interp_trajectory(
                grasp_pos, grasp_quat, grasp_stage_joints,
                grasp_pos, grasp_quat, squeeze_joints,
                self.config.squeeze_steps(), include_start=False,
            )
            clearance_report = {
                **clearance_report,
                "uses_record_stages": True,
                "pregrasp_to_grasp_planner": leg2_planner,
                "stage_report": grasp_record.get("stage_report"),
            }
        else:
            # --- Legacy records (no stages): repair the single action here. ---
            # Failed-grasp records can place even the PALM inside the object;
            # no finger projection can repair that. Pull the grasp wrist pose
            # back along its own approach axis until the wide-open hand clears
            # the surface, and use the corrected pose everywhere.
            grasp_pos, wrist_backoff = self._project_wrist_pose(
                world, grasp_pos, grasp_quat, pregrasp_joints,
                clearance_target_m=self.config.approach_clearance_m,
            )

            approach_pos, approach_quat, approach_joints, transit_report = self._plan_transit(
                surface, world,
                switch_pos, switch_quat,
                grasp_pos, grasp_quat, pregrasp_joints,
            )
            clearance_report = {**clearance_report, "transit": transit_report, "uses_record_stages": False}

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
            # Hold the contact (grasp) pose through the squeeze segment and
            # carry -- no drive-through past contact.
            squeeze_joints = contact_joints
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

        static_obj_pos0, static_obj_quat0 = matrix_to_pos_quat(static_object_pose[None])
        static_obj_pos0, static_obj_quat0 = static_obj_pos0[0], static_obj_quat0[0]

        # --- Settle hold: park the wrist at the squeeze-end pose with the
        # squeeze targets held so the physics contacts converge before any
        # load transfer (appended to the squeeze segment: same drive gains,
        # and the carry-only lift metrics stay unpolluted). ---
        settle_steps = int(round(self.config.settle_seconds * self.config.fps))
        if settle_steps > 0:
            _append(
                np.tile(hand_pos_chunks[-1][-1][None], (settle_steps, 1)),
                np.tile(hand_quat_chunks[-1][-1][None], (settle_steps, 1)),
                np.tile(squeeze_joints[None], (settle_steps, 1)),
                np.tile(static_obj_pos0[None], (settle_steps, 1)),
                np.tile(static_obj_quat0[None], (settle_steps, 1)),
                SEGMENT_SQUEEZE, settle_steps,
            )

        # --- Segment c: carry ------------------------------------------------
        carry_frame = (
            analysis.grasp_frame_index
            if self.config.carry_start == CARRY_START_GRASP_FRAME
            else analysis.pickup_frame_index
        )
        carry_report: dict = {"carry_style": self.config.carry_style, "settle_steps": settle_steps}
        if self.config.carry_style == CARRY_STYLE_VERTICAL_LIFT:
            # Straight world-vertical lift from the squeeze-end pose (fixed
            # orientation, squeeze finger targets), then a hold at the top.
            # The cosine ease starts and ends at zero velocity, so there is no
            # boundary blend and no jerk at lift-off; its peak speed is
            # (pi/2) * height / duration, kept under the wrist speed cap by
            # growing the step count.
            up_cam, up_report = world_up_in_camera(sequence_dir, self.config.dexycb_manifest)
            n_lift = max(1, int(round(self.config.carry_lift_seconds * self.config.fps)))
            if self.config.max_wrist_speed_mps > 0:
                n_lift = max(n_lift, int(np.ceil(
                    (np.pi / 2.0) * self.config.carry_lift_height_m * self.config.fps
                    / self.config.max_wrist_speed_mps
                )))
            n_hold = int(round(self.config.carry_hold_seconds * self.config.fps))
            alpha = (1.0 - np.cos(np.pi * np.arange(1, n_lift + 1) / n_lift)) / 2.0
            alpha = np.concatenate([alpha, np.ones(n_hold)])
            offsets = alpha[:, None] * self.config.carry_lift_height_m * up_cam[None]
            hand_pos_cam = hand_pos_chunks[-1][-1][None] + offsets
            hand_quat_cam = np.tile(hand_quat_chunks[-1][-1][None], (alpha.size, 1))
            obj_pos_cam = static_obj_pos0[None] + offsets
            obj_quat_cam = np.tile(static_obj_quat0[None], (alpha.size, 1))
            joints_carry = np.tile(squeeze_joints[None], (alpha.size, 1))
            _append(hand_pos_cam, hand_quat_cam, joints_carry, obj_pos_cam, obj_quat_cam, SEGMENT_CARRY, alpha.size)
            carry_report.update({
                "lift_steps": int(n_lift),
                "hold_steps": int(n_hold),
                "lift_height_m": float(self.config.carry_lift_height_m),
                "peak_wrist_speed_mps": float(
                    (np.pi / 2.0) * self.config.carry_lift_height_m * self.config.fps / n_lift
                ),
                **up_report,
            })
        else:
            # Follow the recorded human carry trajectory (object-relative).
            carry_indices = demo.valid_indices[demo.valid_indices >= carry_frame]
            carry_resample_report: dict = {"raw_steps": 0, "resampled_steps": 0}
            if carry_indices.size:
                obj_pose_cam = demo.object_pose_camera[carry_indices]
                obj_pos_cam, obj_quat_cam = matrix_to_pos_quat(obj_pose_cam)
                obj_pos_cam, obj_quat_cam, carry_resample_report = self._resample_carry_object_poses(
                    obj_pos_cam, obj_quat_cam, grasp_root_tf
                )
                obj_pose_cam = pos_quat_to_matrix(obj_pos_cam, obj_quat_cam)
                hand_pose_cam = obj_pose_cam @ grasp_root_tf[None]
                hand_pos_cam, hand_quat_cam = matrix_to_pos_quat(hand_pose_cam)
                # Ease out of the squeeze-end pose: segment b froze the object at
                # its switch-frame pose, while carry follows the live recording,
                # so without a blend the boundary has a visible pose jump.
                n_blend = max(1, int(round(self.config.carry_blend_seconds * self.config.fps)))
                hand_pos_cam, hand_quat_cam = blend_into_trajectory(
                    hand_pos_chunks[-1][-1], hand_quat_chunks[-1][-1], hand_pos_cam, hand_quat_cam, n_blend
                )
                steps_before_final_cap = int(hand_pos_cam.shape[0])
                hand_pos_cam, hand_quat_cam, obj_pos_cam, obj_quat_cam = self._cap_synchronized_pose_speed(
                    hand_pos_cam, hand_quat_cam, obj_pos_cam, obj_quat_cam
                )
                actual_step = np.linalg.norm(np.diff(hand_pos_cam, axis=0), axis=1)
                carry_resample_report["steps_before_boundary_speed_cap"] = steps_before_final_cap
                carry_resample_report["steps_after_boundary_speed_cap"] = int(hand_pos_cam.shape[0])
                carry_resample_report["max_hand_speed_after_boundary_blend_mps"] = (
                    float(actual_step.max() * self.config.fps) if actual_step.size else 0.0
                )
                joints_carry = np.tile(squeeze_joints[None], (hand_pos_cam.shape[0], 1))
                _append(hand_pos_cam, hand_quat_cam, joints_carry, obj_pos_cam, obj_quat_cam, SEGMENT_CARRY, hand_pos_cam.shape[0])
            carry_report.update({
                "num_carry_frames": int(carry_indices.size),
                "carry_resample": carry_resample_report,
            })

        hand_pos_all = np.concatenate(hand_pos_chunks, axis=0)
        hand_quat_all = np.concatenate(hand_quat_chunks, axis=0)
        finger_targets = np.concatenate(joints_chunks, axis=0)
        object_pos_all = np.concatenate(object_pos_chunks, axis=0)
        object_quat_all = np.concatenate(object_quat_chunks, axis=0)

        # --- Initial-posture self-clearance correction: the simulation
        # initializes the articulation by teleporting the joints to frame 0,
        # and with PhysX hand self-collision ON an interpenetrating start
        # posture (the retargeted open hand reads -2.7 mm on the finger-base
        # sphere pairs, i.e. overlapping hulls) is resolved impulsively on
        # the first step -- the frame-0 blowup. Only the initialization is
        # dangerous: once running, targets that bring fingers close together
        # merely produce steady, drive-absorbable contact forces. So frame 0
        # alone is spread clear (or to its anatomical maximum), and the
        # correction decays back to the original trajectory over
        # self_clearance_decay_seconds so the commanded targets stay
        # continuous; the rest of the trajectory is untouched.
        if self.config.self_clearance_buffer_m >= 0.0:
            finger_targets_orig = finger_targets.copy()
            tuner = SelfClearanceTuner(
                self.asset, self.device_cfg, self.clearance_checker,
                buffer_m=self.config.self_clearance_buffer_m,
            )
            before_min = float(tuner.min_clearances(finger_targets_orig[:1])[0])
            report_sc: dict = {
                "buffer_m": self.config.self_clearance_buffer_m,
                "frame0_before_min_clearance_m": before_min,
                "frame0_before_worst_pairs": tuner.worst_pairs(finger_targets_orig[0]),
            }
            tuned0, achieved = tuner.solve_posture(finger_targets_orig[0])
            correction = tuned0 - finger_targets_orig[0]
            n_decay = min(
                max(int(round(self.config.self_clearance_decay_seconds * self.config.fps)), 1),
                finger_targets.shape[0] - 1,
            )
            for k in range(n_decay):
                factor = 1.0 - k / float(n_decay)
                finger_targets[k] = self._clamp(finger_targets_orig[k] + correction * factor)

            # Object-clearance guard: the spread must not push a finger into
            # the object on the affected frames (hand pose re-expressed in
            # the object frame for the SDF query). Per frame, bisect the
            # largest correction scale keeping at least 0.5 mm of object
            # clearance where the original frame had it (never worsening a
            # frame already at/inside the surface). Frame 0 is typically far
            # from the object, so this rarely binds.
            idx = np.arange(n_decay)
            rel = np.linalg.inv(pos_quat_to_matrix(object_pos_all[idx], object_quat_all[idx])) @ pos_quat_to_matrix(
                hand_pos_all[idx], hand_quat_all[idx]
            )
            rel_pos, rel_quat = matrix_to_pos_quat(rel)

            def _object_clearances(joints_sub: np.ndarray) -> np.ndarray:
                actions = np.concatenate([rel_pos, rel_quat, joints_sub], axis=-1).astype(np.float32)
                return self.clearance_checker.compute_clearances(
                    self.device_cfg.to_device(actions), world
                ).cpu().numpy()

            obj_before = _object_clearances(finger_targets_orig[idx])
            obj_after = _object_clearances(finger_targets[idx])
            floor = np.minimum(obj_before, 0.0005)
            offending = obj_after < floor - 1e-6
            if offending.any():
                lo_s = np.zeros(idx.size)
                hi_s = np.ones(idx.size)
                per_frame_corr = finger_targets[idx] - finger_targets_orig[idx]
                for _ in range(10):
                    mid = 0.5 * (lo_s + hi_s)
                    cand = finger_targets_orig[idx] + mid[:, None] * per_frame_corr
                    ok = _object_clearances(cand) >= floor - 1e-6
                    lo_s = np.where(ok, mid, lo_s)
                    hi_s = np.where(ok, hi_s, mid)
                scale = np.where(offending, lo_s, 1.0)
                finger_targets[idx] = finger_targets_orig[idx] + scale[:, None] * per_frame_corr
                obj_after = _object_clearances(finger_targets[idx])
                report_sc["object_guard_min_scale"] = float(scale.min())
            report_sc.update(
                frame0_after_min_clearance_m=float(tuner.min_clearances(finger_targets[:1])[0]),
                frame0_after_worst_pairs=tuner.worst_pairs(finger_targets[0]),
                converged=bool(achieved >= self.config.self_clearance_buffer_m - 1e-4),
                max_abs_delta_rad=float(np.abs(finger_targets[0] - finger_targets_orig[0]).max()),
                decay_frames=int(n_decay),
                object_guard_frames=int(offending.sum()),
                object_clearance_before_m=float(obj_before.min()),
                object_clearance_after_m=float(obj_after.min()),
                max_step_delta_rad=float(np.abs(np.diff(finger_targets[: n_decay + 1], axis=0)).max()),
            )
            clearance_report = {**clearance_report, "self_clearance": report_sc}

        traj = GraspTrajectory(
            hand_pos_camera=hand_pos_all,
            hand_quat_camera=hand_quat_all,
            finger_targets=finger_targets,
            object_pos_camera=object_pos_all,
            object_quat_camera=object_quat_all,
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
                    "pregrasp_open_fraction": self.config.pregrasp_open_fraction,
                    "approach_clearance_m": self.config.approach_clearance_m,
                    "carry_style": self.config.carry_style,
                    "settle_seconds": self.config.settle_seconds,
                    "carry_lift_height_m": self.config.carry_lift_height_m,
                    "carry_lift_seconds": self.config.carry_lift_seconds,
                    "carry_hold_seconds": self.config.carry_hold_seconds,
                    "carry_start": self.config.carry_start,
                    "max_wrist_speed_mps": self.config.max_wrist_speed_mps,
                    "carry_blend_seconds": self.config.carry_blend_seconds,
                    "open_clearance_m": self.config.open_clearance_m,
                    "open_horizon_seconds": self.config.open_horizon_seconds,
                    "planner": self.config.planner,
                    "final_close_seconds": self.config.final_close_seconds,
                    "near_contact_margin_m": self.config.near_contact_margin_m,
                    "self_clearance_buffer_m": self.config.self_clearance_buffer_m,
                    "self_clearance_decay_seconds": self.config.self_clearance_decay_seconds,
                },
                "grasp_frame_index": int(analysis.grasp_frame_index),
                "pickup_frame_index": int(analysis.pickup_frame_index),
                "carry_start_frame": int(carry_frame),
                "object_name": grasp_record.get("object_name"),
                "object_mesh": str(surface.object_mesh_path),
                "num_retarget_frames": int(retarget_frames.size) if retarget_frames.size else 0,
                "carry": carry_report,
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
