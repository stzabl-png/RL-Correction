"""cuRobo v2 motion planning for the wide-open hand's transit to the grasp
standoff (Stage A, grasp-synthesis conda env).

The hand is modeled as a floating-base robot: a 6-DOF virtual joint chain
(3 prismatic x/y/z + 3 revolute roll/pitch/yaw, the same construction as
MagicSim's ``SharpaWaveFloating``) generated on demand from the asset's own
URDF (``ensure_floating_urdf``), with all 22 finger joints locked at the
pregrasp (wide-open) posture and the object mesh as a collision obstacle.
``MotionPlanner.plan_pose`` then produces a collision-free joint-space
trajectory whose virtual-joint values ARE the wrist pose.

Everything here is planning in the object-canonical frame, matching the rest
of Stage A's segment-b math.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch

from curobo._src.types.device_cfg import DeviceCfg

from ocir.grasp_synthesis.assets import SharpaWaveAsset
from ocir.grasp_synthesis.bodex_curobo_v2.solver import _load_yaml

VIRTUAL_JOINT_NAMES = (
    "virtual_x_joint",
    "virtual_y_joint",
    "virtual_z_joint",
    "virtual_roll_joint",
    "virtual_pitch_joint",
    "virtual_yaw_joint",
)

_FLOATING_CHAIN_TEMPLATE = """
  <!-- Generated floating-base virtual chain (see ocir.grasp_traj.planner). -->
  <link name="base_root">
    <inertial><origin xyz="0 0 0"/><mass value="1.0E-06"/><inertia ixx="2.0E-12" ixy="0" ixz="0" iyy="2.0E-12" iyz="0" izz="2.0E-12"/></inertial>
  </link>
  <link name="virtual_root_1"><inertial><origin xyz="0 0 0"/><mass value="1.0E-06"/><inertia ixx="2.0E-12" ixy="0" ixz="0" iyy="2.0E-12" iyz="0" izz="2.0E-12"/></inertial></link>
  <link name="virtual_root_2"><inertial><origin xyz="0 0 0"/><mass value="1.0E-06"/><inertia ixx="2.0E-12" ixy="0" ixz="0" iyy="2.0E-12" iyz="0" izz="2.0E-12"/></inertial></link>
  <link name="virtual_root_3"><inertial><origin xyz="0 0 0"/><mass value="1.0E-06"/><inertia ixx="2.0E-12" ixy="0" ixz="0" iyy="2.0E-12" iyz="0" izz="2.0E-12"/></inertial></link>
  <link name="virtual_root_4"><inertial><origin xyz="0 0 0"/><mass value="1.0E-06"/><inertia ixx="2.0E-12" ixy="0" ixz="0" iyy="2.0E-12" iyz="0" izz="2.0E-12"/></inertial></link>
  <link name="virtual_root_5"><inertial><origin xyz="0 0 0"/><mass value="1.0E-06"/><inertia ixx="2.0E-12" ixy="0" ixz="0" iyy="2.0E-12" iyz="0" izz="2.0E-12"/></inertial></link>
  <joint name="virtual_x_joint" type="prismatic">
    <parent link="base_root"/><child link="virtual_root_1"/><axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="2.0" effort="200" velocity="2.0"/>
  </joint>
  <joint name="virtual_y_joint" type="prismatic">
    <parent link="virtual_root_1"/><child link="virtual_root_2"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="200" velocity="2.0"/>
  </joint>
  <joint name="virtual_z_joint" type="prismatic">
    <parent link="virtual_root_2"/><child link="virtual_root_3"/><axis xyz="0 0 1"/>
    <limit lower="-2.0" upper="2.0" effort="200" velocity="2.0"/>
  </joint>
  <joint name="virtual_roll_joint" type="revolute">
    <parent link="virtual_root_3"/><child link="virtual_root_4"/><axis xyz="1 0 0"/>
    <limit lower="-6.283185307179586" upper="6.283185307179586" effort="200" velocity="2.0"/>
  </joint>
  <joint name="virtual_pitch_joint" type="revolute">
    <parent link="virtual_root_4"/><child link="virtual_root_5"/><axis xyz="0 1 0"/>
    <limit lower="-6.283185307179586" upper="6.283185307179586" effort="200" velocity="2.0"/>
  </joint>
  <joint name="virtual_yaw_joint" type="revolute">
    <parent link="virtual_root_5"/><child link="{base_link}"/><axis xyz="0 0 1"/>
    <limit lower="-6.283185307179586" upper="6.283185307179586" effort="200" velocity="2.0"/>
  </joint>
"""


def ensure_floating_urdf(asset: SharpaWaveAsset) -> Path:
    """Generate (and cache next to the source URDF, so relative mesh paths
    keep resolving) a floating-base variant of the hand URDF: the asset's own
    URDF with a 6-DOF virtual joint chain from a new ``base_root`` link to
    the hand's base link. Regenerated whenever the source URDF is newer."""

    src = Path(asset.urdf_path)
    dst = src.with_name(src.stem + "_floating.generated.urdf")
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst

    base_link = asset.config["base_link"]
    text = src.read_text(encoding="utf-8")
    match = re.search(r"<robot[^>]*>", text)
    if match is None:
        raise ValueError(f"no <robot> element found in {src}")
    chain = _FLOATING_CHAIN_TEMPLATE.format(base_link=base_link)
    out = text[: match.end()] + chain + text[match.end() :]
    dst.write_text(out, encoding="utf-8")
    return dst


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic X-Y-Z (the virtual chain applies roll, then pitch, then yaw
    each in the previously rotated frame): R = Rx(roll) @ Ry(pitch) @ Rz(yaw)."""

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rx @ ry @ rz


def matrix_to_rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`rpy_to_matrix` (intrinsic X-Y-Z Tait-Bryan)."""

    pitch = float(np.arcsin(np.clip(rot[0, 2], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(-rot[1, 2], rot[2, 2]))
        yaw = float(np.arctan2(-rot[0, 1], rot[0, 0]))
    else:
        # Gimbal lock: yaw is unobservable, fold everything into roll.
        roll = float(np.arctan2(rot[2, 1], rot[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


class TransitPlanner:
    """Plans the wide-open hand's wrist path (object-canonical frame) with
    cuRobo v2's MotionPlanner. One instance per (sequence, pregrasp posture):
    the finger joints are locked at construction and the object mesh loaded
    as the collision world."""

    def __init__(
        self,
        asset: SharpaWaveAsset,
        pregrasp_joints_by_name: dict[str, float],
        object_mesh_path: str | Path,
        device_cfg: DeviceCfg,
        *,
        collision_activation_distance_m: float = 0.002,
    ):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        self.device_cfg = device_cfg
        floating_urdf = ensure_floating_urdf(asset)
        collision_config = _load_yaml(asset.collision_spheres_path)
        spheres_map = collision_config["collision_spheres"]
        collision_link_names = list(spheres_map.keys())
        # The hand's neighboring-link sphere overlaps are intentional; reuse
        # the ignore/buffer maps from the existing BODex robot config rather
        # than re-deriving them (an empty ignore map flags every pose as
        # self-colliding and the planner never succeeds).
        bodex_robot_kin = _load_yaml(asset.bodex_path("robot_config"))["robot_cfg"]["kinematics"]
        self_collision_ignore = bodex_robot_kin.get("self_collision_ignore", {})
        self_collision_buffer = bodex_robot_kin.get("self_collision_buffer", {})

        finger_names = list(pregrasp_joints_by_name.keys())
        joint_names = list(VIRTUAL_JOINT_NAMES) + finger_names
        n = len(joint_names)
        robot_dict = {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": str(floating_urdf),
                    "base_link": "base_root",
                    "tool_frames": [asset.config["base_link"]],
                    "collision_link_names": collision_link_names,
                    "collision_spheres": spheres_map,
                    "collision_sphere_buffer": 0.0,
                    "lock_joints": {k: float(v) for k, v in pregrasp_joints_by_name.items()},
                    "self_collision_ignore": self_collision_ignore,
                    "self_collision_buffer": self_collision_buffer,
                    "cspace": {
                        "joint_names": joint_names,
                        "default_joint_position": [0.0] * len(VIRTUAL_JOINT_NAMES)
                        + [float(pregrasp_joints_by_name[k]) for k in finger_names],
                        "cspace_distance_weight": [1.0] * n,
                        "null_space_weight": [1.0] * n,
                        "max_acceleration": 15.0,
                        "max_jerk": 500.0,
                    },
                }
            }
        }

        # Self-collision must also be dropped from the METRICS rollout (which
        # gates plan success): MotionPlannerCfg.create's self_collision_check
        # flag only disables it in the optimizer rollouts, and the hand's
        # locked wide-open fingers trip cross-finger sphere overlaps that are
        # irrelevant here.
        from curobo._src.util.config_io import resolve_config
        from curobo.content import get_task_configs_path

        import tempfile

        import yaml

        metrics_dict = resolve_config(str(get_task_configs_path() / "metrics_base.yml"))
        metrics_dict["rollout"]["constraint_cfg"].pop("self_collision_cfg", None)
        # MotionPlannerCfg.create hands the same metrics config to several
        # consumers, each of which mutates a dict in place -- but re-reads
        # string paths fresh. Passing a file path is the only sharing-safe
        # form.
        metrics_file = tempfile.NamedTemporaryFile(
            mode="w", suffix="_metrics_no_selfcol.yml", delete=False
        )
        yaml.safe_dump(metrics_dict, metrics_file)
        metrics_file.close()
        self._metrics_yaml_path = metrics_file.name

        from curobo._src.geom.types import Mesh, SceneCfg

        scene = SceneCfg(
            mesh=[
                Mesh(
                    name="object",
                    file_path=str(object_mesh_path),
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                )
            ]
        )
        cfg = MotionPlannerCfg.create(
            robot=robot_dict,
            scene_model=scene,
            self_collision_check=False,
            metrics_rollout=self._metrics_yaml_path,
            graph_planner_rollout=self._metrics_yaml_path,
            device_cfg=device_cfg,
            optimizer_collision_activation_distance=collision_activation_distance_m,
            # The endpoint is re-pinned to the exact goal pose after
            # resampling, so a slightly loose orientation tolerance only
            # trades a tiny slerp kink at the join for far fewer IK failures
            # near rpy-chart boundaries.
            orientation_tolerance=0.08,
        )
        self.planner = MotionPlanner(cfg)
        self.tool_frame = asset.config["base_link"]

    @staticmethod
    def _pose_to_virtual_q(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        from ocir.grasp_traj.trajectory_schema import quat_wxyz_to_matrix

        roll, pitch, yaw = matrix_to_rpy(quat_wxyz_to_matrix(np.asarray(quat_wxyz, dtype=np.float64)))
        p = np.asarray(pos, dtype=np.float64)
        return np.array([p[0], p[1], p[2], roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def _virtual_q_to_poses(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from ocir.grasp_traj.trajectory_schema import matrix_to_quat_wxyz

        q = np.asarray(q, dtype=np.float64)
        positions = q[:, :3].copy()
        quats = np.stack([matrix_to_quat_wxyz(rpy_to_matrix(*row[3:6])) for row in q], axis=0)
        return positions, quats

    def plan(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        *,
        seconds: float,
        fps: float,
        max_speed_mps: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Plan start -> goal wrist poses; returns ``(positions, quats)``
        resampled at ``fps`` (goal-inclusive, start-exclusive, matching the
        include_start=False convention of the segment builders) with the step
        count derived from the PLANNED path length (so the speed cap holds on
        detours, not just the straight-line distance), or ``None`` if
        planning failed."""

        from ocir.grasp_traj.segments import steps_for_leg

        from curobo._src.state.state_joint import JointState
        from curobo._src.types.tool_pose import GoalToolPose

        q_start = self._pose_to_virtual_q(start_pos, start_quat)
        start_state = JointState.from_position(
            self.device_cfg.to_device(q_start.astype(np.float32)).unsqueeze(0),
            joint_names=list(VIRTUAL_JOINT_NAMES),
        )
        goal_position = self.device_cfg.to_device(
            np.asarray(goal_pos, dtype=np.float32).reshape(1, 1, 1, 1, 3)
        )
        goal_quaternion = self.device_cfg.to_device(
            np.asarray(goal_quat, dtype=np.float32).reshape(1, 1, 1, 1, 4)
        )
        goal = GoalToolPose(
            tool_frames=[self.tool_frame],
            position=goal_position,
            quaternion=goal_quaternion,
        )
        result = self.planner.plan_pose(goal, start_state)
        if result is None or result.success is None or not bool(result.success.reshape(-1)[0].item()):
            return None

        traj = result.interpolated_trajectory
        if traj is None:
            traj = result.js_solution
        if traj is None:
            return None
        q_path = traj.position
        q_path = q_path.reshape(-1, q_path.shape[-1]).detach().cpu().numpy().astype(np.float64)
        if result.interpolated_last_tstep is not None:
            last = int(result.interpolated_last_tstep.reshape(-1)[0].item())
            if last > 1:
                q_path = q_path[:last]

        # Arc-length parameterization (with a tiny per-step epsilon so pure-
        # rotation or dwell frames stay monotonic): the solver's interpolated
        # output contains dwell frames that index-based resampling would keep
        # as dead time.
        step_norms = np.linalg.norm(np.diff(q_path, axis=0), axis=1)
        path_length = float(np.linalg.norm(np.diff(q_path[:, :3], axis=0), axis=1).sum())
        n_steps = steps_for_leg(path_length, seconds, fps, max_speed_mps)

        arc = np.concatenate([[0.0], np.cumsum(step_norms + 1e-9)])
        ts = arc / arc[-1]
        ts_new = np.linspace(0.0, 1.0, int(n_steps) + 1)[1:]
        q_resampled = np.stack(
            [np.interp(ts_new, ts, q_path[:, j]) for j in range(q_path.shape[1])], axis=1
        )
        # Pin the endpoint exactly to the requested goal pose (the planner is
        # within tolerance, not exact).
        positions, quats = self._virtual_q_to_poses(q_resampled)
        positions[-1] = np.asarray(goal_pos, dtype=np.float64)
        quats[-1] = np.asarray(goal_quat, dtype=np.float64)
        return positions, quats
