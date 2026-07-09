"""Official-cuRobo-v2 rollout/optimizer wiring for exact BODex Sharpa synthesis."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
import yaml

from ocir.grasp_synthesis.bodex_curobo_v2.backend import import_official_curobo

import_official_curobo()

from curobo._src.geom.transform import torch_quaternion_to_matrix
from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.robot.kinematics.kinematics import Kinematics

from ocir.grasp_synthesis.assets import load_sharpa_wave_right
from ocir.grasp_synthesis.bodex_curobo_v2.contact_world import SingleObjectContactWorld
from ocir.grasp_synthesis.bodex_curobo_v2.grasp_cost import BodexGraspCost, BodexGraspCostConfig
from ocir.grasp_synthesis.bodex_curobo_v2.grasp_energy import normalize_vector
from ocir.grasp_synthesis.bodex_curobo_v2.newton_opt import BodexNewtonOpt, BodexNewtonOptCfg
from ocir.grasp_synthesis.bodex_curobo_v2.seed_generator import (
    HeurGraspSeedGenerator,
    load_hand_pose_transfer,
    sample_surface_points_and_normals,
)
from ocir.grasp_synthesis.object_surface import ObjectSurface


# BODex full-hand (fingertip + pad + palm) contact points for the Sharpa
# right hand, "LinkName/sphere_index" format. Matches
# assets/robots/hands/sharpa_wave/grasp_synthesis/bodex/fc.yml's
# grasp_contact_strategy.contact_points_name, and the original-BODex pipeline's
# SHARPA_CONTACT_POINTS (src/ocir/grasp_synthesis/bodex/synthesize_original_bodex.py)
# so both pipelines optimize the exact same contact points in the same order.
# (fc_5finger.yml's fingertip-only 5-point subset is still available for
# small/handle-shaped objects but is no longer the default.)
SHARPA_CONTACT_POINTS = [
    "right_pinky_DP/1",
    "right_pinky_PP/0",
    "right_ring_PP/0",
    "right_ring_DP/1",
    "right_middle_PP/0",
    "right_middle_DP/1",
    "right_index_PP/0",
    "right_index_DP/1",
    "right_thumb_PP/0",
    "right_thumb_DP/1",
    "right_hand_C_MC/0",
]

# Real Sharpa fc.yml values (verified against
# assets/robots/hands/sharpa_wave/grasp_synthesis/bodex/fc.yml and
# third_party/BODex/src/curobo/content/configs/task/gradient_grasp_fc.yml).
DEFAULT_TASK_DICT = {"f": [0.0, 0.0, 1.0], "t": [0.0, 0.0, 0.0], "gamma": 180.0}
DEFAULT_CONTACT_STRATEGY = {
    "opt_progress": [0.0, 0.6, 0.8],
    "contact_query_mode": [-1, 0, 0],
    "distance": [0.02, 0.01, 0.0],
    "save_qpos": [False, True],
    "max_ge_stage": 0,
}
DEFAULT_GE_PARAM = {
    "type": "qp",
    "k_lower": 0.3,
    # Contact index order: 0-1 pinky, 2-3 ring, 4-5 middle, 6-7 index,
    # 8-9 thumb, 10 palm.
    "pressure_constraints": [
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1.0],
        [[8, 9], 0.5],
        [[0, 1, 2, 3, 4, 5, 6, 7], 0.7],
        [[0, 1, 2, 3], 0.4],
        [[4, 5, 6, 7], 0.5],
    ],
    "solver_type": "batch_reluqp",
    "miu_coef": [0.3, 0.0],
    "enable_density": False,
    "solve_interval": 5,
}
DEFAULT_WEIGHT = [100.0, 1000.0, 10.0]
DEFAULT_PERTURB_STRENGTH_BOUND = [[0.01], [0.02]]
DEFAULT_GRASP_THRESHOLD = 0.001
DEFAULT_DISTANCE_THRESHOLD = 0.01


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _sphere_lookup(collision_config: dict, link: str, sphere_index: int) -> tuple[np.ndarray, float]:
    spheres_map = collision_config.get("collision_spheres")
    if not isinstance(spheres_map, dict):
        spheres_map = collision_config.get("geometry", {}).get("collision_spheres", {}).get("spheres", {})
    spheres = spheres_map.get(link)
    if not spheres:
        raise KeyError(f"no collision spheres for link {link}")
    if sphere_index >= len(spheres):
        raise IndexError(f"sphere index {sphere_index} out of range for {link}")
    sphere = spheres[int(sphere_index)]
    return np.asarray(sphere["center"], dtype=np.float64), float(sphere["radius"])


def transform_points(root_pos: torch.Tensor, root_quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate+translate ``points`` by the (root_pos, root_quat) pose.

    Uses official cuRobo v2's ``torch_quaternion_to_matrix`` (a plain,
    stride-agnostic op with no custom backward) for the rotation, then a
    broadcasting matrix multiply for the actual point transform. Deliberately
    does NOT use cuRobo v2's ``transform_points``/``pose_multiply`` here:
    those are custom ``torch.autograd.Function``s (Warp-kernel-backed) whose
    backward requires every downstream consumer of their output to hand back
    a contiguous gradient -- fine when isolated inside another custom
    Function's own hand-written backward (as `contact_world.py` does), but
    fragile here, where the result flows through many further ordinary
    autograd ops (concatenation, indexing, staged costs) before the final
    loss. A plain broadcast implementation tolerates arbitrary strides
    throughout that whole downstream graph.
    """

    rot = torch_quaternion_to_matrix(root_quat)
    if points.ndim == root_pos.ndim:
        return (rot @ points.unsqueeze(-1)).squeeze(-1) + root_pos
    if points.ndim == root_pos.ndim + 1:
        return (rot.unsqueeze(-3) @ points.unsqueeze(-1)).squeeze(-1) + root_pos.unsqueeze(-2)
    raise ValueError(f"unsupported point shape {tuple(points.shape)} for root_pos {tuple(root_pos.shape)}")


def compose_root_pose(root_pos: torch.Tensor, root_quat: torch.Tensor, local_pos: torch.Tensor, local_quat: torch.Tensor) -> torch.Tensor:
    """Compose a per-batch root pose with N per-batch-item local poses.

    See `transform_points` docstring for why this uses plain broadcasting
    math (via cuRobo v2's `torch_quaternion_to_matrix`) rather than cuRobo
    v2's custom-autograd `pose_multiply`.
    """

    world_pos = transform_points(root_pos, root_quat, local_pos)
    rw, rx, ry, rz = torch.unbind(root_quat.unsqueeze(-2), dim=-1)
    lw, lx, ly, lz = torch.unbind(local_quat, dim=-1)
    world_quat = torch.stack(
        [
            rw * lw - rx * lx - ry * ly - rz * lz,
            rw * lx + rx * lw + ry * lz - rz * ly,
            rw * ly - rx * lz + ry * lw + rz * lx,
            rw * lz + rx * ly - ry * lx + rz * lw,
        ],
        dim=-1,
    )
    return torch.cat([world_pos, normalize_vector(world_quat)], dim=-1)


def batched_transform_points(pos: torch.Tensor, quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform (B, N, 3) points by per-item (B, N, 3)/(B, N, 4) poses (plain
    broadcasting matmul; see `transform_points` docstring for why this
    avoids cuRobo v2's custom-autograd transform ops)."""

    rot = torch_quaternion_to_matrix(quat)
    return (rot @ points.unsqueeze(-1)).squeeze(-1) + pos


def find_object_mesh_from_surface(surface: ObjectSurface, explicit: Path | None = None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(explicit)
    raw = surface.metadata.get("object_points_path")
    if raw is None:
        raise FileNotFoundError("surface artifact has no object_points_path metadata")
    model_dir = Path(str(raw)).expanduser().parent
    candidates = [
        model_dir / "textured_simple.obj",
        model_dir / "textured.obj",
        model_dir / f"{model_dir.name}.stl",
    ]
    candidates.extend(sorted(model_dir.glob("*.obj")))
    candidates.extend(sorted(model_dir.glob("*.stl")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no mesh found near {model_dir}")


def expand_active_action_to_full_joint_order(
    active_action: np.ndarray,
    active_joint_names: list[str],
    full_joint_order: list[str],
    full_neutral_q: np.ndarray,
) -> np.ndarray:
    """Expand optimized cuRobo-active joints into the full Sharpa action vector."""

    active_action = np.asarray(active_action, dtype=float)
    full_action = np.zeros((7 + len(full_joint_order),), dtype=float)
    full_action[:7] = active_action[:7]
    full_q = np.asarray(full_neutral_q, dtype=float).copy()
    active_q = active_action[7:]
    active_index = {name: idx for idx, name in enumerate(active_joint_names)}
    for full_idx, joint_name in enumerate(full_joint_order):
        if joint_name in active_index:
            full_q[full_idx] = active_q[active_index[joint_name]]
    full_action[7:] = full_q
    return full_action


@dataclass
class SharpaBodexRollout:
    device_cfg: DeviceCfg
    surface: ObjectSurface
    object_mesh_path: Path
    root_bounds_center: np.ndarray
    root_bounds_radius: float
    opt_iters: int = 500

    def __post_init__(self):
        self.asset = load_sharpa_wave_right()
        self.full_joint_order = list(self.asset.config["joint_order"])
        self.full_neutral_q = np.zeros((len(self.full_joint_order),), dtype=np.float64)

        collision_config = _load_yaml(self.asset.collision_spheres_path)
        contact_link_order: list[str] = []
        local_centers = []
        local_radii = []
        for entry in SHARPA_CONTACT_POINTS:
            link_name, sphere_idx = entry.split("/")
            center, radius = _sphere_lookup(collision_config, link_name, int(sphere_idx))
            if link_name not in contact_link_order:
                contact_link_order.append(link_name)
            local_centers.append(center)
            local_radii.append(radius)
        self.contact_link_names = tuple(contact_link_order)
        self.contact_sphere_links = [entry.split("/")[0] for entry in SHARPA_CONTACT_POINTS]
        local = [np.concatenate([c, [r]], axis=0) for c, r in zip(local_centers, local_radii)]
        self.contact_spheres_local = self.device_cfg.to_device(np.asarray(local, dtype=np.float32))

        self.kin_link_names = list(dict.fromkeys(list(self.contact_link_names) + [self.asset.config["base_link"]]))
        robot_cfg = RobotCfg.from_basic(
            urdf_path=str(self.asset.urdf_path),
            base_link=self.asset.config["base_link"],
            tool_frames=self.kin_link_names,
            device_cfg=self.device_cfg,
        )
        self.kinematics = Kinematics(robot_cfg.kinematics)
        self.joint_names = list(self.kinematics.joint_names)
        extra = [name for name in self.joint_names if name not in self.full_joint_order]
        if extra:
            raise ValueError(f"official cuRobo loaded joints not present in Sharpa asset config: {extra}")
        self.passive_joint_names = [name for name in self.full_joint_order if name not in self.joint_names]

        self._batch_size = 1
        self.time_action_horizon = 1
        self.time_horizon = 1
        self.sum_horizon = True
        center = self.device_cfg.to_device(np.asarray(self.root_bounds_center, dtype=np.float32))
        rad = float(self.root_bounds_radius)
        joint_lower = torch.tensor(
            [self.asset.config["joint_limits"][name][0] for name in self.joint_names],
            device=self.device_cfg.device, dtype=self.device_cfg.dtype,
        )
        joint_upper = torch.tensor(
            [self.asset.config["joint_limits"][name][1] for name in self.joint_names],
            device=self.device_cfg.device, dtype=self.device_cfg.dtype,
        )
        self._action_bound_lows = torch.cat([center - rad, self.device_cfg.to_device([-1, -1, -1, -1]), joint_lower])
        self._action_bound_highs = torch.cat([center + rad, self.device_cfg.to_device([1, 1, 1, 1]), joint_upper])

        contact_idx = torch.arange(len(SHARPA_CONTACT_POINTS), device=self.device_cfg.device, dtype=torch.long)
        self.contact_world = SingleObjectContactWorld(
            object_mesh_path=self.object_mesh_path,
            robot_urdf_path=self.asset.urdf_path,
            contact_link_names=self.contact_link_names,
            device_cfg=self.device_cfg,
        )
        points = self.device_cfg.to_device(self.surface.points_object_frame.astype(np.float32))
        center_obj = points.mean(dim=0, keepdim=True)
        extent = torch.clamp(points.max(dim=0).values - points.min(dim=0).values, min=1e-3).max().view(1, 1)
        self.object_gravity_center = center_obj
        self.object_obb_length = extent
        cfg = BodexGraspCostConfig(
            weight=list(DEFAULT_WEIGHT),
            device_cfg=self.device_cfg,
            task_dict=dict(DEFAULT_TASK_DICT),
            ge_param={
                **DEFAULT_GE_PARAM,
                "obj_gravity_center": center_obj,
                "obj_obb_length": extent,
            },
            perturb_strength_bound=DEFAULT_PERTURB_STRENGTH_BOUND,
            contact_points_idx=contact_idx,
            contact_mesh_idx=list(range(len(self.contact_link_names))),
            world_coll_checker=self.contact_world,
            finger_num=len(SHARPA_CONTACT_POINTS),
            contact_strategy=dict(DEFAULT_CONTACT_STRATEGY),
        )
        self.grasp_cost = BodexGraspCost(cfg)
        # Separate instance dedicated to final/exact metric evaluation
        # (`evaluate_exact_grasp_metrics`), mirroring original BODex's
        # independent `grasp_convergence` GraspCost. `QPEnergy` only re-solves
        # its QP every `solve_interval` calls (a perf optimization for the
        # in-loop optimization cost); reusing `self.grasp_cost` here would
        # read a QP solution stale from hundreds of earlier, less-converged
        # optimization iterations. A fresh instance has `count=0`, so its
        # first (and only) `evaluate()` call always re-solves the QP against
        # the actual final contact geometry.
        self.grasp_convergence_cost = BodexGraspCost(cfg)

        # BODex-faithful surface-normal-facing seed generator (replaces a
        # naive random-position/identity-orientation/random-joints seed,
        # which has no mechanism to orient the palm toward the object and
        # can converge to a physically backwards grasp, e.g. fingernails
        # touching instead of fingerpads).
        grasp_synthesis_cfg = _load_yaml(self.asset.bodex_path("grasp_synthesis_config"))
        seeder_cfg = grasp_synthesis_cfg["seeder_cfg"]
        robot_config_yaml = _load_yaml(self.asset.bodex_path("robot_config"))
        cspace_joint_names = list(robot_config_yaml["robot_cfg"]["kinematics"]["cspace"]["joint_names"])
        hand_pose_transfer = load_hand_pose_transfer(self.asset.bodex_path("hand_pose_transfer"), self.device_cfg)
        transfer_rot, transfer_trans = hand_pose_transfer[self.asset.config["base_link"]]
        self.seed_generator = HeurGraspSeedGenerator(
            device_cfg=self.device_cfg,
            base_link=self.asset.config["base_link"],
            joint_order=self.joint_names,
            seeder_cfg=seeder_cfg,
            cspace_joint_names=cspace_joint_names,
            transfer_rot=transfer_rot,
            transfer_trans=transfer_trans,
        )
        obj_sample_cfg = seeder_cfg["obj_sample"]
        object_trimesh = trimesh.load(str(self.object_mesh_path), force="mesh", process=False)
        surface_points_np, surface_normals_np = sample_surface_points_and_normals(
            object_trimesh,
            num=int(obj_sample_cfg["num"]),
            inflate=float(obj_sample_cfg["inflate"]),
            convex_hull=bool(obj_sample_cfg["convex_hull"]),
        )
        self.seed_generator.reset(
            self.device_cfg.to_device(surface_points_np.astype(np.float32)),
            self.device_cfg.to_device(surface_normals_np.astype(np.float32)),
        )

    @property
    def action_dim(self) -> int:
        return 7 + len(self.joint_names)

    @property
    def action_horizon(self) -> int:
        return self.time_action_horizon

    @property
    def horizon(self) -> int:
        return self.time_horizon

    @property
    def action_bound_lows(self) -> torch.Tensor:
        return self._action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        return self._action_bound_highs

    @property
    def dt(self) -> float:
        return 1.0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = int(value)

    def update_batch_size(self, batch_size: int) -> None:
        self._batch_size = int(batch_size)

    def update_params(self, **kwargs) -> bool:
        return True

    def update_dt(self, dt, **kwargs) -> bool:
        return True

    def reset(self, reset_problem_ids=None, **kwargs) -> bool:
        return True

    def reset_shape(self) -> bool:
        return True

    def reset_seed(self) -> None:
        return None

    def reset_cuda_graph(self) -> bool:
        return True

    def get_initial_action(self, use_random: bool = True, use_zero: bool = False, **kwargs) -> torch.Tensor:
        n = self.batch_size or 1
        if use_random:
            action = self.seed_generator.get_samples(n)
        else:
            center = (self.action_bound_lows + self.action_bound_highs) * 0.5
            action = center.unsqueeze(0).repeat(n, 1)
            action[:, 3:7] = self.device_cfg.to_device([1, 0, 0, 0]).view(1, 4)
        return action.view(n, self.action_horizon, self.action_dim)

    def _state_from_action(self, action: torch.Tensor) -> JointState:
        return JointState.from_position(action)

    def _fk_contact_inputs(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = action[:, 0, :]
        root_pos = flat[:, :3]
        root_quat = normalize_vector(flat[:, 3:7])
        q = flat[:, 7:].contiguous()
        kin_state = self.kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=self.joint_names)
        )
        link_poses = []
        for link in self.contact_link_names:
            pose = kin_state.tool_poses.get_link_pose(link)
            local_pose = torch.cat([pose.position, pose.quaternion], dim=-1)
            link_poses.append(local_pose)
        local_link_pose = torch.stack(link_poses, dim=1)
        world_link_pose = compose_root_pose(root_pos, root_quat, local_link_pose[..., :3], local_link_pose[..., 3:])

        b = root_pos.shape[0]
        n_contacts = len(self.contact_sphere_links)
        sphere_link_pos = torch.stack(
            [kin_state.tool_poses.get_link_pose(link).position for link in self.contact_sphere_links], dim=1
        )  # (B, n_contacts, 3)
        sphere_link_quat = torch.stack(
            [kin_state.tool_poses.get_link_pose(link).quaternion for link in self.contact_sphere_links], dim=1
        )  # (B, n_contacts, 4)
        local_offsets = self.contact_spheres_local[:, :3].unsqueeze(0).expand(b, n_contacts, 3)
        radii = self.contact_spheres_local[:, 3].view(1, n_contacts, 1).expand(b, n_contacts, 1)
        sphere_in_link_frame_world = batched_transform_points(sphere_link_pos, sphere_link_quat, local_offsets)
        sphere_world_pos = transform_points(root_pos, root_quat, sphere_in_link_frame_world)
        robot_spheres = torch.cat([sphere_world_pos, radii], dim=-1).unsqueeze(1)
        return robot_spheres, world_link_pose.unsqueeze(1)

    def _costs_and_constraints(self, state: JointState, opt_progress: float = 0.0) -> CostsAndConstraints:
        action = state.position
        robot_spheres, link_pos_quat = self._fk_contact_inputs(action)
        env_idx = torch.zeros((action.shape[0],), device=self.device_cfg.device, dtype=torch.int32)
        e_angle, e_dist, e_regu, debug = self.grasp_cost.forward(robot_spheres, link_pos_quat, env_idx, opt_progress)
        quat_norm_cost = ((action[:, 0, 3:7].norm(dim=-1) - 1.0) ** 2).view(-1, 1, 1) * 10.0
        costs = CostsAndConstraints()
        costs.costs.add(e_angle.view(-1, 1, 1), "bodex_grasp_energy")
        costs.costs.add(e_dist.view(-1, 1, 1), "bodex_contact_distance")
        costs.costs.add(e_regu.view(-1, 1, 1), "bodex_regularization")
        costs.costs.add(quat_norm_cost, "root_quaternion_unit")
        costs.debug = debug
        return costs

    def evaluate_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutResult:
        if act_seq.shape[0] != self._batch_size:
            self.update_batch_size(act_seq.shape[0])
        act_seq = torch.cat(
            [
                act_seq[:, :, :3],
                normalize_vector(act_seq[:, :, 3:7]),
                act_seq[:, :, 7:],
            ],
            dim=-1,
        )
        state = self._state_from_action(act_seq)
        costs = self._costs_and_constraints(state, kwargs.get("opt_progress", 0.0))
        return RolloutResult(actions=act_seq, state=state, costs_and_constraints=costs, debug=getattr(costs, "debug", None))

    def compute_metrics_from_state(self, state: JointState, **kwargs) -> RolloutMetrics:
        costs = self._costs_and_constraints(state, kwargs.get("opt_progress", 1.0))
        convergence = CostCollection()
        convergence.add(costs.get_sum_cost(sum_horizon=False), "sum_cost")
        return RolloutMetrics(
            actions=state.position,
            state=state,
            costs_and_constraints=costs,
            feasible=torch.ones((state.position.shape[0], 1), device=self.device_cfg.device, dtype=torch.bool),
            convergence=convergence,
            debug=getattr(costs, "debug", None),
        )

    def compute_metrics_from_action(self, act_seq: torch.Tensor, **kwargs) -> RolloutMetrics:
        result = self.evaluate_action(act_seq, **kwargs)
        return self.compute_metrics_from_state(result.state, **kwargs)

    def evaluate_exact_grasp_metrics(self, act_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exact final grasp/distance error via `grasp_cost.evaluate`, bypassing
        the staged contact-mode schedule (matches original BODex's dedicated
        `grasp_convergence` evaluation, which always uses the exact mesh
        contact query regardless of which stage the optimizer ended on)."""

        act_seq = torch.cat(
            [act_seq[:, :, :3], normalize_vector(act_seq[:, :, 3:7]), act_seq[:, :, 7:]], dim=-1
        )
        _, link_pos_quat = self._fk_contact_inputs(act_seq)
        env_idx = torch.zeros((act_seq.shape[0],), device=self.device_cfg.device, dtype=torch.int32)
        # `grasp_cost.evaluate` adds its own horizon dim via `.unsqueeze(1)`;
        # `_fk_contact_inputs` already includes one, so squeeze it back out.
        dist_error, grasp_error, contact_point, contact_frame, contact_force = self.grasp_convergence_cost.evaluate(
            link_pos_quat.squeeze(1), env_idx
        )
        return {
            "dist_error": dist_error,
            "grasp_error": grasp_error,
            "contact_point": contact_point,
            "contact_frame": contact_frame,
            "contact_force": contact_force,
        }


def compute_success(
    grasp_error: torch.Tensor, dist_error: torch.Tensor, grasp_threshold: float, distance_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """BODex `get_success` semantics: all target-wrench directions must pass.

    No world-collision/bound-violation "feasible" channel is wired into this
    rollout (single-object canonical-frame scope, no scene-collision cost
    registered), so unlike original BODex this only checks the grasp-energy
    and contact-distance error terms -- consistent with the sibling
    ``bodex/`` pipeline's own diagnostics, which document the same
    "feasible_count_available: False" caveat.
    """

    grasp_error_max = grasp_error.amax(dim=-1)
    success = (grasp_error_max <= grasp_threshold) & (dist_error <= distance_threshold)
    return success, grasp_error_max


def build_metric_summary(
    *,
    success: torch.Tensor,
    grasp_error_max: torch.Tensor,
    dist_error: torch.Tensor,
    costs: torch.Tensor,
    grasp_threshold: float,
    distance_threshold: float,
) -> dict:
    seed_count = int(costs.shape[0])
    strict_success_count = int(success.sum().item())
    loose_counts = {
        "grasp_error_max_lt_0_2": int((grasp_error_max < 0.2).sum().item()),
        "grasp_error_max_lt_0_1": int((grasp_error_max < 0.1).sum().item()),
        "grasp_error_max_le_configured_threshold": int((grasp_error_max <= float(grasp_threshold)).sum().item()),
    }
    distance_count = int((dist_error <= float(distance_threshold)).sum().item())
    best_idx = int(torch.argmin(costs).item())
    return {
        "seed_count_returned": seed_count,
        "strict_success_count": strict_success_count,
        "success_definition": {
            "strict_success": "grasp_error_max <= grasp_threshold AND dist_error <= distance_threshold",
            "grasp_threshold": float(grasp_threshold),
            "distance_threshold": float(distance_threshold),
            "feasible_count_available": False,
            "feasible_note": "This cuRobo-v2 port does not register a world-collision/bound 'feasible' "
            "channel; matches the same caveat documented by the sibling original-BODex pipeline.",
        },
        "loose_diagnostic_counts": loose_counts,
        "distance_le_configured_threshold_count": distance_count,
        "best_ranked_seed": {
            "seed_index": best_idx,
            "score": float(costs[best_idx].item()),
            "strict_success": bool(success[best_idx].item()),
            "grasp_error_max": float(grasp_error_max[best_idx].item()),
            "dist_error": float(dist_error[best_idx].item()),
        },
    }


def _build_grasp_record(
    *,
    full_action: np.ndarray,
    optimized_action: np.ndarray,
    success_flag: bool,
    seed_idx: int,
    score: float,
    grasp_error_max: float,
    dist_error: float,
    successful_seed_count: int,
    metric_summary: dict,
    surface_artifact: Path,
    object_mesh_path: Path,
    surface: ObjectSurface,
    rollout: SharpaBodexRollout,
    rank: int,
) -> dict:
    full_action = np.asarray(full_action, dtype=float).copy()
    full_action[3:7] = full_action[3:7] / max(np.linalg.norm(full_action[3:7]), 1e-9)
    return {
        "ok": bool(success_flag),
        "backend": "curobo_v2_bodex_exact_single_object",
        "hand": "sharpa_wave_right",
        "rank": int(rank),
        "seed_index": int(seed_idx),
        "sequence_id": str(surface.metadata.get("sequence_id")) if surface.metadata.get("sequence_id") is not None else None,
        "object_name": str(surface.metadata.get("object_name")) if surface.metadata.get("object_name") is not None else None,
        "object_index": int(surface.metadata.get("object_index")) if surface.metadata.get("object_index") is not None else None,
        "object_mesh": str(object_mesh_path),
        "object_gravity_center": rollout.object_gravity_center.detach().cpu().numpy().reshape(-1).tolist(),
        "object_obb_length": float(rollout.object_obb_length.detach().cpu().item()),
        "surface_artifact": str(surface_artifact),
        "action": full_action.astype(float).tolist(),
        "joint_names": rollout.full_joint_order,
        "success": bool(success_flag),
        "successful_seed_count": int(successful_seed_count),
        "grasp_error_max": float(grasp_error_max),
        "dist_error": float(dist_error),
        "dist_error_final": float(dist_error),
        "metric_summary": metric_summary,
        "score": float(score),
        "optimized_action": np.asarray(optimized_action, dtype=float).tolist(),
        "optimized_joint_names": rollout.joint_names,
        "passive_joint_names": rollout.passive_joint_names,
    }


def solve_sharpa_bodex(
    surface_artifact: Path,
    out_dir: Path,
    object_mesh: Path | None = None,
    seeds: int = 20,
    top_k: int = 8,
    opt_iters: int = 500,
    seed: int = 0,
    grasp_threshold: float = DEFAULT_GRASP_THRESHOLD,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("official cuRobo v2 BODex grasp synthesis requires CUDA")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    surface = ObjectSurface.load(surface_artifact)
    object_mesh_path = find_object_mesh_from_surface(surface, object_mesh)
    pts = np.asarray(surface.points_object_frame, dtype=np.float32)
    center = pts.mean(axis=0)
    radius = float(max(np.linalg.norm(pts - center[None, :], axis=1).max() + 0.18, 0.25))
    rollout_cfg = dict(
        device_cfg=device_cfg,
        surface=surface,
        object_mesh_path=object_mesh_path,
        root_bounds_center=center,
        root_bounds_radius=radius,
        opt_iters=opt_iters,
    )
    rollouts = [SharpaBodexRollout(**rollout_cfg), SharpaBodexRollout(**rollout_cfg)]

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
    optimizer = MultiStageOptimizer([BodexNewtonOpt(opt_cfg, rollouts, use_cuda_graph=False)], rollouts)
    optimizer.update_num_problems(seeds)
    for rollout in rollouts:
        rollout.batch_size = seeds
    init_action = rollouts[0].get_initial_action(use_random=True)
    optimizer.reinitialize(init_action)
    result = optimizer.optimize(init_action)

    metrics = rollouts[0].compute_metrics_from_action(result, opt_progress=1.0)
    costs = metrics.costs_and_constraints.get_sum_cost(sum_horizon=True).detach()
    exact = rollouts[0].evaluate_exact_grasp_metrics(result)
    success, grasp_error_max = compute_success(exact["grasp_error"], exact["dist_error"], grasp_threshold, distance_threshold)
    dist_error = exact["dist_error"].detach()
    grasp_error_max = grasp_error_max.detach()

    metric_summary = build_metric_summary(
        success=success,
        grasp_error_max=grasp_error_max,
        dist_error=dist_error,
        costs=costs,
        grasp_threshold=grasp_threshold,
        distance_threshold=distance_threshold,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("grasp_*.json", "failed_grasp_*.json"):
        for path in out_dir.glob(pattern):
            path.unlink()

    successful_seed_count = int(success.sum().item())
    result_cpu = result.detach()

    def _write_records(order: list[int], prefix: str) -> list[dict]:
        entries = []
        for rank, seed_idx in enumerate(order):
            optimized_action = result_cpu[seed_idx, 0].cpu().numpy().copy()
            optimized_action[3:7] = optimized_action[3:7] / max(np.linalg.norm(optimized_action[3:7]), 1e-9)
            full_action = expand_active_action_to_full_joint_order(
                optimized_action, rollouts[0].joint_names, rollouts[0].full_joint_order, rollouts[0].full_neutral_q
            )
            record = _build_grasp_record(
                full_action=full_action,
                optimized_action=optimized_action,
                success_flag=bool(success[seed_idx].item()),
                seed_idx=int(seed_idx),
                score=float(costs[seed_idx].item()),
                grasp_error_max=float(grasp_error_max[seed_idx].item()),
                dist_error=float(dist_error[seed_idx].item()),
                successful_seed_count=successful_seed_count,
                metric_summary=metric_summary,
                surface_artifact=surface_artifact,
                object_mesh_path=object_mesh_path,
                surface=surface,
                rollout=rollouts[0],
                rank=rank,
            )
            path = out_dir / f"{prefix}_{rank:03d}.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            entries.append({"rank": int(rank), "seed_index": int(seed_idx), "score": float(costs[seed_idx].item()), "grasp_json": str(path)})
        return entries

    costs_np = costs.cpu().numpy()
    if successful_seed_count == 0:
        failed_order = list(np.argsort(costs_np)[: min(int(top_k), int(seeds))])
        top_failed_grasps = _write_records(failed_order, "failed_grasp")
        summary = {
            "ok": False,
            "backend": "curobo_v2_bodex_exact_single_object",
            "hand": "sharpa_wave_right",
            "sequence_id": str(surface.metadata.get("sequence_id")) if surface.metadata.get("sequence_id") is not None else None,
            "object_name": str(surface.metadata.get("object_name")) if surface.metadata.get("object_name") is not None else None,
            "object_mesh": str(object_mesh_path),
            "surface_artifact": str(surface_artifact),
            "success": False,
            "successful_seed_count": 0,
            "metric_summary": metric_summary,
            "seed_count": int(seeds),
            "top_k": 0,
            "top_grasps": [],
            "top_failed_grasps": top_failed_grasps,
            "failed_grasp_json": str(top_failed_grasps[0]["grasp_json"]) if top_failed_grasps else None,
            "opt_iters": int(opt_iters),
            "error": "No successful grasp seeds under the configured thresholds.",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    success_np = success.cpu().numpy()
    candidate_indices = np.flatnonzero(success_np)
    candidate_scores = costs_np[candidate_indices]
    order = list(candidate_indices[np.argsort(candidate_scores)[: min(int(top_k), len(candidate_indices))]])
    top_grasps = _write_records(order, "grasp")

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
