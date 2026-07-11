"""Anchored-BODex rollout: cuRobo v2 rollout wiring with human guidance.

An adapted copy of ``bodex_curobo_v2.solver.SharpaBodexRollout`` (that module
is a frozen reference port and is never modified) that:

- builds its contact spheres / kinematics from a per-sequence **subset** of
  the 11 Sharpa contact points (selected from the human demo's contact roles)
  and passes regenerated pressure constraints into the unmodified
  ``BodexGraspCost`` / ``QPEnergy`` (both are shape-driven);
- takes its initial actions from an :class:`AnchoredSeedGenerator` (retargeted
  human contact-frame poses) instead of object-surface sampling;
- adds two guidance costs on top of the exact BODex staged cost:

  - ``human_pose_prior``: per-seed deviation from that seed's own un-relaxed
    retargeted anchor (position, sign-invariant quaternion, joints), annealed
    to zero before the stage-0 -> 1 contact switch;
  - ``affordance_attraction``: chamfer distance from the FK contact-sphere
    centers to the high-affordance object points. Pulling the sphere centers
    (differentiable through FK in every stage) instead of the object-side
    contact points sidesteps the stage-0-only custom-backward gradient path;
    since the BODex distance term pins spheres to the surface anyway, the two
    coincide at convergence. Decayed to zero across stages 1 -> 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ocir.grasp_synthesis.bodex_curobo_v2.backend import import_official_curobo

import_official_curobo()

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult

from ocir.grasp_synthesis.assets import load_sharpa_wave_right
from ocir.grasp_synthesis.anchored_bodex.guidance import GuidanceWeights
from ocir.grasp_synthesis.anchored_bodex.seed_generator import AnchoredSeedGenerator
from ocir.grasp_synthesis.clearance import ClearanceChecker
from ocir.grasp_synthesis.bodex_curobo_v2.contact_world import SingleObjectContactWorld
from ocir.grasp_synthesis.bodex_curobo_v2.grasp_cost import BodexGraspCost, BodexGraspCostConfig
from ocir.grasp_synthesis.bodex_curobo_v2.grasp_energy import normalize_vector
from ocir.grasp_synthesis.bodex_curobo_v2.solver import (
    DEFAULT_CONTACT_STRATEGY,
    DEFAULT_GE_PARAM,
    DEFAULT_PERTURB_STRENGTH_BOUND,
    DEFAULT_TASK_DICT,
    DEFAULT_WEIGHT,
    _load_yaml,
    _sphere_lookup,
    batched_transform_points,
    compose_root_pose,
    transform_points,
)
from ocir.grasp_synthesis.object_surface import ObjectSurface


@dataclass
class AnchoredBodexRollout:
    device_cfg: DeviceCfg
    surface: ObjectSurface
    object_mesh_path: Path
    root_bounds_center: np.ndarray
    root_bounds_radius: float
    contact_points: list[str]                 # active subset, canonical order
    pressure_constraints: list[list]
    seed_generator: AnchoredSeedGenerator     # shared across rollout instances
    afford_points: torch.Tensor | None        # (M, 3) high-affordance points
    weights: GuidanceWeights = field(default_factory=GuidanceWeights)
    opt_iters: int = 500

    def __post_init__(self):
        self.asset = load_sharpa_wave_right()
        self.full_joint_order = list(self.asset.config["joint_order"])
        self.full_neutral_q = np.zeros((len(self.full_joint_order),), dtype=np.float64)

        collision_config = _load_yaml(self.asset.collision_spheres_path)
        contact_link_order: list[str] = []
        local_centers = []
        local_radii = []
        for entry in self.contact_points:
            link_name, sphere_idx = entry.split("/")
            center, radius = _sphere_lookup(collision_config, link_name, int(sphere_idx))
            if link_name not in contact_link_order:
                contact_link_order.append(link_name)
            local_centers.append(center)
            local_radii.append(radius)
        self.contact_link_names = tuple(contact_link_order)
        self.contact_sphere_links = [entry.split("/")[0] for entry in self.contact_points]
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

        # The seed generator's fitter may expose a different active-joint set
        # (its kinematics is built from the fingertip/pad keypoint links);
        # remap its [pos, quat, q] layout into this rollout's joint order.
        fitter_names = list(self.seed_generator.fitter.joint_names)
        missing = [name for name in self.joint_names if name not in fitter_names]
        if missing:
            raise ValueError(f"seed generator does not provide joints {missing}")
        self._seed_joint_map = torch.tensor(
            [fitter_names.index(name) for name in self.joint_names],
            device=self.device_cfg.device,
            dtype=torch.long,
        )

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

        contact_idx = torch.arange(len(self.contact_points), device=self.device_cfg.device, dtype=torch.long)
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
                "pressure_constraints": [list(c) for c in self.pressure_constraints],
                "obj_gravity_center": center_obj,
                "obj_obb_length": extent,
            },
            perturb_strength_bound=DEFAULT_PERTURB_STRENGTH_BOUND,
            contact_points_idx=contact_idx,
            contact_mesh_idx=list(range(len(self.contact_link_names))),
            world_coll_checker=self.contact_world,
            finger_num=len(self.contact_points),
            contact_strategy=dict(DEFAULT_CONTACT_STRATEGY),
        )
        self.grasp_cost = BodexGraspCost(cfg)
        # Fresh instance for exact final metrics (same rationale as the
        # reference port: the in-loop cost re-solves its QP only every
        # solve_interval calls, so its solution can be stale for evaluation).
        self.grasp_convergence_cost = BodexGraspCost(cfg)

        # Affordance target set for the attraction cost (palm excluded: the
        # palm's placement is a consequence of the grasp, not an intent
        # signal from the demo).
        palm_link = self.asset.config["base_link"]
        self._afford_mask = torch.tensor(
            [not cp.startswith(f"{palm_link}/") for cp in self.contact_points],
            device=self.device_cfg.device,
            dtype=torch.bool,
        )

        # All-sphere FK for the non-penetration penalty: the contact-subset
        # kinematics above only covers the contact links, but penetration can
        # happen anywhere on the hand (phalanges, palm bulk). ClearanceChecker
        # brings its own all-sphere-link Kinematics; its sphere positions are
        # differentiable through FK, and get_sphere_contact_pdn's backward
        # carries the exact SDF gradient for the distance term.
        self._pene_checker = ClearanceChecker(self.asset, self.device_cfg) if self.weights.w_pene > 0.0 else None
        self._pene_expand_idx = torch.tensor(
            [self.full_joint_order.index(name) for name in self.joint_names],
            device=self.device_cfg.device,
            dtype=torch.long,
        )
        self._pene_perturb_placeholder = self.device_cfg.to_device(np.zeros((1, 1, 3), dtype=np.float32))

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

    def _remap_seed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """[pos, quat, q(fitter order)] -> [pos, quat, q(this rollout's order)]."""

        return torch.cat([actions[:, :7], actions[:, 7:][:, self._seed_joint_map]], dim=-1)

    def get_initial_action(self, use_random: bool = True, use_zero: bool = False, **kwargs) -> torch.Tensor:
        n = self.batch_size or 1
        if use_random:
            action = self._remap_seed_actions(self.seed_generator.get_samples(n))
        else:
            center = (self.action_bound_lows + self.action_bound_highs) * 0.5
            action = center.unsqueeze(0).repeat(n, 1)
            action[:, 3:7] = self.device_cfg.to_device([1, 0, 0, 0]).view(1, 4)
        return action.view(n, self.action_horizon, self.action_dim)

    def _ref_actions_for_batch(self, batch: int) -> torch.Tensor:
        """Per-seed anchors broadcast to the rollout batch (the optimizer
        evaluates num_problems x num_particles actions, problem index slowest)."""

        ref = self._remap_seed_actions(self.seed_generator.ref_actions)
        n = ref.shape[0]
        if batch == n:
            return ref
        if batch % n == 0:
            return ref.repeat_interleave(batch // n, dim=0)
        raise ValueError(f"rollout batch {batch} is not a multiple of seed count {n}")

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
        )
        sphere_link_quat = torch.stack(
            [kin_state.tool_poses.get_link_pose(link).quaternion for link in self.contact_sphere_links], dim=1
        )
        local_offsets = self.contact_spheres_local[:, :3].unsqueeze(0).expand(b, n_contacts, 3)
        radii = self.contact_spheres_local[:, 3].view(1, n_contacts, 1).expand(b, n_contacts, 1)
        sphere_in_link_frame_world = batched_transform_points(sphere_link_pos, sphere_link_quat, local_offsets)
        sphere_world_pos = transform_points(root_pos, root_quat, sphere_in_link_frame_world)
        robot_spheres = torch.cat([sphere_world_pos, radii], dim=-1).unsqueeze(1)
        return robot_spheres, world_link_pose.unsqueeze(1)

    def _guidance_costs(
        self, flat_action: torch.Tensor, robot_spheres: torch.Tensor, opt_progress: float
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        b = flat_action.shape[0]

        w_pose = self.weights.pose_prior_weight(opt_progress)
        if w_pose > 0.0:
            ref = self._ref_actions_for_batch(b)
            wp, wr, wj = self.weights.w_pose
            e_pos = ((flat_action[:, :3] - ref[:, :3]) ** 2).sum(dim=-1)
            e_rot = 1.0 - (flat_action[:, 3:7] * ref[:, 3:7]).sum(dim=-1) ** 2
            e_joint = ((flat_action[:, 7:] - ref[:, 7:]) ** 2).mean(dim=-1)
            out["human_pose_prior"] = w_pose * (wp * e_pos + wr * e_rot + wj * e_joint)

        w_afford = self.weights.afford_weight(opt_progress) if self.afford_points is not None else 0.0
        if w_afford > 0.0:
            centers = robot_spheres[:, 0, self._afford_mask, :3]  # (b, n_masked, 3)
            d2 = torch.cdist(centers, self.afford_points.view(1, -1, 3).expand(b, -1, -1)).min(dim=-1).values ** 2
            out["affordance_attraction"] = self.weights.w_afford * w_afford * d2.mean(dim=-1)

        if self._pene_checker is not None:
            out["penetration_penalty"] = self.weights.w_pene * self._penetration_cost(flat_action)
        return out

    def _penetration_cost(self, flat_action: torch.Tensor) -> torch.Tensor:
        """Asymmetric non-penetration energy over ALL hand collision spheres:
        sum(relu(-signed_distance)^2), meters^2. Zero whenever the hand is
        clear of the object, so it never competes with the staged contact
        schedule -- it only forbids the overshoot INTO the mesh that the
        symmetric distance term is indifferent to."""

        b = flat_action.shape[0]
        n_full = len(self.full_joint_order)
        full_q = torch.zeros((b, n_full), device=flat_action.device, dtype=flat_action.dtype)
        full_q.index_copy_(1, self._pene_expand_idx, flat_action[:, 7:])
        full_actions = torch.cat([flat_action[:, :7], full_q], dim=-1)

        centers = self._pene_checker.sphere_world_positions(full_actions)  # (b, N, 3)
        n_spheres = centers.shape[1]
        radii = self._pene_checker.radii.view(1, n_spheres, 1).expand(b, n_spheres, 1)
        spheres = torch.cat([centers, radii], dim=-1)
        _, distance, _, _, _ = self.contact_world.get_sphere_contact_pdn(
            spheres, None, self._pene_perturb_placeholder, env_query_idx=None
        )
        penetration = torch.relu(-distance)  # (b, N) depth in meters
        return (penetration**2).sum(dim=-1)

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
        for name, value in self._guidance_costs(action[:, 0, :], robot_spheres, opt_progress).items():
            costs.costs.add(value.view(-1, 1, 1), name)
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
        """Exact final grasp/distance error via the dedicated convergence cost
        (always the exact mesh contact query, independent of stage schedule)."""

        act_seq = torch.cat(
            [act_seq[:, :, :3], normalize_vector(act_seq[:, :, 3:7]), act_seq[:, :, 7:]], dim=-1
        )
        _, link_pos_quat = self._fk_contact_inputs(act_seq)
        env_idx = torch.zeros((act_seq.shape[0],), device=self.device_cfg.device, dtype=torch.int32)
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
