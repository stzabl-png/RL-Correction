"""Exact BODex staged grasp cost adapted to official cuRobo v2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import numpy as np
import torch
from scipy.stats.qmc import Halton

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.geom.transform import torch_quaternion_to_matrix
from curobo._src.types.device_cfg import DeviceCfg

from ocir.grasp_synthesis.bodex_curobo_v2.grasp_energy import GraspEnergyBase, init_grasp_energy, normalize_vector


class BodexContactChecker(Protocol):
    """BODex contact-query API required by the exact grasp cost."""

    def get_sphere_contact_pdn(
        self,
        contact_robot_sphere: torch.Tensor,
        contact_query_buffer: "ContactBuffer",
        perturb: torch.Tensor,
        env_query_idx: torch.Tensor,
    ):
        ...

    def get_mesh_contact_pdn(
        self,
        contact_robot_pose: torch.Tensor,
        perturb: torch.Tensor,
        env_query_idx: torch.Tensor,
        dist_upper_bound: torch.Tensor,
    ):
        ...


@dataclass
class ContactBuffer:
    shape: tuple[int, ...] | None = None
    pos_buf: torch.Tensor | None = None
    dist_buf: torch.Tensor | None = None
    normal_buf: torch.Tensor | None = None
    debug_posi: torch.Tensor | None = None
    debug_normal: torch.Tensor | None = None
    grad_pos_buf: torch.Tensor | None = None
    grad_dist_buf: torch.Tensor | None = None
    grad_normal_buf: torch.Tensor | None = None

    def update_buffer_shape(self, shape: Sequence[int], device_cfg: DeviceCfg) -> None:
        shape = tuple(int(v) for v in shape)
        if len(shape) != 4:
            raise ValueError(f"BODex contact buffer shape must be [batch, horizon, n_spheres, n_perturb], got {shape}")
        if self.shape == shape:
            self.pos_buf.zero_()
            self.dist_buf.zero_()
            self.normal_buf.zero_()
            self.debug_posi.zero_()
            self.debug_normal.zero_()
            self.grad_pos_buf.zero_()
            self.grad_dist_buf.zero_()
            self.grad_normal_buf.zero_()
            return

        batch, horizon, n_spheres, n_perturb = shape
        self.shape = shape
        self.pos_buf = torch.zeros((batch, horizon, n_spheres, 3), device=device_cfg.device, dtype=device_cfg.dtype)
        self.dist_buf = torch.zeros((batch, horizon, n_spheres), device=device_cfg.device, dtype=device_cfg.dtype)
        self.normal_buf = torch.zeros((batch, horizon, n_spheres, 3), device=device_cfg.device, dtype=device_cfg.dtype)
        self.debug_posi = torch.zeros((batch, horizon, n_spheres, n_perturb, 3), device=device_cfg.device, dtype=device_cfg.dtype)
        self.debug_normal = torch.zeros((batch, horizon, n_spheres, n_perturb, 3), device=device_cfg.device, dtype=device_cfg.dtype)
        self.grad_pos_buf = torch.zeros((batch, horizon, n_spheres, n_perturb, 3, 4), device=device_cfg.device, dtype=device_cfg.dtype)
        self.grad_dist_buf = torch.zeros((batch, horizon, n_spheres, n_perturb, 4), device=device_cfg.device, dtype=device_cfg.dtype)
        self.grad_normal_buf = torch.zeros((batch, horizon, n_spheres, n_perturb, 3, 4), device=device_cfg.device, dtype=device_cfg.dtype)


class HaltonGenerator:
    """Small exact-compatible Halton sampler used by BODex perturbations."""

    def __init__(
        self,
        ndims: int,
        device_cfg: DeviceCfg,
        up_bounds,
        low_bounds,
        seed: int = 123,
        store_buffer: int | None = 2000,
    ):
        self._seed = seed
        self.device_cfg = device_cfg
        self.sequencer = Halton(d=ndims, seed=seed, scramble=False)
        self.range_b = device_cfg.to_device(up_bounds) - device_cfg.to_device(low_bounds)
        self.low_bounds = device_cfg.to_device(low_bounds)
        self._sample_buffer = None
        self._store_buffer = store_buffer
        self._index_buffer = None
        if store_buffer is not None:
            self._sample_buffer = torch.tensor(
                self.sequencer.random(store_buffer),
                device=device_cfg.device,
                dtype=device_cfg.dtype,
            )
            self._int_gen = torch.Generator(device=device_cfg.device)
            self._int_gen.manual_seed(seed)

    def get_samples(self, num_samples: int, bounded: bool = False) -> torch.Tensor:
        if self._sample_buffer is not None:
            out_buffer = self._index_buffer if self._index_buffer is not None and self._index_buffer.shape[0] == num_samples else None
            index = torch.randint(
                0,
                self._sample_buffer.shape[0],
                (num_samples,),
                generator=self._int_gen,
                device=self.device_cfg.device,
                out=out_buffer,
            )
            samples = self._sample_buffer[index]
            if self._index_buffer is None:
                self._index_buffer = index
        else:
            samples = torch.tensor(self.sequencer.random(num_samples), device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        if bounded:
            samples = samples * self.range_b + self.low_bounds
        return samples


@dataclass
class BodexGraspCostConfig(BaseCostCfg):
    task_dict: dict | None = None
    ge_param: dict | None = None
    perturb_strength_bound: Sequence[float] | None = None
    contact_points_idx: torch.Tensor | None = None
    contact_mesh_idx: Sequence[int] | None = None
    world_coll_checker: BodexContactChecker | None = None
    finger_num: int | None = None
    contact_strategy: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.contact_strategy is not None and isinstance(self.contact_strategy.get("opt_progress"), list):
            self.contact_strategy["opt_progress"] = self.device_cfg.to_device(self.contact_strategy["opt_progress"])
        if self.contact_points_idx is not None:
            self.contact_points_idx = self.contact_points_idx.to(device=self.device_cfg.device, dtype=torch.long)


class BodexGraspCost(BaseCost):
    """BODex ``GraspCost`` as an official cuRobo v2 custom cost."""

    def __init__(self, config: BodexGraspCostConfig, grasp_energy: GraspEnergyBase | None = None):
        if config.world_coll_checker is None:
            raise ValueError("BodexGraspCost requires an exact BODexContactChecker")
        super().__init__(config)
        self.config: BodexGraspCostConfig
        self.contact_stage = 0
        self.contact_query_mode = None
        self._contact_buffer = ContactBuffer()
        self.grasp_energy = grasp_energy or init_grasp_energy(self.config.ge_param, self.device_cfg)
        self.TWS = self._init_tws(self.config.task_dict)
        self.count = 0
        self.update_perturb_info()

    def _init_tws(self, task: dict) -> dict:
        tws = {}
        if task["f"] is None:
            raise ValueError("BODex task_dict['f'] target wrench force is required")
        external_f = normalize_vector(self.device_cfg.to_device(task["f"]))
        axis_0, axis_1, axis_2 = self.grasp_energy.utils_1axis_to_3axes(external_f)
        tws["f"] = [external_f]
        robust_angle = task["gamma"]
        if robust_angle < 0 or robust_angle > 180:
            raise ValueError("BODex task gamma must be in [0, 180]")
        if robust_angle > 0:
            if robust_angle > 90:
                tws["f"].append(-external_f)
                robust_angle = 90 / 180 * np.pi
            else:
                robust_angle = robust_angle / 180 * np.pi
            tws["f"].append(axis_0 * math.cos(robust_angle) + axis_1 * math.sin(robust_angle))
            tws["f"].append(axis_0 * math.cos(robust_angle) - axis_1 * math.sin(robust_angle))
            tws["f"].append(axis_0 * math.cos(robust_angle) - axis_2 * math.sin(robust_angle))
            tws["f"].append(axis_0 * math.cos(robust_angle) + axis_2 * math.sin(robust_angle))
        tws["f"] = torch.stack(tws["f"], dim=0)
        if task["t"] is None:
            task["t"] = [0, 0, 0]
        tws["t"] = self.device_cfg.to_device(task["t"]).unsqueeze(0).expand_as(tws["f"])
        if tws["t"].norm() != 0:
            raise NotImplementedError("BODex nonzero task torque target is not implemented upstream")
        tws["w"] = torch.cat([tws["f"], tws["t"]], dim=-1).unsqueeze(-1)
        return tws

    def update_perturb_info(self) -> None:
        pert_dim = 3
        self.perturb_template = self.device_cfg.to_device(torch.zeros((pert_dim * 2, pert_dim)))
        for i in range(pert_dim):
            self.perturb_template[i, i] = 1
            self.perturb_template[i + pert_dim, i] = -1
        self.perturb_strength_generator = HaltonGenerator(
            self.perturb_template.shape[0],
            self.device_cfg,
            up_bounds=self.config.perturb_strength_bound[1],
            low_bounds=self.config.perturb_strength_bound[0],
            seed=123,
        )

    def reset_problem(self, gravity_center, obb_length) -> None:
        self.contact_stage = 0
        self.count = 0
        self.grasp_energy.reset(gravity_center, obb_length)

    def _get_contact_stage(self, opt_progress: float):
        strategy = self.config.contact_strategy
        max_ge_stage = strategy["max_ge_stage"]
        new_contact_stage = (strategy["opt_progress"] <= opt_progress).int().sum() - 1
        contact_query_mode = strategy["contact_query_mode"][self.contact_stage]
        contact_distance = strategy["distance"][self.contact_stage]
        switch_stage_flag = self.contact_stage < new_contact_stage
        ge_stage_flag = (self.contact_stage <= max_ge_stage) or (new_contact_stage <= max_ge_stage)
        save_qpos_flag = switch_stage_flag and strategy["save_qpos"][self.contact_stage]
        self.contact_stage = int(new_contact_stage)
        if self.contact_query_mode != contact_query_mode:
            self.contact_query_mode = contact_query_mode
            self.update_perturb_info()
        return contact_distance, switch_stage_flag, ge_stage_flag, save_qpos_flag

    @torch.no_grad()
    def evaluate(self, link_pos_quat: torch.Tensor, env_query_idx: torch.Tensor):
        self._get_contact_stage(0.0)
        contact_robot_pose = link_pos_quat[..., self.config.contact_mesh_idx, :].unsqueeze(1)
        dist_upper_bound = self.device_cfg.to_device(torch.ones_like(contact_robot_pose[..., -1]).float() * 50)
        ho_contact_point, contact_dist, contact_normal, _, _ = self.config.world_coll_checker.get_mesh_contact_pdn(
            contact_robot_pose.detach(), self.perturb_template, env_query_idx, dist_upper_bound
        )
        contact_point = ho_contact_point[..., :3]
        grasp_energy, grasp_error, contact_frame, contact_force = self.grasp_energy.forward(
            contact_point.squeeze(1), contact_normal.squeeze(1), self.TWS["w"]
        )
        dist_error = contact_dist.abs().mean(dim=-1).squeeze(1)
        return dist_error, grasp_error, contact_point, contact_frame, contact_force

    def loss_w_ge(self, pos, dist, normal, contact_distance, opt_progress):
        grasp_energy, grasp_error, contact_frame, contact_force = self.grasp_energy.forward(
            pos.squeeze(1), normal.squeeze(1), self.TWS["w"]
        )
        E_angle = self._weight[0] * grasp_energy.mean(dim=-1, keepdim=True)
        E_dist = self._weight[1] * ((dist - contact_distance) ** 2).mean(dim=-1)
        E_regu = 0 * E_dist
        dist_error = (dist - contact_distance).abs().max(dim=-1)[0]
        return E_angle, E_dist, E_regu, dist_error, grasp_error.detach(), contact_frame.detach(), contact_force.detach()

    def loss_wo_ge(self, raw_spheres, contact_distance, mesh_contacts, mesh_dist, opt_progress):
        target_sphere_center_inner = self.target_contact_position - self.target_contact_normal * (
            raw_spheres[..., -1] + contact_distance
        ).unsqueeze(-1)
        E_dist = (
            self._weight[1]
            * ((raw_spheres[..., :3] - target_sphere_center_inner) ** 2).sum(dim=-1).mean(dim=-1)
            * min(max(10 - 10 * opt_progress, 0.0), 1.0)
        )
        E_angle = 0 * E_dist
        if mesh_dist is not None:
            target_sphere_center = self.target_contact_position - self.target_contact_normal * contact_distance
            E_dist += self._weight[1] * ((mesh_contacts - target_sphere_center) ** 2).sum(dim=-1).mean(dim=-1)
            E_regu = self._weight[2] * (((mesh_dist - contact_distance) ** 2).mean(dim=-1))
            dist_error = (mesh_dist - contact_distance).abs().max(dim=-1)[0]
        else:
            target_sphere_center = self.target_contact_position - self.target_contact_normal * (
                raw_spheres[..., -1] + contact_distance
            ).unsqueeze(-1)
            E_regu = 0 * E_dist
            dist_error = (raw_spheres[..., :3] - target_sphere_center).norm(dim=-1).max(dim=-1)[0]
        return E_angle, E_dist, E_regu, dist_error

    def forward(
        self,
        robot_sphere_in: torch.Tensor,
        link_pos_quat: torch.Tensor,
        env_query_idx: torch.Tensor,
        opt_progress: float,
        **kwargs,
    ):
        contact_distance, switch_stage_flag, ge_stage_flag, save_qpos_flag = self._get_contact_stage(opt_progress)

        if self.contact_query_mode == -1:
            robot_contact_points = robot_sphere_in[..., self.config.contact_points_idx, :]
            b, h, n_points, _ = robot_contact_points.shape
            n_pert = self.perturb_template.shape[-2]
            strength_sample = self.perturb_strength_generator.get_samples(b * h * n_points, bounded=True)
            perturb = (self.perturb_template * strength_sample.unsqueeze(-1)).view(b, h, n_points, n_pert, -1)
            self._contact_buffer.update_buffer_shape([b, h, n_points, n_pert], self.device_cfg)
            raw_pos, raw_dist, raw_normal, debug_pos, debug_normal = self.config.world_coll_checker.get_sphere_contact_pdn(
                robot_contact_points, self._contact_buffer, perturb, env_query_idx
            )
            raw_pos_robot_with_grad = None
            mesh_dist_with_grad = None
        elif self.contact_query_mode == 0:
            b, h, _, _ = robot_sphere_in.shape
            n_points = len(self.config.contact_mesh_idx)
            n_pert = self.perturb_template.shape[-2]
            strength_sample = self.perturb_strength_generator.get_samples(b * h * n_points, bounded=True)
            perturb = (self.perturb_template * strength_sample.unsqueeze(-1)).view(b, h, n_points, n_pert, -1)
            query_info = link_pos_quat[..., self.config.contact_mesh_idx, :]
            robot_contact_points = robot_sphere_in[..., self.config.contact_points_idx, :]
            rot = torch_quaternion_to_matrix(query_info[..., 3:])
            cache_invalid = (
                not hasattr(self, "raw_pos_robot_frame")
                or self.raw_pos_robot_frame.shape[:-1] != query_info.shape[:-1]
            )
            if self.count % 5 == 0 or cache_invalid:
                self._contact_buffer.update_buffer_shape([b, h, n_points, n_pert], self.device_cfg)
                _, dist_upper_bound, _, _, _ = self.config.world_coll_checker.get_sphere_contact_pdn(
                    robot_contact_points.detach(), self._contact_buffer, perturb, env_query_idx
                )
                raw_pos_two, self.raw_dist, self.raw_normal, self.debug_pos, self.debug_normal = self.config.world_coll_checker.get_mesh_contact_pdn(
                    query_info.detach(), perturb, env_query_idx, dist_upper_bound
                )
                self.raw_pos_robot_frame = (rot.transpose(-1, -2) @ (raw_pos_two[..., 3:] - query_info[..., :3]).unsqueeze(-1)).squeeze(-1).detach()
                self.raw_pos = raw_pos_two[..., :3]
                self.dist_sign = (self.raw_dist > 0) * 2 - 1
            raw_pos = self.raw_pos
            raw_dist = self.raw_dist
            raw_normal = self.raw_normal
            debug_pos = self.debug_pos
            debug_normal = self.debug_normal
            raw_pos_robot_with_grad = (rot @ self.raw_pos_robot_frame.unsqueeze(-1)).squeeze(-1) + query_info[..., :3]
            mesh_dist_with_grad = self.dist_sign * (raw_pos_robot_with_grad - self.raw_pos).norm(dim=-1)
            self.count += 1
        else:
            raise NotImplementedError(f"Unsupported BODex contact query mode {self.contact_query_mode}")

        target_invalid = (
            not hasattr(self, "target_contact_position")
            or self.target_contact_position.shape != raw_pos.shape
            or self.target_contact_normal.shape != raw_normal.shape
        )
        if target_invalid:
            self.target_contact_position = raw_pos.detach().clone()
            self.target_contact_normal = raw_normal.detach().clone()

        if not ge_stage_flag:
            E_angle, E_dist, E_regu, dist_error = self.loss_wo_ge(
                robot_contact_points, contact_distance, raw_pos_robot_with_grad, mesh_dist_with_grad, opt_progress
            )
            grasp_error = contact_frame = contact_force = None
        else:
            E_angle, E_dist, E_regu, dist_error, grasp_error, contact_frame, contact_force = self.loss_w_ge(
                raw_pos, raw_dist, raw_normal, contact_distance, opt_progress
            )

        if switch_stage_flag and ge_stage_flag:
            self.target_contact_position = raw_pos.detach().clone()
            self.target_contact_normal = raw_normal.detach().clone()

        debug_pert_p = (robot_contact_points[..., :3].unsqueeze(-2) + perturb[..., :6, :3]).view(b, h, -1, 3)
        debug = {
            "input": robot_sphere_in,
            "op": debug_pert_p,
            "debug_posi": debug_pos[..., :6, :3].contiguous(),
            "debug_normal": debug_normal[..., :6, :].contiguous(),
            "normal": raw_normal,
            "save_qpos_flag": save_qpos_flag,
            "dist_error": dist_error.detach().squeeze(1),
            "grasp_error": grasp_error,
            "contact_point": raw_pos.detach().squeeze(1),
            "contact_frame": contact_frame,
            "contact_force": contact_force,
        }
        return E_angle, E_dist, E_regu, debug
