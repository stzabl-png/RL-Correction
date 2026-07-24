"""Exact BODex QP grasp energy ported to official cuRobo v2 types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch

from curobo._src.types.device_cfg import DeviceCfg

from ocir.grasp_synthesis.bodex_curobo_v2.qp import init_qp_solver


def normalize_vector(value: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return value / torch.clamp(torch.linalg.norm(value, dim=-1, keepdim=True), min=eps)


class GraspEnergyBase(ABC):
    """BODex grasp-matrix construction and target wrench bookkeeping."""

    def __init__(
        self,
        miu_coef: Sequence[float],
        obj_gravity_center,
        obj_obb_length,
        device_cfg: DeviceCfg,
        enable_density: bool,
    ):
        self.miu_coef = list(miu_coef)
        if not (0 < self.miu_coef[0] <= 1):
            raise ValueError("BODex friction coefficient miu_coef[0] must be in (0, 1]")
        self.device_cfg = device_cfg
        self.obj_gravity_center: torch.Tensor | None = None
        self.obj_obb_length: torch.Tensor | None = None
        self.reset(obj_gravity_center, obj_obb_length)
        self.rot_base1 = self.device_cfg.to_device([0, 1, 0])
        self.rot_base2 = self.device_cfg.to_device([0, 0, 1])
        self.enable_density = enable_density
        self.grasp_batch = 0
        self.force_batch = 0

    def utils_1axis_to_3axes(self, axis_0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tmp_rot_base1 = self.rot_base1.view([1] * (len(axis_0.shape) - 1) + [3])
        tmp_rot_base2 = self.rot_base2.view([1] * (len(axis_0.shape) - 1) + [3])

        proj_xy = (axis_0 * tmp_rot_base1).sum(dim=-1, keepdim=True).abs()
        axis_1 = torch.where(proj_xy > 0.99, tmp_rot_base2, tmp_rot_base1)
        axis_1 = normalize_vector(axis_1 - (axis_1 * axis_0).sum(dim=-1, keepdim=True) * axis_0).detach()
        axis_1 = normalize_vector(axis_1 - (axis_1 * axis_0).sum(dim=-1, keepdim=True) * axis_0)
        axis_2 = torch.cross(axis_0, axis_1, dim=-1)
        return axis_0, axis_1, axis_2

    def construct_grasp_matrix(self, pos: torch.Tensor, normal: torch.Tensor):
        axis_0, axis_1, axis_2 = self.utils_1axis_to_3axes(normal)
        env_num = self.obj_gravity_center.shape[0]
        batch_num = pos.shape[0] // env_num
        relative_pos = (
            (
                pos.view(env_num, batch_num, -1, 3)
                - self.obj_gravity_center.view(-1, 1, 1, 3)
            )
            / self.obj_obb_length.view(-1, 1, 1, 1)
        ).view(env_num * batch_num, -1, 3)
        w0 = torch.cat([axis_0, torch.cross(relative_pos, axis_0, dim=-1)], dim=-1)
        w1 = torch.cat([axis_1, torch.cross(relative_pos, axis_1, dim=-1)], dim=-1)
        w2 = torch.cat([axis_2, torch.cross(relative_pos, axis_2, dim=-1)], dim=-1)
        if self.miu_coef[1] > 0:
            w3 = torch.cat([axis_0 * 0.0, axis_0 * self.miu_coef[1]], dim=-1)
            grasp_matrix = torch.stack([w0, w1, w2, w3], dim=-1)
        else:
            grasp_matrix = torch.stack([w0, w1, w2], dim=-1)
        contact_frame = torch.stack([axis_0, axis_1, axis_2], dim=-1)
        contact_force = self.device_cfg.to_device([1.0, 0.0, 0.0]).view(1, 1, -1).expand_as(pos)
        return grasp_matrix, contact_frame, contact_force

    def estimate_density(self, normal: torch.Tensor) -> torch.Tensor:
        cos_theta = (normal.unsqueeze(-2) * normal.unsqueeze(-3)).sum(dim=-1)
        density = 1 / torch.clamp(torch.clamp(cos_theta, min=0).sum(dim=-1), min=1e-4)
        return density.detach()

    @abstractmethod
    def forward(self, pos: torch.Tensor, normal: torch.Tensor, test_wrenches: torch.Tensor):
        raise NotImplementedError

    def reset(self, gravity_center, obb_length) -> None:
        if gravity_center is not None:
            gravity_center_tensor = self.device_cfg.to_device(gravity_center)
            if self.obj_gravity_center is not None and gravity_center_tensor.shape == self.obj_gravity_center.shape:
                self.obj_gravity_center.copy_(gravity_center_tensor)
            else:
                self.obj_gravity_center = gravity_center_tensor
        if obb_length is not None:
            obb_length_tensor = self.device_cfg.to_device(obb_length)
            if self.obj_obb_length is not None and obb_length_tensor.shape == self.obj_obb_length.shape:
                self.obj_obb_length.copy_(obb_length_tensor)
            else:
                self.obj_obb_length = obb_length_tensor


class QPEnergy(GraspEnergyBase):
    """BODex lower-level force-closure QP energy."""

    def __init__(
        self,
        k_lower: float,
        pressure_constraints: Sequence[tuple[Sequence[int], float]],
        solver_type: str,
        miu_coef: Sequence[float],
        obj_gravity_center,
        obj_obb_length,
        device_cfg: DeviceCfg,
        enable_density: bool,
        solve_interval: int,
        **kwargs,
    ):
        super().__init__(miu_coef, obj_gravity_center, obj_obb_length, device_cfg, enable_density)
        self.num_friction_approx = 4 if self.miu_coef[1] > 0 else 8
        self.k_lower = k_lower
        self.pressure_constraints = list(pressure_constraints)
        self.qpsolver = init_qp_solver(solver_type)
        self.count = 0
        self.solve_interval = solve_interval
        self.qp_size = None

    def _init_lcqp_lu(self, batch: int, num_points: int):
        num_f_strength = num_points * 3
        row_count = (self.num_friction_approx + 1) * num_points + 1 + len(self.pressure_constraints)
        G_matrix = self.device_cfg.to_device(torch.zeros((batch, row_count, num_f_strength + 1)))
        l_matrix = self.device_cfg.to_device(torch.zeros((batch, row_count)) - torch.inf)
        h_matrix = self.device_cfg.to_device(torch.zeros((batch, row_count)))

        A_end = self.num_friction_approx * num_points
        pressure_ind = range(0, num_f_strength, 3)
        friction1_ind = range(1, num_f_strength, 3)
        friction2_ind = range(2, num_f_strength, 3)
        select_ind = range(0, num_points)
        A_tmp = G_matrix[:, :A_end, :num_f_strength].view(-1, self.num_friction_approx, num_points, num_f_strength)
        A_tmp[..., select_ind, pressure_ind] = -self.miu_coef[0]
        angles = self.device_cfg.to_device(torch.arange(self.num_friction_approx) * 2 * np.pi / self.num_friction_approx).unsqueeze(-1)
        A_tmp[..., select_ind, friction1_ind] = torch.sin(angles)
        A_tmp[..., select_ind, friction2_ind] = torch.cos(angles)

        A_end2 = A_end + num_points
        G_matrix[:, range(A_end, A_end2), pressure_ind] = 1
        l_matrix[:, A_end:A_end2] = 0.01
        h_matrix[:, A_end:A_end2] = 1

        G_matrix[:, A_end2, -1] = -1
        h_matrix[:, A_end2] = -self.k_lower - 0.01
        l_matrix[:, A_end2] = -self.k_lower + 0.01

        for i, constraint in enumerate(self.pressure_constraints):
            press_lst = [pressure_ind[k] for k in constraint[0]]
            G_matrix[:, A_end2 + 1 + i, press_lst] = -1
            h_matrix[:, A_end2 + 1 + i] = -constraint[1]
        return G_matrix, l_matrix, h_matrix

    def _init_lcqp_lu_soft(self, batch: int, num_points: int):
        num_f_strength = num_points * 4
        row_count = (2 * self.num_friction_approx + 1) * num_points + 1 + len(self.pressure_constraints)
        G_matrix = self.device_cfg.to_device(torch.zeros((batch, row_count, num_f_strength + 1)))
        l_matrix = self.device_cfg.to_device(torch.zeros((batch, row_count)) - torch.inf)
        h_matrix = self.device_cfg.to_device(torch.zeros((batch, row_count)))

        A_end = 2 * self.num_friction_approx * num_points
        pressure_ind = range(0, num_f_strength, 4)
        friction1_ind = range(1, num_f_strength, 4)
        friction2_ind = range(2, num_f_strength, 4)
        friction3_ind = range(3, num_f_strength, 4)
        select_ind = range(0, num_points)
        A_tmp = G_matrix[:, :A_end, :num_f_strength].view(-1, self.num_friction_approx, 2, num_points, num_f_strength)
        A_tmp[..., select_ind, pressure_ind] = -self.miu_coef[0]
        angles = self.device_cfg.to_device(torch.arange(self.num_friction_approx) * 2 * np.pi / self.num_friction_approx).view(-1, 1, 1)
        A_tmp[..., :, select_ind, friction1_ind] = torch.sin(angles)
        A_tmp[..., :, select_ind, friction2_ind] = torch.cos(angles)
        A_tmp[..., 0, select_ind, friction3_ind] = -1.0
        A_tmp[..., 1, select_ind, friction3_ind] = 1.0

        A_end2 = A_end + num_points
        G_matrix[:, range(A_end, A_end2), pressure_ind] = 1
        l_matrix[:, A_end:A_end2] = 0.001
        h_matrix[:, A_end:A_end2] = 1

        G_matrix[:, A_end2, -1] = -1
        h_matrix[:, A_end2] = -self.k_lower

        for i, constraint in enumerate(self.pressure_constraints):
            press_lst = [pressure_ind[k] for k in constraint[0]]
            G_matrix[:, A_end2 + 1 + i, press_lst] = -1
            h_matrix[:, A_end2 + 1 + i] = -constraint[1]
        return G_matrix, l_matrix, h_matrix

    def init_lcqp(self, batch_num: int, point_num: int, wrench_num: int) -> None:
        prob_size = torch.Size((batch_num, point_num, wrench_num))
        if self.qp_size == prob_size:
            return
        self.qp_size = prob_size
        if self.qpsolver.glh_type != "glh":
            raise NotImplementedError("OCIR exact port currently supports BODex glh QP format")
        if self.miu_coef[1] > 0:
            G_matrix, l_matrix, h_matrix = self._init_lcqp_lu_soft(batch_num * wrench_num, point_num)
        else:
            G_matrix, l_matrix, h_matrix = self._init_lcqp_lu(batch_num * wrench_num, point_num)
        self.qpsolver.init_problem(G_matrix, l_matrix, h_matrix)

    def construct_Q_matrix(self, grasp_matrix: torch.Tensor, target_wrenches: torch.Tensor):
        batch_num, point_num = grasp_matrix.shape[:2]
        grasp_matrix = grasp_matrix.transpose(-3, -2).reshape(batch_num, 6, -1)
        grasp_matrix = grasp_matrix.repeat_interleave(target_wrenches.shape[0], dim=0)
        repeated_target_wrench = target_wrenches.repeat(batch_num, 1, 1)
        grasp_matrix_with_target = torch.cat([grasp_matrix, -repeated_target_wrench], dim=-1)
        Q_matrix = grasp_matrix_with_target.transpose(-2, -1) @ grasp_matrix_with_target
        Q_matrix2 = Q_matrix.clone()
        press_ind = range(0, point_num * (4 if self.miu_coef[1] > 0 else 3), 4 if self.miu_coef[1] > 0 else 3)
        Q_matrix[:, press_ind, press_ind] += 0.01
        return Q_matrix, Q_matrix2, grasp_matrix_with_target

    def forward(self, pos: torch.Tensor, normal: torch.Tensor, target_wrenches: torch.Tensor):
        batch, n_points = pos.shape[:-1]
        self.init_lcqp(batch, n_points, target_wrenches.shape[0])
        grasp_matrix, contact_frame, _ = self.construct_grasp_matrix(pos, normal)
        Q_matrix, Q_matrix2, semi_Q_matrix = self.construct_Q_matrix(grasp_matrix, target_wrenches)
        if self.count % self.solve_interval == 0:
            self.solution = self.qpsolver.solve(Q_matrix, semi_Q_matrix)
        self.count += 1
        grasp_energy = (self.solution.unsqueeze(-2) @ Q_matrix2 @ self.solution.unsqueeze(-1)).view(batch, -1).clamp(min=0.0)
        grasp_error = grasp_energy**0.5
        contact_force = self.solution[..., :-1].reshape(batch, target_wrenches.shape[0], n_points, -1)
        return grasp_energy, grasp_error, contact_frame, contact_force

    def reset(self, gravity_center, obb_length) -> None:
        super().reset(gravity_center, obb_length)
        self.count = 0


def init_grasp_energy(params: dict, device_cfg: DeviceCfg) -> GraspEnergyBase:
    kind = params.get("type", "qp")
    if kind != "qp":
        raise NotImplementedError(f"Only BODex QP grasp energy is ported, got {kind!r}")
    kwargs = {key: value for key, value in params.items() if key != "type"}
    return QPEnergy(device_cfg=device_cfg, **kwargs)
