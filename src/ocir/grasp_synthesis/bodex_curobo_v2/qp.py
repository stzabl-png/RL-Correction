"""BODex batched ReLU-QP solver ported without importing third_party/BODex."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from curobo._src.types.device_cfg import DeviceCfg


class QPSolver(ABC):
    G_matrix: torch.Tensor
    l_matrix: torch.Tensor | None
    h_matrix: torch.Tensor

    def init_problem(
        self,
        G_matrix: torch.Tensor,
        l_matrix: torch.Tensor | None,
        h_matrix: torch.Tensor,
    ) -> None:
        self.G_matrix = G_matrix
        self.l_matrix = l_matrix
        self.h_matrix = h_matrix

    @property
    @abstractmethod
    def glh_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def solve(
        self,
        Q_matrix: torch.Tensor,
        semi_Q_matrix: torch.Tensor,
        solution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class BatchedReluQp(QPSolver):
    """Exact BODex default QP solver.

    This is a direct port of BODex's ``BATCHED_RELUQP``.  It solves the lower
    force-closure QP used by ``QPEnergy`` and has the same ``glh`` bound format
    as the original implementation.
    """

    @property
    def glh_type(self) -> str:
        return "glh"

    def init_problem(
        self,
        G_matrix: torch.Tensor,
        l_matrix: torch.Tensor | None,
        h_matrix: torch.Tensor,
    ) -> None:
        if l_matrix is None:
            raise ValueError("BatchedReluQp requires lower and upper bounds")
        super().init_problem(G_matrix, l_matrix, h_matrix)
        self.solver = _ReluQp(
            self.G_matrix.shape[0],
            self.G_matrix.shape[2],
            self.G_matrix.shape[1],
            DeviceCfg(device=self.G_matrix.device, dtype=self.G_matrix.dtype),
        )

    def solve(
        self,
        Q_matrix: torch.Tensor,
        semi_Q_matrix: torch.Tensor,
        solution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.solver.solve(Q_matrix, self.G_matrix, self.l_matrix, self.h_matrix)


def init_qp_solver(solver_type: str = "batch_reluqp") -> QPSolver:
    if solver_type == "batch_reluqp":
        return BatchedReluQp()
    raise NotImplementedError(
        f"QP solver '{solver_type}' is not ported. Exact OCIR BODex-curobo-v2 currently "
        "supports BODex's default batch_reluqp solver."
    )


@torch.jit.script
def _relu_step(
    input_tensor: torch.Tensor,
    W: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    rho_ind: torch.Tensor,
    help_ind: torch.Tensor,
    nx: int,
    nc: int,
) -> torch.Tensor:
    torch.bmm(W[rho_ind, help_ind], input_tensor.unsqueeze(-1), out=input_tensor.unsqueeze(-1))
    input_tensor[:, nx : nx + nc].clamp_(l, u)
    return input_tensor


@torch.jit.script
def _compute_residuals(
    H: torch.Tensor,
    A: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    lam: torch.Tensor,
    rho: torch.Tensor,
    rho_min: float,
    rho_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t1 = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
    t2 = torch.matmul(H, x.unsqueeze(-1)).squeeze(-1)
    t3 = torch.matmul(A.transpose(-1, -2), lam.unsqueeze(-1)).squeeze(-1)
    primal_res = torch.linalg.vector_norm(t1 - z, dim=-1, ord=torch.inf)
    dual_res = torch.linalg.vector_norm(t2 + t3, dim=-1, ord=torch.inf)
    numerator = torch.div(
        primal_res,
        torch.max(
            torch.linalg.vector_norm(t1, dim=-1, ord=torch.inf),
            torch.linalg.vector_norm(z, dim=-1, ord=torch.inf),
        ),
    )
    denom = torch.div(
        dual_res,
        torch.max(
            torch.linalg.vector_norm(t2, dim=-1, ord=torch.inf),
            torch.linalg.vector_norm(t3, dim=-1, ord=torch.inf),
        ).clamp(min=0),
    )
    rho = torch.clamp(rho * torch.sqrt(numerator / denom), rho_min, rho_max)
    return primal_res, dual_res, rho


class _ReluQp:
    def __init__(self, batch: int, nx: int, nc: int, device_cfg: DeviceCfg = DeviceCfg()):
        self.device_cfg = device_cfg
        self.batch = batch
        self.nx = nx
        self.nc = nc

        self.rho = 0.1
        self.rho_min = 1e-3
        self.rho_max = 1e3
        self.adaptive_rho_tolerance = 5
        self._setup_rhos(nc, batch)

        self.max_iter = 1000
        self.eps_abs = 1e-3
        self.check_interval = 25
        self.sigma = 1e-6 * self.device_cfg.to_device(torch.eye(nx)).unsqueeze(0).unsqueeze(1)
        self.W_ks = self.device_cfg.to_device(torch.zeros(len(self.rhos) + 1, batch, nx + nc + nc, nx + nc + nc))
        self.W_ks[0, ...] = torch.eye(nx, device=self.device_cfg.device, dtype=self.device_cfg.dtype).new_zeros(
            nx + nc + nc, nx + nc + nc
        )
        self.W_ks[0, ...] = torch.eye(nx + nc + nc, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        self.W_ks[1:, :, -nc:, -nc:] = torch.eye(nc, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        self.W_ks[1:, :, -nc:, nx:-nc] = -self.rhos_matrix
        self.help_arange = torch.arange(batch, device=self.device_cfg.device).long()
        self.output = self.device_cfg.to_device(torch.zeros((batch, nx + nc + nc)))

    def _setup_rhos(self, nc: int, batch: int) -> None:
        rhos = [self.rho]
        rho = self.rho / self.adaptive_rho_tolerance
        while rho >= self.rho_min:
            rhos.append(rho)
            rho = rho / self.adaptive_rho_tolerance
        rho = self.rho * self.adaptive_rho_tolerance
        while rho <= self.rho_max:
            rhos.append(rho)
            rho = rho * self.adaptive_rho_tolerance
        rhos.sort()

        self.rhos = self.device_cfg.to_device(rhos)
        self.rho_ind = torch.argmin(torch.abs(self.rhos - self.rho)).repeat(self.batch)
        eye = torch.eye(nc, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        self.rhos_matrix = self.rhos.view(-1, 1, 1, 1) * eye.unsqueeze(0).unsqueeze(1)
        self.rhos_inv_matrix = (1 / self.rhos.view(-1, 1, 1, 1)) * eye.unsqueeze(0).unsqueeze(1)

    @torch.no_grad()
    def solve(self, H: torch.Tensor, A: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        H = H.unsqueeze(0)
        A = A.unsqueeze(0)
        self.output[:] = 0

        rhosA = torch.matmul(self.rhos_matrix, A, out=self.W_ks[1:, :, -self.nc :, : self.nx])
        ArA = torch.matmul(
            A.transpose(-1, -2),
            rhosA,
            out=self.W_ks[1:, :, self.nx : 2 * self.nx, self.nx : 2 * self.nx],
        )
        if self.nx >= self.nc:
            raise ValueError("BODex ReLU-QP expects nx < nc")
        nK_inv = torch.add(
            self.sigma + H,
            ArA,
            out=self.W_ks[1:, :, self.nx : 2 * self.nx, 2 * self.nx : 3 * self.nx],
        )
        sig_ArA = torch.sub(
            self.sigma,
            ArA,
            out=self.W_ks[1:, :, self.nx : 2 * self.nx, self.nx : 2 * self.nx],
        )
        nK_ArA = torch.linalg.solve(nK_inv, sig_ArA, out=self.W_ks[1:, :, : self.nx, : self.nx])
        nK_AT = torch.linalg.solve(
            nK_inv,
            A.transpose(-1, -2),
            out=self.W_ks[1:, :, : self.nx, -self.nc :],
        )
        K_AT_rhos = torch.matmul(nK_AT, self.rhos_matrix, out=self.W_ks[1:, :, : self.nx, self.nx : -self.nc])
        K_AT_rhos.mul_(2.0)
        torch.sub(0.0, nK_AT, out=self.W_ks[1:, :, : self.nx, -self.nc :])
        torch.matmul(A, self.W_ks[1:, :, : self.nx, :], out=self.W_ks[1:, :, self.nx : -self.nc, :])
        self.W_ks[1:, :, self.nx : -self.nc, : self.nx].add_(A)
        self.W_ks[1:, :, self.nx : -self.nc, self.nx : -self.nc].sub_(self.W_ks[1:, :, -self.nc :, -self.nc :])
        self.W_ks[1:, :, self.nx : -self.nc, -self.nc :].add_(self.rhos_inv_matrix)

        rho_ind = self.rho_ind
        rho = self.rhos[rho_ind]
        x = self.output[:, : self.nx]

        for k in range(1, self.max_iter + 1):
            self.output = _relu_step(
                input_tensor=self.output,
                W=self.W_ks,
                l=l,
                u=u,
                rho_ind=(rho_ind + 1),
                help_ind=self.help_arange,
                nx=self.nx,
                nc=self.nc,
            )
            if k % self.check_interval == 0:
                x = self.output[:, : self.nx]
                z = self.output[:, self.nx : self.nx + self.nc]
                lam = self.output[:, self.nx + self.nc : self.nx + 2 * self.nc]
                primal_res, dual_res, rho = _compute_residuals(
                    H.squeeze(0),
                    A.squeeze(0),
                    x,
                    z,
                    lam,
                    rho,
                    self.rho_min,
                    self.rho_max,
                )
                rho_larger = (rho > self.rhos[rho_ind] * self.adaptive_rho_tolerance) & (rho_ind < len(self.rhos) - 1) & (rho_ind > -1)
                rho_smaller = (rho < self.rhos[rho_ind] / self.adaptive_rho_tolerance) & (rho_ind > 0)
                rho_ind = rho_ind + rho_larger.int() - rho_smaller.int()
                converge_flag = (primal_res < self.eps_abs * (self.nc**0.5)) & (
                    dual_res < self.eps_abs * (self.nx**0.5)
                )
                rho_ind = torch.where(converge_flag, -1, rho_ind)
                if torch.all(converge_flag):
                    break

        return x
