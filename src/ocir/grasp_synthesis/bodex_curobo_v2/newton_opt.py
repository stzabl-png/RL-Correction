"""BODex-faithful optimizer (momentum + per-DOF-group greedy grid line search)
built on official cuRobo v2's `GradientOptCore`.

BODex's grasp solver is configured with a YAML section named ``lbfgs:``, but
its actual optimizer (``third_party/BODex/src/curobo/opt/newton/newton_base.py``,
``NewtonOptBase``) is plain steepest descent with:

- per-DOF-group (translation / quaternion / joints) L2-normalized gradient,
- momentum: ``buf = (buf * 0.9 + gq) / 2`` (not a standard EMA),
- a "greedy" line search over the Cartesian product of per-group step
  scales (``base_scale`` x ``line_search_scale``), not Wolfe/Armijo,
- a per-outer-iteration-block learning-rate decay (``lr_decay_rate``).

Official cuRobo v2's `GradientOptCore` supplies the rollout/gradient
evaluation, line-search dispatch, and best-solution tracking infrastructure,
but its own optimizers (LBFGS, etc.) implement different step-direction and
line-search logic. This module supplies BODex's own step-direction function
and a custom `GreedyLineSearchStrategy` subclass, plumbed into
`GradientOptCore` the same way `LBFGSOpt` is -- without modifying
`third_party/curobo` at all.

It also restores BODex's `opt_progress = current_iteration / num_iters`
signal, which official cuRobo v2 does not forward into the rollout's cost
function on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, List, Optional

import torch

from ocir.grasp_synthesis.bodex_curobo_v2.backend import import_official_curobo

import_official_curobo()

from curobo._src.optim.components.gradient_opt_core import GradientOptCore
from curobo._src.optim.gradient.line_search_strategy import GreedyLineSearchStrategy, LineSearchType
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_info

from ocir.grasp_synthesis.bodex_curobo_v2.grasp_energy import normalize_vector


@dataclass
class BodexNewtonOptCfg:
    """Flat configuration mirroring `LBFGSOptCfg`'s shape for `GradientOptCore`."""

    num_iters: int = 500
    solver_type: str = "bodex_newton"
    solver_name: str = "bodex_newton"
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    store_debug: bool = False
    debug_info: Any = None
    num_problems: int = 1
    num_particles: Optional[int] = None
    sync_cuda_time: bool = True
    use_coo_sparse: bool = True
    step_scale: float = 1.0
    inner_iters: int = 50
    _num_rollout_instances: int = 2

    cost_convergence: float = 1e-7
    cost_delta_threshold: float = 0.0
    cost_relative_threshold: float = 0.0
    converged_ratio: float = 0.8
    fixed_iters: bool = True
    convergence_iteration: int = 0
    minimum_iters: Optional[int] = None
    return_best_action: bool = False  # BODex retain_best=False -> return final iterate

    #: BODex greedy-line-search per-group step scales (multiplied against
    #: base_scale, Cartesian product across the 3 DOF groups).
    bodex_line_search_scale: List[float] = field(default_factory=lambda: [0.1])
    #: placeholder, sized to num_particles in __post_init__; values unused
    #: (BodexGreedyLineSearchStrategy overrides candidate generation).
    line_search_scale: List[float] = field(default_factory=list)
    line_search_type: LineSearchType = LineSearchType.GREEDY
    use_cuda_kernel_line_search: bool = False
    fix_terminal_action: bool = False
    line_search_wolfe_c_1: float = 1e-5
    line_search_wolfe_c_2: float = 0.9

    #: BODex per-DOF-group base step scale: [translation, quaternion, joints].
    base_scale: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.1])
    translation_dim: int = 3
    quaternion_dim: int = 4
    momentum: bool = True
    normalize_grad: bool = True
    momentum_decay: float = 0.9
    lr_decay_rate: float = 0.95
    initial_step_scale: float = 0.1

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = len(self.bodex_line_search_scale) ** 3
        self.line_search_scale = [0.0] * self.num_particles
        self.line_search_type = LineSearchType(self.line_search_type)
        if self.fixed_iters:
            self.cost_delta_threshold = 0.0
            self.cost_relative_threshold = 0.0
        if self._num_rollout_instances != 2:
            raise ValueError("BodexNewtonOptCfg requires 2 rollout instances (GradientOptCore dual-rollout contract)")

    @property
    def num_rollout_instances(self) -> int:
        return self._num_rollout_instances

    @property
    def outer_iters(self) -> int:
        outer = math.ceil(self.num_iters / self.inner_iters)
        if outer <= 0:
            raise ValueError(f"outer_iters {outer} <= 0; num_iters must be a positive multiple of inner_iters")
        return outer

    def update_niters(self, niters: int) -> None:
        self.num_iters = niters


class BodexGreedyLineSearchStrategy(GreedyLineSearchStrategy):
    """Replicates BODex's per-DOF-group Cartesian-product greedy line search.

    Official cuRobo v2's `GreedyLineSearchStrategy._prepare_search_points`
    applies a single scalar per candidate to the whole action vector. BODex
    instead scales translation / quaternion / joint DOF groups independently
    (`base_scale`), tries the Cartesian product of `line_search_scale`
    values across the 3 groups, and clamps + renormalizes the quaternion
    block. Only `_prepare_search_points` is overridden; candidate selection
    (argmin cost) is inherited unmodified from `GreedyLineSearchStrategy`.
    """

    def __init__(
        self,
        base_scale: List[float],
        line_search_scale: List[float],
        translation_dim: int,
        quaternion_dim: int,
        action_bound_lows: torch.Tensor,
        action_bound_highs: torch.Tensor,
    ):
        super().__init__()
        opt_dim = action_bound_lows.shape[-1]
        joint_dim = opt_dim - translation_dim - quaternion_dim
        if joint_dim < 0:
            raise ValueError(f"action dim {opt_dim} smaller than translation+quaternion dims {translation_dim + quaternion_dim}")
        self._translation_dim = translation_dim
        self._quaternion_dim = quaternion_dim
        device = action_bound_lows.device
        dtype = action_bound_lows.dtype
        dof_scale = torch.cat(
            [
                torch.full((translation_dim,), float(base_scale[0]), device=device, dtype=dtype),
                torch.full((quaternion_dim,), float(base_scale[1]), device=device, dtype=dtype),
                torch.full((joint_dim,), float(base_scale[2]), device=device, dtype=dtype),
            ]
        )
        t0, q1 = translation_dim, translation_dim + quaternion_dim
        candidates = []
        for i in line_search_scale:
            for j in line_search_scale:
                for k in line_search_scale:
                    candidates.append(
                        torch.cat([dof_scale[:t0] * i, dof_scale[t0:q1] * j, dof_scale[q1:] * k])
                    )
        self._base_alpha_grid = torch.stack(candidates, dim=0)  # (n_particles, opt_dim)
        self._alpha_grid = self._base_alpha_grid.clone()
        self._action_bound_lows = action_bound_lows
        self._action_bound_highs = action_bound_highs

    def reset_alpha(self) -> None:
        self._alpha_grid = self._base_alpha_grid.clone()

    def decay_alpha(self, rate: float) -> None:
        self._alpha_grid = self._alpha_grid * rate

    def _prepare_search_points(self, x, step_direction, context):
        step_direction = step_direction.detach()
        x = x.detach()
        b = x.shape[0]
        opt_dim = context.action_horizon * context.action_dim
        x_flat = x.view(b, 1, opt_dim)
        step_flat = step_direction.view(b, 1, opt_dim)
        alpha = self._alpha_grid.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        x_set = x_flat + alpha * step_flat  # (b, n_particles, opt_dim)
        lows = self._action_bound_lows.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        highs = self._action_bound_highs.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        x_set = torch.clamp(x_set, lows, highs)
        t0 = self._translation_dim
        q1 = t0 + self._quaternion_dim
        x_set = torch.cat(
            [x_set[..., :t0], torch.nn.functional.normalize(x_set[..., t0:q1], dim=-1), x_set[..., q1:]],
            dim=-1,
        )
        x_set = x_set.detach().requires_grad_(True)
        return x_set, step_flat


class BodexProgressGradientOptCore(GradientOptCore):
    """`GradientOptCore` with BODex's `opt_progress` hook and greedy line search swap.

    Official cuRobo v2's optimizer core does not compute or forward a
    per-iteration progress signal into the rollout. BODex's staged contact
    cost (`grasp_cost.py`) needs `opt_progress = current_iteration/n_iters`
    on every gradient evaluation to drive its stage schedule, so this
    subclass tracks it and threads it through `evaluate_action(...,
    opt_progress=...)`.
    """

    def __init__(self, config, rollout_list, step_direction_fn, **kwargs):
        super().__init__(config, rollout_list, step_direction_fn, **kwargs)
        self.current_iteration = 0
        self._bodex_progress = 0.0

    def _opt_iters(self, iteration_state: OptimizationIterationState) -> OptimizationIterationState:
        for _ in range(self.config.inner_iters):
            self._bodex_progress = float(self.current_iteration) / float(max(self.config.num_iters, 1))
            iteration_state = self._opt_step(iteration_state)
            self.current_iteration += 1
            self._record_iteration_state(iteration_state)
        if isinstance(self._line_search_strategy, BodexGreedyLineSearchStrategy):
            self._line_search_strategy.decay_alpha(self.config.lr_decay_rate)
        return iteration_state

    def reinitialize(self, *args, **kwargs) -> None:
        self._bodex_progress = 0.0
        self.current_iteration = 0
        if isinstance(self._line_search_strategy, BodexGreedyLineSearchStrategy):
            self._line_search_strategy.reset_alpha()
        return super().reinitialize(*args, **kwargs)

    def _prepare_initial_iteration_state(self, q: torch.Tensor):
        self._bodex_progress = 0.0
        self.current_iteration = 0
        return super()._prepare_initial_iteration_state(q)

    def _compute_cost_constraint_and_gradient(self, x: torch.Tensor):
        x_n = x.detach().requires_grad_(True)
        x_in = x_n.view(
            self.config.num_problems * self.config.num_particles,
            self.action_horizon,
            self.action_dim,
        )
        trajectories = self.rollout_fn.evaluate_action(
            x_in, use_cuda_graph=False, opt_progress=getattr(self, "_bodex_progress", 0.0)
        )
        costs = trajectories.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
        cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        cost.backward(gradient=self._l_vec, retain_graph=False)
        return cost, x_n.grad.detach()

    def _compute_cost_constraint_and_gradient_initial(self, x: torch.Tensor):
        x_n = x.detach().requires_grad_(True)
        x_in = x_n.view(
            self.config.num_problems * self.config.num_particles,
            self.action_horizon,
            self.action_dim,
        )
        initial_rollout = self._rollout_list[1] if len(self._rollout_list) > 1 else self._rollout_list[0]
        trajectories = initial_rollout.evaluate_action(x_in, use_cuda_graph=False, opt_progress=0.0)
        costs = trajectories.costs_and_constraints.get_sum_cost_and_constraint(sum_horizon=True)
        cost = costs.view(self.config.num_problems, self.config.num_particles, 1)
        cost.backward(gradient=self._l_vec, retain_graph=False)
        return cost, x_n.grad.detach()

    def update_num_problems(self, num_problems: int) -> None:
        super().update_num_problems(num_problems)
        lows = self.action_bound_lows
        highs = self.action_bound_highs
        for attr in ("_line_search_strategy", "_initial_line_search_strategy"):
            if not isinstance(getattr(self, attr), BodexGreedyLineSearchStrategy):
                setattr(
                    self,
                    attr,
                    BodexGreedyLineSearchStrategy(
                        base_scale=self.config.base_scale,
                        line_search_scale=self.config.bodex_line_search_scale,
                        translation_dim=self.config.translation_dim,
                        quaternion_dim=self.config.quaternion_dim,
                        action_bound_lows=lows,
                        action_bound_highs=highs,
                    ),
                )


class BodexNewtonOpt:
    """BODex-faithful momentum/normalized-gradient optimizer on `GradientOptCore`.

    Exposes the same delegated property/method surface as official cuRobo
    v2's `LBFGSOpt`, so it is a drop-in for `MultiStageOptimizer`.
    """

    def __init__(self, config: BodexNewtonOptCfg, rollout_list: list, use_cuda_graph: bool = False):
        self._momentum_buf: Optional[torch.Tensor] = None
        self._core = BodexProgressGradientOptCore(
            config,
            rollout_list,
            step_direction_fn=self._get_step_direction_impl,
            on_reinitialize=self._on_reinitialize,
            on_initial_state=None,
            on_resize=self._on_resize,
            on_shift=None,
            use_cuda_graph=use_cuda_graph,
        )
        self._core.update_num_problems(config.num_problems)
        self._core.finish_init()

    # -- BODex-specific step direction: momentum + per-group L2 normalize --

    def _on_resize(self, num_problems: int) -> None:
        opt_dim = self._core.opt_dim
        self._momentum_buf = torch.zeros(
            (num_problems, opt_dim), device=self._core.device_cfg.device, dtype=self._core.device_cfg.dtype
        )

    def _on_reinitialize(self, mask) -> None:
        if self._momentum_buf is None:
            return
        if mask is None:
            self._momentum_buf.zero_()
        else:
            self._momentum_buf[mask] = 0.0

    @torch.no_grad()
    def _get_step_direction_impl(self, iteration_state: OptimizationIterationState) -> torch.Tensor:
        cfg = self._core.config
        grad_q = iteration_state.exploration_gradient.view(-1, self._core.opt_dim)
        gq = -1.0 * grad_q
        if cfg.normalize_grad:
            t0 = cfg.translation_dim
            q1 = t0 + cfg.quaternion_dim
            groups = [gq[..., :t0], gq[..., t0:q1], gq[..., q1:]]
            gq = torch.cat([normalize_vector(g) for g in groups], dim=-1)
        if cfg.momentum:
            # BODex: best_grad_q = (best_grad_q * 0.9 + gq) / 2 (not a standard EMA).
            self._momentum_buf = (self._momentum_buf * cfg.momentum_decay + gq) / 2.0
            direction = self._momentum_buf
        else:
            direction = gq
        return direction.view(-1, self._core.action_horizon, self._core.action_dim)

    # -- Protocol: delegate to core (mirrors official cuRobo v2's LBFGSOpt) --

    @property
    def config(self):
        return self._core.config

    @property
    def device_cfg(self):
        return self._core.device_cfg

    @property
    def opt_dt(self):
        return self._core.opt_dt

    @opt_dt.setter
    def opt_dt(self, value):
        self._core.opt_dt = value

    @property
    def use_cuda_graph(self):
        return self._core.use_cuda_graph

    @property
    def enabled(self):
        return self._core.enabled

    def enable(self):
        self._core.enable()

    def disable(self):
        self._core.disable()

    @property
    def action_horizon(self):
        return self._core.action_horizon

    @property
    def action_dim(self):
        return self._core.action_dim

    @property
    def opt_dim(self):
        return self._core.opt_dim

    @property
    def outer_iters(self):
        return self._core.outer_iters

    @property
    def horizon(self):
        return self._core.horizon

    @property
    def action_bound_lows(self):
        return self._core.action_bound_lows

    @property
    def action_bound_highs(self):
        return self._core.action_bound_highs

    @property
    def action_step_max(self):
        return self._core.action_step_max

    @property
    def action_horizon_step_max(self):
        return self._core.action_horizon_step_max

    @property
    def action_horizon_bounds_lows(self):
        return self._core.action_horizon_bounds_lows

    @property
    def action_horizon_bounds_highs(self):
        return self._core.action_horizon_bounds_highs

    @property
    def solve_time(self):
        return self._core.solve_time

    @property
    def solver_names(self):
        return self._core.solver_names

    @property
    def rollout_fn(self):
        return self._core.rollout_fn

    @property
    def _rollout_list(self):
        return self._core._rollout_list

    @property
    def _graphable_methods(self):
        return self._core._graphable_methods

    @property
    def _executors(self):
        return self._core._executors

    @_executors.setter
    def _executors(self, value):
        self._core._executors = value

    def optimize(self, seed_action):
        return self._core.optimize(seed_action)

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        return self._core.reinitialize(action, mask, clear_optimizer_state, reset_num_iters)

    def shift(self, shift_steps=0):
        return self._core.shift(shift_steps)

    def _shift(self, shift_steps=0):
        return self._core._shift(shift_steps)

    def update_num_problems(self, num_problems):
        return self._core.update_num_problems(num_problems)

    def update_rollout_params(self, goal):
        return self._core.update_rollout_params(goal)

    def update_goal_dt(self, goal):
        return self._core.update_goal_dt(goal)

    def get_all_rollout_instances(self):
        return self._core.get_all_rollout_instances()

    def compute_metrics(self, action):
        return self._core.compute_metrics(action)

    def reset_shape(self):
        return self._core.reset_shape()

    def reset_seed(self):
        return self._core.reset_seed()

    def reset_cuda_graph(self):
        return self._core.reset_cuda_graph()

    def get_recorded_trace(self):
        return self._core.get_recorded_trace()

    def update_solver_params(self, solver_params):
        return self._core.update_solver_params(solver_params)

    def update_niters(self, niters):
        return self._core.update_niters(niters)

    def debug_dump(self, file_path: str = ""):
        return self._core.debug_dump(file_path)
