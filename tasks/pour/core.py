"""Dependency-light task logic for the bimanual pouring environment.

The Isaac environment owns rigid bodies and robot tensors.  This module owns
the task protocol: action layout, phases, analytic liquid transfer, reward
ablation, success criteria, and deterministic failure precedence.  Keeping
these rules here makes them testable without launching Kit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from tasks.pour.enums import (
    FAILURE_NAMES,
    AblationMode,
    CurriculumStage,
    FailureCode,
    PourPhase,
)


SIDE_ACTION_DIM = 13
ACTION_DIM = 2 * SIDE_ACTION_DIM


def split_bimanual_action(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return left and right 13-D commands from ``[..., 26]`` actions."""

    if actions.ndim < 1 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"bimanual action must end in {ACTION_DIM} values; got {tuple(actions.shape)}"
        )
    return actions[..., :SIDE_ACTION_DIM], actions[..., SIDE_ACTION_DIM:]


@dataclass(frozen=True)
class LiquidProxyConfig:
    initial_mass: float = 1.0
    flow_rate_per_second: float = 0.55
    onset_tilt_deg: float = 60.0
    full_flow_tilt_deg: float = 85.0
    cup_radius_m: float = 0.045
    min_mouth_height_m: float = 0.01
    max_mouth_height_m: float = 0.25
    stream_margin_m: float = 0.005


@dataclass
class LiquidState:
    bottle: torch.Tensor
    cup: torch.Tensor
    spill: torch.Tensor

    @classmethod
    def full(
        cls, count: int, *, device: torch.device | str, mass: float = 1.0
    ) -> "LiquidState":
        bottle = torch.full((count,), float(mass), device=device)
        return cls(bottle=bottle, cup=torch.zeros_like(bottle), spill=torch.zeros_like(bottle))

    def total(self) -> torch.Tensor:
        return self.bottle + self.cup + self.spill


@dataclass
class LiquidStep:
    state: LiquidState
    transferred: torch.Tensor
    spilled: torch.Tensor
    tilt_rad: torch.Tensor
    radial_error_m: torch.Tensor
    mouth_height_m: torch.Tensor
    flow_gate: torch.Tensor


def step_liquid_proxy(
    state: LiquidState,
    *,
    bottle_up_axis_w: torch.Tensor,
    bottle_mouth_pos_w: torch.Tensor,
    cup_center_pos_w: torch.Tensor,
    cup_up_axis_w: torch.Tensor,
    in_pour_phase: torch.Tensor,
    dt: float,
    cfg: LiquidProxyConfig,
) -> LiquidStep:
    """Advance the conservative analytic liquid model by one control step.

    The stream is projected along the cup normal.  Mass can leave the bottle
    only in POUR, above the opening, and after the video-derived tilt onset.
    Alignment decides which fraction reaches the cup; the rest is spill.
    """

    eps = 1.0e-8
    cup_up = torch.nn.functional.normalize(cup_up_axis_w, dim=-1, eps=eps)
    bottle_up = torch.nn.functional.normalize(bottle_up_axis_w, dim=-1, eps=eps)
    world_up = torch.zeros_like(bottle_up)
    world_up[..., 2] = 1.0
    tilt = torch.acos((bottle_up * world_up).sum(dim=-1).clamp(-1.0, 1.0))

    rel = bottle_mouth_pos_w - cup_center_pos_w
    height = (rel * cup_up).sum(dim=-1)
    radial_vec = rel - height.unsqueeze(-1) * cup_up
    radial = radial_vec.norm(dim=-1)

    onset = torch.deg2rad(torch.as_tensor(cfg.onset_tilt_deg, device=tilt.device))
    full = torch.deg2rad(torch.as_tensor(cfg.full_flow_tilt_deg, device=tilt.device))
    tilt_fraction = ((tilt - onset) / (full - onset).clamp(min=eps)).clamp(0.0, 1.0)
    height_ok = (height >= cfg.min_mouth_height_m) & (height <= cfg.max_mouth_height_m)
    gate = in_pour_phase.bool() & height_ok & (tilt_fraction > 0.0)

    released = torch.minimum(
        state.bottle,
        cfg.flow_rate_per_second * float(dt) * tilt_fraction * gate.float(),
    )
    capture_radius = max(cfg.cup_radius_m + cfg.stream_margin_m, eps)
    capture_fraction = (1.0 - radial / capture_radius).clamp(0.0, 1.0)
    transferred = released * capture_fraction
    spilled = released - transferred
    next_state = LiquidState(
        bottle=(state.bottle - released).clamp(min=0.0),
        cup=state.cup + transferred,
        spill=state.spill + spilled,
    )
    return LiquidStep(
        state=next_state,
        transferred=transferred,
        spilled=spilled,
        tilt_rad=tilt,
        radial_error_m=radial,
        mouth_height_m=height,
        flow_gate=gate,
    )


@dataclass(frozen=True)
class SuccessThresholds:
    min_cup_fraction: float = 0.80
    max_spill_fraction: float = 0.20
    max_cup_tilt_deg: float = 15.0
    max_return_tilt_deg: float = 20.0
    verify_hold_steps: int = 10


def success_conditions(
    *,
    phase: torch.Tensor,
    liquid: LiquidState,
    initial_mass: float,
    left_grasp: torch.Tensor,
    right_grasp: torch.Tensor,
    cup_tilt_rad: torch.Tensor,
    bottle_tilt_rad: torch.Tensor,
    thresholds: SuccessThresholds,
) -> dict[str, torch.Tensor]:
    denom = max(float(initial_mass), 1.0e-8)
    return {
        "verify_phase": phase == int(PourPhase.VERIFY),
        "liquid": liquid.cup / denom >= thresholds.min_cup_fraction,
        "spill": liquid.spill / denom <= thresholds.max_spill_fraction,
        "left_grasp": left_grasp.bool(),
        "right_grasp": right_grasp.bool(),
        "cup_upright": cup_tilt_rad <= torch.deg2rad(
            torch.as_tensor(thresholds.max_cup_tilt_deg, device=cup_tilt_rad.device)
        ),
        "bottle_returned": bottle_tilt_rad <= torch.deg2rad(
            torch.as_tensor(thresholds.max_return_tilt_deg, device=bottle_tilt_rad.device)
        ),
    }


def combine_success(conditions: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = list(conditions.values())
    if not values:
        raise ValueError("at least one success condition is required")
    result = torch.ones_like(values[0], dtype=torch.bool)
    for value in values:
        result &= value.bool()
    return result


def failure_codes(
    *,
    left_drop: torch.Tensor,
    right_drop: torch.Tensor,
    cup_tip: torch.Tensor,
    align_miss: torch.Tensor,
    spill: torch.Tensor,
    timeout: torch.Tensor,
) -> torch.Tensor:
    """Assign one deterministic reason using safety-first precedence."""

    code = torch.zeros_like(left_drop, dtype=torch.long)
    # Assign in reverse precedence; later writes win.
    for mask, value in (
        (timeout, FailureCode.TIMEOUT),
        (spill, FailureCode.SPILL),
        (align_miss, FailureCode.ALIGN_MISS),
        (cup_tip, FailureCode.CUP_TIP),
        (right_drop, FailureCode.RIGHT_DROP),
        (left_drop, FailureCode.LEFT_DROP),
    ):
        code = torch.where(mask.bool(), torch.full_like(code, int(value)), code)
    return code


@dataclass(frozen=True)
class RewardWeights:
    trajectory: float = 6.0
    contact: float = 4.0
    align: float = 5.0
    transfer: float = 20.0
    spill: float = 20.0
    cup_upright: float = 3.0
    return_upright: float = 5.0
    action_rate: float = 0.01
    success: float = 25.0


def compute_reward(
    terms: Mapping[str, torch.Tensor],
    *,
    ablation: AblationMode,
    weights: RewardWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine named normalized terms and apply the registered ablation."""

    required = {
        "trajectory",
        "contact",
        "align",
        "transfer",
        "spill",
        "cup_upright",
        "return_upright",
        "action_rate",
        "success",
    }
    missing = required.difference(terms)
    if missing:
        raise KeyError(f"missing reward terms: {sorted(missing)}")
    weighted = {
        "trajectory": terms["trajectory"] * weights.trajectory
        if ablation.use_video_trajectory
        else torch.zeros_like(terms["trajectory"]),
        "contact": terms["contact"] * weights.contact
        if ablation.use_video_contact
        else torch.zeros_like(terms["contact"]),
        "align": terms["align"] * weights.align,
        "transfer": terms["transfer"] * weights.transfer,
        "spill": -terms["spill"] * weights.spill,
        "cup_upright": terms["cup_upright"] * weights.cup_upright,
        "return_upright": terms["return_upright"] * weights.return_upright,
        "action_rate": -terms["action_rate"] * weights.action_rate,
        "success": terms["success"] * weights.success,
    }
    total = torch.stack(list(weighted.values()), dim=0).sum(dim=0)
    return total, weighted


def reward_phase_gates(phase: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the explicit phase switch for every shaped reward family."""

    return {
        "trajectory": (phase >= int(PourPhase.APPROACH))
        & (phase <= int(PourPhase.RETURN)),
        "contact": phase >= int(PourPhase.DUAL_GRASP),
        "align": (phase == int(PourPhase.ALIGN)) | (phase == int(PourPhase.POUR)),
        "transfer": phase == int(PourPhase.POUR),
        "spill": phase == int(PourPhase.POUR),
        "cup_upright": phase >= int(PourPhase.ALIGN),
        "return_upright": phase >= int(PourPhase.RETURN),
        "action_rate": torch.ones_like(phase, dtype=torch.bool),
        "success": phase == int(PourPhase.VERIFY),
    }


def should_unlock_stage(
    stage: CurriculumStage,
    *,
    left_grasp_rate: float,
    right_grasp_rate: float,
    dual_grasp_rate: float,
    approach_grasp_rate: float,
) -> bool:
    if stage == CurriculumStage.SINGLE_GRASP:
        return min(left_grasp_rate, right_grasp_rate) >= 0.90
    if stage == CurriculumStage.DUAL_GRASP:
        return dual_grasp_rate >= 0.85
    if stage == CurriculumStage.APPROACH_GRASP:
        return approach_grasp_rate >= 0.80
    return False
