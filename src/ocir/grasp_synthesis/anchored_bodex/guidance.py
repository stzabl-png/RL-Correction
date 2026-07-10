"""Per-sequence human-guidance context: contact-point subsets, regenerated QP
pressure constraints, affordance targets, and guidance energy weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ocir.grasp_synthesis.anchored_bodex.demo_analysis import ROLE_ORDER

#: Role -> Sharpa contact point ("LinkName/sphere_index"), one-to-one and
#: order-preserving with bodex_curobo_v2.solver.SHARPA_CONTACT_POINTS.
ROLE_TO_CONTACT: dict[str, str] = {
    "pinky_tip": "right_pinky_DP/1",
    "pinky_pad": "right_pinky_PP/0",
    "ring_pad": "right_ring_PP/0",
    "ring_tip": "right_ring_DP/1",
    "middle_pad": "right_middle_PP/0",
    "middle_tip": "right_middle_DP/1",
    "index_pad": "right_index_PP/0",
    "index_tip": "right_index_DP/1",
    "thumb_pad": "right_thumb_PP/0",
    "thumb_tip": "right_thumb_DP/1",
    "palm": "right_hand_C_MC/0",
}

#: Role-group generalization of the hard-coded pressure-constraint index
#: lists in bodex_curobo_v2.solver.DEFAULT_GE_PARAM (which assume all 11
#: contact points): (set of roles, total pressure cap).
PRESSURE_TEMPLATES: tuple[tuple[frozenset[str], float], ...] = (
    (frozenset(ROLE_ORDER), 1.0),
    (frozenset({"thumb_pad", "thumb_tip"}), 0.5),
    (
        frozenset(
            {
                "pinky_tip",
                "pinky_pad",
                "ring_pad",
                "ring_tip",
                "middle_pad",
                "middle_tip",
                "index_pad",
                "index_tip",
            }
        ),
        0.7,
    ),
    (frozenset({"pinky_tip", "pinky_pad", "ring_pad", "ring_tip"}), 0.4),
    (frozenset({"middle_pad", "middle_tip", "index_pad", "index_tip"}), 0.5),
)

MIN_ACTIVE_CONTACTS = 3


def select_contact_points(active_roles: tuple[str, ...] | list[str]) -> list[str]:
    """Active subset of Sharpa contact points, canonical order preserved."""

    unknown = [role for role in active_roles if role not in ROLE_TO_CONTACT]
    if unknown:
        raise KeyError(f"unknown contact roles {unknown}")
    active = set(active_roles)
    return [ROLE_TO_CONTACT[role] for role in ROLE_ORDER if role in active]


def build_pressure_constraints(active_roles: tuple[str, ...] | list[str]) -> list[list]:
    """Instantiate PRESSURE_TEMPLATES for a role subset.

    Indices are remapped to positions within the active subset (canonical
    order); groups with no active member are dropped; caps are unchanged.
    The full ROLE_ORDER reproduces bodex_curobo_v2's DEFAULT_GE_PARAM lists.
    """

    active = set(active_roles)
    if len(active) < MIN_ACTIVE_CONTACTS:
        raise ValueError(f"need at least {MIN_ACTIVE_CONTACTS} active contact roles, got {sorted(active)}")
    ordered = [role for role in ROLE_ORDER if role in active]
    index_of = {role: i for i, role in enumerate(ordered)}
    constraints: list[list] = []
    for roles, cap in PRESSURE_TEMPLATES:
        indices = [index_of[role] for role in ordered if role in roles]
        if indices:
            constraints.append([indices, float(cap)])
    return constraints


@dataclass(frozen=True)
class GuidanceWeights:
    """Guidance energy weights and their opt_progress schedules.

    The anneal boundaries are derived from the BODex contact-strategy stage
    fractions (``DEFAULT_CONTACT_STRATEGY["opt_progress"]``): the pose prior
    must reach zero before the stage-0 -> 1 switch (after which the staged
    cost chases frozen contact-target snapshots that a live prior would fight),
    and the affordance attraction decays to zero across stages 1 -> 2.
    """

    w_afford: float = 20.0
    w_pose: tuple[float, float, float] = (300.0, 30.0, 3.0)  # pos, rot, joints
    pose_anneal_end: float = 0.6
    afford_decay: tuple[float, float] = (0.6, 0.8)

    @classmethod
    def from_contact_strategy(
        cls, contact_strategy: dict, *, w_afford: float = 20.0, pose_scale: float = 1.0
    ) -> "GuidanceWeights":
        stages = list(contact_strategy["opt_progress"])
        stage1 = float(stages[1]) if len(stages) > 1 else 0.6
        stage2 = float(stages[2]) if len(stages) > 2 else min(stage1 + 0.2, 1.0)
        base = cls()
        return cls(
            w_afford=float(w_afford),
            w_pose=tuple(pose_scale * w for w in base.w_pose),
            pose_anneal_end=stage1,
            afford_decay=(stage1, stage2),
        )

    def pose_prior_weight(self, opt_progress: float) -> float:
        if self.pose_anneal_end <= 0:
            return 0.0
        return max(0.0, 1.0 - float(opt_progress) / self.pose_anneal_end)

    def afford_weight(self, opt_progress: float) -> float:
        start, end = self.afford_decay
        p = float(opt_progress)
        if p <= start:
            return 1.0
        if p >= end:
            return 0.0
        return (end - p) / max(end - start, 1e-9)


def make_afford_contact_mask(contact_points: list[str], device: torch.device) -> torch.Tensor:
    """Mask selecting fingertip/pad contacts for the affordance attraction
    (the palm follows as a consequence; it is not an intent signal)."""

    palm = ROLE_TO_CONTACT["palm"]
    return torch.tensor([cp != palm for cp in contact_points], device=device, dtype=torch.bool)
