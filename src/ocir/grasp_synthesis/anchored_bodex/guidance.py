"""Per-sequence human-guidance context: contact-point subsets, regenerated QP
pressure constraints, affordance targets, and guidance energy weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
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
    #: Asymmetric non-penetration penalty over ALL hand collision spheres
    #: (relu(-signed_distance)^2, summed). DISABLED by default (weight 0):
    #: with the force-closure QP energy live in every stage
    #: (rollout.ANCHORED_CONTACT_STRATEGY), the staged contact-distance term
    #: plus the QP's own contact geometry keep the hand at the surface, and
    #: the penalty's per-sphere SDF pushes were contorting poses. When
    #: enabled (``--penetration-weight`` > 0) it is scheduled opposite the
    #: other guidance terms: zero through stage 0, linearly ramped in across
    #: ``pene_ramp`` (the stage-1 window), full weight in the final
    #: (distance=0) stage only.
    w_pene: float = 0.0
    pene_ramp: tuple[float, float] = (0.6, 0.8)
    #: Pairwise sphere-vs-sphere self-collision energy (relu(min_dist -
    #: center_dist)^2, summed over non-adjacent sphere pairs). Unlike every
    #: other guidance term this is NOT scheduled: fingers must never
    #: interpenetrate each other at any point in the optimization, so it is
    #: at full weight in stages 0, 1, and 2 alike.
    w_selfcol: float = 1000.0

    @classmethod
    def from_contact_strategy(
        cls,
        contact_strategy: dict,
        *,
        w_afford: float = 20.0,
        pose_scale: float = 1.0,
        w_pene: float = 0.0,
        w_selfcol: float = 1000.0,
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
            w_pene=float(w_pene),
            pene_ramp=(stage1, stage2),
            w_selfcol=float(w_selfcol),
        )

    def pene_weight(self, opt_progress: float) -> float:
        start, end = self.pene_ramp
        p = float(opt_progress)
        if p <= start:
            return 0.0
        if p >= end:
            return self.w_pene
        return self.w_pene * (p - start) / max(end - start, 1e-9)

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


def _urdf_parent_map(urdf_path: Path) -> dict[str, str]:
    parent_of: dict[str, str] = {}
    for joint in ET.parse(str(urdf_path)).getroot().findall("joint"):
        parent_of[joint.find("child").get("link")] = joint.find("parent").get("link")
    return parent_of


def build_self_collision_pairs(
    urdf_path: Path, sphere_link_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Sphere-index pairs ``(i, j)``, ``i < j``, to check for pairwise
    self-collision: every cross-link pair EXCEPT same-link spheres and
    URDF-adjacent links.

    Adjacent links share a joint surface and are expected to sit close (or
    overlapping) in most poses -- the standard convention is to exclude
    them from self-collision, same as ``bodex_curobo_v2``'s own robot config
    (``self_collision_ignore`` in ``curobo_sharpa_right.yml``). That map is
    keyed to a finer virtual-link naming scheme (``*_MCP_VL``, ``*_elastomer``,
    ``*_fingertip``) used by cuRobo's full mesh-collision model, which does
    not correspond 1:1 to this pipeline's coarser per-finger-segment
    collision-sphere set -- so adjacency is instead derived directly, once,
    from this asset's own URDF joint tree (nearest ancestor among the given
    sphere-bearing links), which is correct by construction for whatever
    link set is actually passed in.
    """

    parent_of = _urdf_parent_map(urdf_path)
    named = set(sphere_link_names)

    def nearest_named_ancestor(link: str) -> str | None:
        cur = parent_of.get(link)
        while cur is not None and cur not in named:
            cur = parent_of.get(cur)
        return cur

    adjacent: set[frozenset[str]] = set()
    for link in named:
        ancestor = nearest_named_ancestor(link)
        if ancestor is not None:
            adjacent.add(frozenset((link, ancestor)))

    n = len(sphere_link_names)
    pairs_i, pairs_j = [], []
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = sphere_link_names[i], sphere_link_names[j]
            if li == lj or frozenset((li, lj)) in adjacent:
                continue
            pairs_i.append(i)
            pairs_j.append(j)
    return np.asarray(pairs_i, dtype=np.int64), np.asarray(pairs_j, dtype=np.int64)
