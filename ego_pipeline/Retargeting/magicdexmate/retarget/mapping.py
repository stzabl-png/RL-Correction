"""Joint-order mapping: retargeting output (pinocchio dof order) -> SharpaWave SDK order.

The SDK's `set_joint_position` order equals the URDF joint order
(docs/references/02_sharpa_wave_hand.md §2). `SeqRetargeting.retarget()` returns
the full qpos in pinocchio dof order, whose names are `retargeting.joint_names`;
the two orders generally differ, so we always map by name.
"""

from __future__ import annotations

import numpy as np
from dex_retargeting.seq_retarget import SeqRetargeting

_SHARPA_JOINTS_TEMPLATE = [
    "{s}_thumb_CMC_FE",
    "{s}_thumb_CMC_AA",
    "{s}_thumb_MCP_FE",
    "{s}_thumb_MCP_AA",
    "{s}_thumb_IP",
    "{s}_index_MCP_FE",
    "{s}_index_MCP_AA",
    "{s}_index_PIP",
    "{s}_index_DIP",
    "{s}_middle_MCP_FE",
    "{s}_middle_MCP_AA",
    "{s}_middle_PIP",
    "{s}_middle_DIP",
    "{s}_ring_MCP_FE",
    "{s}_ring_MCP_AA",
    "{s}_ring_PIP",
    "{s}_ring_DIP",
    "{s}_pinky_CMC",
    "{s}_pinky_MCP_FE",
    "{s}_pinky_MCP_AA",
    "{s}_pinky_PIP",
    "{s}_pinky_DIP",
]


def sharpa_sdk_joint_names(hand: str = "right") -> list[str]:
    assert hand in ("right", "left")
    return [t.format(s=hand) for t in _SHARPA_JOINTS_TEMPLATE]


class JointMapper:
    """Reorders retargeting qpos to SDK order and clips into scaled joint limits."""

    def __init__(self, retargeting: SeqRetargeting, hand: str = "right", limit_scale: float = 0.9):
        robot = retargeting.optimizer.robot
        pin_names = list(retargeting.joint_names)  # == robot.dof_joint_names
        self.sdk_names = sharpa_sdk_joint_names(hand)

        missing = [n for n in self.sdk_names if n not in pin_names]
        if missing:
            raise ValueError(
                f"URDF/retargeting joints do not cover SDK joints, missing: {missing}; "
                f"retargeting has: {pin_names}"
            )
        self.idx = np.array([pin_names.index(n) for n in self.sdk_names], dtype=int)

        limits = np.asarray(robot.joint_limits)[self.idx]  # (22, 2) in SDK order
        center = limits.mean(axis=1)
        half = (limits[:, 1] - limits[:, 0]) / 2.0 * limit_scale
        self.lo = center - half
        self.hi = center + half

    def to_sdk(self, qpos_pin_order: np.ndarray) -> np.ndarray:
        """(dof,) retargeting output -> (22,) SDK-order, clipped to scaled limits."""
        return np.clip(np.asarray(qpos_pin_order)[self.idx], self.lo, self.hi)
