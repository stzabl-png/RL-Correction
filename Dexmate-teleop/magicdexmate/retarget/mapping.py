"""Joint-order mapping: retargeting output (pinocchio dof order) -> SharpaWave SDK order.

The SDK's `set_joint_position` order equals the URDF joint order
(docs/references/02_sharpa_wave_hand.md §2). `SeqRetargeting.retarget()` returns
the full qpos in pinocchio dof order, whose names are `retargeting.joint_names`;
the two orders generally differ, so we always map by name.
"""

from __future__ import annotations

import numpy as np
from dex_retargeting.seq_retarget import SeqRetargeting

from magicdexmate.retarget.natural_distal import distal_blend

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

    def relax_distal(self, q_sdk: np.ndarray, kp: np.ndarray,
                     thumb_ip_margin: float = 0.10) -> np.ndarray:
        """Blend thumb_IP / pinky_DIP toward the operator's measured flexion when the
        thumb is NOT pinching, while keeping DexPilot's value during a pinch.

        DexPilot over-curls those two under-constrained distal joints (thumb tip
        very bent, pinky tip stuck curled) even on the open hand; that curl is only
        needed to reach the 1-2 mm fingertip opposition while pinching. `kp` is the
        source frame's (21,3) keypoints (any consistent frame). Validated on real
        glove data: opposition unchanged, open hand matches the operator. See
        retarget/natural_distal.py.

        Step 5c (fj): additionally CAP thumb_IP so the robot thumb tip never bends more
        than the operator's own IP flexion by more than ``thumb_ip_margin`` (rad), even
        during a pinch (dexpilot_w=1). The over-curl the optimizer parks in thumb_IP is no
        longer needed for opposition -- Step 5b's conditional pinch vector provides that
        via CMC/MCP -- so we clip it away. This is what fixes "thumb tip looks very bent
        even though I only bent it a little". margin=0 => robot IP <= operator IP exactly.
        """
        q = np.array(q_sdk, dtype=float, copy=True)
        for suffix, (flexion, dexpilot_w) in distal_blend(kp).items():
            if flexion is None:
                continue
            for k, n in enumerate(self.sdk_names):
                if n.endswith(suffix):
                    relaxed = float(np.clip(flexion, self.lo[k], self.hi[k]))
                    q[k] = dexpilot_w * q[k] + (1.0 - dexpilot_w) * relaxed
                    if suffix == "thumb_IP":
                        cap = float(np.clip(flexion + thumb_ip_margin, self.lo[k], self.hi[k]))
                        q[k] = min(q[k], cap)
        return q
