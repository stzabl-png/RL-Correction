"""Relax DexPilot's redundant distal over-curl on the open hand.

Real-glove data (2026-06-22): DexPilot holds thumb_IP ~0.8-1.3 and pinky_DIP
~1.33 even when the operator's own joints are near-straight (~0.2), because those
distal joints are under-constrained and the optimiser parks them high. That
over-curl is only *needed* while pinching - it is how the 1-2 mm fingertip
opposition is reached - and is pure artifact when the thumb is far from the
fingers. So we blend thumb_IP / pinky_DIP toward the operator's measured flexion,
weighted by how close the thumb is to a pinch: full DexPilot while pinching,
operator flexion when open. Opposition is preserved; the open hand looks natural.
"""

from __future__ import annotations

import numpy as np

# joint suffix -> (proximal, vertex, tip) MediaPipe indices for the flexion angle
_TRIPLET = {"thumb_IP": (2, 3, 4), "pinky_DIP": (18, 19, 20)}
# joint suffix -> fingertip indices whose nearness to the thumb means "pinching"
_PINCH_PARTNERS = {"thumb_IP": (8, 12, 16, 20), "pinky_DIP": (20,)}
_THUMB_TIP = 4


def _flex(kp: np.ndarray, a: int, b: int, c: int) -> float | None:
    u, v = kp[a] - kp[b], kp[c] - kp[b]
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return None
    return float(np.pi - np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0)))


def _dexpilot_weight(kp: np.ndarray, partners, near_mm: float, far_mm: float) -> float:
    """1.0 when the thumb is pinching a partner (keep DexPilot), 0.0 when open."""
    d_mm = min(np.linalg.norm(kp[_THUMB_TIP] - kp[j]) for j in partners) * 1000.0
    return float(np.clip((far_mm - d_mm) / (far_mm - near_mm), 0.0, 1.0))


def distal_blend(kp: np.ndarray, near_mm: float = 30.0, far_mm: float = 70.0) -> dict:
    """{joint_suffix: (operator_flexion_rad | None, dexpilot_weight in [0,1])}."""
    return {
        j: (_flex(kp, *t), _dexpilot_weight(kp, _PINCH_PARTNERS[j], near_mm, far_mm))
        for j, t in _TRIPLET.items()
    }
