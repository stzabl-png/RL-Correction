"""When the operator's arms are mirror-symmetric, make the robot's arms too.

The operator's requirement, stated 2026-08-01:

    "if the human skeleton is left-right similar, then Dexmate's two arms must
     also be highly similar"

and their observation that they were not, repeatedly, across several different
mappings.

Why it does not happen by itself
--------------------------------
The solver is not at fault. Fed EXACTLY mirrored targets, Pink returns mirrored
joints to within 0.1 deg (checked over four poses, 400 settle iterations each).
The trouble is that a person is never exactly mirrored: over the thirteen
authored poses the operator's own two wrists sit 19-182 mm apart in mirror
distance, median 42 mm, with 4-21 deg of orientation difference even on the
poses they call symmetric. A 7-DoF arm has one redundant dimension - the elbow
swinging about the shoulder-wrist axis - and nothing in a wrist target pins it,
so those small input differences resolve into left/right joint differences of
90-150 deg.

So symmetry cannot be recovered after the solve, and weighting the solver does
not produce it either. It has to be put into the TARGETS, where it is exact by
construction: mirror-average the pair, and the proven property above does the
rest.

The blend
---------
Full symmetrisation below D_LO, none above D_HI, smooth in between, so a pose
that drifts across the boundary does not pop. The thresholds come from the
operator's own spread: 9 of their 12 poses sit at 19-61 mm (the ones they call
symmetric) while front-horizontal at 182 mm and bow-30 at 105 mm are genuinely
asymmetric and must be passed through untouched - forcing those symmetric would
be the posture task eating the input all over again, which is the failure this
whole line of work started from.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

MIRROR = np.diag([1.0, -1.0, 1.0])      # sagittal plane, y -> -y
D_LO = 0.060    # m of mirror distance: at or below, fully symmetrise
D_HI = 0.120    # m: at or above, pass through unchanged


def mirror_distance(p_right: np.ndarray, p_left: np.ndarray) -> float:
    """How far the operator is from being mirror-symmetric, in metres."""
    return float(np.linalg.norm(np.asarray(p_left) - MIRROR @ np.asarray(p_right)))


def blend_weight(d: float) -> float:
    """1 = fully symmetrise, 0 = pass through. Smoothstep between the bounds."""
    if d <= D_LO:
        return 1.0
    if d >= D_HI:
        return 0.0
    t = (D_HI - d) / (D_HI - D_LO)
    return float(t * t * (3.0 - 2.0 * t))


def symmetrise(p_right, p_left, R_right, R_left, weight: float | None = None):
    """-> (p_right, p_left, R_right, R_left, weight) pulled toward mirror symmetry.

    Positions are mirror-averaged; rotations are averaged in the mirrored frame
    and then mirrored back, so the returned pair is an exact mirror image of
    itself at weight 1 whatever the inputs were.
    """
    p_r = np.asarray(p_right, float)
    p_l = np.asarray(p_left, float)
    if weight is None:
        weight = blend_weight(mirror_distance(p_r, p_l))
    if weight <= 0.0:
        return p_r, p_l, R_right, R_left, 0.0

    p_mid = 0.5 * (p_r + MIRROR @ p_l)
    R_mid = R.from_matrix(np.stack([R_right, MIRROR @ R_left @ MIRROR])).mean()

    out_p_r = (1.0 - weight) * p_r + weight * p_mid
    out_p_l = (1.0 - weight) * p_l + weight * (MIRROR @ p_mid)
    # slerp each side toward its symmetric counterpart
    out_R_r = R.from_matrix(np.stack([R_right, R_mid.as_matrix()])).mean(
        weights=[1.0 - weight, weight]).as_matrix()
    R_l_sym = MIRROR @ R_mid.as_matrix() @ MIRROR
    out_R_l = R.from_matrix(np.stack([R_left, R_l_sym])).mean(
        weights=[1.0 - weight, weight]).as_matrix()
    return out_p_r, out_p_l, out_R_r, out_R_l, float(weight)
