"""Arm swivel: carry the operator's elbow ORIENTATION, not its position.

The problem this exists for
--------------------------
A 7-DoF arm tracking a 6-DoF wrist pose has one free dimension, and it is the
elbow swinging around the shoulder-wrist axis. Nothing in the wrist target
constrains it, so the solver resolves it from the posture task alone - and on
real data the two arms land on different branches. Measured over 3000 frames of
a real take, with the wrist targets only 113 mm / 29 deg apart left-vs-right:

    joint   |q_left - mirror(q_right)|, median
    j3      146 deg      <- the swivel joints
    j5      112 deg
    j1       88 deg
    j4        9 deg      <- elbow FLEXION agrees; only the swivel differs

The solver itself is symmetric: fed exactly mirrored targets it returns
mirrored joints to within 0.1 deg. So this is redundancy resolution, not a bug.

Why not just target the operator's elbow position
-------------------------------------------------
That is what `--elbow-weight` does today and it does not work: the target is
the human elbow relative to their waist, dropped onto the robot's bind point,
and the two bodies do not share proportions (shoulder-offset/arm = 0.22 on the
robot, ~0.35 on a person). The target frequently sits where the robot's elbow
physically cannot go, and the solver then trades the wrist away trying to
reach it - measured wrist residual goes 30 mm -> 475 mm, at both 0.25 and 1.0
weight. The operator's own description of it: the elbow follows better but the
hand "easily loses it".

What this does instead
----------------------
Take only the ANGLE - where the operator's elbow sits around their own
shoulder-wrist axis - and place the robot's elbow at the same angle around the
ROBOT's shoulder-wrist axis, on the circle its own link lengths allow. The
angle is dimensionless, so differing proportions never enter, and the target is
reachable by construction, so the elbow task can never fight the wrist task.
"""
from __future__ import annotations

import numpy as np


def _basis(shoulder: np.ndarray, wrist: np.ndarray, ref_up: np.ndarray):
    """Frame on the shoulder->wrist axis: (axis, u, v), u in the ref_up plane."""
    axis = wrist - shoulder
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return None
    axis = axis / n
    u = ref_up - axis * (ref_up @ axis)          # ref_up, made perpendicular
    nu = np.linalg.norm(u)
    if nu < 1e-6:                                # arm points along ref_up
        alt = np.array([1.0, 0.0, 0.0])
        u = alt - axis * (alt @ axis)
        nu = np.linalg.norm(u)
        if nu < 1e-6:
            return None
    u = u / nu
    return axis, u, np.cross(axis, u), n


def swivel_angle(shoulder, elbow, wrist, ref_up=np.array([0.0, 0.0, 1.0])) -> float | None:
    """Angle [rad] of the elbow about the shoulder->wrist axis, or None.

    Measured from ref_up projected into the plane, so it is comparable between
    two bodies of different size as long as both use the same ref_up.
    """
    b = _basis(np.asarray(shoulder, float), np.asarray(wrist, float),
               np.asarray(ref_up, float))
    if b is None:
        return None
    axis, u, v, _ = b
    r = np.asarray(elbow, float) - np.asarray(shoulder, float)
    r = r - axis * (r @ axis)                    # into the circle plane
    if np.linalg.norm(r) < 1e-6:                 # elbow on the axis: no angle
        return None
    return float(np.arctan2(r @ v, r @ u))


# SMPL body24 关节序里的肩与肘(Unity 侧固定)。此前唯一居所在
# sim/teleop_vega_pico.py 的 _SMPL_SHOULDER/_SMPL_ELBOW(两处必须一致;
# teleop 侧待其解冻后应改为从这里 import,消掉副本)。
SMPL_SHOULDER = {"left": 16, "right": 17}
SMPL_ELBOW = {"left": 18, "right": 19}


def operator_swivel_from_frame(hand: str, body24, wrist_pose7, waist_pose7
                               ) -> float | None:
    """操作者的肘绕自己肩-腕轴的角度,弧度;取不到时返回 None。

    与 sim/teleop_vega_pico.py 的 _operator_swivel 同一套约定(2026-08-11
    抽出,供离线台架对回放素材直接计算):肩/肘取 body24 的 16/17、18/19,
    腕取腕部 tracker,全部先经 process_xr_pose 换到腰 tracker 参考系再算角。
    只搬角度不搬位置的理由见本文件模块 docstring。
    """
    from magicdexmate.pico.xr_pose import mat_to_pos_quat_wxyz, process_xr_pose
    if waist_pose7 is None or body24 is None or wrist_pose7 is None:
        return None
    ref = np.asarray(waist_pose7, dtype=np.float64)
    try:
        p_w = mat_to_pos_quat_wxyz(process_xr_pose(
            np.asarray(wrist_pose7, dtype=np.float64), ref))[0]
        p_s = mat_to_pos_quat_wxyz(process_xr_pose(
            np.asarray(body24[SMPL_SHOULDER[hand]], dtype=np.float64), ref))[0]
        p_e = mat_to_pos_quat_wxyz(process_xr_pose(
            np.asarray(body24[SMPL_ELBOW[hand]], dtype=np.float64), ref))[0]
    except (IndexError, TypeError, ValueError):
        return None
    return swivel_angle(p_s, p_e, p_w)


def elbow_from_swivel(shoulder, wrist, l_upper: float, l_fore: float,
                      angle: float, ref_up=np.array([0.0, 0.0, 1.0])):
    """Where the robot's elbow goes for a given swivel angle.

    Solves the two-link circle: with the shoulder and wrist fixed, the elbow
    lies on a circle perpendicular to the shoulder-wrist axis. Returns a point
    ON that circle, so it is always reachable - which is the whole point.
    Returns None if the arm is stretched beyond its own reach (then there is no
    circle, and the elbow is determined anyway).
    """
    S = np.asarray(shoulder, float)
    b = _basis(S, np.asarray(wrist, float), np.asarray(ref_up, float))
    if b is None:
        return None
    axis, u, v, d = b
    if d > l_upper + l_fore - 1e-4 or d < abs(l_upper - l_fore) + 1e-4:
        return None                              # straight, or folded: no circle
    d1 = (l_upper ** 2 - l_fore ** 2 + d ** 2) / (2.0 * d)
    r2 = l_upper ** 2 - d1 ** 2
    if r2 <= 0.0:
        return None
    r = np.sqrt(r2)
    return S + axis * d1 + r * (np.cos(angle) * u + np.sin(angle) * v)
