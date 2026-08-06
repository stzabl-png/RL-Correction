"""Pure-NumPy pose/joint interpolation helpers for the synthetic approach ->
close -> squeeze segment (Stage A). No torch, no pxr.
"""

from __future__ import annotations

import numpy as np

#: Base-frame approach axis: the palm's local +X points away from the object
#: once grasping (confirmed via bodex_curobo_v2/seed_generator.py's
#: init_r_from_axis convention, and matching the Articulation_Bodex
#: reference's PALM_DIRECTION_LOCAL = [1,0,0]).
PALM_APPROACH_AXIS_LOCAL = np.asarray([1.0, 0.0, 0.0])


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    from dexisaac.grasp_traj.trajectory_schema import quat_wxyz_to_matrix

    return quat_wxyz_to_matrix(np.asarray(quat_wxyz, dtype=np.float64))


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    from dexisaac.grasp_traj.trajectory_schema import matrix_to_quat_wxyz

    return matrix_to_quat_wxyz(np.asarray(matrix, dtype=np.float64))


def slerp_wxyz(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Shortest-path slerp between two wxyz quaternions at parameter t in [0,1]."""

    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(dot)
    theta = theta0 * t
    q2_orth = q2 - q1 * dot
    q2_orth = q2_orth / np.linalg.norm(q2_orth)
    return q1 * np.cos(theta) + q2_orth * np.sin(theta)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a * (1.0 - t) + b * t


def clamp_joints(joints: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(joints, dtype=np.float64), lower, upper)


def standoff_pose(pos: np.ndarray, quat_wxyz: np.ndarray, distance: float) -> np.ndarray:
    """Pull ``pos`` back along the palm's local +X (approach) axis by ``distance``."""

    pos = np.asarray(pos, dtype=np.float64)
    rot = quat_to_matrix(quat_wxyz)
    approach_world = rot @ PALM_APPROACH_AXIS_LOCAL
    return pos - float(distance) * approach_world


def interp_trajectory(
    pos_a: np.ndarray,
    quat_a: np.ndarray,
    joints_a: np.ndarray,
    pos_b: np.ndarray,
    quat_b: np.ndarray,
    joints_b: np.ndarray,
    n_steps: int,
    *,
    include_start: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear position + slerp orientation + linear joint interpolation.

    Returns arrays of length ``n_steps + 1`` (inclusive of both endpoints) if
    ``include_start``, else length ``n_steps`` (endpoint b only, step 0
    dropped) -- used to avoid duplicating a pose already emitted by the
    previous segment.
    """

    pos_a = np.asarray(pos_a, dtype=np.float64)
    pos_b = np.asarray(pos_b, dtype=np.float64)
    joints_a = np.asarray(joints_a, dtype=np.float64)
    joints_b = np.asarray(joints_b, dtype=np.float64)
    n_steps = int(n_steps)
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")

    ts = np.linspace(0.0, 1.0, n_steps + 1)
    if not include_start:
        ts = ts[1:]
    positions = np.stack([lerp(pos_a, pos_b, t) for t in ts], axis=0)
    quats = np.stack([slerp_wxyz(quat_a, quat_b, t) for t in ts], axis=0)
    joints = np.stack([lerp(joints_a, joints_b, t) for t in ts], axis=0)
    return positions, quats, joints


def steps_for_leg(distance_m: float, seconds: float, fps: float, max_speed_mps: float) -> int:
    """Step count for a straight wrist leg: at least the duration-based count,
    more if that would exceed ``max_speed_mps``."""

    duration_steps = max(1, int(round(seconds * fps)))
    if max_speed_mps <= 0.0:
        return duration_steps
    speed_steps = int(np.ceil(float(distance_m) * fps / max_speed_mps))
    return max(duration_steps, speed_steps, 1)


def polyline_wrist_trajectory(
    waypoints_pos: np.ndarray,
    waypoints_quat: np.ndarray,
    n_steps: int,
    *,
    include_start: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise linear position + slerp orientation through ``waypoints``
    ((W,3) and (W,4) wxyz, W >= 2). ``n_steps`` interior steps are allocated
    to legs proportionally to leg length (each leg gets at least 1). Returns
    ``(positions, quats)`` of length ``n_steps + 1`` (both endpoints) if
    ``include_start``, else ``n_steps``."""

    waypoints_pos = np.asarray(waypoints_pos, dtype=np.float64)
    waypoints_quat = np.asarray(waypoints_quat, dtype=np.float64)
    n_legs = waypoints_pos.shape[0] - 1
    if n_legs < 1:
        raise ValueError("polyline needs at least 2 waypoints")
    n_steps = int(n_steps)
    if n_steps < n_legs:
        n_steps = n_legs

    leg_lengths = np.linalg.norm(np.diff(waypoints_pos, axis=0), axis=1)
    total = float(leg_lengths.sum())
    if total <= 0.0:
        leg_steps = np.full(n_legs, n_steps // n_legs, dtype=int)
    else:
        leg_steps = np.maximum(1, np.round(n_steps * leg_lengths / total).astype(int))
    # nudge the longest leg so counts sum exactly to n_steps
    leg_steps[int(np.argmax(leg_lengths))] += n_steps - int(leg_steps.sum())

    positions: list[np.ndarray] = []
    quats: list[np.ndarray] = []
    for i in range(n_legs):
        pos_leg, quat_leg, _ = interp_trajectory(
            waypoints_pos[i], waypoints_quat[i], np.zeros(1),
            waypoints_pos[i + 1], waypoints_quat[i + 1], np.zeros(1),
            int(leg_steps[i]),
            include_start=(i == 0 and include_start),
        )
        positions.append(pos_leg)
        quats.append(quat_leg)
    return np.concatenate(positions, axis=0), np.concatenate(quats, axis=0)


def finger_open_schedule(
    joints_a: np.ndarray,
    joints_b: np.ndarray,
    n_total: int,
    open_fraction: float,
    *,
    include_start: bool = True,
) -> np.ndarray:
    """Joint schedule over ``n_total`` steps: interpolate ``joints_a`` ->
    ``joints_b`` over the first ``ceil(open_fraction * n_total)`` steps, hold
    ``joints_b`` for the rest. Length matches ``polyline_wrist_trajectory``:
    ``n_total + 1`` rows if ``include_start`` else ``n_total``."""

    joints_a = np.asarray(joints_a, dtype=np.float64)
    joints_b = np.asarray(joints_b, dtype=np.float64)
    n_total = int(n_total)
    n_open = max(1, int(np.ceil(float(open_fraction) * n_total)))
    n_open = min(n_open, n_total)
    ts = np.minimum(np.arange(n_total + 1, dtype=np.float64) / n_open, 1.0)
    if not include_start:
        ts = ts[1:]
    return np.stack([lerp(joints_a, joints_b, t) for t in ts], axis=0)


def blend_into_trajectory(
    pos_from: np.ndarray,
    quat_from: np.ndarray,
    pos_traj: np.ndarray,
    quat_traj: np.ndarray,
    n_blend: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ease the first ``n_blend`` steps of a trajectory out of a fixed start
    pose: step ``i`` blends ``pos_from``/``quat_from`` toward
    ``pos_traj[i]``/``quat_traj[i]`` with weight ``(i+1)/n_blend`` (fully on
    the trajectory from step ``n_blend - 1`` onward). Returns copies."""

    pos_out = np.asarray(pos_traj, dtype=np.float64).copy()
    quat_out = np.asarray(quat_traj, dtype=np.float64).copy()
    n_blend = min(int(n_blend), pos_out.shape[0])
    for i in range(n_blend):
        t = (i + 1) / n_blend
        pos_out[i] = lerp(pos_from, pos_out[i], t)
        quat_out[i] = slerp_wxyz(quat_from, quat_out[i], t)
    return pos_out, quat_out
