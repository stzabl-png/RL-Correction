"""Timing-independent ordered SE(3) path matching metrics."""

from __future__ import annotations

import numpy as np


def _pairwise_pose_errors(actual: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.linalg.norm(
        actual[:, None, :3, 3] - reference[None, :, :3, 3],
        axis=-1,
    )
    rotation = np.empty((actual.shape[0], reference.shape[0]), dtype=np.float64)
    for i in range(actual.shape[0]):
        relative = reference[:, :3, :3] @ actual[i, :3, :3].T
        trace = np.trace(relative, axis1=1, axis2=2)
        rotation[i] = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return translation, rotation


def paired_pose_path_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    """Pose errors for two paths sampled at the same open-loop timestamps."""

    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape or actual.ndim != 3 or actual.shape[1:] != (4, 4):
        raise ValueError("paired pose paths must have identical shape (N,4,4)")
    if actual.shape[0] == 0:
        raise ValueError("paired pose path metrics require non-empty paths")
    translation = np.linalg.norm(actual[:, :3, 3] - reference[:, :3, 3], axis=1)
    relative = reference[:, :3, :3] @ np.swapaxes(actual[:, :3, :3], 1, 2)
    trace = np.trace(relative, axis1=1, axis2=2)
    rotation = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(translation**2))),
        "translation_max_m": float(translation.max()),
        "translation_final_m": float(translation[-1]),
        "orientation_rmse_rad": float(np.sqrt(np.mean(rotation**2))),
        "orientation_max_rad": float(rotation.max()),
        "orientation_final_rad": float(rotation[-1]),
    }


def ordered_pose_path_metrics(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> dict:
    """Align two pose paths monotonically with dynamic time warping.

    The combined DTW cost normalizes translation and rotation by their
    requested tolerances.  Reported translation/orientation errors retain
    their physical units and use the recovered monotonic alignment path.
    """

    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.ndim != 3 or actual.shape[1:] != (4, 4) or reference.ndim != 3 or reference.shape[1:] != (4, 4):
        raise ValueError("actual and reference paths must have shape (N,4,4)")
    if actual.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("path metrics require non-empty paths")
    if position_tolerance_m <= 0.0 or orientation_tolerance_rad <= 0.0:
        raise ValueError("path tolerances must be positive")

    translation, rotation = _pairwise_pose_errors(actual, reference)
    local_cost = np.sqrt(
        (translation / float(position_tolerance_m)) ** 2
        + (rotation / float(orientation_tolerance_rad)) ** 2
    )
    n, m = local_cost.shape
    cumulative = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cumulative[0, 0] = 0.0
    parent = np.zeros((n, m), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (cumulative[i - 1, j - 1], cumulative[i - 1, j], cumulative[i, j - 1])
            selected = int(np.argmin(choices))
            cumulative[i, j] = local_cost[i - 1, j - 1] + choices[selected]
            parent[i - 1, j - 1] = selected

    aligned_actual: list[int] = []
    aligned_reference: list[int] = []
    i, j = n - 1, m - 1
    while True:
        aligned_actual.append(i)
        aligned_reference.append(j)
        if i == 0 and j == 0:
            break
        move = int(parent[i, j])
        if move == 0:
            i, j = max(i - 1, 0), max(j - 1, 0)
        elif move == 1:
            i = max(i - 1, 0)
        else:
            j = max(j - 1, 0)
    aligned_actual = aligned_actual[::-1]
    aligned_reference = aligned_reference[::-1]
    aligned_t = translation[aligned_actual, aligned_reference]
    aligned_r = rotation[aligned_actual, aligned_reference]

    # Coverage counts each demonstrated reference pose once, using its best
    # error among the actual poses paired to it by the monotonic alignment.
    reference_t = np.full(m, np.inf, dtype=np.float64)
    reference_r = np.full(m, np.inf, dtype=np.float64)
    for ai, ri in zip(aligned_actual, aligned_reference):
        if local_cost[ai, ri] < np.sqrt(
            (reference_t[ri] / position_tolerance_m) ** 2
            + (reference_r[ri] / orientation_tolerance_rad) ** 2
        ):
            reference_t[ri] = translation[ai, ri]
            reference_r[ri] = rotation[ai, ri]
    return {
        "dtw_normalized_cost": float(cumulative[n, m] / len(aligned_actual)),
        "alignment_length": int(len(aligned_actual)),
        "translation_rmse_m": float(np.sqrt(np.mean(aligned_t**2))),
        "translation_max_m": float(aligned_t.max()),
        "translation_final_m": float(aligned_t[-1]),
        "orientation_rmse_rad": float(np.sqrt(np.mean(aligned_r**2))),
        "orientation_max_rad": float(aligned_r.max()),
        "orientation_final_rad": float(aligned_r[-1]),
        "translation_path_coverage_fraction": float(np.mean(reference_t <= position_tolerance_m)),
        "orientation_path_coverage_fraction": float(np.mean(reference_r <= orientation_tolerance_rad)),
        "joint_pose_path_coverage_fraction": float(np.mean(
            (reference_t <= position_tolerance_m) & (reference_r <= orientation_tolerance_rad)
        )),
        "aligned_actual_index": aligned_actual,
        "aligned_reference_index": aligned_reference,
    }
