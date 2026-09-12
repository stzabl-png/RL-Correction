from __future__ import annotations

import numpy as np


def deterministic_folds(count: int, folds: int = 5) -> list[tuple[list[int], list[int]]]:
    if count < 3:
        raise ValueError("At least 3 samples are required")
    folds = min(max(2, int(folds)), count)
    result: list[tuple[list[int], list[int]]] = []
    for fold in range(folds):
        held_out = [index for index in range(count) if index % folds == fold]
        train = [index for index in range(count) if index % folds != fold]
        if len(train) >= 3 and held_out:
            result.append((train, held_out))
    return result


def camera_matrix_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    matrix_a = np.asarray(a, dtype=np.float64).reshape(3, 3)
    matrix_b = np.asarray(b, dtype=np.float64).reshape(3, 3)
    return {
        "fx_relative_difference": float(
            abs(matrix_a[0, 0] - matrix_b[0, 0])
            / max((abs(matrix_a[0, 0]) + abs(matrix_b[0, 0])) / 2, 1e-12)
        ),
        "fy_relative_difference": float(
            abs(matrix_a[1, 1] - matrix_b[1, 1])
            / max((abs(matrix_a[1, 1]) + abs(matrix_b[1, 1])) / 2, 1e-12)
        ),
        "principal_point_difference_px": float(np.linalg.norm(matrix_a[:2, 2] - matrix_b[:2, 2])),
    }


def summarize_camera_matrices(matrices: list[np.ndarray]) -> dict | None:
    if len(matrices) < 2:
        return None
    values = np.asarray(
        [[matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]] for matrix in matrices],
        dtype=np.float64,
    )
    return {
        "folds": len(matrices),
        "fx_range_px": float(np.ptp(values[:, 0])),
        "fy_range_px": float(np.ptp(values[:, 1])),
        "cx_range_px": float(np.ptp(values[:, 2])),
        "cy_range_px": float(np.ptp(values[:, 3])),
        "fx_relative_range": float(np.ptp(values[:, 0]) / max(abs(np.mean(values[:, 0])), 1e-12)),
        "fy_relative_range": float(np.ptp(values[:, 1]) / max(abs(np.mean(values[:, 1])), 1e-12)),
        "principal_point_span_px": float(np.linalg.norm(np.ptp(values[:, 2:4], axis=0))),
    }
