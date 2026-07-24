"""Small NumPy SE(3) helpers used by open-loop retargeting and tests."""

from __future__ import annotations

import numpy as np

from ocir.grasp_traj.trajectory_schema import matrix_to_quat_wxyz, quat_wxyz_to_matrix


def rotation_angle(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def interpolate_transform(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate translation linearly and rotation by shortest-path slerp."""

    alpha = float(np.clip(alpha, 0.0, 1.0))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    qa = matrix_to_quat_wxyz(a[:3, :3])
    qb = matrix_to_quat_wxyz(b[:3, :3])
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    dot = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
    if dot > 0.9995:
        quat = qa * (1.0 - alpha) + qb * alpha
        quat /= np.linalg.norm(quat)
    else:
        theta = np.arccos(dot)
        quat = (np.sin((1.0 - alpha) * theta) * qa + np.sin(alpha * theta) * qb) / np.sin(theta)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_wxyz_to_matrix(quat)
    out[:3, 3] = a[:3, 3] * (1.0 - alpha) + b[:3, 3] * alpha
    return out


def batch_pos_quat_to_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    out = np.tile(np.eye(4, dtype=np.float64), (pos.shape[0], 1, 1))
    out[:, :3, :3] = quat_wxyz_to_matrix(quat)
    out[:, :3, 3] = pos
    return out
