"""Small NumPy SE(3) helpers used by generation, control, and tests."""

from __future__ import annotations

import numpy as np

from ocir.grasp_traj.trajectory_schema import matrix_to_quat_wxyz, quat_wxyz_to_matrix


def rotation_angle(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    return float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the shortest axis-angle vector for a 3x3 rotation."""

    rotation = np.asarray(rotation, dtype=np.float64)
    angle = rotation_angle(rotation)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-6:
        # The usual skew extraction is singular at pi.  Recover an axis from
        # the symmetric part and choose signs from the off-diagonal entries.
        diag = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diag)
        if axis[0] >= axis[1] and axis[0] >= axis[2] and axis[0] > 1e-8:
            axis[1] = np.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = np.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        elif axis[1] >= axis[2] and axis[1] > 1e-8:
            axis[0] = np.copysign(axis[0], rotation[0, 1] + rotation[1, 0])
            axis[2] = np.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
        elif axis[2] > 1e-8:
            axis[0] = np.copysign(axis[0], rotation[0, 2] + rotation[2, 0])
            axis[1] = np.copysign(axis[1], rotation[1, 2] + rotation[2, 1])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis * angle
    skew = np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
        dtype=np.float64,
    )
    return skew * (angle / (2.0 * np.sin(angle)))


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def transform_from_twist(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build a transform from a world-frame translation and rotation vector.

    This is intentionally the controller's decoupled pose correction rather
    than the coupled translational part of the Lie exponential.
    """

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation_from_vector(rotation)
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def pose_error(target: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decoupled world-frame translation and rotation error."""

    target = np.asarray(target, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    return target[:3, 3] - actual[:3, 3], rotation_vector(target[:3, :3] @ actual[:3, :3].T)


def apply_world_correction(pose: np.ndarray, translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Translate a pose in world axes and pre-rotate its orientation only."""

    pose = np.asarray(pose, dtype=np.float64)
    out = pose.copy()
    out[:3, :3] = rotation_from_vector(rotation) @ pose[:3, :3]
    out[:3, 3] = pose[:3, 3] + np.asarray(translation, dtype=np.float64)
    return out


def rotate_pose_about_point(pose: np.ndarray, rotation: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Rotate a pose rigidly about a world-frame pivot point.

    Unlike :func:`apply_world_correction`, the pose's position also swings
    around the pivot, so rotating a wrist about the grasped object's center
    reorients the object without displacing it.
    """

    pose = np.asarray(pose, dtype=np.float64)
    pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
    rot = rotation_from_vector(rotation)
    out = pose.copy()
    out[:3, :3] = rot @ pose[:3, :3]
    out[:3, 3] = pivot + rot @ (pose[:3, 3] - pivot)
    return out


def interpolate_transform(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
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


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def clamp_norm(vector: np.ndarray, maximum: float) -> tuple[np.ndarray, bool]:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if maximum <= 0.0 or norm <= maximum:
        return vector.copy(), False
    return vector * (maximum / max(norm, 1e-12)), True


def batch_pos_quat_to_matrix(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    out = np.tile(np.eye(4, dtype=np.float64), (pos.shape[0], 1, 1))
    out[:, :3, :3] = quat_wxyz_to_matrix(quat)
    out[:, :3, 3] = pos
    return out
