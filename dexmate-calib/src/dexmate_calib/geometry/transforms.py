"""Minimal SO(3)/SE(3) utilities.

Conventions used throughout dexmate-calib:

* ``T_a_b`` is a 4x4 homogeneous matrix that maps points expressed in frame ``b``
  into frame ``a``; equivalently it is the pose of frame ``b`` seen from ``a``.
* Twists are 6-vectors ``[omega, v]`` (rotation first) and small perturbations are
  applied on the left: ``T <- se3_exp(delta) @ T``.

Everything here is plain numpy so the solvers can be unit-tested without OpenCV.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-12


def skew(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def so3_exp(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-9:
        return np.eye(3) + K + 0.5 * K @ K
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if math.pi - theta < 1e-6:
        # Near pi the sine formula is ill conditioned; use the symmetric part.
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # Fix signs using the off-diagonal terms.
        idx = int(np.argmax(axis))
        for j in range(3):
            if j != idx and A[idx, j] < 0.0:
                axis[j] = -axis[j]
        axis /= max(np.linalg.norm(axis), _EPS)
        return axis * theta
    return (
        np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        * theta
        / (2.0 * math.sin(theta))
    )


def _so3_left_jacobian(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-9:
        return np.eye(3) + 0.5 * K
    a = (1.0 - math.cos(theta)) / (theta * theta)
    b = (theta - math.sin(theta)) / (theta**3)
    return np.eye(3) + a * K + b * K @ K


def _so3_left_jacobian_inv(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-9:
        return np.eye(3) - 0.5 * K
    half = 0.5 * theta
    cot = 1.0 / math.tan(half)
    b = (1.0 / (theta * theta)) - (half * cot / (theta * theta))
    return np.eye(3) - 0.5 * K + b * K @ K


def se3_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    w, v = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(w)
    T[:3, 3] = _so3_left_jacobian(w) @ v
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    w = so3_log(T[:3, :3])
    v = _so3_left_jacobian_inv(w) @ T[:3, 3]
    return np.concatenate([w, v])


def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rvec_to_R(rvec: np.ndarray) -> np.ndarray:
    return so3_exp(rvec)


def R_to_rvec(R: np.ndarray) -> np.ndarray:
    return so3_log(R)


def rotation_angle_deg(R: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(so3_log(R))))


def pose_error(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return (rotation error in degrees, translation error in metres) between two poses."""
    delta = inv_T(T_a) @ T_b
    return rotation_angle_deg(delta[:3, :3]), float(np.linalg.norm(delta[:3, 3]))


def project_rotation(R: np.ndarray) -> np.ndarray:
    """Closest proper rotation matrix (Frobenius) to ``R``."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def is_rigid(T: np.ndarray, atol: float = 1e-6) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    return (
        np.allclose(R.T @ R, np.eye(3), atol=atol)
        and abs(float(np.linalg.det(R)) - 1.0) < atol
        and np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol)
    )
