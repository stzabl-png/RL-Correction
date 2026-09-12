from __future__ import annotations

import numpy as np

from dexmate_calib.geometry.transforms import (
    inv_T,
    is_rigid,
    pose_error,
    rotation_angle_deg,
    rt_to_T,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)


def test_so3_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        w = rng.normal(size=3)
        w = w / np.linalg.norm(w) * rng.uniform(0.0, np.pi - 1e-3)
        R = so3_exp(w)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.allclose(so3_log(R), w, atol=1e-8)


def test_so3_log_near_pi():
    w = np.array([0.0, 0.0, np.pi - 1e-7])
    R = so3_exp(w)
    assert abs(rotation_angle_deg(R) - 180.0) < 1e-4
    assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-6)


def test_se3_exp_log_roundtrip_and_inverse():
    rng = np.random.default_rng(2)
    for _ in range(50):
        xi = rng.normal(size=6)
        xi[:3] *= 0.8
        T = se3_exp(xi)
        assert is_rigid(T)
        assert np.allclose(se3_log(T), xi, atol=1e-8)
        assert np.allclose(inv_T(T) @ T, np.eye(4), atol=1e-12)
        assert np.allclose(se3_exp(-xi), inv_T(T), atol=1e-8)


def test_pose_error():
    T = rt_to_T(so3_exp([0.0, 0.1, 0.0]), [0.0, 0.0, 0.02])
    rot_deg, trans_m = pose_error(np.eye(4), T)
    assert abs(rot_deg - np.degrees(0.1)) < 1e-9
    assert abs(trans_m - 0.02) < 1e-12
