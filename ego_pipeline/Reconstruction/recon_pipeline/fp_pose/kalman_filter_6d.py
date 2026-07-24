"""6D Kalman filter for FoundationPose++ style tracking (from STEP_6_pose)."""

import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter6D:
    """Track [tx, ty, tz, rx, ry, rz, v_*] with optional 2D xy measurements."""

    def __init__(self, measurement_noise_scale: float = 0.05):
        ndim, dt = 6, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._update_mat_xy = np.zeros((2, 2 * ndim))
        self._update_mat_xy[0, 0] = 1
        self._update_mat_xy[1, 1] = 1
        self._std_weight_trans_xy = 1.0 / 40
        self._std_weight_trans = 1.0 / 10
        self._std_weight_rot = 1.0 / 20
        self._std_weight_vel_trans = 1.0 / 20
        self._std_weight_vel_rot = 1.0 / 40
        self.measurement_noise_scale = measurement_noise_scale

    def initiate(self, measurement: np.ndarray):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        scale_xyz = max(np.linalg.norm(measurement[:3]), 1e-5)
        scale_rot = max(np.linalg.norm(measurement[3:]), 1e-5)
        std = [
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_trans * scale_xyz,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            0.2 * self._std_weight_rot * scale_rot,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_trans * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
            1 * self._std_weight_vel_rot * scale_xyz,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        scale_xyz = mean[2]
        scale_rot = max(abs(mean[3]), abs(mean[4]), abs(mean[5]), 1e-5)
        std_pos = [
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_trans * scale_xyz,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
            self._std_weight_rot * scale_rot,
        ]
        std_vel = [
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_trans * scale_xyz,
            self._std_weight_vel_rot * scale_xyz,
            self._std_weight_vel_rot * scale_xyz,
            self._std_weight_vel_rot * scale_xyz,
        ]
        motion_cov = np.diag(np.square(std_pos + std_vel))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        std = [
            self._std_weight_trans * max(abs(mean[2]), 1e-5),
            self._std_weight_trans * max(abs(mean[2]), 1e-5),
            self._std_weight_trans * max(abs(mean[2]), 1e-5),
            self._std_weight_rot * 1e-2,
            self._std_weight_rot * 1e-2,
            self._std_weight_rot * 1e-2,
        ]
        std = [s * self.measurement_noise_scale for s in std]
        innovation_cov = np.diag(np.square(std))
        return self._update(mean, covariance, measurement, innovation_cov, self._update_mat)

    def update_from_xy(self, mean, covariance, meas_xy):
        std = [
            self._std_weight_trans_xy * max(abs(mean[2]), 1e-5),
            self._std_weight_trans_xy * max(abs(mean[2]), 1e-5),
        ]
        std = [s * self.measurement_noise_scale for s in std]
        innovation_cov = np.diag(np.square(std))
        return self._update(mean, covariance, meas_xy, innovation_cov, self._update_mat_xy)

    def _update(self, mean, covariance, measurement, innovation_cov, H):
        projected_mean = H @ mean
        projected_cov = H @ covariance @ H.T + innovation_cov
        chol = scipy.linalg.cho_factor(projected_cov, lower=True)
        kalman_gain = scipy.linalg.cho_solve(chol, (covariance @ H.T).T).T
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov


def mat_to_6d(pose: np.ndarray) -> np.ndarray:
    return np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")]


def mat_from_6d(arr: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = arr[:3]
    m[:3, :3] = Rotation.from_euler("xyz", arr[3:]).as_matrix()
    return m


def adjust_xy(pose: np.ndarray, k: np.ndarray, u: float, v: float) -> np.ndarray:
    p = pose.copy()
    tz = p[2, 3]
    p[0, 3] = (u - k[0, 2]) * tz / k[0, 0]
    p[1, 3] = (v - k[1, 2]) * tz / k[1, 1]
    return p


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())
