"""Temporal smoothing for HaWoR world-space MANO predictions."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def _valid_segments(valid: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_valid in enumerate(valid.astype(bool)):
        if is_valid and start is None:
            start = idx
        elif not is_valid and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(valid)))
    return segments


def _gaussian_kernel(sigma: float) -> tuple[np.ndarray, int]:
    radius = max(1, int(round(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel, radius


def _smooth_euclidean_segment(values: np.ndarray, sigma: float) -> np.ndarray:
    if values.shape[0] < 2 or sigma <= 0:
        return values
    return gaussian_filter1d(values, sigma=sigma, axis=0, mode="reflect")


def _unwrap_quaternions(quats: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive frames stay in the same hemisphere."""
    out = quats.copy()
    for idx in range(1, out.shape[0]):
        if np.dot(out[idx], out[idx - 1]) < 0.0:
            out[idx] = -out[idx]
    return out


def _weighted_quaternion_average(quats: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Markley et al. weighted quaternion average."""
    if len(quats) == 1:
        q = quats[0]
        return q / np.linalg.norm(q)

    aligned = quats.copy()
    reference = aligned[0]
    for idx in range(len(aligned)):
        if np.dot(aligned[idx], reference) < 0.0:
            aligned[idx] = -aligned[idx]

    matrix = np.zeros((4, 4), dtype=np.float64)
    for quat, weight in zip(aligned, weights):
        matrix += weight * np.outer(quat, quat)
    _, vectors = np.linalg.eigh(matrix)
    average = vectors[:, -1]
    return average / np.maximum(np.linalg.norm(average), 1e-8)


def _smooth_rotvec_segment(rotvecs: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth axis-angle trajectories with a Gaussian kernel in SO(3)."""
    if rotvecs.ndim == 2:
        rotvecs = rotvecs[:, None, :]
        squeeze = True
    else:
        squeeze = False

    num_frames, num_joints, _ = rotvecs.shape
    if num_frames < 2 or sigma <= 0:
        out = rotvecs.copy()
    else:
        quats = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat().reshape(num_frames, num_joints, 4)
        kernel, radius = _gaussian_kernel(sigma)
        smoothed = np.empty((num_frames, num_joints, 4), dtype=np.float64)
        for joint_idx in range(num_joints):
            joint_quats = _unwrap_quaternions(quats[:, joint_idx, :])
            for frame_idx in range(num_frames):
                lo = max(0, frame_idx - radius)
                hi = min(num_frames, frame_idx + radius + 1)
                window = joint_quats[lo:hi]
                window_offsets = np.arange(lo, hi) - frame_idx
                window_kernel = kernel[radius + window_offsets]
                window_kernel = window_kernel / window_kernel.sum()
                smoothed[frame_idx, joint_idx] = _weighted_quaternion_average(window, window_kernel)
        out = Rotation.from_quat(smoothed.reshape(-1, 4)).as_rotvec().reshape(num_frames, num_joints, 3)

    if squeeze:
        return out[:, 0, :]
    return out


def _smooth_hand_trajectory(
    *,
    pred_trans: np.ndarray,
    pred_rot: np.ndarray,
    pred_hand_pose: np.ndarray,
    pred_betas: np.ndarray,
    valid: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trans_out = pred_trans.copy()
    rot_out = pred_rot.copy()
    pose_out = pred_hand_pose.copy()
    betas_out = pred_betas.copy()

    for start, end in _valid_segments(valid):
        if end - start < 2:
            continue
        trans_out[start:end] = _smooth_euclidean_segment(pred_trans[start:end], sigma)
        rot_out[start:end] = _smooth_rotvec_segment(pred_rot[start:end], sigma)
        pose_out[start:end] = _smooth_rotvec_segment(
            pred_hand_pose[start:end].reshape(end - start, 15, 3),
            sigma,
        ).reshape(end - start, 45)

    return trans_out, rot_out, pose_out, betas_out


def smooth_world_poses(
    pred_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_hand_pose: torch.Tensor,
    pred_betas: torch.Tensor,
    pred_valid: torch.Tensor,
    *,
    sigma: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth observed world-space MANO params per hand on contiguous valid segments."""
    if sigma <= 0:
        return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid

    trans = np.array(pred_trans.detach().cpu().numpy(), copy=True)
    rot = np.array(pred_rot.detach().cpu().numpy(), copy=True)
    hand_pose = np.array(pred_hand_pose.detach().cpu().numpy(), copy=True)
    betas = np.array(pred_betas.detach().cpu().numpy(), copy=True)
    valid = pred_valid.detach().cpu().numpy() > 0

    for hand_idx in range(trans.shape[0]):
        hand_valid = valid[hand_idx]
        if not hand_valid.any():
            continue
        t, r, p, b = _smooth_hand_trajectory(
            pred_trans=trans[hand_idx],
            pred_rot=rot[hand_idx],
            pred_hand_pose=hand_pose[hand_idx],
            pred_betas=betas[hand_idx],
            valid=hand_valid,
            sigma=sigma,
        )
        trans[hand_idx] = t
        rot[hand_idx] = r
        hand_pose[hand_idx] = p
        betas[hand_idx] = b

    return (
        torch.from_numpy(trans).to(dtype=pred_trans.dtype),
        torch.from_numpy(rot).to(dtype=pred_rot.dtype),
        torch.from_numpy(hand_pose).to(dtype=pred_hand_pose.dtype),
        torch.from_numpy(betas).to(dtype=pred_betas.dtype),
        pred_valid,
    )


def smooth_slam_cameras(
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporally smooth SLAM world-to-camera trajectories for visualization."""
    if sigma <= 0:
        return R_w2c, t_w2c

    rotvecs = Rotation.from_matrix(R_w2c.cpu().numpy()).as_rotvec()
    trans = t_w2c.cpu().numpy()
    rot_smooth = _smooth_rotvec_segment(rotvecs, sigma)
    trans_smooth = _smooth_euclidean_segment(trans, sigma)
    return (
        torch.from_numpy(Rotation.from_rotvec(rot_smooth).as_matrix()).to(
            dtype=R_w2c.dtype, device=R_w2c.device
        ),
        torch.from_numpy(trans_smooth).to(dtype=t_w2c.dtype, device=t_w2c.device),
    )


def save_world_poses(
    path,
    pred_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_hand_pose: torch.Tensor,
    pred_betas: torch.Tensor,
    pred_valid: torch.Tensor,
) -> None:
    import joblib

    joblib.dump(
        [pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid],
        path,
    )
