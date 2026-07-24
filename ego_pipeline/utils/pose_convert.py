"""
Utilities for converting ViPE pose format to HaWoR SLAM format.

ViPE outputs:
  - cam_c2w: (N, 4, 4) camera-to-world 4x4 matrices
  - K: (3, 3) intrinsics

HaWoR expects (hawor_slam_w_scale_{start}_{end}.npz):
  - traj: (N, 7) [tx, ty, tz, qx, qy, qz, qw]
  - tstamp: (K,) keyframe indices
  - disps: (K, H, W) disparity maps
  - img_focal: float
  - img_center: (2,) [cx, cy]
  - scale: float (1.0 for ViPE since already metric)
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial.transform import Rotation


def c2w_to_traj7(poses_c2w: np.ndarray) -> np.ndarray:
    """Convert (N, 4, 4) c2w matrices to (N, 7) [t, q_xyzw] format.

    HaWoR's load_slam_cam() reads quaternions as [qx, qy, qz, qw] from
    traj[:, 3:] and then reorders to [qw, qx, qy, qz] before calling
    quaternion_to_matrix. So we store as [qx, qy, qz, qw] in the file.
    """
    N = len(poses_c2w)
    traj = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        R_mat = poses_c2w[i, :3, :3]
        t = poses_c2w[i, :3, 3]

        # scipy returns quaternions in xyzw order
        quat = Rotation.from_matrix(R_mat).as_quat()  # [qx, qy, qz, qw]

        traj[i, :3] = t
        traj[i, 3:] = quat  # [qx, qy, qz, qw]

    return traj


def save_hawor_slam_npz(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    save_path: str,
    start_idx: int = 0,
    end_idx: int | None = None,
):
    """Save ViPE outputs in HaWoR SLAM format.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world matrices
        intrinsics: (3, 3) K matrix
        save_path: path to .npz file
        start_idx: start frame index
        end_idx: end frame index
    """
    N = len(poses_c2w)
    if end_idx is None:
        end_idx = N

    traj = c2w_to_traj7(poses_c2w)

    fx = float(intrinsics[0, 0])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    # ViPE poses are already metric-scale, so scale=1.0.
    # The translation in traj is already the real-world metric value,
    # and load_slam_cam() will do: t_c2w = traj[:, :3] * scale.
    # With scale=1.0 this is a no-op.

    # Keyframe timestamps: use all frames as keyframes
    tstamp = np.arange(N, dtype=np.int32)

    # Disparity placeholder (HaWoR only reads disps from its own SLAM,
    # but the infiller doesn't use disps — it uses traj + scale)
    disps = np.ones((N, 1, 1), dtype=np.float32)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        traj=traj,
        tstamp=tstamp,
        disps=disps,
        img_focal=fx,
        img_center=np.array([cx, cy], dtype=np.float64),
        scale=1.0,
    )
    print(f"  Saved HaWoR SLAM npz -> {save_path}")
    print(f"    traj: {traj.shape}, focal={fx:.1f}, scale=1.0")


def write_est_focal(seq_folder: str, fx: float):
    """Write estimated focal length to est_focal.txt for HaWoR."""
    path = os.path.join(seq_folder, "est_focal.txt")
    with open(path, "w") as f:
        f.write(str(float(fx)))
    print(f"  Wrote est_focal.txt -> {path} (fx={fx:.1f})")
