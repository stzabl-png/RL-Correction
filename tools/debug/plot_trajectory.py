#!/usr/bin/env python3
"""3D trajectory plot: camera + left/right wrist.

Usage:
    python tools/debug/plot_trajectory.py --dir output/debug/3_smooth/
    python tools/debug/plot_trajectory.py --dir output/debug/3_smooth/ --save traj.png
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(d, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Camera trajectory
    poses_path = os.path.join(d, "poses_c2w.npy")
    if os.path.exists(poses_path):
        poses = np.load(poses_path)
        cam_t = poses[:, :3, 3]
        ax.plot(cam_t[:, 0], cam_t[:, 1], cam_t[:, 2],
                "g-", alpha=0.6, linewidth=1, label="camera")
        ax.scatter(*cam_t[0], c="green", s=50, marker="^", zorder=5)

    # MANO wrist trajectories
    mano_path = os.path.join(d, "mano.npz")
    if os.path.exists(mano_path):
        data = np.load(mano_path)
        colors = ["dodgerblue", "salmon"]
        names = ["left wrist", "right wrist"]
        for h in range(2):
            valid = data["valid"][h].astype(bool)
            if valid.sum() < 2:
                continue
            t = data["trans"][h]
            t_valid = t[valid]
            ax.plot(t_valid[:, 0], t_valid[:, 1], t_valid[:, 2],
                    "-", color=colors[h], alpha=0.7, linewidth=1.5,
                    label=names[h])
            ax.scatter(*t_valid[0], c=colors[h], s=40, marker="o", zorder=5)

    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (left)")
    ax.set_zlabel("Z (up)")
    ax.set_title("Ego Pipeline Trajectories (Z-up)")
    ax.legend()

    # Equal aspect ratio
    all_pts = []
    if os.path.exists(poses_path):
        all_pts.append(cam_t)
    if os.path.exists(mano_path):
        for h in range(2):
            valid = data["valid"][h].astype(bool)
            if valid.sum() > 0:
                all_pts.append(data["trans"][h][valid])
    if all_pts:
        pts = np.concatenate(all_pts)
        center = pts.mean(axis=0)
        max_range = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 * 1.1
        for setter, c in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
            setter([c - max_range, c + max_range])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--save", default=None)
    args = p.parse_args()
    plot(args.dir, args.save)
