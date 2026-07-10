#!/usr/bin/env python3
"""One-off MANO -> Sharpa Wave wrist-frame calibration for anchored BODex.

Kabsch-fits the fixed rigid transform between flat-pose Sharpa fingertip/pad
points (FK at q=0, base_link frame) and flat-pose MANO keypoints expressed in
the keypoint-derived wrist frame, then writes
``assets/robots/hands/sharpa_wave/grasp_synthesis/bodex/mano_transfer.yml``
and an overlay PNG for eyeballing.

Hard-fails if the mean residual exceeds ``--max-mean-residual-mm`` (default
25 mm). The gate exists to catch convention errors -- a palm-normal sign flip
or thumb-side swap shows up as a 50-100+ mm residual. Genuine MANO-vs-Sharpa
morphology mismatch (finger length/spacing; the measured flat-pose value is
~19 mm mean, dominated by the pinky) sits well below that and is absorbed
later by the per-frame IK wrist correction + joint fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_MANO_PKL = Path("/data/users/hangkes2/OCIR/raw_data/mano_models/MANO_RIGHT.pkl")


def mano_flat_keypoints(mano_pkl: Path) -> np.ndarray:
    """Flat-pose (all-zero axis-angle, zero betas) MANO 21 keypoints.

    Uses the same fingertip-vertex + reorder convention as
    ``pose_m_to_vertices_and_joints`` so keypoint indices line up with the
    exported demo trajectories.
    """

    from ocir.dexycb.mano_model import load_mano_model, mano_lbs

    model = load_mano_model(mano_pkl, side="right")
    vertices, joints16 = mano_lbs(model, np.zeros((16, 3)), np.zeros(10))
    tip_indices = [745, 317, 444, 556, 673]
    joints21 = np.concatenate([joints16, vertices[tip_indices]], axis=0)
    return joints21[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]


def write_overlay_png(path: Path, sharpa_aligned: np.ndarray, mano_points: np.ndarray, labels: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*mano_points.T, c="tab:blue", s=40, label="MANO (wrist frame)")
    ax.scatter(*sharpa_aligned.T, c="tab:red", s=40, marker="^", label="Sharpa (aligned)")
    for p, q in zip(sharpa_aligned, mano_points):
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], c="gray", lw=0.8)
    for p, label in zip(mano_points, labels):
        ax.text(p[0], p[1], p[2], label, fontsize=6)
    ax.legend()
    ax.set_title("MANO-Sharpa flat-pose calibration")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano-pkl", type=Path, default=DEFAULT_MANO_PKL)
    parser.add_argument("--asset-config", type=Path, default=None)
    parser.add_argument("--max-mean-residual-mm", type=float, default=25.0)
    parser.add_argument("--out", type=Path, default=None, help="Override the mano_transfer.yml output path.")
    args = parser.parse_args()

    import torch

    from ocir.grasp_synthesis.assets import load_sharpa_wave_right
    from ocir.grasp_synthesis.anchored_bodex.retarget import (
        compute_calibration,
        save_mano_transfer,
        transfer_path,
    )
    from curobo._src.types.device_cfg import DeviceCfg

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device_cfg = DeviceCfg(device=device, dtype=torch.float32)
    asset = load_sharpa_wave_right(args.asset_config)

    mano_keypoints = mano_flat_keypoints(args.mano_pkl.expanduser())
    calib, report = compute_calibration(asset, mano_keypoints, device_cfg)

    print("per-point residuals (mm):")
    for name, value in report["per_point_residual_mm"].items():
        print(f"  {name}: {value:.1f}")
    print(f"mean residual: {report['mean_residual_mm']:.1f} mm, max: {report['max_residual_mm']:.1f} mm")

    out_path = args.out if args.out is not None else transfer_path(asset)
    png_path = out_path.with_suffix(".calibration.png")

    # Rebuild overlay data for the PNG.
    from ocir.grasp_synthesis.anchored_bodex.demo_analysis import wrist_frame_from_keypoints
    from ocir.grasp_synthesis.anchored_bodex.retarget import HandFitter, ManoTransferCalib

    fitter = HandFitter(asset, calib, device_cfg)
    q_flat = torch.zeros((1, len(fitter.joint_names)), device=device, dtype=torch.float32)
    q_flat = torch.clamp(q_flat, fitter.joint_lower.view(1, -1), fitter.joint_upper.view(1, -1))
    with torch.no_grad():
        sharpa_points = fitter._fk_keypoints_base(q_flat)[0].cpu().numpy()
    aligned = sharpa_points @ calib.wrist_rot.T + calib.wrist_trans
    inv = np.linalg.inv(wrist_frame_from_keypoints(mano_keypoints))
    mano_wrist = mano_keypoints[[t.mano_joint for t in calib.keypoints]] @ inv[:3, :3].T + inv[:3, 3]
    labels = [f"m{t.mano_joint}" for t in calib.keypoints]
    write_overlay_png(png_path, aligned, mano_wrist, labels)
    print(f"wrote overlay {png_path}")

    if report["mean_residual_mm"] > args.max_mean_residual_mm:
        print(
            f"FAIL: mean residual {report['mean_residual_mm']:.1f} mm exceeds "
            f"{args.max_mean_residual_mm:.1f} mm -- check palm-normal sign / keypoint map before using this calibration"
        )
        return 1

    save_mano_transfer(out_path, calib, report)
    print(f"wrote calibration {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
