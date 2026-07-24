#!/usr/bin/env python3
"""
Evaluate Ego Pipeline predictions against HOI4D Ground Truth.

Comparison modes:
  A) Camera-space MANO: debug/1_hawor/mano.npz vs gt/mano_gt.npz
  B) 2D reprojection:   MANO FK → project → vs GT kps2D
  C) Depth:             ViPE depth vs GT depth
  D) Camera params:     ViPE intrinsics + poses vs GT

Usage:
    python tools/eval_pipeline_vs_gt.py \
        --pred Output/PipelineOutput/HOI4D/ZY20210800001/H1/C1/N19/S100/s02/T1/ \
        --gt   Output/PipelineOutput/HOI4D/ZY20210800001/H1/C1/N19/S100/s02/T1/gt/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.dirname(SCRIPT_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def rotation_error_deg(aa_pred: np.ndarray, aa_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees between two axis-angle vectors."""
    R_pred = Rotation.from_rotvec(aa_pred).as_matrix()
    R_gt = Rotation.from_rotvec(aa_gt).as_matrix()
    R_diff = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.degrees(angle)


def align_scale(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """Find optimal scale s such that s * pred ≈ gt (least squares)."""
    p = pred[valid].flatten()
    g = gt[valid].flatten()
    mask = (g > 0) & (p > 0)
    if mask.sum() < 10:
        return 1.0
    s = np.median(g[mask] / p[mask])
    return float(s)


def align_pred_to_gt(pred_arr: np.ndarray, n_gt: int) -> np.ndarray:
    """Align prediction frames to GT frames.

    HaWoR extracts at 30fps from 15fps video → 2x frames.
    Subsample every 2nd frame to match GT frame rate.
    """
    # Determine the frame axis based on dimensionality
    # For MANO arrays: (2, N, ...) → axis 1 is frame axis
    # For 1D validity: (2, N) → axis 1
    if pred_arr.ndim >= 3:
        n_pred = pred_arr.shape[1]
    elif pred_arr.ndim == 2:
        n_pred = pred_arr.shape[1]
    else:
        n_pred = len(pred_arr)

    if n_pred == n_gt:
        return pred_arr

    ratio = n_pred / n_gt
    if abs(ratio - 2.0) < 0.1:  # 30fps vs 15fps
        indices = np.arange(0, n_pred, 2)[:n_gt]
        if pred_arr.ndim == 1:
            return pred_arr[indices]
        else:
            # For (2, N, ...) arrays, subsample along axis 1
            return pred_arr[:, indices]

    # Fallback: truncate
    if pred_arr.ndim >= 2:
        return pred_arr[:, :n_gt]
    return pred_arr[:n_gt]


# ═══════════════════════════════════════════════════════════════════════════════
#  A) Camera-Space MANO Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def eval_mano_camera_space(pred_dir: str, gt_dir: str) -> dict:
    """Compare MANO in camera space (before CoordUnify).

    Uses debug/1_hawor/mano.npz if available, otherwise world_space_res.pth
    with a warning.
    """
    print("\n" + "=" * 60)
    print("  [A] Camera-Space MANO Comparison")
    print("=" * 60)

    # Load GT
    gt_path = os.path.join(gt_dir, "mano_gt.npz")
    if not os.path.exists(gt_path):
        print("  ⚠️  GT MANO not found, skipping")
        return {}
    gt = np.load(gt_path)

    # Load predictions (prefer debug snapshot = camera space)
    hawor_debug = os.path.join(pred_dir, "debug", "0_hawor", "mano.npz")
    if os.path.exists(hawor_debug):
        print(f"  Using debug snapshot: {hawor_debug}")
        pred = np.load(hawor_debug)
        pred_trans = pred["trans"]   # (2, N, 3)
        pred_rot = pred["rot"]      # (2, N, 3)
        pred_pose = pred["pose"]    # (2, N, 45)
        pred_betas = pred["betas"]  # (2, N, 10)
        pred_valid = pred["valid"]  # (2, N)
    else:
        # Fallback: world_space_res.pth (Z-up, will warn about coord mismatch)
        pth_path = os.path.join(pred_dir, "world_space_res.pth")
        if not os.path.exists(pth_path):
            print("  ⚠️  No prediction MANO found, skipping")
            return {}
        print(f"  ⚠️  Using world_space_res.pth (Z-up space, coord mismatch with GT!)")
        import joblib
        pred_trans, pred_rot, pred_pose, pred_betas, pred_valid = joblib.load(pth_path)

    gt_trans = gt["trans"]      # (2, N_gt, 3)
    gt_rot = gt["rot"]          # (2, N_gt, 3)
    gt_pose = gt["pose"]        # (2, N_gt, 45)
    gt_betas = gt["betas"]      # (2, N_gt, 10)
    gt_valid = gt["valid"]      # (2, N_gt)
    n_gt = gt_trans.shape[1]

    # Align frame rates (pred may be 30fps, GT is 15fps)
    pred_trans = align_pred_to_gt(pred_trans, n_gt)
    pred_rot = align_pred_to_gt(pred_rot, n_gt)
    pred_pose = align_pred_to_gt(pred_pose, n_gt)
    pred_betas = align_pred_to_gt(pred_betas, n_gt)
    pred_valid = align_pred_to_gt(pred_valid, n_gt)
    print(f"  After alignment: pred={pred_trans.shape[1]} frames, gt={n_gt} frames")

    results = {}
    side_names = ["left", "right"]

    for h in range(2):
        # Valid frames = both pred and GT are valid
        both_valid = gt_valid[h].astype(bool)
        if pred_valid is not None:
            both_valid &= pred_valid[h].astype(bool)

        n_valid = both_valid.sum()
        if n_valid < 1:
            print(f"  {side_names[h]}: No valid frames, skipping")
            continue

        print(f"\n  {side_names[h]} hand ({n_valid} valid frames):")

        # ── Translation error (mm) ──────────────────────────────────────
        t_err = np.linalg.norm(
            pred_trans[h][both_valid] - gt_trans[h][both_valid], axis=1
        ) * 1000  # → mm
        results[f"{side_names[h]}_trans_mean_mm"] = float(t_err.mean())
        results[f"{side_names[h]}_trans_median_mm"] = float(np.median(t_err))
        results[f"{side_names[h]}_trans_std_mm"] = float(t_err.std())
        print(f"    Trans L2: mean={t_err.mean():.2f}mm, "
              f"median={np.median(t_err):.2f}mm, std={t_err.std():.2f}mm")

        # ── Global rotation error (degrees) ─────────────────────────────
        rot_errs = []
        for idx in np.where(both_valid)[0]:
            err = rotation_error_deg(pred_rot[h, idx], gt_rot[h, idx])
            rot_errs.append(err)
        rot_errs = np.array(rot_errs)
        results[f"{side_names[h]}_rot_mean_deg"] = float(rot_errs.mean())
        results[f"{side_names[h]}_rot_median_deg"] = float(np.median(rot_errs))
        print(f"    Rot error: mean={rot_errs.mean():.2f}°, "
              f"median={np.median(rot_errs):.2f}°")

        # ── Joint pose error (degrees, per joint) ───────────────────────
        joint_errs = []  # (n_valid, 15)
        for idx in np.where(both_valid)[0]:
            frame_errs = []
            for j in range(15):
                aa_pred = pred_pose[h, idx, j*3:(j+1)*3]
                aa_gt = gt_pose[h, idx, j*3:(j+1)*3]
                err = rotation_error_deg(aa_pred, aa_gt)
                frame_errs.append(err)
            joint_errs.append(frame_errs)
        joint_errs = np.array(joint_errs)  # (n_valid, 15)
        mean_joint = joint_errs.mean(axis=0)  # (15,)
        results[f"{side_names[h]}_pose_mean_deg"] = float(joint_errs.mean())
        results[f"{side_names[h]}_pose_per_joint_deg"] = mean_joint.tolist()
        print(f"    Pose error: overall mean={joint_errs.mean():.2f}°")
        print(f"    Per-joint: min={mean_joint.min():.2f}°, "
              f"max={mean_joint.max():.2f}°")

        # ── Beta error ──────────────────────────────────────────────────
        beta_err = np.linalg.norm(
            pred_betas[h][both_valid] - gt_betas[h][both_valid], axis=1
        )
        results[f"{side_names[h]}_beta_mean"] = float(beta_err.mean())
        print(f"    Beta L2: mean={beta_err.mean():.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  B) 2D Reprojection Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def eval_mano_2d_reproj(pred_dir: str, gt_dir: str) -> dict:
    """Compare predicted MANO via 2D reprojection against GT 2D keypoints.

    Requires MANO model to perform forward kinematics.
    Falls back to comparing GT kps2D if MANO FK is not available.
    """
    print("\n" + "=" * 60)
    print("  [B] 2D Reprojection Comparison")
    print("=" * 60)

    gt_path = os.path.join(gt_dir, "mano_gt.npz")
    if not os.path.exists(gt_path):
        print("  ⚠️  GT MANO not found, skipping")
        return {}
    gt = np.load(gt_path)
    gt_kps2D = gt["kps2D"]    # (2, N, 21, 2)
    gt_valid = gt["valid"]    # (2, N)

    # Try to load MANO model for FK
    try:
        sys.path.insert(0, os.path.join(BIV2AP_DIR, "third_party", "hawor"))
        from mano_wrapper import MANOWrapper
        has_mano = True
    except ImportError:
        has_mano = False
        print("  ⚠️  MANO model not available, using direct trans→project fallback")

    # Load predicted MANO (camera space preferred)
    hawor_debug = os.path.join(pred_dir, "debug", "0_hawor", "mano.npz")
    if not os.path.exists(hawor_debug):
        print("  ⚠️  No camera-space predictions, skipping 2D reproj")
        return {}

    pred = np.load(hawor_debug)

    # Load intrinsics for projection
    K_gt_path = os.path.join(gt_dir, "K_gt.npy")
    K_pred_path = os.path.join(pred_dir, "K.npy")
    K = np.load(K_gt_path) if os.path.exists(K_gt_path) else np.load(K_pred_path)

    results = {}
    side_names = ["left", "right"]

    n_gt = gt_valid.shape[1]
    pred_valid_aligned = align_pred_to_gt(pred["valid"], n_gt)
    pred_trans_aligned = align_pred_to_gt(pred["trans"], n_gt)

    for h in range(2):
        both_valid = gt_valid[h].astype(bool)
        if pred_valid_aligned is not None:
            both_valid &= pred_valid_aligned[h].astype(bool)

        n_valid = both_valid.sum()
        if n_valid < 1:
            continue

        # If we have MANO FK, use it; otherwise use translation as proxy
        if has_mano:
            # TODO: Full FK-based 2D reprojection
            # For now, project wrist (trans) as a proxy
            pass

        # Project predicted translation to 2D as a wrist proxy
        pred_trans_h = pred_trans_aligned[h][both_valid]  # (n_valid, 3)
        gt_kps = gt_kps2D[h][both_valid]           # (n_valid, 21, 2)

        # Project: p = K @ t → (u, v) = (p[0]/p[2], p[1]/p[2])
        p = (K @ pred_trans_h.T).T  # (n_valid, 3)
        pred_2d = p[:, :2] / p[:, 2:3]  # (n_valid, 2)

        # Compare with GT wrist (joint 0)
        gt_wrist = gt_kps[:, 0, :]  # (n_valid, 2)
        wrist_err = np.linalg.norm(pred_2d - gt_wrist, axis=1)

        results[f"{side_names[h]}_wrist_reproj_mean_px"] = float(wrist_err.mean())
        results[f"{side_names[h]}_wrist_reproj_median_px"] = float(np.median(wrist_err))
        pck10 = (wrist_err < 10).mean() * 100
        pck20 = (wrist_err < 20).mean() * 100
        results[f"{side_names[h]}_wrist_pck10"] = float(pck10)
        results[f"{side_names[h]}_wrist_pck20"] = float(pck20)

        print(f"\n  {side_names[h]} hand ({n_valid} valid):")
        print(f"    Wrist reproj: mean={wrist_err.mean():.1f}px, "
              f"median={np.median(wrist_err):.1f}px")
        print(f"    PCK@10px={pck10:.1f}%, PCK@20px={pck20:.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  C) Depth Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def eval_depth(pred_dir: str, gt_dir: str) -> dict:
    """Compare ViPE estimated depth vs HOI4D GT depth."""
    print("\n" + "=" * 60)
    print("  [C] Depth Comparison")
    print("=" * 60)

    # Load GT depth
    gt_depth_path = os.path.join(gt_dir, "depth_gt.npz")
    if not os.path.exists(gt_depth_path):
        print("  ⚠️  GT depth not found, skipping")
        return {}
    gt_depths = np.load(gt_depth_path)["depths"]  # (N, H, W)

    # Load predicted depth
    pred_depth_path = os.path.join(pred_dir, "depth.npz")
    if not os.path.exists(pred_depth_path):
        print("  ⚠️  Predicted depth not found, skipping")
        return {}
    pred_depths = np.load(pred_depth_path)["depths"]  # (N, H', W')

    n_gt, h_gt, w_gt = gt_depths.shape
    n_pred = pred_depths.shape[0]
    n_common = min(n_gt, n_pred)
    print(f"  GT: {gt_depths.shape}, Pred: {pred_depths.shape}")
    print(f"  Using {n_common} common frames")

    # Resize if needed
    if pred_depths.shape[1:] != gt_depths.shape[1:]:
        import cv2
        print(f"  Resizing predictions from {pred_depths.shape[1:]} to {gt_depths.shape[1:]}")
        resized = np.zeros((n_common, h_gt, w_gt), dtype=np.float32)
        for i in range(n_common):
            resized[i] = cv2.resize(
                pred_depths[i], (w_gt, h_gt), interpolation=cv2.INTER_LINEAR
            )
        pred_depths = resized

    # Valid mask: both GT and pred > 0
    gt_crop = gt_depths[:n_common]
    pred_crop = pred_depths[:n_common]
    valid = (gt_crop > 0) & (pred_crop > 0)
    n_valid = valid.sum()
    print(f"  Valid pixels: {n_valid} ({valid.mean()*100:.1f}%)")

    if n_valid < 100:
        print("  ⚠️  Too few valid pixels, skipping")
        return {}

    gt_v = gt_crop[valid]
    pred_v = pred_crop[valid]

    # ── Standard depth metrics ──────────────────────────────────────────
    abs_rel = np.mean(np.abs(pred_v - gt_v) / gt_v)
    rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    delta_125 = (ratio < 1.25).mean() * 100

    # ── Scale-invariant metrics ─────────────────────────────────────────
    scale = np.median(gt_v / pred_v)
    pred_aligned = pred_v * scale
    si_rmse = np.sqrt(np.mean((pred_aligned - gt_v) ** 2))
    si_abs_rel = np.mean(np.abs(pred_aligned - gt_v) / gt_v)

    results = {
        "depth_abs_rel": float(abs_rel),
        "depth_rmse_m": float(rmse),
        "depth_delta_125_pct": float(delta_125),
        "depth_scale_factor": float(scale),
        "depth_si_rmse_m": float(si_rmse),
        "depth_si_abs_rel": float(si_abs_rel),
    }

    print(f"\n  Raw metrics:")
    print(f"    Abs Rel:  {abs_rel:.4f}")
    print(f"    RMSE:     {rmse:.4f} m")
    print(f"    δ < 1.25: {delta_125:.1f}%")
    print(f"\n  Scale-aligned (scale={scale:.4f}):")
    print(f"    SI-RMSE:    {si_rmse:.4f} m")
    print(f"    SI-Abs Rel: {si_abs_rel:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  D) Camera Parameter Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def eval_camera_params(pred_dir: str, gt_dir: str) -> dict:
    """Compare ViPE intrinsics and camera poses vs GT."""
    print("\n" + "=" * 60)
    print("  [D] Camera Parameter Comparison")
    print("=" * 60)

    results = {}

    # ── Intrinsics ──────────────────────────────────────────────────────
    K_gt_path = os.path.join(gt_dir, "K_gt.npy")
    K_pred_path = os.path.join(pred_dir, "K.npy")

    if os.path.exists(K_gt_path) and os.path.exists(K_pred_path):
        K_gt = np.load(K_gt_path)
        K_pred = np.load(K_pred_path)

        fx_err = abs(K_pred[0, 0] - K_gt[0, 0]) / K_gt[0, 0] * 100
        fy_err = abs(K_pred[1, 1] - K_gt[1, 1]) / K_gt[1, 1] * 100
        cx_err = abs(K_pred[0, 2] - K_gt[0, 2])
        cy_err = abs(K_pred[1, 2] - K_gt[1, 2])

        results["intrinsics_fx_err_pct"] = float(fx_err)
        results["intrinsics_fy_err_pct"] = float(fy_err)
        results["intrinsics_cx_err_px"] = float(cx_err)
        results["intrinsics_cy_err_px"] = float(cy_err)

        print(f"\n  Intrinsics:")
        print(f"    GT:   fx={K_gt[0,0]:.1f}  fy={K_gt[1,1]:.1f}  "
              f"cx={K_gt[0,2]:.1f}  cy={K_gt[1,2]:.1f}")
        print(f"    Pred: fx={K_pred[0,0]:.1f}  fy={K_pred[1,1]:.1f}  "
              f"cx={K_pred[0,2]:.1f}  cy={K_pred[1,2]:.1f}")
        print(f"    fx err: {fx_err:.2f}%,  fy err: {fy_err:.2f}%")
        print(f"    cx err: {cx_err:.1f}px, cy err: {cy_err:.1f}px")
    else:
        print("  ⚠️  Intrinsics not available for comparison")

    # ── Camera Poses ────────────────────────────────────────────────────
    poses_gt_path = os.path.join(gt_dir, "poses_c2w_gt.npy")
    poses_pred_path = os.path.join(pred_dir, "cam_c2w.npy")

    if os.path.exists(poses_gt_path) and os.path.exists(poses_pred_path):
        poses_gt = np.load(poses_gt_path)    # (N, 4, 4)
        poses_pred = np.load(poses_pred_path)  # (N, 4, 4)

        n_gt = len(poses_gt)
        n_pred = len(poses_pred)
        n_common = min(n_gt, n_pred)
        print(f"\n  Camera Poses: GT={n_gt} frames, Pred={n_pred} frames, "
              f"using {n_common}")

        # Note: pred may be in Z-up after CoordUnify, GT is in OpenCV
        # Check debug snapshot for OpenCV-space poses
        debug_poses_path = os.path.join(pred_dir, "debug", "0_vipe", "poses_c2w.npy")
        if os.path.exists(debug_poses_path):
            poses_pred = np.load(debug_poses_path)
            print(f"  Using pre-CoordUnify poses from debug snapshot")
            n_pred = len(poses_pred)
            n_common = min(n_gt, n_pred)

        # ── ATE (Absolute Trajectory Error) ─────────────────────────────
        # Align with Umeyama (similarity transform)
        gt_traj = poses_gt[:n_common, :3, 3]
        pred_traj = poses_pred[:n_common, :3, 3]

        # Simple scale-align via Umeyama
        ate_raw = np.linalg.norm(pred_traj - gt_traj, axis=1)

        # Umeyama alignment (translation + rotation + scale)
        gt_mean = gt_traj.mean(axis=0)
        pred_mean = pred_traj.mean(axis=0)
        gt_c = gt_traj - gt_mean
        pred_c = pred_traj - pred_mean

        H = pred_c.T @ gt_c
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1, 1, d])
        R_align = Vt.T @ D @ U.T
        scale_align = np.trace(R_align @ H) / np.trace(pred_c.T @ pred_c)
        t_align = gt_mean - scale_align * R_align @ pred_mean

        pred_aligned = (scale_align * (R_align @ pred_traj.T).T + t_align)
        ate_aligned = np.linalg.norm(pred_aligned - gt_traj, axis=1)

        results["pose_ate_raw_mean_m"] = float(ate_raw.mean())
        results["pose_ate_raw_median_m"] = float(np.median(ate_raw))
        results["pose_ate_aligned_mean_m"] = float(ate_aligned.mean())
        results["pose_ate_aligned_median_m"] = float(np.median(ate_aligned))
        results["pose_align_scale"] = float(scale_align)

        print(f"\n  ATE (raw):")
        print(f"    mean={ate_raw.mean()*1000:.2f}mm, "
              f"median={np.median(ate_raw)*1000:.2f}mm")
        print(f"  ATE (Umeyama-aligned, scale={scale_align:.4f}):")
        print(f"    mean={ate_aligned.mean()*1000:.2f}mm, "
              f"median={np.median(ate_aligned)*1000:.2f}mm")

        # ── RPE (Relative Pose Error) ───────────────────────────────────
        rpe_trans = []
        rpe_rot = []
        for i in range(n_common - 1):
            # Relative transform: T_{i+1} @ T_i^{-1}
            gt_rel = poses_gt[i + 1] @ np.linalg.inv(poses_gt[i])
            pred_rel = poses_pred[i + 1] @ np.linalg.inv(poses_pred[i])

            # Translation error
            dt = np.linalg.norm(pred_rel[:3, 3] - gt_rel[:3, 3])
            rpe_trans.append(dt)

            # Rotation error
            R_diff = pred_rel[:3, :3] @ gt_rel[:3, :3].T
            trace = np.clip(np.trace(R_diff), -1.0, 3.0)
            angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
            rpe_rot.append(np.degrees(angle))

        rpe_trans = np.array(rpe_trans)
        rpe_rot = np.array(rpe_rot)

        results["pose_rpe_trans_mean_mm"] = float(rpe_trans.mean() * 1000)
        results["pose_rpe_rot_mean_deg"] = float(rpe_rot.mean())

        print(f"\n  RPE (frame-to-frame):")
        print(f"    Trans: mean={rpe_trans.mean()*1000:.3f}mm")
        print(f"    Rot:   mean={rpe_rot.mean():.3f}°")
    else:
        print("  ⚠️  Camera poses not available for comparison")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def save_visualizations(pred_dir: str, gt_dir: str, eval_dir: str) -> None:
    """Generate comparison visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib not available, skipping visualizations")
        return

    # ── Depth comparison (sample frames) ────────────────────────────────
    gt_depth_path = os.path.join(gt_dir, "depth_gt.npz")
    pred_depth_path = os.path.join(pred_dir, "depth.npz")

    if os.path.exists(gt_depth_path) and os.path.exists(pred_depth_path):
        gt_depths = np.load(gt_depth_path)["depths"]
        pred_depths = np.load(pred_depth_path)["depths"]
        n_common = min(len(gt_depths), len(pred_depths))

        sample_frames = [0, n_common // 4, n_common // 2,
                         3 * n_common // 4, n_common - 1]

        fig, axes = plt.subplots(3, len(sample_frames), figsize=(20, 10))

        for col, fi in enumerate(sample_frames):
            gt_d = gt_depths[fi]
            pred_d = pred_depths[fi]

            # Resize pred if needed
            if pred_d.shape != gt_d.shape:
                import cv2
                pred_d = cv2.resize(pred_d, (gt_d.shape[1], gt_d.shape[0]))

            vmax = np.percentile(gt_d[gt_d > 0], 95) if (gt_d > 0).any() else 3.0

            axes[0, col].imshow(gt_d, cmap="turbo", vmin=0, vmax=vmax)
            axes[0, col].set_title(f"GT frame {fi}")
            axes[0, col].axis("off")

            axes[1, col].imshow(pred_d, cmap="turbo", vmin=0, vmax=vmax)
            axes[1, col].set_title(f"Pred frame {fi}")
            axes[1, col].axis("off")

            # Difference
            valid = (gt_d > 0) & (pred_d > 0)
            diff = np.zeros_like(gt_d)
            if valid.any():
                # Scale-align for fair comparison
                scale = np.median(gt_d[valid] / pred_d[valid])
                diff[valid] = np.abs(pred_d[valid] * scale - gt_d[valid])

            axes[2, col].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
            axes[2, col].set_title(f"|Δ| frame {fi}")
            axes[2, col].axis("off")

        fig.suptitle("Depth Comparison: GT vs ViPE", fontsize=14)
        plt.tight_layout()
        path = os.path.join(eval_dir, "depth_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✅ Saved {path}")

    # ── Trajectory comparison ───────────────────────────────────────────
    poses_gt_path = os.path.join(gt_dir, "poses_c2w_gt.npy")
    debug_poses_path = os.path.join(pred_dir, "debug", "0_vipe", "poses_c2w.npy")
    poses_pred_path = os.path.join(pred_dir, "cam_c2w.npy")

    pp = debug_poses_path if os.path.exists(debug_poses_path) else poses_pred_path
    if os.path.exists(poses_gt_path) and os.path.exists(pp):
        poses_gt = np.load(poses_gt_path)
        poses_pred = np.load(pp)
        n = min(len(poses_gt), len(poses_pred))

        fig = plt.figure(figsize=(12, 5))

        # XZ plane (top-down)
        ax1 = fig.add_subplot(121)
        ax1.plot(poses_gt[:n, 0, 3], poses_gt[:n, 2, 3], "b-", label="GT", alpha=0.7)
        ax1.plot(poses_pred[:n, 0, 3], poses_pred[:n, 2, 3], "r--", label="Pred", alpha=0.7)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Z (m)")
        ax1.set_title("Trajectory (XZ plane)")
        ax1.legend()
        ax1.set_aspect("equal")

        # 3D
        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot(poses_gt[:n, 0, 3], poses_gt[:n, 1, 3], poses_gt[:n, 2, 3],
                 "b-", label="GT", alpha=0.7)
        ax2.plot(poses_pred[:n, 0, 3], poses_pred[:n, 1, 3], poses_pred[:n, 2, 3],
                 "r--", label="Pred", alpha=0.7)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.set_title("Trajectory 3D")
        ax2.legend()

        fig.suptitle("Camera Trajectory: GT vs ViPE", fontsize=14)
        plt.tight_layout()
        path = os.path.join(eval_dir, "trajectory_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  ✅ Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ego Pipeline vs HOI4D Ground Truth"
    )
    parser.add_argument(
        "--pred", required=True,
        help="Pipeline output directory"
    )
    parser.add_argument(
        "--gt", required=True,
        help="GT directory (from prepare_gt_for_eval.py)"
    )
    parser.add_argument(
        "--eval-dir", default=None,
        help="Evaluation output directory (default: <pred>/eval/)"
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir or os.path.join(args.pred, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Ego Pipeline Evaluation")
    print(f"  Pred: {args.pred}")
    print(f"  GT:   {args.gt}")
    print(f"  Eval: {eval_dir}")
    print(f"{'='*60}")

    all_metrics = {}

    # A) Camera-space MANO
    mano_metrics = eval_mano_camera_space(args.pred, args.gt)
    all_metrics.update(mano_metrics)

    # B) 2D Reprojection
    reproj_metrics = eval_mano_2d_reproj(args.pred, args.gt)
    all_metrics.update(reproj_metrics)

    # C) Depth
    depth_metrics = eval_depth(args.pred, args.gt)
    all_metrics.update(depth_metrics)

    # D) Camera params
    cam_metrics = eval_camera_params(args.pred, args.gt)
    all_metrics.update(cam_metrics)

    # Save metrics
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✅ Metrics saved: {metrics_path}")

    # Visualizations
    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60)
    save_visualizations(args.pred, args.gt, eval_dir)

    # Summary table
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for k, v in sorted(all_metrics.items()):
        if isinstance(v, float):
            print(f"  {k:40s}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k:40s}: [{len(v)} values]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
