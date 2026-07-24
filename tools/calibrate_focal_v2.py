#!/usr/bin/env python3
"""
Focal Length Self-Calibration via Feature-Based Reprojection.

Uses sparse feature matches between frames, GT/estimated depth at matched
pixels, and known camera poses to solve for the focal length that minimises
3D triangulation error.

For each feature match (u_i, v_i) <-> (u_j, v_j) with depths d_i, d_j:
  - Unproject both to 3D using candidate fx
  - Transform both to world coordinates using their camera poses
  - The correct fx minimises ||P_world_i - P_world_j||

This is much more sensitive to fx than depth-consistency because the
LATERAL 3D coordinates scale as 1/fx.

Usage:
    python calibrate_focal_v2.py \
        --input /path/to/pipeline/output \
        --video /path/to/video.mp4 \
        --gt-fx 1376.8 \
        --use-gt \
        --visualize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import minimize_scalar


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_pipeline_outputs(input_dir: str) -> dict:
    """Load pipeline outputs."""
    # K
    for k_path in [
        os.path.join(input_dir, "debug", "0_vipe", "K.npy"),
        os.path.join(input_dir, "K.npy"),
    ]:
        if os.path.exists(k_path):
            K = np.load(k_path)
            break
    else:
        raise FileNotFoundError(f"K.npy not found under {input_dir}")

    # Poses
    for p_path in [
        os.path.join(input_dir, "debug", "0_vipe", "poses_c2w.npy"),
        os.path.join(input_dir, "cam_c2w.npy"),
    ]:
        if os.path.exists(p_path):
            poses_c2w = np.load(p_path)
            break
    else:
        raise FileNotFoundError(f"poses_c2w not found")

    # Depth
    depth_path = os.path.join(input_dir, "depth.npz")
    depth_data = np.load(depth_path)
    depths = depth_data.get("depths", depth_data[list(depth_data.keys())[0]])

    # GT (optional)
    gt_depth_path = os.path.join(input_dir, "gt", "depth_gt.npz")
    gt_depths = None
    if os.path.exists(gt_depth_path):
        gd = np.load(gt_depth_path)
        gt_depths = gd.get("depths", gd[list(gd.keys())[0]])

    gt_poses_path = os.path.join(input_dir, "gt", "poses_c2w_gt.npy")
    gt_poses = np.load(gt_poses_path) if os.path.exists(gt_poses_path) else None

    print(f"Loaded: K={K.shape}, poses={poses_c2w.shape}, depths={depths.shape}")
    print(f"  fx_init={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

    return {
        "K": K, "poses_c2w": poses_c2w, "depths": depths,
        "gt_depths": gt_depths, "gt_poses": gt_poses,
    }


def load_video_frames(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """Load specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = {}
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Matching
# ═══════════════════════════════════════════════════════════════════════════════

def find_matches(
    img_i: np.ndarray,
    img_j: np.ndarray,
    depth_i: np.ndarray,
    depth_j: np.ndarray,
    max_matches: int = 500,
    margin: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find feature matches between two frames with valid depth.

    Returns:
        pts_i: (M, 2) pixel coords in frame i
        pts_j: (M, 2) pixel coords in frame j
        d_i:   (M,) depth at pts_i
        d_j:   (M,) depth at pts_j
    """
    H, W = img_i.shape[:2]

    # Use SIFT for robust matching
    sift = cv2.SIFT_create(nfeatures=2000)
    kp_i, desc_i = sift.detectAndCompute(img_i, None)
    kp_j, desc_j = sift.detectAndCompute(img_j, None)

    if desc_i is None or desc_j is None or len(kp_i) < 10 or len(kp_j) < 10:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    # BFMatcher with ratio test
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(desc_i, desc_j, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.7 * n.distance:
                good.append(m)

    if len(good) < 10:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0)

    # Extract coordinates
    pts_i = np.array([kp_i[m.queryIdx].pt for m in good])  # (N, 2) as (u, v)
    pts_j = np.array([kp_j[m.trainIdx].pt for m in good])

    # Filter: margin check
    valid = (
        (pts_i[:, 0] >= margin) & (pts_i[:, 0] < W - margin) &
        (pts_i[:, 1] >= margin) & (pts_i[:, 1] < H - margin) &
        (pts_j[:, 0] >= margin) & (pts_j[:, 0] < W - margin) &
        (pts_j[:, 1] >= margin) & (pts_j[:, 1] < H - margin)
    )
    pts_i, pts_j = pts_i[valid], pts_j[valid]

    # Get depth (nearest pixel)
    ui, vi = pts_i[:, 0].astype(int), pts_i[:, 1].astype(int)
    uj, vj = pts_j[:, 0].astype(int), pts_j[:, 1].astype(int)
    d_i = depth_i[vi, ui].astype(np.float64)
    d_j = depth_j[vj, uj].astype(np.float64)

    # Filter: valid depth
    valid_d = (d_i > 0.05) & (d_i < 10.0) & (d_j > 0.05) & (d_j < 10.0)
    pts_i, pts_j = pts_i[valid_d], pts_j[valid_d]
    d_i, d_j = d_i[valid_d], d_j[valid_d]

    # Limit matches
    if len(pts_i) > max_matches:
        idx = np.random.choice(len(pts_i), max_matches, replace=False)
        pts_i, pts_j, d_i, d_j = pts_i[idx], pts_j[idx], d_i[idx], d_j[idx]

    return pts_i, pts_j, d_i, d_j


# ═══════════════════════════════════════════════════════════════════════════════
#  Core: 3D Triangulation Error
# ═══════════════════════════════════════════════════════════════════════════════

def triangulation_error(
    fx: float,
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    d_i: np.ndarray,
    d_j: np.ndarray,
    T_i_c2w: np.ndarray,
    T_j_c2w: np.ndarray,
    cx: float,
    cy: float,
) -> float:
    """Compute 3D triangulation error for a candidate fx.

    Unproject matched pixels from both frames into world space.
    The correct fx minimises the 3D distance between the two sets of points.
    """
    # Unproject frame i → camera coords
    Xi = (pts_i[:, 0] - cx) * d_i / fx
    Yi = (pts_i[:, 1] - cy) * d_i / fx
    Zi = d_i
    P_cam_i = np.stack([Xi, Yi, Zi, np.ones_like(Zi)], axis=1)  # (M, 4)

    # Unproject frame j → camera coords
    Xj = (pts_j[:, 0] - cx) * d_j / fx
    Yj = (pts_j[:, 1] - cy) * d_j / fx
    Zj = d_j
    P_cam_j = np.stack([Xj, Yj, Zj, np.ones_like(Zj)], axis=1)  # (M, 4)

    # Transform to world
    P_world_i = (T_i_c2w @ P_cam_i.T).T[:, :3]  # (M, 3)
    P_world_j = (T_j_c2w @ P_cam_j.T).T[:, :3]  # (M, 3)

    # 3D distance
    dists = np.linalg.norm(P_world_i - P_world_j, axis=1)

    # Robust: median
    return float(np.median(dists))


def reprojection_error(
    fx: float,
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    d_i: np.ndarray,
    T_ij: np.ndarray,
    cx: float,
    cy: float,
) -> float:
    """Compute 2D reprojection error for a candidate fx.

    Unproject frame i pixels, transform to frame j, project, compare with actual j pixels.
    """
    # Unproject frame i
    Xi = (pts_i[:, 0] - cx) * d_i / fx
    Yi = (pts_i[:, 1] - cy) * d_i / fx
    Zi = d_i
    P_i = np.stack([Xi, Yi, Zi], axis=1)  # (M, 3)

    R = T_ij[:3, :3]
    t = T_ij[:3, 3]
    P_j = (R @ P_i.T).T + t  # (M, 3)

    # Project to frame j
    valid = P_j[:, 2] > 0.01
    if valid.sum() < 5:
        return float("inf")

    u_j_pred = fx * P_j[valid, 0] / P_j[valid, 2] + cx
    v_j_pred = fx * P_j[valid, 1] / P_j[valid, 2] + cy

    # 2D error
    errs = np.sqrt(
        (u_j_pred - pts_j[valid, 0]) ** 2 +
        (v_j_pred - pts_j[valid, 1]) ** 2
    )

    return float(np.median(errs))


# ═══════════════════════════════════════════════════════════════════════════════
#  Analytic Focal Length from Single Correspondence
# ═══════════════════════════════════════════════════════════════════════════════

def solve_fx_quadratic(
    u_i: float, v_i: float, d_i: float,
    u_j: float, v_j: float,
    R: np.ndarray, t: np.ndarray,
    cx: float, cy: float,
) -> Optional[float]:
    """Solve for fx analytically from a single correspondence.

    From the projection equation u_j = fx * P_j[0]/P_j[2] + cx, we get:
        B*fx² + (d_i*A - a_j*D)*fx - a_j*d_i*C = 0
    """
    a_i = u_i - cx
    b_i = v_i - cy
    a_j = u_j - cx

    A = R[0, 0] * a_i + R[0, 1] * b_i
    B = d_i * R[0, 2] + t[0]
    C = R[2, 0] * a_i + R[2, 1] * b_i
    D = d_i * R[2, 2] + t[2]

    # Quadratic: B*f^2 + (d_i*A - a_j*D)*f - a_j*d_i*C = 0
    qa = B
    qb = d_i * A - a_j * D
    qc = -a_j * d_i * C

    if abs(qa) < 1e-12:
        # Linear: qb*f + qc = 0
        if abs(qb) < 1e-12:
            return None
        f = -qc / qb
        return f if f > 0 else None

    disc = qb ** 2 - 4 * qa * qc
    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    f1 = (-qb + sqrt_disc) / (2 * qa)
    f2 = (-qb - sqrt_disc) / (2 * qa)

    # Return positive root
    candidates = [f for f in [f1, f2] if f > 100 and f < 5000]
    if not candidates:
        return None
    # If both positive, prefer the one closer to typical values
    return min(candidates, key=lambda f: abs(f - 1000))


# ═══════════════════════════════════════════════════════════════════════════════
#  Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_focal(
    video_path: str,
    depths: np.ndarray,
    poses_c2w: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
    n_pairs: int = 20,
    verbose: bool = True,
) -> dict:
    """Optimize focal length using feature matching + reprojection error."""
    t0 = time.time()
    N = len(depths)

    # Select frame pairs with variety
    gaps = [5, 10, 15, 20, 30, 40, 50]
    pair_list = []
    for gap in gaps:
        for start in range(0, N - gap, max(1, gap)):
            i, j = start, start + gap
            if j < N:
                baseline = np.linalg.norm(poses_c2w[j, :3, 3] - poses_c2w[i, :3, 3])
                if baseline > 0.003:
                    pair_list.append((i, j, baseline))
    pair_list.sort(key=lambda x: x[2], reverse=True)
    pairs = [(i, j) for i, j, _ in pair_list[:n_pairs]]

    if verbose:
        print(f"\nSelected {len(pairs)} frame pairs")

    # Load needed frames
    frame_indices = sorted(set(i for pair in pairs for i in pair))
    if verbose:
        print(f"Loading {len(frame_indices)} video frames...")
    frames = load_video_frames(video_path, frame_indices)
    if verbose:
        print(f"  Loaded {len(frames)} frames")

    # Find matches for all pairs
    np.random.seed(42)
    all_matches = []  # list of (pts_i, pts_j, d_i, d_j, T_i_c2w, T_j_c2w)
    total_matches = 0

    for i, j in pairs:
        if i not in frames or j not in frames:
            continue
        pts_i, pts_j, d_i, d_j = find_matches(
            frames[i], frames[j], depths[i], depths[j], max_matches=300
        )
        if len(pts_i) >= 10:
            all_matches.append((pts_i, pts_j, d_i, d_j, poses_c2w[i], poses_c2w[j]))
            total_matches += len(pts_i)

    if verbose:
        print(f"  Total matches: {total_matches} across {len(all_matches)} pairs")

    if len(all_matches) < 3:
        print("ERROR: Too few valid matches!")
        return {"fx_init": fx_init, "fx_optimized": fx_init, "error": "too few matches"}

    # ── Method 1: Analytic solution (per-match fx estimates) ──
    fx_estimates = []
    for pts_i, pts_j, d_i, d_j, T_i_c2w, T_j_c2w in all_matches:
        T_ij = np.linalg.inv(T_j_c2w) @ T_i_c2w
        R = T_ij[:3, :3]
        t_vec = T_ij[:3, 3]
        for k in range(len(pts_i)):
            f_est = solve_fx_quadratic(
                pts_i[k, 0], pts_i[k, 1], d_i[k],
                pts_j[k, 0], pts_j[k, 1],
                R, t_vec, cx, cy,
            )
            if f_est is not None:
                fx_estimates.append(f_est)

    fx_analytic = float(np.median(fx_estimates)) if fx_estimates else fx_init
    fx_analytic_std = float(np.std(fx_estimates)) if fx_estimates else 0
    if verbose:
        print(f"\n[Analytic] {len(fx_estimates)} estimates, median={fx_analytic:.2f}, std={fx_analytic_std:.2f}")

    # ── Method 2: Reprojection error minimisation ──
    def total_reproj_error(fx):
        errors = []
        for pts_i, pts_j, d_i, d_j, T_i_c2w, T_j_c2w in all_matches:
            T_ij = np.linalg.inv(T_j_c2w) @ T_i_c2w
            err = reprojection_error(fx, pts_i, pts_j, d_i, T_ij, cx, cy)
            if np.isfinite(err):
                errors.append(err)
        return float(np.median(errors)) if errors else float("inf")

    def total_triang_error(fx):
        errors = []
        for pts_i, pts_j, d_i, d_j, T_i_c2w, T_j_c2w in all_matches:
            err = triangulation_error(fx, pts_i, pts_j, d_i, d_j, T_i_c2w, T_j_c2w, cx, cy)
            if np.isfinite(err):
                errors.append(err)
        return float(np.median(errors)) if errors else float("inf")

    # Coarse search
    if verbose:
        print(f"\n[Reprojection] Coarse search: fx ∈ [400, 2500], step=10")
    fx_range = np.arange(400, 2501, 10)
    errs_reproj = np.array([total_reproj_error(fx) for fx in fx_range])
    errs_triang = np.array([total_triang_error(fx) for fx in fx_range])

    best_reproj_idx = np.argmin(errs_reproj)
    best_triang_idx = np.argmin(errs_triang)
    fx_reproj_coarse = fx_range[best_reproj_idx]
    fx_triang_coarse = fx_range[best_triang_idx]
    if verbose:
        print(f"  Reproj best:  fx={fx_reproj_coarse:.0f}, error={errs_reproj[best_reproj_idx]:.4f} px")
        print(f"  Triang best:  fx={fx_triang_coarse:.0f}, error={errs_triang[best_triang_idx]:.6f} m")

    # Fine search around reprojection best
    lo = max(400, fx_reproj_coarse - 80)
    hi = min(2500, fx_reproj_coarse + 80)
    if verbose:
        print(f"\n[Reprojection] Fine search: fx ∈ [{lo:.0f}, {hi:.0f}], step=1")
    fx_fine = np.arange(lo, hi + 1, 1)
    errs_fine = np.array([total_reproj_error(fx) for fx in fx_fine])
    best_fine_idx = np.argmin(errs_fine)
    fx_reproj_fine = fx_fine[best_fine_idx]
    if verbose:
        print(f"  Best: fx={fx_reproj_fine:.0f}, error={errs_fine[best_fine_idx]:.4f} px")

    # Scipy refinement
    result = minimize_scalar(
        total_reproj_error,
        bounds=(max(400, fx_reproj_fine - 10), min(2500, fx_reproj_fine + 10)),
        method="bounded",
        options={"xatol": 0.01},
    )
    fx_reproj_opt = result.x

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Final (reproj):   fx={fx_reproj_opt:.2f}")
        print(f"  Final (analytic): fx={fx_analytic:.2f}")
        print(f"  Total time: {elapsed:.1f}s")

    return {
        "fx_init": float(fx_init),
        "fx_reproj": float(fx_reproj_opt),
        "fx_analytic": float(fx_analytic),
        "fx_analytic_std": float(fx_analytic_std),
        "fx_triang": float(fx_triang_coarse),
        "n_matches": total_matches,
        "n_pairs": len(all_matches),
        "n_analytic_estimates": len(fx_estimates),
        "elapsed_sec": elapsed,
        "reproj_error_at_init": float(total_reproj_error(fx_init)),
        "reproj_error_at_opt": float(total_reproj_error(fx_reproj_opt)),
        "coarse_curve": (fx_range.tolist(), errs_reproj.tolist()),
        "coarse_triang_curve": (fx_range.tolist(), errs_triang.tolist()),
        "fine_curve": (fx_fine.tolist(), errs_fine.tolist()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(result: dict, gt_fx: Optional[float], save_path: str):
    """Plot error-vs-fx curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Coarse reproj
    fx_c, err_c = result["coarse_curve"]
    axes[0].plot(fx_c, err_c, "b-", lw=1.5, label="Reproj error")
    axes[0].axvline(result["fx_init"], color="orange", ls="--", lw=1.5,
                     label=f"ViPE: {result['fx_init']:.0f}")
    axes[0].axvline(result["fx_reproj"], color="green", ls="-", lw=2,
                     label=f"Reproj opt: {result['fx_reproj']:.1f}")
    axes[0].axvline(result["fx_analytic"], color="purple", ls="-.", lw=1.5,
                     label=f"Analytic: {result['fx_analytic']:.1f}")
    if gt_fx:
        axes[0].axvline(gt_fx, color="red", ls=":", lw=2, label=f"Ref GT: {gt_fx:.1f}")
    axes[0].set_xlabel("Focal length (px)")
    axes[0].set_ylabel("Median reprojection error (px)")
    axes[0].set_title("Coarse: Reprojection Error")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Coarse triang
    fx_ct, err_ct = result["coarse_triang_curve"]
    axes[1].plot(fx_ct, err_ct, "r-", lw=1.5, label="3D triang error")
    axes[1].axvline(result["fx_init"], color="orange", ls="--", lw=1.5)
    axes[1].axvline(result["fx_triang"], color="darkred", ls="-", lw=2,
                     label=f"Triang opt: {result['fx_triang']:.0f}")
    if gt_fx:
        axes[1].axvline(gt_fx, color="red", ls=":", lw=2, label=f"Ref GT: {gt_fx:.1f}")
    axes[1].set_xlabel("Focal length (px)")
    axes[1].set_ylabel("Median 3D error (m)")
    axes[1].set_title("Coarse: Triangulation Error")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Fine reproj
    fx_f, err_f = result["fine_curve"]
    axes[2].plot(fx_f, err_f, "b.-", markersize=3)
    axes[2].axvline(result["fx_reproj"], color="green", ls="-", lw=2,
                     label=f"Reproj opt: {result['fx_reproj']:.1f}")
    axes[2].axvline(result["fx_analytic"], color="purple", ls="-.", lw=1.5,
                     label=f"Analytic: {result['fx_analytic']:.1f}")
    if gt_fx:
        axes[2].axvline(gt_fx, color="red", ls=":", lw=2, label=f"Ref GT: {gt_fx:.1f}")
    axes[2].set_xlabel("Focal length (px)")
    axes[2].set_ylabel("Median reprojection error (px)")
    axes[2].set_title("Fine: Reprojection Error")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Focal length self-calibration via feature-based reprojection"
    )
    parser.add_argument("--input", required=True, help="Pipeline output directory")
    parser.add_argument("--video", required=True, help="Input video path (for feature matching)")
    parser.add_argument("--output", default=None, help="Output dir")
    parser.add_argument("--gt-fx", type=float, default=None, help="Reference GT fx")
    parser.add_argument("--use-gt", action="store_true", help="Use GT depth + poses")
    parser.add_argument("--visualize", action="store_true", help="Generate plots")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.input, "focal_calibration_v2")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Focal Length Self-Calibration v2 (Feature-Based)")
    print("=" * 60)

    data = load_pipeline_outputs(args.input)
    K = data["K"]
    fx_init = float(K[0, 0])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    if args.use_gt:
        if data["gt_depths"] is None or data["gt_poses"] is None:
            print("ERROR: --use-gt requires GT files")
            sys.exit(1)
        depths = data["gt_depths"].astype(np.float64)
        poses_c2w = data["gt_poses"].astype(np.float64)
        print("\n** Using GT depth + GT poses **")
    else:
        depths = data["depths"].astype(np.float64)
        poses_c2w = data["poses_c2w"].astype(np.float64)
        print("\n** Using ViPE depth + ViPE poses **")

    # Run
    result = optimize_focal(
        args.video, depths, poses_c2w, cx, cy, fx_init,
    )

    # Report
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  fx_init (ViPE):       {result['fx_init']:.2f}")
    print(f"  fx_reproj (optimized):{result['fx_reproj']:.2f}")
    print(f"  fx_analytic (median): {result['fx_analytic']:.2f} ± {result['fx_analytic_std']:.2f}")
    print(f"  fx_triang (coarse):   {result['fx_triang']:.2f}")
    if args.gt_fx:
        for name, val in [("reproj", result["fx_reproj"]),
                          ("analytic", result["fx_analytic"]),
                          ("triang", result["fx_triang"])]:
            err = abs(val - args.gt_fx) / args.gt_fx * 100
            print(f"  {name} error vs GT:  {err:.2f}%")
        result["fx_gt"] = args.gt_fx
    print(f"  Reproj error: {result['reproj_error_at_init']:.4f} → {result['reproj_error_at_opt']:.4f} px")
    print(f"  Matches: {result['n_matches']} across {result['n_pairs']} pairs")
    print("=" * 60)

    # Save
    K_new = K.copy()
    K_new[0, 0] = result["fx_reproj"]
    K_new[1, 1] = result["fx_reproj"]
    np.save(os.path.join(output_dir, "K_corrected.npy"), K_new)

    log = {k: v for k, v in result.items()
           if k not in ("coarse_curve", "fine_curve", "coarse_triang_curve")}
    with open(os.path.join(output_dir, "calibration_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    if args.visualize:
        plot_results(result, args.gt_fx, os.path.join(output_dir, "error_vs_fx.png"))


if __name__ == "__main__":
    main()
