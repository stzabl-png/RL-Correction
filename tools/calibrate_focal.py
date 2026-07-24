#!/usr/bin/env python3
"""
Focal Length Self-Calibration via Multi-View Depth Consistency.

Uses ViPE's accurate metric depth and camera poses to recover the true
focal length of the input video, without modifying ViPE itself.

Algorithm:
  For a candidate focal length f, unproject pixels from frame i using depth,
  transform to frame j via relative pose, project back, and compare
  the predicted depth with the actual depth at the landing pixel.
  The correct f minimises this inconsistency.

Usage:
    python calibrate_focal.py \
        --input /path/to/pipeline/output \
        --gt-fx 1060.3          # optional, for validation
        --visualize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_pipeline_outputs(input_dir: str) -> dict:
    """Load ViPE pipeline outputs (depth, poses, K).

    Tries several path conventions to find the files.
    """
    # --- K matrix ---
    for k_path in [
        os.path.join(input_dir, "debug", "0_vipe", "K.npy"),
        os.path.join(input_dir, "K.npy"),
    ]:
        if os.path.exists(k_path):
            K = np.load(k_path)
            break
    else:
        raise FileNotFoundError(f"K.npy not found under {input_dir}")

    # --- Poses c2w ---
    for p_path in [
        os.path.join(input_dir, "debug", "0_vipe", "poses_c2w.npy"),
        os.path.join(input_dir, "cam_c2w.npy"),
    ]:
        if os.path.exists(p_path):
            poses_c2w = np.load(p_path)
            break
    else:
        raise FileNotFoundError(f"poses_c2w not found under {input_dir}")

    # --- Depth ---
    depth_path = os.path.join(input_dir, "depth.npz")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"depth.npz not found: {depth_path}")
    depth_data = np.load(depth_path)
    # Handle different key names
    if "depths" in depth_data:
        depths = depth_data["depths"]
    elif "depth" in depth_data:
        depths = depth_data["depth"]
    else:
        key = list(depth_data.keys())[0]
        depths = depth_data[key]

    # --- GT depth (optional, for validation with GT poses) ---
    gt_depth_path = os.path.join(input_dir, "gt", "depth_gt.npz")
    gt_depths = None
    if os.path.exists(gt_depth_path):
        gd = np.load(gt_depth_path)
        gt_depths = gd["depths"] if "depths" in gd else gd[list(gd.keys())[0]]

    # --- GT poses (optional) ---
    gt_poses_path = os.path.join(input_dir, "gt", "poses_c2w_gt.npy")
    gt_poses = None
    if os.path.exists(gt_poses_path):
        gt_poses = np.load(gt_poses_path)

    print(f"Loaded: K={K.shape}, poses={poses_c2w.shape}, depths={depths.shape}")
    print(f"  fx_init={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    if gt_depths is not None:
        print(f"  GT depth: {gt_depths.shape}")
    if gt_poses is not None:
        print(f"  GT poses: {gt_poses.shape}")

    return {
        "K": K,
        "poses_c2w": poses_c2w,
        "depths": depths,
        "gt_depths": gt_depths,
        "gt_poses": gt_poses,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame Pair Selection
# ═══════════════════════════════════════════════════════════════════════════════

def select_frame_pairs(
    poses_c2w: np.ndarray,
    n_pairs: int = 30,
    min_gap: int = 5,
    max_gap: int = 50,
    min_baseline: float = 0.005,
) -> list[tuple[int, int]]:
    """Select frame pairs with sufficient baseline for calibration."""
    N = len(poses_c2w)
    candidates = []

    for gap in range(min_gap, min(max_gap + 1, N)):
        for i in range(0, N - gap, max(1, gap // 3)):
            j = i + gap
            if j >= N:
                continue
            t_i = poses_c2w[i, :3, 3]
            t_j = poses_c2w[j, :3, 3]
            baseline = np.linalg.norm(t_j - t_i)
            if baseline > min_baseline:
                candidates.append((i, j, baseline))

    # Sort by baseline (prefer larger), take top-N
    candidates.sort(key=lambda x: x[2], reverse=True)
    pairs = [(i, j) for i, j, _ in candidates[:n_pairs]]

    if len(pairs) < 5:
        print(f"  WARNING: only {len(pairs)} valid frame pairs found")

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
#  Pixel Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def sample_pixels(
    depth: np.ndarray,
    n_pixels: int = 3000,
    margin: int = 30,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> np.ndarray:
    """Sample valid pixels from a depth map.

    Returns (M, 2) array of (u, v) integer coordinates.
    """
    H, W = depth.shape
    valid = (depth > min_depth) & (depth < max_depth)

    # Exclude image borders
    valid[:margin, :] = False
    valid[-margin:, :] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False

    ys, xs = np.where(valid)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    if len(ys) > n_pixels:
        idx = np.random.choice(len(ys), n_pixels, replace=False)
        ys, xs = ys[idx], xs[idx]

    return np.stack([xs, ys], axis=1)  # (M, 2) as (u, v)


# ═══════════════════════════════════════════════════════════════════════════════
#  Bilinear Sampling
# ═══════════════════════════════════════════════════════════════════════════════

def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a 2D array at (u, v) float coordinates."""
    H, W = image.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    # Clamp
    u0 = np.clip(u0, 0, W - 1)
    u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)
    v1 = np.clip(v1, 0, H - 1)

    du = u - u0.astype(np.float64)
    dv = v - v0.astype(np.float64)

    val = (
        image[v0, u0] * (1 - du) * (1 - dv)
        + image[v0, u1] * du * (1 - dv)
        + image[v1, u0] * (1 - du) * dv
        + image[v1, u1] * du * dv
    )
    return val


# ═══════════════════════════════════════════════════════════════════════════════
#  Core: Depth Consistency Error
# ═══════════════════════════════════════════════════════════════════════════════

def depth_consistency_error(
    fx: float,
    depth_i: np.ndarray,
    depth_j: np.ndarray,
    T_ij: np.ndarray,
    cx: float,
    cy: float,
    pixels: np.ndarray,
) -> float:
    """Compute depth consistency error for a candidate focal length.

    Args:
        fx: candidate focal length (assumes fx == fy)
        depth_i: depth map of frame i, shape (H, W)
        depth_j: depth map of frame j, shape (H, W)
        T_ij: relative pose from i to j, shape (4, 4), i.e. T_j_w2c @ T_i_c2w
        cx, cy: principal point
        pixels: sampled pixels in frame i, shape (M, 2) as (u, v)

    Returns:
        Median relative depth error (float). Lower is better.
    """
    if len(pixels) == 0:
        return float("inf")

    H, W = depth_i.shape
    R = T_ij[:3, :3]
    t = T_ij[:3, 3]

    us = pixels[:, 0].astype(np.float64)
    vs = pixels[:, 1].astype(np.float64)
    ds = depth_i[pixels[:, 1], pixels[:, 0]].astype(np.float64)

    # Unproject to 3D (camera i coordinates)
    X = (us - cx) * ds / fx
    Y = (vs - cy) * ds / fx
    Z = ds
    P_i = np.stack([X, Y, Z], axis=1)  # (M, 3)

    # Transform to camera j coordinates
    P_j = (R @ P_i.T).T + t  # (M, 3)

    # Project to frame j pixel plane
    valid = P_j[:, 2] > 0.01
    u_j = fx * P_j[:, 0] / P_j[:, 2] + cx
    v_j = fx * P_j[:, 1] / P_j[:, 2] + cy

    # Bounds check
    valid &= (u_j >= 1) & (u_j < W - 2) & (v_j >= 1) & (v_j < H - 2)

    n_valid = valid.sum()
    if n_valid < 20:
        return float("inf")

    # Bilinear sample depth_j at projected locations
    d_j_actual = bilinear_sample(
        depth_j.astype(np.float64), u_j[valid], v_j[valid]
    )
    d_j_pred = P_j[valid, 2]

    # Exclude invalid depth lookups
    depth_ok = d_j_actual > 0.1
    if depth_ok.sum() < 20:
        return float("inf")

    # Relative depth error
    rel_err = np.abs(d_j_pred[depth_ok] - d_j_actual[depth_ok]) / d_j_actual[depth_ok]

    return float(np.median(rel_err))


# ═══════════════════════════════════════════════════════════════════════════════
#  Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def compute_total_error(
    fx: float,
    depths: np.ndarray,
    poses_c2w: np.ndarray,
    cx: float,
    cy: float,
    pairs: list[tuple[int, int]],
    pixel_cache: dict,
) -> float:
    """Compute total depth consistency error across all frame pairs."""
    errors = []
    for i, j in pairs:
        # Relative pose: j_w2c @ i_c2w
        T_ij = np.linalg.inv(poses_c2w[j]) @ poses_c2w[i]
        pixels = pixel_cache[i]
        if len(pixels) == 0:
            continue
        err = depth_consistency_error(fx, depths[i], depths[j], T_ij, cx, cy, pixels)
        if np.isfinite(err):
            errors.append(err)

    if not errors:
        return float("inf")
    return float(np.median(errors))


def optimize_focal(
    depths: np.ndarray,
    poses_c2w: np.ndarray,
    cx: float,
    cy: float,
    fx_init: float,
    search_range: tuple[float, float] = (400.0, 2500.0),
    n_pixels: int = 3000,
    n_pairs: int = 30,
    verbose: bool = True,
) -> dict:
    """Three-stage focal length optimization.

    Stage 1: Coarse grid search (step=20)
    Stage 2: Fine grid search (step=2)
    Stage 3: Scipy bounded minimization
    """
    t0 = time.time()

    # Select frame pairs
    pairs = select_frame_pairs(poses_c2w, n_pairs=n_pairs)
    if verbose:
        print(f"\nSelected {len(pairs)} frame pairs")

    # Pre-sample pixels for each source frame (deterministic)
    np.random.seed(42)
    pixel_cache = {}
    unique_frames = sorted(set(i for i, _ in pairs))
    for frame_idx in unique_frames:
        pixel_cache[frame_idx] = sample_pixels(depths[frame_idx], n_pixels=n_pixels)
    if verbose:
        print(f"Sampled pixels for {len(unique_frames)} source frames")

    def cost(fx):
        return compute_total_error(fx, depths, poses_c2w, cx, cy, pairs, pixel_cache)

    # ── Stage 1: Coarse search ──
    if verbose:
        print(f"\n[Stage 1] Coarse search: fx ∈ [{search_range[0]:.0f}, {search_range[1]:.0f}], step=20")
    fx_coarse = np.arange(search_range[0], search_range[1] + 1, 20)
    err_coarse = np.array([cost(fx) for fx in fx_coarse])
    best_idx = np.argmin(err_coarse)
    fx_best_coarse = fx_coarse[best_idx]
    if verbose:
        print(f"  Best: fx={fx_best_coarse:.0f}, error={err_coarse[best_idx]:.6f}")

    # ── Stage 2: Fine search ──
    lo = max(search_range[0], fx_best_coarse - 100)
    hi = min(search_range[1], fx_best_coarse + 100)
    if verbose:
        print(f"\n[Stage 2] Fine search: fx ∈ [{lo:.0f}, {hi:.0f}], step=2")
    fx_fine = np.arange(lo, hi + 1, 2)
    err_fine = np.array([cost(fx) for fx in fx_fine])
    best_idx_fine = np.argmin(err_fine)
    fx_best_fine = fx_fine[best_idx_fine]
    if verbose:
        print(f"  Best: fx={fx_best_fine:.0f}, error={err_fine[best_idx_fine]:.6f}")

    # ── Stage 3: Scipy refinement ──
    lo3 = max(search_range[0], fx_best_fine - 20)
    hi3 = min(search_range[1], fx_best_fine + 20)
    if verbose:
        print(f"\n[Stage 3] Scipy bounded: fx ∈ [{lo3:.0f}, {hi3:.0f}]")
    result = minimize_scalar(cost, bounds=(lo3, hi3), method="bounded",
                             options={"xatol": 0.01, "maxiter": 50})
    fx_opt = result.x
    err_opt = result.fun
    if verbose:
        print(f"  Optimal: fx={fx_opt:.2f}, error={err_opt:.6f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nTotal time: {elapsed:.1f}s")

    return {
        "fx_init": float(fx_init),
        "fx_optimized": float(fx_opt),
        "error_at_init": float(cost(fx_init)),
        "error_at_opt": float(err_opt),
        "n_pairs": len(pairs),
        "n_pixels": n_pixels,
        "elapsed_sec": elapsed,
        "coarse_curve": (fx_coarse.tolist(), err_coarse.tolist()),
        "fine_curve": (fx_fine.tolist(), err_fine.tolist()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_curve(result: dict, gt_fx: Optional[float], save_path: str):
    """Plot error-vs-fx curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Coarse curve
    fx_c, err_c = result["coarse_curve"]
    ax1.plot(fx_c, err_c, "b.-", markersize=3, label="Depth consistency error")
    ax1.axvline(result["fx_init"], color="orange", ls="--", lw=1.5, label=f"ViPE init: {result['fx_init']:.0f}")
    ax1.axvline(result["fx_optimized"], color="green", ls="-", lw=2, label=f"Optimized: {result['fx_optimized']:.1f}")
    if gt_fx:
        ax1.axvline(gt_fx, color="red", ls=":", lw=2, label=f"Reference GT: {gt_fx:.1f}")
    ax1.set_xlabel("Focal length (px)")
    ax1.set_ylabel("Median relative depth error")
    ax1.set_title("Coarse Search")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Fine curve
    fx_f, err_f = result["fine_curve"]
    ax2.plot(fx_f, err_f, "b.-", markersize=4)
    ax2.axvline(result["fx_init"], color="orange", ls="--", lw=1.5, label=f"ViPE init: {result['fx_init']:.0f}")
    ax2.axvline(result["fx_optimized"], color="green", ls="-", lw=2, label=f"Optimized: {result['fx_optimized']:.1f}")
    if gt_fx:
        ax2.axvline(gt_fx, color="red", ls=":", lw=2, label=f"Reference GT: {gt_fx:.1f}")
    ax2.set_xlabel("Focal length (px)")
    ax2.set_ylabel("Median relative depth error")
    ax2.set_title("Fine Search")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved error curve: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Focal length self-calibration via multi-view depth consistency"
    )
    parser.add_argument("--input", required=True, help="Pipeline output directory")
    parser.add_argument("--output", default=None, help="Output directory (default: input/focal_calibration/)")
    parser.add_argument("--gt-fx", type=float, default=None, help="Ground truth fx for validation")
    parser.add_argument("--use-gt", action="store_true", help="Use GT depth + GT poses instead of ViPE outputs")
    parser.add_argument("--visualize", action="store_true", help="Generate error-vs-fx plot")
    parser.add_argument("--fx-min", type=float, default=400, help="Min fx search range")
    parser.add_argument("--fx-max", type=float, default=2500, help="Max fx search range")
    parser.add_argument("--n-pairs", type=int, default=30, help="Number of frame pairs")
    parser.add_argument("--n-pixels", type=int, default=3000, help="Pixels per frame")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(args.input, "focal_calibration")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Focal Length Self-Calibration")
    print("=" * 60)

    # Load data
    data = load_pipeline_outputs(args.input)
    K = data["K"]
    fx_init = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Choose which depth/poses to use
    if args.use_gt:
        if data["gt_depths"] is None or data["gt_poses"] is None:
            print("ERROR: --use-gt requires GT depth and poses in gt/ subdirectory")
            sys.exit(1)
        depths = data["gt_depths"].astype(np.float64)
        poses_c2w = data["gt_poses"].astype(np.float64)
        print("\n** Using GT depth + GT poses for calibration **")
    else:
        depths = data["depths"].astype(np.float64)
        poses_c2w = data["poses_c2w"].astype(np.float64)
        print("\n** Using ViPE depth + ViPE poses for calibration **")

    # Run optimization
    result = optimize_focal(
        depths, poses_c2w, cx, cy, fx_init,
        search_range=(args.fx_min, args.fx_max),
        n_pixels=args.n_pixels,
        n_pairs=args.n_pairs,
    )

    # ── Report ──
    print("\n" + "=" * 60)
    print("  Calibration Results")
    print("=" * 60)
    print(f"  fx_init (ViPE):     {result['fx_init']:.2f}")
    print(f"  fx_optimized:       {result['fx_optimized']:.2f}")
    if args.gt_fx:
        gt_err_init = abs(result["fx_init"] - args.gt_fx) / args.gt_fx * 100
        gt_err_opt = abs(result["fx_optimized"] - args.gt_fx) / args.gt_fx * 100
        print(f"  fx_gt (reference):  {args.gt_fx:.2f}")
        print(f"  Init error vs GT:   {gt_err_init:.2f}%")
        print(f"  Optimized error:    {gt_err_opt:.2f}%")
        result["fx_gt"] = args.gt_fx
        result["error_pct_init"] = gt_err_init
        result["error_pct_opt"] = gt_err_opt
    print(f"  Depth error @ init: {result['error_at_init']:.6f}")
    print(f"  Depth error @ opt:  {result['error_at_opt']:.6f}")
    print("=" * 60)

    # ── Save ──
    # Corrected K
    K_new = K.copy()
    K_new[0, 0] = result["fx_optimized"]
    K_new[1, 1] = result["fx_optimized"]
    np.save(os.path.join(output_dir, "K_corrected.npy"), K_new)
    print(f"\nSaved K_corrected.npy")

    # Log
    log = {k: v for k, v in result.items() if k not in ("coarse_curve", "fine_curve")}
    with open(os.path.join(output_dir, "calibration_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved calibration_log.json")

    # Visualization
    if args.visualize:
        plot_error_curve(
            result,
            args.gt_fx,
            os.path.join(output_dir, "error_vs_fx.png"),
        )


if __name__ == "__main__":
    main()
