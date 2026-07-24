#!/usr/bin/env python3
"""
Quantitative Benchmark: MegaSAM vs ViPE on HOI4D-32.

Metrics (depth):
  - AbsRel: |d_pred - d_gt| / d_gt
  - RMSE:   sqrt(mean((d_pred - d_gt)^2))
  - δ<1.25: % of pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25
  - δ<1.25²: same with threshold 1.5625
  - δ<1.25³: same with threshold 1.953125

Metrics (pose trajectory):
  - ATE (Absolute Trajectory Error): RMSE of aligned translation
  - RPE (Relative Pose Error): per-frame relative rotation/translation error

Scale alignment:
  - MegaSAM: median-scaling to GT (standard monocular evaluation)
  - ViPE: same median-scaling

Usage:
    conda activate biv2ap
    python tools/eval_benchmark.py
"""

import os, sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict

BENCH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "Data", "Benchmark")


# ══════════════════════════════════════════════════════════════════════════════
# Depth Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_depth_metrics(pred, gt, min_depth=0.1, max_depth=10.0):
    """Compute standard monocular depth metrics with median-scaling.

    Args:
        pred: (H, W) predicted depth in metres
        gt:   (H, W) ground-truth depth in metres
        min_depth, max_depth: valid depth range

    Returns:
        dict of metrics, or None if insufficient valid pixels
    """
    # Valid mask
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt)
    valid &= (pred > 1e-4) & np.isfinite(pred)

    if valid.sum() < 100:
        return None

    pred_v = pred[valid]
    gt_v = gt[valid]

    # Median scaling (standard for monocular depth eval)
    scale = np.median(gt_v) / np.median(pred_v)
    pred_v = pred_v * scale

    # Clamp
    pred_v = np.clip(pred_v, min_depth, max_depth)

    # AbsRel
    abs_rel = np.mean(np.abs(pred_v - gt_v) / gt_v)

    # SqRel
    sq_rel = np.mean(((pred_v - gt_v) ** 2) / gt_v)

    # RMSE
    rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))

    # RMSE log
    rmse_log = np.sqrt(np.mean((np.log(pred_v) - np.log(gt_v)) ** 2))

    # Thresholds
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    d1 = np.mean(ratio < 1.25)
    d2 = np.mean(ratio < 1.25 ** 2)
    d3 = np.mean(ratio < 1.25 ** 3)

    return {
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "d1": float(d1),
        "d2": float(d2),
        "d3": float(d3),
        "scale": float(scale),
        "n_valid": int(valid.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory Metrics (ATE)
# ══════════════════════════════════════════════════════════════════════════════

def align_trajectories_umeyama(pred_xyz, gt_xyz):
    """Umeyama alignment (sim3) of predicted trajectory to GT.

    Returns aligned_pred, (scale, R, t)
    """
    mu_pred = pred_xyz.mean(axis=0)
    mu_gt = gt_xyz.mean(axis=0)

    pred_c = pred_xyz - mu_pred
    gt_c = gt_xyz - mu_gt

    # Covariance
    H = pred_c.T @ gt_c / len(pred_xyz)
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T

    # Scale
    var_pred = np.sum(pred_c ** 2) / len(pred_xyz)
    scale = np.sum(S * np.diag(D)) / var_pred

    # Translation
    t = mu_gt - scale * R @ mu_pred

    aligned = scale * (R @ pred_xyz.T).T + t
    return aligned, (scale, R, t)


def compute_ate(pred_c2w, gt_c2w):
    """Compute ATE (Absolute Trajectory Error) after Umeyama alignment.

    Args:
        pred_c2w: (N, 4, 4) predicted camera-to-world poses
        gt_c2w:   (N, 4, 4) ground-truth camera-to-world poses

    Returns:
        dict with ate_rmse, ate_mean, ate_median
    """
    pred_xyz = pred_c2w[:, :3, 3]
    gt_xyz = gt_c2w[:, :3, 3]

    # Filter out any degenerate poses
    valid = np.all(np.isfinite(pred_xyz), axis=1) & np.all(np.isfinite(gt_xyz), axis=1)
    if valid.sum() < 3:
        return None

    pred_xyz = pred_xyz[valid]
    gt_xyz = gt_xyz[valid]

    aligned_pred, _ = align_trajectories_umeyama(pred_xyz, gt_xyz)
    errors = np.linalg.norm(aligned_pred - gt_xyz, axis=1)

    return {
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean": float(np.mean(errors)),
        "ate_median": float(np.median(errors)),
        "n_poses": int(len(errors)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation per sequence
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_sequence(name, pipeline, bench_root):
    """Evaluate a single sequence for a given pipeline."""
    import cv2

    gt_depth_path = os.path.join(bench_root, "HOI4D", "gt_depth_npz", f"{name}.npz")
    gt_pose_path = os.path.join(bench_root, "HOI4D", "gt_pose", f"{name}.npy")
    pred_dir = os.path.join(bench_root, "HOI4D", pipeline, name)
    pred_depth_path = os.path.join(pred_dir, "depth.npz")
    pred_pose_path = os.path.join(pred_dir, "cam_c2w.npy")
    meta_path = os.path.join(pred_dir, "meta.json")

    if not os.path.exists(pred_depth_path) or not os.path.exists(gt_depth_path):
        return None

    # Load GT
    gt_depths = np.load(gt_depth_path)["depths"]  # (N_gt, H_gt, W_gt)
    N_gt, H_gt, W_gt = gt_depths.shape

    # Load pred
    pred_depths = np.load(pred_depth_path)["depths"]  # (N_pred, H_pred, W_pred)
    N_pred, H_pred, W_pred = pred_depths.shape

    # Load meta
    meta = json.load(open(meta_path))
    stride = meta.get("stride", 1)

    # Frame alignment: pred may have fewer frames (subsampled)
    # MegaSAM: 60 frames uniformly sampled from 300
    # ViPE: stride-based sampling
    if pipeline == "megasam":
        # MegaSAM uniformly samples MAX_FRAMES=60 from total
        total_frames = N_gt
        step = max(1, total_frames // N_pred)
        pred_frame_idxs = list(range(0, total_frames, step))[:N_pred]
    else:
        # ViPE uses stride
        pred_frame_idxs = list(range(0, N_gt, stride))[:N_pred]

    # Compute per-frame depth metrics
    frame_metrics = []
    for pred_i, gt_i in enumerate(pred_frame_idxs):
        if pred_i >= N_pred or gt_i >= N_gt:
            break

        pred_d = pred_depths[pred_i]
        gt_d = gt_depths[gt_i]

        # Resize pred to GT resolution if needed
        if pred_d.shape != gt_d.shape:
            pred_d = cv2.resize(pred_d, (W_gt, H_gt),
                                interpolation=cv2.INTER_LINEAR)

        m = compute_depth_metrics(pred_d, gt_d)
        if m is not None:
            frame_metrics.append(m)

    if not frame_metrics:
        return None

    # Average depth metrics across frames
    avg_depth = {}
    for key in frame_metrics[0]:
        if key in ("scale", "n_valid"):
            avg_depth[key] = float(np.median([m[key] for m in frame_metrics]))
        else:
            avg_depth[key] = float(np.mean([m[key] for m in frame_metrics]))
    avg_depth["n_frames_eval"] = len(frame_metrics)

    # Trajectory metrics
    traj_metrics = None
    if os.path.exists(pred_pose_path) and os.path.exists(gt_pose_path):
        try:
            pred_c2w = np.load(pred_pose_path)  # (N_pred, 4, 4)
            gt_c2w_full = np.load(gt_pose_path)  # (N_gt, 4, 4)

            # Align frame indices
            gt_c2w_aligned = []
            pred_c2w_aligned = []
            for pred_i, gt_i in enumerate(pred_frame_idxs):
                if pred_i >= len(pred_c2w) or gt_i >= len(gt_c2w_full):
                    break
                pred_c2w_aligned.append(pred_c2w[pred_i])
                gt_c2w_aligned.append(gt_c2w_full[gt_i])

            if len(pred_c2w_aligned) >= 3:
                pred_c2w_aligned = np.array(pred_c2w_aligned)
                gt_c2w_aligned = np.array(gt_c2w_aligned)
                traj_metrics = compute_ate(pred_c2w_aligned, gt_c2w_aligned)
        except Exception as e:
            print(f"    Pose eval error: {e}")

    return {
        "depth": avg_depth,
        "trajectory": traj_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*70}")
    print(f"  MegaSAM vs ViPE Quantitative Benchmark on HOI4D-32")
    print(f"{'═'*70}\n")

    # Get all sequences
    gt_dir = os.path.join(BENCH_ROOT, "HOI4D", "gt_depth_npz")
    sequences = sorted([f.replace(".npz", "") for f in os.listdir(gt_dir)
                        if f.endswith(".npz")])
    print(f"  Sequences: {len(sequences)}")

    pipelines = ["megasam", "vipe"]
    all_results = {}

    for pipeline in pipelines:
        print(f"\n  ━━━ Evaluating {pipeline.upper()} ━━━")
        results = []
        for i, name in enumerate(sequences):
            r = evaluate_sequence(name, pipeline, BENCH_ROOT)
            if r is not None:
                results.append({"name": name, **r})
                d = r["depth"]
                t = r["trajectory"]
                ate_str = f"ATE={t['ate_rmse']:.4f}" if t else "ATE=N/A"
                print(f"    [{i+1:2d}/{len(sequences)}] {name[-40:]:40s}  "
                      f"AbsRel={d['abs_rel']:.4f}  RMSE={d['rmse']:.4f}  "
                      f"δ<1.25={d['d1']:.3f}  {ate_str}")
            else:
                print(f"    [{i+1:2d}/{len(sequences)}] {name[-40:]:40s}  SKIPPED")

        all_results[pipeline] = results

    # ── Summary Table ──────────────────────────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print(f"  SUMMARY: HOI4D-32 Depth Evaluation (median-scaled)")
    print(f"{'═'*70}")
    print(f"\n  {'Pipeline':<12s} {'AbsRel↓':>8s} {'SqRel↓':>8s} {'RMSE↓':>8s} "
          f"{'RMSElog↓':>8s} {'δ<1.25↑':>8s} {'δ<1.56↑':>8s} {'δ<1.95↑':>8s} "
          f"{'#Seq':>5s}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")

    summary = {}
    for pipeline in pipelines:
        results = all_results[pipeline]
        if not results:
            continue
        depth_metrics = [r["depth"] for r in results]
        avg = {}
        for key in ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]:
            avg[key] = float(np.mean([m[key] for m in depth_metrics]))

        print(f"  {pipeline:<12s} {avg['abs_rel']:>8.4f} {avg['sq_rel']:>8.4f} "
              f"{avg['rmse']:>8.4f} {avg['rmse_log']:>8.4f} "
              f"{avg['d1']:>8.3f} {avg['d2']:>8.3f} {avg['d3']:>8.3f} "
              f"{len(results):>5d}")
        summary[pipeline] = avg

    # ── Trajectory Summary ─────────────────────────────────────────────────────
    print(f"\n\n  {'Pipeline':<12s} {'ATE_RMSE↓':>10s} {'ATE_Mean↓':>10s} "
          f"{'ATE_Med↓':>10s} {'#Seq':>5s}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*5}")

    for pipeline in pipelines:
        results = all_results[pipeline]
        traj_results = [r for r in results if r.get("trajectory") is not None]
        if not traj_results:
            print(f"  {pipeline:<12s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {0:>5d}")
            continue
        traj_metrics = [r["trajectory"] for r in traj_results]
        ate_rmse = float(np.mean([m["ate_rmse"] for m in traj_metrics]))
        ate_mean = float(np.mean([m["ate_mean"] for m in traj_metrics]))
        ate_med = float(np.mean([m["ate_median"] for m in traj_metrics]))
        print(f"  {pipeline:<12s} {ate_rmse:>10.4f} {ate_mean:>10.4f} "
              f"{ate_med:>10.4f} {len(traj_results):>5d}")

    # ── Winner ─────────────────────────────────────────────────────────────────
    if len(summary) == 2:
        print(f"\n\n  {'═'*50}")
        better_depth = "megasam" if summary["megasam"]["abs_rel"] < summary["vipe"]["abs_rel"] else "vipe"
        better_d1 = "megasam" if summary["megasam"]["d1"] > summary["vipe"]["d1"] else "vipe"
        print(f"  Depth AbsRel winner:   {better_depth.upper()}")
        print(f"  Depth δ<1.25 winner:   {better_d1.upper()}")

        # Improvement percentages
        abs_rel_diff = (summary["vipe"]["abs_rel"] - summary["megasam"]["abs_rel"]) / summary["vipe"]["abs_rel"] * 100
        d1_diff = (summary["megasam"]["d1"] - summary["vipe"]["d1"]) / summary["vipe"]["d1"] * 100
        print(f"\n  MegaSAM vs ViPE AbsRel: {abs_rel_diff:+.1f}% ({'better' if abs_rel_diff > 0 else 'worse'} for MegaSAM)")
        print(f"  MegaSAM vs ViPE δ<1.25: {d1_diff:+.1f}% ({'better' if d1_diff > 0 else 'worse'} for MegaSAM)")
        print(f"  {'═'*50}")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = os.path.join(BENCH_ROOT, "HOI4D", "eval_results.json")
    save_data = {
        "summary": summary,
        "per_sequence": {p: all_results[p] for p in pipelines},
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
