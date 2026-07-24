#!/usr/bin/env python3
"""
Benchmark: MegaSAM vs DA3 on EgoDex add_remove_lid
Compares intrinsics, poses, and depth across 26 episodes.
"""

import os, sys, json, time, glob
import numpy as np
import cv2
from natsort import natsorted

# ── Paths ──────────────────────────────────────────────────────────────────────
EGODEX_ROOT = "/home/lyh/Project/Affordance2Grasp/data_hub/RawData/EgoRawData/egodex/test"
TASK = "add_remove_lid"
TASK_DIR = os.path.join(EGODEX_ROOT, TASK)
MEGASAM_DIR = "/home/lyh/Project/Reconstruct_and_Retarget/Output/Depth/MegaSAM/Egodex"
OUT_DIR = "/home/lyh/Project/Reconstruct_and_Retarget/Output/Benchmark/MegaSAM_vs_DA3"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_frames(mp4_path, out_dir, max_frames=None):
    """Extract frames from mp4 to jpg files."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and idx >= max_frames:
            break
        path = os.path.join(out_dir, f"{idx:06d}.jpg")
        if not os.path.exists(path):
            cv2.imwrite(path, frame)
        frames.append(path)
        idx += 1
    cap.release()
    return frames


def load_gt(hdf5_path):
    """Load GT intrinsics and poses from HDF5."""
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        K_gt = f['camera/intrinsic'][:]           # (3,3)
        poses_gt = f['transforms/camera'][:]       # (T, 4, 4)
    return K_gt, poses_gt


def load_megasam(ep_dir, n_total_frames):
    """Load MegaSAM results."""
    depth = np.load(os.path.join(ep_dir, 'depth.npz'))['depths']  # (60, H, W)
    K = np.load(os.path.join(ep_dir, 'K.npy'))                    # (3, 3)
    c2w = np.load(os.path.join(ep_dir, 'cam_c2w.npy'))            # (60, 4, 4)
    with open(os.path.join(ep_dir, 'meta.json')) as f:
        meta = json.load(f)

    # MegaSAM uses 60 uniformly sampled frames
    n_mega = depth.shape[0]
    frame_indices = np.linspace(0, n_total_frames - 1, n_mega, dtype=int)

    return {
        'depth': depth,
        'K': K,
        'c2w': c2w,
        'meta': meta,
        'frame_indices': frame_indices,
        'hw': (meta['hw'][0], meta['hw'][1]),
    }


def run_da3(frame_paths, device='cuda'):
    """Run DA3 inference."""
    import torch
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    model = model.to(device=torch.device(device))

    t0 = time.time()
    prediction = model.inference(frame_paths)
    elapsed = time.time() - t0

    return {
        'depth': prediction.depth,             # (N, H, W) meters
        'intrinsics': prediction.intrinsics,   # (N, 3, 3)
        'extrinsics': prediction.extrinsics,   # (N, 3, 4)  w2c
        'conf': prediction.conf,               # (N, H, W)
        'elapsed': elapsed,
        'n_frames': len(frame_paths),
    }, model


def rescale_intrinsics(K, from_hw, to_hw):
    """Rescale intrinsics from one resolution to another."""
    K_out = K.copy()
    sy = to_hw[0] / from_hw[0]
    sx = to_hw[1] / from_hw[1]
    K_out[0, 0] *= sx
    K_out[0, 2] *= sx
    K_out[1, 1] *= sy
    K_out[1, 2] *= sy
    return K_out


def intrinsic_error(K_est, K_gt, est_hw, gt_hw=(1080, 1920)):
    """Compute intrinsic estimation error (rescaled to GT resolution)."""
    K_rescaled = rescale_intrinsics(K_est, est_hw, gt_hw)
    fx_err = abs(K_rescaled[0, 0] - K_gt[0, 0]) / K_gt[0, 0] * 100
    fy_err = abs(K_rescaled[1, 1] - K_gt[1, 1]) / K_gt[1, 1] * 100
    return fx_err, fy_err, K_rescaled


def align_poses_sim3(poses_est, poses_gt):
    """Align estimated poses to GT using Sim(3) (Umeyama)."""
    # Extract translations
    t_est = poses_est[:, :3, 3]  # (N, 3)
    t_gt = poses_gt[:, :3, 3]    # (N, 3)

    # Umeyama alignment
    mu_est = t_est.mean(0)
    mu_gt = t_gt.mean(0)
    t_est_c = t_est - mu_est
    t_gt_c = t_gt - mu_gt

    # Scale
    s = np.sqrt((t_gt_c ** 2).sum()) / (np.sqrt((t_est_c ** 2).sum()) + 1e-8)

    # Rotation (Procrustes)
    H = t_est_c.T @ t_gt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ S @ U.T

    # Translation
    t = mu_gt - s * R @ mu_est

    # Apply
    t_aligned = s * (R @ t_est.T).T + t
    return t_aligned, s, R, t


def compute_ate(t_est, t_gt):
    """Absolute Trajectory Error (RMSE of aligned translations)."""
    diff = t_est - t_gt
    ate = np.sqrt((diff ** 2).sum(1)).mean()
    return ate


def compute_rpe(poses_est, poses_gt):
    """Relative Pose Error (translation and rotation)."""
    n = min(len(poses_est), len(poses_gt))
    t_errs = []
    r_errs = []
    for i in range(n - 1):
        # Relative transform
        rel_est = np.linalg.inv(poses_est[i]) @ poses_est[i + 1]
        rel_gt = np.linalg.inv(poses_gt[i]) @ poses_gt[i + 1]
        # Translation error
        dt = np.linalg.norm(rel_est[:3, 3] - rel_gt[:3, 3])
        t_errs.append(dt)
        # Rotation error (angle)
        dR = rel_est[:3, :3].T @ rel_gt[:3, :3]
        angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        r_errs.append(np.degrees(angle))
    return np.mean(t_errs), np.mean(r_errs)


def evaluate_episode(ep_id, da3_model, device='cuda'):
    """Evaluate one episode."""
    import torch

    mp4_path = os.path.join(TASK_DIR, f"{ep_id}.mp4")
    hdf5_path = os.path.join(TASK_DIR, f"{ep_id}.hdf5")
    mega_dir = os.path.join(MEGASAM_DIR, TASK, str(ep_id))

    if not os.path.exists(hdf5_path):
        return None
    if not os.path.exists(mega_dir):
        return None

    # ── Load GT ────────────────────────────────────────────────────────────────
    K_gt, poses_gt = load_gt(hdf5_path)
    n_total = poses_gt.shape[0]

    # ── Extract frames ─────────────────────────────────────────────────────────
    frame_dir = os.path.join(OUT_DIR, "frames", TASK, str(ep_id))
    frames = extract_frames(mp4_path, frame_dir)

    # ── Load MegaSAM ───────────────────────────────────────────────────────────
    mega = load_megasam(mega_dir, n_total)

    # ── Run DA3 (on same subset of frames as MegaSAM for fair comparison) ─────
    mega_frame_paths = [frames[i] for i in mega['frame_indices'] if i < len(frames)]

    t0 = time.time()
    prediction = da3_model.inference(mega_frame_paths)
    da3_elapsed = time.time() - t0

    da3_depth = prediction.depth        # (N, H, W)
    da3_K = prediction.intrinsics       # (N, 3, 3)
    da3_ext = prediction.extrinsics     # (N, 3, 4) w2c
    da3_conf = prediction.conf

    # DA3 intrinsics: use median across frames
    da3_K_median = np.median(da3_K, axis=0)
    da3_hw = (da3_depth.shape[1], da3_depth.shape[2])

    # DA3 extrinsics → c2w (4x4)
    da3_c2w = []
    for ext in da3_ext:
        R = ext[:3, :3]
        t = ext[:3, 3]
        c2w_44 = np.eye(4)
        c2w_44[:3, :3] = R.T
        c2w_44[:3, 3] = -R.T @ t
        da3_c2w.append(c2w_44)
    da3_c2w = np.array(da3_c2w)

    # ── GT poses at MegaSAM frame indices ──────────────────────────────────────
    gt_at_mega = poses_gt[mega['frame_indices'][:len(mega_frame_paths)]]

    # ── 1. Intrinsic Error ─────────────────────────────────────────────────────
    mega_fx_err, mega_fy_err, mega_K_rescaled = intrinsic_error(
        mega['K'], K_gt, mega['hw'])
    da3_fx_err, da3_fy_err, da3_K_rescaled = intrinsic_error(
        da3_K_median, K_gt, da3_hw)

    # ── 2. Pose Error (Sim3 align) ─────────────────────────────────────────────
    n_poses = min(len(mega['c2w']), len(gt_at_mega), len(da3_c2w))

    # MegaSAM poses
    mega_t_aligned, _, _, _ = align_poses_sim3(
        mega['c2w'][:n_poses], gt_at_mega[:n_poses])
    mega_ate = compute_ate(mega_t_aligned, gt_at_mega[:n_poses, :3, 3])
    mega_rpe_t, mega_rpe_r = compute_rpe(mega['c2w'][:n_poses], gt_at_mega[:n_poses])

    # DA3 poses
    da3_t_aligned, _, _, _ = align_poses_sim3(
        da3_c2w[:n_poses], gt_at_mega[:n_poses])
    da3_ate = compute_ate(da3_t_aligned, gt_at_mega[:n_poses, :3, 3])
    da3_rpe_t, da3_rpe_r = compute_rpe(da3_c2w[:n_poses], gt_at_mega[:n_poses])

    # ── 3. Depth stats ─────────────────────────────────────────────────────────
    mega_depth_range = (float(mega['depth'].min()), float(mega['depth'].max()))
    da3_depth_range = (float(da3_depth.min()), float(da3_depth.max()))
    da3_conf_mean = float(da3_conf.mean())

    result = {
        'ep_id': ep_id,
        'n_frames': n_total,
        'n_eval_frames': n_poses,
        # Intrinsics
        'gt_fx': float(K_gt[0, 0]),
        'mega_fx_1080p': float(mega_K_rescaled[0, 0]),
        'da3_fx_1080p': float(da3_K_rescaled[0, 0]),
        'mega_fx_err_pct': round(mega_fx_err, 2),
        'da3_fx_err_pct': round(da3_fx_err, 2),
        # Poses
        'mega_ate': round(mega_ate, 5),
        'da3_ate': round(da3_ate, 5),
        'mega_rpe_t': round(mega_rpe_t, 5),
        'da3_rpe_t': round(da3_rpe_t, 5),
        'mega_rpe_r_deg': round(mega_rpe_r, 3),
        'da3_rpe_r_deg': round(da3_rpe_r, 3),
        # Depth
        'mega_depth_range': mega_depth_range,
        'da3_depth_range': da3_depth_range,
        'da3_conf_mean': round(da3_conf_mean, 4),
        # Speed
        'da3_elapsed_s': round(da3_elapsed, 2),
    }
    return result


def main():
    import torch

    print("=" * 60)
    print("  MegaSAM vs DA3 Benchmark")
    print(f"  Task: {TASK}")
    print("=" * 60)

    # Find episodes
    hdf5_files = natsorted(glob.glob(os.path.join(TASK_DIR, "*.hdf5")))
    episodes = [os.path.splitext(os.path.basename(f))[0] for f in hdf5_files]
    # Filter to those with MegaSAM results
    episodes = [e for e in episodes
                if os.path.exists(os.path.join(MEGASAM_DIR, TASK, e, 'depth.npz'))]
    print(f"  Episodes with both GT & MegaSAM: {len(episodes)}")
    print()

    # Load DA3 model once
    print("Loading DA3 model...")
    os.environ["XFORMERS_DISABLED"] = "1"
    da3_model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    da3_model = da3_model.to(device=torch.device('cuda'))
    print("DA3 model ready.\n")

    results = []
    for i, ep in enumerate(episodes):
        print(f"[{i+1}/{len(episodes)}] Episode {ep}...", end=" ", flush=True)
        try:
            r = evaluate_episode(ep, da3_model)
            if r:
                results.append(r)
                print(f"✅ ATE: mega={r['mega_ate']:.4f} da3={r['da3_ate']:.4f}  "
                      f"fx_err: mega={r['mega_fx_err_pct']:.1f}% da3={r['da3_fx_err_pct']:.1f}%")
            else:
                print("⏭️ skipped (missing data)")
        except Exception as e:
            print(f"❌ {e}")

    if not results:
        print("No results!")
        return

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Averages
    avg = lambda key: np.mean([r[key] for r in results])
    med = lambda key: np.median([r[key] for r in results])

    print(f"\n  {'Metric':<25s} {'MegaSAM':>12s} {'DA3':>12s} {'Winner':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    metrics = [
        ('fx error (%)', 'mega_fx_err_pct', 'da3_fx_err_pct', 'lower'),
        ('ATE (m)', 'mega_ate', 'da3_ate', 'lower'),
        ('RPE trans', 'mega_rpe_t', 'da3_rpe_t', 'lower'),
        ('RPE rot (°)', 'mega_rpe_r_deg', 'da3_rpe_r_deg', 'lower'),
    ]
    for name, mk, dk, better in metrics:
        mv = avg(mk)
        dv = avg(dk)
        winner = 'DA3' if (dv < mv if better == 'lower' else dv > mv) else 'MegaSAM'
        print(f"  {name:<25s} {mv:>12.4f} {dv:>12.4f} {'🏆 '+winner:>10s}")

    # Speed
    da3_total = sum(r['da3_elapsed_s'] for r in results)
    da3_per_frame = da3_total / sum(r['n_eval_frames'] for r in results)
    print(f"\n  DA3 speed: {da3_per_frame:.3f}s/frame ({da3_total:.1f}s total)")
    print(f"  MegaSAM speed: ~0.37s/frame (from batch stats)")

    # Save results
    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_json}")
    print("=" * 70)


if __name__ == "__main__":
    # Import DA3 here to set env first
    os.environ["XFORMERS_DISABLED"] = "1"
    from depth_anything_3.api import DepthAnything3
    main()
