#!/usr/bin/env python3
"""
Smooth HaWoR output to reduce hand jitter.

Methods:
  - savgol   : Savitzky-Golay filter (default, preserves trajectory shape)
  - gaussian : Gaussian smoothing (stronger suppression)
  - oneeuro  : One-Euro filter (adaptive, low latency)

Usage:
  python smooth_hawor.py --input hawor_output.npz --output hawor_smooth.npz --method savgol --window 7
"""

import argparse
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


# ──────────────────────────────────────────────
# One-Euro Filter implementation
# ──────────────────────────────────────────────
class OneEuroFilter:
    """Adaptive low-pass filter that adjusts cutoff based on signal speed."""
    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        # Derivative estimation
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def smooth_oneeuro(signal, freq=30.0, min_cutoff=1.0, beta=0.007):
    """Apply One-Euro filter to a signal [T, D]."""
    T, D = signal.shape
    out = np.zeros_like(signal)
    filt = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
    for t in range(T):
        out[t] = filt(signal[t])
    return out


# ──────────────────────────────────────────────
# Smoothing dispatcher
# ──────────────────────────────────────────────
def smooth_signal(signal, method='savgol', window=7, sigma=2.0, **kwargs):
    """
    Smooth a 2D signal [T, D] along axis=0.

    Args:
        signal: [T, D] numpy array
        method: 'savgol', 'gaussian', or 'oneeuro'
        window: window size for savgol (must be odd)
        sigma:  sigma for gaussian filter
    Returns:
        Smoothed [T, D] array
    """
    T = signal.shape[0]

    if method == 'savgol':
        # Window must be odd and <= T
        w = min(window, T)
        if w % 2 == 0:
            w -= 1
        w = max(w, 3)
        polyorder = min(2, w - 1)
        return savgol_filter(signal, window_length=w, polyorder=polyorder, axis=0)

    elif method == 'gaussian':
        return gaussian_filter1d(signal, sigma=sigma, axis=0)

    elif method == 'oneeuro':
        freq = kwargs.get('freq', 30.0)
        min_cutoff = kwargs.get('min_cutoff', 1.0)
        beta = kwargs.get('beta', 0.007)
        return smooth_oneeuro(signal, freq=freq, min_cutoff=min_cutoff, beta=beta)

    else:
        raise ValueError(f"Unknown method: {method}")


def smooth_per_hand(data, valid, method, window, sigma, **kwargs):
    """
    Smooth per-hand data [T, D], only on valid segments.
    Handles gaps (invalid frames) by smoothing valid segments independently.
    """
    T = data.shape[0]
    result = data.copy()

    # Find contiguous valid segments
    segments = []
    start = None
    for t in range(T):
        if valid[t]:
            if start is None:
                start = t
        else:
            if start is not None:
                segments.append((start, t))
                start = None
    if start is not None:
        segments.append((start, T))

    # Smooth each valid segment
    for s, e in segments:
        seg_len = e - s
        if seg_len >= 3:  # Need at least 3 frames to smooth
            result[s:e] = smooth_signal(data[s:e], method=method,
                                        window=window, sigma=sigma, **kwargs)

    return result


def smooth_rotations(rot_aa, valid, method, window, sigma, **kwargs):
    """
    Smooth rotation (axis-angle [T, 3]) via quaternion space to avoid 360° flips.
    
    Steps:
      1. axis-angle → quaternion
      2. ensure quaternion continuity (flip sign if dot < 0)
      3. smooth in quaternion space
      4. normalize quaternions
      5. quaternion → axis-angle
    """
    T = rot_aa.shape[0]
    result = rot_aa.copy()

    # Find contiguous valid segments
    segments = []
    start = None
    for t in range(T):
        if valid[t]:
            if start is None:
                start = t
        else:
            if start is not None:
                segments.append((start, t))
                start = None
    if start is not None:
        segments.append((start, T))

    for s, e in segments:
        seg_len = e - s
        if seg_len < 3:
            continue

        seg = rot_aa[s:e]  # [L, 3]

        # axis-angle → quaternion [L, 4] (scipy uses xyzw internally, as_quat returns xyzw)
        quats = R.from_rotvec(seg).as_quat()  # [L, 4] in xyzw format

        # Ensure quaternion continuity: flip sign if consecutive quats are on opposite hemisphere
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]

        # Smooth quaternion components
        quats_smooth = smooth_signal(quats, method=method, window=window,
                                     sigma=sigma, **kwargs)

        # Re-normalize to unit quaternions
        norms = np.linalg.norm(quats_smooth, axis=1, keepdims=True)
        quats_smooth = quats_smooth / np.clip(norms, 1e-8, None)

        # quaternion → axis-angle
        result[s:e] = R.from_quat(quats_smooth).as_rotvec()

    return result


def main():
    parser = argparse.ArgumentParser(description='Smooth HaWoR output')
    parser.add_argument('--input', required=True, help='Input hawor_output.npz')
    parser.add_argument('--output', required=True, help='Output smoothed npz')
    parser.add_argument('--method', default='savgol',
                        choices=['savgol', 'gaussian', 'oneeuro'],
                        help='Smoothing method (default: savgol)')
    parser.add_argument('--window', type=int, default=7,
                        help='Window size for savgol (odd number, default: 7)')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Sigma for gaussian (default: 2.0)')
    parser.add_argument('--smooth_verts', action='store_true',
                        help='Also directly smooth vertices (quick mode)')
    parser.add_argument('--mano_dir', default=None,
                        help='Path to MANO models dir to recompute vertices')
    args = parser.parse_args()

    # Load data
    data = dict(np.load(args.input))
    T = data['pred_trans'].shape[1]
    print(f"Loaded {args.input}: {T} frames")

    # ── Smooth MANO parameters per hand ──
    # Hand 0 = right, Hand 1 = left
    hand_names = ['right', 'left']
    for h in range(2):
        valid = data['pred_valid'][h].astype(bool)
        n_valid = valid.sum()
        print(f"  {hand_names[h]} hand: {n_valid}/{T} valid frames")

        if n_valid < 3:
            print(f"    Skip (too few valid frames)")
            continue

        # Smooth translation [T, 3]
        data['pred_trans'][h] = smooth_per_hand(
            data['pred_trans'][h], valid,
            args.method, args.window, args.sigma)

        # Smooth rotation (axis-angle) [T, 3] — via quaternion space
        data['pred_rot'][h] = smooth_rotations(
            data['pred_rot'][h], valid,
            args.method, args.window, args.sigma)

        # Smooth hand pose [T, 45] — each joint is axis-angle [3], smooth via quat
        hand_pose = data['pred_hand_pose'][h]  # [T, 45]
        for j in range(15):  # 15 joints × 3 axis-angle
            joint_aa = hand_pose[:, j*3:(j+1)*3]  # [T, 3]
            hand_pose[:, j*3:(j+1)*3] = smooth_rotations(
                joint_aa, valid,
                args.method, args.window, args.sigma)
        data['pred_hand_pose'][h] = hand_pose

        # Smooth betas [T, 10] — mild smoothing
        data['pred_betas'][h] = smooth_per_hand(
            data['pred_betas'][h], valid,
            args.method, max(3, args.window // 2), args.sigma * 0.5)

    # ── Optionally smooth vertices directly ──
    if args.smooth_verts:
        for side in ['right', 'left']:
            key = f'{side}_verts'
            if key in data and not np.isnan(data[key]).all():
                h = 0 if side == 'right' else 1
                valid = data['pred_valid'][h].astype(bool)
                verts = data[key]  # [T, 778, 3]
                T_v, V, _ = verts.shape
                # Smooth each vertex coordinate
                verts_flat = verts.reshape(T_v, -1)  # [T, 778*3]
                verts_flat = smooth_per_hand(
                    verts_flat, valid,
                    args.method, args.window, args.sigma)
                data[key] = verts_flat.reshape(T_v, V, 3)
                print(f"  Smoothed {key} vertices directly")

    # ── Recompute vertices if MANO available ──
    if args.mano_dir is not None:
        try:
            import torch
            import smplx
            print("\nRecomputing MANO vertices with smoothed params...")

            for h, side in enumerate(['right', 'left']):
                is_rhand = (side == 'right')
                mano = smplx.create(
                    args.mano_dir, 'mano',
                    is_rhand=is_rhand,
                    use_pca=False,
                    flat_hand_mean=False,
                    num_betas=10
                )

                valid = data['pred_valid'][h].astype(bool)
                trans = torch.tensor(data['pred_trans'][h], dtype=torch.float32)
                rot = torch.tensor(data['pred_rot'][h], dtype=torch.float32)
                pose = torch.tensor(data['pred_hand_pose'][h], dtype=torch.float32)
                betas = torch.tensor(data['pred_betas'][h], dtype=torch.float32)

                with torch.no_grad():
                    out = mano(
                        global_orient=rot,
                        hand_pose=pose,
                        betas=betas,
                        transl=trans
                    )
                    data[f'{side}_verts'] = out.vertices.numpy()
                    print(f"  Recomputed {side}_verts: {data[f'{side}_verts'].shape}")

        except Exception as e:
            print(f"  MANO recompute failed: {e}, using direct vertex smoothing")

    # ── Save ──
    np.savez(args.output, **data)
    print(f"\nSaved smoothed output → {args.output}")

    # ── Report jitter reduction ──
    orig = dict(np.load(args.input))
    for h, side in enumerate(['right', 'left']):
        valid = orig['pred_valid'][h].astype(bool)
        if valid.sum() < 3:
            continue
        orig_trans = orig['pred_trans'][h][valid]
        new_trans = data['pred_trans'][h][valid]
        orig_jitter = np.mean(np.linalg.norm(np.diff(orig_trans, axis=0), axis=1))
        new_jitter = np.mean(np.linalg.norm(np.diff(new_trans, axis=0), axis=1))
        reduction = (1 - new_jitter / orig_jitter) * 100
        print(f"  {side} jitter: {orig_jitter:.4f} → {new_jitter:.4f} ({reduction:+.1f}%)")


if __name__ == '__main__':
    main()
