#!/usr/bin/env python3
"""
Prepare HOI4D Ground Truth data for Ego Pipeline evaluation.

Extracts GT from HOI4D raw data and saves in a unified format:
  - K_gt.npy:         (3, 3) camera intrinsics
  - poses_c2w_gt.npy: (N, 4, 4) camera-to-world poses
  - depth_gt.npz:     (N, H, W) metric depth in metres
  - mano_gt.npz:      MANO parameters (trans/rot/pose/betas/valid/kps2D)

Usage:
    python tools/prepare_gt_for_eval.py \
        --seq ZY20210800001/H1/C1/N19/S100/s02/T1 \
        --output Output/PipelineOutput/HOI4D/ZY20210800001/H1/C1/N19/S100/s02/T1/gt/
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.dirname(SCRIPT_DIR)
HOI4D_ROOT = os.path.join(BIV2AP_DIR, "Data", "HOI4D")

# HOI4D RealSense D435i intrinsics (identical across all 2317 sessions)
GT_INTRINSICS = np.array(
    [[1376.8,    0.0, 967.7],
     [   0.0, 1376.2, 526.8],
     [   0.0,    0.0,   1.0]],
    dtype=np.float64,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera Intrinsics
# ═══════════════════════════════════════════════════════════════════════════════

def save_intrinsics(output_dir: str) -> None:
    """Save GT camera intrinsics (fixed RealSense D435i)."""
    path = os.path.join(output_dir, "K_gt.npy")
    np.save(path, GT_INTRINSICS)
    K = GT_INTRINSICS
    print(f"  ✅ K_gt.npy: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera Poses (from 3Dseg/output.log)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_output_log(log_path: str) -> np.ndarray:
    """Parse 3Dseg/output.log into (N, 4, 4) c2w poses.

    Format: each frame = 5 lines:
        header: frame_idx start_frame end_frame
        4 lines of 4x4 matrix (c2w, OpenCV convention)
    """
    with open(log_path) as f:
        lines = f.readlines()

    n_frames = len(lines) // 5
    poses = np.zeros((n_frames, 4, 4), dtype=np.float32)

    for i in range(n_frames):
        base = i * 5
        # Skip header line (lines[base])
        for r in range(4):
            row = lines[base + 1 + r].strip().split()
            poses[i, r] = [float(x) for x in row]

    return poses


def save_poses(seq_path: str, output_dir: str) -> None:
    """Extract and save GT camera poses."""
    log_path = os.path.join(
        HOI4D_ROOT, "HOI4D_annotations", seq_path, "3Dseg", "output.log"
    )
    if not os.path.exists(log_path):
        print(f"  ⚠️  Camera pose log not found: {log_path}")
        return

    poses = parse_output_log(log_path)
    path = os.path.join(output_dir, "poses_c2w_gt.npy")
    np.save(path, poses)

    # Sanity checks
    det0 = np.linalg.det(poses[0, :3, :3])
    is_identity = np.allclose(poses[0], np.eye(4), atol=1e-5)
    t_range = np.linalg.norm(poses[:, :3, 3], axis=1)
    print(f"  ✅ poses_c2w_gt.npy: {poses.shape}, "
          f"frame0={'identity' if is_identity else 'NOT identity'}, "
          f"det(R[0])={det0:.4f}, "
          f"t_range=[{t_range.min():.4f}, {t_range.max():.4f}]m")


# ═══════════════════════════════════════════════════════════════════════════════
#  Depth (from align_depth/depth_video.avi)
# ═══════════════════════════════════════════════════════════════════════════════

def decode_depth_video(avi_path: str) -> np.ndarray | None:
    """Decode HOI4D depth AVI to (N, H, W) float32 metres.

    HOI4D encodes 16-bit depth across 2 channels of an AVI:
        depth_uint16 = ch0 | (ch1 << 8)     [millimetres]
        depth_m = depth_uint16 / 1000.0      [metres]
    """
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        print(f"  ❌ Cannot open depth video: {avi_path}")
        return None

    depths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 16-bit depth from 2 channels
        depth_raw = (frame[:, :, 0].astype(np.uint16)
                     | (frame[:, :, 1].astype(np.uint16) << 8))
        depth_m = depth_raw.astype(np.float32) / 1000.0
        depth_m[depth_raw == 0] = 0.0  # invalid pixels
        depths.append(depth_m)

    cap.release()
    return np.stack(depths) if depths else None


def save_depth(seq_path: str, output_dir: str) -> None:
    """Extract and save GT depth maps."""
    avi_path = os.path.join(
        HOI4D_ROOT, "HOI4D_depth_video", seq_path, "align_depth", "depth_video.avi"
    )
    if not os.path.exists(avi_path):
        print(f"  ⚠️  Depth video not found: {avi_path}")
        return

    print(f"  Decoding depth video...")
    depths = decode_depth_video(avi_path)
    if depths is None:
        return

    path = os.path.join(output_dir, "depth_gt.npz")
    np.savez_compressed(path, depths=depths)

    valid = depths > 0
    print(f"  ✅ depth_gt.npz: {depths.shape}, "
          f"range=[{depths[valid].min():.3f}, {depths[valid].max():.3f}]m, "
          f"valid={valid.mean()*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  MANO Hand Pose (from Hand_pose/handpose_{left,right}_hand)
# ═══════════════════════════════════════════════════════════════════════════════

def load_hand_pickles(hand_dir: str, n_frames: int) -> dict:
    """Load per-frame hand pose pickles into arrays.

    Each pickle contains:
        poseCoeff: (48,) → [0:3]=global_rot, [3:48]=joint_angles
        beta:      (10,)
        trans:     (3,)
        kps2D:     (21, 2)

    Returns dict with arrays of shape (n_frames, ...) and valid mask.
    """
    trans = np.zeros((n_frames, 3), dtype=np.float32)
    rot = np.zeros((n_frames, 3), dtype=np.float32)
    pose = np.zeros((n_frames, 45), dtype=np.float32)
    betas = np.zeros((n_frames, 10), dtype=np.float32)
    kps2D = np.zeros((n_frames, 21, 2), dtype=np.float32)
    valid = np.zeros(n_frames, dtype=bool)

    for frame_idx in range(n_frames):
        pkl_path = os.path.join(hand_dir, f"{frame_idx}.pickle")
        if not os.path.exists(pkl_path):
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        trans[frame_idx] = data["trans"]
        rot[frame_idx] = data["poseCoeff"][:3]    # global rotation (axis-angle)
        pose[frame_idx] = data["poseCoeff"][3:48]  # 15 joints × 3
        betas[frame_idx] = data["beta"]
        kps2D[frame_idx] = data["kps2D"]
        valid[frame_idx] = True

    return {
        "trans": trans,
        "rot": rot,
        "pose": pose,
        "betas": betas,
        "kps2D": kps2D,
        "valid": valid,
    }


def save_mano(seq_path: str, output_dir: str, n_frames: int = 300) -> None:
    """Extract and save GT MANO parameters for both hands.

    Output format matches Pipeline: dim 0 = [left=0, right=1]
    """
    # Initialize combined arrays
    trans = np.zeros((2, n_frames, 3), dtype=np.float32)
    rot = np.zeros((2, n_frames, 3), dtype=np.float32)
    pose = np.zeros((2, n_frames, 45), dtype=np.float32)
    betas = np.zeros((2, n_frames, 10), dtype=np.float32)
    kps2D = np.zeros((2, n_frames, 21, 2), dtype=np.float32)
    valid = np.zeros((2, n_frames), dtype=bool)

    side_names = ["left", "right"]
    side_dirs = ["handpose_left_hand", "handpose_right_hand"]

    for h, (side_name, side_dir) in enumerate(zip(side_names, side_dirs)):
        hand_dir = os.path.join(HOI4D_ROOT, "Hand_pose", side_dir, seq_path)

        if not os.path.isdir(hand_dir):
            print(f"  ⚠️  {side_name} hand GT not found: {hand_dir}")
            continue

        data = load_hand_pickles(hand_dir, n_frames)
        n_valid = data["valid"].sum()

        trans[h] = data["trans"]
        rot[h] = data["rot"]
        pose[h] = data["pose"]
        betas[h] = data["betas"]
        kps2D[h] = data["kps2D"]
        valid[h] = data["valid"]

        print(f"  ✅ {side_name} hand: {n_valid}/{n_frames} valid frames")

    path = os.path.join(output_dir, "mano_gt.npz")
    np.savez_compressed(
        path,
        trans=trans,
        rot=rot,
        pose=pose,
        betas=betas,
        kps2D=kps2D,
        valid=valid,
    )
    print(f"  ✅ mano_gt.npz saved")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Prepare HOI4D GT data for Ego Pipeline evaluation"
    )
    parser.add_argument(
        "--seq", required=True,
        help="HOI4D sequence relative path, e.g. ZY20210800001/H1/C1/N19/S100/s02/T1"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for GT files"
    )
    parser.add_argument(
        "--n-frames", type=int, default=300,
        help="Expected number of frames (default: 300)"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"{'='*60}")
    print(f"  Preparing GT for: {args.seq}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    # 1. Camera intrinsics
    print("[1/4] Camera Intrinsics")
    save_intrinsics(args.output)

    # 2. Camera poses
    print("\n[2/4] Camera Poses")
    save_poses(args.seq, args.output)

    # 3. Depth
    print("\n[3/4] Depth Maps")
    save_depth(args.seq, args.output)

    # 4. MANO
    print("\n[4/4] MANO Hand Pose")
    save_mano(args.seq, args.output, args.n_frames)

    print(f"\n{'='*60}")
    print(f"  GT preparation complete!")
    print(f"  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
