#!/usr/bin/env python3
"""
Convert hawor NPZ output to world_space_res.pth format for vis_hand_motion.py.

Usage:
  python npz_to_pth.py --npz hawor_smooth.npz --out_dir /path/to/seq_dir
  # This creates world_space_res.pth in the specified directory
"""

import argparse
import numpy as np
import torch
import joblib
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True, help='Input NPZ file')
    parser.add_argument('--out_dir', required=True, help='Output directory (seq_dir for vis)')
    args = parser.parse_args()

    d = np.load(args.npz)
    
    trans = torch.tensor(d['pred_trans'], dtype=torch.float32)       # (2, T, 3)
    rot   = torch.tensor(d['pred_rot'], dtype=torch.float32)         # (2, T, 3)
    pose  = torch.tensor(d['pred_hand_pose'], dtype=torch.float32)   # (2, T, 45)
    betas = torch.tensor(d['pred_betas'], dtype=torch.float32)       # (2, T, 10)
    valid = d['pred_valid'].astype(bool)                              # (2, T)

    ws = (trans, rot, pose, betas, valid)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'world_space_res.pth')
    joblib.dump(ws, out_path)
    print(f"Saved → {out_path}  ({trans.shape[1]} frames)")


if __name__ == '__main__':
    main()
