#!/usr/bin/env python3
"""
Full object-branch smoke test on HOI4D, end-to-end across both envs:

  ObjectMaskStage  (EgoHOS egohos-CPU subprocess -> SAM2 propagate, biv2ap GPU)
    -> ObjectMeshStage  (SAM3D single-image recon + depth-based scale.json)
    -> ObjectPoseStage  (FoundationPose register@mesh_frame_idx + track)

Builds an EgoContext from V2AP precomputed depth/K + HOI4D RGB (MegaSAM-subsampled
to N_depth, normalized to depth resolution). No precomputed mask/mesh used —
everything is produced by the new stages.

Run: ~/anaconda3/envs/biv2ap/bin/python tools/test_object_branch_full_hoi4d.py --seq <name>
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from glob import glob

BIV2AP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BIV2AP)

from ego_pipeline.context import EgoContext
from ego_pipeline.stages.object_mask_stage import ObjectMaskStage
from ego_pipeline.stages.object_mesh_stage import ObjectMeshStage
from ego_pipeline.stages.object_pose_stage import ObjectPoseStage

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "ego_pipeline"))
from repo_paths import V2AP_ROOT  # noqa: E402

V2AP = str(V2AP_ROOT)
DEPTH_BASE = os.path.join(V2AP, "data_hub/ProcessedData/egocentric_depth/hoi4d")
RGB_ROOTS = [os.path.join(BIV2AP, "Data/HOI4D/HOI4D_release"),
             os.path.join(V2AP, "data/egocentric/hoi4d/HOI4D_release")]
RGB_REL = "align_rgb/image/extracted_images"


def find_rgb_dir(seq):
    rel = os.path.join(seq.replace("_", "/"), RGB_REL)
    for root in RGB_ROOTS:
        d = os.path.join(root, rel)
        if glob(os.path.join(d, "*.jpg")):
            return d
    raise FileNotFoundError(seq)


def build_ctx(seq, out_dir, n_max):
    depths = np.load(os.path.join(DEPTH_BASE, seq, "depth.npz"))["depths"].astype(np.float32)
    K = np.load(os.path.join(DEPTH_BASE, seq, "K.npy")).astype(np.float64)
    N, Hd, Wd = depths.shape
    if n_max and N > n_max:                       # uniform subsample to cap runtime
        sel = np.linspace(0, N - 1, n_max).round().astype(int)
        depths = depths[sel]
    else:
        sel = np.arange(N)

    rgb_files = sorted(glob(os.path.join(find_rgb_dir(seq), "*.jpg")))
    step = max(1, len(rgb_files) // N)
    frames = np.empty((len(sel), Hd, Wd, 3), np.uint8)
    for k, i in enumerate(sel):
        bgr = cv2.imread(rgb_files[min(int(i) * step, len(rgb_files) - 1)])
        frames[k] = cv2.cvtColor(cv2.resize(bgr, (Wd, Hd)), cv2.COLOR_BGR2RGB)

    print(f"[ctx] {seq}: {len(frames)} frames @ {Hd}x{Wd}")
    ctx = EgoContext(video_path="", output_dir=out_dir)
    ctx.frames = frames
    ctx.depth = depths
    ctx.intrinsics = K
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="ZY20210800001_H1_C3_N23_S19_s01_T1")
    ap.add_argument("--n-max", type=int, default=30)
    ap.add_argument("--pose-backend", choices=["pp_ros", "local"], default="pp_ros")
    ap.add_argument("--out", default=os.path.join(BIV2AP, "Output", "obj_branch_full"))
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.seq)
    ctx = build_ctx(args.seq, out_dir, args.n_max)

    for stage in [ObjectMaskStage(min_area=200), ObjectMeshStage(),
                  ObjectPoseStage(backend=args.pose_backend, debug=1)]:
        print(f"\n{'='*56}\n  {stage.name()}\n{'='*56}")
        miss = stage.check_deps(ctx)
        if miss:
            print(f"[FAIL] {stage.name()} missing: {miss}"); sys.exit(1)
        ctx = stage.run(ctx)

    o = ctx.objects
    print(f"\n{'#'*56}\n[RESULT] {args.seq}")
    print(f"  object_mask: {o['object_mask'].shape} "
          f"frames_with_obj={(o['object_mask']>0).any(axis=(1,2)).sum()}")
    print(f"  mesh: {o['mesh_path']}  (recon frame {o['mesh_frame_idx']})")
    print(f"  pose: {o['pose'].shape}  reg_idx={o.get('pose_reg_idx')}  -> {o['pose_dir']}")
    t = o["pose"][:, :3, 3]
    print(f"  translation z range (m): [{t[:,2].min():.3f}, {t[:,2].max():.3f}]")


if __name__ == "__main__":
    main()
