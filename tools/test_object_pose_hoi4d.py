#!/usr/bin/env python3
"""
Smoke-test the biv2ap object branch (ObjectPoseStage = FoundationPose) on HOI4D
using V2AP's PRE-COMPUTED assets — no EgoHOS, no SAM3D needed:

  * register-frame object mask : V2AP obj_recon_input/egocentric/<seq>/0.png  (manual)
  * object mesh + scale.json   : V2AP obj_meshes/hoi4d/<seq>/{mesh.ply,scale.json}
  * metric depth + K           : V2AP egocentric_depth/hoi4d/<seq>/{depth.npz,K.npy}
  * RGB frames                 : HOI4D extracted_images (MegaSAM-subsampled to N_depth)

Everything is normalized to DEPTH resolution before being placed into EgoContext
so ObjectPoseStage.build_fp_scene_dir (which assumes uniform frame/depth/K res)
behaves exactly like V2AP batch_obj_pose_ego.prepare_scene_ego.

Run:  ~/anaconda3/envs/biv2ap/bin/python tools/test_object_pose_hoi4d.py --seq <name>
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
from ego_pipeline.stages.object_pose_stage import ObjectPoseStage

V2AP = "/home/lyh/Project/V2AP"
DEPTH_BASE = os.path.join(V2AP, "data_hub/ProcessedData/egocentric_depth/hoi4d")
MESH_BASE = os.path.join(V2AP, "data_hub/ProcessedData/obj_meshes/hoi4d")
RECON_BASE = os.path.join(V2AP, "data_hub/ProcessedData/obj_recon_input/egocentric")
# HOI4D release frames: prefer Reconstruct_and_Retarget local copy, fall back to V2AP.
RGB_REL = "align_rgb/image/extracted_images"
RGB_ROOTS = [
    os.path.join(BIV2AP, "Data/HOI4D/HOI4D_release"),
    os.path.join(V2AP, "data/egocentric/hoi4d/HOI4D_release"),
]


def seq_to_release_path(seq: str) -> str:
    # ZY20210800004_H4_C18_N48_S364_s03_T2 -> ZY20210800004/H4/C18/N48/S364/s03/T2
    return seq.replace("_", "/")


def find_rgb_dir(seq: str) -> str:
    rel = os.path.join(seq_to_release_path(seq), RGB_REL)
    for root in RGB_ROOTS:
        d = os.path.join(root, rel)
        if os.path.isdir(d) and glob(os.path.join(d, "*.jpg")):
            return d
    raise FileNotFoundError(f"No RGB extracted_images for {seq} under {RGB_ROOTS}")


def build_ctx(seq: str, out_dir: str) -> EgoContext:
    depth_dir = os.path.join(DEPTH_BASE, seq)
    depths = np.load(os.path.join(depth_dir, "depth.npz"))["depths"].astype(np.float32)
    K = np.load(os.path.join(depth_dir, "K.npy")).astype(np.float64)  # at depth res
    N, Hd, Wd = depths.shape

    # RGB: MegaSAM subsampling (depth frame i -> rgb[i*step]), then resize to depth res.
    rgb_files = sorted(glob(os.path.join(find_rgb_dir(seq), "*.jpg")))
    step = max(1, len(rgb_files) // N)
    frames = np.empty((N, Hd, Wd, 3), np.uint8)
    for i in range(N):
        bgr = cv2.imread(rgb_files[min(i * step, len(rgb_files) - 1)])
        frames[i] = cv2.cvtColor(cv2.resize(bgr, (Wd, Hd)), cv2.COLOR_BGR2RGB)

    # Register-frame mask (manual), resize to depth res.
    mask_path = os.path.join(RECON_BASE, seq, "0.png")
    mask0 = cv2.resize(cv2.imread(mask_path, 0), (Wd, Hd), interpolation=cv2.INTER_NEAREST)

    mesh_path = os.path.join(MESH_BASE, seq, "mesh.ply")
    for p in (mask_path, mesh_path, os.path.join(MESH_BASE, seq, "scale.json")):
        assert os.path.exists(p), f"missing {p}"

    print(f"[ctx] {seq}: N={N} depth_res={Hd}x{Wd} "
          f"mask_fg={(mask0>0).mean()*100:.2f}% "
          f"depth(m) min={depths[depths>0].min():.3f} max={depths.max():.3f}")

    ctx = EgoContext(video_path="", output_dir=out_dir)
    ctx.frames = frames
    ctx.depth = depths
    ctx.intrinsics = K
    ctx.objects = {"mesh_path": mesh_path, "mesh_mask": (mask0 > 0).astype(np.uint8)}
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="ZY20210800004_H4_C18_N48_S364_s03_T2")
    ap.add_argument("--out", default=os.path.join(BIV2AP, "Output", "obj_pose_test"))
    ap.add_argument("--est-iter", type=int, default=5)
    ap.add_argument("--track-iter", type=int, default=2)
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.seq)
    os.makedirs(out_dir, exist_ok=True)
    ctx = build_ctx(args.seq, out_dir)

    stage = ObjectPoseStage(est_iter=args.est_iter, track_iter=args.track_iter, debug=1)
    missing = stage.check_deps(ctx)
    if missing:
        print("[FAIL] missing deps:", missing)
        sys.exit(1)

    ctx = stage.run(ctx)

    poses = ctx.objects["pose"]
    pose_dir = ctx.objects["pose_dir"]
    n_vis = len(glob(os.path.join(pose_dir, "track_vis", "*.png")))
    n_txt = len(glob(os.path.join(pose_dir, "ob_in_cam", "*.txt")))
    print(f"\n[OK] poses={poses.shape} ob_in_cam={n_txt} track_vis={n_vis}")
    print(f"[OK] frame0 ob_in_cam=\n{poses[0]}")
    t = poses[:, :3, 3]
    print(f"[OK] translation range (m): x[{t[:,0].min():.3f},{t[:,0].max():.3f}] "
          f"y[{t[:,1].min():.3f},{t[:,1].max():.3f}] z[{t[:,2].min():.3f},{t[:,2].max():.3f}]")
    print(f"[OK] outputs -> {pose_dir}")


if __name__ == "__main__":
    main()
