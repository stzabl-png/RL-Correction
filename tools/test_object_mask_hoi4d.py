#!/usr/bin/env python3
"""
Smoke-test the full ObjectMaskStage end-to-end across BOTH envs:
  EgoHOS obj1 (subprocess, egohos CPU env) -> SAM2 propagation (biv2ap GPU) ->
  dense per-frame object mask.

Loads a short contiguous HOI4D clip, runs ObjectMaskStage, reports coverage and
saves a few mask overlays.

Run:  ~/anaconda3/envs/biv2ap/bin/python tools/test_object_mask_hoi4d.py --seq <name>
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

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "ego_pipeline"))
from repo_paths import HOI4D_RELEASE  # noqa: E402

RGB_ROOTS = [
    os.path.join(BIV2AP, "Data/HOI4D/HOI4D_release"),
    str(HOI4D_RELEASE),
]
RGB_REL = "align_rgb/image/extracted_images"


def find_rgb_dir(seq):
    rel = os.path.join(seq.replace("_", "/"), RGB_REL)
    for root in RGB_ROOTS:
        d = os.path.join(root, rel)
        if glob(os.path.join(d, "*.jpg")):
            return d
    raise FileNotFoundError(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="ZY20210800001_H1_C3_N23_S19_s01_T1")
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--downscale", type=float, default=2.0)
    ap.add_argument("--out", default=os.path.join(BIV2AP, "Output", "obj_mask_test"))
    args = ap.parse_args()

    files = sorted(glob(os.path.join(find_rgb_dir(args.seq), "*.jpg")))
    idx = list(range(args.start, min(args.start + args.n * args.step, len(files)), args.step))
    frames = []
    for i in idx:
        bgr = cv2.imread(files[i])
        if args.downscale != 1.0:
            bgr = cv2.resize(bgr, (int(bgr.shape[1] / args.downscale),
                                   int(bgr.shape[0] / args.downscale)))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    frames = np.stack(frames)
    print(f"[ctx] {args.seq}: {len(frames)} frames @ {frames.shape[1]}x{frames.shape[2]}")

    out_dir = os.path.join(args.out, args.seq)
    ctx = EgoContext(video_path="", output_dir=out_dir)
    ctx.frames = frames

    stage = ObjectMaskStage(min_area=200)
    missing = stage.check_deps(ctx)
    if missing:
        print("[FAIL] missing deps:", missing)
        sys.exit(1)
    ctx = stage.run(ctx)

    om = ctx.objects["object_mask"]
    inst = ctx.objects["instance_mask"]
    cov = (om > 0).any(axis=(1, 2)).sum()
    print(f"\n[OK] object_mask={om.shape} frames_with_object={cov}/{len(om)}")
    print(f"[OK] instance ids present: {sorted(np.unique(inst).tolist())}")

    ovl = os.path.join(out_dir, "overlays")
    os.makedirs(ovl, exist_ok=True)
    for j in range(0, len(frames), max(1, len(frames) // 5)):
        bgr = cv2.cvtColor(frames[j], cv2.COLOR_RGB2BGR).copy()
        m = om[j] > 0
        bgr[m] = (0.4 * bgr[m] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(ovl, f"{j:03d}.png"), bgr)
    print(f"[OK] overlays -> {ovl}")


if __name__ == "__main__":
    main()
