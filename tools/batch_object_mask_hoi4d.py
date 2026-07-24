#!/usr/bin/env python3
"""
Run ObjectMaskStage (Kailang's auto hand/object mask: EgoHOS L/R + SAM2 propagate)
on the 10 manually-curated HOI4D sequences and save an instance-colored montage
per sequence so the auto-mask quality can be eyeballed.

Instance colors: 1=left-hand obj (red), 2=right-hand obj (green), 3=both (blue).

Run: ~/anaconda3/envs/biv2ap/bin/python tools/batch_object_mask_hoi4d.py [--n 16]
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

V2AP = "/home/lyh/Project/V2AP"
DEPTH_BASE = os.path.join(V2AP, "data_hub/ProcessedData/egocentric_depth/hoi4d")
RGB_ROOTS = [os.path.join(BIV2AP, "Data/HOI4D/HOI4D_release"),
             os.path.join(V2AP, "data/egocentric/hoi4d/HOI4D_release")]
RGB_REL = "align_rgb/image/extracted_images"
COLORS = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}  # L / R / both (RGB)


def find_rgb_dir(seq):
    rel = os.path.join(seq.replace("_", "/"), RGB_REL)
    for root in RGB_ROOTS:
        d = os.path.join(root, rel)
        if glob(os.path.join(d, "*.jpg")):
            return d
    raise FileNotFoundError(seq)


def build_ctx(seq, out_dir, n_max):
    depths = np.load(os.path.join(DEPTH_BASE, seq, "depth.npz"))["depths"].astype(np.float32)
    N, Hd, Wd = depths.shape
    sel = np.linspace(0, N - 1, min(n_max, N)).round().astype(int)
    rgb_files = sorted(glob(os.path.join(find_rgb_dir(seq), "*.jpg")))
    step = max(1, len(rgb_files) // N)
    frames = np.empty((len(sel), Hd, Wd, 3), np.uint8)
    for k, i in enumerate(sel):
        bgr = cv2.imread(rgb_files[min(int(i) * step, len(rgb_files) - 1)])
        frames[k] = cv2.cvtColor(cv2.resize(bgr, (Wd, Hd)), cv2.COLOR_BGR2RGB)
    ctx = EgoContext(video_path="", output_dir=out_dir)
    ctx.frames = frames
    return ctx


def overlay(frame_rgb, inst):
    out = frame_rgb.copy()
    for v, col in COLORS.items():
        m = inst == v
        if m.any():
            out[m] = (0.45 * out[m] + 0.55 * np.array(col)).astype(np.uint8)
    return out


def montage(seq, frames, inst, cov, out_png, k=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(frames)
    idx = np.linspace(0, n - 1, min(k, n)).round().astype(int)
    cols = 3
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.6))
    for ax, j in zip(np.atleast_1d(axes).ravel(), idx):
        present = sorted(int(v) for v in np.unique(inst[j]) if v > 0)
        ax.imshow(overlay(frames[j], inst[j]))
        ax.set_title(f"f{j}  inst={present}", fontsize=8); ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(idx):]:
        ax.axis("off")
    fig.suptitle(f"{seq}   object-frames {cov}/{n}   (1=L red 2=R green 3=both blue)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="frames per seq (CPU EgoHOS cost)")
    ap.add_argument("--out", default=os.path.join(BIV2AP, "Output", "obj_mask_batch"))
    args = ap.parse_args()

    seqs = sorted(os.listdir(DEPTH_BASE))
    summary = []
    for i, seq in enumerate(seqs):
        print(f"\n########## [{i+1}/{len(seqs)}] {seq} ##########", flush=True)
        out_dir = os.path.join(args.out, seq)
        try:
            ctx = build_ctx(seq, out_dir, args.n)
            ctx = ObjectMaskStage(min_area=200).run(ctx)
            inst = ctx.objects["instance_mask"]
            om = ctx.objects["object_mask"]
            cov = int((om > 0).any(axis=(1, 2)).sum())
            ids = sorted(int(v) for v in np.unique(inst) if v > 0)
            png = os.path.join(out_dir, "mask_montage.png")
            montage(seq, ctx.frames, inst, cov, png)
            summary.append((seq, f"{cov}/{len(om)}", ids, png))
            print(f"  -> {png}  object-frames={cov}/{len(om)} instance_ids={ids}")
        except Exception as e:
            import traceback; traceback.print_exc()
            summary.append((seq, "FAIL", str(e), ""))

    print("\n================ SUMMARY ================")
    for seq, cov, ids, png in summary:
        print(f"  {seq:40s} obj-frames={cov:8s} ids={ids}")


if __name__ == "__main__":
    main()
