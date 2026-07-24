#!/usr/bin/env python3
"""
For each of the 10 HOI4D seqs: run ObjectMaskStage (auto mask) then STEP_5's
best-frame selection (object_io.pick_best_frame), and mark the chosen frame.

Outputs per seq: <out>/<seq>/selected_frame.png (chosen frame RGB + magenta mask
+ caption) and object_mask.npz. Plus a 2x5 overview of all 10 chosen frames.

Run: ~/anaconda3/envs/biv2ap/bin/python tools/select_frame_hoi4d.py [--n 30]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from glob import glob
from scipy.ndimage import label

BIV2AP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BIV2AP)

from ego_pipeline.context import EgoContext
from ego_pipeline.stages.object_mask_stage import ObjectMaskStage
from ego_pipeline.utils.object_io import pick_best_frame

V2AP = "/home/lyh/Project/V2AP"
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


def frame_stats(m):
    mb = m > 0
    area = int(mb.sum())
    lab, n = label(mb)
    largest = max((lab == k).sum() for k in range(1, n + 1)) if n else 0
    frag = largest / area if area else 0.0
    border = bool(mb[0].any() or mb[-1].any() or mb[:, 0].any() or mb[:, -1].any())
    return area, frag, border


def overlay(frame, mask):
    out = frame.copy()
    m = mask > 0
    out[m] = (0.45 * out[m] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
    ys, xs = np.where(m)
    if len(xs):
        cv2.rectangle(out, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(BIV2AP, "Output", "select_frame_batch"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = sorted(os.listdir(DEPTH_BASE))
    picks = []
    for i, seq in enumerate(seqs):
        print(f"\n##### [{i+1}/{len(seqs)}] {seq} #####", flush=True)
        out_dir = os.path.join(args.out, seq)
        os.makedirs(out_dir, exist_ok=True)
        ctx = build_ctx(seq, out_dir, args.n)
        ctx = ObjectMaskStage(min_area=200).run(ctx)
        om = ctx.objects["object_mask"]
        np.savez_compressed(os.path.join(out_dir, "object_mask.npz"), object_mask=om)
        idx = pick_best_frame(om)
        area, frag, border = frame_stats(om[idx])
        ov = overlay(ctx.frames[idx], om[idx])
        cv2.imwrite(os.path.join(out_dir, "selected_frame.png"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        picks.append((seq, idx, len(om), area, frag, border, ov))
        print(f"  picked subset-frame {idx}/{len(om)-1}  area={area}px "
              f"frag={frag:.2f} border={border}")

    # 2x5 overview of all 10 chosen frames
    fig, axes = plt.subplots(2, 5, figsize=(20, 6))
    for ax, (seq, idx, n, area, frag, border, ov) in zip(axes.ravel(), picks):
        ax.imshow(ov)
        short = seq.split("_", 1)[1] if "_" in seq else seq
        ax.set_title(f"{short}\nbest f{idx}/{n-1} area={area} frag={frag:.2f}", fontsize=7)
        ax.axis("off")
    fig.suptitle("STEP_5 pick_best_frame — chosen frame per HOI4D seq "
                 "(magenta=mask, green=bbox)", fontsize=12)
    fig.tight_layout()
    ovw = os.path.join(args.out, "selected_overview.png")
    fig.savefig(ovw, dpi=95); plt.close(fig)

    print("\n========= SUMMARY =========")
    for seq, idx, n, area, frag, border, _ in picks:
        print(f"  {seq:40s} best=f{idx:<2d}/{n-1}  area={area:6d}  frag={frag:.2f}  border={border}")
    print(f"\noverview -> {ovw}")


if __name__ == "__main__":
    main()
