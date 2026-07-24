#!/usr/bin/env python3
"""
Batch-evaluate grasp_detect over every annotated clip and report aggregate metrics.

Finds all Data/**/grasp_annotation.json, maps each to its reconstruction npz,
runs detection, and prints per-hand IoU / precision / recall plus a macro average.
Use this to (re)tune thresholds as more annotated clips are added.
"""
import glob, json, os
import numpy as np
from grasp_detect import detect, evaluate, HAND_NAME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ann_to_npz(ann_path):
    rel = ann_path.split("HOI4D_release/")[1].split("/align_rgb")[0]
    return os.path.join(ROOT, "Output/ReconstructOutput/hoi4d", rel, "world_fused.npz"), rel


def main():
    anns = sorted(glob.glob(os.path.join(ROOT, "Data/**/grasp_annotation.json"), recursive=True))
    rows = []
    for ann in anns:
        npz, rel = ann_to_npz(ann)
        if not os.path.exists(npz):
            print(f"skip (no recon): {rel}"); continue
        gt = json.load(open(ann))["annotations"]
        results, _ = detect(npz)
        name2idx = {v: k for k, v in HAND_NAME.items()}
        for side, segs_gt in gt.items():
            if not segs_gt:
                continue
            h = name2idx.get(side)
            det = results.get(h, [])
            m = evaluate(det, segs_gt)
            rows.append(m)
            print(f"{rel:45s} [{side:5s}] det={det}")
            print(f"{'':45s}         gt ={segs_gt}  {m}")
    if rows:
        agg = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("IoU", "precision", "recall")}
        print("\n==== MACRO AVERAGE over %d annotated hand-tracks ====" % len(rows), agg)


if __name__ == "__main__":
    main()
