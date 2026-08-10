#!/usr/bin/env python
"""Decide whether a take's data is good enough to be worth using, and record the verdict.

Some clips are simply not reconstructable: the object sits inside a bowl and is partly
hidden from frame 0, then the hand reaches in and hides more of it. There is no clean
view for SAM3D to build a mesh from, and no uncontaminated frame for the contact
evidence. Chasing those with better parameters is wasted effort -- they should be
identified once and skipped from then on.

Two independent rejection criteria, both read off measurements rather than guessed:

  object never matches      median object-projection IoU (scored only where the object
                            is observable) below `--min-iou`. The reconstructed mesh/pose
                            simply does not agree with what the camera sees, at any frame.
  no clean view exists      fewer than `--min-clean-frames` frames outside the grasp with
                            the hand touching <=2% of the object mask, or the best such
                            frame is smaller than `--min-clean-px`.

The thresholds sit in the gap the data itself shows: across this EgoDex batch the median
IoU is 0.24 for the one broken take and >=0.54 for every other; clean-frame counts are 3
for the one hopeless take and >=20 for the rest.

Writes <take>/data_quality.json. contact_align_heatmap.py reads it and refuses to run on
a rejected take unless --ignore-quality is passed.

Usage:
  python tools/take_quality_gate.py <ReconstructOutput subtree> [--write] [--min-iou 0.35]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ego_pipeline"))
from contact import frames as F                                        # noqa: E402

SCHEMA = "take_data_quality_v1"


def annotated_side(take_dir: Path) -> str:
    """Which hand the grasp annotation is for (takes 10 and 12 in this batch are left)."""
    ga = take_dir / "grasp_annotation.json"
    if ga.exists():
        ann = json.loads(ga.read_text())["annotations"]
        for side in ("right", "left"):
            if ann.get(side):
                return side
    return "right"


def measure(take_dir: Path, sample=12) -> dict:
    import cv2
    import trimesh

    side = annotated_side(take_dir)
    take = F.load_take(take_dir, side=side, require_qpos=False)
    mesh = trimesh.load(take.obj_mesh_path, force="mesh", process=False)
    grasp = set(take.contact_frames().tolist()) if take.intervals else set()

    ious = []
    for f in range(0, take.Tv, max(1, take.Tv // sample)):
        try:
            ious.append(F.object_projection_iou(take, f, mesh)["iou"])
        except Exception:
            pass

    contamination, clean = [], []
    k = np.ones((9, 9), np.uint8)
    for f in range(take.Tv):
        om = cv2.imread(str(take.mask_path("object", f)), cv2.IMREAD_UNCHANGED)
        if om is None:
            continue
        om = om > 0
        if not om.any():
            continue
        hm = None
        for s in ("right", "left"):
            p = take_dir / f"masks/hands/frames/frame_{f:06d}_masks/{s}_hand_0.png"
            h = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if h is not None:
                hm = h if hm is None else np.maximum(hm, h)
        c = 0.0 if hm is None else float(
            (om & cv2.dilate((hm > 0).astype(np.uint8), k).astype(bool)).sum()) / max(int(om.sum()), 1)
        contamination.append(c)
        if c <= 0.02 and f not in grasp:
            clean.append(int(om.sum()))

    return dict(
        take=take_dir.name, side=side, num_frames=take.Tv, intervals=take.intervals,
        object_iou_median=float(np.median(ious)) if ious else float("nan"),
        object_iou_max=float(np.max(ious)) if ious else float("nan"),
        clean_frames=len(clean),
        best_clean_object_px=int(max(clean)) if clean else 0,
        hand_contamination_median=float(np.median(contamination)) if contamination else float("nan"),
        mesh_extents_cm=[round(float(x) * 100, 1) for x in mesh.extents],
        mesh_anisotropy=float(max(mesh.extents) / max(min(mesh.extents), 1e-9)),
    )


def judge(m: dict, min_iou: float, min_clean_frames: int, min_clean_px: int) -> tuple:
    reasons = []
    if not (m["object_iou_median"] >= min_iou):
        reasons.append(
            f"the reconstructed object never matches the image (median projection IoU "
            f"{m['object_iou_median']:.2f} < {min_iou}); its mesh is "
            f"{m['mesh_extents_cm']} cm, {m['mesh_anisotropy']:.1f}:1")
    if m["clean_frames"] < min_clean_frames or m["best_clean_object_px"] < min_clean_px:
        reasons.append(
            f"no clean view of the object exists ({m['clean_frames']} frames outside the "
            f"grasp with <=2% hand overlap, best {m['best_clean_object_px']} px; hand "
            f"covers {m['hand_contamination_median']*100:.0f}% of the object in a median "
            f"frame) -- occluded throughout, so re-labelling cannot help")
    return ("rejected" if reasons else "usable"), reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--write", action="store_true", help="write <take>/data_quality.json")
    ap.add_argument("--min-iou", type=float, default=0.35)
    ap.add_argument("--min-clean-frames", type=int, default=10)
    ap.add_argument("--min-clean-px", type=int, default=3000)
    a = ap.parse_args()

    takes = sorted({p.parent for p in a.root.rglob("world_fused.npz") if "interim" not in p.parts},
                   key=lambda p: (len(p.name), p.name))
    hdr = (f"{'take':>6} {'side':>5} {'objIoU':>7} {'clean':>6} {'bestpx':>7} "
           f"{'hand%':>6} {'aniso':>6}  verdict")
    print(hdr)
    print("-" * len(hdr))
    rejected = []
    for t in takes:
        try:
            m = measure(t)
        except Exception as e:
            print(f"{t.name:>6}  measurement failed: {e}")
            continue
        v, reasons = judge(m, a.min_iou, a.min_clean_frames, a.min_clean_px)
        m.update(schema_version=SCHEMA, verdict=v, reasons=reasons,
                 thresholds=dict(min_iou=a.min_iou, min_clean_frames=a.min_clean_frames,
                                 min_clean_px=a.min_clean_px))
        print(f"{m['take']:>6} {m['side']:>5} {m['object_iou_median']:>7.3f} "
              f"{m['clean_frames']:>6} {m['best_clean_object_px']:>7} "
              f"{m['hand_contamination_median']*100:>5.1f}% {m['mesh_anisotropy']:>6.2f}  {v}")
        if v == "rejected":
            rejected.append(m)
        if a.write:
            (t / "data_quality.json").write_text(json.dumps(m, indent=2))

    print(f"\n{len(takes)} takes | usable {len(takes)-len(rejected)} | rejected {len(rejected)}")
    for m in rejected:
        print(f"\n  take {m['take']} REJECTED")
        for r in m["reasons"]:
            print(f"    - {r}")
    if a.write:
        print(f"\nwrote data_quality.json into {len(takes)} takes")
    else:
        print("\n(dry run -- pass --write to record the verdicts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
