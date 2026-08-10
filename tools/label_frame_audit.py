#!/usr/bin/env python
"""Audit the frame each take's object was labelled on, and recommend a better one.

SAM3D builds the object mesh from a SINGLE view: whatever the object looks like on the
labelled frame is what the mesh becomes. If the hand is already on the object there,
SAM2's mask has the covered part cut out of it, SAM3D sees a partial silhouette, and it
extrapolates a wrong shape -- typically stretched along one axis. That is exactly what
happened to take 7 (labelled on frame 70, inside the grasp interval [41, 88]: mesh came
out 1.9:1 anisotropic and ~2x oversized).

The check is cheap -- masks only, no mesh, no camera -- so it can run right after
reconstruction instead of waiting for the contact stage to expose the damage.

A fixed offset like "grasp start minus 5 frames" does NOT solve this: measured on take 7,
frame 36 (= start - 5) still had 3.1%% hand contamination and 13%% less object area than
frame 22, because the hand is already closing in by then. Pick by measurement.

Usage:
  python tools/label_frame_audit.py <ReconstructOutput subtree> [--max-occlusion 0.02]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ego_pipeline"))
from contact import frames as F                                        # noqa: E402


def hand_contamination(obj_mask, hand_mask, dilate_px=4) -> float:
    """Fraction of the object mask touched by (or immediately adjacent to) the hand."""
    import cv2
    if hand_mask is None:
        return 0.0
    k = np.ones((2 * dilate_px + 1,) * 2, np.uint8)
    near = cv2.dilate((hand_mask > 0).astype(np.uint8), k).astype(bool)
    return float((obj_mask & near).sum()) / max(int(obj_mask.sum()), 1)


def scan_take(take_dir: Path, max_occl: float) -> dict | None:
    import cv2

    lp = take_dir / "masks/objects/label_prompt.json"
    if not lp.exists():
        return None
    prompt = json.loads(lp.read_text())["objects"][0]
    labelled = int(prompt["frame_idx"])

    ga = take_dir / "grasp_annotation.json"
    intervals = []
    if ga.exists():
        ann = json.loads(ga.read_text())["annotations"]
        for side in ("left", "right"):
            intervals += [[int(a), int(b)] for a, b in ann.get(side, [])]

    def masks(f):
        om = cv2.imread(str(take_dir / f"masks/objects/frames/frame_{f:06d}_masks/object_0.png"),
                        cv2.IMREAD_UNCHANGED)
        if om is None:
            return None, None
        hm = None
        for side in ("right", "left"):
            p = take_dir / f"masks/hands/frames/frame_{f:06d}_masks/{side}_hand_0.png"
            h = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if h is not None:
                hm = h if hm is None else np.maximum(hm, h)
        return om > 0, hm

    n = len(list((take_dir / "masks/objects/frames").glob("frame_*_masks")))
    in_grasp = lambda f: any(a <= f <= b for a, b in intervals)          # noqa: E731

    rows = []
    for f in range(n):
        om, hm = masks(f)
        if om is None or not om.any():
            continue
        rows.append((f, int(om.sum()), hand_contamination(om, hm), in_grasp(f)))
    if not rows:
        return None

    cur = next((r for r in rows if r[0] == labelled), None)
    clean = [r for r in rows if not r[3] and r[2] <= max_occl]
    best = max(clean, key=lambda r: r[1]) if clean else None

    problems = []
    if cur is None:
        problems.append(f"labelled frame {labelled} has no object mask")
    else:
        if cur[3]:
            problems.append(f"labelled INSIDE a grasp interval {intervals} -- SAM3D saw a "
                            f"hand-occluded partial silhouette")
        if cur[2] > max_occl:
            problems.append(f"hand touches {cur[2]*100:.0f}% of the object mask there")
        if best and cur[1] < 0.75 * best[1]:
            problems.append(f"only {cur[1]} px of object visible; frame {best[0]} offers "
                            f"{best[1]} px ({best[1]/max(cur[1],1):.1f}x)")

    return dict(take=take_dir.name, path=take_dir, labelled=labelled, intervals=intervals,
                n_frames=n, cur=cur, best=best, problems=problems)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--max-occlusion", type=float, default=0.02,
                    help="hand/object mask overlap allowed in a candidate frame")
    a = ap.parse_args()

    takes = sorted({p.parents[2] for p in a.root.rglob("masks/objects/label_prompt.json")},
                   key=lambda p: (len(p.name), p.name))
    if not takes:
        print(f"no labelled takes under {a.root}")
        return 1

    hdr = f"{'take':>6}  {'labelled':>8}  {'obj px':>7}  {'hand%':>6}  {'grasp':>6}  " \
          f"{'best':>5}  {'obj px':>7}  verdict"
    print(hdr)
    print("-" * len(hdr))
    bad = []
    for d in filter(None, (scan_take(t, a.max_occlusion) for t in takes)):
        c, b = d["cur"], d["best"]
        v = "PROBLEM" if d["problems"] else "ok"
        if d["problems"]:
            bad.append(d)
        print(f"{d['take']:>6}  {d['labelled']:>8}  {c[1] if c else 0:>7}  "
              f"{(c[2]*100 if c else 0):>5.1f}%  {('YES' if c and c[3] else 'no'):>6}  "
              f"{b[0] if b else '-':>5}  {b[1] if b else 0:>7}  {v}")

    if not bad:
        print("\nall labelled frames are clean")
        return 0

    print(f"\n{len(bad)} take(s) need re-labelling:\n")
    for d in bad:
        print(f"  take {d['take']}: " + "; ".join(d["problems"]))
        if d["best"]:
            print(f"    -> re-label on frame {d['best'][0]} "
                  f"({d['best'][1]} px object, {d['best'][2]*100:.1f}% hand)")
    print("\nRe-label + rebuild (per take; run it in your own terminal, it opens a browser):")
    for d in bad:
        vid = d["take"] + ".mp4"
        print(f"  ./reconstruct.sh <Data>/.../{vid} --dataset egodex --root <Data root> "
              f"--web --force --keep-interim      # go to frame {d['best'][0] if d['best'] else '?'}")
    print("\nNOTE: run_batch_queue has no per-step switch, so --force redoes all 8 steps "
          "(vipe and hawor included), not just the object branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
