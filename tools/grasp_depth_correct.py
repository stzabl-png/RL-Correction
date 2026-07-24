#!/usr/bin/env python
"""Single rigid-translation grasp correction.

Decided approach: treat the ENTIRE hand trajectory as one rigid body and apply ONE
translation to the whole thing (no per-frame, no per-segment, no internal deformation).
The translation is the least-squares shift that brings the hand as close as possible to
the object across ALL grasp frames, leaving a target wrist<->object-centroid gap:

    T = mean_g(obj_g - wrist_g) - TARGET * mean_g(unit(obj_g - wrist_g))

where g ranges over every valid grasp frame (all frames inside the annotated intervals),
wrist = joints[0], obj = object_ob_in_world[:, :3, 3] (centroid). T is added to every
frame and every joint/vertex of the grasped hand, so the trajectory shape is untouched.

Inputs:
  world_fused.npz       : object_ob_in_world (Tv,4,4)
  replay_world.npz      : joints_{L,R} (Tv,21,3), mano_verts_{L,R}, valid_{L,R}
  grasp_annotation.json : {"annotations": {"left":[[a,b]...], "right":[...]}}

Usage:
  python tools/grasp_depth_correct.py --recon <world_fused.npz> --replay <replay_world.npz>
      --annot <grasp_annotation.json> [--target 0.06] [--out corrected.npz]
"""
import argparse
import json
import os

import numpy as np


def load_grasp(annot_path, n):
    ann = json.load(open(annot_path))["annotations"]
    out, tips = {}, {}
    for side in ("left", "right"):
        idx, bnd = [], []
        for a, b in ann.get(side, []):
            idx.extend(range(max(0, a), min(n - 1, b) + 1))
            bnd += [a, b]                                    # contact tips: grab=start, place=end
        out[side] = np.array(sorted(set(idx)), dtype=int)
        tips[side] = np.array(sorted(set(f for f in bnd if 0 <= f < n)), dtype=int)
    return out, tips


def fit_translation(wrist, obj, g, target):
    """One rigid shift over all grasp frames g, leaving a `target` wrist<->obj gap."""
    off = obj[g] - wrist[g]                                  # wrist -> obj vectors
    u = off / np.clip(np.linalg.norm(off, axis=1, keepdims=True), 1e-9, None)
    return off.mean(0) - target * u.mean(0)


def dist(wrist, obj, g):
    return np.linalg.norm(wrist[g] - obj[g], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--annot", required=True)
    ap.add_argument("--target", type=float, default=0.06, help="wrist<->object-centroid gap (m)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = np.load(args.recon)
    p = np.load(args.replay)
    obj = r["object_ob_in_world"][:, :3, 3]
    Tv = obj.shape[0]
    grasp, tips = load_grasp(args.annot, Tv)

    payload = {k: p[k] for k in p.files}
    for side, jkey, vkey, mkey in (
        ("left", "joints_left", "valid_left", "mano_verts_left"),
        ("right", "joints_right", "valid_right", "mano_verts_right"),
    ):
        g = grasp[side]
        if len(g) == 0:
            continue
        g = g[p[vkey][g] > 0.5]
        tp = tips[side][p[vkey][tips[side]] > 0.5]
        if len(g) == 0:
            continue
        wrist = p[jkey][:, 0, :]
        T = fit_translation(wrist, obj, g, args.target)
        wc = wrist + T                                       # single rigid shift, whole trajectory
        print(f"[{side}] {len(g)} grasp frames | ONE T = {np.round(T*100,1)}cm |T|={np.linalg.norm(T)*100:.1f}cm")
        print(f"  wrist<->obj over ALL grasp frames:  before mean={dist(wrist,obj,g).mean()*100:.1f}cm"
              f"  ->  after mean={dist(wc,obj,g).mean()*100:.1f} max={dist(wc,obj,g).max()*100:.1f}cm")
        print(f"  at contact tips {list(tp)}:")
        print(f"    before (cm): {np.round(dist(wrist,obj,tp)*100,1)}")
        print(f"    after  (cm): {np.round(dist(wc,obj,tp)*100,1)}")

        if args.out:
            payload[jkey] = p[jkey] + T                      # broadcast over frames & joints
            if mkey in payload and payload[mkey].shape[0] == Tv:
                payload[mkey] = p[mkey] + T
            print(f"  applied ONE T to {jkey}/{mkey} (whole trajectory rigid)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez_compressed(args.out, **payload)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
