#!/usr/bin/env python
"""Combine a HANDS trajectory npz and an OBJECT trajectory npz into one replay npz
for sim/replay_isaacsim.py, aligned on shared video-frame indices.

Hands come from one source (e.g. HaWoR camera-frame joints), the object pose from
another (e.g. HOI4D GT obj_pose). Both must be in the SAME camera frame (meters).
Rows are aligned by their `frames` index; the output spans the object's frames (the
annotated action window), pulling the matching hand row for each.

  python tools/combine_hands_object.py \
      --hands hawor_camjoints_chair.npz --object replay_hoi4d_2hand.npz \
      --out replay_hawor_2hand.npz
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--hands", required=True, help="npz with joints_left/right (+ valid_*, frames)")
ap.add_argument("--object", required=True, help="npz with obj_pose (+ frames)")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=None, help="override fps (else from --object)")
args = ap.parse_args()

H = np.load(args.hands)
O = np.load(args.object)
hf = np.asarray(H["frames"]).astype(int) if "frames" in H.files else np.arange(len(H["joints_right"]))
of = np.asarray(O["frames"]).astype(int) if "frames" in O.files else np.arange(len(O["obj_pose"]))
h_row = {int(f): i for i, f in enumerate(hf)}

# master = object frames that also have hand data (keeps the annotated action window)
master = [int(f) for f in of if int(f) in h_row]
o_row = {int(f): i for i, f in enumerate(of)}
hi = np.array([h_row[f] for f in master])
oi = np.array([o_row[f] for f in master])
T = len(master)
if T == 0:
    raise SystemExit("no overlapping frames between hands and object npz")

out = {"obj_pose": np.asarray(O["obj_pose"])[oi].astype(np.float32),
       "frames": np.asarray(master, np.int32),
       "fps": np.float32(args.fps if args.fps else (O["fps"] if "fps" in O.files else 15.0))}
for s in ("left", "right"):
    if f"joints_{s}" in H.files:
        out[f"joints_{s}"] = np.asarray(H[f"joints_{s}"])[hi].astype(np.float32)
        v = H[f"valid_{s}"] if f"valid_{s}" in H.files else np.ones(len(hf))
        out[f"valid_{s}"] = np.asarray(v)[hi].astype(np.float32)

np.savez_compressed(args.out, **out)
print(f"[combine] {args.out}: T={T} frames {master[0]}..{master[-1]}  "
      f"hands={[k for k in out if k.startswith('joints')]}  "
      f"obj_pose{out['obj_pose'].shape}  fps={float(out['fps'])}")
