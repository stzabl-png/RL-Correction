#!/usr/bin/env python
"""HOI4D objpose annotations -> obj_pose npz (camera frame). No hand GT needed, so it
works on takes that lack Hand_pose pickles (hands come from HaWoR). Feeds the --object
input of hawor_to_world_replay.py.

  python extract_hoi4d_objpose.py --hoi4d-root <root> --rel <rel> --out <npz> [--fps 15]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--hoi4d-root", required=True)
ap.add_argument("--rel", required=True, help="ZY.../Cxx/Nxx/.../T#")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--target-label", default=None, help="pick this label from dataList (default first)")
args = ap.parse_args()

obj_dir = os.path.join(args.hoi4d_root, "HOI4D_annotations", args.rel, "objpose")
frames = sorted(int(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(obj_dir, "*.json")))
if not frames:
    raise SystemExit(f"no objpose json under {obj_dir}")
T = len(frames)
obj_pose = np.zeros((T, 7), np.float32)
last = None
label = None
for i, idx in enumerate(frames):
    fp = os.path.join(obj_dir, f"{idx}.json")
    item = None
    if os.path.exists(fp):
        dl = json.load(open(fp)).get("dataList", [])
        if args.target_label:
            item = next((x for x in dl if x.get("label") == args.target_label), None)
        item = item or (dl[0] if dl else None)
    if item is None:
        obj_pose[i] = last if last is not None else [0, 0, 0, 1, 0, 0, 0]
        continue
    label = item.get("label")
    c, r = item["center"], item["rotation"]
    q = Rotation.from_euler("XYZ", [r["x"], r["y"], r["z"]]).as_quat()  # xyzw
    obj_pose[i] = [c["x"], c["y"], c["z"], q[3], q[0], q[1], q[2]]
    last = obj_pose[i]

os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
np.savez_compressed(args.out, obj_pose=obj_pose, frames=np.asarray(frames, np.int32),
                    fps=np.float32(args.fps), label=label or "")
print(f"saved {args.out}: T={T} frames {frames[0]}..{frames[-1]} label={label} "
      f"center span(cm)={((obj_pose[:, :3].max(0) - obj_pose[:, :3].min(0)) * 100).round(1).tolist()}")
