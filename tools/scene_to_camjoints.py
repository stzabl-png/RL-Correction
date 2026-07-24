#!/usr/bin/env python
"""World-frame hand joints (replay npz) -> per-frame CAMERA-frame joints for video overlay,
using the scene reconstruction's moving camera (c2w). Each frame's joints are expressed in
that frame's camera coords, so traj_viz can project them with the static intrinsics K.

  python tools/scene_to_camjoints.py --replay replay_world.npz --scene scene_min.npz --out camjoints.npz
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--replay", required=True, help="npz with joints_left/right in the scene WORLD frame")
ap.add_argument("--scene", required=True, help="scene_reconstruction_minimal npz (has c2w, K)")
ap.add_argument("--out", required=True)
args = ap.parse_args()

rep = np.load(args.replay)
scn = np.load(args.scene, allow_pickle=True)
c2w = scn["c2w"].astype(np.float64)                 # (N,4,4) camera->world
N = len(c2w)
K = scn["K"].astype(np.float64)

out = {"frames": np.arange(N, dtype=np.int32),
       "fps": np.float32(float(scn["video_fps"]) if "video_fps" in scn.files else 15.0)}
for side in ("left", "right"):
    if f"joints_{side}" not in rep.files:
        continue
    jw = np.asarray(rep[f"joints_{side}"], np.float64)   # (N,21,3) world
    jc = np.full_like(jw, np.nan)
    for i in range(min(N, len(jw))):
        w2c = np.linalg.inv(c2w[i])
        jc[i] = jw[i] @ w2c[:3, :3].T + w2c[:3, 3]       # world -> camera-i
    out[f"joints_{side}"] = jc.astype(np.float32)
    if f"valid_{side}" in rep.files:
        out[f"valid_{side}"] = rep[f"valid_{side}"]

np.savez_compressed(args.out, **out)
print(f"[cam] {args.out}: N={N}  K fx={K[0,0]:.1f} cx={K[0,2]:.1f}  "
      f"sides={[k for k in out if k.startswith('joints')]}")
print(f"  use:  python tools/traj_viz.py --video <mp4> --traj {args.out} "
      f"--intrinsics {K[0,0]:.1f},{K[1,1]:.1f},{K[0,2]:.1f},{K[1,2]:.1f} --mano --mano-traj")
