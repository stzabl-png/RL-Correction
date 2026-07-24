#!/usr/bin/env python
"""Estimate the world gravity-up vector for a scene-reconstruction package using ViPE's
GeoCalib (per-frame camera gravity) + the package c2w. Prints a ready --scene-rot up:... .

  python tools/estimate_gravity_geocalib.py --video <mp4> --scene <scene_min.npz>
"""
import argparse
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(VIPE_ROOT))
from vipe.priors.geocalib import GeoCalib  # noqa: E402

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "ego_pipeline"))
from repo_paths import VIPE_ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--frames", default="0,60,120,180,240")
args = ap.parse_args()

c2w = np.load(args.scene, allow_pickle=True)["c2w"].astype(np.float64)  # (N,4,4)
model = GeoCalib(weights="pinhole").cuda().eval()

cap = cv2.VideoCapture(args.video)
finds = [int(x) for x in args.frames.split(",")]
world_grav = []
for fi in finds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, img = cap.read()
    if not ok:
        continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().moveaxis(-1, 0) / 255.0   # (3,H,W)
    with torch.no_grad():
        res = model.calibrate(t.cuda(), shared_intrinsics=False)
    g_cam = res["gravity"].vec3d.squeeze().cpu().numpy()        # camera-frame gravity (down)
    wg = c2w[fi][:3, :3] @ g_cam                                # -> world frame
    world_grav.append(wg / np.linalg.norm(wg))
    print(f"  frame {fi}: gravity(cam)={g_cam.round(3)}  gravity(world)={(wg/np.linalg.norm(wg)).round(3)}")

up = np.mean(world_grav, axis=0); up /= np.linalg.norm(up)      # GeoCalib vec3d points UP
spread = np.degrees(np.arccos(np.clip([wg @ up for wg in world_grav], -1, 1)))
print(f"\nworld UP = {up.round(3)}   consistency spread = {spread.max():.0f} deg (low = reliable)")
print(f"\n--scene-rot up:{up[0]:.3f},{up[1]:.3f},{up[2]:.3f}")
