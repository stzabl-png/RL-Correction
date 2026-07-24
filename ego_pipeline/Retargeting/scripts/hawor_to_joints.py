#!/usr/bin/env python
"""Convert a HaWoR sequence .npz (MANO params) -> a lightweight joints .npz that
magicdexmate's HaWoRSource replays. Run in the HaWoR env (needs MANO/smplx;
magicdexmate's own venvs stay MANO-free).

  python scripts/hawor_to_joints.py --in hawor_out.npz --out hawor_joints.npz [--fps 30]

Input  (keys from hawor/run_hawor_seq.py), [0]=left [1]=right:
  pred_trans (2,T,3)  pred_rot (2,T,3)  pred_hand_pose (2,T,45)
  pred_betas (2,T,10)  pred_valid (2,T)
Output:
  joints_left / joints_right : (T,21,3) float32, world meters, OpenPose order
                               (== MediaPipe order that magicdexmate expects)
  valid : (2,T) float32   fps : scalar

run_mano / run_mano_left (hawor.utils.process) return outputs["joints"] of shape
(B,T,21,3): MANO 16 joints + 5 fingertip verts, reordered to OpenPose by the
mano_wrapper joint_map. We take batch 0 per hand.
"""
import argparse
import os
import sys

import numpy as np
import torch

_DEFAULT_HAWOR = os.path.expanduser("~/Project/Reconstruct_and_Retarget/third_party/hawor")

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="inp", required=True, help="HaWoR sequence .npz")
ap.add_argument("--out", required=True, help="output joints .npz")
ap.add_argument("--fps", type=float, default=30.0, help="capture fps to record in the output")
ap.add_argument("--hawor-dir", default=_DEFAULT_HAWOR, help="HaWoR repo root (for hawor.utils.process)")
ap.add_argument("--cpu", action="store_true", help="run MANO on CPU")
args = ap.parse_args()

sys.path.insert(0, args.hawor_dir)
from hawor.utils.process import run_mano, run_mano_left  # noqa: E402

use_cuda = not args.cpu and torch.cuda.is_available()
d = np.load(args.inp)
pt = torch.from_numpy(d["pred_trans"]).float()
pr = torch.from_numpy(d["pred_rot"]).float()
pp = torch.from_numpy(d["pred_hand_pose"]).float()
pb = torch.from_numpy(d["pred_betas"]).float()
T = pt.shape[1]

out_l = run_mano_left(pt[0:1], pr[0:1], pp[0:1], betas=pb[0:1], use_cuda=use_cuda)
out_r = run_mano(pt[1:2], pr[1:2], pp[1:2], betas=pb[1:2], use_cuda=use_cuda)
jl = out_l["joints"][0].detach().cpu().numpy().astype(np.float32)  # (T,21,3)
jr = out_r["joints"][0].detach().cpu().numpy().astype(np.float32)
valid = np.asarray(d["pred_valid"]) if "pred_valid" in d else np.ones((2, T), np.float32)

os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
np.savez_compressed(args.out, joints_left=jl, joints_right=jr,
                    valid=valid.astype(np.float32), fps=np.float32(args.fps))
print(f"saved {args.out}: joints_left{jl.shape} joints_right{jr.shape} "
      f"fps={args.fps} valid_frames L/R={int(valid[0].sum())}/{int(valid[1].sum())}")
