#!/usr/bin/env python
"""HOI4D take -> replay.npz for sim/retarget_isaacsim.py. Run in the HaWoR conda env
(needs MANO/smplx + hawor.utils.process):

  "$HAWOR_PYTHON" scripts/hoi4d_to_replay.py \
      --hoi4d-root "$RR_DATA_ROOT/HOI4D" \
      --rel ZY20210800001/H1/C20/N27/S295/s03/T4 --out replay_hoi4d.npz

Exports BOTH hands by default (--hands both). Output keys:
    joints_left / joints_right : (T,21,3) float32  (whichever side(s) requested)
    valid_left / valid_right   : (T,)     float32  (0 where a frame was gap-filled)
    obj_pose                   : (T,7)    [x,y,z, qw,qx,qy,qz]
    fps                        : scalar
Use --hands left|right (or legacy --hand) for a single side.

Hand: Hand_pose/handpose_<hand>_hand/<rel>/<frame>.pickle = MANO {poseCoeff(48)=
global_orient(3)+hand_pose(45), beta(10), trans(3)}, camera frame. run_mano ->
joints (T,21,3) OpenPose order (== MediaPipe order replay/retarget expect).
Object: HOI4D_annotations/<rel>/objpose/<frame>.json dataList[*] {center, rotation
(euler xyz, rad), dimensions}. -> obj_pose (T,7)=[x,y,z, qw,qx,qy,qz], camera frame.
The two hands are aligned on the UNION of their frame indices (gaps forward-filled,
flagged in valid_*); the object track shares that same timeline. Everything stays
in CAMERA frame; retarget_isaacsim.py places it into the world (--scene-rot cv2zup
--recenter table). fps = 15 (HOI4D video).
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--hoi4d-root", required=True)
ap.add_argument("--rel", required=True, help="ZY.../Cxx/Nxx/.../T4")
ap.add_argument("--hands", choices=["both", "left", "right"], default="both",
                help="which hand(s) to export (default both -> joints_left+joints_right)")
ap.add_argument("--hand", choices=["right", "left"], default=None,
                help="(legacy) export a single side; overrides --hands")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--target-label", default=None, help="object label to pick from dataList (default: first)")
ap.add_argument("--hawor-dir", default=os.path.expanduser("~/Project/Reconstruct_and_Retarget/third_party/hawor"))
args = ap.parse_args()

sys.path.insert(0, args.hawor_dir)
from hawor.utils.process import run_mano, run_mano_left  # noqa: E402

sides = [args.hand] if args.hand else (["left", "right"] if args.hands == "both" else [args.hands])
use_cuda = torch.cuda.is_available()


def side_frames(side):
    hd = os.path.join(args.hoi4d_root, "Hand_pose", f"handpose_{side}_hand", args.rel)
    fs = sorted(int(os.path.basename(f)[:-7]) for f in glob.glob(os.path.join(hd, "*.pickle")))
    return hd, fs

# ---- master timeline: union of every requested hand's frames ---------------
hand_dirs, side_fs = {}, {}
for side in sides:
    hd, fs = side_frames(side)
    if not fs:
        sys.exit(f"no {side}-hand pickles under {hd}")
    hand_dirs[side], side_fs[side] = hd, fs
master = sorted(set().union(*side_fs.values()))
T = len(master)
print(f"[hoi4d] sides={sides}  master frames T={T}  idx {master[0]}..{master[-1]}  "
      + " ".join(f"{s}:{len(side_fs[s])}" for s in sides))


def hand_joints_for(side):
    """MANO params over the master timeline (forward/back-filled across gaps) -> joints."""
    avail = {}
    for idx in side_fs[side]:
        d = pickle.load(open(os.path.join(hand_dirs[side], f"{idx}.pickle"), "rb"))
        avail[idx] = (np.asarray(d["poseCoeff"], np.float32),
                      np.asarray(d["beta"], np.float32), np.asarray(d["trans"], np.float32))
    first = avail[side_fs[side][0]]
    pose, beta, trans, valid, last = [], [], [], [], None
    for idx in master:
        cur = avail.get(idx, last if last is not None else first)
        last = cur
        pose.append(cur[0]); beta.append(cur[1]); trans.append(cur[2])
        valid.append(1.0 if idx in avail else 0.0)
    pose = np.asarray(pose, np.float32)
    root_orient = torch.from_numpy(pose[None, :, :3])              # (1,T,3)
    hand_pose = torch.from_numpy(pose[None, :, 3:48])              # (1,T,45)
    betas = torch.from_numpy(np.asarray(beta, np.float32)[None])   # (1,T,10)
    transl = torch.from_numpy(np.asarray(trans, np.float32)[None]) # (1,T,3)
    runner = run_mano if side == "right" else run_mano_left
    out = runner(transl, root_orient, hand_pose, betas=betas, use_cuda=use_cuda)
    joints = out["joints"][0].detach().cpu().numpy().astype(np.float32)  # (T,21,3)
    valid = np.asarray(valid, np.float32)
    print(f"[hoi4d] {side}: joints {joints.shape}  valid {int(valid.sum())}/{T}  "
          f"wrist span(m)={(joints[:, 0].max(0) - joints[:, 0].min(0)).round(3).tolist()}")
    return joints, valid


hand_out = {}      # joints_<side>, valid_<side>
for side in sides:
    j, v = hand_joints_for(side)
    hand_out[f"joints_{side}"] = j
    hand_out[f"valid_{side}"] = v

# ---- object: objpose euler+center -> (T,7) over the master timeline ---------
obj_dir = os.path.join(args.hoi4d_root, "HOI4D_annotations", args.rel, "objpose")
obj_pose = np.zeros((T, 7), np.float32)
last = None
for i, idx in enumerate(master):
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
    c, r = item["center"], item["rotation"]
    quat_xyzw = Rotation.from_euler("XYZ", [r["x"], r["y"], r["z"]]).as_quat()
    obj_pose[i] = [c["x"], c["y"], c["z"], quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    last = obj_pose[i]
print(f"[hoi4d] obj label='{item.get('label')}'  obj center span(m)="
      f"{(obj_pose[:, :3].max(0) - obj_pose[:, :3].min(0)).round(3).tolist()}")

os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
# `frames` = video frame index for each row (lets a video overlay align npz rows -> mp4 frames)
np.savez_compressed(args.out, obj_pose=obj_pose, fps=np.float32(args.fps),
                    frames=np.asarray(master, np.int32), rel=args.rel, **hand_out)
print(f"[hoi4d] saved {args.out}: T={T} @ {args.fps}fps  frames {master[0]}..{master[-1]}  "
      f"keys={sorted(['obj_pose','fps','frames','rel',*hand_out])}")
