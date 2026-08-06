"""Filter out unnatural hand orientations (thumb side / '虎口' pointing down).

Criterion: world-z component of the hand root frame's **thumb-side axis**.
Natural human grasps have the thumb up (+1) or horizontal (~0, e.g. top-down
pinches); thumb-down (-1) means an upside-down wrist.

⚠ 左右手的拇指侧轴方向相反。SharpaWave 左手是右手关于 xz 平面的精确镜像
  (M = diag(1,-1,1)), 所以拇指侧从 +Y 变成 -Y:

      right:  thumb_z = +R[2,1]
      left :  thumb_z = -R[2,1]

  用 --side 指定 (默认 right, 保持旧行为); 或让 --side auto 从 npy 的
  ``hand_name`` 字段读 (sharpa_wave_left / sharpa_wave)。照搬右手公式去筛左手,
  判据整体翻号, 会把姿态正确的全否掉、把虎口朝下的全留下。
  实测 (2026-08-05, bottle_cap 左手 10 个候选): 左手约定下人工认可的 3 个落在
  [-0.17, 0.00], 被否的 6 个落在 [-0.98, -0.74], -0.25 这条线正好分开。

  # flat dir of npys (isaac_succ):
  python tools/filter_hand_orientation.py --dir output/pp0_sharpa_wave/isaac_succ
  # grasp_data tree:
  python tools/filter_hand_orientation.py --exp-dir output/pp0_sharpa_wave --data grasp_data
  # 左手:
  python tools/filter_hand_orientation.py --exp-dir output/bottle_cap_sharpa_wave_left --side auto

Rejected files (and their _video/_report siblings if present) move to
<dir>_thumb_down/. Default threshold: thumb_z >= -0.25.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
from scipy.spatial.transform import Rotation as R


def thumb_z(npy_path, side="right"):
    d = np.load(npy_path, allow_pickle=True).item()
    q = d["grasp_qpos"][0] if d["grasp_qpos"].ndim > 1 else d["grasp_qpos"]
    if side == "auto":
        side = "left" if str(d.get("hand_name", "")).endswith("_left") else "right"
    sgn = -1.0 if side == "left" else 1.0        # 左手拇指侧是 -Y, 见模块 docstring
    return sgn * float(R.from_quat(np.roll(q[3:7], -1)).as_matrix()[2, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="flat dir of <name>.npy (e.g. isaac_succ)")
    ap.add_argument("--exp-dir", help="experiment dir (use with --data)")
    ap.add_argument("--data", default="grasp_data")
    ap.add_argument("--side", default="right", choices=("left", "right", "auto"),
                    help="拇指侧轴; left 用 -Y, auto 从 npy 的 hand_name 读")
    ap.add_argument("--min-thumb-z", type=float, default=-0.25)
    args = ap.parse_args()

    if args.dir:
        root = args.dir.rstrip("/")
        files = sorted(glob.glob(f"{root}/*_grasp.npy"))
        reject_root = root + "_thumb_down"
    else:
        assert args.exp_dir
        root = f"{args.exp_dir.rstrip('/')}/{args.data}"
        files = [f for f in sorted(glob.glob(f"{root}/**/*.npy", recursive=True))
                 if os.path.isfile(f)]
        reject_root = root + "_thumb_down"

    kept, moved = [], []
    for f in files:
        z = thumb_z(f, args.side)
        ok = z >= args.min_thumb_z
        print(f"{os.path.basename(f):22s} thumb_z={z:+.2f}  {'OK' if ok else 'THUMB-DOWN -> moved'}")
        if ok:
            kept.append(os.path.basename(f))
            continue
        moved.append(os.path.basename(f))
        group = [f]
        if args.dir:  # move video/report siblings too
            stem = os.path.basename(f).replace(".npy", "")
            group += [p for p in (f"{root}/{stem}_video.mp4", f"{root}/{stem}_report.json")
                      if os.path.exists(p)]
        for src in group:
            dst = src.replace(root, reject_root, 1)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.realpath(src), dst)
            if os.path.islink(src):
                os.unlink(src)

    # keep isaac_succ summary.json consistent
    sj = f"{root}/summary.json" if args.dir else None
    if sj and os.path.exists(sj) and moved:
        s = json.load(open(sj))
        names = {m.replace(".npy", "") for m in moved}
        s["success"] = [n for n in s["success"] if n not in names]
        s["results"] = {k: v for k, v in s["results"].items() if k not in names}
        s["n_success"] = len(s["success"])
        json.dump(s, open(sj, "w"), indent=1)
    print(f"\nkept {len(kept)}, moved {len(moved)} -> {reject_root}")


if __name__ == "__main__":
    main()
