#!/usr/bin/env python3
"""ARCTIC egocentric view -> an mp4 our reconstruction pipeline can take.

Three things have to be right or the ground-truth comparison downstream is meaningless.

1. The stored "cropped_images" are only cropped for the 8 static cameras (crop_images.py
   cuts a hand-tracking window and resizes it to 1000x1000 -- that is no longer a
   consistent camera view).  View 0 takes the other branch, a plain
   `im.resize(0.3 * size)`: nothing cut, nothing stretched, 2800x2000 -> 840x600.  So the
   ego view IS a valid pinhole view and its intrinsics are just K * 0.3.

2. Frame numbering.  Filenames start at 00001.jpg but annotations are 0-based, offset by
   misc.json's per-subject ioi_offset (`vidx = int(name) - ioi_offset`, as in the repo's
   own crop_images.py and arctic_dataset.py).  We write the mapping out so the comparison
   never has to guess.

3. Distortion.  dist8 is NOT removed from the stored images.  Measured on this camera the
   corners move 26-40 px on the 840x600 image (3.5 px near the centre), which is enough to
   bias ViPE's camera track.  We undistort with the scaled K, and record the new K -- after
   undistortion the video is a clean pinhole camera, which is what the pipeline assumes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

ARCTIC = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")
EGO_SCALE = 0.3          # crop_images.py: EGO_IMAGE_SCALE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True)              # e.g. s05
    ap.add_argument("--seq", required=True)                  # e.g. laptop_grab_01
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=30.0, help="输出 fps(已按 stride 折算后的值)")
    ap.add_argument("--stride", type=int, default=1,
                    help="每 stride 帧取一帧。ARCTIC 是 30fps, stride=2 -> 15fps, "
                         "与我们平时的 EgoDex 输入帧率一致, 且重建耗时减半")
    ap.add_argument("--dark-thresh", type=float, default=40.0,
                    help="开头连续亮度低于此值的帧丢弃(自动曝光未稳定)")
    ap.add_argument("--no-undistort", action="store_true",
                    help="留下畸变(仅用于对照实验; 默认去畸变)")
    a = ap.parse_args()

    src = ARCTIC / "cropped_images" / a.subject / a.seq / "0"
    frames = sorted(f for f in glob.glob(str(src / "*.jpg")) if "/._" not in f)
    if not frames:
        raise SystemExit(f"no ego frames under {src}")

    misc = json.loads((ARCTIC / "meta" / "misc.json").read_text())
    ioi = int(misc[a.subject]["ioi_offset"])
    ego = np.load(ARCTIC / "raw_seqs" / a.subject / f"{a.seq}.egocam.dist.npy",
                  allow_pickle=True).item()
    K_full = np.asarray(ego["intrinsics"], float)
    dist = np.asarray(ego["dist8"], float).ravel()
    n_ann = int(np.asarray(ego["R_k_cam_np"]).shape[0])

    K = K_full.copy()                    # intrinsics of the STORED (0.3x) image
    K[:2] *= EGO_SCALE

    h, w = cv2.imread(frames[0]).shape[:2]
    if not a.no_undistort:
        # alpha=0 keeps only valid pixels, so the result has no black wedges for SAM2/vipe
        K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (w, h), cv2.CV_16SC2)
        x, y, rw, rh = roi
    else:
        K_new, map1, map2 = K, None, None
        x, y, rw, rh = 0, 0, w, h
    rw -= rw % 2
    rh -= rh % 2                          # libx264 needs even dimensions

    # Leading frames are the camera's auto-exposure warm-up: s05/laptop_grab_01 starts with
    # 2 near-black frames (mean 18.8, 18.7) before jumping to 97.  They are worthless and sit
    # exactly where vipe initialises and where auto-label picks its earliest accepted frame,
    # so drop them.  Only the LEADING run is dropped -- a dark frame mid-sequence is real
    # content (a hand covering the camera) and must stay, or the timeline would silently
    # desynchronise from the annotations.
    n_skip = 0
    for f in frames:
        if cv2.imread(f).mean() >= a.dark_thresh:
            break
        n_skip += 1
    if n_skip:
        print(f"  丢弃开头 {n_skip} 个暗帧(亮度<{a.dark_thresh}, 自动曝光未稳定)")
    frames = frames[n_skip:]
    if a.stride > 1:
        frames = frames[::a.stride]      # 抽帧在丢暗帧之后, 保证第一帧就是有效帧

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = a.out_dir / f"{a.subject}__{a.seq}.mp4"
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (rw, rh))
    index = []
    kept = 0
    for f in frames:
        vidx = int(Path(f).stem) - ioi
        if not 0 <= vidx < n_ann:         # crop_images.py drops these too (vidx < 0)
            continue
        im = cv2.imread(f)
        if map1 is not None:
            im = cv2.remap(im, map1, map2, cv2.INTER_LINEAR)
        im = im[y:y + rh, x:x + rw]
        vw.write(im)
        index.append({"video_frame": kept, "arctic_vidx": vidx, "file": Path(f).name})
        kept += 1
    vw.release()

    # principal point moves with the ROI crop; record the K that matches the mp4
    K_out = K_new.copy()
    K_out[0, 2] -= x
    K_out[1, 2] -= y
    meta = {
        "subject": a.subject, "seq": a.seq, "video": out_mp4.name,
        "num_video_frames": kept, "num_annotation_frames": n_ann,
        "leading_dark_frames_dropped": n_skip,
        "fps": a.fps, "size_wh": [rw, rh], "ioi_offset": ioi,
        "ego_image_scale": EGO_SCALE, "undistorted": not a.no_undistort,
        "K_stored_image": K.tolist(), "dist8": dist.tolist(),
        "K_video": K_out.tolist(),
        "note": ("video_frame -> arctic_vidx via index[]; GT object pose in ego camera = "
                 "world2ego[vidx] @ obj_pose_world[vidx], world2ego from egocam R_k_cam_np/"
                 "T_k_cam_np, object translation in object.npy is in MILLIMETRES"),
        "index": index,
    }
    (a.out_dir / f"{a.subject}__{a.seq}.meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{out_mp4}  {kept}/{len(frames)} 帧 -> {rw}x{rh} @ {a.fps}fps "
          f"({'去畸变' if not a.no_undistort else '未去畸变'})")
    print(f"  标注帧数 {n_ann}, ioi_offset {ioi}, 视频首帧 -> arctic_vidx {index[0]['arctic_vidx']}")
    print(f"  K_video fx={K_out[0,0]:.1f} fy={K_out[1,1]:.1f} cx={K_out[0,2]:.1f} cy={K_out[1,2]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
