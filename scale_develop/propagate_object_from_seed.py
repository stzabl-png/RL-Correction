#!/usr/bin/env python3
"""从一个种子 mask(取内部点)对整段视频做 SAM2 双向传播,产出:
  1. recon 管线 sam2_object 布局: <out>/video_segmentation/masks/frame_%06d_masks/object_0.png
     + label_prompt.json(种子帧+点;之后可被人工选帧覆盖 frame_idx)
  2. v17A persistent_mask_sequence_v1 兼容 manifest: <out>/seed_propagation_manifest.json
     (非空帧=accepted,供 make_object_mask_candidates.py 直接消费)

与 recon 管线 segment_object_on_video 的差别只有一个: init_state 加
offload_video_to_cpu=True,长视频(数千帧)不炸显存(帧张量放内存,~13MB/帧)。

用法(sam3 env, CUDA_VISIBLE_DEVICES 选卡):
  python propagate_object_from_seed.py --video v.mp4 --seed-mask m.png --seed-frame 34 \
      --out <sam2_object目录> --sam2-root <sam2仓库> --object-name dustpan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def interior_point(mb: np.ndarray) -> tuple[float, float]:
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(mb)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x), float(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--seed-mask", required=True, type=Path)
    ap.add_argument("--seed-frame", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sam2-root", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--object-id", default="object_0")
    ap.add_argument("--object-name", default="object")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(args.sam2_root))
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    ckpt = args.checkpoint or (args.sam2_root / "checkpoints" / "sam2.1_hiera_large.pt")

    mb = cv2.imread(str(args.seed_mask), cv2.IMREAD_GRAYSCALE) > 0
    assert mb.any(), f"seed mask 为空: {args.seed_mask}"
    px, py = interior_point(mb)
    print(f"[seed] frame={args.seed_frame} point=({px:.0f},{py:.0f})")

    predictor = build_sam2_video_predictor(args.model_cfg, str(ckpt), device="cuda")
    state = predictor.init_state(video_path=str(args.video), offload_video_to_cpu=True)
    num_frames = int(state["num_frames"])
    h, w = int(state["video_height"]), int(state["video_width"])

    outputs: dict[int, np.ndarray] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.add_new_points_or_box(
            state, frame_idx=args.seed_frame, obj_id=1,
            points=np.array([[px, py]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
            clear_old_points=True, normalize_coords=True,
        )
        for reverse in (False, True):
            for fidx, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=args.seed_frame, reverse=reverse,
            ):
                if 1 in obj_ids:
                    m = (logits[list(obj_ids).index(1)][0] > 0).cpu().numpy()
                    outputs[int(fidx)] = m

    masks_base = args.out / "video_segmentation" / "masks"
    masks_base.mkdir(parents=True, exist_ok=True)
    manifest_frames, detected = [], 0
    for fidx in range(num_frames):
        m = outputs.get(fidx)
        arr = (m.astype(np.uint8) * 255) if m is not None and m.any() else np.zeros((h, w), np.uint8)
        fdir = masks_base / f"frame_{fidx:06d}_masks"
        fdir.mkdir(exist_ok=True)
        fpath = fdir / f"{args.object_id}.png"
        cv2.imwrite(str(fpath), arr)
        if arr.any():
            detected += 1
            manifest_frames.append({
                "frame_idx": fidx,
                "objects": {args.object_id: {"status": "accepted", "mask": str(fpath)}},
            })
        else:
            manifest_frames.append({"frame_idx": fidx, "objects": {}})

    (args.out / "label_prompt.json").write_text(json.dumps({
        "schema_version": "sam2_object_prompt_v2",
        "objects": [{
            "object_id": args.object_id, "frame_idx": args.seed_frame,
            "points": [[px, py]], "labels": [1],
            "locked": False, "name": args.object_name,
        }],
    }, indent=2))

    manifest = {
        "schema_version": "persistent_mask_sequence_v1",
        "status": "ready",
        "video": str(args.video),
        "object_ids": [args.object_id],
        "frames": manifest_frames,
    }
    mpath = args.out / "seed_propagation_manifest.json"
    mpath.write_text(json.dumps(manifest))
    print(f"[done] frames={num_frames} detected={detected} -> {mpath}")


if __name__ == "__main__":
    main()
