#!/usr/bin/env python3
"""基线自动选帧评测:对 v17A 的 video_mask_sequence.json 逐物体跑
ego_pipeline.utils.object_io.pick_best_frame(原代码,零改动),
输出选帧结果 + 逐帧统计 + 可视化 contact sheet,供人工判断选得好不好。

用法(sam3 env):
  python eval_baseline_frame_selection.py \
      --manifest <.../video_mask_sequence/video_mask_sequence.json> \
      --video <video.mp4> --out <输出目录>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ego_pipeline"))
from utils.object_io import pick_best_frame  # noqa: E402  原代码,零改动


def load_accepted(manifest: dict, object_id: str) -> dict[int, Path]:
    out = {}
    for frame in manifest.get("frames") or []:
        obj = (frame.get("objects") or {}).get(object_id)
        if obj and obj.get("status") == "accepted" and obj.get("mask"):
            out[int(frame["frame_idx"])] = Path(obj["mask"])
    return out


def mask_stats(mb: np.ndarray) -> dict:
    from scipy.ndimage import label

    area = int(mb.sum())
    lab, n = label(mb)
    largest = int(max(((lab == k).sum() for k in range(1, n + 1)), default=0))
    frag = largest / area if area else 0.0
    border = bool(mb[0].any() or mb[-1].any() or mb[:, 0].any() or mb[:, -1].any())
    return {"area": area, "n_components": int(n), "frag": round(frag, 4), "border": border}


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx}")
    return frame


def overlay(frame_bgr: np.ndarray, mb: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    out = frame_bgr.copy()
    out[mb] = (0.55 * out[mb] + 0.45 * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sheet-cols", type=int, default=4)
    ap.add_argument("--sheet-n", type=int, default=12, help="contact sheet 采样帧数")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    # 真实输出是 persistent_video_mask_sequence_v1;adapter 文档里写的旧名也兼容
    assert manifest.get("schema_version") in (
        "persistent_video_mask_sequence_v1", "persistent_mask_sequence_v1",
    ), manifest.get("schema_version")
    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    assert cap.isOpened(), args.video

    summary = {"manifest": str(args.manifest), "video": str(args.video), "objects": {}}
    for object_id in manifest.get("object_ids") or []:
        accepted = load_accepted(manifest, object_id)
        if not accepted:
            summary["objects"][object_id] = {"error": "no accepted masks"}
            continue
        idxs = sorted(accepted)

        # 与原调用方 object_mesh_stage.py 一致:按帧序喂 mask 栈(生成器省内存)
        def gen():
            for i in idxs:
                yield cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE)

        chosen_pos = pick_best_frame(gen())  # 返回的是喂入序列中的位置
        chosen = idxs[chosen_pos]

        rows = {}
        for i in idxs:
            mb = cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0
            rows[i] = mask_stats(mb)
        summary["objects"][object_id] = {
            "chosen_frame": chosen,
            "n_accepted": len(idxs),
            "accepted_range": [idxs[0], idxs[-1]],
            "per_frame": rows,
        }

        # 选中帧全分辨率 overlay
        mb = cv2.imread(str(accepted[chosen]), cv2.IMREAD_GRAYSCALE) > 0
        cv2.imwrite(str(args.out / f"{object_id}_chosen_f{chosen:06d}.jpg"),
                    overlay(read_frame(cap, chosen), mb))

        # contact sheet: 均匀采样 + 必含选中帧
        sample = sorted(set(np.linspace(0, len(idxs) - 1, args.sheet_n, dtype=int).tolist()) | {chosen_pos})
        tiles = []
        for pos in sample:
            i = idxs[pos]
            mb = cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0
            tile = overlay(read_frame(cap, i), mb, (0, 255, 0) if i == chosen else (0, 0, 255))
            tile = cv2.resize(tile, (480, int(480 * tile.shape[0] / tile.shape[1])))
            r = rows[i]
            txt = f"f{i} a={r['area']} frag={r['frag']} b={int(r['border'])}"
            if i == chosen:
                txt = "CHOSEN " + txt
            cv2.putText(tile, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if i == chosen else (0, 200, 255), 2)
            tiles.append(tile)
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        ncol = args.sheet_cols
        rows_img = [np.hstack(tiles[r:r + ncol]) for r in range(0, len(tiles), ncol)]
        w = max(r.shape[1] for r in rows_img)
        rows_img = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows_img]
        cv2.imwrite(str(args.out / f"{object_id}_contact_sheet.jpg"), np.vstack(rows_img))
        print(f"[{object_id}] accepted={len(idxs)} ({idxs[0]}..{idxs[-1]}) -> chosen frame {chosen}")

    cap.release()
    (args.out / "report.json").write_text(json.dumps(summary, indent=2))
    print("report ->", args.out / "report.json")


if __name__ == "__main__":
    main()
