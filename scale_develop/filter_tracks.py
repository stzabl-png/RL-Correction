#!/usr/bin/env python3
"""Track 级背景过滤(A+B 层):在 Qwen 终审之前砍掉"非交互目标"的 track。

A. 手物连接分:HOI-DETR detections.json 的 links.hf 给出每帧手-物连接的物体框;
   track 在该帧记"与手相连"当且仅当 mask 外接框与某连接框 IoU≥0.3(双向匹配——
   单向覆盖会把连接框内的桌面碎片也算进去,已踩坑)。
   hand_link_fraction = 相连帧数 / track 的 accepted 帧数。
B. 绝对运动:mask 质心相对首帧质心的最大位移 / 画面对角线。被拿起的物体位移大,
   台面/木盒≈0。
判定(保守,两条都弱才杀): hand_link_fraction < 0.10 且 max_disp < 0.03
   -> static_background;另 accepted 帧数 < 10 -> insufficient_track(选帧无意义)。
   跟着主体动的部件级虚假 track(如微波炉凹槽)不在此层处理,交给 Qwen 终审的
   is_discrete_object(纵深防御)。

只读已有产物,mask 以 1/4 分辨率读(质心/覆盖比不受影响)。
用法: python filter_tracks.py --run runs/s01_ketchup_grab_01
输出: <run>/track_filter.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

LINK_IOU_THRESH = 0.3
MIN_LINK_FRACTION = 0.10
MIN_DISP_DIAGONALS = 0.03
MIN_TRACK_FRAMES = 10


def linked_boxes_per_frame(detections: dict) -> dict[int, list[tuple[float, float, float, float]]]:
    out: dict[int, list] = {}
    for frame in detections["frames"]:
        dets = {str(d.get("detection_id")): d for d in frame.get("detections", [])}
        boxes = []
        for link in (frame.get("links") or {}).get("hf", []):
            src = dets.get(str(link["source_detection_id"]))
            tgt = dets.get(str(link["target_detection_id"]))
            if src is None or tgt is None:
                continue
            if src.get("class_name") == "hand" and tgt.get("class_name") != "hand":
                obj = tgt
            elif tgt.get("class_name") == "hand" and src.get("class_name") != "hand":
                obj = src
            else:
                continue
            boxes.append(tuple(float(v) for v in obj["box_xyxy"]))
        if boxes:
            out[int(frame["frame_idx"])] = boxes
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--detections", type=Path, default=None)
    args = ap.parse_args()
    run = args.run.resolve()

    manifest = json.loads((run / "instance/video_mask_sequence/video_mask_sequence.json").read_text())
    det_path = args.detections or run / "hoi_detr_probe/detections.json"
    linked = linked_boxes_per_frame(json.loads(det_path.read_text()))

    # 1/4 分辨率读 mask;坐标系数 0.25
    S = 0.25
    stats: dict[str, dict] = {}
    diag = None
    for frame in manifest["frames"]:
        fidx = int(frame["frame_idx"])
        for oid, obj in (frame.get("objects") or {}).items():
            if not obj.get("mask"):
                continue
            m = cv2.imread(str(obj["mask"]), cv2.IMREAD_REDUCED_GRAYSCALE_4)
            if m is None:
                continue
            if diag is None:
                diag = float(np.hypot(*m.shape))
            mb = m > 0
            area = int(mb.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(mb)
            st = stats.setdefault(oid, {"frames": 0, "linked": 0, "centroids": []})
            st["frames"] += 1
            st["centroids"].append((float(xs.mean()), float(ys.mean())))
            mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()
            m_area = (mx2 - mx1) * (my2 - my1)
            best = 0.0
            for (x1, y1, x2, y2) in linked.get(fidx, []):
                bx1, by1, bx2, by2 = x1 * S, y1 * S, x2 * S, y2 * S
                iw = max(0.0, min(mx2, bx2) - max(mx1, bx1))
                ih = max(0.0, min(my2, by2) - max(my1, by1))
                inter = iw * ih
                union = m_area + (bx2 - bx1) * (by2 - by1) - inter
                if union > 0:
                    best = max(best, inter / union)
            if best >= LINK_IOU_THRESH:
                st["linked"] += 1

    result = {}
    for oid, st in stats.items():
        c = np.asarray(st["centroids"])
        disp = float(np.hypot(*(c - c[0]).T).max() / diag) if len(c) > 1 else 0.0
        frac = st["linked"] / st["frames"]
        background = frac < MIN_LINK_FRACTION and disp < MIN_DISP_DIAGONALS
        insufficient = st["frames"] < MIN_TRACK_FRAMES
        verdict = ("insufficient_track" if insufficient
                   else "static_background" if background else "keep")
        result[oid] = {
            "n_frames": st["frames"],
            "hand_link_fraction": round(frac, 4),
            "max_centroid_disp_diagonals": round(disp, 4),
            "static_background": bool(background),
            "insufficient_track": bool(insufficient),
            "verdict": verdict,
        }
        print(f"[{oid}] frames={st['frames']} hand_link={frac:.2%} "
              f"disp={disp:.4f} -> {verdict.upper() if verdict != 'keep' else 'keep'}")

    out = run / "track_filter.json"
    out.write_text(json.dumps({
        "thresholds": {
            "link_iou": LINK_IOU_THRESH,
            "min_link_fraction": MIN_LINK_FRACTION,
            "min_disp_diagonals": MIN_DISP_DIAGONALS,
            "min_track_frames": MIN_TRACK_FRAMES,
        },
        "tracks": result,
    }, ensure_ascii=False, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
