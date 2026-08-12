#!/usr/bin/env python3
"""为人工挑选"重建参考帧"生成候选 mask 帧。

输入 v17A video_mask_sequence manifest + 原视频 + (可选)anylink box track 用于
定位目标物体实例 + (可选)HOI-DETR detections 用于手遮挡评分。
输出: 目标物体的 top-K 候选帧全分辨率 overlay + contact sheet + report.json,
供人工挑选;挑中的帧号之后作为 import_v17a_masks --reconstruction-frame。

评分 = 面积(相对中位数) × (1-手遮挡)^2 × 完整度(frag),再按时间分桶保证多样性。
基线 pick_best_frame(零改动)的选择也一并渲染,标 baseline。

用法(sam3 env):
  python make_object_mask_candidates.py \
      --manifest .../video_mask_sequence/video_mask_sequence.json \
      --video 0.mp4 --out <候选目录> \
      [--track anylink filtered_track.json] [--detections hoi_detr detections.json] \
      [--object-id object_0001] [--topk 16]
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


def bbox_of(mb: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mb)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def hand_boxes_by_frame(detections_json: Path | None) -> dict[int, list]:
    if not detections_json:
        return {}
    data = json.loads(detections_json.read_text())
    out = {}
    for fr in data.get("frames") or []:
        boxes = [d["box_xyxy"] for d in fr.get("detections") or []
                 if d.get("class_name") == "hand"]
        if boxes:
            out[int(fr["frame_idx"])] = boxes
    return out


def hand_occlusion(mb: np.ndarray, boxes: list) -> float:
    """mask 像素落在任一手 box 内的比例(遮挡代理)。"""
    if not boxes:
        return 0.0
    area = int(mb.sum())
    if not area:
        return 0.0
    inside = np.zeros_like(mb)
    h, w = mb.shape
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 > x1 and y2 > y1:
            inside[y1:y2, x1:x2] = True
    return float((mb & inside).sum()) / area


def track_boxes_by_frame(track_json: Path | None) -> dict[int, list]:
    """anylink filtered_track.json -> {frame: [object_box,...]}"""
    if not track_json:
        return {}
    data = json.loads(track_json.read_text())
    out = {}
    for k, v in (data.get("per_frame") or {}).items():
        boxes = [ln["object_box"] for ln in v.get("links") or [] if ln.get("object_box")]
        if boxes:
            out[int(k)] = boxes
    return out


def match_object(manifest: dict, track: dict[int, list]) -> tuple[str, dict]:
    """用 track 的 object_box 对每个实例的 mask bbox 做平均 IoU,选最高者。"""
    scores = {}
    for object_id in manifest.get("object_ids") or []:
        accepted = load_accepted(manifest, object_id)
        common = sorted(set(accepted) & set(track))
        vals = []
        for i in common[:: max(1, len(common) // 60)]:  # 采样≤60帧够稳
            mb = cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0
            box = bbox_of(mb)
            if box:
                vals.append(max(iou(box, tb) for tb in track[i]))
        scores[object_id] = {
            "mean_iou_vs_track": round(float(np.mean(vals)), 4) if vals else 0.0,
            "n_accepted": len(accepted),
            "n_common_with_track": len(common),
        }
    best = max(scores, key=lambda k: scores[k]["mean_iou_vs_track"]) if scores else None
    return best, scores


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx}")
    return frame


def overlay(frame_bgr: np.ndarray, mb: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = frame_bgr.copy()
    out[mb] = (0.55 * out[mb] + 0.45 * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 3)
    return out


def banner(img: np.ndarray, text: str) -> np.ndarray:
    out = cv2.copyMakeBorder(img, 46, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--track", type=Path, default=None, help="anylink filtered_track.json 定位目标实例")
    ap.add_argument("--detections", type=Path, default=None, help="HOI-DETR detections.json 用于手遮挡")
    ap.add_argument("--object-id", default=None, help="直接指定实例,跳过 track 匹配")
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--sheet-cols", type=int, default=4)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    assert manifest.get("schema_version") in (
        "persistent_video_mask_sequence_v1", "persistent_mask_sequence_v1",
    ), manifest.get("schema_version")

    track = track_boxes_by_frame(args.track)
    if args.object_id:
        object_id, match_scores = args.object_id, {}
    else:
        assert track, "未指定 --object-id 时必须给 --track"
        object_id, match_scores = match_object(manifest, track)
        assert object_id, "track 匹配不到任何实例"

    accepted = load_accepted(manifest, object_id)
    assert accepted, f"{object_id} 没有 accepted mask"
    idxs = sorted(accepted)
    hands = hand_boxes_by_frame(args.detections)

    # 逐帧统计
    rows = {}
    for i in idxs:
        mb = cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0
        st = mask_stats(mb)
        st["occ"] = round(hand_occlusion(mb, hands.get(i, [])), 4)
        rows[i] = st

    # 基线选择(零改动 pick_best_frame)
    def gen():
        for i in idxs:
            yield cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE)

    baseline = idxs[pick_best_frame(gen())]

    # 打分: 面积(相对中位数,上限1) × (1-occ)^2 × frag;贴边直接不合格
    med = float(np.median([r["area"] for r in rows.values()]))
    for i, r in rows.items():
        a = min(r["area"] / med, 1.5) / 1.5
        r["score"] = round(a * (1.0 - r["occ"]) ** 2 * r["frag"], 4) if not r["border"] else 0.0

    # 时间分桶保证多样性: topk 桶,每桶取分最高;不足则全局补位
    buckets = np.array_split(np.array(idxs), args.topk)
    cands = []
    for b in buckets:
        if not len(b):
            continue
        best = max(b.tolist(), key=lambda i: rows[i]["score"])
        if rows[best]["score"] > 0:
            cands.append(best)
    pool = sorted((i for i in idxs if i not in cands), key=lambda i: -rows[i]["score"])
    cands += pool[: args.topk - len(cands)]
    cands = sorted(set(cands))

    args.out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    assert cap.isOpened(), args.video

    tiles = []
    for rank, i in enumerate(sorted(cands, key=lambda i: -rows[i]["score"]), 1):
        mb = cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0
        color = (0, 255, 255) if i == baseline else (0, 255, 0)
        img = overlay(read_frame(cap, i), mb, color)
        r = rows[i]
        tag = f"f{i:04d} score={r['score']:.3f} occ={r['occ']:.2f} frag={r['frag']:.2f} area={r['area']}"
        if i == baseline:
            tag += " [baseline pick]"
        img = banner(img, tag)
        cv2.imwrite(str(args.out / f"cand{rank:02d}_f{i:04d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        tile = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
        tiles.append(tile)

    if baseline not in cands:  # 基线未入选也渲染出来供对照
        mb = cv2.imread(str(accepted[baseline]), cv2.IMREAD_GRAYSCALE) > 0
        img = banner(overlay(read_frame(cap, baseline), mb, (0, 255, 255)),
                     f"f{baseline:04d} [baseline pick] occ={rows[baseline]['occ']:.2f}")
        cv2.imwrite(str(args.out / f"baseline_f{baseline:04d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        cols = args.sheet_cols
        rows_img = [np.hstack(tiles[r: r + cols]) for r in range(0, len(tiles), cols)]
        w = max(r.shape[1] for r in rows_img)
        rows_img = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows_img]
        cv2.imwrite(str(args.out / "contact_sheet.jpg"), np.vstack(rows_img),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    report = {
        "manifest": str(args.manifest),
        "video": str(args.video),
        "object_id": object_id,
        "instance_match_scores": match_scores,
        "baseline_pick_best_frame": baseline,
        "candidates_ranked": sorted(cands, key=lambda i: -rows[i]["score"]),
        "per_frame": rows,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"object={object_id} baseline=f{baseline} candidates={len(cands)} -> {args.out}")


if __name__ == "__main__":
    main()
