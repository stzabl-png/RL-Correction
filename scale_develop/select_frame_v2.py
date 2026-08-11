#!/usr/bin/env python3
"""选帧 v2:两段式筛选(纯 CPU,不重跑任何 GPU stage)。

第一段 几何粗筛(尺度不变,比 stage B 的 accept 更严):
  硬门: 贴边剔除 / frag>=0.98 / 时间窗相对面积 [0.7,1.3](替代全局绝对面积带)
  相对门(对物体形状不做绝对假设): 填充率 area/diag^2、凸度 area/hull 均相对本物体
  中位数打折剔除; 清晰度(bbox 裁剪 Laplacian 方差,归一分辨率)剔除最糊的 25%。
第二段 遮挡分(HOI-DETR 手框版 v0):
  occ = |手框并集 ∩ mask凸包| / |mask凸包|,凸包近似"包含被手指盖住缺口的完整轮廓"。
  按 occ 升序排,occ 相近(0.05 档内)按干净度(凸度x填充率x清晰度的相对分)择优。

输出: report_v2.json + top 候选 contact sheet(含基线选帧对比)。
用法(sam3 env): python select_frame_v2.py --run runs/s01_ketchup_grab_01 [--top 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label

HAND_SCORE_MIN = 0.5
SAM2_CKPT = "/home/bangdu/HumanVideo2RobotData/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
RING_PX = 25         # 边界接触带宽度(全分辨率像素)
CONTACT_W = 0.25     # contact 在合成遮挡分里的权重
WINDOW = 15          # 时间窗半径(帧)
REL_AREA = (0.7, 1.3)
FRAG_MIN = 0.98
SOLIDITY_REL = 0.90  # 凸度 >= 本物体中位数 * 该系数
FILL_REL = 0.75      # 填充率 >= 本物体中位数 * 该系数
BLUR_DROP = 0.25     # 剔除最糊的分位
STAGE1_CAP = 60
OCC_TIE = 0.05
NMS_GAP = 20         # top-K 候选之间的最小帧距(时间 NMS,保证多样性)


def load_accepted(manifest: dict, object_id: str) -> dict[int, Path]:
    out = {}
    for frame in manifest.get("frames") or []:
        obj = (frame.get("objects") or {}).get(object_id)
        if obj and obj.get("status") == "accepted" and obj.get("mask"):
            out[int(frame["frame_idx"])] = Path(obj["mask"])
    return out


def hand_boxes_by_frame(detections: dict) -> dict[int, list[list[float]]]:
    out = {}
    for frame in detections["frames"]:
        boxes = [d["box_xyxy"] for d in frame.get("detections") or []
                 if d.get("class_name") == "hand" and d.get("score", 0) >= HAND_SCORE_MIN]
        out[int(frame["frame_idx"])] = boxes
    return out


def geom_features(mb: np.ndarray) -> dict:
    area = int(mb.sum())
    lab, n = label(mb)
    largest = int(max(((lab == k).sum() for k in range(1, n + 1)), default=0))
    ys, xs = np.nonzero(mb)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    diag = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1))
    pts = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    hull = cv2.convexHull(np.vstack([p.reshape(-1, 2) for p in pts]))
    hull_area = float(cv2.contourArea(hull))
    return {
        "area": area,
        "frag": largest / area if area else 0.0,
        "border": bool(mb[0].any() or mb[-1].any() or mb[:, 0].any() or mb[:, -1].any()),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "fill": area / (diag * diag) if diag else 0.0,
        "solidity": area / hull_area if hull_area else 0.0,
        "hull": hull.reshape(-1, 2).tolist(),
    }


def blur_score(frame_bgr: np.ndarray, bbox: list[int]) -> float:
    x0, y0, x1, y1 = bbox
    crop = cv2.cvtColor(frame_bgr[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2GRAY)
    if crop.shape[1] > 512:  # 统一分辨率,否则 Laplacian 方差不可比
        crop = cv2.resize(crop, (512, max(1, int(512 * crop.shape[0] / crop.shape[1]))))
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def occlusion(hands: list[list[float]], hull: np.ndarray, shape: tuple[int, int], ds: int = 4) -> float:
    h, w = shape[0] // ds, shape[1] // ds
    hull_m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hull_m, [np.asarray(hull, np.int32) // ds], 1)
    hand_m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in hands:
        cv2.rectangle(hand_m, (int(x0) // ds, int(y0) // ds), (int(x1) // ds, int(y1) // ds), 1, -1)
    denom = int(hull_m.sum())
    return float((hull_m & hand_m).sum() / denom) if denom else 1.0


def hand_masks_pixel(run: Path, cap, idxs: list[int], hands: dict[int, list[list[float]]]) -> dict[int, np.ndarray]:
    """SAM2 图像预测器 + HOI-DETR 手框 prompt -> 逐帧手部 mask 并集(磁盘缓存)。"""
    cache = run / "hand_masks"
    cache.mkdir(exist_ok=True)
    need = [i for i in idxs if not (cache / f"f{i:06d}.png").exists()]
    if need:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device="cuda"))
        for i in need:
            frame = read_frame(cap, i)
            union = np.zeros(frame.shape[:2], np.uint8)
            boxes = hands.get(i, [])
            if boxes:
                predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for box in boxes:
                    m, _, _ = predictor.predict(box=np.asarray(box), multimask_output=False)
                    union |= (m[0] > 0).astype(np.uint8)
            cv2.imwrite(str(cache / f"f{i:06d}.png"), union * 255)
    return {i: cv2.imread(str(cache / f"f{i:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0 for i in idxs}


def occlusion_pixel(hand: np.ndarray, hull: list, mb: np.ndarray) -> tuple[float, float, float]:
    """(合成遮挡分, occ_hull, contact)。
    occ_hull: 手mask落在物体凸包内的占比——手在物体前方压住剪影才计入(凸物体下天然区分前后)。
    contact: 手mask与物体轮廓膨胀环的交并——手压在物体边界上的证据(抓握从背后包边也能感知)。"""
    hull_m = np.zeros(mb.shape, np.uint8)
    cv2.fillPoly(hull_m, [np.asarray(hull, np.int32)], 1)
    hull_area = int(hull_m.sum())
    occ_hull = float((hand & (hull_m > 0)).sum() / hull_area) if hull_area else 1.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_PX, RING_PX))
    ring = (cv2.dilate(mb.astype(np.uint8), kernel) > 0) & ~mb
    ring_area = int(ring.sum())
    contact = float((hand & ring).sum() / ring_area) if ring_area else 0.0
    return occ_hull + CONTACT_W * contact, occ_hull, contact


def rel_norm(vals: dict[int, float]) -> dict[int, float]:
    med = float(np.median(list(vals.values()))) or 1e-9
    return {k: v / med for k, v in vals.items()}


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    assert ok, idx
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--hand-mode", choices=["box", "pixel"], default="box",
                    help="box=HOI-DETR手框(v2.0) pixel=SAM2像素手mask+边界接触带(v2.1,需GPU)")
    args = ap.parse_args()

    run = args.run.resolve()
    manifest = json.loads((run / "instance/video_mask_sequence/video_mask_sequence.json").read_text())
    detections = json.loads((run / "hoi_detr_probe/detections.json").read_text())
    hands = hand_boxes_by_frame(detections)
    video = next(run.glob("*.mp4"))
    baseline = json.loads((run / "frame_selection_baseline/report.json").read_text())["objects"]
    out_dir = run / ("frame_selection_v21" if args.hand_mode == "pixel" else "frame_selection_v2")
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    summary = {"video": str(video), "objects": {}}
    for object_id in manifest.get("object_ids") or []:
        accepted = load_accepted(manifest, object_id)
        idxs = sorted(accepted)
        feats = {}
        for i in idxs:
            feats[i] = geom_features(cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0)

        # ── 第一段:硬门 + 相对门 ──
        areas = {i: feats[i]["area"] for i in idxs}
        rel_area = {i: areas[i] / (np.median([areas[j] for j in idxs if abs(j - i) <= WINDOW]) or 1e-9)
                    for i in idxs}
        sol_rel = rel_norm({i: feats[i]["solidity"] for i in idxs})
        fill_rel = rel_norm({i: feats[i]["fill"] for i in idxs})
        drop = {}
        for i in idxs:
            why = []
            if feats[i]["border"]: why.append("border")
            if feats[i]["frag"] < FRAG_MIN: why.append("frag")
            if not (REL_AREA[0] <= rel_area[i] <= REL_AREA[1]): why.append("rel_area")
            if sol_rel[i] < SOLIDITY_REL: why.append("solidity")
            if fill_rel[i] < FILL_REL: why.append("fill")
            if why: drop[i] = why
        stage1 = [i for i in idxs if i not in drop]

        blur = {i: blur_score(read_frame(cap, i), feats[i]["bbox"]) for i in stage1}
        if len(stage1) > 8:  # 太少就别再砍清晰度了
            cut = float(np.quantile(list(blur.values()), BLUR_DROP))
            for i in [i for i in stage1 if blur[i] < cut]:
                drop[i] = ["blur"]
            stage1 = [i for i in stage1 if blur[i] >= cut]
        clean = {i: sol_rel[i] * fill_rel[i] * (blur[i] / (np.median(list(blur.values())) or 1e-9))
                 for i in stage1}
        stage1 = sorted(stage1, key=lambda i: -clean[i])[:STAGE1_CAP]

        # ── 第二段:遮挡分 ──
        detail = {}
        if args.hand_mode == "pixel":
            hmasks = hand_masks_pixel(run, cap, stage1, hands)
            occ = {}
            for i in stage1:
                mb = cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0
                occ[i], oh, ct = occlusion_pixel(hmasks[i], feats[i]["hull"], mb)
                detail[i] = {"occ_hull": round(oh, 4), "contact": round(ct, 4)}
        else:
            occ = {i: occlusion(hands.get(i, []), feats[i]["hull"], shape) for i in stage1}
        ranked = sorted(stage1, key=lambda i: (round(occ[i] / OCC_TIE), -clean[i]))
        # 时间 NMS:top-K 内两两至少隔 NMS_GAP 帧,避免连号扎堆(排名第一永远保留)
        top = []
        for i in ranked:
            if all(abs(i - j) >= NMS_GAP for j in top):
                top.append(i)
            if len(top) >= args.top:
                break

        base_chosen = baseline.get(object_id, {}).get("chosen_frame")
        summary["objects"][object_id] = {
            "n_accepted": len(idxs), "n_stage1": len(stage1),
            "dropped": {str(k): v for k, v in sorted(drop.items())},
            "hand_mode": args.hand_mode,
            "occ": {str(i): round(occ[i], 4) for i in ranked},
            "occ_detail": {str(i): detail[i] for i in ranked if i in detail},
            "clean": {str(i): round(clean[i], 3) for i in ranked},
            "ranked": ranked, "candidates_nms": top,
            "chosen_frame": top[0] if top else None,
            "baseline_chosen": base_chosen,
        }
        print(f"[{object_id}] accepted={len(idxs)} stage1={len(stage1)} "
              f"-> v2 chosen {top[0] if top else None} (baseline {base_chosen})")

        # ── 对比 contact sheet: v2 top-N + 基线选帧 ──
        tiles = []
        show = [(f"v2 #{r+1}", i) for r, i in enumerate(top)]
        if base_chosen is not None and base_chosen not in top:
            show.append(("BASELINE", base_chosen))
        for tag, i in show:
            mb = cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0
            color = (0, 255, 0) if tag == "v2 #1" else (0, 165, 255) if tag == "BASELINE" else (0, 0, 255)
            img = read_frame(cap, i).copy()
            img[mb] = (0.55 * img[mb] + 0.45 * np.array(color)).astype(np.uint8)
            if args.hand_mode == "pixel":
                hb = hmasks.get(i)
                if hb is not None:
                    img[hb] = (0.5 * img[hb] + 0.5 * np.array([255, 80, 0])).astype(np.uint8)
            for x0, y0, x1, y1 in hands.get(i, []):
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 6)
            img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
            o = occ.get(i)
            txt = f"{tag} f{i} occ={o:.2f}" if o is not None else f"{tag} f{i} occ=n/a"
            cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            tiles.append(img)
        if tiles:
            h = max(t.shape[0] for t in tiles)
            tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
            rows = [np.hstack(tiles[r:r + 3]) for r in range(0, len(tiles), 3)]
            w = max(r.shape[1] for r in rows)
            rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
            cv2.imwrite(str(out_dir / f"{object_id}_v2_vs_baseline.jpg"), np.vstack(rows))

    cap.release()
    (out_dir / "report_v2.json").write_text(json.dumps(summary, indent=2))
    print("report ->", out_dir / "report_v2.json")


if __name__ == "__main__":
    main()
