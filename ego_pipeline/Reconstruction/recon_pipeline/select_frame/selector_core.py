"""选帧器核心(从 agent/v17a-scale-develop 的 scale_develop/ 移植, 2026-08-14)。

三层结构(与验证时一致, 阈值原封不动):
  0. track 过滤(A+B): 手物连接帧占比(mask外接框 x HOI-DETR links.hf 连接框 IoU>=0.3,
     双向匹配) + 质心最大位移 + accepted 帧数下限。两弱才判 static_background(保守);
     部件级虚假 track(如微波炉凹槽)留给 Qwen verify(纵深防御)。
  1. 几何粗筛: 贴边/碎裂/时间窗相对面积/凸度/填充率(全部相对量) + Laplacian 清晰度。
  2. 像素级遮挡精排: SAM2 图像模式以 HOI-DETR 手框为 prompt 出手 mask,
     occ = |手∩物体凸包|/|凸包| + 0.25 x |手∩边界膨胀环|/|环|; SAM2 不可用时退回手框版。
  3. Qwen 终审(可选): 分项验证 top-1, 不合格从 top-K 比选; 多 track 通过时主体仲裁
     (hand_link 差距悬殊免调用直判)。VLM 不可达 -> 几何 top-1(geometric_fallback)。

验证记录: ketchup f591 / 扫把 f295 / microwave f618(凹槽 track 被 verify 判 spurious)。
本模块路径无关: 输入都是显式参数, 不假设任何目录布局。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

# ── 阈值(验证值, 勿随意改动) ──────────────────────────────────────────────
HAND_SCORE_MIN = 0.5
RING_PX = 25
CONTACT_W = 0.25
WINDOW = 15
REL_AREA = (0.7, 1.3)
FRAG_MIN = 0.98
SOLIDITY_REL = 0.90
FILL_REL = 0.75
BLUR_DROP = 0.25
STAGE1_CAP = 60
OCC_TIE = 0.05
NMS_GAP = 20

LINK_IOU_THRESH = 0.3
MIN_LINK_FRACTION = 0.10
MIN_DISP_DIAGONALS = 0.03
MIN_TRACK_FRAMES = 10

HARD_CRITERIA = ("occlusion", "completeness", "mask_quality")
CROP_EXPAND = 1.5
CROP_LONG_SIDE = 1024

QWEN_SYSTEM = (
    "你是 3D 重建管线的选帧审核员。用户会给你视频帧(原图裁剪 + 物体mask叠加图),"
    "评估该帧是否适合作为单图 3D 重建的参考帧。严格按要求输出 JSON,不要输出多余文字。"
)
VERIFY_PROMPT = """图1是视频帧的裁剪,图2是同一裁剪上的物体 mask 叠加(红色半透明区域+轮廓线)。
请先描述 mask 圈住的东西,然后逐项评估。只输出如下 JSON:
{
  "object_description": "mask圈住的东西是什么(一句话)",
  "is_discrete_object": true/false,   // 独立物体或物体的一个部件(如翻盖/盖子/把手)都算 true;仅当是桌面/背景/贴纸,或把多个不相干物体混在一个mask里时为 false
  "occlusion":    {"pass": true/false, "reason": "物体可见表面是否几乎无手或他物遮挡"},
  "completeness": {"pass": true/false, "reason": "物体是否完整在画面内,未被画幅裁切"},
  "mask_quality": {"pass": true/false, "reason": "mask是否贴合物体:没漏掉物体部分,也没把手/桌面/别的物体包进来"},
  "view_informative": {"pass": true/false, "reason": "该视角是否体现物体三维形状(非完全正对的退化平面视角)"},
  "sharpness":    {"pass": true/false, "reason": "是否清晰无明显运动模糊"}
}"""
FALLBACK_PROMPT = """以下 {n} 张图是同一物体在不同帧的 mask 叠加裁剪,标号 {labels}(与图片顺序一一对应)。
请选出最适合做单图 3D 重建参考帧的一张:优先无遮挡、物体完整、mask 贴合、视角有三维信息、清晰。
只输出如下 JSON:
{{
  "choice": "标号字母",
  "reason": "选它的理由(一句话)",
  "rejected": {{"标号": "不选的主因(短语)", ...}}
}}"""
TARGET_PROMPT = """以下 {n} 张图分别是同一段"人手操作物体"视频中 {n} 个不同物体 track 的
mask 叠加(红色区域,标号 {labels},与图片顺序一一对应,均取各自与手接触最多的帧)。
请判断哪个标号是**手主要抓握/操作的目标物体**(而非桌面、支撑台、背景或大件家具)。
只输出如下 JSON:
{{"target": "标号字母", "reason": "判断依据(一句话)"}}"""


# ══════════════════════════ 输入解析 ══════════════════════════

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


def linked_boxes_per_frame(detections: dict) -> dict[int, list[tuple[float, ...]]]:
    """每帧 links.hf 里手所连接的物体框。"""
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


def read_frame(cap, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    assert ok, idx
    return frame


# ══════════════════════════ 第 0 层: track 过滤 ══════════════════════════

def filter_tracks(manifest: dict, detections: dict) -> dict[str, dict]:
    """A+B 层: 每个 v17A track 的 hand_link_fraction / 位移 / verdict。1/4 分辨率读 mask。"""
    linked = linked_boxes_per_frame(detections)
    S = 0.25
    stats: dict[str, dict] = {}
    diag = None
    for frame in manifest.get("frames") or []:
        fidx = int(frame["frame_idx"])
        for oid, obj in (frame.get("objects") or {}).items():
            if obj.get("status") != "accepted" or not obj.get("mask"):
                continue
            m = cv2.imread(str(obj["mask"]), cv2.IMREAD_REDUCED_GRAYSCALE_4)
            if m is None:
                continue
            if diag is None:
                diag = float(np.hypot(*m.shape))
            mb = m > 0
            if not mb.any():
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
        result[oid] = {
            "n_frames": st["frames"],
            "hand_link_fraction": round(frac, 4),
            "max_centroid_disp_diagonals": round(disp, 4),
            "verdict": ("insufficient_track" if insufficient
                        else "static_background" if background else "keep"),
        }
    return result


# ══════════════════════════ 第 1+2 层: 两段式选帧 ══════════════════════════

def _geom_features(mb: np.ndarray) -> dict:
    from scipy.ndimage import label as cc_label
    area = int(mb.sum())
    lab, n = cc_label(mb)
    largest = int(max(((lab == k).sum() for k in range(1, n + 1)), default=0))
    ys, xs = np.nonzero(mb)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    diag = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1))
    pts = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    hull = cv2.convexHull(np.vstack([p.reshape(-1, 2) for p in pts]))
    hull_area = float(cv2.contourArea(hull))
    return {"area": area, "frag": largest / area if area else 0.0,
            "border": bool(mb[0].any() or mb[-1].any() or mb[:, 0].any() or mb[:, -1].any()),
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "fill": area / (diag * diag) if diag else 0.0,
            "solidity": area / hull_area if hull_area else 0.0,
            "hull": hull.reshape(-1, 2).tolist()}


def _blur_score(frame_bgr: np.ndarray, bbox: list[int]) -> float:
    x0, y0, x1, y1 = bbox
    crop = cv2.cvtColor(frame_bgr[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2GRAY)
    if crop.shape[1] > 512:
        crop = cv2.resize(crop, (512, max(1, int(512 * crop.shape[0] / crop.shape[1]))))
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def _occlusion_box(hands: list[list[float]], hull: list, shape: tuple[int, int], ds: int = 4) -> float:
    h, w = shape[0] // ds, shape[1] // ds
    hull_m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hull_m, [np.asarray(hull, np.int32) // ds], 1)
    hand_m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in hands:
        cv2.rectangle(hand_m, (int(x0) // ds, int(y0) // ds), (int(x1) // ds, int(y1) // ds), 1, -1)
    denom = int(hull_m.sum())
    return float((hull_m & hand_m).sum() / denom) if denom else 1.0


def _occlusion_pixel(hand: np.ndarray, hull: list, mb: np.ndarray) -> tuple[float, float, float]:
    hull_m = np.zeros(mb.shape, np.uint8)
    cv2.fillPoly(hull_m, [np.asarray(hull, np.int32)], 1)
    hull_area = int(hull_m.sum())
    occ_hull = float((hand & (hull_m > 0)).sum() / hull_area) if hull_area else 1.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_PX, RING_PX))
    ring = (cv2.dilate(mb.astype(np.uint8), kernel) > 0) & ~mb
    ring_area = int(ring.sum())
    contact = float((hand & ring).sum() / ring_area) if ring_area else 0.0
    return occ_hull + CONTACT_W * contact, occ_hull, contact


def _hand_masks_pixel(cache_dir: Path, cap, idxs: list[int], hands: dict,
                      sam2_cfg: str, sam2_ckpt: Path, device: str) -> dict[int, np.ndarray] | None:
    """SAM2 图像模式手 mask(磁盘缓存)。SAM2 不可用返回 None -> 调用方退回手框版。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    need = [i for i in idxs if not (cache_dir / f"f{i:06d}.png").exists()]
    if need:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            print(f"[select_frame] SAM2 不可用({exc!r}), 第二段退回手框遮挡", flush=True)
            return None
        if not Path(sam2_ckpt).is_file():
            print(f"[select_frame] SAM2 checkpoint 缺失({sam2_ckpt}), 第二段退回手框遮挡", flush=True)
            return None
        predictor = SAM2ImagePredictor(build_sam2(sam2_cfg, str(sam2_ckpt), device=device))
        for i in need:
            frame = read_frame(cap, i)
            union = np.zeros(frame.shape[:2], np.uint8)
            boxes = hands.get(i, [])
            if boxes:
                predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for box in boxes:
                    m, _, _ = predictor.predict(box=np.asarray(box), multimask_output=False)
                    union |= (m[0] > 0).astype(np.uint8)
            cv2.imwrite(str(cache_dir / f"f{i:06d}.png"), union * 255)
    return {i: cv2.imread(str(cache_dir / f"f{i:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0 for i in idxs}


def _rel_norm(vals: dict[int, float]) -> dict[int, float]:
    med = float(np.median(list(vals.values()))) or 1e-9
    return {k: v / med for k, v in vals.items()}


def select_frames_for_track(manifest: dict, object_id: str, cap, hands: dict,
                            *, cache_dir: Path, sam2_cfg: str, sam2_ckpt: Path,
                            device: str = "cuda", top_k: int = 6) -> dict | None:
    """两段式选帧。返回 report dict(含 ranked/candidates_nms/occ_detail), 无可用帧返回 None。"""
    accepted = load_accepted(manifest, object_id)
    idxs = sorted(accepted)
    if not idxs:
        return None
    feats = {i: _geom_features(cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0) for i in idxs}

    areas = {i: feats[i]["area"] for i in idxs}
    rel_area = {i: areas[i] / (np.median([areas[j] for j in idxs if abs(j - i) <= WINDOW]) or 1e-9)
                for i in idxs}
    sol_rel = _rel_norm({i: feats[i]["solidity"] for i in idxs})
    fill_rel = _rel_norm({i: feats[i]["fill"] for i in idxs})
    drop: dict[int, list[str]] = {}
    for i in idxs:
        why = []
        if feats[i]["border"]:
            why.append("border")
        if feats[i]["frag"] < FRAG_MIN:
            why.append("frag")
        if not (REL_AREA[0] <= rel_area[i] <= REL_AREA[1]):
            why.append("rel_area")
        if sol_rel[i] < SOLIDITY_REL:
            why.append("solidity")
        if fill_rel[i] < FILL_REL:
            why.append("fill")
        if why:
            drop[i] = why
    stage1 = [i for i in idxs if i not in drop]
    if not stage1:  # 全被硬门砍掉: 放宽为按干净度取前 STAGE1_CAP
        stage1 = sorted(idxs, key=lambda i: -(sol_rel[i] * fill_rel[i]))[:STAGE1_CAP]

    blur = {i: _blur_score(read_frame(cap, i), feats[i]["bbox"]) for i in stage1}
    if len(stage1) > 8:
        cut = float(np.quantile(list(blur.values()), BLUR_DROP))
        for i in [i for i in stage1 if blur[i] < cut]:
            drop[i] = ["blur"]
        stage1 = [i for i in stage1 if blur[i] >= cut]
    clean = {i: sol_rel[i] * fill_rel[i] * (blur[i] / (np.median(list(blur.values())) or 1e-9))
             for i in stage1}
    stage1 = sorted(stage1, key=lambda i: -clean[i])[:STAGE1_CAP]

    detail: dict[int, dict] = {}
    hmasks = _hand_masks_pixel(cache_dir, cap, stage1, hands, sam2_cfg, sam2_ckpt, device)
    if hmasks is not None:
        occ = {}
        for i in stage1:
            mb = cv2.imread(str(accepted[i]), cv2.IMREAD_GRAYSCALE) > 0
            occ[i], oh, ct = _occlusion_pixel(hmasks[i], feats[i]["hull"], mb)
            detail[i] = {"occ_hull": round(oh, 4), "contact": round(ct, 4)}
        hand_mode = "pixel"
    else:
        shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        occ = {i: _occlusion_box(hands.get(i, []), feats[i]["hull"], shape) for i in stage1}
        hand_mode = "box"

    ranked = sorted(stage1, key=lambda i: (round(occ[i] / OCC_TIE), -clean[i]))
    top = []
    for i in ranked:
        if all(abs(i - j) >= NMS_GAP for j in top):
            top.append(i)
        if len(top) >= top_k:
            break
    return {
        "n_accepted": len(idxs), "n_stage1": len(stage1), "hand_mode": hand_mode,
        "dropped": {str(k): v for k, v in sorted(drop.items())},
        "occ": {str(i): round(occ[i], 4) for i in ranked},
        "occ_detail": {str(i): detail[i] for i in ranked if i in detail},
        "clean": {str(i): round(clean[i], 3) for i in ranked},
        "ranked": ranked, "candidates_nms": top,
        "chosen_frame": top[0] if top else None,
    }


# ══════════════════════════ 第 3 层: Qwen 终审 ══════════════════════════

def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(match.group(0))


def _crop_pair(frame: np.ndarray, mb: np.ndarray, out_raw: Path, out_overlay: Path) -> None:
    ys, xs = np.nonzero(mb)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max((x1 - x0 + 1) * CROP_EXPAND / 2, (y1 - y0 + 1) * CROP_EXPAND / 2, 80)
    X0, X1 = max(0, int(cx - half)), min(frame.shape[1], int(cx + half))
    Y0, Y1 = max(0, int(cy - half)), min(frame.shape[0], int(cy + half))
    raw = frame[Y0:Y1, X0:X1]
    over = raw.copy()
    sub = mb[Y0:Y1, X0:X1]
    over[sub] = (0.6 * over[sub] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (0, 0, 255), 3)
    for img, path in ((raw, out_raw), (over, out_overlay)):
        scale = CROP_LONG_SIDE / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        cv2.imwrite(str(path), img)


def _load_mask_for(manifest: dict, object_id: str, frame_idx: int) -> np.ndarray:
    for frame in manifest["frames"]:
        if int(frame["frame_idx"]) == frame_idx:
            obj = (frame.get("objects") or {}).get(object_id)
            return cv2.imread(str(obj["mask"]), cv2.IMREAD_GRAYSCALE) > 0
    raise KeyError(f"{object_id} f{frame_idx}")


def make_qwen_caller(v17a_root: Path):
    """返回 (call, ok)。导入失败或 key 缺失 -> (None, 原因)。
    统一走 experimental/hoi_detr_v17a 的 qwen_client(端点/模型/本地 vLLM 均由环境变量控制,
    enable_thinking 默认 False —— 满足 Qwen 收口硬约束 1)。"""
    import sys
    sys.path.insert(0, str(v17a_root))
    try:
        from experiments.hoi_detr.qwen_client import build_user_content, call_qwen, make_client
    except (ImportError, SystemExit) as exc:
        return None, f"qwen_client unavailable: {exc}"

    client = make_client()

    def call(system: str, text: str, images: list[Path], log: list, tag: str, retries: int = 2):
        content = build_user_content(text, images)
        for attempt in range(retries + 1):
            try:
                resp = call_qwen(system, content, client=client)
                log.append({"tag": tag, "attempt": attempt, "response": resp.content})
                return _parse_json(resp.content)
            except Exception as exc:  # API 或解析失败都重试
                log.append({"tag": tag, "attempt": attempt, "error": repr(exc)})
        return None

    return call, "ok"


def arbitrate(reports: dict[str, dict], track_stats: dict[str, dict], manifest: dict,
              cap, audit_dir: Path, qwen_call) -> dict[str, dict]:
    """终审: 每 track verify(+top-K 比选), 然后跨 track 主体仲裁。qwen_call=None 时全走几何。"""
    audit_dir.mkdir(parents=True, exist_ok=True)
    final: dict[str, dict] = {}
    for object_id, rep in reports.items():
        cands = rep.get("candidates_nms") or []
        if not cands:
            final[object_id] = {"final_frame": None, "source": "no_candidates"}
            continue
        top1 = cands[0]
        log: list = []
        entry = {"geometric_top1": top1, "candidates": cands}
        verify = None
        if qwen_call is not None:
            raw_p = audit_dir / f"{object_id}_f{top1:06d}_raw.jpg"
            over_p = audit_dir / f"{object_id}_f{top1:06d}_overlay.jpg"
            _crop_pair(read_frame(cap, top1), _load_mask_for(manifest, object_id, top1), raw_p, over_p)
            verify = qwen_call(QWEN_SYSTEM, VERIFY_PROMPT, [raw_p, over_p], log, "verify")
            entry["verify"] = verify

        if verify is None:
            entry.update(final_frame=top1, source="geometric_fallback")
        elif not verify.get("is_discrete_object", True):
            entry.update(final_frame=None, source="spurious_track", skip_reconstruction=True)
        elif all((verify.get(c) or {}).get("pass") for c in HARD_CRITERIA):
            entry.update(final_frame=top1, source="qwen_accept")
        else:
            labels = [chr(ord("A") + n) for n in range(len(cands))]
            paths = []
            for lab, idx in zip(labels, cands):
                p = audit_dir / f"{object_id}_cand{lab}_f{idx:06d}.jpg"
                _crop_pair(read_frame(cap, idx), _load_mask_for(manifest, object_id, idx),
                           audit_dir / f"{object_id}_cand{lab}_raw_unused.jpg", p)
                paths.append(p)
            prompt = FALLBACK_PROMPT.format(
                n=len(cands), labels=", ".join(f"{l}=帧{i}" for l, i in zip(labels, cands)))
            pick = qwen_call(QWEN_SYSTEM, prompt, paths, log, "fallback")
            if pick and pick.get("choice") in labels:
                entry.update(final_frame=cands[labels.index(pick["choice"])],
                             source="qwen_override", fallback=pick)
            else:
                entry.update(final_frame=top1, source="geometric_fallback",
                             low_confidence=True, fallback=pick)
        entry["qwen_log"] = log
        final[object_id] = entry

    # ── 主体仲裁 ──
    passers = [oid for oid, e in final.items() if e.get("final_frame") is not None]
    if len(passers) == 1:
        final[passers[0]]["interaction_target"] = True
    elif len(passers) > 1:
        fracs = {oid: (track_stats.get(oid) or {}).get("hand_link_fraction", 0.0) for oid in passers}
        ranked_p = sorted(passers, key=lambda o: -fracs[o])
        if fracs[ranked_p[0]] >= 0.5 and all(fracs[o] < 0.25 for o in ranked_p[1:]):
            winner, how = ranked_p[0], f"hand_link_margin({fracs[ranked_p[0]]:.0%})"
        elif qwen_call is not None:
            labels = [chr(ord("A") + n) for n in range(len(ranked_p))]
            paths = []
            for lab, oid in zip(labels, ranked_p):
                detail = reports[oid].get("occ_detail") or {}
                cf = (int(max(detail, key=lambda k: detail[k].get("contact", 0)
                              + detail[k].get("occ_hull", 0)))
                      if detail else int(final[oid]["final_frame"]))
                p = audit_dir / f"target_{lab}_{oid}_f{cf:06d}.jpg"
                _crop_pair(read_frame(cap, cf), _load_mask_for(manifest, oid, cf),
                           audit_dir / f"target_{lab}_{oid}_raw_unused.jpg", p)
                paths.append(p)
            log: list = []
            prompt = TARGET_PROMPT.format(
                n=len(ranked_p), labels=", ".join(f"{l}={o}" for l, o in zip(labels, ranked_p)))
            pick = qwen_call(QWEN_SYSTEM, prompt, paths, log, "target")
            if pick and pick.get("target") in labels:
                winner, how = ranked_p[labels.index(pick["target"])], "qwen_target"
            else:
                winner, how = ranked_p[0], "hand_link_fallback"
            final[winner].setdefault("qwen_log", []).extend(log)
        else:
            winner, how = ranked_p[0], "hand_link_fallback"
        for oid in passers:
            final[oid]["interaction_target"] = oid == winner
        final[winner]["target_source"] = how
    return final
