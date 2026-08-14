"""尺度三路共识融合(从 agent/v17a-scale-develop 移植, 2026-08-14)。

L_extent(多帧点云跨度) x L_hand(手长锚点) x L_typical(类别常识), 一次 Qwen 调用。
核心洞察 **不信朝向**: 单帧 PCA 主轴比的误差全部来自 mesh<->观测的朝向对齐
(扫把: 观测 18.4cm 对, 朝向歪 -> 40cm), 多帧 mask 反投影点云的鲁棒跨度(P2-P98)
不经过任何朝向, 直接当最长边。

融合规则(验证值 tol=1.6):
  1. Qwen 锚点仲裁 extent 双聚合口径(median_all vs median_topclean, 单口径无全胜)
  2. 三路两两一致 -> consensus, 用 L_extent(唯一"量出来"的)
  3. L_extent 离群且两锚点互洽 -> corrected_by_prior, 用 geomean(锚点)
  4. 锚点互相打架 -> extent + low_confidence
  5. identity confidence 低 -> 弃 L_typical(手长锚点不依赖认对类别 ——
     laptop 被认成"砧板"但尺寸照样准, 双锚点容错的实证)
实测: ketchup 1.60x->0.96x / laptop 3.07x->1.15x / 扫把 ~2x->约1x。

主线约定(SCALE_TODO.md): 保留几何路径当对照, 产物记 scale_geometric / scale_fused /
scale_verdict; Qwen 不可达 -> 不改 final mesh(退回旧行为), 只记 extent 候选。
帧池优先复用 select_frame step 的 ranked 干净帧(同一套质量判据), 缺失则均匀采样。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

TOL = 1.6
HAND_CM = 18.5
MAX_FRAMES = 60
MIN_PTS = 200
CROP_LONG_SIDE = 1024

RECON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RECON_ROOT.parent.parent.parent
V17A_ROOT = REPO_ROOT / "experimental" / "hoi_detr_v17a"

SYSTEM = (
    "你是 3D 重建管线的尺度审核员。依据画面里人手与物体的相对大小以及你对该类物体的"
    "常识,估计物体的真实尺寸。人手(腕根到中指尖)长度约 18-19cm。"
    "只输出要求的 JSON,不要多余文字。"
)
# ⚠ 本 prompt 为验证版原文(ketchup 0.96x / laptop 1.15x / 扫把 ~1x), 字段行内注释是
# 语义定义的一部分 —— 2026-08-14 实测删掉注释后 Qwen 会把 hand_lengths 按厘米填
# (17-20cm -> 锚点 3.4m)。任何改动都要在对照 take 上重验。
PROMPT = """图1:手与物体接触/抓握的帧(手和物体距离相机深度相近,像素大小可直接比较)。
图2:物体的 mask 叠加图(红色区域即目标物体,请以这个物体为准)。
请估计目标物体的**最长跨度**。注意:一律按物体在**画面中的当前构型**估计
(例如摊开的笔记本按摊开后的跨度,而非合拢状态)。
只输出 JSON:
{
  "object_description": "目标物体是什么(一句话)",
  "identity_confidence": "high/medium/low",  // 你对认出这是什么物体的把握
  "current_configuration": "物体当前构型/状态(一句话,如'完全摊开'/'直立放置')",
  "hand_lengths": {"low": 数值, "high": 数值},   // 当前构型最长跨度 = 多少个手长,保守区间
  "typical_size_cm": {"low": 数值, "high": 数值}, // 这类物体在该构型下最长跨度通常多少厘米
  "reason": "判断依据(一句话)"
}"""


def _geomean(lo: float, hi: float) -> float:
    return float(np.sqrt(max(lo, 1e-6) * max(hi, 1e-6)))


def _ratio_ok(a: float, b: float, tol: float = TOL) -> bool:
    return 1.0 / tol <= a / b <= tol


def _mask_at(obj_dir: Path, frame_idx: int, object_id: str):
    import cv2
    p = obj_dir / "video_segmentation" / "masks" / f"frame_{frame_idx:06d}_masks" / f"{object_id}.png"
    if not p.is_file():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def _frame_estimate(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> dict | None:
    """单帧: 清理 mask -> 反投影 -> PCA 主轴 P2-P98 跨度(米)。"""
    import cv2
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    m = cv2.erode(m, np.ones((9, 9), np.uint8)) > 0
    valid = m & np.isfinite(depth) & (depth > 1e-4)
    if valid.sum() < MIN_PTS:
        return None
    z = depth[valid]
    z_lo, z_hi = np.percentile(z, [5, 95])
    band = max(z_hi - z_lo, 0.02)
    keep = (z >= z_lo - 0.5 * band) & (z <= z_hi + 0.5 * band)
    vs, us = np.nonzero(valid)
    us, vs, z = us[keep], vs[keep], z[keep]
    if len(z) < MIN_PTS:
        return None
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    pts = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=1)
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    span = float(np.percentile(proj[:, 0], 98) - np.percentile(proj[:, 0], 2))
    return {"n_points": int(len(z)), "principal_span_m": round(span, 4)}


def _select_frame_report(dataset: str, video_id: str) -> dict:
    from _common.paths import interim_step_dir
    p = interim_step_dir(dataset, video_id, "select_frame") / "report.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _extent_candidates(job, object_id: str, obj_dir: Path, vipe_dir: Path,
                       k_mat: np.ndarray, depth_scale: float) -> dict | None:
    """多帧跨度的两个聚合口径 + 帧池信息。"""
    from _common.io import iter_vipe_depth_frames
    from fp_common import _depth_to_meters

    sf = _select_frame_report(job.dataset, job.video_id)
    v17a_oid = None
    for pid, voi in (sf.get("pipeline_mapping") or {}).items():
        if pid == object_id:
            v17a_oid = voi
    sf_obj = (sf.get("objects") or {}).get(v17a_oid or "", {})
    ranked = [int(f) for f in sf_obj.get("ranked") or []]

    mask_root = obj_dir / "video_segmentation" / "masks"
    all_frames = sorted(int(p.name.split("_")[1]) for p in mask_root.glob("frame_*_masks")
                        if (p / f"{object_id}.png").is_file())
    pool = ranked if ranked else all_frames
    if len(pool) > MAX_FRAMES:
        pool = pool[:MAX_FRAMES] if ranked else \
            [pool[i] for i in np.linspace(0, len(pool) - 1, MAX_FRAMES).astype(int)]
    pool_set = set(pool)
    if not pool_set:
        return None

    per_frame: dict[int, dict] = {}
    for idx, depth_raw in iter_vipe_depth_frames(vipe_dir, job.video_id):
        if idx not in pool_set:
            continue
        mask = _mask_at(obj_dir, idx, object_id)
        if mask is None or not mask.any():
            continue
        depth = _depth_to_meters(depth_raw, depth_scale)
        if depth.shape != mask.shape:
            import cv2
            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        est = _frame_estimate(depth, mask, k_mat)
        if est is not None:
            per_frame[idx] = est
    if not per_frame:
        return None

    spans = np.array([e["principal_span_m"] for e in per_frame.values()])
    npts = [e["n_points"] for e in per_frame.values()]
    gate = max(1000, int(np.percentile(npts, 25)))
    order = ranked if ranked else sorted(per_frame, key=lambda f: -per_frame[f]["n_points"])
    top_clean = [f for f in order if f in per_frame and per_frame[f]["n_points"] >= gate][:10]
    top_spans = np.array([per_frame[f]["principal_span_m"] for f in top_clean])
    return {
        "n_frames": len(spans),
        "frame_pool": "select_frame_ranked" if ranked else "sam2_object_uniform",
        "median_all_m": round(float(np.median(spans)), 4),
        "median_topclean_m": round(float(np.median(top_spans)), 4) if len(top_spans) else None,
        "std_m": round(float(spans.std()), 4),
        "min_npts_gate": gate,
        "contact_frame": (int(max(d, key=lambda k_: d[k_].get("contact", 0) + d[k_].get("occ_hull", 0)))
                          if (d := sf_obj.get("occ_detail") or {}) else None),
    }


def _qwen_anchors(job, object_id: str, obj_dir: Path, step_dir: Path,
                  frame_idx: int, contact_frame: int | None) -> dict | None:
    """一次 Qwen 调用拿手长/常识两个锚点。不可达返回 None(退回旧行为)。"""
    import cv2
    sys.path.insert(0, str(V17A_ROOT))
    try:
        from experiments.hoi_detr.qwen_client import build_user_content, call_qwen
    except (ImportError, SystemExit) as exc:
        print(f"[scale_fusion] qwen_client 不可用({exc!r}) -> 保持几何尺度", flush=True)
        return None

    def save(idx: int, path: Path, with_mask: bool) -> Path | None:
        cap = cv2.VideoCapture(str(job.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        if with_mask:
            mb = _mask_at(obj_dir, idx, object_id)
            if mb is not None:
                frame[mb] = (0.6 * frame[mb] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
                cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, cnts, -1, (0, 0, 255), 3)
        s = CROP_LONG_SIDE / max(frame.shape[:2])
        if s < 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * s), int(frame.shape[0] * s)))
        cv2.imwrite(str(path), frame)
        return path

    imgs = []
    cf = contact_frame if contact_frame is not None else frame_idx
    p1 = save(cf, step_dir / f"fusion_contact_f{cf:06d}.jpg", with_mask=False)
    p2 = save(frame_idx, step_dir / f"fusion_recon_f{frame_idx:06d}_overlay.jpg", with_mask=True)
    imgs = [p for p in (p1, p2) if p is not None]
    if not imgs:
        return None
    try:
        resp = call_qwen(SYSTEM, build_user_content(PROMPT, imgs))
        import re
        text = re.sub(r"^```(?:json)?|```$", "", resp.content.strip(), flags=re.M).strip()
        m = re.search(r"\{.*\}", text, re.S)
        est = json.loads(m.group(0))
        est["_raw"] = resp.content
        return est
    except Exception as exc:
        print(f"[scale_fusion] Qwen 调用失败({exc!r}) -> 保持几何尺度", flush=True)
        return None


def fuse_object_scale(job, *, object_id: str, step_dir: Path, obj_dir: Path, vipe_dir: Path,
                      raw_mesh_longest: float, frame_idx: int, k_mat: np.ndarray,
                      depth_scale: float, scale_geometric: float,
                      final_mesh_path: Path) -> dict:
    """返回要并入 scale metadata 的字段; 融合成立时改写 final mesh(几何版留对照副本)。"""
    out: dict = {"scale_geometric": float(scale_geometric),
                 "L_geometric_m": round(scale_geometric * raw_mesh_longest, 4),
                 "scale_fused": None, "scale_verdict": "fusion_unavailable"}
    try:
        ext = _extent_candidates(job, object_id, obj_dir, vipe_dir, k_mat, depth_scale)
    except Exception as exc:
        print(f"[scale_fusion] extent 计算失败({exc!r}) -> 保持几何尺度", flush=True)
        ext = None
    if ext is None:
        out["scale_verdict"] = "extent_unavailable"
        return out
    out["extent"] = ext

    qwen = _qwen_anchors(job, object_id, obj_dir, step_dir, frame_idx, ext.get("contact_frame"))
    if qwen is None:
        out["scale_verdict"] = "prior_unavailable_kept_geometric"
        return out

    try:
        L_hand = _geomean(float(qwen["hand_lengths"]["low"]),
                          float(qwen["hand_lengths"]["high"])) * HAND_CM / 100.0
        L_typ = _geomean(float(qwen["typical_size_cm"]["low"]),
                         float(qwen["typical_size_cm"]["high"])) / 100.0
    except (KeyError, TypeError, ValueError) as exc:
        print(f"[scale_fusion] Qwen 输出不完整({exc!r}) -> 保持几何尺度", flush=True)
        out["scale_verdict"] = "prior_malformed_kept_geometric"
        out["qwen"] = {k: v for k, v in qwen.items() if k != "_raw"}
        return out
    conf = str(qwen.get("identity_confidence", "low")).lower()
    anchors = [L_hand] + ([L_typ] if conf in ("high", "medium") else [])
    anchor_mid = float(np.exp(np.mean(np.log(anchors))))

    cand = {k: ext[k] for k in ("median_all_m", "median_topclean_m") if ext.get(k)}
    extent_key = min(cand, key=lambda k: abs(np.log(cand[k] / anchor_mid)))
    L_extent = float(cand[extent_key])

    hand_typ_agree = conf in ("high", "medium") and _ratio_ok(L_hand, L_typ)
    if _ratio_ok(L_extent, anchor_mid):
        L_final, verdict = L_extent, ("consensus" if (hand_typ_agree or conf == "low")
                                      else "extent_with_partial_prior")
    elif hand_typ_agree or conf == "low":
        L_final, verdict = anchor_mid, "corrected_by_prior"
    else:
        L_final, verdict = L_extent, "low_confidence_extent"

    scale_fused = L_final / raw_mesh_longest
    out.update({
        "qwen": {k: v for k, v in qwen.items() if k != "_raw"},
        "L_hand_m": round(L_hand, 4), "L_typical_m": round(L_typ, 4),
        "identity_confidence": conf, "anchor_mid_m": round(anchor_mid, 4),
        "extent_aggregator_chosen": extent_key, "L_extent_m": round(L_extent, 4),
        "L_fused_m": round(L_final, 4), "scale_fused": float(scale_fused),
        "scale_verdict": verdict, "fusion_tol": TOL,
        "fused_over_geometric": round(scale_fused / scale_geometric, 4),
    })

    if verdict != "low_confidence_extent":
        geometric_copy = final_mesh_path.with_name("object_mesh_scaled_geometric.obj")
        shutil.copy2(final_mesh_path, geometric_copy)
        factor = scale_fused / scale_geometric
        with open(geometric_copy) as src, open(final_mesh_path, "w") as dst:
            for line in src:
                if line.startswith("v "):
                    parts = line.split()
                    xyz = [float(x) * factor for x in parts[1:4]]
                    dst.write(f"v {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"
                              + (" " + " ".join(parts[4:]) if len(parts) > 4 else "") + "\n")
                else:
                    dst.write(line)
        out["object_mesh_scaled_geometric"] = str(geometric_copy)
        out["final_mesh_rescaled_to_fused"] = True
        print(f"[scale_fusion] {object_id}: geo={out['L_geometric_m']}m extent={L_extent}m"
              f"({extent_key}) hand={out['L_hand_m']}m typical={out['L_typical_m']}m"
              f" -> {verdict} L_final={L_final:.3f}m (x{factor:.3f})", flush=True)
    else:
        out["final_mesh_rescaled_to_fused"] = False
        print(f"[scale_fusion] {object_id}: 锚点互斥且 extent 离群 -> 保持几何尺度"
              f" (extent={L_extent}m hand={out['L_hand_m']}m typical={out['L_typical_m']}m)",
              flush=True)
    return out
