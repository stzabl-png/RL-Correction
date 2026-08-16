#!/usr/bin/env python
"""物体位姿的投影自检 —— 把 mesh 按估计位姿投影回图像, 和 2D mask 比对。

动机: 重建产物 `world_fused.npz` 里的 `object_confidence` / `object_pose_valid_by_frame`
在实测中是**恒定值**(27_scene 全 188 帧 confidence=1.000), 即管线目前没有任何逐帧位姿
质量信号。本工具用"网格投影轮廓 vs 2D mask"提供第一个独立的几何信号。

★ 关键: 不要用朴素 IoU。
  v17A/SAM2 的 mask 是 **modal**(只含当前帧可见区域, 见 v17A README), 而网格投影是
  **amodal**(完整轮廓)。手一遮住物体, 投影必然大于 mask —— 这不是位姿误差。
  所以主信号取非对称的 `explained`, 并另算一个扣掉手 mask 的遮挡感知 IoU。

逐帧指标:
  explained  = |mask ∩ proj| / |mask|              主信号, 对遮挡鲁棒
                                                   "每个可见的物体像素都被投影解释到了吗"
  iou_occ    = IoU(mask, proj \\ hands)             扣掉手遮挡后的 IoU
  iou_raw    = IoU(mask, proj)                     参考; 与 iou_occ 的差 = modal/amodal 缺口
  area_ratio = |proj| / |mask|                     尺度线索(网格 scale 错会系统性偏离)
  d_centroid = 质心像素距离

⚠ 解读边界:
  1. **多物体 take 必须给 `--object`。** 投影侧是单物体(一份网格 + 一条位姿), 不给的话
     mask 侧会合并全部物体, `explained` 被另一个物体整体拉低 —— 那是口径错配, 不是位姿误差。
     实测 pour/17(杯+瓶): 合并 mask **0.479 / 质心距 177px**(看着像位姿烂透了),
     `--object object_0` **0.950 / 质心距 8px**。同一份位姿, 差别全在口径。
     ★ 2026-08-15 我本人就是这么误判的, 还据此对外说"绝对值不可信"。
  2. 轮廓对"绕自身对称轴的自转"是盲的。水瓶近似旋转对称, 本工具**无法**校验/修正
     拧盖角度。它能查的是平移、倾角和尺度。
  3. **跨重建版本比位姿时不要比四元数。** 若两版各自重建了网格, 物体系(网格规范系)不同,
     同一物理朝向会对应相差上百度的四元数 —— 实测 pour/17 两版四元数差 161.7°, 而摆进
     世界后点云中位差仅 **0.7mm**、本工具两版 explained 0.950 vs 0.9xx 无差异。
     要比就摆进世界比几何, 或用本工具比像素。

★ 内参: 从 `world_fused.npz` 的 `K` 逐视频读取(EgoDex pour/17 实测 fx=fy=736.63)。
  (旧注释曾写"内参是 dataset_constant fx=741.44 给投影设了误差地板" —— 那是本工具早期
   用在 HOI4D 上时的情况, **代码早已按视频读 K**, 该说法 2026-08-15 已订正。
   教训: 文档与实现会不同步, 判断"当前行为"要看代码。)

用法:
  # 单物体
  python confidence/pose_projection_check.py --scene <take> --video <mp4> --out <dir>
  # ★ 多物体: 必须指定
  ... --object object_1
  # 换成 v17A 的独立 mask 做交叉验证:
  ... --mask-source v17a --v17a-take /path/to/VideoPrior/takes/screw_unscrew_bottle_cap/27
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_MASK = (255, 200, 0)      # 2D mask       青
COL_PROJ = (200, 60, 235)     # 投影轮廓       品红
COL_MISS = (0, 220, 255)      # mask 有投影无  黄 ← 真失配的强信号
COL_EXTRA = (90, 90, 235)     # 投影有 mask 无 红 (遮挡时正常)
COL_HAND = {"left": (96, 232, 96), "right": (255, 168, 64)}
STRIP_H = 96
LABEL_W = 78


# ------------------------------------------------------------------ 几何
def project_silhouette(V: np.ndarray, F_front_idx, K: np.ndarray,
                       T: np.ndarray, shape) -> np.ndarray:
    """把网格按 object->camera 变换 T 投影成轮廓 mask。

    闭合网格只填正面三角形即可得到精确轮廓, 省一半光栅化。
    """
    H, W = shape
    Vc = (T[:3, :3] @ V.T).T + T[:3, 3]
    z = Vc[:, 2]
    if not np.isfinite(z).all() or (z <= 1e-6).all():
        return np.zeros((H, W), bool)
    uv = (K @ Vc.T).T
    uv = uv[:, :2] / np.where(np.abs(uv[:, 2:3]) < 1e-9, 1e-9, uv[:, 2:3])
    tri = uv[F_front_idx]
    # 背面剔除按当前投影重算(朝向随位姿变)
    area = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = tri[area < 0]
    img = np.zeros((H, W), np.uint8)
    if len(front):
        cv2.fillPoly(img, [t.astype(np.int32) for t in front], 255)
    return img > 0


def centroid(m: np.ndarray):
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def metrics(mask: np.ndarray, proj: np.ndarray, hands: np.ndarray | None) -> dict:
    am, ap = int(mask.sum()), int(proj.sum())
    inter = int((mask & proj).sum())
    union = int((mask | proj).sum())
    proj_vis = proj & ~hands if hands is not None else proj
    iv = int((mask & proj_vis).sum())
    uv = int((mask | proj_vis).sum())
    cm, cp = centroid(mask), centroid(proj)
    d = (float(np.hypot(cm[0] - cp[0], cm[1] - cp[1]))
         if (cm and cp) else None)
    return {
        "mask_px": am, "proj_px": ap,
        "explained": round(inter / am, 4) if am else None,
        "iou_raw": round(inter / union, 4) if union else None,
        "iou_occ": round(iv / uv, 4) if uv else None,
        "area_ratio": round(ap / am, 4) if am else None,
        "d_centroid": round(d, 2) if d is not None else None,
    }


# ------------------------------------------------------------------ mask 源
def scene_object_mask(scene: Path, i: int, object_id: str | None = None):
    """object_id=None 时**合并全部物体**的 mask; 给了就只取那一个。

    ⚠ 多物体 take 上必须给 object_id ——本工具的投影侧是**单物体**(一份网格 + 一条位姿),
      mask 侧若合并了两个物体, `explained = |mask ∩ proj| / |mask|` 会被另一个物体
      整体拉低, 而这与位姿好坏无关。
      实测 pour/17(杯+瓶): 合并 mask -> explained 0.479 / 质心距 177px(看着像位姿很烂),
      只取 object_0 -> **explained 0.950 / 质心距 8px**。同一份位姿, 差别全是口径。
    """
    d = scene / "masks" / "objects" / "frames" / f"frame_{i:06d}_masks"
    if not d.is_dir():
        return None
    paths = ([d / f"{object_id}.png"] if object_id else sorted(d.glob("object_*.png")))
    acc = None
    for p in paths:
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.is_file() else None
        if a is None:
            continue
        b = a > 127
        acc = b if acc is None else (acc | b)
    return acc


def v17a_object_mask(frames_by_idx: dict, i: int):
    acc = None
    for _oid, o in (frames_by_idx.get(i) or {}).items():
        p = o.get("mask")
        if not p or str(p) == "None":
            p = o.get("raw_mask") or o.get("unresolved_mask")   # tracking 语义
        if not p or str(p) == "None":
            continue
        a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if a is None:
            continue
        b = a > 127
        acc = b if acc is None else (acc | b)
    return acc


def hand_masks(dirs: list[Path], i: int):
    acc = None
    for base in dirs:
        d = base / f"frame_{i:06d}_masks"
        if not d.is_dir():
            continue
        for side in ("left", "right"):
            a = cv2.imread(str(d / f"{side}_hand_0.png"), cv2.IMREAD_GRAYSCALE)
            if a is None:
                continue
            b = a > 127
            acc = b if acc is None else (acc | b)
    return acc


# ------------------------------------------------------------------ 渲染
def build_strip(W: int, n: int, series: dict) -> np.ndarray:
    """两条曲线: explained(主) 与 iou_occ。纵轴 0~1。"""
    strip = np.full((STRIP_H, W, 3), 26, np.uint8)
    x0, x1, y0, y1 = LABEL_W, W - 10, 8, STRIP_H - 20
    cv2.rectangle(strip, (x0, y0), (x1, y1), (44, 44, 44), -1)
    for v, lab in ((1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")):
        y = int(y1 - v * (y1 - y0))
        cv2.line(strip, (x0, y), (x1, y), (70, 70, 70), 1)
        cv2.putText(strip, lab, (6, y + 4), FONT, 0.32, (130, 130, 130), 1, cv2.LINE_AA)
    for key, col in (("explained", (120, 255, 120)), ("iou_occ", COL_PROJ)):
        pts = []
        for i, v in enumerate(series[key]):
            if v is None:
                continue
            x = int(x0 + (i / max(1, n - 1)) * (x1 - x0))
            pts.append((x, int(y1 - float(v) * (y1 - y0))))
        if len(pts) > 1:
            cv2.polylines(strip, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
    cv2.putText(strip, "explained", (6, STRIP_H - 6), FONT, 0.36, (120, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(strip, "iou_occ", (100, STRIP_H - 6), FONT, 0.36, COL_PROJ, 1, cv2.LINE_AA)
    for i in range(0, n, 30):
        x = int(x0 + (i / max(1, n - 1)) * (x1 - x0))
        cv2.line(strip, (x, y1), (x, y1 + 4), (90, 90, 90), 1)
    return strip


def blend(img, m, color, alpha):
    if m is None or not m.any():
        return
    sub = img[m].astype(np.float32)
    img[m] = (sub * (1 - alpha) + np.asarray(color, np.float32) * alpha).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--npz", type=Path, default=None,
                    help="world_fused.npz (默认 <scene>/world_fused.npz)")
    ap.add_argument("--mesh", type=Path, default=None)
    ap.add_argument("--mask-source", choices=("scene", "v17a"), default="scene")
    ap.add_argument("--object", default=None,
                    help="物体 id(如 object_1)。多物体 take **必须给** —— 不给会合并所有"
                         "物体的 mask 去比单物体投影, explained 被另一个物体拉低(实测 0.95->0.48)")
    ap.add_argument("--v17a-take", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--clean-from", type=int, default=150,
                    help="判定为无遮挡的起始帧, 用来量内参/网格造成的误差地板")
    a = ap.parse_args(argv)

    npz = a.npz or (a.scene / "world_fused.npz")
    z = np.load(npz, allow_pickle=True)
    K = np.asarray(z["K"], float)
    T_all = np.asarray(z["object_ob_in_cam"], float)
    fi = np.asarray(z["object_frame_indices"]).astype(int)
    idx_of = {int(f): k for k, f in enumerate(fi)}

    # ★ 逐物体: 网格 / 位姿 / mask 三者必须同指一个物体, 否则量的是口径错配不是位姿误差
    oid = a.object or "object_0"
    if a.object and "object_ob_in_cam_all" in z.files:
        ids = [str(x) for x in z["object_ids"]] if "object_ids" in z.files else []
        if a.object in ids:
            T_all = np.asarray(z["object_ob_in_cam_all"], float)[ids.index(a.object)]
            print(f"[proj] 逐物体: {a.object} (取 object_ob_in_cam_all[{ids.index(a.object)}])")
        else:
            print(f"[proj] ⚠ world_fused 里没有 {a.object}, 现有 {ids}; 退回单物体位姿")
    mesh_p = a.mesh or (a.scene / "objects" / oid / "object_mesh_scaled_final.obj")
    mesh = trimesh.load(str(mesh_p), force="mesh")
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, int)
    print(f"[proj] mesh V={len(V)} F={len(F)} extents={np.round(mesh.extents,4)}  K fx={K[0,0]:.2f}")
    isrc = str(z["intrinsics_source"]) if "intrinsics_source" in z.files else "unknown"
    print(f"[proj] intrinsics_source={isrc}  位姿帧数={len(T_all)}")

    frames_by_idx = {}
    if a.mask_source == "v17a":
        man = json.loads((a.v17a_take / "v17a" / "video_mask_sequence.json").read_text())
        frames_by_idx = {int(f["frame_idx"]): f["objects"] for f in man["frames"]}
    hand_dirs = [a.scene / "masks" / "hands" / "frames"]
    if a.v17a_take:
        hand_dirs.append(a.v17a_take / "hands")
    hand_dirs = [d for d in hand_dirs if d.is_dir()]

    cap = cv2.VideoCapture(str(a.video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    Hf = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Wf = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    W = int(Wf * a.scale) // 2 * 2
    H = int(Hf * a.scale) // 2 * 2

    # ---- 第一遍: 只算指标 ----
    t0 = time.time()
    per = []
    for i in range(n):
        k = idx_of.get(i)
        mask = (scene_object_mask(a.scene, i, a.object) if a.mask_source == "scene"
                else v17a_object_mask(frames_by_idx, i))
        rec = {"frame": i, "t": round(i / fps, 3)}
        if k is None or mask is None or not mask.any():
            rec.update({"explained": None, "iou_occ": None, "iou_raw": None,
                        "area_ratio": None, "d_centroid": None,
                        "reason": "no_pose" if k is None else "no_mask"})
        else:
            proj = project_silhouette(V, F, K, T_all[k], (Hf, Wf))
            rec.update(metrics(mask, proj, hand_masks(hand_dirs, i)))
        per.append(rec)
        if (i + 1) % 40 == 0:
            print(f"  指标 {i+1}/{n} ({time.time()-t0:.0f}s)")

    series = {kk: [r.get(kk) for r in per] for kk in ("explained", "iou_occ")}
    strip = build_strip(W, n, series)
    x0, x1 = LABEL_W, W - 10

    # ---- 第二遍: 渲染 ----
    a.out.mkdir(parents=True, exist_ok=True)
    outmp4 = a.out / f"{a.scene.name}_poseproj_{a.mask_source}.mp4"
    vw = cv2.VideoWriter(str(outmp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + STRIP_H))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        k = idx_of.get(i)
        mask = (scene_object_mask(a.scene, i, a.object) if a.mask_source == "scene"
                else v17a_object_mask(frames_by_idx, i))
        r = per[i]
        if k is not None and mask is not None and mask.any():
            proj = project_silhouette(V, F, K, T_all[k], (Hf, Wf))
            sh = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            ms, ps = sh(mask), sh(proj)
            blend(small, ms & ps, COL_MASK, 0.30)        # 一致
            blend(small, ms & ~ps, COL_MISS, 0.55)       # mask 有投影无 ← 真失配
            blend(small, ps & ~ms, COL_EXTRA, 0.22)      # 投影有 mask 无 (遮挡时正常)
            cs, _ = cv2.findContours(ps.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(small, cs, -1, COL_PROJ, 2, cv2.LINE_AA)
        f = lambda v: "  --" if v is None else f"{v:5.3f}"
        lines = [
            (f"{a.scene.name}  mask-source={a.mask_source}", (235, 235, 235)),
            (f"frame {i:04d}/{n-1:04d}  t={i/fps:5.2f}s", (200, 200, 200)),
            (f"explained {f(r.get('explained'))}   (mask covered by projection)",
             (120, 255, 120)),
            (f"iou_occ   {f(r.get('iou_occ'))}   iou_raw {f(r.get('iou_raw'))}", COL_PROJ),
            (f"area_ratio{f(r.get('area_ratio'))}   d_centroid "
             f"{'--' if r.get('d_centroid') is None else format(r['d_centroid'],'.1f')}px",
             (190, 190, 190)),
            ("cyan=agree  YELLOW=mask w/o proj  red=proj w/o mask", (170, 170, 170)),
        ]
        x, y = 10, 10
        hh = 17 * len(lines) + 10
        ww = 8 + max(int(cv2.getTextSize(t, FONT, 0.44, 1)[0][0]) for t, _ in lines) + 10
        small[y:y+hh, x:x+ww] = (small[y:y+hh, x:x+ww].astype(np.float32) * 0.35).astype(np.uint8)
        for j, (t, c) in enumerate(lines):
            cv2.putText(small, t, (x + 6, y + 17 + j * 17), FONT, 0.44, c, 1, cv2.LINE_AA)
        s = strip.copy()
        px = int(x0 + (i / max(1, n - 1)) * (x1 - x0))
        cv2.line(s, (px, 8), (px, STRIP_H - 20), (255, 255, 255), 1)
        vw.write(np.vstack([small, s]))
    cap.release(); vw.release()

    # ---- 汇总 ----
    val = [r for r in per if r.get("explained") is not None]
    clean = [r for r in val if r["frame"] >= a.clean_from]
    inter = [r for r in val if r["frame"] < a.clean_from]

    def stat(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        if not v:
            return None
        v = np.array(v, float)
        return {"n": len(v), "min": round(float(v.min()), 4),
                "p10": round(float(np.percentile(v, 10)), 4),
                "median": round(float(np.median(v)), 4),
                "max": round(float(v.max()), 4)}

    summary = {
        "scene": str(a.scene), "mask_source": a.mask_source, "object": a.object,
        "intrinsics_source": isrc,
        "K_fx": float(K[0, 0]), "num_frames": n, "frames_with_metric": len(val),
        "clean_from": a.clean_from,
        "clean(no-occlusion) 误差地板": {k: stat(clean, k) for k in
                                        ("explained", "iou_occ", "iou_raw", "area_ratio", "d_centroid")},
        "interaction(occluded)": {k: stat(inter, k) for k in
                                  ("explained", "iou_occ", "iou_raw", "area_ratio", "d_centroid")},
        "worst_explained": sorted(
            [{k: r[k] for k in ("frame", "t", "explained", "iou_occ", "area_ratio", "d_centroid")}
             for r in val], key=lambda r: r["explained"])[:12],
    }
    (a.out / f"{a.scene.name}_poseproj_{a.mask_source}.json").write_text(
        json.dumps({"summary": summary, "per_frame": per}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n[proj] 视频: {outmp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
