#!/usr/bin/env python3
"""尺度估计第 1 层原型:去朝向化的多帧跨度比。

原理:在选帧器 stage1 幸存帧(几何干净:不贴边/不碎裂/凸度填充率达标)上,
物体 mask 反投影点云的鲁棒主轴跨度 ≈ 物体真实最长边 × cos(透视缩短角)。
不依赖 SAM3D/FoundationPose 朝向 ——
  scale_i = obs_span_i / mesh_longest_extent(归一化 mesh)
透视缩短只会低估不会高估,故聚合用高分位(P90)兼顾鲁棒与抗缩短;中位数一并报告。

轻量:只读已有产物(vipe 深度 zip、实例 manifest mask、SAM3D raw mesh),
每帧一次反投影+PCA,无模型推理。

用法(sam3 env):
  python scale_extent_v1.py --run runs/sweep_dustpan_1 --dataset egodex --video-id sweep_dustpan_1
输出: <run>/scale_extent_v1/<object_id>.json + 终端摘要
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

SD = Path(__file__).resolve().parent
RECON = SD.parent / "ego_pipeline/Reconstruction"


def _load_io_module():
    path = RECON / "recon_pipeline/_common/io.py"
    spec = importlib.util.spec_from_file_location("_recon_io", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def mesh_longest_extent(obj_path: Path) -> float:
    vs = []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(vs)
    return float((v.max(0) - v.min(0)).max())


def clean_mask(mask: np.ndarray, erode_px: int = 4) -> np.ndarray:
    """最大连通域 + 腐蚀(去掉硅影边缘的深度渗色)。"""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8)
    if erode_px > 0:
        m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    return m > 0


def robust_span(points: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, np.ndarray]:
    """PCA 主轴上的 P2-P98 跨度(米)。返回 (span, 三轴跨度)。"""
    c = points - points.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    spans = np.percentile(proj, hi, axis=0) - np.percentile(proj, lo, axis=0)
    return float(spans[0]), spans


def frame_estimate(depth: np.ndarray, mask: np.ndarray, k: np.ndarray) -> dict | None:
    m = clean_mask(mask)
    valid = m & np.isfinite(depth) & (depth > 1e-4)
    if valid.sum() < 200:
        return None
    z = depth[valid]
    # 深度内点剔除:mask 边缘常混入背景深度,用 P5-P95 截断
    z_lo, z_hi = np.percentile(z, [5, 95])
    band = max(z_hi - z_lo, 0.02)
    keep = (z >= z_lo - 0.5 * band) & (z <= z_hi + 0.5 * band)
    vs, us = np.nonzero(valid)
    us, vs, z = us[keep], vs[keep], z[keep]
    if len(z) < 200:
        return None
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    pts = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=1)
    span, spans3 = robust_span(pts)
    return {
        "n_points": int(len(z)),
        "median_depth_m": float(np.median(z)),
        "principal_span_m": round(span, 4),
        "spans_m": [round(float(s), 4) for s in spans3],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--object-id", default=None, help="v17A track id,默认 final_selection 第一个有效项")
    ap.add_argument("--recon-object", default="object_0", help="sam3d 输出的物体目录名")
    ap.add_argument("--max-frames", type=int, default=60)
    args = ap.parse_args()
    run = args.run.resolve()
    interim = RECON / "data/interim" / args.dataset / args.video_id

    final = json.loads((run / "final_selection.json").read_text())
    object_id = args.object_id or next(
        oid for oid, e in final.items() if e.get("final_frame") is not None)
    report = json.loads((run / "frame_selection_v21/report_v2.json").read_text())["objects"][object_id]
    frames = [int(f) for f in report["ranked"][: args.max_frames]]

    manifest = json.loads(
        (run / "instance/video_mask_sequence/video_mask_sequence.json").read_text())
    mask_path = {int(fr["frame_idx"]): fr["objects"][object_id]["mask"]
                 for fr in manifest["frames"] if object_id in fr.get("objects", {})}

    io = _load_io_module()
    k, _flip = io.load_vipe_intrinsics(interim / "vipe", args.video_id)
    mesh_len = mesh_longest_extent(interim / "sam3d/objects" / args.recon_object / "object_mesh_raw.obj")

    wanted = {f for f in frames if f in mask_path}
    per_frame: dict[int, dict] = {}
    for idx, depth in io.iter_vipe_depth_frames(interim / "vipe", args.video_id):
        if idx not in wanted:
            continue
        mask = cv2.imread(mask_path[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if depth.shape != mask.shape:
            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        est = frame_estimate(depth, mask, k)
        if est is not None:
            per_frame[idx] = est

    if not per_frame:
        sys.exit("没有任何帧产生有效估计")
    spans = np.array([e["principal_span_m"] for e in per_frame.values()])
    # 主口径:干净度 top-K 且点数达标的帧取中位数。
    # 低点数帧 = 物体过远/过小/看不全,跨度系统性截断(span 与 log npts 正相关),先滤掉。
    min_npts = max(1000, int(np.percentile([e["n_points"] for e in per_frame.values()], 25)))
    top_clean = [f for f in frames if f in per_frame and per_frame[f]["n_points"] >= min_npts][:10]
    top_spans = np.array([per_frame[f]["principal_span_m"] for f in top_clean])
    agg = {
        "n_frames": len(spans),
        "median_all_m": round(float(np.median(spans)), 4),
        "p90_all_m": round(float(np.percentile(spans, 90)), 4),
        "max_m": round(float(spans.max()), 4),
        "min_m": round(float(spans.min()), 4),
        "std_m": round(float(spans.std()), 4),
        "min_npts_gate": min_npts,
        "top_clean_frames": top_clean,
        "median_topclean_m": round(float(np.median(top_spans)), 4) if len(top_spans) else None,
    }
    # 最长边估计 = 干净 top-K 中位跨度 / mesh 最长边(归一化单位)
    L_est = agg["median_topclean_m"] if agg["median_topclean_m"] else agg["median_all_m"]
    scale = L_est / mesh_len
    result = {
        "object_id": object_id,
        "mesh_longest_extent_norm": round(mesh_len, 4),
        "scale_extent_v1": round(scale, 6),
        "L_extent_m": round(L_est, 4),
        "aggregate": agg,
        "frames_used": sorted(per_frame),
        "per_frame": {str(k_): v for k_, v in sorted(per_frame.items())},
    }
    out_dir = run / "scale_extent_v1"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{object_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"[{object_id}] frames={agg['n_frames']} median_all={agg['median_all_m']}m "
          f"median_topclean={agg['median_topclean_m']}m max={agg['max_m']}m std={agg['std_m']}m")
    print(f"mesh_longest(norm)={mesh_len:.4f} -> scale={scale:.4f} L_extent={L_est}m")
    print("->", out_dir / f"{object_id}.json")


if __name__ == "__main__":
    main()
