#!/usr/bin/env python3
"""汇总所有 take -> 哪个可观测信号能预测真实误差? 按物体划分 dev / held-out。

为什么按**物体**划分而不是随机划分帧:同一条 take 内的帧高度相关,随机划分会让
留出集里混进训练集同一条视频的帧,相关系数会虚高。判据要能换个没见过的物体还成立,
划分就必须切在物体上。

为什么盯"深度尺度"而不是总误差:laptop 实测深度分量 236mm、横向 38mm,
而所有现有判据都是图像内轮廓比对 —— 对深度结构性失明。深度是缺口所在。

★ 纪律:任何"改进"必须在 held-out 上复现才算数。dev 上的相关系数只用来选形式,不用来报成绩。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 只用我们自己产物就能算出的量(判据必须可上线);目标是它们能否预测真值误差
SIGNALS = ["obj_over_hand_depth", "contact_gap_mm", "contact_gap_min_mm",
           "hand_palm_cm", "hand_total_cm", "hand_depth_m", "obj_depth_m",
           "take_conf_pos_median", "take_conf_rot_median", "take_rot_factor",
           "take_mesh_scale_fitted", "take_median_explained", "take_median_occl"]
TARGETS = ["depth_ratio_med", "err_pos_med", "err_dep_med", "err_lat_med", "err_rot_med"]


def spearman(x, y):
    """秩相关:样本少 + 关系可能非线性时比 Pearson 稳。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, default=Path(__file__).resolve().parents[2] / "Output/arctic_eval/rows")
    ap.add_argument("--dev-objects", default="laptop,box,microwave,mixer,notebook,phone",
                    help="dev 集物体(逗号分隔);其余为 held-out")
    a = ap.parse_args()

    rows = [json.loads(p.read_text()) for p in sorted(a.rows.glob("*.json"))]
    if not rows:
        raise SystemExit(f"{a.rows} 下没有 take 记录")
    dev_objs = set(a.dev_objects.split(","))
    dev = [r for r in rows if r["object"] in dev_objs]
    hold = [r for r in rows if r["object"] not in dev_objs]

    print(f"共 {len(rows)} 条 take | dev {len(dev)} ({sorted({r['object'] for r in dev})})"
          f" | held-out {len(hold)} ({sorted({r['object'] for r in hold})})")

    print("\n=== 每条 take 概览 ===")
    hdr = f"{'物体':16}{'帧':>5}{'深度比':>8}{'位置mm':>8}{'深度mm':>8}{'横向mm':>8}{'旋转°':>7}" \
          f"{'confP':>7}{'confR':>7}{'物/手深度':>9}{'接触隙mm':>9}{'掌长cm':>8}"
    print(hdr)
    for r in sorted(rows, key=lambda r: r["object"]):
        def g(k, d="-"):
            v = r.get(k)
            return f"{v:.2f}" if isinstance(v, float) else (str(v) if v is not None else d)
        print(f"{r['object']:16}{r['n_cmp']:>5}{g('depth_ratio_med'):>8}{g('err_pos_med'):>8}"
              f"{g('err_dep_med'):>8}{g('err_lat_med'):>8}{g('err_rot_med'):>7}"
              f"{g('take_conf_pos_median'):>7}{g('take_conf_rot_median'):>7}"
              f"{g('obj_over_hand_depth'):>9}{g('contact_gap_mm'):>9}{g('hand_palm_cm'):>8}")

    print("\n=== dev 集: 信号 vs 目标 的秩相关(|rho|>=0.6 才值得继续) ===")
    print(f"{'信号':26}" + "".join(f"{t:>18}" for t in TARGETS))
    keep = []
    for s in SIGNALS:
        x = np.array([r.get(s, np.nan) if isinstance(r.get(s), (int, float)) else np.nan
                      for r in dev], float)
        cells, best = [], 0.0
        for t in TARGETS:
            y = np.array([r.get(t, np.nan) for r in dev], float)
            rho = spearman(x, y)
            cells.append(f"{rho:+.3f}" if np.isfinite(rho) else "  -  ")
            if np.isfinite(rho):
                best = max(best, abs(rho))
        print(f"{s:26}" + "".join(f"{c:>18}" for c in cells))
        if best >= 0.6:
            keep.append(s)
    print(f"\ndev 上够强的信号: {keep if keep else '(无 —— 老实报告: 没找到可用信号)'}")

    if keep and hold:
        print("\n=== held-out 复现(唯一算数的成绩) ===")
        print(f"{'信号':26}" + "".join(f"{t:>18}" for t in TARGETS))
        for s in keep:
            x = np.array([r.get(s, np.nan) if isinstance(r.get(s), (int, float)) else np.nan
                          for r in hold], float)
            cells = []
            for t in TARGETS:
                y = np.array([r.get(t, np.nan) for r in hold], float)
                rho = spearman(x, y)
                cells.append(f"{rho:+.3f}" if np.isfinite(rho) else "  -  ")
            print(f"{s:26}" + "".join(f"{c:>18}" for c in cells))
        print("  (held-out 样本少,符号一致 + 量级相近才算复现;数值本身不必相同)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
