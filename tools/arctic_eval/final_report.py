#!/usr/bin/env python3
"""arctic15 全量结论：帧级（用户真正关心的）+ take 级（按物体划分 dev/held-out）。

两层必须分开，因为它们回答的是不同问题、样本量差两个数量级：

* **帧级**（约 4000 帧）：一条视频里能不能挑出位姿更准的帧 —— confidence 的实际用途。
  样本充足，结论可靠。
* **take 级**（11 条）：整条视频好不好、能不能跨视频比较。11 个样本只能看强效应，
  且必须按**物体**划分 dev/held-out —— 同一条视频内的帧高度相关，随机划分会让相关系数虚高。

误差口径一律是**零拟合的质心距**（两边网格顶点各按自己的位姿变换后取质心，那是同一个
物理点），并分深度/横向 —— 实测主误差在深度而所有判据都是图像内轮廓比对，只报总误差
会把这个事实盖住。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROWS = Path(__file__).resolve().parents[2] / "Output" / "arctic_eval" / "rows15"
DEV_OBJECTS = {"box", "laptop", "microwave", "mixer", "notebook", "phone"}


def rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return np.nan
    return float(np.corrcoef(np.argsort(np.argsort(x[m])),
                             np.argsort(np.argsort(y[m])))[0, 1])


def main() -> int:
    js = sorted(ROWS.glob("*.json"))
    rows = [json.loads(p.read_text()) for p in js]
    if not rows:
        print("(没有评测结果)")
        return 0

    # ---------- take 级概览 ----------
    print(f"\n{'='*104}\n【每条 take 概览】{len(rows)} 条\n{'='*104}")
    print(f"{'物体':16}{'比对帧':>6}{'有分帧':>7}{'形状中轴':>9}{'形状短轴':>9}"
          f"{'位置mm':>8}{'深度mm':>8}{'横向mm':>8}{'深度比':>7}{'confP':>7}{'confR':>7}")
    for r in sorted(rows, key=lambda r: r["object"]):
        f = lambda k, d=float("nan"): r.get(k, d)  # noqa: E731
        print(f"{r['object']:16}{r['n_cmp']:>6}{r.get('conf_frames',0):>7}"
              f"{f('shape_err_mid_pct'):>8.1f}%{f('shape_err_short_pct'):>8.0f}%"
              f"{r['err_pos_med']:>8.0f}{r['err_dep_med']:>8.0f}{r['err_lat_med']:>8.0f}"
              f"{r['depth_ratio_med']:>7.2f}"
              f"{str(f('take_conf_pos_median','-')):>7}{str(f('take_conf_rot_median','-')):>7}")

    # ---------- 帧级：核心问题 ----------
    print(f"\n{'='*104}\n【帧级】confidence 能不能在一条视频里挑出位姿更准的帧\n{'='*104}")
    print(f"{'物体':16}{'有分帧':>7}{'相关':>7}   {'低分25%':>9}{'高分25%':>9}{'倍数':>7}")
    allc, alle, per_take = [], [], []
    for p in sorted(ROWS.glob("*.npz")):
        d = np.load(p)
        nm = p.stem.split("__")[1].replace("_grab_01", "")
        c, e = d["conf_pos"], d["err_pos"]
        m = np.isfinite(c) & np.isfinite(e)
        if m.sum() < 20:
            print(f"{nm:16}{int(m.sum()):>7}   (有分的帧太少)")
            continue
        c, e = c[m], e[m]
        allc.append(c); alle.append(e)
        lo = e[c <= np.percentile(c, 25)]; hi = e[c >= np.percentile(c, 75)]
        r_ = rho(c, e)
        per_take.append(r_)
        print(f"{nm:16}{len(c):>7}{r_:>7.2f}   {np.median(lo):>8.0f}mm{np.median(hi):>8.0f}mm"
              f"{np.median(lo)/max(np.median(hi),1e-9):>6.1f}x")
    if allc:
        C, E = np.concatenate(allc), np.concatenate(alle)
        print(f"\n合计 {len(C)} 帧   相关 {rho(C,E):+.2f}   "
              f"逐 take 相关中位 {np.median(per_take):+.2f}（{sum(r<0 for r in per_take)}/{len(per_take)} 为负=方向正确）")
        print("  conf_pos 分箱 → 位置误差中位:")
        for lo_, hi_ in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]:
            k = (C >= lo_) & (C < hi_)
            if k.sum() > 5:
                print(f"    {lo_:3d}-{hi_:<3d} {int(k.sum()):>5}帧   {np.median(E[k]):>6.0f}mm")

    # ---------- take 级：dev / held-out ----------
    print(f"\n{'='*104}\n【take 级】按物体划分 dev/held-out（同一视频内帧高度相关，不能随机划分）\n{'='*104}")
    dev = [r for r in rows if r["object"] in DEV_OBJECTS]
    hold = [r for r in rows if r["object"] not in DEV_OBJECTS]
    print(f"dev {len(dev)}: {sorted(r['object'] for r in dev)}")
    print(f"held-out {len(hold)}: {sorted(r['object'] for r in hold)}")
    for name, grp in (("dev", dev), ("held-out", hold)):
        if len(grp) < 4:
            print(f"  {name}: 样本 {len(grp)} 条，太少，不算")
            continue
        cp = np.array([r.get("take_conf_pos_median", np.nan) for r in grp], float)
        print(f"  {name} (n={len(grp)}):", end="")
        for t, lbl in (("err_pos_med", "位置"), ("err_dep_med", "深度"),
                       ("depth_ratio_med", "深度比"), ("shape_err_short_pct", "形状短轴")):
            y = np.array([r.get(t, np.nan) for r in grp], float)
            print(f"  conf_pos vs {lbl} {rho(cp,y):+.2f}", end="")
        print()
    n = len(rows)
    print(f"\n  ⚠ n={n} 时，相关系数需 |rho|>{1.96/np.sqrt(max(n-3,1)):.2f} 才在 0.05 水平显著。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
