"""通用设计 G-B: 接触起始段置信档位下限 + 档位对照变换 (打乱 / 反向)。

档位数值约定 (tasks/Pour/17/A_Design/L3_Learning/progress.py): 2=绿(最紧) 1=黄 0=红(最松)。
"下限 = 黄" 指**自由度**不低于黄档 ⟹ 数值上 tier = min(tier, 1); 红档 (0) 不受影响。

G-B (2026-08-31 用户裁定, 台账 L5-35.1 / L5-36): 交互段 [0, r_lift+margin] 行档位上限黄,
r_lift = 参考物体首次抬升 ≥ rise_m (= 认证阈值 CERT_RISE 5mm) 的行。窗长按母带算, 不写死。
v3 母带实测: r_lift=15 ⟹ 19 行 (0~18, ≈1.3s)。末尾不设 (已是红档, 设黄是收紧, 另一假设)。

为什么通用: 抓握/认证瞬间的重建置信度高 ≠ 抓法物理可行; 认证时策略要做的辅助动作
(调腕贴垫 / 容忍微倾) 超出绿档预算 —— 这是跨任务的事实, 不是 pour 特有。

环境变量:
  POUR_TIER_FLOOR_START  auto(默认) | <行数> | 0(关)
  POUR_TIER_SHUFFLE      <seed>: 同一多重集标签按固定种子置换到各行 (主对照; 各档行数不变)
  POUR_TIER_REVERSE      1: 绿↔红对调 (次对照; 绿/红行数不等, 总量略变)
施加顺序: 拍平(POUR_CONF_FLAT, 调用方负责) → 打乱/反向 → 下限。
对照组同样享有下限 —— 比的是"档位携带的信息", 不是窗本身。
"""
from __future__ import annotations

import os

import numpy as np


def start_window_rows(obj_pos_rows, rise_m, margin=3):
    """参考物体轨迹 (N,3) [交互段] → (窗长行数, r_lift)。找不到抬升 → (0, -1) 即不设窗。"""
    z = np.asarray(obj_pos_rows, np.float64)[:, 2]
    idx = np.nonzero(z - z[0] >= float(rise_m))[0]
    if len(idx) == 0:
        return 0, -1
    return int(idx[0]) + int(margin) + 1, int(idx[0])


def spec():
    """(floor_mode, shuffle_seed, reverse) —— 横幅/指纹共用一处解析。"""
    f = (os.environ.get("POUR_TIER_FLOOR_START") or "auto").strip().lower()
    s = int(os.environ.get("POUR_TIER_SHUFFLE") or 0)
    r = os.environ.get("POUR_TIER_REVERSE") == "1"
    return f, s, r


def transform(tp, tr, obj_pos_rows, rise_m, tag="PB"):
    """tp/tr: {oi: (N,) 档位整数}; 返回 (tp, tr, tmix, info)。tmix = min(tp[1], tr[1]) (主档=瓶)。
    batch 版与标量版进度机都走这里, 保证两边一致。"""
    tp = {k: np.asarray(v, np.int64).copy() for k, v in tp.items()}
    tr = {k: np.asarray(v, np.int64).copy() for k, v in tr.items()}
    n = len(tp[1])
    f, s, rev = spec()
    info = {"tier_floor_start": 0, "tier_shuffle": int(s), "tier_reverse": bool(rev),
            "r_lift": -1}
    cnt0 = np.bincount(np.minimum(tp[1], tr[1]), minlength=3)
    if s:
        perm = np.random.RandomState(s).permutation(n)
        for d in (tp, tr):
            for k in d:
                d[k] = d[k][perm]
        cnt1 = np.bincount(np.minimum(tp[1], tr[1]), minlength=3)
        assert (cnt0 == cnt1).all(), f"打乱后档位行数变了: {cnt0} -> {cnt1}"
        print(f"[{tag}] ★POUR_TIER_SHUFFLE={s}: 档位标签按固定种子置换到各行 "
              f"(红/黄/绿 行数 {cnt0.tolist()} 不变)", flush=True)
    if rev:
        for d in (tp, tr):
            for k in d:
                d[k] = 2 - d[k]
        print(f"[{tag}] ★POUR_TIER_REVERSE=1: 绿↔红对调 "
              f"(红/黄/绿 行数 {cnt0.tolist()} -> {cnt0[::-1].tolist()})", flush=True)
    if f == "auto":
        rows, r_lift = start_window_rows(obj_pos_rows, rise_m)
        info["r_lift"] = r_lift
    elif f in ("0", "off"):
        rows = 0
    else:
        rows = int(f)
    rows = int(min(max(rows, 0), n))
    if rows:
        before = np.minimum(tp[1], tr[1])[:rows].copy()
        for d in (tp, tr):
            for k in d:
                d[k][:rows] = np.minimum(d[k][:rows], 1)
        after = np.minimum(tp[1], tr[1])[:rows]
        print(f"[{tag}] ★G-B 接触起始黄窗: 交互行 0~{rows - 1} 档位上限黄 "
              f"(r_lift={info['r_lift']}, 主档 绿→黄 {int((before != after).sum())} 行)",
              flush=True)
    else:
        print(f"[{tag}] G-B 接触起始黄窗: 关 (POUR_TIER_FLOOR_START={f})", flush=True)
    info["tier_floor_start"] = rows
    return tp, tr, np.minimum(tp[1], tr[1]), info
