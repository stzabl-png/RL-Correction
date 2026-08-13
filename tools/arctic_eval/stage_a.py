#!/usr/bin/env python3
"""阶段 A：把 confidence 的**每条分项判据**在锚后口径下重测一遍。

背景见 `docs/CONFIDENCE_REDESIGN.md` §0：现行摆放（`ref_builders/replay_grasp.py` 第③步
"物体听手"）会把交互开始帧的偏移归零，所以 RL 承受的是**锚后**误差，而现有判据是拿
**绝对**误差验证的。本脚本对同一批帧同时给出两个口径，差距即"评错了对象"的程度。

除了各判据，必须一起测两个**对照预测器**，否则会把平凡效应当成判据的功劳：

* `t_since_anchor` —— 距锚点的帧数。锚后误差是累积漂移，天然随时间涨。
  若它比所有判据都强，说明逐帧打分在这个口径下没有增量信息。
* `disp_from_anchor` —— 我们自己算的"物体离锚点多远"（**无需真值**）。
  若锚后误差 ≈ (幅度比−1) × 该位移，那么 take 内的误差几乎由**全局幅度**决定，
  逐帧分数无事可做 —— 这会直接改变 §2 的规划（① 让位给 ②）。

用法：
    python3 tools/arctic_eval/stage_a.py            # 全量表
    python3 tools/arctic_eval/stage_a.py --reversed # 只深挖 3 条方向反的 take
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anchored_conf import ARCTIC, META, RR, rho, streams  # noqa: E402

AUDIT = Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad/audit15.json")

# 逐帧判据 → (人类可读名, 期望符号)。期望符号 = 与"误差"的期望相关方向：
#   +1 表示"该量越大越糟"（如抖动幅度）, -1 表示"越大越好"（如 explained 贴合度）。
SIGNALS = {
    "explained":    ("轮廓解释度", -1),
    "d_cent_norm":  ("质心偏移(归一)", +1),
    "occl":         ("遮挡比例", +1),
    "tilt_deg":     ("倾斜角", +1),
    "above_mm":     ("离支撑面高度", +1),
    "step_mm":      ("逐帧位移跳变", +1),
    "step_deg":     ("逐帧转角跳变", +1),
    "mask_step_px": ("mask 位移(像素)", +1),
    "proj_step_px": ("投影位移(像素)", +1),
    "conf_pos":     ("★conf_pos(合成)", -1),
    "conf_rot":     ("conf_rot(合成)", -1),
    "conf":         ("conf(总)", -1),
}


def load_audit():
    doc = json.loads(AUDIT.read_text())
    out = {}
    for rec in doc["takes"]:
        tk = str(rec.get("take", ""))
        if "arctic15" not in tk or rec.get("object", "object_0") != "object_0":
            continue
        # ⚠ 尾部三级：arctic 与 arctic15 的 <subject>/<seq> 同名，取两级会串台
        out["/".join(Path(tk).parts[-3:])] = rec
    return out


def per_frame_signals(rec, n):
    """audit 的 per_frame → {名: (n,) 数组}，缺帧填 nan。"""
    cols = {k: np.full(n, np.nan) for k in SIGNALS}
    for r in rec.get("per_frame", []):
        i = int(r["frame"])
        if i < n:
            for k in SIGNALS:
                v = r.get(k)
                if v is not None and not isinstance(v, (list, dict)):
                    cols[k][i] = float(v)
    return cols


# ★ 定标集只含这些 subject。s01 是 B-2b 的**留出 subject**，任何定标都不得看到它 ——
# 否则留出验证失效。此前它是靠"缺 rows15/*.npz"被偶然排除的，那是隐患：一旦有人给
# s01 生成了 rows15，定标集会**静默**从 11 条涨到 15 条而没有任何报错。
DEV_SUBJECTS = {"s02", "s04", "s05", "s06", "s07", "s10"}


def collect(subjects=DEV_SUBJECTS):
    """→ [(obj, 各信号(锚后帧上), 锚后误差, 绝对误差, 距锚帧数, 离锚位移)]

    默认只返回定标集（见 DEV_SUBJECTS）。要拿全量请显式传 subjects=None。
    """
    audit = load_audit()
    out = []
    for take in sorted((RR / "Output/ReconstructOutput/arctic15").glob("*/*")):
        if subjects is not None and take.parent.name not in subjects:
            continue
        st = streams(take)
        if st is None:
            continue
        obj, ours, gt, _, idx, f = st
        rec = audit.get("/".join(take.resolve().parts[-3:]))
        if rec is None:
            continue
        a0 = idx[0]
        e_anc = np.linalg.norm((ours[idx] - ours[a0]) - (gt[idx] - gt[a0]), axis=1) * 1000
        e_abs = np.linalg.norm(ours[idx] - gt[idx], axis=1) * 1000
        t_since = np.arange(len(idx), dtype=float)
        disp = np.linalg.norm(ours[idx] - ours[a0], axis=1) * 1000     # 无需真值
        cols = per_frame_signals(rec, int(f.max()) + 1)
        sig = {k: v[f[idx]] for k, v in cols.items()}
        out.append((obj, sig, e_anc, e_abs, t_since, disp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reversed", action="store_true", help="只深挖方向反的 take")
    a = ap.parse_args()
    data = collect()
    if not data:
        print("(没有可比对的 take)")
        return 0

    if a.reversed:
        return deep_dive(data)

    # ---------- 1. 各判据在两个口径下的相关（合并全部帧） ----------
    print(f"\n{'='*92}\n【1】各分项判据 vs 误差（合并 {sum(len(d[2]) for d in data)} 帧，仅接触后）\n{'='*92}")
    print(f"{'判据':22}{'期望符号':>8}{'vs绝对误差':>11}{'vs锚后误差':>11}{'方向对(锚后)':>12}   判读")
    for k, (nm, sgn) in SIGNALS.items():
        X = np.concatenate([d[1][k] for d in data])
        A = np.concatenate([d[3] for d in data])
        B = np.concatenate([d[2] for d in data])
        ok = sum(1 for d in data if np.sign(rho(d[1][k], d[2])) == sgn and abs(rho(d[1][k], d[2])) > 0.1)
        ra, rb = rho(X, A), rho(X, B)
        verdict = ("存活" if np.sign(rb) == sgn and abs(rb) >= 0.15
                   else "反向 ⚠" if np.sign(rb) == -sgn and abs(rb) >= 0.15 else "失效")
        print(f"{nm:22}{sgn:>+8}{ra:>11.2f}{rb:>11.2f}{ok:>8}/{len(data)}   {verdict}")

    # ---------- 2. 对照预测器 ----------
    print(f"\n{'='*92}\n【2】对照预测器（不是判据，是平凡效应；判据必须打得过它们才有价值）\n{'='*92}")
    for nm, i in (("距锚点帧数 t_since", 4), ("离锚点位移 disp（无需真值）", 5)):
        X = np.concatenate([d[i] for d in data])
        B = np.concatenate([d[2] for d in data])
        per = [rho(d[i], d[2]) for d in data]
        print(f"  {nm:30} vs 锚后误差  合并 {rho(X,B):+.2f}   逐条中位 {np.median(per):+.2f}"
              f"   ({sum(r>0.1 for r in per)}/{len(per)} 条为正)")

    # ---------- 3. ② 是否吞掉了 ①：take 内误差能否被"位移×常数"解释 ----------
    print(f"\n{'='*92}\n【3】take 内：锚后误差 ≈ (幅度比−1) × 离锚位移？\n"
          f"    R² 高 ⇒ 误差主要由**全局幅度**决定，逐帧分数无事可做\n{'='*92}")
    print(f"{'物体':16}{'R²':>7}{'斜率k':>8}{'残差中位':>10}{'原误差中位':>11}{'逐帧还剩多少':>13}")
    r2s, red = [], []
    for obj, _, e_anc, _, _, disp in data:
        m = disp > 1e-6
        k = float(np.sum(disp[m] * e_anc[m]) / np.sum(disp[m] ** 2))   # 过原点最小二乘
        resid = e_anc - k * disp
        ss = 1 - np.sum(resid[m] ** 2) / max(np.sum((e_anc[m] - e_anc[m].mean()) ** 2), 1e-9)
        r2s.append(ss); red.append(np.median(np.abs(resid)) / max(np.median(e_anc), 1e-9))
        print(f"{obj:16}{ss:>7.2f}{k:>8.2f}{np.median(np.abs(resid)):>8.0f}mm"
              f"{np.median(e_anc):>9.0f}mm{np.median(np.abs(resid))/max(np.median(e_anc),1e-9):>12.0%}")
    print(f"\n  R² 中位 {np.median(r2s):.2f}   扣掉「位移×常数」后仍剩 {np.median(red):.0%} 的误差")

    # ---------- 4. 判据对**残差**还有没有价值 ----------
    # 位移效应是全局幅度(②)的表现, 且无需真值就能预测。把它扣掉之后剩下的才是逐帧品质(①)
    # 该管的部分。判据若连残差都预测不了, 说明 ① 在锚后口径下没有可测的抓手。
    print(f"\n{'='*92}\n【4】扣掉位移效应后，各判据 vs **残差**（这才是 ① 真正该管的部分）\n{'='*92}")
    resids = []
    for _, _, e_anc, _, _, disp in data:
        m = disp > 1e-6
        k = float(np.sum(disp[m] * e_anc[m]) / np.sum(disp[m] ** 2))
        resids.append(np.abs(e_anc - k * disp))
    print(f"{'判据':22}{'期望符号':>8}{'vs锚后误差':>11}{'vs残差':>9}{'方向对(残差)':>12}   判读")
    for k_, (nm, sgn) in SIGNALS.items():
        X = np.concatenate([d[1][k_] for d in data])
        B = np.concatenate([d[2] for d in data])
        Rr = np.concatenate(resids)
        per = [rho(d[1][k_], r) for d, r in zip(data, resids)]
        ok = sum(1 for v in per if np.sign(v) == sgn and abs(v) > 0.1)
        rr = rho(X, Rr)
        verdict = ("存活" if np.sign(rr) == sgn and abs(rr) >= 0.15
                   else "反向 ⚠" if np.sign(rr) == -sgn and abs(rr) >= 0.15 else "失效")
        print(f"{nm:22}{sgn:>+8}{rho(X,B):>11.2f}{rr:>9.2f}{ok:>8}/{len(data)}   {verdict}")
    return 0


def deep_dive(data) -> int:
    """3 条方向反的 take：高分帧是不是集中在幅度被放大最狠的段落。"""
    BAD = {"espressomachine", "microwave", "waffleiron"}
    print(f"\n{'='*92}\n【方向反的 take 深挖】高 conf 帧是否落在离锚点更远处（=幅度错被放大处）\n{'='*92}")
    print(f"{'物体':16}{'反向?':>6}{'rho(conf,锚后)':>15}{'rho(conf,离锚位移)':>18}"
          f"{'低分帧离锚':>11}{'高分帧离锚':>11}")
    for obj, sig, e_anc, _, _, disp in data:
        c = sig["conf_pos"]
        m = np.isfinite(c)
        if m.sum() < 20:
            continue
        c_, e_, d_ = c[m], e_anc[m], disp[m]
        lo = d_[c_ <= np.percentile(c_, 25)]; hi = d_[c_ >= np.percentile(c_, 75)]
        print(f"{obj:16}{'★' if obj in BAD else '':>6}{rho(c_,e_):>15.2f}{rho(c_,d_):>18.2f}"
              f"{np.median(lo):>9.0f}mm{np.median(hi):>9.0f}mm")
    print("\n  判读：若 ★ 三条的 rho(conf,离锚位移) 明显为正（高分帧离锚点更远），"
          "\n        则 conf 的「反向」是 ② 污染 ① 的表现，不是判据本身坏。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
