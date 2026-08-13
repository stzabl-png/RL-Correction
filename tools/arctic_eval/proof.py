#!/usr/bin/env python3
"""B-2a：新设计到底比旧 confidence 准多少 —— 留一物体交叉验证的正面对拍。

**任务定义**（必须先钉死，否则"更准"没有意义）：
    给定一帧，预测它的**锚后位置误差**（= RL 实际承受的量，见 docs/CONFIDENCE_REDESIGN.md §0）。

**协议**：留一物体（leave-one-object-out）。每次把一条 take 完全拿出去，所有参数
（斜率 k、conf→mm 校准表、残差修正系数）只在**其余 10 条**上定标，再去预测被留出的那条。
位移法是在这 11 条上发现的，不做留出就是自己考自己。

**公平性**：旧 conf_pos 是 0–100 的分数，天生只能排序，直接比毫米数对它不公平。
所以先用其余 10 条给它拟一张**单调校准表**（分位分箱 → 该箱误差中位数 → 线性插值），
让它也能输出毫米预测。这是给旧方法的最好待遇。

**指标**（三个都要，因为它们回答不同问题）：
* `rho`      —— 排序能力：能不能分出哪些帧更准
* `MAE(mm)`  —— 定量能力：预测的毫米数离真值多远
* 保留曲线   —— 决策能力：按预测挑掉最差的帧后，剩下的帧实际有多准
                （附随机基线与 oracle 上限，否则看不出提升是真是假）
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from stage_a import collect

KEEP = (0.25, 0.50, 0.75)


def calib_curve(x, y, nbin=10):
    """单调校准：按 x 分位分箱 → 每箱 y 中位数 → 可插值的 (箱心, 值)。给旧分数用。"""
    q = np.unique(np.quantile(x, np.linspace(0, 1, nbin + 1)))
    if len(q) < 3:
        return np.array([x.min(), x.max()]), np.array([np.median(y)] * 2)
    cx, cy = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (x >= lo) & (x <= hi)
        if m.sum() >= 5:
            cx.append(0.5 * (lo + hi)); cy.append(np.median(y[m]))
    if len(cx) < 2:
        return np.array([x.min(), x.max()]), np.array([np.median(y)] * 2)
    return np.array(cx), np.array(cy)


def fit_models(train):
    """train = [(sig, e_anc, disp), ...] 的其余 10 条 → 各模型的参数。"""
    E = np.concatenate([t[1] for t in train])
    D = np.concatenate([t[2] for t in train])
    C = np.concatenate([t[0]["conf_pos"] for t in train])
    S = np.concatenate([t[0]["step_mm"] for t in train])

    # ① 旧法：conf_pos → mm 的单调校准（给旧方法的最好待遇）
    m = np.isfinite(C) & np.isfinite(E)
    cx, cy = calib_curve(C[m], E[m])

    # ② 位移法：逐 take 拟斜率再取中位（比池化更抗单条 take 主导）
    ks = []
    for _, e, d in train:
        ok = d > 1e-6
        if ok.sum() > 20:
            ks.append(float(np.sum(d[ok] * e[ok]) / np.sum(d[ok] ** 2)))
    k = float(np.median(ks))

    # ③ 位移法 + 残差修正：resid = e - k*disp 用存活判据 step_mm 线性拟合
    ok = np.isfinite(S) & np.isfinite(E) & np.isfinite(D)
    A = np.stack([np.ones(ok.sum()), S[ok]], 1)
    b = np.linalg.lstsq(A, E[ok] - k * D[ok], rcond=None)[0]

    return dict(cx=cx, cy=cy, k=k, b=b, const=float(np.median(E)))


def predict(p, sig, disp):
    c, s = sig["conf_pos"], sig["step_mm"]
    old = np.interp(np.nan_to_num(c, nan=np.median(p["cx"])), p["cx"], p["cy"])
    new = p["k"] * disp
    both = new + p["b"][0] + p["b"][1] * np.nan_to_num(s, nan=0.0)
    return {"旧 conf_pos(已校准)": old, "新·位移法": new,
            "新·位移法+残差判据": np.maximum(both, 0.0),
            "基线·常数": np.full(len(disp), p["const"])}


def main() -> int:
    data = collect()
    takes = [(o, s, e, d) for o, s, e, _, _, d in data]
    names = list(predict(fit_models([(t[1], t[2], t[3]) for t in takes[1:]]),
                         takes[0][1], takes[0][3]))

    rows = {n: {"rho": [], "mae": [], "keep": {r: [] for r in KEEP},
                "cov": {r: [] for r in KEEP}} for n in names}
    rand = {r: [] for r in KEEP}
    orac = {r: [] for r in KEEP}
    base_all = []

    print(f"\n{'='*98}\n【留一物体交叉验证】每条 take 的参数只来自其余 10 条\n{'='*98}")
    print(f"{'物体':16}{'实际误差':>9}", end="")
    for n in names:
        print(f"{n[:12]:>14}", end="")
    print()

    for i, (obj, sig, e, disp) in enumerate(takes):
        train = [(t[1], t[2], t[3]) for j, t in enumerate(takes) if j != i]
        pr = predict(fit_models(train), sig, disp)
        base_all.append(np.median(e))
        print(f"{obj:16}{np.median(e):>7.0f}mm", end="")
        for n in names:
            p = pr[n]
            rows[n]["rho"].append(spearmanr(p, e).statistic)
            rows[n]["mae"].append(float(np.median(np.abs(p - e))))
            for r in KEEP:
                keep = np.argsort(p)[:max(int(len(p) * r), 5)]
                rows[n]["keep"][r].append(float(np.median(e[keep])))
                # ⚠ 退化检查: 位移法挑出的"低误差帧"可能只是"物体没动的帧" ——
                # 误差小但对 RL 没信息。覆盖率 = 保留帧离锚位移中位 / 全体中位。
                rows[n]["cov"][r].append(float(np.median(disp[keep]) / max(np.median(disp), 1e-9)))
            print(f"{np.median(np.abs(p-e)):>12.0f}mm", end="")
        print()
        rng = np.random.default_rng(i)
        for r in KEEP:
            nk = max(int(len(e) * r), 5)
            rand[r].append(float(np.median(e[rng.permutation(len(e))[:nk]])))
            orac[r].append(float(np.median(e[np.argsort(e)[:nk]])))

    print(f"\n{'='*98}\n【汇总】11 条留一结果\n{'='*98}")
    print(f"{'方法':24}{'排序 rho':>10}{'定量 MAE':>11}   保留 25%/50%/75% 后的实际误差      ★运动覆盖率")
    for n in names:
        r = rows[n]
        ks = "  ".join(f"{np.median(r['keep'][x]):>5.0f}mm" for x in KEEP)
        cv = "  ".join(f"{np.median(r['cov'][x]):>4.0%}" for x in KEEP)
        print(f"{n:24}{np.median(r['rho']):>+10.2f}{np.median(r['mae']):>9.0f}mm   {ks}   {cv}")
    print(f"{'（随机挑帧）':24}{0.0:>+10.2f}{'—':>11}   "
          + "  ".join(f"{np.median(rand[x]):>5.0f}mm" for x in KEEP))
    print(f"{'（oracle 上限）':24}{1.0:>+10.2f}{0:>9.0f}mm   "
          + "  ".join(f"{np.median(orac[x]):>5.0f}mm" for x in KEEP))
    print(f"\n  全部帧不做任何筛选时的实际误差中位: {np.median(base_all):.0f}mm")
    print("  判读：保留曲线要和「随机挑帧」比才算数 —— 随机挑也会因为分位而略低于全体中位。")
    print("  ★运动覆盖率 = 保留帧的离锚位移中位 / 全体中位。**远低于 100% = 退化**：")
    print("    挑出的只是「物体还没动」的帧，误差当然小，但轨迹信息也一起被挑没了。")
    stratified(takes)
    return 0


def stratified(takes) -> None:
    """公平的**挑帧**对拍：在同等离锚位移的帧之间比，退化优势被消掉。

    位移法在分层内几乎是常数（同一层里 disp 相近），所以它在这里理应退化成随机 ——
    这正是要验证的：**定量预测**和**挑帧**是两件事，位移法只赢前者。
    挑帧该由残差判据（A-1 存活的「跳变」家族）来做。
    """
    # (名字, 取哪个信号, 方向)  方向 -1 = 值越小越好
    CAND = [("旧 conf_pos", "conf_pos", +1), ("conf_rot", "conf_rot", +1),
            ("★step_mm(残差判据)", "step_mm", -1), ("proj_step_px", "proj_step_px", -1)]
    print(f"\n{'='*98}\n【分层挑帧】按离锚位移分 4 层，层内各留最好的 50% —— 运动覆盖率被强制拉平\n{'='*98}")
    print(f"{'挑帧依据':24}{'保留后实际误差':>14}{'vs 随机':>10}{'运动覆盖率':>11}")
    res = {n: [] for n, _, _ in CAND}
    res["（随机挑帧）"] = []
    cov = {n: [] for n in res}
    for i, (obj, sig, e, disp) in enumerate(takes):
        edges = np.quantile(disp, [0, .25, .5, .75, 1.0])
        rng = np.random.default_rng(1000 + i)
        for nm, key, sgn in CAND + [("（随机挑帧）", None, 0)]:
            sel = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = np.where((disp >= lo) & (disp <= hi))[0]
                if len(b) < 4:
                    continue
                if key is None:
                    order = rng.permutation(len(b))
                else:
                    v = np.nan_to_num(sig[key][b], nan=np.inf if sgn < 0 else -np.inf)
                    order = np.argsort(sgn * -v)      # sgn=-1 → 小的在前
                sel.extend(b[order[:max(len(b) // 2, 2)]])
            if sel:
                res[nm].append(float(np.median(e[sel])))
                cov[nm].append(float(np.median(disp[sel]) / max(np.median(disp), 1e-9)))
    base = np.median(res["（随机挑帧）"])
    for nm in [c[0] for c in CAND] + ["（随机挑帧）"]:
        if not res[nm]:
            continue
        m = np.median(res[nm])
        print(f"{nm:24}{m:>12.0f}mm{(m-base)/base:>+9.0%}{np.median(cov[nm]):>10.0%}")
    print("\n  判读：分层后位移法失去作用（层内 disp 近似常数），挑帧只能靠残差判据。")


if __name__ == "__main__":
    raise SystemExit(main())
