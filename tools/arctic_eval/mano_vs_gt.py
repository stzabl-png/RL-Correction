#!/usr/bin/env python
"""C1 验收: MANO 全手探头 vs ARCTIC GT, 与旧指垫探头同台对拍。

从 contact_mano_*.npz 的距离时间线离线重投票, 扫 τ×vote 网格:
  - dev 物体(4)上选最优 (τ, vote), held-out(7) 只报 before/after —— 反作弊纪律
  - 口径与 contact_vs_gt 一致: vs GT 整段抓握模式(5mm)的指级 P/R + 掌命中 + 指数误差
  - 时间线: 我们的 grasp_core(重投票后) vs GT 接触段 IoU
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GT = Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic_gt_contacts")
OURS = Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic15_ours")
META = Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic15_local")
F = ("thumb", "index", "middle", "ring", "pinky", "palm")
DEV = {"box", "laptop", "ketchup", "mixer"}
TAKES = ["s05__laptop_grab_01", "s07__ketchup_grab_01", "s05__box_grab_01",
         "s04__capsulemachine_grab_01", "s05__espressomachine_grab_01",
         "s05__microwave_grab_01", "s05__mixer_grab_01", "s06__notebook_grab_01",
         "s10__phone_grab_01", "s02__scissors_grab_01", "s05__waffleiron_grab_01"]


def longest_run(flags, max_gap=5):
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return None
    best = cur = [idx[0], idx[0]]
    for i in idx[1:]:
        if i - cur[1] <= max_gap + 1:
            cur[1] = i
        else:
            cur = [i, i]
        if cur[1] - cur[0] > best[1] - best[0]:
            best = list(cur)
    return tuple(best)


def revote(npz, tau, vote):
    """从距离时间线重投票 → (指集合, palm, 核心段视频帧区间) 或 None。"""
    hit = npz["dists_m"] < tau
    core = longest_run(hit.sum(axis=1) >= 2)
    if core is None:
        return None
    c0, c1 = core
    frac = hit[c0:c1 + 1].mean(axis=0)
    return ({F[i] for i in range(5) if frac[i] >= vote}, bool(frac[5] >= vote),
            (int(npz["frames"][c0]), int(npz["frames"][c1])))


def eval_takes(takes, tau, vote):
    """返回 (指级tp/fp/fn, palm对/总, 指数误差列表, IoU列表)."""
    s = {"tp": 0, "fp": 0, "fn": 0, "palm_ok": 0, "palm_n": 0, "nerr": [], "iou": []}
    for take in takes:
        subj, seq = take.split("__", 1)
        gtz = np.load(GT / take / "gt_contact.npz", allow_pickle=True)
        summ = json.loads((GT / take / "gt_summary.json").read_text())
        index = json.loads((META / f"{take}.meta.json").read_text())["index"]
        v_of = {r["video_frame"]: r["arctic_vidx"] for r in index}
        for si, side in enumerate(("left", "right")):
            gm = summ["hands"][side]["tau_5mm"]["grasp_mode"]
            npz_p = OURS / subj / seq / "contact" / f"contact_mano_object_0_{side}.npz"
            if not npz_p.is_file():
                if gm:                                   # GT 有抓、我们无提取 → 全漏
                    s["fn"] += len(gm["fingers"]); s["nerr"].append(gm["n_fingers_median"])
                continue
            r = revote(np.load(npz_p), tau, vote)
            if r is None:
                if gm:
                    s["fn"] += len(gm["fingers"]); s["nerr"].append(gm["n_fingers_median"])
                continue
            ours, palm, core = r
            if not gm:                                   # GT 整段无抓握 → 我们报的全是误报
                s["fp"] += len(ours)
                continue
            gt_set = set(gm["fingers"])
            s["tp"] += len(ours & gt_set); s["fp"] += len(ours - gt_set); s["fn"] += len(gt_set - ours)
            s["nerr"].append(abs(len(ours) - gm["n_fingers_median"]))
            s["palm_n"] += 1
            s["palm_ok"] += (palm == (gm["palm_contact_frac"] >= 0.5))
            gt_c = set(np.where(gtz["contact_5mm"][:, si, :5].any(1))[0].tolist())
            our_c = {v_of[f] for f in range(core[0], core[1] + 1) if f in v_of}
            u = our_c | gt_c
            s["iou"].append(len(our_c & gt_c) / len(u) if u else 1.0)
    return s


def fmt(s):
    P = s["tp"] / max(1, s["tp"] + s["fp"]); R = s["tp"] / max(1, s["tp"] + s["fn"])
    F1 = 2 * P * R / max(1e-9, P + R)
    return (f"P {P:.2f} R {R:.2f} F1 {F1:.2f} | palm对 {s['palm_ok']}/{s['palm_n']} "
            f"| n误差中位 {np.median(s['nerr']):.0f} | IoU中位 {np.median(s['iou']):.2f}"
            if s["iou"] else "无数据")


def main():
    import sys
    takes = TAKES
    if len(sys.argv) > 1:                      # 可选: 传 take 清单文件(每行 s01/box_grab_01 或 s01__box_grab_01)
        takes = [l.strip().replace("/", "__") for l in open(sys.argv[1]) if l.strip()]
        takes = [t for t in takes
                 if (GT / t / "gt_contact.npz").is_file()]
    dev = [t for t in takes if t.split("__")[1].split("_grab")[0] in DEV]
    held = [t for t in takes if t not in dev]
    print(f"dev {len(dev)} 条 {sorted(DEV)} / held-out {len(held)} 条\n")
    print("── dev 网格标定 (F1):")
    best, best_f1 = None, -1
    for tau in (0.015, 0.025, 0.035, 0.045):
        row = []
        for vote in (0.3, 0.4, 0.5):
            s = eval_takes(dev, tau, vote)
            P = s["tp"] / max(1, s["tp"] + s["fp"]); R = s["tp"] / max(1, s["tp"] + s["fn"])
            f1 = 2 * P * R / max(1e-9, P + R)
            row.append(f"v{vote}:{f1:.2f}")
            if f1 > best_f1:
                best, best_f1 = (tau, vote), f1
        print(f"  τ={tau*1000:.0f}mm  " + "  ".join(row))
    tau, vote = best
    print(f"\n★ dev 最优: τ={tau*1000:.0f}mm vote={vote} (F1 {best_f1:.2f})")
    print(f"\n── held-out 终评 (τ={tau*1000:.0f}mm, vote={vote}):")
    print("  新探头(MANO全手):", fmt(eval_takes(held, tau, vote)))
    print("  dev(参考):       ", fmt(eval_takes(dev, tau, vote)))
    print("\n  旧探头基线(全11条, Phase B): P 0.94 R 0.46 | 掌不可测 | n误差中位 3 | IoU 0.45")


if __name__ == "__main__":
    main()
