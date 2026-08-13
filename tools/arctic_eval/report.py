#!/usr/bin/env python3
"""B-1 验收：新打分(anchored_sigma)对物体轨迹的实际效果 + 与 ARCTIC 真值的逐帧比对。

产出 PNG 到 --out，供报告页嵌入。四张图各回答一个问题：

1. `pred_vs_true.png`   预测的毫米数准不准（散点，理想=对角线）
2. `timeline.png`       一条视频里预测跟不跟得上真实误差的起伏（逐帧曲线）
3. `old_vs_new.png`     新打分比旧 conf_pos 强多少（同一批帧，三个指标）
4. `traj_xy.png`        我们的轨迹 vs 真值轨迹（俯视，锚点对齐后）

口径一律是**锚后误差**：RL 摆放会把交互开始帧的偏移归零，绝对误差里那部分进不了仿真。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from anchored_conf import ARCTIC, META, RR, umeyama
from stage_a import load_audit, per_frame_signals

# 55 条 = 7 subject × 11 物体。协议为 leave-one-subject-out(见 calib55.py);
# 本页的散点/曲线用全量, 表格标注 subject 便于按人查看。
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "figure.facecolor": "white", "axes.axisbelow": True})
for fam in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [fam]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def gather():
    """→ [dict(take, obj, subject, 真实锚后误差, 预测, 我们的轨迹, 真值轨迹, conf_pos)]"""
    audit = load_audit()
    rows = []
    for take in sorted((RR / "Output/ReconstructOutput/arctic15").glob("*/*")):
        sub, seq = take.parent.name, take.name
        mp_, ca, ts = (META / f"{sub}__{seq}.meta.json", take / "contact_auto.json",
                       take / "traj_sigma.npz")
        if not (mp_.is_file() and ca.is_file() and ts.is_file()):
            continue
        v2a = {int(r["video_frame"]): int(r["arctic_vidx"])
               for r in json.loads(mp_.read_text())["index"]}
        z = np.load(take / "world_fused.npz", allow_pickle=True)
        W = np.asarray(z["object_ob_in_world"], float)
        c2w = np.asarray(z["c2w"], float)
        obj = seq.split("_")[0]
        Vo = np.asarray(trimesh.load(take / "objects/object_0/object_mesh_scaled_final.obj",
                                     force="mesh").vertices, float).mean(0)
        Vg = np.asarray(trimesh.load(ARCTIC / "meta/object_vtemplates" / obj / "mesh.obj",
                                     force="mesh").vertices, float).mean(0) / 1000.0
        o = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.object.npy", allow_pickle=True)
        ego = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.egocam.dist.npy",
                      allow_pickle=True).item()
        T = len(o)
        w2e = np.tile(np.eye(4), (T, 1, 1))
        w2e[:, :3, :3] = np.asarray(ego["R_k_cam_np"], float)
        w2e[:, :3, 3] = np.asarray(ego["T_k_cam_np"], float).reshape(T, 3)
        P = np.tile(np.eye(4), (T, 1, 1))
        P[:, :3, :3] = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
        P[:, :3, 3] = o[:, 4:7] / 1000.0
        f = np.array([t for t in range(len(W)) if t in v2a and v2a[t] < T])
        g = np.array([v2a[t] for t in f])
        s, R, tv = umeyama(c2w[f, :3, 3], np.linalg.inv(w2e)[g, :3, 3])
        ours = s * (R @ (np.einsum("tij,j->ti", W[f, :3, :3], Vo) + W[f, :3, 3]).T).T + tv
        gt = np.einsum("tij,j->ti", P[g, :3, :3], Vg) + P[g, :3, 3]

        ann = json.loads(ca.read_text()).get("annotations", {})
        st = [sg[0] for sd in ("left", "right") for sg in (ann.get(sd) or [])]
        if not st:
            continue
        idx = np.where(f >= min(st))[0]
        if len(idx) < 20:
            continue
        a0 = idx[0]
        true = np.linalg.norm((ours[idx] - ours[a0]) - (gt[idx] - gt[a0]), axis=1) * 1000
        sig = np.load(ts)
        # v2 schema: sigma_obj_mm(物体自身 σ) / sigma_fused_mm(融合后 σ)
        key = ("object_0__sigma_obj_mm" if "object_0__sigma_obj_mm" in sig.files
               else "object_0__pred_err_mm")
        pred = np.asarray(sig[key], float)[f[idx]]
        fus = (np.asarray(sig["object_0__ref_disp_fused_m"], float)     # 米, 与 ours/gt 同单位
               if "object_0__ref_disp_fused_m" in sig.files else None)
        rec = audit.get("/".join(take.resolve().parts[-3:]))
        cp = (per_frame_signals(rec, int(f.max()) + 1)["conf_pos"][f[idx]]
              if rec else np.full(len(idx), np.nan))
        # 融合参考(工具产物, 我们世界系) -> 施加 s·R 才能与真值比
        fz = None
        if fus is not None and len(fus) > f[idx].max():
            fz = (s * (R @ (fus[f[idx]] - fus[f[idx][0]]).T).T)
        # ⚠ 必须取**握住物体的那只手**(最长接触段的所属手), 不能取"第一只数据完整的手" ——
        # 另一只手在场但没碰物体时会被误选, 腕参考随之失真(实测 p90 385 vs 254mm)。
        seg_all = [(0 if sd == "left" else 1, sg)
                   for sd in ("left", "right") for sg in (ann.get(sd) or [])]
        hand_i, (s0, s1) = max(seg_all, key=lambda x: x[1][1] - x[1][0])
        hz = None
        if "hand_trans" in z.files:
            ht = np.asarray(z["hand_trans"], float)
            nh = ht.shape[1]
            ix = (np.round(np.linspace(0, nh - 1, len(W))).astype(int)
                  if nh != len(W) else np.arange(len(W)))
            hh = ht[hand_i, ix][f[idx]]
            if np.isfinite(hh).all():
                hz = (s * (R @ (hh - hh[0]).T).T)
        # 最长接触段掩码: 物体没被握住时腕会独立移动, 拿它当物体参考本就不成立,
        # 融合对比必须限制在握住的时段, 否则是在惩罚一个没人主张的用法。
        held = (f[idx] >= s0) & (f[idx] <= s1)
        rows.append(dict(obj=obj, sub=sub, frames=f[idx], true=true, pred=pred, conf=cp,
                         ours=ours[idx] - ours[a0], gt=gt[idx] - gt[a0],
                         fused=fz, hand=hz, held=held, heldout=False))
    return rows


def fig_scatter(rows, out):
    """左: 预测 vs 真实(全 55 条)。右: 逐 subject 的预测误差(留一交叉验证的可视化)。"""
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.4),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    subs = sorted({r["sub"] for r in rows})
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(subs):
        for r in [x for x in rows if x["sub"] == s]:
            ax[0].scatter(r["pred"], r["true"], s=2.5, alpha=.13, color=cmap(i % 10))
    P = np.concatenate([r["pred"] for r in rows]); Tr = np.concatenate([r["true"] for r in rows])
    m = np.isfinite(P) & np.isfinite(Tr)
    lim = float(np.nanpercentile(np.r_[P[m], Tr[m]], 98))
    ax[0].plot([0, lim], [0, lim], "k--", lw=1.2, label="完美预测")
    b = np.linspace(0, lim, 10); c, v = [], []
    for lo, hi in zip(b[:-1], b[1:]):
        k = m & (P >= lo) & (P < hi)
        if k.sum() > 40:
            c.append(.5 * (lo + hi)); v.append(np.median(Tr[k]))
    ax[0].plot(c, v, "o-", color="#c2502e", lw=2, ms=5, label="分箱中位")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_xlabel("预测的误差 σ (mm)"); ax[0].set_ylabel("真实锚后误差 (mm)")
    ax[0].set_title(f"55 条 / {int(m.sum())} 帧   |预测−真实| 中位 "
                    f"{np.median(np.abs(P[m]-Tr[m])):.0f}mm", fontsize=9.5, loc="left")
    ax[0].legend(fontsize=8, loc="upper left")
    # 右: 逐 subject
    mae, base = [], []
    for s in subs:
        rs = [r for r in rows if r["sub"] == s]
        p_ = np.concatenate([r["pred"] for r in rs]); t_ = np.concatenate([r["true"] for r in rs])
        k = np.isfinite(p_) & np.isfinite(t_)
        mae.append(np.median(np.abs(p_[k] - t_[k])))
        base.append(np.median(np.abs(np.median(Tr[m]) - t_[k])))
    y = np.arange(len(subs))
    ax[1].barh(y + .2, base, .38, color="#c7cfd0", label="常数基线")
    ax[1].barh(y - .2, mae, .38, color="#c2502e", label="σ 预测")
    ax[1].set_yticks(y); ax[1].set_yticklabels(subs, fontsize=9)
    ax[1].set_xlabel("|预测 − 真实| 中位 (mm)")
    ax[1].set_title("逐 subject（留一交叉验证）", fontsize=9.5, loc="left")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pred_vs_true.png", dpi=135); plt.close(fig)


def fig_timeline(rows, out):
    pick = [r for r in rows if (r["obj"], r["sub"]) in
            {("laptop", "s05"), ("microwave", "s02"), ("scissors", "s01"), ("mixer", "s06")}]
    pick = pick or rows[:4]
    fig, ax = plt.subplots(len(pick), 1, figsize=(9.4, 2.1 * len(pick)), sharex=False)
    ax = np.atleast_1d(ax)
    for a, r in zip(ax, pick):
        x = np.arange(len(r["true"]))
        a.plot(x, r["true"], color="#2a9d8f", lw=1.4, label="真实锚后误差")
        a.plot(x, r["pred"], color="#e76f51", lw=1.6, label="预测 σ（0.876×离锚位移+1.2）")
        cf = r["conf"]
        if np.isfinite(cf).any():                      # 旧分数缩放到同轴，只为看形状
            a2 = a.twinx(); a2.plot(x, cf, color="#6c757d", lw=.9, alpha=.55)
            a2.set_ylim(0, 105); a2.set_ylabel("conf_pos", fontsize=7, color="#6c757d")
            a2.grid(False); a2.tick_params(labelsize=7, colors="#6c757d")
        a.set_ylabel("mm"); a.legend(fontsize=7.5, loc="upper left", ncol=2)
        a.set_title(f"{r['sub']}/{r['obj']}", fontsize=9, loc="left")
    ax[-1].set_xlabel("接触起始帧之后的帧序")
    fig.tight_layout(); fig.savefig(out / "timeline.png", dpi=135); plt.close(fig)


def fig_old_vs_new(rows, out):
    """三个指标各一栏：排序能力 / 定量能力 / 分层挑帧（层内 disp 相近，去掉退化优势）。"""
    def rho(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 20:
            return np.nan
        return float(np.corrcoef(np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m])))[0, 1])
    names = [f"{r['sub']}/{r['obj']}" for r in rows]
    rn = [rho(r["pred"], r["true"]) for r in rows]
    ro = [-rho(r["conf"], r["true"]) for r in rows]      # conf 越高越好，取负使同向
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6),
                           gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(rows))
    ax[0].barh(y - .2, ro, .38, color="#adb5bd", label="旧 conf_pos")
    ax[0].barh(y + .2, rn, .38, color="#e76f51", label="新 预测")
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=7.5)
    ax[0].axvline(0, color="k", lw=.8)
    ax[0].set_xlabel("与真实锚后误差的秩相关（越高越好）")
    ax[0].set_title("排序能力：能不能分出哪些帧更准", fontsize=9.5, loc="left")
    ax[0].legend(fontsize=8)
    # 定量：绝对预测误差
    P = np.concatenate([r["pred"] for r in rows]); Tr = np.concatenate([r["true"] for r in rows])
    C = np.concatenate([r["conf"] for r in rows])
    m = np.isfinite(P) & np.isfinite(Tr)
    mae_new = np.median(np.abs(P[m] - Tr[m]))
    const = np.median(Tr[m])
    mae_const = np.median(np.abs(const - Tr[m]))
    mc = np.isfinite(C) & np.isfinite(Tr)
    q = np.unique(np.quantile(C[mc], np.linspace(0, 1, 11)))
    cx, cy = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        k = mc & (C >= lo) & (C <= hi)
        if k.sum() > 20:
            cx.append(.5 * (lo + hi)); cy.append(np.median(Tr[k]))
    mae_old = (np.median(np.abs(np.interp(C[mc], cx, cy) - Tr[mc])) if len(cx) > 1 else np.nan)
    bars = [("旧 conf_pos\n(已单调校准)", mae_old, "#adb5bd"),
            ("常数基线", mae_const, "#ced4da"), ("新 预测", mae_new, "#e76f51")]
    ax[1].bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    for i, b in enumerate(bars):
        ax[1].text(i, b[1] + 2, f"{b[1]:.0f}mm", ha="center", fontsize=9)
    ax[1].set_ylabel("|预测 − 真实| 中位 (mm)")
    ax[1].set_title("定量能力：能不能报出毫米数", fontsize=9.5, loc="left")
    ax[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(out / "old_vs_new.png", dpi=135); plt.close(fig)


def fig_traj(rows, out):
    pick = [r for r in rows if (r["obj"], r["sub"]) in
            {("box", "s05"), ("laptop", "s05"), ("scissors", "s01"), ("phone", "s10")}] or rows[:4]
    fig, ax = plt.subplots(1, len(pick), figsize=(3.1 * len(pick), 3.3))
    ax = np.atleast_1d(ax)
    for a, r in zip(ax, pick):
        o, g = r["ours"] * 1000, r["gt"] * 1000
        a.plot(g[:, 0], g[:, 1], color="#2a9d8f", lw=1.8, label="真值")
        a.plot(o[:, 0], o[:, 1], color="#e76f51", lw=1.5, label="我们的重建")
        a.scatter([0], [0], c="k", s=30, zorder=5, marker="x", label="锚点(接触起始)")
        a.set_aspect("equal"); a.set_title(f"{r['sub']}/{r['obj']}", fontsize=9)
        a.set_xlabel("X (mm)"); a.set_ylabel("Y (mm)")
        a.tick_params(labelsize=7)
    ax[0].legend(fontsize=7.5)
    fig.suptitle("锚点对齐后的物体轨迹（俯视）—— 位置已被手锚住，剩下的就是形状与幅度",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(out / "traj_xy.png", dpi=135); plt.close(fig)


def fig_fusion(rows, out):
    """三种物体参考来源 vs 真值。这是工具真正交付给 RL 的东西的验收。"""
    A, B, C = [], [], []
    for r in rows:
        if r["fused"] is None or r["hand"] is None or r["held"].sum() < 30:
            continue
        h = r["held"]
        g = (r["gt"][h] - r["gt"][h][0]) * 1000          # 各自以接触段首为锚
        A.append(np.linalg.norm((r["ours"][h] - r["ours"][h][0]) * 1000 - g, axis=1))
        B.append(np.linalg.norm((r["hand"][h] - r["hand"][h][0]) * 1000 - g, axis=1))
        C.append(np.linalg.norm((r["fused"][h] - r["fused"][h][0]) * 1000 - g, axis=1))
    A, B, C = (np.concatenate(x) for x in (A, B, C))
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2),
                           gridspec_kw={"width_ratios": [1, 1.15]})
    q = np.linspace(0, 95, 60)
    for nm, v, col, lw in (("只用重建物体", A, "#8b9698", 1.4),
                           ("只用腕（现行做法）", B, "#2a7f74", 1.6),
                           ("σ 融合（本工具）", C, "#c2502e", 2.2)):
        ax[0].plot(q, np.percentile(v, q), color=col, lw=lw, label=nm)
    ax[0].set_xlabel("分位数 (%)"); ax[0].set_ylabel("与真值的偏差 (mm)")
    ax[0].set_title("物体参考轨迹的误差分布（越低越好）", fontsize=9.5, loc="left")
    ax[0].legend(fontsize=8.5, loc="upper left")
    stats = [("中位", 50), ("p75", 75), ("p90", 90)]
    x = np.arange(len(stats)); w = .26
    for i, (nm, v, col) in enumerate((("只用物体", A, "#8b9698"),
                                      ("只用腕", B, "#2a7f74"),
                                      ("σ 融合", C, "#c2502e"))):
        vals = [np.percentile(v, s) for _, s in stats]
        ax[1].bar(x + (i - 1) * w, vals, w, color=col, label=nm)
        for j, val in enumerate(vals):
            ax[1].text(x[j] + (i - 1) * w, val + 4, f"{val:.0f}", ha="center", fontsize=7.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels([s[0] for s in stats])
    ax[1].set_ylabel("与真值的偏差 (mm)")
    ax[1].set_title(f"{len(A)} 帧汇总", fontsize=9.5, loc="left")
    ax[1].legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "fusion.png", dpi=135); plt.close(fig)
    return dict(obj=float(np.median(A)), hand=float(np.median(B)), fused=float(np.median(C)),
                obj90=float(np.percentile(A, 90)), hand90=float(np.percentile(B, 90)),
                fused90=float(np.percentile(C, 90)), n=int(len(A)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad/report"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = gather()
    print(f"take: {len(rows)} 条（留出 s01 {sum(r['heldout'] for r in rows)} 条）")
    fig_scatter(rows, a.out); fig_timeline(rows, a.out)
    fig_old_vs_new(rows, a.out); fig_traj(rows, a.out)
    fus = fig_fusion(rows, a.out)
    print(f"\n融合验收 {fus['n']} 帧: 只用物体 {fus['obj']:.0f} / 只用腕 {fus['hand']:.0f} / "
          f"融合 {fus['fused']:.0f} mm (p90 {fus['obj90']:.0f}/{fus['hand90']:.0f}/{fus['fused90']:.0f})")

    # 逐条数字表（报告页用）
    tbl = []
    for r in rows:
        m = np.isfinite(r["pred"]) & np.isfinite(r["true"])
        tbl.append(dict(take=f"{r['sub']}/{r['obj']}", sub=r["sub"],
                        n=int(m.sum()),
                        true_med=round(float(np.median(r["true"][m])), 0),
                        mae=round(float(np.median(np.abs(r["pred"][m] - r["true"][m]))), 0),
                        rho_new=round(float(np.corrcoef(
                            np.argsort(np.argsort(r["pred"][m])),
                            np.argsort(np.argsort(r["true"][m])))[0, 1]), 2)))
    (a.out / "table.json").write_text(json.dumps(tbl, ensure_ascii=False, indent=1))
    for t in tbl[:6]:
        print(f"  {t['take']:24}真实 {t['true_med']:>5.0f}mm  预测误差 {t['mae']:>4.0f}mm  "
              f"rho {t['rho_new']:+.2f}")
    print(f"  ... 共 {len(tbl)} 条")
    print(f"\n图已写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
