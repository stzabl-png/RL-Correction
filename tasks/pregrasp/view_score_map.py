"""把接触分数图画成物体表面的热力图 (V2AP vis_3panel 风格: 密集点云 + jet + 无坐标轴)。

    python3 -m tasks.pregrasp.view_score_map \\
        --npz logs/G3_8_5/<ts>/score_map.npz --clip Grasp3 \\
        --prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz

三列 (与 V2AP 的 Human Prior / Robot GT / Model Prediction 同构):
  ① GraspPose 规划接触区  —— Dexonomy 说"该碰这里"
  ② RL 实测抓握性 q       —— 训练真跑出来的"碰这里能不能抓稳"
  ③ 覆盖度 log10(访问次数) —— 这张图的**支持域**, 哪里的数据够多

两行 = 正反两个视角。

⚠ **灰点 = 没有数据 (unknown), 不是分低。** 512 个 FPS 采样点里只有一部分被碰过,
  密集点云是从它们**最近邻插值**出来的, 离最近有效样本超过 `--radius` 的一律画灰 ——
  不这么做, 138 个样本铺到 3 万个点上会看起来比实际有信息得多。
"""
import argparse
import json
import os

import numpy as np


def dense_surface(mesh, n):
    """密集表面采样, 连同每点的面法向 (法向用于**背面剔除**)."""
    import trimesh
    p, fid = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(p), np.asarray(mesh.face_normals)[fid]


def facing(nrm, elev, azim):
    """朝向相机的点 —— 不剔背面的话前后两层点会叠在一起, 看不出实面."""
    e, a = np.radians(elev), np.radians(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return (nrm @ d) > 0.0


def paint(dense, src_pts, src_val, radius):
    """把稀疏采样点的值最近邻铺到密集点上; 超出 radius 的标 NaN (=unknown)."""
    d = np.linalg.norm(dense[:, None, :] - src_pts[None, :, :], axis=2)
    j = d.argmin(1)
    v = src_val[j].astype(float)
    v[d.min(1) > radius] = np.nan
    return v


def panel(fig, pos, dense, nrm, val, title, elev, azim, cmap="jet",
          vmin=None, vmax=None, ps=7.0):
    ax = fig.add_subplot(*pos, projection="3d")
    vis = facing(nrm, elev, azim)
    d, v = dense[vis], val[vis]
    ok = ~np.isnan(v)
    if (~ok).any():
        ax.scatter(*d[~ok].T, c="0.85", s=ps, marker=".", linewidths=0)
    if ok.any():
        ax.scatter(*d[ok].T, c=v[ok], cmap=cmap, vmin=vmin, vmax=vmax,
                   s=ps, marker=".", linewidths=0)
    r = np.abs(dense).max() * 0.72
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold", pad=0)
    return ax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--clip", default="Grasp3")
    p.add_argument("--prior", default="")
    p.add_argument("--out", default=None)
    p.add_argument("--n_dense", type=int, default=120000)
    p.add_argument("--ps", type=float, default=7.0, help="点径")
    p.add_argument("--radius", type=float, default=0.006,
                   help="m, 密集点离最近有效样本超过它就画灰 (默认 6mm ≈ FPS 间距)")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    import trimesh

    from rl_rebuild.correction import clips

    z = np.load(a.npz, allow_pickle=True)
    pts, s, n, q, w = z["points"], z["s"], z["n"], z["heatmap"], z["weight"]
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    m = w > 0
    mesh = trimesh.load(clips.clip_entry(a.clip)["mesh"], force="mesh")
    dense, nrm = dense_surface(mesh, a.n_dense)

    # ① 规划接触区: 离 prior 接触点的高斯衰减 (σ=8mm), 2cm 外算 unknown
    prior_f = None
    if a.prior:
        cp = np.load(a.prior)["contact_pos"]
        dmin = np.linalg.norm(dense[:, None, :] - cp[None, :, :], axis=2).min(1)
        prior_f = np.exp(-(dmin ** 2) / (2 * 0.008 ** 2))
        prior_f[dmin > 0.02] = np.nan
    # ② 实测分数 (只铺有效点)
    score_f = paint(dense, pts[m], q[m], a.radius) if m.any() else np.full(len(dense), np.nan)
    # ③ 覆盖度 (全部点都有, 0 也算数据)
    cov_f = paint(dense, pts, np.log10(n + 1), a.radius * 2)

    fig = plt.figure(figsize=(16, 9.6), facecolor="white")
    lo, hi = (float(q[m].min()), float(q[m].max())) if m.any() else (0.0, 1.0)
    for r_i, (el, az) in enumerate([(20, 45), (20, 225)]):
        tag = "正面" if r_i == 0 else "背面"
        if prior_f is not None:
            panel(fig, (2, 3, r_i * 3 + 1), dense, nrm, prior_f,
                  f"① GraspPose 规划接触区 ({tag})", el, az, vmin=0, vmax=1, ps=a.ps)
        panel(fig, (2, 3, r_i * 3 + 2), dense, nrm, score_f,
                   f"② RL 实测抓握性 q ({tag})", el, az, vmin=lo, vmax=hi, ps=a.ps)
        panel(fig, (2, 3, r_i * 3 + 3), dense, nrm, cov_f,
              f"③ 覆盖度 log10(访问+1) ({tag})", el, az, ps=a.ps)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=fig.axes, shrink=.4, pad=.01, location="right")
    cb.set_label(f"抓握性 q = (s+1)/(n+2)    [{lo:.2f} ~ {hi:.2f}]", fontsize=10)

    fig.suptitle(
        f"接触分数图  {meta.get('clip', a.clip)} / prior {meta.get('prior', '—')} / "
        f"task={meta.get('task', '—')}\n"
        f"有效点 {int(m.sum())}/{len(n)} ({m.mean()*100:.0f}%)  ·  访问总量 {n.sum():.0f}  ·  "
        f"q 中位 {np.median(q[m]) if m.any() else float('nan'):.3f}  ·  "
        f"灰色 = 没有数据 (unknown), 不是分低",
        fontsize=12, fontweight="bold")
    out = a.out or os.path.splitext(a.npz)[0] + ".png"
    fig.savefig(out, dpi=110, facecolor="white", bbox_inches="tight")
    print(f"写出 {out}  (密集点 {len(dense)}, 有效样本 {int(m.sum())})")


if __name__ == "__main__":
    main()
