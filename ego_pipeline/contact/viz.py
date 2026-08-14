#!/usr/bin/env python3
"""接触提取结果的可视化。产物落在 `<take>/contact/` 里, 与数据同处。

    contact_heatmap.png    物体点云 + 接触频率热力图(4 视角 × 每个成功的组合)
    contact_overlay.jpg    把接触区投回**原视频**帧上 —— 用来肉眼核对位置对不对

两张图回答不同问题, 都要看:

  * 点云图看**接触区在物体上的哪里**(高度、包裹角度、成不成形)。
    形状散成几块通常意味着物体位姿在飘 —— 位姿越准接触区越集中。
  * 叠加图看**这个位置在真实画面里对不对**。3D 里再漂亮, 投回去落在手背上就是错的。

⚠ 只画 status=ok 的组合。失败的组合在 contact_v2_summary.json 里有原因, 不在图上 ——
  图里没有的组合不等于"没碰过", 可能是"碰了但不构成抓握"。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def heatmap(recon_dir: Path, out: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(str(recon_dir / "contact" / "contact_v2_*.npz")))
    if not files:
        return None
    fig = plt.figure(figsize=(20, 6.4 * len(files)))
    sc = None
    for row, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        V = z["probe_local"].astype(float) - z["probe_local"].mean(0)
        w = z["weight"]
        hot = w > 0
        idx = np.arange(len(V))
        # 物体本体抽稀但画实, 否则形状看不出来(热点浮在空中没有参照)
        bg = idx[~hot][:: max(1, (~hot).sum() // 12000)]
        for k, (az, el, name) in enumerate([(-60, 15, "3/4 view"), (0, 8, "front"),
                                            (90, 8, "side"), (0, 88, "top")]):
            ax = fig.add_subplot(len(files), 4, row * 4 + k + 1, projection="3d")
            ax.scatter(V[bg, 0], V[bg, 1], V[bg, 2], s=3.0, c="#b8c4cc", alpha=0.55,
                       linewidths=0, depthshade=False)
            sc = ax.scatter(V[hot, 0], V[hot, 1], V[hot, 2], s=34, c=w[hot], cmap="turbo",
                            vmin=0, vmax=1, linewidths=0, depthshade=False)
            ax.view_init(elev=el, azim=az)
            r = np.abs(V).max() * 1.02
            ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
            ax.set_box_aspect([1, 1, 1]); ax.set_axis_off()
            ax.set_title(name, fontsize=12, pad=0)
        sw = m["stable_window"]
        fig.text(0.012, 1 - (row + 0.05) / len(files),
                 f"[MEASURED]  {m['object_id']} x {m['side']} hand   hot={m['hot_verts']}   "
                 f"window f{sw['span'][0]}-f{sw['span'][1]} ({sw['n']}f)   "
                 f"opposition {m['opposition_tips']:.2f}   gap {m['min_dist_mm_median']:.1f}mm   "
                 f"2D-veto {int(z['vetoed'].sum())}",
                 fontsize=14, color="#1a6b3c", va="top")
    cb = fig.colorbar(sc, ax=fig.axes, fraction=0.010, pad=0.01)
    cb.set_label("contact frequency  (frames touched / frames in window)", fontsize=12)
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def overlay(recon_dir: Path, video: Path, out: Path) -> Path | None:
    """接触区投回原视频帧。绿=被碰到过, 红=热点(≥半数帧)。"""
    import cv2
    files = sorted(glob.glob(str(recon_dir / "contact" / "contact_v2_*.npz")))
    if not files or not video.is_file():
        return None
    w = np.load(recon_dir / "world_fused.npz", allow_pickle=True)
    allT = (w["object_ob_in_world_all"] if "object_ob_in_world_all" in w.files
            else w["object_ob_in_world"][None])
    oids = ([str(x) for x in w["object_ids"]] if "object_ids" in w.files else ["object_0"])
    c2w, K = w["c2w"], w["K"]
    cap = cv2.VideoCapture(str(video))
    panels = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        V, wt = z["probe_local"].astype(float), z["weight"]
        sw = m["stable_window"]
        t = int((sw["span"][0] + sw["span"][1]) // 2)          # 稳定窗中点最有代表性
        if t >= allT.shape[1] or m["object_id"] not in oids:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap.read()
        if not ok:
            continue
        M = allT[oids.index(m["object_id"]), t]
        Vw = (M[:3, :3] @ V.T).T + M[:3, 3]
        Kc = np.linalg.inv(c2w[t])
        P = (Kc[:3, :3] @ Vw.T).T + Kc[:3, 3]
        vis = P[:, 2] > 1e-6
        u = P[:, 0] / np.clip(P[:, 2], 1e-6, None) * K[0, 0] + K[0, 2]
        v = P[:, 1] / np.clip(P[:, 2], 1e-6, None) * K[1, 1] + K[1, 2]
        H, W = img.shape[:2]
        inb = vis & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        for msk, col in (((wt > 0) & inb, (0, 255, 0)), ((wt >= 0.5) & inb, (0, 0, 255))):
            img[v[msk].astype(int), u[msk].astype(int)] = col
        cv2.putText(img, f"{m['object_id']} x {m['side']}  f{t} (win f{sw['span'][0]}-"
                    f"f{sw['span'][1]})  hot={m['hot_verts']}  oppo={m['opposition_tips']:.2f}",
                    (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
        cv2.putText(img, f"{m['object_id']} x {m['side']}  f{t} (win f{sw['span'][0]}-"
                    f"f{sw['span'][1]})  hot={m['hot_verts']}  oppo={m['opposition_tips']:.2f}",
                    (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.putText(img, "green=touched  red=hot(>=50% frames)", (14, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, "green=touched  red=hot(>=50% frames)", (14, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        panels.append(cv2.resize(img, (W // 2, H // 2)))
    cap.release()
    if not panels:
        return None
    cv2.imwrite(str(out), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir", type=Path)
    ap.add_argument("--video", type=Path, default=None, help="原视频; 给了才画叠加图")
    a = ap.parse_args(argv)
    cd = a.recon_dir / "contact"
    cd.mkdir(parents=True, exist_ok=True)
    p = heatmap(a.recon_dir, cd / "contact_heatmap.png")
    print(f"  热力图  -> {p}" if p else "  热力图: 没有 status=ok 的组合, 跳过")
    if a.video:
        q = overlay(a.recon_dir, a.video, cd / "contact_overlay.jpg")
        print(f"  叠加图  -> {q}" if q else "  叠加图: 跳过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
