"""视频接触带 -> affordance npz(**自动蒸馏**)。

还的是台账 §2.22 记的那笔欠账:"接触点(空间)目前是**人工做的一条**,自动蒸馏链路待建"。
Screw27 的 72 点环带当年是手工从热图蒸馏的;本模块把它变成一条命令。

输入 = 重建产出的 `contact/contact_v2_<oid>_<hand>.npz`:
    probe_local (N,3)  物体**局部系**的探针点
    weight      (N,)   该点被手碰到的证据权重(0/1 型为主)
输出 = 训练侧 affordance npz(与 `Screw27_cap_affordance.npz` 同格式):
    pts (K,3) 接触带点 / centroid (3,) / band_s (2,) 沿主轴范围 / radius 平均半径 / source

⚠ 只在**网格未被缩放**时可直接用;若训练用的网格做过缩放, 必须传 --scale 同步缩放,
否则接触带与网格对不上(2026-08-15 起 pour17 不再缩放, 故默认 1.0)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def distill(npz_path: Path, *, k_max: int = 200, scale: float = 1.0,
            axis: int | None = None, axis_origin=(0.0, 0.0, 0.0)) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    P = np.asarray(d["probe_local"], float) * float(scale)
    W = np.asarray(d["weight"], float)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    sel = W > 0
    if not sel.any():
        raise RuntimeError(f"{npz_path}: 没有正权重点(接触带为空)")
    P, W = P[sel], W[sel]
    if len(P) > k_max:                      # 按权重取前 k_max, 权重同分时随机但可复现
        rng = np.random.default_rng(0)
        order = np.lexsort((rng.random(len(W)), -W))
        P, W = P[order[:k_max]], W[order[:k_max]]
    c = P.mean(0)
    if axis is None:
        # ⚠ 主轴必须取**物体网格**的长轴, 不能取接触点云的展布 —— 接触带是**环绕**
        #   物体的, 其径向展布常大于轴向, 会把主轴判成径向(2026-08-15 实测踩到)。
        raise ValueError("必须给 axis(物体网格主轴索引)或 --mesh")
    other = [i for i in range(3) if i != axis]
    # ⚠ 半径必须从**主轴**量, 不能从点云质心量 —— 接触带常只包一部分圆周(实测瓶 150°),
    #   质心根本不在轴上, 从质心量会把 2.6cm 的环带算成 1.1cm(2026-08-15 踩到)。
    r = float(np.linalg.norm(P[:, other] - np.asarray(axis_origin)[other], axis=1).mean())
    band = np.array([float(P[:, axis].min()), float(P[:, axis].max())])
    return {"pts": P.astype(np.float32),
            # 同一份文件喂两个消费者: `affordance`(对齐目标, 要 points_raw+heatmap)
            # 与 `affordance_npz`(pad_approach 塑形, 要 pts)。
            "points_raw": P.astype(np.float32), "heatmap": W.astype(np.float32),
            "centroid": c.astype(np.float32),
            "band_s": band.astype(np.float32), "radius": np.float32(r),
            "axis": np.int32(axis), "n_src": np.int32(int(sel.sum())),
            "source": np.frombuffer(
                (f"{npz_path} | scale={scale} | k={len(P)}/{int(sel.sum())} | "
                 f"frames_used={meta.get('frames_used')} "
                 f"hand_mask_agreement={meta.get('hand_mask_agreement')}").encode(),
                dtype=np.uint8)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="训练网格若缩放过, 同步缩放")
    ap.add_argument("--k-max", type=int, default=200)
    ap.add_argument("--mesh", type=Path, default=None, help="用它的最长边定主轴")
    ap.add_argument("--axis", type=int, default=None, choices=[0, 1, 2])
    a = ap.parse_args(argv)
    ax = a.axis
    if ax is None and a.mesh is not None:
        import trimesh
        ax = int(np.argmax(trimesh.load(a.mesh, process=False, force="mesh").extents))
    if ax is None:
        raise SystemExit("请给 --axis 或 --mesh(主轴来自物体网格, 不能从接触点云猜)")
    org = (0.0, 0.0, 0.0)
    if a.mesh is not None:
        import trimesh
        org = tuple(np.asarray(trimesh.load(a.mesh, process=False,
                                            force="mesh").vertices).mean(0) * a.scale)
    d = distill(a.contact_npz, k_max=a.k_max, scale=a.scale, axis=ax, axis_origin=org)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **d)
    ax = int(d["axis"])
    print(f"  接触带 {len(d['pts'])} 点(源 {int(d['n_src'])} 个正权重) | 主轴={'xyz'[ax]} | "
          f"沿轴 {d['band_s'][0]*100:.1f}~{d['band_s'][1]*100:.1f}cm | "
          f"平均半径 {float(d['radius'])*100:.2f}cm | 质心 {np.round(d['centroid']*100, 2).tolist()}cm")
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
