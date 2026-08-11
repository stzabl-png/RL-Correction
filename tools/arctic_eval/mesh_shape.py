#!/usr/bin/env python3
"""重建网格 vs ARCTIC 真值网格 —— 只比形状, 不比尺度和朝向。

为什么不能直接比 bbox 三轴:SAM3D 输出的朝向是任意的(不保留 mask 里的方向), 轴对齐 bbox
会随朝向变化, 比出来的数没有意义。这里先 PCA 对齐到主轴, 再按最长轴归一化,
剩下的就是纯形状(长宽比 + 轮廓)。

每个网格出三张正交剪影(沿三个主轴看), 并列真值与我们的, 形状像不像一眼可判。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

A = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data/meta/object_vtemplates")
RR = Path("/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/arctic")
TILE = 260


def pca_norm(V: np.ndarray) -> np.ndarray:
    """PCA 对齐 + 按最长轴归一化 -> 尺度/朝向无关的形状表示。"""
    V = V - V.mean(0)
    _, _, Vt = np.linalg.svd(V - V.mean(0), full_matrices=False)
    V = V @ Vt.T                                  # 主轴对齐
    ext = V.ptp(0)
    order = np.argsort(-ext)                      # 长轴在前, 保证两边可比
    V = V[:, order]
    return V / V.ptp(0).max()


def silhouettes(V: np.ndarray) -> np.ndarray:
    """沿三个主轴各出一张正交剪影。"""
    tiles = []
    for drop, name in ((2, "axis1-2"), (1, "axis1-3"), (0, "axis2-3")):
        keep = [i for i in range(3) if i != drop]
        p = V[:, keep]
        img = np.full((TILE, TILE, 3), 22, np.uint8)
        q = ((p - p.min(0)) / max(p.ptp(0).max(), 1e-9) * (TILE - 40) + 20).astype(int)
        for x, y in q:
            cv2.circle(img, (int(x), TILE - 1 - int(y)), 1, (235, 235, 235), -1)
        cv2.putText(img, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 255), 1)
        tiles.append(img)
    return np.hstack(tiles)


def row(label: str, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Vn = pca_norm(V)
    ratio = np.sort(Vn.ptp(0))[::-1]
    strip = silhouettes(Vn)
    bar = np.full((30, strip.shape[1], 3), 12, np.uint8)
    cv2.putText(bar, f"{label}   ratio {ratio[0]:.2f} : {ratio[1]:.2f} : {ratio[2]:.2f}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return np.vstack([bar, strip]), ratio


def load(p: Path, n=20000) -> np.ndarray | None:
    if not p.is_file():
        return None
    V = np.asarray(trimesh.load(p, force="mesh").vertices, float)
    if len(V) > n:
        V = V[np.random.default_rng(0).choice(len(V), n, replace=False)]
    return V


def main() -> int:
    groups = {
        "ketchup": [("GT (ARCTIC)", A / "ketchup/mesh.obj"),
                    ("OURS f347 (current)", RR / "s07/ketchup_grab_01/objects/object_0/object_mesh_scaled_final.obj"),
                    ("OURS f300 (selector)", RR / "s07/ketchup_grab_01_f300/objects/object_0/object_mesh_scaled_final.obj")],
        "laptop": [("GT (ARCTIC)", A / "laptop/mesh.obj"),
                   ("OURS f642", RR / "s05/laptop_grab_01/objects/object_0/object_mesh_scaled_final.obj")],
    }
    for name, items in groups.items():
        rows, ratios = [], {}
        for label, p in items:
            V = load(p)
            if V is None:
                print(f"  缺: {p}")
                continue
            img, r = row(f"{name} · {label}", V)
            rows.append(img)
            ratios[label] = r
        if not rows:
            continue
        out = Path(f"/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/"
                   f"scratchpad/shape_{name}.jpg")
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{name} -> {out}")
        gt = ratios.get("GT (ARCTIC)")
        for k, v in ratios.items():
            if k == "GT (ARCTIC)" or gt is None:
                continue
            print(f"    {k}: 轴比 {np.round(v,3)}  vs 真值 {np.round(gt,3)}  "
                  f"相对偏差 {np.round((v/gt-1)*100,1)}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
