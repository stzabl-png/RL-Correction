#!/usr/bin/env python3
"""ARCTIC 真值 vs 我们的重建 —— 三联可视化。

为什么要俯视图: 本次的主误差是**深度**(中位 236mm), 而深度误差在画面里几乎不可见
(投影质心只差 33px, 两边网格投影都精确贴合物体)。只看画面会得出"重建很好"的错误结论。
俯视图把相机-物体的距离画出来, 才看得见我们把物体放远了 23%。

左: 原视频 + 两边网格投影(绿=ARCTIC真值, 洋红=我们)
右: 相机系俯视图(X-Z平面), 相机在原点, 两个物体的实际位置
下: conf_pos 与真实误差随时间, 带游标 —— 看曲线是否同起同落
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

S = Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad")
A = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")
D = Path("/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/arctic/s05/laptop_grab_01")
GT_C, OUR_C = (60, 220, 60), (220, 60, 220)
PANEL_W, STRIP_H = 620, 210

meta = json.loads((S / "arctic_ego/s05__laptop_grab_01.meta.json").read_text())
v2a = {int(r["video_frame"]): int(r["arctic_vidx"]) for r in meta["index"]}
K = np.asarray(meta["K_video"], float)

z = np.load(D / "world_fused.npz", allow_pickle=True)
C = np.asarray(z["object_ob_in_cam"], float)
Vo = np.asarray(trimesh.load(D / "objects/object_0/object_mesh_scaled_final.obj",
                             force="mesh").vertices, float)
Vo = Vo[np.random.default_rng(0).choice(len(Vo), min(4000, len(Vo)), replace=False)]
Vg = np.asarray(trimesh.load(A / "meta/object_vtemplates/laptop/mesh.obj",
                             force="mesh").vertices, float) / 1000.0
Vg = Vg[np.random.default_rng(0).choice(len(Vg), min(4000, len(Vg)), replace=False)]

o = np.load(A / "raw_seqs/s05/laptop_grab_01.object.npy", allow_pickle=True)
ego = np.load(A / "raw_seqs/s05/laptop_grab_01.egocam.dist.npy", allow_pickle=True).item()
T = len(o)
w2e = np.tile(np.eye(4), (T, 1, 1))
w2e[:, :3, :3] = np.asarray(ego["R_k_cam_np"], float)
w2e[:, :3, 3] = np.asarray(ego["T_k_cam_np"], float).reshape(T, 3)
P = np.tile(np.eye(4), (T, 1, 1))
P[:, :3, :3] = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
P[:, :3, 3] = o[:, 4:7] / 1000.0
G = np.einsum("tij,tjk->tik", w2e, P)

frames = sorted(v2a)
err = np.full(len(C), np.nan)
for t in frames:
    co = C[t, :3, :3] @ Vo.mean(0) + C[t, :3, 3]
    cg = G[v2a[t], :3, :3] @ Vg.mean(0) + G[v2a[t], :3, 3]
    err[t] = np.linalg.norm(co - cg) * 1000

audit = json.loads((S / "arctic_audit.json").read_text())
rec = next(t for t in audit["takes"] if "laptop_grab_01" in t.get("take", ""))
conf = np.full(len(C), np.nan)
for r in rec["per_frame"]:
    if r["frame"] < len(C):
        conf[r["frame"]] = r.get("conf_pos", np.nan)

cap = cv2.VideoCapture(str(S / "arctic_ego/s05__laptop_grab_01.mp4"))
W, H = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(str(S / "arctic_gt_vs_recon.mp4"),
                      cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W + PANEL_W, H + STRIP_H))

# top-down extent, fixed so the viewer can compare frames
allz = np.concatenate([[C[t, 2, 3] for t in frames], [G[v2a[t], 2, 3] for t in frames]])
ZMAX = float(np.nanmax(allz)) * 1.15
XR = 0.6


def topdown(t):
    p = np.full((H, PANEL_W, 3), 24, np.uint8)
    def to_px(x, zz):
        return (int(PANEL_W / 2 + x / XR * (PANEL_W / 2 - 20)),
                int(H - 30 - zz / ZMAX * (H - 70)))
    for zz in np.arange(0.2, ZMAX, 0.2):                       # 深度刻度
        y = to_px(0, zz)[1]
        cv2.line(p, (10, y), (PANEL_W - 10, y), (55, 55, 55), 1)
        cv2.putText(p, f"{zz:.1f}m", (12, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
    cx, cy = to_px(0, 0)
    cv2.drawMarker(p, (cx, cy), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)
    cv2.putText(p, "camera", (cx - 26, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for V, M, col in ((Vg, G[v2a[t]], GT_C), (Vo, C[t], OUR_C)):
        X = (M[:3, :3] @ V.T).T + M[:3, 3]
        for x, zz in X[::12, [0, 2]]:
            u, vv = to_px(x, zz)
            if 0 <= u < PANEL_W and 0 <= vv < H:
                p[vv, u] = col
    zo, zg = C[t, 2, 3], G[v2a[t], 2, 3]
    cv2.putText(p, "TOP-DOWN (camera frame)", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.putText(p, f"GT depth   {zg*1000:4.0f} mm", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GT_C, 2)
    cv2.putText(p, f"OURS depth {zo*1000:4.0f} mm", (14, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OUR_C, 2)
    cv2.putText(p, f"ratio {zo/zg:5.3f}", (14, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 210, 90), 2)
    return p


def strip(t):
    s = np.full((STRIP_H, W + PANEL_W, 3), 18, np.uint8)
    n = len(C)
    def X(i):
        return int(i / max(n - 1, 1) * (s.shape[1] - 60)) + 30
    for name, arr, col, lo, hi, row in (("conf_pos", conf, (90, 200, 255), 0, 100, 0),
                                        ("err (mm)", err, (120, 120, 255), 0, 900, 1)):
        y0, y1 = 24 + row * 92, 96 + row * 92
        cv2.putText(s, name, (4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        pts = [(X(i), int(y1 - (np.clip(arr[i], lo, hi) - lo) / (hi - lo) * (y1 - y0)))
               for i in range(n) if not np.isnan(arr[i])]
        for a, b in zip(pts, pts[1:]):
            if b[0] - a[0] <= 3:
                cv2.line(s, a, b, col, 1)
        cv2.line(s, (30, y1), (s.shape[1] - 30, y1), (70, 70, 70), 1)
    cv2.line(s, (X(t), 10), (X(t), STRIP_H - 6), (255, 255, 255), 1)
    return s


i = 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i in v2a:
        # 交错采样, 否则后画的一方会把另一方完全盖住 —— 而"两边投影都贴合"
        # 恰恰是本图要传达的核心(图像看不出深度错误)
        for (V, M, col), off in (((Vg, G[v2a[i]], GT_C), 0), ((Vo, C[i], OUR_C), 3)):
            X = (M[:3, :3] @ V.T).T + M[:3, 3]
            X = X[X[:, 2] > 1e-3]
            uv = (K @ X.T).T
            uv = (uv[:, :2] / uv[:, 2:3]).astype(int)
            for u, v in uv[off::6]:
                if 0 <= u < W and 0 <= v < H:
                    f[v, u] = col
        cv2.putText(f, "GT", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GT_C, 2)
        cv2.putText(f, "OURS", (60, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, OUR_C, 2)
        c = conf[i]
        cv2.putText(f, f"f{i}   conf_pos {('%.0f' % c) if not np.isnan(c) else '--'}"
                       f"   err {err[i]:.0f}mm",
                    (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        out.write(np.vstack([np.hstack([f, topdown(i)]), strip(i)]))
    i += 1
cap.release()
out.release()
print(f"wrote {S/'arctic_gt_vs_recon.mp4'}")
