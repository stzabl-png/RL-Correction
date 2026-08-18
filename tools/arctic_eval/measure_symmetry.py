#!/usr/bin/env python3
"""从 ARCTIC **真值网格**测每个物体的旋转对称群 —— 独立于任何一次重建。

为什么不能用 rot_observability 推(GraspPose 2026-08-17 指出): 那是 pose_audit 从**被评估的
那次重建**算的, 拿它决定 GT 度量的对称群 = 让尺子依赖被测物, 重建越差反而获得越松的群,
方向正好反了。对称性是网格的固有属性, 应当独立测。

位姿的物理模糊性: 图像上无法区分 (R,t) 与 (R·S_R, t + R·S_t), 只要 S_R·V + S_t ≈ V。
⇒ 判定某个旋转 S_R 是否属于对称群时**必须允许配一个平移** S_t, 否则对称中心不在坐标原点的
   物体会被漏掉。这里对每个候选旋转用质心差当 S_t, 再量单向最近点距离。

阈值: 残差中位 < 物体**最小轴长的 2%** 判为对称(GraspPose 建议的量级)。
"""
import json, sys, glob, os
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ARCTIC = os.environ.get("ARCTIC_DATA",
                        "/media/lyh/DATA2/arctic/repo/data/arctic_data/data")

def load_obj(p):
    V = []
    for ln in open(p, "r", errors="ignore"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
    return np.asarray(V, float)

def residual(V, tree, R, extent):
    """把 V 转一下, 配最优平移(质心差), 量单向最近点距离中位 / 最小轴长"""
    W = V @ R.T
    W = W + (V.mean(0) - W.mean(0))          # S_t
    d, _ = tree.query(W, k=1)
    return float(np.median(d)) / extent

def main():
    tmpl = sorted(glob.glob(os.path.join(ARCTIC, "meta/object_vtemplates/*/mesh.obj")))
    if not tmpl:
        raise SystemExit("没找到 ARCTIC 物体模板: " + ARCTIC)
    TH = 0.02
    out = {}
    print("%-18s %-9s %-34s %s" % ("物体", "顶点数", "最小/最大轴长(m)", "对称群"))
    for p in tmpl:
        name = os.path.basename(os.path.dirname(p))
        V = load_obj(p)
        if len(V) < 50:
            print("%-18s 顶点太少, 跳过" % name); continue
        # ARCTIC 模板单位是 mm
        if np.ptp(V, axis=0).max() > 10:
            V = V / 1000.0
        ext = np.ptp(V, axis=0)
        emin = float(ext.min())
        tree = cKDTree(V)
        axes = np.eye(3)
        found = []
        # ① 连续对称(回转体): 绕某轴多个角度**全部**匹配
        cont = []
        for ai, ax in enumerate(axes):
            angs = np.arange(15, 360, 15)
            r = [residual(V, tree, Rotation.from_rotvec(ax * np.radians(t)).as_matrix(), emin)
                 for t in angs]
            if max(r) < TH:
                cont.append(ai)
        # ② 离散对称: 90/180 度
        for ai, ax in enumerate(axes):
            if ai in cont: continue
            for deg in (90, 180):
                r = residual(V, tree, Rotation.from_rotvec(ax * np.radians(deg)).as_matrix(), emin)
                if r < TH:
                    found.append((ai, deg, round(r, 4)))
        grp = {"continuous_axes": cont,
               "discrete": [{"axis": a, "deg": d, "resid": r} for a, d, r in found],
               "extent_m": [round(float(x), 4) for x in ext]}
        out[name] = grp
        desc = []
        if cont: desc.append("★连续(轴 %s)" % cont)
        for a, d, r in found: desc.append("%s轴%d°(残差%.3f)" % ("xyz"[a], d, r))
        print("%-18s %-9d %-34s %s" % (name, len(V), "%.3f / %.3f" % (emin, float(ext.max())),
                                       " ".join(desc) or "无(不对称)"))
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "arctic_symmetry.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n写入", sys.argv[1] if len(sys.argv) > 1 else "arctic_symmetry.json")

main()
