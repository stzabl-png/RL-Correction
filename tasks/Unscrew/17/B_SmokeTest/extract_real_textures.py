"""U15: 从 SAM3D 重建 (egodex_auto 的 textured glb) 抽取真实纹理+UV, 对齐到入库网格帧存档。
产物: datasets/unscrew_bottle/17/cache/textures/real_{bottle,cap}.png / real_{tag}_uv.npz
       (verts=对齐后 GLB 顶点[入库 z-up 落底帧], uv=GLB 原 UV)
对齐: GLTF y-up -> z-up, 上下符号按「入库网格顶点->对齐GLB 最近邻距离中位」硬判 (剖面 L2 会选错, 实测瓶被翻正);
      z 线性映射到入库高度, 径向按最大半径比。texture_objects.apply_textures 运行时最近邻迁移 UV。
用法: $PY tasks/Unscrew/17/B_SmokeTest/extract_real_textures.py   (纯 numpy/trimesh/scipy, 无 Isaac)
"""
import numpy as np, trimesh, os
from scipy.spatial import cKDTree
def align_by_long_axis(V, OV, tag):
    """U16.1 数据驱动对齐: 不假设任何 up 轴 (pour staged 实测非 z-up 的教训)。
    两边各取延伸最大轴为长轴, 构造置换旋转 ±号, 长轴线性映射 + 径向按最大半径比;
    NN 中位选优并**硬断言** (<15mm; 对齐后各向延伸比 ∈[0.7,1.4]) —— 配错对/配错轴必须炸, 不许静默。
    返回 (nn_med, M4x4, 对齐后顶点)。"""
    from scipy.spatial import cKDTree as _KD
    def _sym_axis(P):
        e = P.ptp(0); m = np.median(e)
        return int(np.argmax(np.abs(np.log(np.maximum(e, 1e-9) / max(m, 1e-9)))))
    # 对称轴 = 与三向延伸中位数偏离最大的轴 (瓶=长轴, 盖/碟=短轴; 长轴 argmax 对碟形退化, 断言抓过)
    ax_g = _sym_axis(V); ax_s = _sym_axis(OV)
    sub = OV[np.random.RandomState(0).choice(len(OV), min(6000, len(OV)), replace=False)]
    s_lo = OV[:, ax_s].min(); s_h = OV[:, ax_s].ptp()
    rad_axes_s = [a for a in range(3) if a != ax_s]
    r_s = np.hypot(OV[:, rad_axes_s[0]], OV[:, rad_axes_s[1]]).max()
    best = None
    for sgn in (1, -1):
        # 置换旋转: glb 长轴 -> staged 长轴(±), 余两轴任意正交完成 (回转体 roll 自由)
        R = np.zeros((3, 3)); R[ax_s, ax_g] = sgn
        rest_g = [a for a in range(3) if a != ax_g]
        R[rad_axes_s[0], rest_g[0]] = 1.0
        R[rad_axes_s[1], rest_g[1]] = float(np.linalg.det(
            np.eye(3)) if True else 1.0) or 1.0
        R[rad_axes_s[1], rest_g[1]] = 1.0
        if np.linalg.det(R) < 0:
            R[rad_axes_s[1], rest_g[1]] = -1.0
        W = V @ R.T
        sz = s_h / max(W[:, ax_s].ptp(), 1e-9)
        r_g = np.hypot(W[:, rad_axes_s[0]], W[:, rad_axes_s[1]]).max()
        rr = r_s / max(r_g, 1e-9)
        S = np.eye(3); S[ax_s, ax_s] = sz
        S[rad_axes_s[0], rad_axes_s[0]] = rr; S[rad_axes_s[1], rad_axes_s[1]] = rr
        W2 = W @ S.T
        t = np.zeros(3); t[ax_s] = s_lo - W2[:, ax_s].min()
        W2 = W2 + t
        d, _ = _KD(W2).query(sub, k=1)
        med = float(np.median(d))
        print(f"[{tag}] 长轴 {ax_g}->{ax_s} sgn {sgn:+d}: NN中位 {med*1000:.2f}mm")
        if best is None or med < best[0]:
            M = np.eye(4); M[:3, :3] = S @ R; M[:3, 3] = t
            best = (med, M, W2)
    med, M, W2 = best
    assert med < 0.015, f"{tag}: 对齐 NN 中位 {med*1000:.1f}mm ≥15mm —— 配对/轴错, 拒绝静默产出"
    ratio = W2.ptp(0) / np.maximum(OV.ptp(0), 1e-9)
    assert (ratio > 0.7).all() and (ratio < 1.4).all(), f"{tag}: 对齐后延伸比 {np.round(ratio,2)} 出界"
    return med, M, W2

import sys
TASK = sys.argv[1] if len(sys.argv) > 1 else "unscrew17"
PRESET = {
    "unscrew17": dict(G="/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/screw_unscrew_bottle_cap/17/objects",
                      R="/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/screw_unscrew_bottle_cap/17/retarget",
                      D="datasets/unscrew_bottle/17/cache/textures",
                      objs=[(0, "bottle", "datasets/unscrew_bottle/17/objects/object_0/object_mesh_scaled_final.obj"),
                            (1, "cap", "datasets/unscrew_bottle/17/objects/object_1/object_mesh_scaled_final.obj")]),
    "pour17": dict(G="/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/pour/17/objects",
                   R="/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/pour/17/retarget",
                   D="datasets/pour17/cache/textures",
                   objs=[(1, "bottle", "datasets/pour17/objects/object_1/object_mesh_scaled_final.obj"),
                         (0, "cup", "datasets/pour17/objects/object_0/object_mesh_scaled_final.obj")]),
}[TASK]
G, D = PRESET["G"], PRESET["D"]
os.makedirs(D, exist_ok=True)
for oi, tag, staged in PRESET["objs"]:
    glb = trimesh.load(f"{G}/object_{oi}/textured/object_mesh_scaled_final_textured.glb", force="mesh", process=False)
    V = np.asarray(glb.vertices, np.float64)
    UV = np.asarray(glb.visual.uv, np.float32)
    glb.visual.material.baseColorTexture.save(f"{D}/real_{tag}.png")
    obj = trimesh.load(staged, force="mesh", process=False)
    OV = np.asarray(obj.vertices); zo0, zo1 = OV[:, 2].min(), OV[:, 2].max()
    sub = OV[np.random.RandomState(0).choice(len(OV), min(8000, len(OV)), replace=False)]
    med, R4, W = align_by_long_axis(V, OV, tag)
    np.savez(f"{D}/real_{tag}_uv.npz", verts=W.astype(np.float32), uv=UV)
    print(f"[{tag}] ★对齐存档 (NN中位 {med*1000:.1f}mm) -> {D}/real_{tag}*")

# ---- U16: 视觉挂载变换 (retarget/object_*_textured.usd -> 入库帧 4x4) ----
# 视觉网格直接换 SAM3D USD (跨图集 UV 迁移的碎花病根绕开); 上下符号同款 NN 硬判。
import json
from pxr import Usd, UsdGeom
RD = PRESET.get("R")
if RD:
    for oi, tag, staged in PRESET["objs"]:
        up = f"{RD}/object_{oi}_textured.usd"
        if not os.path.isfile(up):
            print(f"[{tag}] 无 textured usd, 跳过 visual json"); continue
        st = Usd.Stage.Open(up); V = None; phys = 0
        for pr in st.Traverse():
            if pr.IsA(UsdGeom.Mesh) and V is None:
                V = np.array(UsdGeom.Mesh(pr).GetPointsAttr().Get(), float)
            phys += sum(1 for sc in pr.GetAppliedSchemas() if "Physics" in sc)
        obj = trimesh.load(staged, force="mesh", process=False); OV = np.asarray(obj.vertices)
        med, M, _W2 = align_by_long_axis(V, OV, f"{tag}(visual)")
        json.dump({"usd": up, "matrix": M.tolist(), "nn_med_mm": round(med * 1000, 2),
                   "phys_apis": phys}, open(f"{D}/real_{tag}_visual.json", "w"), indent=1)
        print(f"[{tag}] visual json: NN中位 {med*1000:.1f}mm physAPI={phys}")
