"""U15: 从 SAM3D 重建 (egodex_auto 的 textured glb) 抽取真实纹理+UV, 对齐到入库网格帧存档。
产物: datasets/unscrew_bottle/17/cache/textures/real_{bottle,cap}.png / real_{tag}_uv.npz
       (verts=对齐后 GLB 顶点[入库 z-up 落底帧], uv=GLB 原 UV)
对齐: GLTF y-up -> z-up, 上下符号按「入库网格顶点->对齐GLB 最近邻距离中位」硬判 (剖面 L2 会选错, 实测瓶被翻正);
      z 线性映射到入库高度, 径向按最大半径比。texture_objects.apply_textures 运行时最近邻迁移 UV。
用法: $PY tasks/Unscrew/17/B_SmokeTest/extract_real_textures.py   (纯 numpy/trimesh/scipy, 无 Isaac)
"""
import numpy as np, trimesh, os
from scipy.spatial import cKDTree
import sys
TASK = sys.argv[1] if len(sys.argv) > 1 else "unscrew17"
PRESET = {
    "unscrew17": dict(G="/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/screw_unscrew_bottle_cap/17/objects",
                      D="datasets/unscrew_bottle/17/cache/textures",
                      objs=[(0, "bottle", "datasets/unscrew_bottle/17/objects/object_0/object_mesh_scaled_final.obj"),
                            (1, "cap", "datasets/unscrew_bottle/17/objects/object_1/object_mesh_scaled_final.obj")]),
    "pour17": dict(G="/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/pour/17/objects",
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
    best = None
    for sgn in (1, -1):
        W = np.stack([V[:, 0], -sgn * V[:, 2], sgn * V[:, 1]], 1)
        W[:, 2] = (W[:, 2] - W[:, 2].min()) / max(W[:, 2].ptp(), 1e-9) * (zo1 - zo0) + zo0
        rs = np.hypot(OV[:, 0], OV[:, 1]).max() / max(np.hypot(W[:, 0], W[:, 1]).max(), 1e-9)
        W[:, 0] *= rs; W[:, 1] *= rs
        d, _ = cKDTree(W).query(sub, k=1)
        print(f"[{tag}] sgn {sgn:+d}: NN中位 {np.median(d)*1000:.2f}mm")
        if best is None or np.median(d) < best[0]:
            best = (np.median(d), sgn, W)
    _, sgn, W = best
    np.savez(f"{D}/real_{tag}_uv.npz", verts=W.astype(np.float32), uv=UV)
    print(f"[{tag}] ★sgn {sgn:+d} 存档 -> {D}/real_{tag}*")
