"""核验 sdf_rz.npz 近似查表 vs trimesh 精确 signed_distance.

建表用的是"表面采样+KD树最近点, 符号取最近面法向"的近似. 这个近似在**中轴面附近**
会翻符号 (最近点落在薄壁的哪一侧是不定的). 所以不能只看全域统计, 要专门看
**接触判据真正用到的那条带**: 垫心离表面 -20mm ~ +2mm.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
SP = "/home/lyh/Project/RL_Correction/tools/grasp_design"
OBJ = np.array([-0.0506, -0.0833, 0.8652])

c = np.load(f"{SP}/sdf_rz.npz")
R_AX, Z_AX, SDF = c["r"], c["z"], c["sdf"]
rr, zz = np.meshgrid(R_AX, Z_AX, indexing="ij")
pts = np.column_stack([rr.ravel() + OBJ[0], np.full(rr.size, OBJ[1]), zz.ravel()])
sd = SDF.ravel()

import trimesh                                                        # noqa: E402
from rl_rebuild.correction import clips as _clips, frames as _F       # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import _flat_rest_quat  # noqa: E402
e = _clips.clip_entry("Grasp2")
m = trimesh.load(e["mesh"], process=False)
v = np.asarray(m.vertices)
m.vertices = _F.rot_apply(np.broadcast_to(_flat_rest_quat(_F.load_obj_verts(e["mesh"])),
                                          (len(v), 4)), v) + OBJ

rng = np.random.default_rng(1)
BANDS = [("接触带 (-20~+2mm, 判据真正用的)", (sd > -0.020) & (sd < 0.002)),
         ("物体内部 (>0)", sd > 0),
         ("远场 (<-20mm)", sd <= -0.020)]
N = 60
print(f"网格 {sd.size} 点, sd 范围 [{sd.min()*100:.2f}, {sd.max()*100:.2f}]cm\n")
for tag, msk in BANDS:
    idx = np.flatnonzero(msk)
    if len(idx) == 0:
        print(f"{tag}: 无点"); continue
    pick = rng.choice(idx, min(N, len(idx)), replace=False)
    ref = trimesh.proximity.signed_distance(m, pts[pick])
    err = np.abs(ref - sd[pick])
    same = int((np.sign(ref) == np.sign(sd[pick])).sum())
    print(f"{tag}  ({len(idx)} 点, 抽 {len(pick)})")
    print(f"   最大差 {err.max()*1000:7.3f}mm   中位 {np.median(err)*1000:7.3f}mm   "
          f"p95 {np.percentile(err,95)*1000:7.3f}mm   符号一致 {same}/{len(pick)}")
    if err.max() > 0.0005:
        w = pick[np.argmax(err)]
        print(f"   最差点: r={np.linalg.norm(pts[w,:2]-OBJ[:2])*100:.2f}cm "
              f"z-obj={(pts[w,2]-OBJ[2])*100:+.2f}cm  "
              f"查表 {sd[w]*1000:+.2f}mm vs 精确 {ref[np.argmax(err)]*1000:+.2f}mm")
