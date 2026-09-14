"""Sweep2 格式母带的"朝口笔画"分析 (2026-09-13, 台账 408 §9.15 的方法固化): 刷面心在簸箕系的轨迹, 连续朝口 (−z) 且在口宽内的段,
以及每段起点前方可摆方块的候选位。用法: python stroke_scan.py <sweep2fmt.npz> [--min_rows 5]"""
import argparse, os, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
p = argparse.ArgumentParser(); p.add_argument("tape"); p.add_argument("--min_rows", type=int, default=5); a = p.parse_args()
z = np.load(a.tape, allow_pickle=True); q2R = lambda q: Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix()
pp, pq, bp, bq, bl = z["obj_pos_0"], z["obj_quat_0"], z["obj_pos_1"], z["obj_quat_1"], z["brush_contact_local"]
T = len(pp); B = np.array([q2R(pq[t]).T @ (bp[t] + q2R(bq[t]) @ bl - pp[t]) for t in range(T)]) * 100
TZ = float(z["table_z"]); _m = os.path.join(os.path.dirname(os.path.abspath(a.tape)), "..", "..", "..", "..", "..")
_V = np.asarray(trimesh.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../datasets/sweep408/objects/object_0/object_mesh_scaled_final.obj"), process=False).vertices)
_face = _V[(_V[:, 1] < -0.046) & (_V[:, 2] > 0.005) & (_V[:, 2] < 0.110)]; _face = _face[np.random.RandomState(0).choice(len(_face), 2000, replace=False)]
hmin = np.array([(_face @ q2R(bq[t]).T + bp[t])[:, 2].min() - TZ for t in range(T)]) * 100   # 刷毛最低点离桌 cm
tilt = np.degrees(np.arccos([q2R(pq[t])[2, 1] for t in range(T)]))
inside = (abs(B[:, 0]) < 7) & (B[:, 2] < 4.7)
print(f"刷毛最低点离桌: 全带中位 {np.median(hmin):.2f}cm | 口内行 ({inside.sum()} 行) 中位 {np.median(hmin[inside]) if inside.any() else float('nan'):.2f}cm | 口外行 中位 {np.median(hmin[~inside]):.2f}cm | 簸箕倾角中位 {np.median(tilt):.1f}°")
print(f"{T} 行; 簸箕总位移 {np.linalg.norm(pp[-1]-pp[0])*100:.1f}cm; 刷面心簸箕系 z 范围 [{B[:,2].min():.1f}, {B[:,2].max():.1f}]cm, |x|<7 且 z<4.7 (口内) 的行占 {((abs(B[:,0])<7)&(B[:,2]<4.7)).mean()*100:.0f}%")
segs = []; s = None
for t in range(1, T):
    ok = (B[t, 2] < B[t-1, 2] - 0.05) and abs(B[t, 0]) < 7 and B[t, 2] < 25
    if ok and s is None: s = t - 1
    if not ok and s is not None: segs.append((s, t - 1)); s = None
if s is not None: segs.append((s, T - 1))
segs = [(a_, b_) for a_, b_ in segs if b_ - a_ >= a.min_rows]
print("朝口段 (起行, 止行, 朝口行程cm, 起点(x,z), 止点(x,z)):")
for a_, b_ in segs: print(f"  {a_:4d}~{b_:4d}  {B[a_,2]-B[b_,2]:5.1f}cm  ({B[a_,0]:5.1f},{B[a_,2]:5.1f}) → ({B[b_,0]:5.1f},{B[b_,2]:5.1f})  刷离桌 {hmin[a_]:.1f}→{hmin[b_]:.1f}cm  止点{'过口沿' if B[b_,2] < 4.7 else '未到口沿'}")
print("每 10 行 刷面心 (x,y,z | 刷离桌cm):"); print("  " + " ".join(f"{t}:({B[t,0]:.1f},{B[t,1]:.1f},{B[t,2]:.1f}|{hmin[t]:.1f})" for t in range(0, T, 10)))
