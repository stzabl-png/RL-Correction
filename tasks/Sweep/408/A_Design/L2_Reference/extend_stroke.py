"""把 Sweep2 格式母带里指定的朝口笔画沿其水平方向**延长** ext_cm (2026-09-14, 用户裁定 175: "延长3cm 改母带").
只动扫把 (obj_1) 的位置 + 重解右臂 IK (手 = 物体 ∘ T_oh, 与构带器同式); 簸箕/左臂/人手臂 q 不动。
位移剖面: 笔画 [a,b] 内 smoothstep 从 0 升到 ext, 之后 blend 行内 smoothstep 回 0 (连续, 每行增量 ≤ ext/行数)。
用法: python extend_stroke.py --src v4.npz --out v5.npz --strokes 46:78,145:174 --ext_cm 3 [--blend 15]"""
import argparse, hashlib, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../../../.."))
sys.path.insert(0, ROOT)
from rl_rebuild.correction.kinematics import ArmIK
p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
p.add_argument("--strokes", default="");
p.add_argument("--lateral", default="", help="a:b:cm[,...] 该行段内扫把沿簸箕系 +x (横向) 平移 cm, 两侧各 blend 行 smoothstep 过渡 (2026-09-14: 175 第二笔偏左 3cm, 出走廊)"); p.add_argument("--ext_cm", type=float, default=3.0); p.add_argument("--blend", type=int, default=15)
p.add_argument("--prior", default=os.path.join(ROOT, "tasks/pregrasp/priors/Sweep408_broom.npz"))
p.add_argument("--ik_pos_tol_mm", type=float, default=0.2); p.add_argument("--ik_rot_tol_deg", type=float, default=0.29); a = p.parse_args()
z = np.load(a.src, allow_pickle=True); out = {k: z[k] for k in z.files}
q2R = lambda q: Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix()
P = np.asarray(z["obj_pos_1"], np.float64).copy(); Q = np.asarray(z["obj_quat_1"], np.float64); RQ = np.asarray(z["right_q"], np.float64).copy(); T = len(P)
g = np.asarray(np.load(a.prior)["grasp"], np.float64); p_oh, R_oh = g[:3], q2R(g[3:7])
ik = ArmIK("right", anchor_link="arm_center", anchor_T=np.asarray(z["anchor_T"], np.float64))
# 自检: 原母带 FK(right_q) 应 = 物体 ∘ T_oh
res = [np.linalg.norm(ik.fk(RQ[t])[0] - (P[t] + q2R(Q[t]) @ p_oh)) for t in range(0, T, max(1, T // 10))]
print(f"[ext] 自检 FK(right_q) vs 物体∘T_oh: 位置残差 中位 {np.median(res)*100:.2f}cm max {np.max(res)*100:.2f}cm (应为毫米级)")
assert np.max(res) < 0.01, "T_oh 约定不对, 停"
smooth = lambda x: (lambda u: u * u * (3 - 2 * u))(np.clip(x, 0, 1))
d = np.zeros(T); dirs = {}
for seg in [x for x in a.strokes.split(",") if x]:
    s0, s1 = [int(v) for v in seg.split(":")]
    u = P[s1] - P[s0]; u[2] = 0; u /= max(np.linalg.norm(u), 1e-9); dirs[(s0, s1)] = u
    for t in range(s0, min(s1 + a.blend, T - 1) + 1):
        w = smooth((t - s0) / max(s1 - s0, 1)) if t <= s1 else 1.0 - smooth((t - s1) / a.blend)
        P[t] += (a.ext_cm / 100.0) * w * u; d[t] = max(d[t], w)
PQ = np.asarray(z["obj_quat_0"], np.float64)
for seg in ([x for x in a.lateral.split(",") if x] if a.lateral else []):
    s0, s1, cm = seg.split(":"); s0, s1, cm = int(s0), int(s1), float(cm)
    for t in range(max(s0 - a.blend, 0), min(s1 + a.blend, T - 1) + 1):
        w = smooth((t - (s0 - a.blend)) / a.blend) if t < s0 else (1.0 if t <= s1 else 1.0 - smooth((t - s1) / a.blend))
        xdir = q2R(PQ[t])[:, 0]; xdir[2] = 0; xdir /= max(np.linalg.norm(xdir), 1e-9)      # 簸箕系 +x 的水平方向
        P[t] += (cm / 100.0) * w * xdir; d[t] = max(d[t], w)
    print(f"[ext] 横移 行 {s0}~{s1} +{cm}cm (簸箕系 +x), 过渡 {a.blend} 行")
mod = np.flatnonzero(d > 0); lo, hi = int(mod.min()), int(mod.max())
print(f"[ext] 笔画 {list(dirs)} 各延长 {a.ext_cm}cm (水平方向), 受影响行 {lo}~{hi} ({len(mod)} 行), 每行最大增量 {np.max(np.abs(np.diff(P,axis=0)))*1000:.1f}mm")
# 重解右臂 IK (连续, 从 lo-1 的原解出发)
q = RQ[max(lo - 1, 0)].copy(); perr = []; rerr = []
for t in range(lo, hi + 1):
    tgt_p = P[t] + q2R(Q[t]) @ p_oh; tgt_R = q2R(Q[t]) @ R_oh
    r = ik.solve(tgt_p, tgt_R, q0=q, iters=200, pos_tol=a.ik_pos_tol_mm / 1000.0, rot_tol=np.radians(a.ik_rot_tol_deg))
    q = r["q"]; RQ[t] = q; perr.append(r["pos_err"]); rerr.append(np.degrees(r["rot_err"]))
step = np.degrees(np.abs(np.diff(RQ, axis=0))).max(axis=1)
print(f"[ext] 右臂 IK: 位置误差 中位 {np.median(perr)*100:.3f}cm max {np.max(perr)*100:.3f}cm | 姿态 max {np.max(rerr):.2f}° | 关节单步 max {step.max():.2f}° (行 {int(step.argmax())}) | 接回原解处 (行 {hi}→{hi+1}) 关节差 {np.degrees(np.abs(RQ[hi+1]-RQ[hi])).max():.2f}°")
out["obj_pos_1"] = P.astype(np.float32); out["right_q"] = RQ.astype(np.float32)
out["meta"] = np.asarray(f"{str(z['meta'])} | extend_stroke: strokes [{a.strokes}] +{a.ext_cm}cm, lateral [{a.lateral}], blend {a.blend}, rows {lo}~{hi} right_q re-IK")
np.savez(a.out, **out); print(f"[ext] → {a.out} sha {hashlib.sha256(open(a.out,'rb').read()).hexdigest()[:16]}")
