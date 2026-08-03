"""严格接触判定: 胶垫到**真实 mesh 表面**的距离 (不是圆柱近似).

圆环是环面, 外半径随高度变化 —— 用 r=3.86cm 当外壁会高估接触.
这里对每个胶垫算: 垫心到最近 mesh 顶点的距离, 以及沿垫法向到表面的距离.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
SP = "/home/lyh/Project/RL_Correction/tools/grasp_design"
from rl_rebuild.correction import clips, frames as F
from rl_rebuild.correction.kinematics import Urdf, _T

TZ = 0.85
PAD_LOCAL = np.array([0.0, -0.0216, 0.0013])
PAD_N_LOCAL = np.array([0.0, -1.0, 0.0])
PAD_HALF = 0.0072                      # 胶垫自身半厚 (elastomer 包围盒 ~1.44cm)
FING = ("thumb", "index", "middle", "ring", "pinky")

GRIP = os.environ.get("GRIP_NPZ", "grip_cfg_v6.npz")
print(f"构型文件: {GRIP}")
g = np.load(f"{SP}/{GRIP}", allow_pickle=True)
T = g["wrist_T"]; OBJ = g["obj"]
JN = [str(s) for s in g["joint_names"]]
u = Urdf()

e = clips.clip_entry("Grasp2")
v = F.load_obj_verts(e["mesh"])
from rl_rebuild.correction.ref_builders.replay_grasp import _flat_rest_quat
rq = _flat_rest_quat(v)
V = F.rot_apply(np.broadcast_to(rq, (len(v), 4)), v) + OBJ      # 世界系顶点
print(f"物体 mesh {len(V)} 顶点, 世界 z [{V[:,2].min():.4f}, {V[:,2].max():.4f}]")
print(f"外半径随高度变化 (环面!):")
r_all = np.linalg.norm(V[:, :2] - OBJ[:2], axis=1)
for z0 in np.arange(0.0, 0.026, 0.005):
    m = (V[:, 2] - V[:, 2].min() >= z0) & (V[:, 2] - V[:, 2].min() < z0 + 0.005)
    if m.sum():
        print(f"    h={z0*100:4.1f}-{(z0+0.005)*100:4.1f}cm  外半径 {r_all[m].max()*100:5.2f}cm  "
              f"内半径 {r_all[m].min()*100:5.2f}cm")

for tag, key in (("张开 PreGrasp", "q_open"), ("夹紧 Grip", "q_grip")):
    q = {n: float(x) for n, x in zip(JN, g[key])}
    print(f"\n--- {tag} ---")
    print(f"{'指':<8}{'垫心到最近顶点cm':>18}{'指尖到最近顶点cm':>18}"
          f"{'沿法向到表面cm':>17}{'接触?':>8}{'谁先碰':>8}")
    n_ct = 0
    n_pad_first = 0
    ct_ang = []
    for f in FING:
        M = u.link_pose(f"right_{f}_elastomer", q, T, "right_hand_C_MC")
        p = M[:3, 3] + M[:3, :3] @ PAD_LOCAL
        n = M[:3, :3] @ PAD_N_LOCAL
        d = np.linalg.norm(V - p, axis=1)
        dmin = d.min()
        # 指尖端 (fingertip link 原点): 指腹接触的判据是它必须比垫心**更远**离物体
        Mt = u.link_pose(f"right_{f}_fingertip", q, T, "right_hand_C_MC")
        dtip = np.linalg.norm(V - Mt[:3, 3], axis=1).min()
        # 沿法向: 取法向前方 (投影>0) 的顶点里最近的
        proj = (V - p) @ n
        lat = np.linalg.norm((V - p) - proj[:, None] * n[None], axis=1)
        m = (proj > -0.002) & (lat < 0.010)
        dn = proj[m].min() if m.any() else np.inf
        hit = dmin < PAD_HALF + 0.002
        pad_first = dmin < dtip
        n_ct += hit
        n_pad_first += bool(hit and pad_first)
        if hit:
            dd = p[:2] - OBJ[:2]
            ct_ang.append(np.degrees(np.arctan2(dd[1], dd[0])) % 360)
        print(f"{f:<8}{dmin*100:>18.2f}{dtip*100:>18.2f}"
              f"{(dn*100 if np.isfinite(dn) else float('nan')):>17.2f}"
              f"{'✅' if hit else '❌':>8}{'指腹' if pad_first else '指尖⚠':>8}")
    print(f"  真实接触指数 {n_ct}/5   (判据: 垫心到表面 < 垫半厚 {PAD_HALF*100:.2f}cm + 2mm 容差)")
    print(f"  其中**用指腹**接触 {n_pad_first}/{n_ct} "
          f"{'✅' if n_ct and n_pad_first == n_ct else '❌ 有指尖先碰到物体'}")
    if len(ct_ang) >= 2:
        a = np.sort(np.array(ct_ang))
        gp = np.diff(np.concatenate([a, [a[0] + 360]]))
        MU = 0.5
        cone = np.degrees(np.arctan(MU))
        print(f"  接触点角向 {np.round(a,1).tolist()}°  最大缺口 {gp.max():.1f}°")
        print(f"  水平力封闭: 无摩擦 {'✅' if gp.max()<180 else '❌'};  "
              f"计摩擦锥±{cone:.1f}° 后缺口 {max(gp.max()-2*cone,0):.1f}° "
              f"{'✅' if gp.max()-2*cone<180 else '❌'}")
    # 垂直承重: 摩擦力 vs 重力
    if n_ct:
        print(f"  垂直承重: 物体 0.1kg = 0.98N; μ=0.5 下每指需法向力 "
              f"{0.98/(0.5*n_ct):.2f}N ({n_ct} 指分担)")
