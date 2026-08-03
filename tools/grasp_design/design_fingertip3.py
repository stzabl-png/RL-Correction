"""指尖环抓构型 v3 — 变量改成**手臂关节角**, 腕位姿由 FK 生成, 天然可达.

v2 的教训: 只优化腕位姿会解出手臂够不到的位姿 (实测位置差 10.4cm, 3 个关节顶限位).
变量: 右臂 7 关节 + 张开指角(22) + 夹紧指角(22) = 51
      (躯干锁在训练 env 站姿 45/90/0°)
残差: 每指胶垫 [半径, 高度, 法向对准圆心] × 两态 + 最大角向缺口 + 桌面 + 掌心朝下 + 正则
"""
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
from rl_rebuild.correction.kinematics import (Urdf, _T, ArmIK, quat_to_R,
                                              DEFAULT_TORSO_DEG)
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_OPEN, GENERIC_CLOSED

HAND, TZ = "right", 0.85
OBJ = np.array([-0.0506, -0.0833, 0.8652])
R_OBJ = 0.0386
R_OPEN, R_GRIP = R_OBJ + 0.012, R_OBJ - 0.003
H_CT = OBJ[2]
CLEAR = 0.005
PAD_LOCAL = np.array([0.0, -0.0216, 0.0013])
PAD_N_LOCAL = np.array([0.0, -1.0, 0.0])
MAX_GAP = np.deg2rad(float(sys.argv[1]) if len(sys.argv) > 1 else 120.0)
W_DOWN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0     # 掌心朝下权重
BASE_T = _T(np.eye(3), np.array([-0.5, 0.0, 0.0]))

FING = ("thumb", "index", "middle", "ring", "pinky")
JN = [n.replace("right_", f"{HAND}_") for n in
      ["right_thumb_CMC_FE","right_thumb_CMC_AA","right_thumb_MCP_FE","right_thumb_MCP_AA",
       "right_thumb_IP","right_index_MCP_FE","right_index_MCP_AA","right_index_PIP",
       "right_index_DIP","right_middle_MCP_FE","right_middle_MCP_AA","right_middle_PIP",
       "right_middle_DIP","right_ring_MCP_FE","right_ring_MCP_AA","right_ring_PIP",
       "right_ring_DIP","right_pinky_CMC","right_pinky_MCP_FE","right_pinky_MCP_AA",
       "right_pinky_PIP","right_pinky_DIP"]]
QI = {n: i for i, n in enumerate(JN)}
BASE = f"{HAND}_hand_C_MC"
u = Urdf()
LO = np.array([u.joints[n]["lower"] for n in JN])
UP = np.array([u.joints[n]["upper"] for n in JN])
I3 = np.eye(3)

# ---- 手臂链 (root -> hand_C_MC), 躯干角写死, 臂 7 关节为变量 ----
TORSO = {k: np.deg2rad(v) for k, v in DEFAULT_TORSO_DEG.items()}
ARM_CHAIN_J = u.chain_to(BASE)
ARM_JN = [n for n in ARM_CHAIN_J if n.startswith("R_arm_j") and u.joints[n]["type"] != "fixed"]
A_LO = np.array([u.joints[n]["lower"] for n in ARM_JN])
A_UP = np.array([u.joints[n]["upper"] for n in ARM_JN])
AQI = {n: i for i, n in enumerate(ARM_JN)}
print(f"手臂变量关节 {ARM_JN}")


def _K(ax):
    return np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], float)


def compile_chain(joints, qi_map):
    out = []
    for n in joints:
        j = u.joints[n]
        K = None if j["type"] == "fixed" else _K(j["axis"])
        out.append((j["origin"], K, qi_map.get(n, -1), float(TORSO.get(n, 0.0))))
    return out


CH_ARM = compile_chain(ARM_CHAIN_J, AQI)
CH_PAD = [compile_chain(u.chain_between(BASE, f"{HAND}_{f}_elastomer"), QI) for f in FING]
CHK = [l for f in FING for s in ("PP", "MP", "DP", "elastomer", "fingertip")
       for l in [f"{HAND}_{f}_{s}"] if l in u.parent_joint]
CH_CHK = [compile_chain(u.chain_between(BASE, l), QI) for l in CHK]


def fk(chain, qv, T0):
    T = T0.copy()
    for org, K, qi, fx in chain:
        T = T @ org
        if K is not None:
            a = qv[qi] if qi >= 0 else fx
            T[:3, :3] = T[:3, :3] @ (I3 + np.sin(a) * K + (1 - np.cos(a)) * (K @ K))
    return T


def pads(T, q):
    P, N = np.empty((5, 3)), np.empty((5, 3))
    for i, ch in enumerate(CH_PAD):
        M = fk(ch, q, T)
        P[i] = M[:3, 3] + M[:3, :3] @ PAD_LOCAL
        N[i] = M[:3, :3] @ PAD_N_LOCAL
    return P, N


def link_zs(T, q):
    return np.array([fk(ch, q, T)[2, 3] for ch in CH_CHK] + [T[2, 3]])


def unpack(x):
    return fk(CH_ARM, x[:7], BASE_T), x[7:29], x[29:51]


def residual(x):
    T, qo, qg = unpack(x)
    res = []
    for q, r_t, wn in ((qo, R_OPEN, 0.3), (qg, R_GRIP, 1.0)):
        P, N = pads(T, q)
        d = P[:, :2] - OBJ[:2]
        rad = np.linalg.norm(d, axis=1) + 1e-9
        inw = np.column_stack([-d / rad[:, None], np.zeros(5)])
        res += [(rad - r_t) * 80, (P[:, 2] - H_CT) * 30,
                (1 - np.einsum("ij,ij->i", N, inw)) * 3 * wn]
        ang = np.sort(np.arctan2(d[:, 1], d[:, 0]))
        gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
        res.append(np.maximum(gaps - MAX_GAP, 0.0) * 4.0)
        res.append(np.minimum(link_zs(T, q) - (TZ + CLEAR), 0) * 200)
    res.append([(1.0 + T[2, 2]) * W_DOWN])            # 掌心 +z 朝下 -> T[2,2] = -1
    res.append((x[7:29] - x[29:51]) * 0.02)
    return np.concatenate([np.atleast_1d(np.asarray(r, float)) for r in res])


BND = (np.concatenate([A_LO, LO, LO]), np.concatenate([A_UP, UP, UP]))
ikA = ArmIK("right", urdf=u, base_pos=(-0.5, 0.0, 0.0))
rng = np.random.default_rng(0)
Rd = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)

# 种子: 用 ArmIK 解"腕在物体正上方 h、掌心朝下、绕竖直轴 yaw"的位姿
seeds = []
for yaw in np.deg2rad(np.arange(0, 360, 30)):
    c, s = np.cos(yaw), np.sin(yaw)
    R0 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ Rd
    for h in (0.13, 0.16):
        r = ikA.solve_best(OBJ + [0, 0, h], R0,
                           [ikA.q_default] + [rng.uniform(ikA.lower, ikA.upper)
                                              for _ in range(6)],
                           pos_tol=0.01, rot_tol=np.deg2rad(25))
        seeds.append((np.degrees(yaw), h, r["ok"], r["pos_err"], r["q"]))
n_ok = sum(s[2] for s in seeds)
print(f"种子: {n_ok}/{len(seeds)} 个'掌心朝下悬停'位姿手臂可达")

coarse = []
for yd, h, ok, pe, qa in seeds:
    x0 = np.concatenate([qa, np.clip(GENERIC_OPEN + rng.normal(0, .08, 22), LO, UP),
                         np.clip(GENERIC_CLOSED * .4 + rng.normal(0, .08, 22), LO, UP)])
    r = least_squares(residual, np.clip(x0, *BND), bounds=BND, xtol=1e-11, ftol=1e-11,
                      max_nfev=60)
    coarse.append((r.cost, yd, h, ok, r.x))
    print(f"  [粗] yaw={yd:5.0f}° h={h:.2f} 种子IK{'✅' if ok else '❌'}({pe*100:4.1f}cm)"
          f"  cost={r.cost:.4f}", flush=True)
coarse.sort(key=lambda t: t[0])
best = None
for cst, yd, h, ok, x0 in coarse[:3]:
    r = least_squares(residual, x0, bounds=BND, xtol=1e-11, ftol=1e-11, max_nfev=500)
    print(f"  [精] yaw={yd:5.0f}° h={h:.2f}  {cst:.4f} -> {r.cost:.4f}", flush=True)
    if best is None or r.cost < best.cost:
        best = r

x = best.x
T, qo, qg = unpack(x)


def R_to_quat(R):
    w = np.sqrt(max(1 + np.trace(R), 0)) / 2
    if w < 1e-8:
        i = int(np.argmax(np.diag(R))); q = np.zeros(4); q[i + 1] = 1.0; return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


print("\n" + "=" * 78)
print(f"最优 cost = {best.cost:.4f}   (最大缺口目标 {np.degrees(MAX_GAP):.0f}°)")
print(f"右臂关节角(deg) = {np.round(np.degrees(x[:7]),2).tolist()}")
print(f"  顶限位: {[ARM_JN[i] for i in range(7) if min(abs(x[i]-A_LO[i]),abs(x[i]-A_UP[i]))<1e-3]}")
print(f"手基座 {BASE} 位置 {np.round(T[:3,3],4).tolist()} (离桌 {(T[2,3]-TZ)*100:.2f}cm)")
print(f"姿态 wxyz {np.round(R_to_quat(T[:3,:3]),4).tolist()}")
print(f"掌心指向(+z) {np.round(T[:3,2],3).tolist()}  "
      f"与竖直向下夹角 {np.degrees(np.arccos(np.clip(-T[2,2],-1,1))):.1f}°")
MU = 0.5
cone = np.degrees(np.arctan(MU))
for nm, q, r_t in (("张开 PreGrasp", qo, R_OPEN), ("夹紧 Grip", qg, R_GRIP)):
    P, N = pads(T, q)
    d = P[:, :2] - OBJ[:2]
    rad = np.linalg.norm(d, axis=1)
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
    inw = np.column_stack([-d / rad[:, None], np.zeros(5)])
    al = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", N, inw), -1, 1)))
    z = link_zs(T, q)
    print(f"\n--- {nm} ---  目标半径 {r_t*100:.2f}cm  目标高度(离桌) {(H_CT-TZ)*100:.2f}cm")
    print(f"{'指':<8}{'垫心半径cm':>11}{'离桌cm':>9}{'角向°':>8}{'法向偏离°':>11}")
    for i, f in enumerate(FING):
        print(f"{f:<8}{rad[i]*100:>11.2f}{(P[i,2]-TZ)*100:>9.2f}{ang[i]:>8.1f}{al[i]:>11.1f}")
    o = np.sort(ang)
    gp = np.diff(np.concatenate([o, [o[0] + 360]]))
    print(f"  角向间隔 {np.round(gp,1).tolist()}°  最小 {gp.min():.1f}°  最大 {gp.max():.1f}°")
    print(f"  手上最低 link 离桌 {(z.min()-TZ)*100:+.2f}cm ({(CHK+[BASE])[int(np.argmin(z))]})")
    v = inw[:, :2]
    a2 = np.sort(np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360)
    g2 = np.diff(np.concatenate([a2, [a2[0] + 360]]))
    print(f"  夹持力方向最大缺口 {g2.max():.1f}° (无摩擦 {'✅' if g2.max()<180 else '❌'});  "
          f"计摩擦锥±{cone:.1f}° 后 {max(g2.max()-2*cone,0):.1f}° "
          f"({'✅ 力封闭' if g2.max()-2*cone<180 else '❌'})")

np.savez("/home/lyh/Project/RL_Correction/tools/grasp_design/grip_cfg.npz",
         wrist_T=T, q_open=qo, q_grip=qg, joint_names=np.array(JN), obj=OBJ,
         wrist_quat=R_to_quat(T[:3, :3]), arm_q=x[:7], arm_joints=np.array(ARM_JN))
print("\n张开指角(rad) =", np.round(qo, 4).tolist())
print("夹紧指角(rad) =", np.round(qg, 4).tolist())
