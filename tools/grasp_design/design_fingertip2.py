"""指尖环抓构型求解 (预编译链的快速 FK).

变量: 腕位姿(6, 张开/夹紧**共享** —— 手臂不动只动手指) + 张开指角(22) + 夹紧指角(22)
残差: 每指胶垫 [半径-目标, 高度-目标, 法向对准圆心] × 两态 + 角向分散 + 桌面 + 正则
"""
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
from rl_rebuild.correction.kinematics import Urdf, _T
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_OPEN, GENERIC_CLOSED

HAND, TZ = "right", 0.85
OBJ = np.array([-0.0506, -0.0833, 0.8652])
R_OBJ, H_OBJ = 0.0386, 0.0262
R_OPEN, R_GRIP = R_OBJ + 0.012, R_OBJ - 0.003
H_CT = OBJ[2]
CLEAR = 0.005
PAD_LOCAL = np.array([0.0, -0.0216, 0.0013])
PAD_N_LOCAL = np.array([0.0, -1.0, 0.0])
SPREAD_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0   # 允许的最大角向缺口

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
PAD_LINKS = [f"{HAND}_{f}_elastomer" for f in FING]
CHK = [l for f in FING for s in ("PP", "MP", "DP", "elastomer", "fingertip")
       for l in [f"{HAND}_{f}_{s}"] if l in u.parent_joint]


def compile_chain(link):
    out = []
    for n in u.chain_between(BASE, link):
        j = u.joints[n]
        k = j["axis"]
        K = None if j["type"] == "fixed" else np.array(
            [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], float)
        out.append((j["origin"], K, QI.get(n, -1)))
    return out


CH_PAD = [compile_chain(l) for l in PAD_LINKS]
CH_CHK = [compile_chain(l) for l in CHK]
I3 = np.eye(3)


def fk(chain, qv, T0):
    T = T0.copy()
    for org, K, qi in chain:
        T = T @ org
        if K is not None:
            a = qv[qi] if qi >= 0 else 0.0
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


def rotvec_to_R(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return I3.copy()
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return I3 + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def R_to_quat(R):
    w = np.sqrt(max(1 + np.trace(R), 0)) / 2
    if w < 1e-8:
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4); q[i + 1] = 1.0
        return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def unpack(x):
    return _T(rotvec_to_R(x[3:6]), x[:3]), x[6:28], x[28:50]


PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
SPREAD = np.deg2rad(SPREAD_DEG)


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
        res.append(np.maximum(gaps - SPREAD, 0.0) * 4.0)      # 罚超过目标的缺口
        res.append(np.minimum(link_zs(T, q) - (TZ + CLEAR), 0) * 200)
    res.append((x[6:28] - x[28:50]) * 0.02)
    return np.concatenate(res)


BND = (np.concatenate([[-1, -1, TZ], [-8] * 3, LO, LO]),
       np.concatenate([[1, 1, TZ + 0.6], [8] * 3, UP, UP]))

rng = np.random.default_rng(0)
Rd = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)


def seed_for(yaw, k):
    c, s = np.cos(yaw), np.sin(yaw)
    R0 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ Rd
    th = np.arccos(np.clip((np.trace(R0) - 1) / 2, -1, 1))
    rv = np.zeros(3) if th < 1e-8 else th / (2 * np.sin(th)) * np.array(
        [R0[2, 1] - R0[1, 2], R0[0, 2] - R0[2, 0], R0[1, 0] - R0[0, 1]])
    return np.concatenate([OBJ + [0, 0, 0.15 + 0.03 * k], rv,
                           np.clip(GENERIC_OPEN + rng.normal(0, .08, 22), LO, UP),
                           np.clip(GENERIC_CLOSED * .4 + rng.normal(0, .08, 22), LO, UP)])


def run(seed, nfev):
    return least_squares(residual, np.clip(seed, *BND), bounds=BND,
                         xtol=1e-11, ftol=1e-11, max_nfev=nfev)


# 粗筛: 12 个 yaw 各一次短跑
coarse = []
for yaw in np.deg2rad(np.arange(0, 360, 30)):
    r = run(seed_for(yaw, 0), 60)
    coarse.append((r.cost, np.degrees(yaw), r.x))
    print(f"  [粗] yaw={np.degrees(yaw):5.0f}°  cost={r.cost:.4f}", flush=True)
coarse.sort(key=lambda t: t[0])
# 精修: 最好的 3 个
best = None
for cst, yaw, x0 in coarse[:3]:
    r = run(x0, 500)
    print(f"  [精] yaw={yaw:5.0f}°  {cst:.4f} -> {r.cost:.4f}", flush=True)
    if best is None or r.cost < best.cost:
        best = r

x = best.x
T, qo, qg = unpack(x)
print("\n" + "=" * 78)
print(f"允许最大缺口 {SPREAD_DEG:.0f}°   最优 cost = {best.cost:.4f}")
print(f"手基座 {BASE} 位置 {np.round(T[:3,3],4).tolist()} (离桌 {(T[2,3]-TZ)*100:.2f}cm)")
print(f"姿态 wxyz {np.round(R_to_quat(T[:3,:3]),4).tolist()}   掌心指向(+z) {np.round(T[:3,2],3).tolist()}")
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
        print(f"{f:<8}{rad[i]*100:>11.2f}{(z[i] if False else P[i,2]-TZ)*100:>9.2f}"
              f"{ang[i]:>8.1f}{al[i]:>11.1f}")
    o = np.sort(ang)
    g = np.diff(np.concatenate([o, [o[0] + 360]]))
    print(f"  角向间隔 {np.round(g,1).tolist()}°  最小 {g.min():.1f}°  最大 {g.max():.1f}°")
    print(f"  手上最低 link 离桌 {(z.min()-TZ)*100:+.2f}cm ({(CHK+[BASE])[int(np.argmin(z))]})")
    v = inw[:, :2]
    a2 = np.sort(np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360)
    g2 = np.diff(np.concatenate([a2, [a2[0] + 360]]))
    MU = 0.5                                   # ObjectSemantics.friction
    cone = np.degrees(np.arctan(MU))
    print(f"  夹持力方向最大缺口 {g2.max():.1f}° (无摩擦判据: "
          f"{'✅' if g2.max()<180 else '❌'})")
    print(f"  计入摩擦锥 ±{cone:.1f}° (μ={MU}) 后有效缺口 {max(g2.max()-2*cone,0):.1f}° "
          f"({'✅ 水平力封闭' if g2.max()-2*cone<180 else '❌ 仍可逃逸'})")

np.savez("/home/lyh/Project/RL_Correction/tools/grasp_design/grip_cfg.npz",
         wrist_T=T, q_open=qo, q_grip=qg, joint_names=np.array(JN), obj=OBJ,
         wrist_quat=R_to_quat(T[:3, :3]))
print("\n张开指角(rad) =", np.round(qo, 4).tolist())
print("夹紧指角(rad) =", np.round(qg, 4).tolist())
