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
SPREAD_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 72.0

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
        res += [(rad - r_t) * 30, (P[:, 2] - H_CT) * 30,
                (1 - np.einsum("ij,ij->i", N, inw)) * 3 * wn]
        ang = np.arctan2(d[:, 1], d[:, 0])
        gaps = np.array([abs(np.angle(np.exp(1j * (ang[i] - ang[j])))) for i, j in PAIRS])
        res.append(np.maximum(SPREAD - gaps, 0.0) * 1.0)
        res.append(np.minimum(link_zs(T, q) - (TZ + CLEAR), 0) * 200)
    res.append((x[6:28] - x[28:50]) * 0.02)
    return np.concatenate(res)


