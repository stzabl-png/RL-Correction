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
                                              DEFAULT_TORSO_DEG, URDF_PATH)
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_OPEN, GENERIC_CLOSED

HAND, TZ = "right", 0.85
OBJ = np.array([-0.0506, -0.0833, 0.8652])
R_OBJ = 0.0386
R_OPEN, R_GRIP = R_OBJ + 0.012, R_OBJ - 0.003
H_CT = OBJ[2]
CLEAR = 0.008
# ⚠ 别再用胶垫**体心**当接触点. URDF 实测: elastomer 网格 y∈[-0.0267,-0.0089],
#   接触面在 y=-0.0267; 而 PAD_LOCAL=[0,-0.0216,·] 在表面后方 4.5mm 的胶垫体内.
#   更坑的是 right_*_fingertip 这个 link **不在手指远端**, 它在 y=-0.026, 就落在
#   胶垫接触面上 (距面 0.14mm) —— 拿它当"指尖"去跟垫心比"谁先碰物体", 比的是
#   "胶垫表面 vs 胶垫体心", 表面永远更近, 这个判据在构造上就不可能满足.
PAD_LOCAL = np.array([0.0, -0.0267, 0.0013])      # 胶垫**接触面**中心
PAD_N_LOCAL = np.array([0.0, -1.0, 0.0])
MAX_GAP = np.deg2rad(float(sys.argv[1]) if len(sys.argv) > 1 else 120.0)
W_DOWN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0     # 掌心朝下权重
W_TIP = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0   # "胶垫先于指骨"权重
BASE_T = _T(np.eye(3), np.array([-0.5, 0.0, 0.0]))
# 接触点现在是胶垫**表面**, 所以目标值直接就是压入量: 胶垫是软的, 压 1mm.
SD_GRIP = 0.001      # 夹紧: 胶垫表面压进物体 1mm
SD_OPEN = -0.012     # 张开: 表面离物体 12mm, 干净让位
Z_BAND = 0.010       # 垫心高度必须落在物体中段 ±1cm 内 (物体半高 1.3cm), 不许爬到顶面
DP_CLEAR = 0.003     # 指骨刚体必须比胶垫接触面**退后** 3mm (胶垫先碰, 指骨不蹭)

# ---- 轴对称 SDF 查表 (圆环 12 扇区外半径极差仅 0.01cm, 轴对称成立) ----
# 有符号距离**必须**用带符号的: 无符号距离分不清"贴着"和"陷进去",
# 实测会把垫心优化到物体内部 4~5mm (= 穿模, 不是接触).
import os as _os                                                    # noqa: E402
_CACHE = "/home/lyh/Project/RL_Correction/tools/grasp_design/sdf_rz.npz"
if _os.path.exists(_CACHE):
    _c = np.load(_CACHE)
    R_AX, Z_AX, SDF = _c["r"], _c["z"], _c["sdf"]
else:
    import trimesh                                                  # noqa: E402
    from rl_rebuild.correction import clips as _clips, frames as _F  # noqa: E402
    from rl_rebuild.correction.ref_builders.replay_grasp import _flat_rest_quat  # noqa: E402
    _e = _clips.clip_entry("Grasp2")
    _m = trimesh.load(_e["mesh"], process=False)
    _v = np.asarray(_m.vertices)
    _m.vertices = _F.rot_apply(np.broadcast_to(_flat_rest_quat(_F.load_obj_verts(_e["mesh"])),
                                               (len(_v), 4)), _v) + OBJ
    R_AX = np.arange(0.0, 0.090, 0.0005)
    Z_AX = np.arange(OBJ[2] - 0.045, OBJ[2] + 0.065, 0.0005)
    _rr, _zz = np.meshgrid(R_AX, Z_AX, indexing="ij")
    _pts = np.column_stack([_rr.ravel() + OBJ[0], np.full(_rr.size, OBJ[1]), _zz.ravel()])
    # ⚠ 不能用 trimesh.proximity.signed_distance: 这个 mesh 有 81 万面, 它对每个查询点
    #   遍历全部三角形, 39600 点跑到 28GB RSS 还没出结果 (实测 5min+ 无输出, 逼近 OOM).
    #   改成"表面稠密采样 + KD 树最近点, 符号取最近点面法向": 近表面处精度 ~ 采样间距,
    #   而 SDF 只在接触判据 (mm 量级) 里用, 完全够.
    from scipy.spatial import cKDTree                                 # noqa: E402
    # 幅值: 表面稠密采样 + KD 树最近点 (实测与精确解逐点吻合到 0.005mm)
    _N_S = 600_000
    print(f"表面采样 {_N_S} 点建 KD 树 (mesh {len(_m.faces)} 面)…", flush=True)
    _sp, _ = trimesh.sample.sample_surface(_m, _N_S)
    _d, _ = cKDTree(_sp).query(_pts, k=1, workers=-1)
    # 符号: ⚠ **不能**用最近面的法向. 这份重建 mesh 的面法向是局部翻转的
    #   (is_winding_consistent 报 True 也没用), 实测材料内部整片被判成体外 —— 而
    #   "垫心穿进物体"被报成"在体外"正好让穿模判据失效, 优化器会把指腹压进物体里.
    #   改用射线奇偶判内外: 精确, 且只有第一次调用要建 intersector (~0.4s),
    #   之后 2000 点/ms, 整张表几乎免费.
    print(f"射线判内外 {_pts.shape[0]} 点…", flush=True)
    _in = _m.contains(_pts)
    SDF = np.where(_in, _d, -_d).reshape(_rr.shape)
    # ---- 对拍: 幅值和符号必须**分开**验, 因为没有一把尺子两样都准 ----
    #   幅值 -> trimesh.signed_distance 的绝对值 (来自 closest_point, 与法向无关, 准);
    #   符号 -> **轴对称截面真值** (顶点半径按 z 薄片取 [min,max]).
    #   为什么不能拿 signed_distance 的符号当基准: 它的符号也是取最近三角形面法向,
    #   在这份法向局部翻转的 mesh 上实测 58/60 (contains 是 60/60).
    _NC = 60
    _band = np.flatnonzero(_d < 0.02)            # 近表面 2cm 内 = 判据真正用到的带
    _cal = np.random.default_rng(0).choice(_band, _NC, replace=False)
    _ref = trimesh.proximity.signed_distance(_m, _pts[_cal])
    _V = np.asarray(_m.vertices)
    _rv = np.linalg.norm(_V[:, :2] - OBJ[:2], axis=1)
    _zv = _V[:, 2]
    _ZB = np.arange(_zv.min(), _zv.max() + 1e-9, 0.0008)
    _tru = np.zeros(_NC, bool)
    _rc = np.linalg.norm(_pts[_cal][:, :2] - OBJ[:2], axis=1)
    for _k in range(len(_ZB) - 1):
        _s = (_zv >= _ZB[_k]) & (_zv < _ZB[_k + 1])
        if _s.sum() < 20:
            continue
        _sel = (_pts[_cal][:, 2] >= _ZB[_k]) & (_pts[_cal][:, 2] < _ZB[_k + 1])
        if _sel.any():
            _tru[_sel] = (_rc[_sel] >= _rv[_s].min()) & (_rc[_sel] <= _rv[_s].max())
    _err = np.abs(np.abs(_ref) - _d[_cal])
    _same = int(((SDF.ravel()[_cal] > 0) == _tru).sum())
    print(f"[对拍] 幅值 vs trimesh closest_point ({_NC} 点, 近表面带): "
          f"最大差 {_err.max()*1000:.3f}mm  中位 {np.median(_err)*1000:.3f}mm")
    print(f"[对拍] 符号 vs 轴对称截面真值: {_same}/{_NC}  "
          f"(参考: trimesh signed_distance 自己只对 "
          f"{int(((_ref > 0) == _tru).sum())}/{_NC})")
    if _err.max() > 0.001 or _same < _NC:
        raise SystemExit(f"❌ SDF 查表对不上 (幅值最大差 {_err.max()*1000:.3f}mm, "
                         f"符号 {_same}/{_NC}) —— 接触判据不能建在这张表上")
    np.savez_compressed(_CACHE, r=R_AX, z=Z_AX, sdf=SDF)
print(f"SDF 查表 {SDF.shape}  (>0=物体内部)  范围 [{SDF.min()*100:.2f}, {SDF.max()*100:.2f}]cm")
_DR, _DZ = R_AX[1] - R_AX[0], Z_AX[1] - Z_AX[0]

# 物体外表面半径随高度 (环面, 逐 z 取最大的"体内"半径; 该高度没有材料则为 0)
R_OUT = np.array([(R_AX[SDF[:, j] > 0].max() if (SDF[:, j] > 0).any() else 0.0)
                  for j in range(SDF.shape[1])])
print(f"外表面半径随高度: 最大 {R_OUT.max()*100:.2f}cm  "
      f"(有材料的高度层 {int((R_OUT > 0).sum())}/{len(R_OUT)})")


def r_out_at(z):
    """物体在高度 z 处的外表面半径 (m)."""
    j = np.clip(((np.asarray(z) - Z_AX[0]) / _DZ).astype(int), 0, len(Z_AX) - 1)
    return R_OUT[j]


def sdf(P):
    """(N,3) 世界点 -> 有符号距离 (m), >0 = 在物体内部. 双线性插值."""
    r = np.linalg.norm(P[:, :2] - OBJ[:2], axis=1)
    fi = np.clip((r - R_AX[0]) / _DR, 0, len(R_AX) - 1.001)
    fj = np.clip((P[:, 2] - Z_AX[0]) / _DZ, 0, len(Z_AX) - 1.001)
    i, j = fi.astype(int), fj.astype(int)
    a, b = fi - i, fj - j
    return ((1 - a) * (1 - b) * SDF[i, j] + a * (1 - b) * SDF[i + 1, j]
            + (1 - a) * b * SDF[i, j + 1] + a * b * SDF[i + 1, j + 1])

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
CH_DP = [compile_chain(u.chain_between(BASE, f"{HAND}_{f}_DP"), QI) for f in FING]

# 指骨刚体 (DP) 的网格点, 用来判"先碰到物体的是胶垫还是指骨".
# 这才是"指腹接触"的正确判据 —— 之前拿 fingertip link 当指尖是错的 (见 PAD_LOCAL 注释).
import trimesh as _tm                                                  # noqa: E402
_dp_dir = _os.path.join(_os.path.dirname(URDF_PATH), "meshes",
                        f"sharpa_{HAND}", f"{HAND}_DP.STL")
_dpm = _tm.load(_dp_dir, process=False)
_rng_dp = np.random.default_rng(0)
DP_PTS = np.asarray(_dpm.vertices)[
    _rng_dp.choice(len(_dpm.vertices), min(60, len(_dpm.vertices)), replace=False)]
print(f"指骨 DP 网格采样 {len(DP_PTS)} 点 (判 胶垫 vs 指骨 谁先碰物体)")


def dp_sd(T, q):
    """每指: 指骨刚体上**最深**的有符号距离 (>0 = 已戳进物体)."""
    out = np.empty(5)
    for i, ch in enumerate(CH_DP):
        M = fk(ch, q, T)
        out[i] = sdf(DP_PTS @ M[:3, :3].T + M[:3, 3]).max()
    return out


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
        sd = sdf(P)
        gap = -(SD_GRIP if wn >= 1.0 else SD_OPEN)     # 垫心该在外壁外侧多少 (4mm/15mm)
        res += [(sd - (SD_GRIP if wn >= 1.0 else SD_OPEN)) * 150,      # 不穿模
                (1 - np.einsum("ij,ij->i", N, inw)) * 8 * wn,
                # ---- 五指环抱**外壁** ----
                # 两条约束缺一不可, 都是被实测的钻空子行为逼出来的:
                #  (a) 只给 SDF: 手指钻进圆环中间的洞 (垫心半径 0.8~2.6cm 全在外半径以内).
                #      洞里的指腹"朝中轴压"是背离材料的, 夹不住, 力封闭还会误报 ✅.
                #  (b) 只加"半径 >= 该高度外半径": 拇指/中指爬到物体**顶面上方**, 那个
                #      高度没材料 r_out=0, 约束自动满足 —— 判据被绕过去, 实测就这么解的.
                # 所以改成: 半径**两侧**都卡在 (外壁 + gap), 且垫心高度必须落在物体中段.
                (rad - r_out_at(P[:, 2]) - gap) * 150,
                np.maximum(np.abs(P[:, 2] - H_CT) - Z_BAND, 0.0) * 200]
        # 指腹判据: 先碰到物体的必须是**胶垫**, 不是指骨刚体.
        # (旧写法拿 fingertip link 当"指尖"跟垫心比, 而那个 link 就在胶垫表面上 ——
        #  比的是"胶垫表面 vs 胶垫体心", 恒不可满足, 权重提上去只会逼出穿模解.)
        res.append(np.maximum(dp_sd(T, q) - (sd - DP_CLEAR), 0.0) * W_TIP * wn)
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
    ro = r_out_at(P[:, 2])
    print(f"{'指':<8}{'垫心半径cm':>11}{'该高度外半径cm':>16}{'离桌cm':>9}{'角向°':>8}{'法向偏离°':>11}")
    for i, f in enumerate(FING):
        print(f"{f:<8}{rad[i]*100:>11.2f}{ro[i]*100:>16.2f}{(P[i,2]-TZ)*100:>9.2f}"
              f"{ang[i]:>8.1f}{al[i]:>11.1f}")
    # "在外壁上" = 高度落在物体中段 且 半径正好卡在外壁外侧一个垫厚以内
    gap_r = rad - ro
    on_wall = (np.abs(P[:, 2] - H_CT) <= Z_BAND + 0.002) & (ro > 0) & \
              (gap_r > -0.002) & (gap_r < 0.020)
    print(f"  {'每指径向间隙(垫心-外壁)cm':<28}{np.round(np.where(ro > 0, gap_r, np.nan)*100, 2).tolist()}")
    print(f"  五指环抱外壁 {int(on_wall.sum())}/5 "
          f"{'✅' if on_wall.all() else '❌ ' + ', '.join(FING[i] for i in range(5) if not on_wall[i]) + ' 不在外壁上'}")
    o = np.sort(ang)
    gp = np.diff(np.concatenate([o, [o[0] + 360]]))
    print(f"  角向间隔 {np.round(gp,1).tolist()}°  最小 {gp.min():.1f}°  最大 {gp.max():.1f}°")
    print(f"  手上最低 link 离桌 {(z.min()-TZ)*100:+.2f}cm ({(CHK+[BASE])[int(np.argmin(z))]})")
    sd = sdf(P)
    sdd = dp_sd(T, q)
    print(f"  {'胶垫接触面 有符号d(cm, 正=压入)':<30}{np.round(sd*100,2).tolist()}")
    print(f"  {'指骨最深 有符号d(cm)':<30}{np.round(sdd*100,2).tolist()}")
    pen = int((sdd > 0).sum())          # 穿模看**指骨**: 胶垫是软的, 压入 1mm 是设计如此
    touch = np.abs(sd) < 0.002          # 胶垫面离物体 2mm 内 = 真接触
    pad_first = sd > sdd
    print(f"  指骨穿模 {pen}/5 {'✅ 无' if pen == 0 else '❌ 指骨戳进物体了'};  "
          f"胶垫真接触 {int(touch.sum())}/5;  "
          f"胶垫先于指骨 {int(pad_first.sum())}/5 {'✅' if pad_first.all() else '❌'}")
    if touch.sum() >= 2:
        aa = np.sort(ang[touch])
        gg = np.diff(np.concatenate([aa, [aa[0] + 360]]))
        print(f"  **只计真实接触点** 角向 {np.round(aa,1).tolist()}°  最大缺口 {gg.max():.1f}° "
              f"(无摩擦 {'✅' if gg.max()<180 else '❌'}; 计摩擦锥后 "
              f"{max(gg.max()-2*cone,0):.1f}° {'✅' if gg.max()-2*cone<180 else '❌'})")
    v = inw[:, :2]
    a2 = np.sort(np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360)
    g2 = np.diff(np.concatenate([a2, [a2[0] + 360]]))
    print(f"  夹持力方向最大缺口 {g2.max():.1f}° (无摩擦 {'✅' if g2.max()<180 else '❌'});  "
          f"计摩擦锥±{cone:.1f}° 后 {max(g2.max()-2*cone,0):.1f}° "
          f"({'✅ 力封闭' if g2.max()-2*cone<180 else '❌'})")

np.savez("/home/lyh/Project/RL_Correction/tools/grasp_design/grip_cfg_v6.npz",
         wrist_T=T, q_open=qo, q_grip=qg, joint_names=np.array(JN), obj=OBJ,
         wrist_quat=R_to_quat(T[:3, :3]), arm_q=x[:7], arm_joints=np.array(ARM_JN))
print("\n张开指角(rad) =", np.round(qo, 4).tolist())
print("夹紧指角(rad) =", np.round(qg, 4).tolist())
