"""B1 frames.py — 单一坐标约定与变换 (与 rl_rebuild/scripts/check_alignment.py 同源).

约定链 (对 clip-11 已验证, 见 check_alignment.py 全 PASS):
  raw ViPE-gauge world --cv2zup--> z-up --recenter(table)--> 桌面局部系
手与物体施加同一刚体变换, 保持重建出的 HOI 相对几何; env_origin 由 env 再加.
"""
from __future__ import annotations

import numpy as np

CV2ZUP = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # 水平相机系 -> z-up

# MANO 21 关节序: 0 腕; 1-4 拇指; 5-8 食指; 9-12 中指; 13-16 无名指; 17-20 小指
MANO_TIPS = [4, 8, 12, 16, 20]


# ---------------- 四元数 (wxyz unless noted) ----------------
def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-8:
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4)
        q[i + 1] = 1.0
        return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w),
                     (R[1, 0] - R[0, 1]) / (4 * w)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def rot_apply(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    qc = q * np.array([1.0, -1, -1, -1])
    return quat_mul(quat_mul(q, qv), qc)[..., 1:]


# ---------------- mesh / 掌面 ----------------
def load_obj_verts(mesh_path):
    """解析 .obj 顶点 (零依赖) -> (V,3)."""
    verts = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64)


def object_aabb_obj(mesh_path):
    """物体局部系 AABB."""
    v = load_obj_verts(mesh_path)
    return v.min(axis=0), v.max(axis=0)


def sample_surface_points(mesh_path, n, seed=0):
    """FPS 最远点采样 mesh 顶点 -> (n,3) 物体规范系表面点 (确定性, 覆盖均匀).
    降采样 mesh 的顶点近似表面均匀; FPS 保证 n 点最大化铺开. 供物体几何观测用."""
    v = load_obj_verts(mesh_path)                      # (V,3) 物体系
    V = len(v)
    if V <= n:                                         # 顶点比目标少: 重复补齐
        rng = np.random.default_rng(seed)
        idx = np.concatenate([np.arange(V), rng.choice(V, n - V)]) if V else np.zeros(n, int)
        return v[idx].astype(np.float32)
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(V))]
    d = np.full(V, np.inf)
    for _ in range(n - 1):
        d = np.minimum(d, np.linalg.norm(v - v[sel[-1]], axis=1))
        sel.append(int(d.argmax()))
    return v[sel].astype(np.float32)


def sharpa_base_quat_from_joints(joints):
    """MANO 关节点 -> SharpaWave 浮动基座姿态 (T,21,3)->(T,4)wxyz.

    与 retarget_isaacsim.base_rot_from_joints 同一约定 (retarget README):
    +z = 腕->四指MCP质心(指向), +y = 食指MCP-小指MCP(拇指侧), +x = y×z(掌法向).
    右手, 无 palm_flip.
    """
    w = joints[:, 0]
    z = joints[:, [5, 9, 13, 17]].mean(1) - w
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    radial = joints[:, 5] - joints[:, 17]
    y = radial - (radial * z).sum(1, keepdims=True) * z
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-9
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=-1)          # (T,3,3) 列向量为基
    return np.stack([rotmat_to_quat(R[t]) for t in range(len(R))])


# ---------------- 平滑 / 离群剔除 / 重采样 ----------------
def _gauss_kernel(sigma):
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth_channels(x, sigma=2.0):
    """(T,D) 逐通道高斯平滑, 边界 edge-pad."""
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, "valid") for i in range(x.shape[1])], 1)


def smooth_quats(q, sigma=2.0):
    """四元数平滑: 先半球对齐(防 q/-q 跳变), 逐通道高斯, 再归一化.
    对帧间小转角是 slerp 的良好近似."""
    q = q.copy()
    for t in range(1, len(q)):
        if (q[t] * q[t - 1]).sum() < 0:
            q[t] = -q[t]
    out = smooth_channels(q, sigma)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def fix_outliers(p, mult=3.0):
    """漂移点检测+修复: 帧 t 偏离邻帧中点超过 mult×中位帧间位移 => 判为漂移,
    用邻帧线性插值顶替. 返回 (修复后, 漂移帧索引)."""
    med = np.median(np.linalg.norm(np.diff(p, axis=0), axis=1)) + 1e-9
    mid = 0.5 * (p[:-2] + p[2:])
    bad = np.zeros(len(p), bool)
    bad[1:-1] = np.linalg.norm(p[1:-1] - mid, axis=1) > mult * med
    out = p.copy()
    out[1:-1][bad[1:-1]] = mid[bad[1:-1]]
    return out, np.flatnonzero(bad)


def _slerp(q0, q1, u):
    if (d := np.dot(q0, q1)) < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def resample(x, T_new, kind="linear"):
    """时间轴均匀重采样 (T,...)->(T_new,...). kind: linear | quat(slerp) | nearest."""
    T = len(x)
    t = np.linspace(0, T - 1, T_new)
    if kind == "nearest":
        return x[np.round(t).astype(int)]
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    u = (t - lo).reshape((-1,) + (1,) * (x.ndim - 1))
    if kind == "quat":
        return np.stack([_slerp(x[a], x[b], float(w)) for a, b, w in zip(lo, hi, u.ravel())])
    return (1 - u) * x[lo] + u * x[hi]


def support_faces(verts, min_area=0.002):
    """凸包上的候选支撑面: [(面积, 单位法向), ...] 按面积降序. 法向按 0.1 精度聚类."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts)
    tri = verts[hull.simplices]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / (2 * area[:, None] + 1e-12)
    buckets = {}
    for i, k in enumerate(map(tuple, np.round(n, 1))):
        buckets.setdefault(k, []).append(i)
    out = []
    for idx in buckets.values():
        A = area[idx].sum()
        if A > min_area:
            nn = (n[idx] * area[idx, None]).sum(0)
            out.append((A, nn / np.linalg.norm(nn)))
    out.sort(key=lambda x: -x[0])
    return out


def stable_pose_projection(verts, q0_wxyz):
    """最小旋转把最接近朝下的支撑面压平到桌面 -> (修正后四元数, 修正角度deg).

    用途: 重建的初始物体姿态常带倾斜噪声 (clip-11 实测 43°), 直接放进物理会倒;
    投影后物体稳稳立在桌上, 参考轨迹姿态不动 (那是跟踪目标, 噪声由 RL 消化)."""
    faces = support_faces(verts)
    if not faces:
        return q0_wxyz, 0.0
    R = quat_to_rotmat(q0_wxyz)
    down = np.array([0.0, 0, -1])
    v = min((R @ nn for _, nn in faces),
            key=lambda w: np.arccos(np.clip(w @ down, -1, 1)))
    axis = np.cross(v, down)
    s, c = np.linalg.norm(axis), float(v @ down)
    if s < 1e-8:
        return q0_wxyz, 0.0
    angle = np.arctan2(s, c)
    axis = axis / s
    q_fix = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])
    return quat_mul(q_fix, q0_wxyz), float(np.degrees(angle))


def scene_rotation(scene_rot: str):
    """scene_rot 字符串 -> 旋转矩阵. identity: 新流水线(hawor_to_world_replay)
    已是 z-up 世界系 (run_retarget.sh 的默认 --scene-rot 1,0,0,0);
    cv2zup: 旧相机系数据兼容; 或 'w,x,y,z' 显式四元数."""
    if scene_rot == "identity":
        return np.eye(3)
    if scene_rot == "cv2zup":
        return CV2ZUP
    return quat_to_rotmat(np.array([float(x) for x in scene_rot.split(",")]))


def align_replay(joints, obj_p, obj_q_wxyz, mesh_path,
                 table_height=0.85, obj_gap=0.01, scene_rot="identity"):
    """scene_rot + 落桌 recenter, 手物同一变换. 返回 (joints', obj_p', obj_q', info)."""
    Rs = scene_rotation(scene_rot)
    qs = rotmat_to_quat(Rs)
    T = len(obj_p)
    obj_p_r = obj_p @ Rs.T
    obj_q_r = quat_mul(np.broadcast_to(qs, (T, 4)), obj_q_wxyz)
    joints_r = joints @ Rs.T

    lo, hi = object_aabb_obj(mesh_path)
    # 落桌高度必须用真实顶点最低点: AABB 角点对圆柱/圆弧面是空气,
    # 会把物体悬空数厘米 (躺置圆柱实测 4.6cm), 落地冲击直接滚走
    verts = load_obj_verts(mesh_path)
    v0 = rot_apply(np.broadcast_to(obj_q_r[0], (len(verts), 4)), verts) + obj_p_r[0]
    shift = np.array([-obj_p_r[0, 0], -obj_p_r[0, 1],
                      table_height + obj_gap - v0[:, 2].min()])
    info = {"aabb_lo": lo, "aabb_hi": hi, "shift": shift,
            "half_diag": np.linalg.norm(hi - lo) / 2}
    return joints_r + shift, obj_p_r + shift, obj_q_r, info
