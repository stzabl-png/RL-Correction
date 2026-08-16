"""B1 frames.py — 单一坐标约定与变换。

约定链:
  raw ViPE-gauge world --cv2zup--> z-up --recenter(table)--> 桌面局部系

⚠ **摆放约定已于 2026-08-07 变更, 本文件只提供变换原语, 不再定义摆放。**
旧约定是"手与物体施加同一刚体变换, 保持重建出的 HOI 相对几何", 配套的
`rl_rebuild/scripts/check_alignment.py` 会断言该相对几何不变 —— **两者都已删除**。
现行约定(唯一): `ref_builders/replay_grasp.py` 的三步 ——
  ① 相机(人头) xy 对齐机器人 ZED 光心;
  ② 手轨迹整体抬升, 全程离桌 clearance;
  ③ **物体 XY 听手**: 交互开始帧手的抓取锚点定物体 xy, Z 贴桌。
第③步**故意改变**手物相对几何 —— 重建的物体 track 在交互期被手遮挡、噪声大,
而手部轨迹是**明显更可信的通道**(ARCTIC 真值实测, 2026-08-12 修正索引后:
**锚后**误差 手 74mm vs 物体 123mm —— 手好 **1.7 倍**; 按"每 1mm 真实位移产生多少误差"
归一后 0.53 vs 0.95, 同样 1.8 倍。且手在接触后走的距离是物体的 2.7 倍, 所以毫米数比法
其实**低估**了手的优势。绝对口径 手 180mm vs 物 264mm)。
⚠ 别用"手 58mm vs 物 157mm / 好 2.7 倍" —— 那是拿两个**不同池**的中位数相比
(30 个手-side 条目 vs 11 条物体 take), 口径不匹配。
⚠ 早期版本记的"手 198mm / 相对 393mm"是错的: 当时用视频帧号索引 hand_*, 而 world_fused
的 hand_* 按 ARCTIC 30fps 标注帧索引(长度是 object 的 2 倍), 取到了错的时刻。
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


def sharpa_base_quat_from_joints(joints, hand="right"):
    """MANO 关节点 -> SharpaWave 浮动基座姿态 (T,21,3)->(T,4)wxyz.

    与 retarget_isaacsim.base_rot_from_joints 同一约定 (retarget README):
    +z = 腕->四指MCP质心(指向), +y = 食指MCP-小指MCP(拇指侧), +x = y×z(掌法向).

    ⚠ **左手必须传 hand="left"**(2026-08-15 修)。下面的公式对两只手都把 +y 定成
    "食指MCP->小指MCP"即**解剖学上的拇指侧**, 而 SharpaWave 的 URDF 左手基座系是右手
    系镜像的: 实测 `grasp_center_local` 左 [4.19,+0.86,14.40] / 右 [4.19,-0.86,14.40],
    **只有 y 反号, x 同号**。于是人手左腕系与机器人左手基座系差一个**绕 z 的 180°**
    (retarget 侧一直用 `--palm-flip left` 补它, 本函数从前没有, 因为历史上只跑右手)。

    判据与实测(pour/17 抓握窗, 判据 = 机器人合拢中心落到人五指尖质心的残差):
        左手  不翻 6.20cm / **绕z翻 2.37cm** / 绕x翻 29.3 / 绕y翻 28.7
        右手  **不翻 1.28cm** / 绕z翻 8.83 / 绕x翻 27.7 / 绕y翻 29.0
    不预设翻转直接解最优修正旋转: 左 158.1° 绕 [0.32,-0.13,0.94](≈绕z 180°),
    右 26.6° —— 右手那 26.6° 是"机器人手≠人手"的形状残差基线, 左手 = 它 + 一个 z180。
    翻正后两手残差同量级(2.4 vs 1.3cm), 即只剩形状残差。

    默认仍是 "right" 且此时**恒等**, 所以历史上所有右手 run(pp0/Grasp0/冠军配方)
    的数值不受本次改动影响。
    """
    if hand not in ("left", "right"):
        raise ValueError(f"hand 必须是 'left'/'right', 收到 {hand!r}")
    w = joints[:, 0]
    z = joints[:, [5, 9, 13, 17]].mean(1) - w
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    radial = joints[:, 5] - joints[:, 17]
    y = radial - (radial * z).sum(1, keepdims=True) * z
    y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-9
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=-1)          # (T,3,3) 列向量为基
    if hand == "left":                        # palm flip: 绕 z 转 180° (x,y 同时反号)
        R = R @ np.diag([-1.0, -1.0, 1.0])
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


# `stable_pose_projection` 于 2026-08-11 删除 (随 load_replay 的 initial_pose_mode 一起):
# 它属于"以物体自身位姿为准摆放"的旧约定。现行摆放见 ref_builders/replay_grasp.py。
# `support_faces` 保留 —— replay_grasp 仍用它算贴桌高度。


def scene_rotation(scene_rot: str):
    """scene_rot 字符串 -> 旋转矩阵. identity: 新流水线(hawor_to_world_replay)
    已是 z-up 世界系 (run_retarget.sh 的默认 --scene-rot 1,0,0,0);
    cv2zup: 旧相机系数据兼容; 或 'w,x,y,z' 显式四元数."""
    if scene_rot == "identity":
        return np.eye(3)
    if scene_rot == "cv2zup":
        return CV2ZUP
    return quat_to_rotmat(np.array([float(x) for x in scene_rot.split(",")]))


def pick_anchor_object(npz_path, mesh_path, recon_dir=None):
    """挑**最可信**的物体来定场景原点。→ (obj_p, obj_q, mesh_path, 说明)

    `align_replay` 用"某个物体首帧的位姿 + 网格最低顶点"定整个场景的平移量。
    原来固定用**主物体**(机器人抓的那个) —— 但主物体是由"机器人是左手还是右手"
    决定的, 与"哪个物体的轨迹可信"毫无关系。

    实测 screw_unscrew_bottle_cap/0:
        object_0 瓶身  conf_pos 82  被证伪 1/133
        object_1 瓶盖  conf_pos 50  被证伪 **104/133**
    两者定出的场景平移差 **10.3cm**。机器人只有右手 -> 只能抓瓶盖 -> 整只手的参考
    轨迹会被那份坏数据带偏 10cm。

    所以改成: 谁可信谁定原点, 与谁被抓无关。判据取 `confidence_complete.json` 的
    `conf_pos_median`(越高越好), 打平时取被证伪帧少的。没有该文件就退回主物体。
    """
    import json as _j
    import os as _os
    import numpy as _np
    d = _np.load(npz_path, allow_pickle=True)
    prim = (d["obj_pose"][:, :3].astype(_np.float64),
            d["obj_pose"][:, 3:7].astype(_np.float64), mesh_path, "主物体(无可信度信息)")
    if "obj_pose_all" not in d.files or "object_ids" not in d.files:
        return prim
    rd = recon_dir or _os.path.dirname(str(npz_path)).replace("RetargetOutput",
                                                              "ReconstructOutput")
    cf = _os.path.join(rd, "confidence_complete.json")
    if not _os.path.isfile(cf):
        return prim
    try:
        co = (_j.load(open(cf)).get("objects") or {})
    except Exception:
        return prim
    oids = [str(x) for x in d["object_ids"]]
    scored = [(co.get(o, {}).get("conf_pos_median") or -1,
               -(co.get(o, {}).get("refuted_frames") or 0), i, o)
              for i, o in enumerate(oids) if o in co]
    if not scored:
        return prim
    scored.sort(reverse=True)
    conf, negref, i, oid = scored[0]
    m = _os.path.join(rd, "objects", oid, "object_mesh_scaled_final.obj")
    if not _os.path.isfile(m):
        m = mesh_path
    P = d["obj_pose_all"][i]
    return (P[:, :3].astype(_np.float64), P[:, 3:7].astype(_np.float64), m,
            f"{oid}(conf_pos={conf}, 被证伪 {-negref} 帧) —— 最可信, 与谁被抓无关")


def align_replay(joints, obj_p, obj_q_wxyz, mesh_path,
                 table_height=0.85, obj_gap=0.01, scene_rot="identity", anchor=None):
    """scene_rot + 落桌 recenter, 手物同一变换. 返回 (joints', obj_p', obj_q', info).

    `anchor` = `pick_anchor_object()` 的返回 (obj_p, obj_q, mesh, 说明):
    **用它来算平移量**, 但平移仍然施加给传进来的 joints/obj。
    为什么要分开: 平移量原本由"机器人抓的那个物体"决定, 而那与"哪个物体可信"无关 ——
    实测 clip0 拿瓶盖(被证伪 104/133)定原点比拿瓶身(1/133)偏 10.3cm。
    anchor=None 时行为与原来完全一致。
    """
    Rs = scene_rotation(scene_rot)
    qs = rotmat_to_quat(Rs)
    T = len(obj_p)
    obj_p_r = obj_p @ Rs.T
    obj_q_r = quat_mul(np.broadcast_to(qs, (T, 4)), obj_q_wxyz)
    joints_r = joints @ Rs.T

    if anchor is not None:
        a_p, a_q, a_mesh, _why = anchor
        a_p_r = a_p @ Rs.T
        a_q_r = quat_mul(np.broadcast_to(qs, (len(a_q), 4)), a_q)
        a_verts = load_obj_verts(a_mesh)
        a_v0 = rot_apply(np.broadcast_to(a_q_r[0], (len(a_verts), 4)), a_verts) + a_p_r[0]
    lo, hi = object_aabb_obj(mesh_path)
    # 落桌高度必须用真实顶点最低点: AABB 角点对圆柱/圆弧面是空气,
    # 会把物体悬空数厘米 (躺置圆柱实测 4.6cm), 落地冲击直接滚走
    verts = load_obj_verts(mesh_path)
    v0 = rot_apply(np.broadcast_to(obj_q_r[0], (len(verts), 4)), verts) + obj_p_r[0]
    if anchor is not None:
        shift = np.array([-a_p_r[0, 0], -a_p_r[0, 1],
                          table_height + obj_gap - a_v0[:, 2].min()])
    else:
        shift = np.array([-obj_p_r[0, 0], -obj_p_r[0, 1],
                          table_height + obj_gap - v0[:, 2].min()])
    info = {"aabb_lo": lo, "aabb_hi": hi, "shift": shift,
            "half_diag": np.linalg.norm(hi - lo) / 2}
    return joints_r + shift, obj_p_r + shift, obj_q_r, info
