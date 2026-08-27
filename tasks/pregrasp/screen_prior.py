"""GraspPose 候选筛选 —— 三关一次跑完。判据与依据见 `docs/GRASPPOSE_SCREENING.md`。

    Gate 0  数据同一性   秒级   离线      mesh OBB 对不上 = 数据问题, 不是候选问题
    Gate 1  可达性       秒级   离线      yaw 扫描: 可达带是否覆盖**视频 yaw**
    Gate 2  prior 质量   ~2min  Isaac     pads* >= 4 且 Q* > 0

Gate 0/1 不需要 GPU, 所以一批几十个候选先秒级砍掉大半, 只有幸存者才付 Isaac 的钱。

用法
----
    PY=/home/lyh/luhr/MagicSim/.venv/bin/python
    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.screen_prior \\
        --clip Grasp1 \\
        --grasp_dir /home/lyh/Project/Dexonomy/output/pp1_sharpa_wave \\
        --info_json /home/lyh/Project/Dexonomy/assets/object/custom/processed_data/pp1/info/simplified.json

`--grasp_dir` 会**递归**收集 `*_grasp.npy` —— 故意同时吃 `grasp_data/` 和
`grasp_data_isaac_failed/`: Dexonomy 的 Isaac 验证器对本平台没有预测力,
不能拿它当候选池 (见规范 §0)。

  --offline_only   只跑 Gate 0/1 (不开 Isaac, 几秒出结果)
  --tol_deg        Gate 1 的 Δψ 容忍度, 默认 10 (严格档, 2026-08-01 定)
  --prior_dir      跳过 npy 转换, 直接吃已有的 prior npz

⚠ Dexonomy 的 npy 是 numpy2 pickle, **必须用系统 python3 转换** —— 本脚本会自动
  subprocess 调 `python3 make_prior.py`, 不要改成用 Isaac venv 的解释器。
⚠ Gate 2 每个候选起一个独立进程 (一个 Isaac session 建多个 env 不安全)。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAD_NAMES = ("thumb", "index", "middle", "ring", "pinky")


# ---------------------------------------------------------------- 小工具
def quat_to_R(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def geo_deg(A, B):
    """两个旋转矩阵的测地角距 (度)."""
    c = (np.trace(A @ B.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def ang_diff(a, b):
    """角度差 (度), 取环上最短."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# ---------------------------------------------------------------- 视频 yaw
def video_yaw_deg(npz_path, canon_rot):
    """重建视频里物体静置段的 yaw (度), 以 canon_rot 为基准姿态.

    找 θ 使 Rz(θ)·R_canon 最接近视频姿态。返回 (θ, 残差角, 静置段稳定性)。
    残差角 = 非绕 z 的部分 —— 大 (>15°) 说明"哪个面朝下"和视频都不一致,
    此时 θ 没有意义 (Grasp3 曾出现 179.9° 翻面)。
    """
    from rl_rebuild.correction import frames as F
    z = np.load(npz_path, allow_pickle=True)
    op = z["obj_pose"]
    # 静置段 = 任一只手开始接触之前
    gs = len(op)
    for k in ("phase_right", "phase_left"):
        if k in z.files:
            idx = np.flatnonzero(np.asarray(z[k]).astype(int) == 1)
            if len(idx):
                gs = min(gs, int(idx[0]))
    gs = max(gs, 2)
    R_vid = [quat_to_R(op[i, 3:7]) for i in range(gs)]
    R_ref = R_vid[len(R_vid) // 2]
    stab = float(np.median([geo_deg(R, R_ref) for R in R_vid]))

    R_canon = quat_to_R(canon_rot)
    best = (1e9, 0.0)
    for deg in np.arange(0, 360, 0.5):
        e = geo_deg(Rz(np.radians(deg)) @ R_canon, R_ref)
        if e < best[0]:
            best = (e, float(deg))
    resid, theta = best
    _ = F  # 保持 import 语义 (mesh 顶点由调用方处理)
    return theta, resid, stab


def object_placement(npz_path, mesh_path, table_top_z, hand="right",
                     affordance=None, semantics=None, scene_layout_json=None):
    """物体在 env 世界系的摆放。**直接取 ref_builder 的结果**, 不再自己算。

    ⚠ 2026-08-15/16 两次踩到:本函数原来自己按"纯相机锚定"算 xy, 而 env 走的是
    `place_mode="ref_builder"` 的**物体听手**(物体 XY = 抓取锚帧的合拢中心)。
    两者对 pour17 差 **5.2cm** —— 于是 Gate 1 的可达带、Gate 1c 的接近方位角
    全都算在一个**错的物体位置**上(结论碰巧没变, 但推理链是断的)。
    docstring 原写"与 env 实测吻合 <0.05mm", 那只对**纯相机锚定**的老 clip 成立。

    ⚠ 而且这个位置**会随 gs 变**:物体听手锚在 gs 帧, 2026-08-16 把 gs 从"接触标注起点"
    改成"物体运动起始"后, pour17 瓶的摆放移动了 **9.1cm**。所以任何 yaw 结论都必须
    在**当前的 gs 口径**下重算, 不能沿用。
    """
    from rl_rebuild.correction.ref_builders.replay_grasp import load_replay_grasp
    import numpy as _np
    _du = load_replay_grasp(
        npz_path, mesh_path, hand=hand,
        affordance_npz=affordance, semantics=semantics,
        scene_layout_json=scene_layout_json,
        table_height=table_top_z, verbose=False)
    return _np.asarray(_du.object_init_pose[:3], float)


def video_yaw_deg_from_clip(clip_cfg, canon_rot):
    """薄包装: clips.py 注册表 -> 具体量。**数学只在上面那个函数里, 这里不复制。**"""
    return video_yaw_deg(clip_cfg["npz"], canon_rot)


def object_placement_from_clip(clip_cfg, canon_rot, table_top_z):
    """薄包装, 同上。Step3 等外部调用方可以绕开注册表直接调 `object_placement`。"""
    return object_placement(
        clip_cfg["npz"], clip_cfg["mesh"], table_top_z,
        hand=clip_cfg.get("robot_hand", clip_cfg.get("hand", "right")),
        affordance=clip_cfg.get("affordance"), semantics=clip_cfg.get("semantics"),
        scene_layout_json=clip_cfg.get("scene_layout_json"))


def _object_placement_camera_anchor(clip_cfg, canon_rot, table_top_z):
    """旧实现(纯相机锚定), 仅为对拍保留。**不要用它做判读** —— 见上面的坑。"""
    from rl_rebuild.correction import frames as F
    from rl_rebuild.correction import place_camera as PC
    raw = np.load(clip_cfg["npz"], allow_pickle=True)
    # c2w 的三处候选统一走 place_camera.find_cam_src(与 replay_grasp 同一份逻辑)。
    # 之前这里只看网格旁, pour17 的软链在上一层, 会误报"缺 c2w 过不了 Gate 0"。
    _cam = PC.find_cam_src(clip_cfg["mesh"], clip_cfg["npz"])
    sxy = (PC.camera_anchor_shift(_cam, PC.ZED_NOMINAL[:2], raw["obj_pose"][0, :2])
           if _cam is not None else None)
    if sxy is None:
        raise RuntimeError("world_fused.npz 缺 c2w, 无法相机锚定 —— 这条 clip 过不了 Gate 0")
    v = F.load_obj_verts(clip_cfg["mesh"]) @ quat_to_R(canon_rot).T
    return np.array([sxy[0], sxy[1], table_top_z + 0.002 - float(v[:, 2].min())])


# ---------------------------------------------------------------- Gate 0
def gate0(clip_cfg, info_json):
    """网格同一性: 重建 mesh 的 OBB 必须与 Dexonomy 的逐位吻合 (±0.5mm)."""
    import trimesh
    d = json.load(open(info_json))
    dex = sorted(float(x) for x in d["obb"])
    m = trimesh.load(clip_cfg["mesh"], force="mesh")
    rec = sorted(float(x) for x in m.bounding_box_oriented.primitive.extents)
    dmax = max(abs(a - b) for a, b in zip(dex, rec))
    return dmax <= 0.0005, dex, rec, dmax



def _make_ik(hand, anchor_link=None, anchor_T=None, _warned=[]):
    """构造 ArmIK。给了 anchor_T 就用**实测基座**(与 env 同口径), 否则退回名义并警告一次。"""
    from rl_rebuild.correction.kinematics import ArmIK
    if anchor_T is not None:
        return ArmIK(hand, anchor_link=(anchor_link or "arm_center"),
                     anchor_T=np.asarray(anchor_T, np.float64))
    if not _warned:
        _warned.append(1)
        print("  ⚠ 可达性用的是**名义基座**(无 anchor_T) —— 结论只能当排序键, **不能当判死线**。"
              "实测基座: tasks/pregrasp/priors/anchor_T_<hand>.json (dump_anchor.py 生成)")
    return ArmIK(hand)


def _ik_err_at(prior_npz, obj_pos, yaw_deg, hand, from_yaw=0.0,
               anchor_link=None, anchor_T=None) -> float:
    """在给定物体 yaw 上解抓姿 IK, 返回位置误差(m)。

    ⚠ **必须暖启动**: Gate 1 的可达带扫描是逐 5° 顺序求解、把上一格的解当种子的
    (`if r["ok"]: q = r["q"]`), 手臂被一点点"绕"过去。单点冷启动会掉进局部极小 ——
    实测 pour17 瓶 yaw=19.5° 冷启动报 18.53cm(假不可达), 而扫描说它可达。
    这里从 `from_yaw` 走 5° 步长过去, 每步用上一步的解当种子, 与 Gate 1 同口径。
    """
    from rl_rebuild.correction.kinematics import ArmIK
    z = np.load(prior_npz)
    canon = np.asarray(z["canon_rot"], np.float64)
    grasp = np.asarray(z["grasp"], np.float64)
    ik = _make_ik(hand, anchor_link, anchor_T)
    q = np.zeros(len(ik.arm_joints))
    q[0] = np.radians(-45.0)
    if len(q) > 3:
        q[3] = np.radians(-90.0)      # ★ 肘部预弯; 漏了它会卡在"伸直臂"的局部极小
    d = (yaw_deg - from_yaw) % 360
    path = list(np.arange(from_yaw, from_yaw + d, 5.0)) + [yaw_deg] if d > 5 else [yaw_deg]
    err = 1e9
    for y in path:
        a = np.radians(y)
        oq2 = qmul(np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]), canon)
        gp = quat_to_R(oq2) @ grasp[:3] + obj_pos
        gq = qmul(oq2, grasp[3:7])
        r = ik.solve(gp, quat_to_R(gq), q0=q, iters=200)
        if r["ok"]:
            q = r["q"]
        err = float(r["pos_err"])
    return err


# ------------------------------------------------------- Gate 1c (接近侧对齐)
YAW_SYM_MM = 5.0        # 绕竖轴转任意角的点云中位偏差 < 它 = 回转体, yaw 不携带物理信息


def yaw_symmetry_mm(mesh_path, canon_rot) -> float:
    """物体绕**竖轴**的旋转对称度: 转若干角度后点云到原点云的最近邻中位偏差(mm)取最大。

    小 = 回转体 ⟹ "物体朝哪"在物理上是空的量。判据阈值 YAW_SYM_MM 取 5mm
    (≈ 重建自身噪声量级)。pour17 实测: 瓶 0.6mm / 杯 3.3mm, 都在这一档。
    """
    from scipy.spatial import cKDTree
    from rl_rebuild.correction import frames as F
    V = F.load_obj_verts(mesh_path) @ quat_to_R(canon_rot).T      # 摆成 canon 姿态
    V = V - V.mean(0)
    rng = np.random.default_rng(0)
    S = V[rng.choice(len(V), min(3000, len(V)), replace=False)]
    T = cKDTree(V)
    return max(float(np.median(T.query((Rz(np.radians(a)) @ S.T).T)[0])) * 1000
               for a in (15, 45, 90, 180))


PRE_WIN = 8          # 接近段末尾窗口长度(帧); 15fps 下约 0.5s


def human_approach_azim(clip_cfg, hand):
    """人手腕在物体的**方位角**(度, 世界 XY, 0=+X, 逆时针), 取接触窗内的圆中位数。

    与 env 系只差一个 XY 平移(相机锚定是纯平移, 见 place_camera), 平移不改方位角,
    所以可以直接和机器人侧的方位角比。取整窗中位数而不是单帧 —— 单帧对重建抖动敏感。
    """
    z = np.load(clip_cfg["npz"], allow_pickle=True)
    J = np.asarray(z[f"joints_{hand}"], float)
    oid = clip_cfg.get("primary_oid")
    oids = [str(v) for v in z["object_ids"]]
    OP = np.asarray(z["obj_pose_all"], float)[oids.index(oid)]
    w = np.flatnonzero(np.asarray(z[f"phase_{hand}"]).astype(int) == 1)
    if not len(w):
        raise RuntimeError(f"{oid}/{hand}: 没有接触帧, 无法定接近方位")
    # ⚠ 窗口只取**接近段末尾**(合拢前后), **不能用整条接触窗**: 抓住之后手与物体刚体
    #   耦合, 而倒水段物体被倾到 84°, "腕相对物体的水平方位角"在那段已经没有意义。
    #   pour17 实测: 整窗中位 193.1° vs 接近段末 216.4°, 差 23° —— 足以改变裁定。
    g0 = int(w[0])
    lo, hi = max(g0 - PRE_WIN, 0), min(g0 + 2, len(J) - 1)
    v = J[lo:hi + 1, 0, :2] - OP[lo:hi + 1, :2]
    ok = np.isfinite(v).all(1)
    ang = np.arctan2(v[ok, 1], v[ok, 0])
    c, sn = np.cos(ang).mean(), np.sin(ang).mean()
    spread = float(np.degrees(np.sqrt(-2.0 * np.log(min(np.hypot(c, sn), 1.0)))))  # 圆标准差
    return float(np.degrees(np.arctan2(sn, c)) % 360), spread, g0, (lo, hi)


def approach_offset_K(canon_rot, grasp) -> float:
    """K = 手腕方位角 − 物体 yaw (度). 由 GraspPose 自身决定(抓法在物体的哪一侧)."""
    wp = quat_to_R(canon_rot) @ np.asarray(grasp, float)[:3]     # yaw=0 时的腕位偏移
    return float(np.degrees(np.arctan2(wp[1], wp[0])) % 360)


def _parse_band(band: str):
    """"0-120, 330-355" -> [(0,120),(330,355)]"""
    out = []
    for seg in (band or "").split(","):
        seg = seg.strip()
        if "-" in seg:
            a, b = seg.split("-")
            out.append((float(a), float(b)))
    return out


# ---------------------------------------------------------------- Gate 1
def gate1(prior_npz, obj_pos, video_yaw, tol_deg, step=5, hand="right",
          anchor_link=None, anchor_T=None):
    """yaw 扫描: 可达带 Ψ 是否够到视频 yaw.

    ★ `anchor_T` (2026-08-17 加): env 解 IK 用的是**实测的 arm_center 位姿**, 而本函数
      原来用 `ArmIK(hand)` = **URDF 名义基座**。`dexmate_env.py:290` 的注释:
        "锚在实测的 arm_center 上, 不用『躯干在配置角度』这个假设 —— 否则躯干沉降几度
         就让整条臂的基座偏掉, q_ref 会是个到不了的目标。"
      ⟹ 两边解的不是同一个末端坐标系, 可达性结论必然打架。2026-08-17 实测: 本函数说
      yaw 175° 可达, env 里 `grasp ok=False` 够不着(杯候选 1_8)。
      **不传 anchor_T 时结论只能当排序键, 不能当判死线**(会打印警告)。
      实测值由 `tasks/pregrasp/dump_anchor.py` 导出到 priors/anchor_T_<hand>.json。

    ⚠ `hand` 必须跟 clip 的 `robot_hand` 走 —— 之前写死 "right", 拿右臂去筛左手候选
    (pour17 的杯是左手)会给出完全无意义的可达带。2026-08-15 修。
    """
    z = np.load(prior_npz)
    if "canon_rot" not in z.files:      # 老 prior(Screw27 等)没有这一项, 跳过而不是崩
        return dict(ok=False, n_reach=0, dpsi=None, best_yaw=None, best_err=float("nan"),
                    band="", skipped="无 canon_rot(老 prior)")
    canon = np.asarray(z["canon_rot"], np.float64)
    grasp = np.asarray(z["grasp"], np.float64)
    ik = _make_ik(hand, anchor_link, anchor_T)
    q = np.zeros(len(ik.arm_joints))
    q[0] = np.radians(-45.0)
    if len(q) > 3:
        q[3] = np.radians(-90.0)
    rows = []
    for deg in range(0, 360, step):
        a = np.radians(deg)
        oq2 = qmul(np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]), canon)
        gp = quat_to_R(oq2) @ grasp[:3] + obj_pos
        gq = qmul(oq2, grasp[3:7])
        r = ik.solve(gp, quat_to_R(gq), q0=q, iters=80)
        if r["ok"]:
            q = r["q"]
        rows.append((float(deg), float(r["pos_err"])))
    reach = [d for d, e in rows if e < 0.01]
    if not reach:
        return dict(ok=False, n_reach=0, dpsi=None, best_yaw=None,
                    best_err=min(e for _, e in rows), band="")
    dpsi = min(ang_diff(d, video_yaw) for d in reach)
    # 可达带里离视频 yaw 最近的那个角 (这才是我们要用的 yaw, 不是 IK 最优的那个)
    best_yaw = min(reach, key=lambda d: ang_diff(d, video_yaw))
    best_err = dict(rows)[best_yaw]
    segs, s, p = [], reach[0], reach[0]
    for v in reach[1:]:
        if v - p > step:
            segs.append((s, p))
            s = v
        p = v
    segs.append((s, p))
    return dict(ok=dpsi <= tol_deg, n_reach=len(reach), dpsi=dpsi,
                best_yaw=best_yaw, best_err=best_err,
                band=", ".join(f"{a:.0f}-{b:.0f}" for a, b in segs))


# ---------------------------------------------------------------- Gate 1b (H3)
# 派生常数, 与 tasks/pregrasp 的管壁参数一致 (见 docs/PLAN_PICK_LIFT.md §4.1)
TUBE_SOFT_FRAC = 0.7      # 管壁从 0.7R 起就往回拉 -> 无阻力半径只有 0.7R
TUBE_JITTER = 0.015       # 物体位置课程最大抖动
TUBE_SLACK = 0.02         # IK 残差 + 控制跟踪滞后
TUBE_R_CAP = 0.20         # H3 判死线


def human_pregrasp_wrist(clip_cfg, clip, table_top_z=0.85, hand=None):
    """人手参考轨迹在 **PreGrasp 帧** 的腕位姿 (env 系) + 该帧号.

    这是接近段的终点参考; H3 量的就是"它离 GraspPose 有多远".
    纯离线 (numpy), 与 env 里 ref builder 走同一条码路.
    """
    from rl_rebuild.correction.ref_builders.replay_grasp import load_replay_grasp
    du = load_replay_grasp(
        clip_cfg["npz"], clip_cfg["mesh"], usd_path=clip_cfg.get("usd", ""),
        # ★ 2026-08-17 修: 原来这里**写死 "right"**, 没有 hand 参数。
        #   于是给**左手** clip 评 H3 时, 人手参考取的是**右手**腕轨迹 —— 左右手近似镜像,
        #   姿态差直接飙到百来度。实测吻合: 瓶(右手) 24° 合理 / 杯(左手) 165° 离谱,
        #   出问题的恰好就是那个左右不一致的。现在按 clip 注册表的 hand 取。
        clip_id=clip, hand=(hand or clip_cfg.get("hand") or "right"),
        table_height=table_top_z,
        affordance_npz=clip_cfg.get("affordance"), semantics=clip_cfg.get("semantics"),
        # ⚠ 必须 False —— 2026-08-02 D7 之后训练侧已关掉 PreGrasp 对齐, 这里若还开着,
        # 算出来的 G 会比训练时**小一半以上** (Grasp3/8_5: 8.9cm vs 实测 17.8cm),
        # 因为对齐会把整条腕轨迹往物体方向挪 13.68cm.
        pregrasp_align=False)   # anchor_mode 已删(相机锚定是唯一模式)
    r = du.ref
    # grasp_phase_frame 是**合拢终点** ge; PreGrasp 帧 gs = ge - close_steps(30)
    gs = int(r.grasp.grasp_phase_frame) - 30
    W = np.asarray(r.track_wrist)
    return W[gs, :3].astype(np.float64), W[gs, 3:7].astype(np.float64), gs


def gate1b(prior_npz, obj_pos, yaw_deg, wrist_p, wrist_q):
    """H3 人手一致性: 缺口 G -> 派生管壁 R_hi, 超过 cap 判死. 见 GRASPPOSE_SCREENING Gate 1b."""
    z = np.load(prior_npz)
    canon = np.asarray(z["canon_rot"], np.float64)
    a = np.radians(yaw_deg)
    oq2 = qmul(np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]), canon)
    gp = quat_to_R(oq2) @ np.asarray(z["grasp"][:3], np.float64) + obj_pos
    gq = qmul(oq2, np.asarray(z["grasp"][3:7], np.float64))
    G = float(np.linalg.norm(gp - wrist_p))
    dth = geo_deg(quat_to_R(gq), quat_to_R(wrist_q))
    R_hi = G / TUBE_SOFT_FRAC + TUBE_JITTER + TUBE_SLACK
    return dict(G=G, d_rot_deg=dth, R_hi=R_hi, ok=bool(R_hi <= TUBE_R_CAP))


# ---------------------------------------------------------------- Gate 2 (子进程)
def gate2_child(clip, prior_npz, yaw_deg=-1.0):
    """在**本进程**里开 Isaac 跑 Gate 2, 结果以 JSON 打到 stdout (由父进程解析)."""
    from isaaclab.app import AppLauncher
    import argparse as _ap
    _p = _ap.ArgumentParser()
    AppLauncher.add_app_launcher_args(_p)
    _a = _p.parse_args(["--headless"])

    from rl_rebuild.utils.gpu_guard import isaac_slot
    _slot = isaac_slot("screen_prior")
    app = AppLauncher(_a).app

    import torch
    from rl_rebuild.correction import clips as _clips
    from tasks.pregrasp.cfg import GraspTaskCfg
    from tasks.pregrasp.env import GraspTaskEnv

    cfg = GraspTaskCfg()
    _clips.configure_cfg(cfg, clip)
    cfg.grasp_prior_npz = prior_npz
    cfg.prior_yaw_deg = float(yaw_deg)      # D1: 钉死在 Gate 1 的 best_yaw
    cfg.scene.num_envs = 16
    cfg.obj_jitter_xy = 0.0
    cfg.closure_init_max = 0.0
    E = GraspTaskEnv(cfg)
    E.gentle = 1.0                       # 全价看信号量级
    N, A = cfg.scene.num_envs, cfg.action_space

    def sweep(thumb_a, steps, settle):
        """压到最深保持, 返回该设定下的稳态读数."""
        E.reset()
        act = torch.zeros((N, A), device=E.device)
        act[:, 7] = 1.0                  # c 全速到 closure_max
        act[:, 8] = thumb_a              # 拇指残差
        act[:, 9:] = 1.0                 # 四指压深
        for _ in range(settle):
            E.step(act)
        acc, nrec = None, 0
        for _ in range(steps):
            E.step(act)
            s = E._sig
            G = cfg.pad_force_sign * torch.cat(
                [c.data.force_matrix_w.view(N, 1, 3) for c in E._contact_sensors], dim=1)
            mag = G.nan_to_num(0.0).norm(dim=-1).clamp(max=50.0)
            over = (mag - cfg.squeeze_f_max).clamp(min=0.0).sum(dim=1)
            row = np.array([
                s["n_pads"].float().mean().item(), s["cent"].mean().item(),
                s["r_imb"].mean().item(), s["tau_n"].mean().item(),
                s["quality"].mean().item(), over.mean().item(),
                mag.max().item(), s["cand_ok"].float().mean().item(),
            ] + mag.mean(dim=0).tolist())
            acc = row if acc is None else acc + row
            nrec += 1
        return acc / max(nrec, 1)

    # ---- 深度轴 (五指全开) + 拇指轴 ----
    settings = [1.0, 0.0, -0.25, -0.5, -0.75, -1.0]
    recs = {}
    for t in settings:
        recs[t] = sweep(t, steps=8, settle=48)

    # 标称位姿 (c=1.0 附近) 的穿透量: 用较浅的 settle 采一次
    nominal = sweep(1.0, steps=6, settle=24)

    Qs = {t: r[4] for t, r in recs.items()}
    Ps = {t: r[0] for t, r in recs.items()}
    t_best = max(Qs, key=lambda k: Qs[k])
    r = recs[t_best]
    pads_star = max(Ps.values())
    n_pass_pads = sum(1 for v in Ps.values() if v >= cfg.success_min_pads)
    pad_f = r[8:13]
    share = float(max(pad_f) / max(sum(pad_f), 1e-6))

    out = dict(
        clip=clip, prior=os.path.basename(prior_npz),
        Q_star=float(max(Qs.values())), pads_star=float(pads_star),
        thumb_a_best=float(t_best),
        cent=float(r[1]), imb=float(r[2]), tau_n=float(r[3]),
        over_N=float(r[5]), peak_pad_N=float(r[6]), cand_ok=float(r[7]),
        pad_forces={n: float(v) for n, v in zip(PAD_NAMES, pad_f)},
        max_pad_share=share,
        n_settings_pads_ok=int(n_pass_pads),      # 0 = 悬崖 (无任何档能拿到 >=4 垫)
        nominal_peak_pad_N=float(nominal[6]),
        min_pads_required=int(cfg.success_min_pads),
    )
    out["H1_pads"] = bool(out["pads_star"] >= cfg.success_min_pads)
    out["H2_Q"] = bool(out["Q_star"] > 0.0)
    out["pass"] = bool(out["H1_pads"] and out["H2_Q"])
    print("###GATE2_JSON###" + json.dumps(out), flush=True)

    E.close()
    app.close()


def gate2(clip, prior_npz, yaw_deg=-1.0, timeout=1800):
    """起独立进程跑 Gate 2 (一个 Isaac session 建多个 env 不安全)."""
    env = dict(os.environ, PYTHONPATH=REPO, SHARPA_WANDB="0")
    cmd = [sys.executable, "-u", "-m", "tasks.pregrasp.screen_prior",
           "--_gate2_child", "--clip", clip, "--prior", prior_npz,
           "--prior_yaw", str(yaw_deg)]
    try:
        p = subprocess.run(cmd, cwd=REPO, env=env, timeout=timeout,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return dict(error="timeout")
    for line in p.stdout.splitlines():
        if line.startswith("###GATE2_JSON###"):
            return json.loads(line[len("###GATE2_JSON###"):])
    tail = "\n".join((p.stdout + p.stderr).splitlines()[-15:])
    return dict(error="no result", tail=tail)


# ---------------------------------------------------------------- 转换
def convert(npy, info_json, out_npz):
    """Dexonomy npy -> prior npz. **必须用系统 python3** (npy 是 numpy2 pickle)."""
    cmd = ["python3", os.path.join(REPO, "tasks/pregrasp/make_prior.py"),
           "--grasp_npy", npy, "--info_json", info_json, "--out", out_npz]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--grasp_dir", help="递归收集 *_grasp.npy (含 isaac_failed)")
    ap.add_argument("--prior_dir", help="直接吃已转换的 prior npz")
    ap.add_argument("--info_json", help="Dexonomy simplified.json (Gate 0 + 转换需要)")
    ap.add_argument("--out_dir", default=None, help="转换产物落盘处")
    ap.add_argument("--tol_deg", type=float, default=10.0,
                    help="Gate 1 的 Δψ 容忍度 (默认 10 = 严格档)")
    ap.add_argument("--offline_only", action="store_true", help="只跑 Gate 0/1/1b")
    ap.add_argument("--no_h3", action="store_true",
                    help="跳过 Gate 1b —— 只在**没有接近段**的任务上才该用")
    ap.add_argument("--table_top_z", type=float, default=0.85)
    ap.add_argument("--_gate2_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--prior", help=argparse.SUPPRESS)
    ap.add_argument("--prior_yaw", type=float, default=-1.0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._gate2_child:
        return gate2_child(args.clip, args.prior, args.prior_yaw)

    from rl_rebuild.correction import clips as _clips
    clip_cfg = _clips.CLIPS[args.clip]
    # 用 --prior_dir 时报告落在同一个目录里, 别新建一个 <clip>_candidates 让人找不到
    out_dir = (args.out_dir or args.prior_dir
               or os.path.join(REPO, f"tasks/pregrasp/priors/{args.clip}_candidates"))
    os.makedirs(out_dir, exist_ok=True)

    # ---- 收集候选 ----
    if args.prior_dir:
        priors = sorted(glob.glob(os.path.join(args.prior_dir, "*.npz")))
    else:
        if not (args.grasp_dir and args.info_json):
            ap.error("--grasp_dir 需要配 --info_json (或改用 --prior_dir)")
        npys = sorted(glob.glob(os.path.join(args.grasp_dir, "**", "*_grasp.npy"),
                                recursive=True))
        print(f"[collect] {len(npys)} 个 Dexonomy 候选 (含 isaac_failed)", flush=True)
        priors = []
        for f in npys:
            tag = os.path.basename(f).replace("_grasp.npy", "")
            o = os.path.join(out_dir, f"{tag}.npz")
            ok, msg = convert(f, args.info_json, o)
            if ok:
                priors.append(o)
            else:
                print(f"  ✗ {tag} 转换失败: {msg}", flush=True)
    if not priors:
        print("没有可筛的候选"); return 1

    # ---- Gate 0 ----
    print("\n" + "=" * 78)
    if args.info_json:
        ok, dex, rec, dmax = gate0(clip_cfg, args.info_json)
        print(f"[Gate 0] 网格同一性: Dexonomy OBB {[round(x,4) for x in dex]}")
        print(f"         重建 OBB          {[round(x,4) for x in rec]}   最大差 {dmax*1000:.2f}mm")
        if not ok:
            print("         ❌ 不吻合 —— 尺度/网格版本不一致, 先解决这个, 筛选无意义")
            return 1
        print("         ✅ 通过")
    else:
        print("[Gate 0] 跳过 (未给 --info_json)")

    canon = np.asarray(np.load(priors[0])["canon_rot"], np.float64)
    obj_pos = object_placement_from_clip(clip_cfg, canon, args.table_top_z)
    vy, resid, stab = video_yaw_deg_from_clip(clip_cfg, canon)
    print(f"\n[基准] 物体摆放 xy=({obj_pos[0]:+.4f}, {obj_pos[1]:+.4f}) z={obj_pos[2]:.4f}")
    print(f"       视频 yaw = {vy:.1f}°  (静置段稳定性 {stab:.1f}°, 翻面残差 {resid:.1f}°)")
    if resid > 15:
        print(f"       ⚠ 翻面残差 {resid:.1f}° 偏大 —— canon 姿态与视频**朝下的面**不一致, "
              f"视频 yaw 无意义, Gate 1 结果不可信")

    # ---- Gate 1 ----
    # ★ 实测臂基座: 有就用(与 env 同口径), 没有就退回名义并在下面打警告。
    #   2026-08-17 起 —— 名义基座会给出 env 够不到的候选(实测: yaw 175° 说可达, env ok=False)。
    _hand = clip_cfg.get("robot_hand", clip_cfg.get("hand", "right"))
    _anchor_T, _anchor_link = None, None
    _apath = os.path.join(os.path.dirname(__file__), "priors", f"anchor_T_{_hand}.json")
    if os.path.exists(_apath):
        with open(_apath) as _f:
            _ar = json.load(_f)
        _anchor_T, _anchor_link = _ar["anchor_T"], _ar.get("anchor_link", "arm_center")
        print(f"  [基座] 用**实测** anchor_T ({_hand}) 自 {os.path.basename(_apath)} "
              f"dump于 {_ar.get('dumped_at','?')} —— 与 env 同口径")
    else:
        print(f"  [基座] ⚠ 找不到 {_apath} —— 退回**名义基座**, "
              f"可达性结论只能当排序键。生成: python -m tasks.pregrasp.dump_anchor")
    print(f"\n[Gate 1] 可达性 (Δψ ≤ {args.tol_deg:.0f}° 才算与视频一致)")
    print(f"  {'候选':>10s} {'可达档':>7s} {'可达带':>22s} {'Δψ':>7s} {'用哪个yaw':>10s} {'IK':>8s}  判定")
    survivors = []
    g1 = {}
    for p in priors:
        tag = os.path.basename(p)[:-4]
        r = gate1(p, obj_pos, vy, args.tol_deg, hand=_hand,
                   anchor_link=_anchor_link, anchor_T=_anchor_T)
        g1[tag] = r
        if r.get("skipped"):
            print(f"  {tag:>10s} {'—':>7s} {'—':>22s} {'—':>7s} {'—':>10s} {'—':>8s}  ⏭ {r['skipped']}")
            continue
        if r["n_reach"] == 0:
            print(f"  {tag:>10s} {'0/72':>7s} {'—':>22s} {'—':>7s} {'—':>10s} "
                  f"{r['best_err']*100:7.2f}cm  ❌ 全域不可达")
            continue
        mark = "✅" if r["ok"] else "❌"
        print(f"  {tag:>10s} {r['n_reach']:>5d}/72 {r['band']:>22s} {r['dpsi']:>6.0f}° "
              f"{r['best_yaw']:>9.0f}° {r['best_err']*100:7.2f}cm  {mark}")
        if r["ok"]:
            survivors.append(p)
    print(f"  -> {len(survivors)}/{len(priors)} 过 Gate 1")

    # ---- Gate 1c: 接近侧对齐 (新范式"照人手轨迹走"才需要, 2026-08-15 用户裁定走 B) ----
    #   Gate 1 只保证"物体朝向与视频一致"; 它**不检查**机器人的手是否从人手那一侧接近。
    #   pour17 实测两者差 108.6° —— 参考轨迹把手领到 216.4°, GraspPose 落在 325°,
    #   两个权威在训练早期互相拽。物体 yaw 只能对齐其中一件(视频接触带与 Dexonomy
    #   抓法在**物体系里**就差 108°), 所以必须选。
    #   裁定规则(可测, 不是拍脑袋): 物体绕竖轴近似对称(< YAW_SYM_MM) ⟹ "它朝哪"在物理
    #   上是空的量, A 方案在保护一个空的量 ⟹ 走 B(对齐人手接近方位); 否则走 A。
    print(f"\n[Gate 1c] 接近侧对齐 (方位角 0=+X 远离机器人, 90=机器人左手边, 逆时针)")
    _hand = clip_cfg.get("robot_hand", clip_cfg.get("hand", "right"))
    _sym = yaw_symmetry_mm(clip_cfg["mesh"], canon)
    _azh, _spread, _g0, _win = human_approach_azim(clip_cfg, _hand)
    _regime = "回转体 -> 走 B" if _sym < YAW_SYM_MM else "朝向有物理意义 -> 走 A"
    print(f"  绕竖轴对称度 {_sym:.1f}mm (阈 {YAW_SYM_MM:.0f}mm) ⇒ {_regime}")
    print(f"  人手接近方位角 {_azh:.1f}° (合拢帧 f{_g0}, 窗 f{_win[0]}~f{_win[1]}, "
          f"圆标准差 {_spread:.1f}°{'  ⚠ 抖动大, 该角不可信' if _spread > 15 else ''})")
    print(f"  {'候选':>10s} {'K':>7s} {'A yaw':>7s} {'A方位':>7s} {'B yaw':>7s} {'B方位':>7s} "
          f"{'B处IK':>8s} {'该用':>7s}")
    use_yaw, b_ok = {}, set()         # 交给 Gate 1b —— 一致性必须在**真要用的角**上评
    for p in priors:
        tag = os.path.basename(p)[:-4]
        if g1.get(tag, {}).get("skipped") or g1.get(tag, {}).get("n_reach", 0) == 0:
            continue
        _z = np.load(p)
        K = approach_offset_K(canon, _z["grasp"])
        yaw_b = (_azh - K) % 360
        # ⚠ 不用"在不在可达带"判 —— 可达带是每 5° 采样的, 边界有 ±5° 的不确定,
        #   pour17 瓶实测 yaw_B=356.6° 落在带外 1.6°, 但那只是采样格点的假边界。
        #   直接在这个角上解一次 IK 才是真判据。
        _err = _ik_err_at(p, obj_pos, yaw_b,
                          clip_cfg.get("robot_hand", clip_cfg.get("hand", "right")))
        in_band = _err < 0.01
        yaw_a = g1[tag]["best_yaw"]
        use = (yaw_b if (_sym < YAW_SYM_MM and in_band) else yaw_a)
        use_yaw[tag] = use
        if in_band:
            b_ok.add(tag)
        print(f"  {tag:>10s} {K:6.1f}° {yaw_a:6.1f}° {(yaw_a+K)%360:6.1f}° "
              f"{yaw_b:6.1f}° {(yaw_b+K)%360:6.1f}° {_err*100:7.2f}cm "
              f"{use:6.1f}°")
        if _sym < YAW_SYM_MM and not in_band:
            print(f"  {'':>10s} ⚠ B 的角上 IK 误差 {_err*100:.2f}cm >1cm —— "
                  f"该候选够不到人手那一侧, 退回 A")

    # ★ 走 B 时改判硬门: Gate 1 的 ok/fail(Δψ≤tol = "物体朝向像不像视频")本身是个
    #   **A 判据**, 拿它卡 B 方案会把好候选误杀(pour17 杯: Gate1 ❌ 差 0.5°, 但 B 角
    #   IK 只有 0.08cm)。走 B 时的硬门 = "B 角可达" + 后面的 "B 角上过 H3"。
    if _regime.endswith("走 B"):
        _sv = [p for p in priors if os.path.basename(p)[:-4] in b_ok]
        print(f"  -> 走 B: 硬门换成'B 角可达', {len(_sv)}/{len(priors)} 通过 "
              f"(Gate 1 的 Δψ 判定仅作参考)")
        survivors = _sv

    # ---- Gate 1b: H3 人手一致性 (只对带接近段的任务是硬判据) ----
    g1b = {}
    if survivors and not args.no_h3:
        wp, wq, gs_f = human_pregrasp_wrist(clip_cfg, args.clip, args.table_top_z)
        print(f"  [H3] 人手参考取的是 **{clip_cfg.get('hand', 'right')}** 手 "
              f"(clip 注册表的 hand 字段) —— 左右取错会让姿态差飙到百来度")
        print(f"\n[Gate 1b] H3 人手一致性 (PreGrasp 帧 {gs_f}, 腕位 "
              f"{np.round(wp, 3)}; R_hi ≤ {TUBE_R_CAP*100:.0f}cm 才过)")
        print(f"  {'候选':>10s} {'用的yaw':>8s} {'缺口G':>9s} {'姿态差':>8s} {'派生管壁R_hi':>13s}  判定")
        keep = []
        for p in survivors:
            tag = os.path.basename(p)[:-4]
            # ★ 用 Gate 1c 定下的角, 不是 Gate 1 的 best_yaw —— 一致性判据必须在
            #   **真要用的摆法**上评。用 A 的角评 B 的方案, 必然报"抓法与人差太远"。
            _uy = use_yaw.get(tag, g1[tag]["best_yaw"])
            r = gate1b(p, obj_pos, _uy, wp, wq)
            g1b[tag] = r
            print(f"  {tag:>10s} {_uy:7.1f}° {r['G']*100:8.1f}cm {r['d_rot_deg']:7.0f}° "
                  f"{r['R_hi']*100:12.1f}cm  {'✅' if r['ok'] else '❌ 抓法与人差太远'}")
            if r["ok"]:
                keep.append(p)
        print(f"  -> {len(keep)}/{len(survivors)} 过 H3")
        survivors = keep

    if args.offline_only or not survivors:
        # 离线模式也要落报告 —— 下游 (选训练候选 / 台账) 读的是它, 不是日志
        json.dump(dict(clip=args.clip, tol_deg=args.tol_deg, video_yaw=vy,
                       obj_pos=obj_pos.tolist(), n_candidates=len(priors),
                       gate1=g1, gate1b=g1b, gate2={}, passed=[]),
                  open(os.path.join(out_dir, "screen_report.json"), "w"),
                  indent=1, ensure_ascii=False, default=float)
        print(f"\n  报告: {os.path.join(out_dir, 'screen_report.json')}")
        if not survivors:
            print("\n【结论】本批全部不过 Gate 1/1b —— 按规则整批丢弃, 重新生成一批。")
        return 0

    # ---- Gate 2 ----
    print(f"\n[Gate 2] prior 质量 (需 pads* ≥ 4 **且** Q* > 0), {len(survivors)} 个候选")
    passed, results = [], {}
    for p in survivors:
        tag = os.path.basename(p)[:-4]
        print(f"  · {tag} ...", end=" ", flush=True)
        r = gate2(args.clip, p, g1[tag]["best_yaw"])
        results[tag] = r
        if "error" in r:
            print(f"❌ {r['error']}")
            continue
        mark = "✅" if r["pass"] else "❌"
        n_ok = r["n_settings_pads_ok"]
        grad = "悬崖(0档)" if n_ok == 0 else f"{n_ok}/6档≥4垫"
        print(f"{mark}  Q*={r['Q_star']:+.3f}  pads*={r['pads_star']:.2f}  "
              f"imb={r['imb']:.3f}  τ={r['tau_n']:.3f}  峰值={r['peak_pad_N']:.1f}N  "
              f"独吞={r['max_pad_share']*100:.0f}%  {grad}")
        if r["pass"]:
            passed.append(tag)

    # ---- 汇总 ----
    rep = dict(clip=args.clip, tol_deg=args.tol_deg, video_yaw=vy,
               obj_pos=obj_pos.tolist(), n_candidates=len(priors),
               gate1={k: v for k, v in g1.items()}, gate1b=g1b,
               gate2=results, passed=passed)
    rp = os.path.join(out_dir, "screen_report.json")
    json.dump(rep, open(rp, "w"), indent=1, ensure_ascii=False, default=float)
    print("\n" + "=" * 78)
    if passed:
        print(f"【结论】{len(passed)}/{len(priors)} 个候选通过全部三关: {', '.join(passed)}")
        print(f"  用法: --grasp_prior {out_dir}/<tag>.npz")
        print(f"  ⚠ 训练前把该候选的 Gate 1 ' 用哪个yaw ' 钉进摆放, 否则加载器会自己搜 IK 最优 yaw")
    else:
        print("【结论】本批全部不通过 —— 按规则整批丢弃, 重新生成一批。")
    print(f"  详细报告: {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
