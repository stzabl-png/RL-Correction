"""Unscrew v1 母带离线构建器 (纯 numpy + ArmIK, 不开 Isaac)。

  UNSCREW_CLIP=32 PYTHONPATH=. python tasks/Unscrew/part4/A_Design/L2_Reference/make_reference.py

产出 A_Design/L2_Reference/<clip>/reference_v1.npz, schema 与 Pour17 v1 相同
(right_q/left_q(T,7) right_f/left_f(T,22) obj_pos/quat_{0,1} conf_pos/rot_{0,1}
 source frame_of_row seg_lens seg_names fin_names meta) + 本任务新增列:
  hand_conf_pos_{r,l}(T,)      人手腕位置置信度 (0-100, 机器行 NaN)
  hand_conf_fin_{r,l}(T,)      五指置信度均值 (形状指引奖的门控原料)
obj 列约定: obj_0=瓶身(env.object,左手) obj_1=盖(env.aux,右手) —— 与数据集编号
无关, 按 mesh 尺寸认件 (17/18 条里 6 条 object_0 其实是盖的教训, 见 clips.py)。

== 构建规则 (全部承 Pour17 拼装规则 + 旧扭盖台账实证配方) ==
1. 交互窗 = [两物体最早运动起始帧, 运动结束帧] (scene_layout onset + 尾部静止检测)
2. 物体换基 (Pour17 L2-6): 平移=场景静置位+原始增量; 朝向=原始世界系增量∘静置姿
   (对接 yaw 不污染轨迹 —— 用瓶的任意 yaw 转整条轨迹是 2026-08-15 已修过的坑)
3. 瓶姿态 U24a 直立投影混合 (<22° 全投影 / >40° 全原始): 重建静置帧 ~21°
   FoundationPose 噪声 > 平底圆柱 18.3° 倾倒极限, 不投则瓶必自倒 (旧台账总根因)
4. 盖行: 脱离前由瓶推导 (瓶行+closed_offset·螺轴, 合拢期盖自身轨迹更噪且冗余);
   脱离帧从数据测 (盖轨迹与装配位偏离首超 4cm); 脱离后接盖自身重建轨迹 (连续拼接)
5. 腕参考**死通道零依赖** (2026-08-30 拍板: 上游 wrist_pos 是静态填充,
   不使用): 右腕 P1->到盖的位置曲线来自活的 MANO 掌根代理，用端点受约束
   相似变换重定向到机器人的 P1/GraspPose；姿态增量来自活的 wrist_quat。
   到盖后右腕刚性锚在盖行上;
   左腕 = Screw27_body GraspPose **镜像**锚在瓶行上 (同 CAD 字节相同,
   绕瓶轴方位角 ArmIK 可达率扫描), 指形 = prior 抓形模板 + env squeeze 加压。
   ArmIK 逐行解臂 (anchor 用 env_rest.json 实测 arm_center, 缺省退 URDF 推导
   —— 有 ~cm 级系统差, v2 重铸时被增量空间锚消掉)
6. 分段契约: P0->P1 = cuRobo Approach；P1->任务末行 = 一条连续数据/RL 轨迹，
   中间不插入机器规划；任务末行->P0 = cuRobo Retreat。没有规划产物时只生成
   smoothstep 诊断占位（⚠ 无碰撞背书，正式训练预检必须拒绝）。
"""
from __future__ import annotations

import argparse
import hashlib
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))

import task_config as TC  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, R_to_quat, quat_to_R  # noqa: E402

TABLE_Z = 0.87
OBJ_GAP = 0.01
APP_ROWS, SEAM1, SEAM2, RET_ROWS = 60, 25, 25, 60
# SEAM1 定档 25 (2026-08-31 实测): 拉到 40 行 (合拢 0.8s) 反而在**进刀段末尾**
# 就把瓶碰倒 —— 这一段的净空只有约 1cm, 手在瓶边多停留反而更容易蹭上。25 行
# 是实测"全程不倒 + 站位 4 垫"的那一组。行数不进规划摘要, 改它不必重规划。
CAP_CLEARANCE = 0.015                       # U35b: 盖顶净空
HAND_DROP = TC.HAND_DROP                    # 腕-指尖垂距 (probe_grasp 实测, 逐手)
SEP_THRESH = 0.04                           # 盖脱离帧: 与装配位偏差首超 4cm


def qmul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate(
        [aw * bw - (av * bv).sum(-1, keepdims=True),
         aw * bv + bw * av + np.cross(av, bv)], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q, v):
    """batch (T,4) wxyz 作用 (T,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)],
        axis=-2)
    return np.einsum("tij,tj->ti", R, v)


def smooth(a, sigma=1.5):
    if sigma <= 0:
        return a
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(a, float), sigma=sigma, axis=0, mode="nearest")


def cont(q):
    """四元数符号连续化."""
    q = np.asarray(q, float).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def smooth_quats(q, sigma=2.0):
    q = smooth(cont(q), sigma)
    return q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)


def upright_project(q_seq):
    """U24a: 倾角<22° 全投影到 yaw-only, >40° 全原始, 线性过渡."""
    q = cont(q_seq)
    r33 = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    tilt = np.degrees(np.arccos(np.clip(r33, -1, 1)))
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                     1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    q_up = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)
    w = np.clip((tilt - 22.0) / 18.0, 0.0, 1.0)[:, None]
    sgn = np.where((q_up * q).sum(1, keepdims=True) < 0, -1.0, 1.0)
    out = (1 - w) * q_up * sgn + w * q
    return out / np.linalg.norm(out, axis=1, keepdims=True), tilt


def motion_window(P, base_n=12):
    """(T,3) -> 尾部静止检测: 最后一个持续运动帧."""
    d = np.linalg.norm(P[3:] - P[:-3], axis=1)
    thr = max(0.008, float(np.percentile(d[:base_n], 25)) * 3.0)
    mv = np.flatnonzero(d > thr)
    return int(mv[-1]) + 3 + 2 if len(mv) else len(P) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rest-json", default=TC.REST_JSON,
                    help="probe_rest.py 产物 (env 实测静置位/anchor_T/站姿)")
    ap.add_argument(
        "--allow-offline-rest", action="store_true",
        help="显式允许无 env_rest 时用离线估计；只供初始脚手架，不可用于正式母带")
    ap.add_argument("--out", default=TC.REF_V1)
    args = ap.parse_args()
    if not os.path.isfile(args.rest_json) and not args.allow_offline_rest:
        raise FileNotFoundError(
            f"缺少 {args.rest_json}; 先运行 C_Wiring/probe_rest.py")
    take = TC.TAKE_DIR
    print(f"[v1] clip={TC.CLIP_ID} take={take}")

    lay = json.load(open(os.path.join(take, "scene_layout.json")))
    objs = lay["objects"]
    by_size = sorted(objs, key=lambda o: -max(objs[o]["extent_cm"]))
    BODY_ID, CAP_ID = by_size[0], by_size[-1]
    oids_order = None

    r = np.load(os.path.join(take, "replay_world.npz"), allow_pickle=True)
    oids_order = [str(x) for x in r["object_ids"]]
    bi, ci = oids_order.index(BODY_ID), oids_order.index(CAP_ID)
    OP = np.asarray(r["obj_pose_all"], np.float64)            # (2,T,7)
    T_src = OP.shape[1]
    # 四元数序核验: 与 world_fused 的旋转矩阵交叉对账 (wxyz 应逐位一致)。
    # 序判据用**对得上的那个物体** (T2-18): replay 管线会对弱重建物体做绕自身轴
    # 的自转正则化 (clip32 是盖、clip17 是瓶身 recon 11/101 帧; 实测两份数据
    # 倾角逐帧一致、差全在自转 —— 旋转体自转本是规范自由度), 修正过的物体
    # 天然与 fused 不逐位一致, 旧版只拿瓶身判序在 clip17 上被误伤。
    w = np.load(os.path.join(take, "world_fused.npz"), allow_pickle=True)
    M = np.asarray(w["object_ob_in_world_all"], np.float64)   # (2,T,4,4)
    _errs = []
    for _oi in range(OP.shape[0]):
        _q0 = OP[_oi, 0, 3:7]
        _errs.append((np.abs(quat_to_R(_q0) - M[_oi, 0, :3, :3]).max(),
                      np.abs(quat_to_R(np.r_[_q0[3], _q0[:3]])
                             - M[_oi, 0, :3, :3]).max()))
    _ew, _ex = min(_errs, key=lambda t: t[0])
    assert _ew < _ex and _ew < 0.02, \
        f"replay obj_pose_all 四元数序存疑: per-object (wxyz,xyzw) = {_errs}"
    for _oi, (_e, _) in enumerate(_errs):
        if _e >= 0.02:
            print(f"[v1] ⚠ replay 对 {oids_order[_oi]} 有绕轴自转正则化 "
                  f"(f0 旋转差元素峰 {_e:.3f}); 物体轨迹以 replay 为准")

    body_raw_p, body_raw_q = OP[bi, :, :3], cont(OP[bi, :, 3:7])
    cap_raw_p, cap_raw_q = OP[ci, :, :3], cont(OP[ci, :, 3:7])

    # ---- 交互窗 ----
    w0 = min(int(objs[BODY_ID]["onset_frame"]), int(objs[CAP_ID]["onset_frame"]))
    w1 = min(max(motion_window(body_raw_p), motion_window(cap_raw_p)), T_src - 1)
    N = w1 - w0 + 1
    print(f"[v1] 交互窗 f{w0}..f{w1} ({N} 行; body onset "
          f"{objs[BODY_ID]['onset_frame']} cap onset {objs[CAP_ID]['onset_frame']})")

    # ---- env 静置位 (probe 实测优先) ----
    rest = None
    if os.path.isfile(args.rest_json):
        rest = json.load(open(args.rest_json))
        if str(rest.get("clip")) != TC.CLIP_ID:
            raise ValueError(f"env_rest clip={rest.get('clip')} != {TC.CLIP_ID}")
        print(f"[v1] env 静置位: {args.rest_json} (实测)")
    else:
        print(f"[v1] ⚠ 无 {args.rest_json} —— 静置位用离线估计, "
              f"冒烟 A 项 (<5mm) 大概率过不了; 先跑 C_Wiring/probe_rest.py")

    # 瓶: U24a 投影
    bq_proj, tilt = upright_project(body_raw_q)
    print(f"[v1] U24a 投影: 首帧倾角 {tilt[0]:.1f}° | 全投影 "
          f"{int((tilt < 22).sum())}/{T_src} 帧")
    # ---- T2-12 右手启动帧: 从原始示范反推, 不用交互窗首行代替 ----
    # wrist_pos 是静态坏道; 用 MANO 掌根 0/5/9/13/17 的均值作稳健右掌代理。
    # 基线取右侧接触候选前最多 12 帧, 并要求瓶身已到水平阈值。clip32 得 f35:
    # 原始瓶倾角 87.6°, 右掌相对基线 4.6mm, 与视觉裁定一致。
    _jr = np.asarray(r["joints_right"], np.float64)
    if _jr.ndim != 3 or _jr.shape[1] < 18:
        raise ValueError("replay_world.joints_right 缺少 MANO 21 点, 无法定右手启动帧")
    _palm_r = _jr[:, [0, 5, 9, 13, 17]].mean(axis=1)
    _contact_r = int(objs[CAP_ID].get(
        "contact_start", objs[CAP_ID].get("onset_frame", w0)))
    _b0 = max(w0, _contact_r - 12)
    _palm_base = np.median(_palm_r[_b0:max(_contact_r, _b0 + 1)], axis=0)
    _palm_disp = np.linalg.norm(_palm_r - _palm_base, axis=1)
    _right_ok = np.flatnonzero(
        (np.arange(T_src) >= _contact_r)
        & (tilt >= float(TC.RIGHT_START_TILT_DEG))
        & (_palm_disp >= float(TC.RIGHT_START_PALM_DISP_M)))
    if not len(_right_ok):
        raise ValueError(
            "原始数据找不到‘瓶身水平且右掌已启动’帧; "
            f"tilt>={TC.RIGHT_START_TILT_DEG}deg, "
            f"palm_disp>={TC.RIGHT_START_PALM_DISP_M * 1000:.1f}mm")
    right_start_src = int(_right_ok[0])
    if not w0 <= right_start_src <= w1:
        raise ValueError(f"右手启动帧 f{right_start_src} 不在交互窗 f{w0}..f{w1}")
    right_start_tilt = float(tilt[right_start_src])
    right_start_palm_disp = float(_palm_disp[right_start_src])
    # 数据里 f35..f60 是连续水平平台，f61 开始回竖。右手的“找盖”轨迹必须
    # 在这段平台内完成；取平台最后一帧作到盖/合指锚，不凭空另造机器段。
    right_grasp_src = right_start_src
    while (right_grasp_src + 1 < T_src
           and tilt[right_grasp_src + 1] >= float(TC.RIGHT_START_TILT_DEG)):
        right_grasp_src += 1
    if right_grasp_src <= right_start_src:
        raise ValueError("瓶身没有足够长的水平平台供右手数据轨迹找盖")
    print(f"[v1] T2-12 时序锚: 右手最早 f{right_start_src} 启动 | "
          f"瓶倾角 {right_start_tilt:.1f}° | 右掌位移 "
          f"{right_start_palm_disp * 1000:.1f}mm (接触候选 f{_contact_r}); "
          f"水平平台到 f{right_grasp_src}")
    # U35c 移植: 携带段持瓶朝向 yaw 重定向 (人举瓶的朝向对人顺手, 对机器人
    # 肘几何可能不可达 —— 旧 clip 实测轴向对握要求肘超臂长, IK 塌方; 转 -35°
    # 后 79.7%→100%)。幅度随倾角权重渐入 (静置 0, 深倾斜满额), 位置不动;
    # 所有下游几何 (螺轴/盖行/腕锚/左握位) 从同一 body_q 派生, 自动一致。
    # 每条 clip 用 UNSCREW_HOLD_YAW 覆写 (probe_ikcheck 出数后定, 默认 0)。
    import trimesh                       # 物体系对齐要用 mesh bbox
    hold_yaw = TC.HOLD_YAW_DEG          # 逐 clip 标定值 (UNSCREW_HOLD_YAW 可覆写)
    if hold_yaw:
        wmix = np.clip((tilt - 22.0) / 18.0, 0.0, 1.0)
        hy = np.deg2rad(hold_yaw) * wmix
        qy = np.stack([np.cos(hy / 2), 0 * hy, 0 * hy, np.sin(hy / 2)], 1)
        bq_proj = qmul(qy, bq_proj)
        bq_proj /= np.linalg.norm(bq_proj, axis=1, keepdims=True)
        print(f"[v1] U35c yaw 重定向: {hold_yaw:+.0f}° × 倾角权重 "
              f"(受影响 {int((wmix > 0).sum())} 帧)")
    # T2-24 横瓶角度上限 (2026-09-02 用户裁定"适度少转一点"): 演示把瓶转到
    # ~88°, 盖嘴几乎贴桌, 右手被压进"盖-桌薄层"(T2-23 实测可达滚转角腕全在
    # 桌面高度)。把超限的倾角部分绕 up×z 轴转回来 (自转/yaw 不动), 盖嘴抬离
    # 桌面还右手净空。右手启动/平台判据用的是上面已算好的原始 tilt, 不受影响;
    # 盖行/腕锚/左握位全部从同一 body_q 派生, 自动一致 (与 U35c 同法)。
    _max_tilt = float(os.environ.get("UNSCREW_MAX_TILT", "0"))
    if _max_tilt > 0:
        _upv = qrot(bq_proj, np.tile([0.0, 0.0, 1.0], (len(bq_proj), 1)))
        _th = np.degrees(np.arccos(np.clip(_upv[:, 2], -1.0, 1.0)))
        _exc = np.radians(np.clip(_th - _max_tilt, 0.0, None))
        if (_exc > 0).any():
            _axc = np.cross(_upv, np.tile([0.0, 0.0, 1.0], (len(bq_proj), 1)))
            _axn = np.linalg.norm(_axc, axis=1, keepdims=True)
            _axc = np.where(_axn > 1e-8, _axc / np.maximum(_axn, 1e-8),
                            np.array([1.0, 0.0, 0.0]))
            _qc = np.concatenate([np.cos(_exc / 2)[:, None],
                                  np.sin(_exc / 2)[:, None] * _axc], axis=1)
            bq_proj = qmul(_qc, bq_proj)
            bq_proj /= np.linalg.norm(bq_proj, axis=1, keepdims=True)
            print(f"[v1] T2-24 倾角上限 {_max_tilt:.0f}°: 受影响 "
                  f"{int((_exc > 0).sum())} 帧 (原峰值 {_th.max():.1f}°)")
    if rest:
        rest_b = np.asarray(rest["body_pose"], float)         # env 系 (7,)
        rest_c = np.asarray(rest["cap_pose"], float)
        rest_axis = quat_to_R(rest_b[3:7])[:, 2]
        rest_rel = rest_c[:3] - rest_b[:3]
        rest_axial = float(np.dot(rest_rel, rest_axis))
        rest_radial = float(np.linalg.norm(
            rest_rel - rest_axial * rest_axis))
        if abs(rest_axial - 0.18) >= 0.005 or rest_radial >= 0.005:
            raise ValueError(
                f"env_rest screw closure invalid: axial={rest_axial:.4f}m "
                f"(expected 0.1800), radial={rest_radial:.4f}m; "
                "rerun C_Wiring/probe_rest.py")
    else:
        mesh_b = trimesh.load(objs[BODY_ID]["mesh"], process=False, force="mesh")
        Vb = np.asarray(mesh_b.vertices)
        q0 = bq_proj[w0]
        zmin = float((quat_to_R(q0) @ Vb.T).T[:, 2].min())
        rest_b = np.r_[0.0, 0.15, TABLE_Z + OBJ_GAP - zmin, q0]
        axis0 = quat_to_R(q0)[:, 2]
        rest_c = np.r_[rest_b[:3] + 0.18 * axis0, q0]

    # ---- 物体行 (交互窗内, env 系) ----
    # 平移: 静置位 + 原始增量; 朝向: 原始世界系增量 ∘ 静置姿
    dq_b = qmul(bq_proj[w0:w1 + 1], np.tile(qconj(bq_proj[w0]), (N, 1)))
    body_q = qmul(dq_b, np.tile(rest_b[3:7], (N, 1)))
    body_q /= np.linalg.norm(body_q, axis=1, keepdims=True)
    body_p = rest_b[None, :3] + (body_raw_p[w0:w1 + 1] - body_raw_p[w0])
    body_p = smooth(body_p, 1.0)
    body_q = smooth_quats(body_q, 1.0)
    axis = qrot(body_q, np.tile([0.0, 0.0, 1.0], (N, 1)))     # 螺轴 (env 系)

    # 盖脱离帧: **瓶体系**里盖相对位置的变化首超 4cm。
    # ⚠ 不能用"投影轴×装配位"口径: 重建瓶姿态带 ~20° 噪声, 投影轴与原始盖位
    #   相差 0.18·sin(20°)≈6cm, 会把脱离帧误报到交互首行 (clip32 实测 6.6cm)。
    #   体系内相对量对瓶姿态系统差鲁棒 (两物体同一坐标系测量)。
    def _q2Rb(q):
        w_, x_, y_, z_ = q / np.linalg.norm(q)
        return np.array([
            [1 - 2*(y_*y_+z_*z_), 2*(x_*y_-w_*z_), 2*(x_*z_+w_*y_)],
            [2*(x_*y_+w_*z_), 1 - 2*(x_*x_+z_*z_), 2*(y_*z_-w_*x_)],
            [2*(x_*z_-w_*y_), 2*(y_*z_+w_*x_), 1 - 2*(x_*x_+y_*y_)]])
    rel_b = np.stack([_q2Rb(body_raw_q[t]).T @ (cap_raw_p[t] - body_raw_p[t])
                      for t in range(T_src)])
    rel0 = np.median(rel_b[w0:w0 + 5], axis=0)
    sep_d = np.linalg.norm(rel_b - rel0, axis=1)
    hit = np.flatnonzero((sep_d > SEP_THRESH)[w0:]) + w0
    geom_sep_src = int(hit[0]) if len(hit) else w1
    geom_sep_src = int(np.clip(geom_sep_src, w0 + 1, w1))
    # 几何重建在 f35 就把盖报成脱离，但此时右手才刚开始找盖；物理上盖不可能
    # 在手到达前自行离瓶。至少钉到水平平台末 f60（数据轨迹的到盖锚）再允许脱离。
    sep_src = max(geom_sep_src, right_grasp_src)
    k_sep = sep_src - w0
    print(f"[v1] 盖脱离帧: 几何候选 f{geom_sep_src} -> 时序约束 f{sep_src} "
          f"(交互行 {k_sep}) | "
          f"脱离前瓶系相对漂移中位 {np.median(sep_d[w0:sep_src]) * 100:.1f}cm")

    # ---- 瓶身自转规范化 (脱离后冻结) ----
    # 与右腕同一条依据: 瓶也是旋转体, 判据侧只看轴倾角 (placed/_axis_tilt),
    # 绕自身轴的自转在**脱离后**既不可判也无任务含义 —— 而演示里人拧完还在
    # 继续转瓶, 左手抓点就跟着绕瓶公转到瓶的另一侧: 左臂要"绕过瓶子"才能跟,
    # 实测搬运段左腕目标 6~21cm 不可达 (多重启也解不出来), 整段左臂冻死。
    # 脱离前不动 (那段的相对自转 = 螺纹相位, 是任务本身)。
    def _twist_free(q_seq, k0):
        """k0 之后只保留**轴的摆动**, 自转平行输运 (轴逐位不变)。

        做法: 用"把上一行的轴最小旋转搬到本行的轴"的那个旋转去推进 —— 轴按
        原样复现, 绕轴的自转不再累积。⚠ 不能改成"逐步剔掉局部 z 增量再积分":
        剔掉自转后后续的摆动是在**另一个自转相位**的局部系里施加的, 轴会跟着
        跑偏 (单元测试实测轴差 0.74)。
        """
        out = q_seq.copy()
        ez = np.array([[0.0, 0.0, 1.0]])
        ax = qrot(q_seq, np.tile([0.0, 0.0, 1.0], (len(q_seq), 1)))
        ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
        for k in range(k0 + 1, len(q_seq)):
            a0, a1 = ax[k - 1], ax[k]
            c = float(np.clip(np.dot(a0, a1), -1.0, 1.0))
            v = np.cross(a0, a1)
            nv = float(np.linalg.norm(v))
            if nv < 1e-9:
                dq = np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else None
                if dq is None:                     # 180° 反向: 取任意垂直轴
                    perp = np.array([1.0, 0.0, 0.0])
                    if abs(a0[0]) > 0.9:
                        perp = np.array([0.0, 1.0, 0.0])
                    v = np.cross(a0, perp)
                    v /= np.linalg.norm(v)
                    dq = np.array([0.0, v[0], v[1], v[2]])
            else:
                ang = np.arctan2(nv, c)
                v = v / nv
                dq = np.concatenate([[np.cos(0.5 * ang)],
                                     np.sin(0.5 * ang) * v])
            out[k] = qmul(dq[None], out[k - 1:k])[0]       # 世界系左乘
            out[k] /= np.linalg.norm(out[k])
        _chk = qrot(out, np.tile([0.0, 0.0, 1.0], (len(q_seq), 1)))
        assert np.abs(_chk - ax).max() < 1e-6, "自转冻结改变了螺轴 (实现错)"
        del ez
        return out

    _spin_before = np.degrees(np.arccos(np.clip(np.abs(
        (body_q[k_sep] * body_q[-1]).sum()), -1, 1))) * 2
    body_q = _twist_free(body_q, k_sep)
    axis = qrot(body_q, np.tile([0.0, 0.0, 1.0], (N, 1)))     # 螺轴 (自转无关)
    print(f"[v1] 瓶自转冻结 (脱离行 {k_sep} 起): 原演示末行相对脱离行转过 "
          f"{_spin_before:.0f}°, 现只保留轴向摆动 (自转不可判)")

    # 盖行: 脱离前=瓶推导 (盖钉在瓶顶, 与螺旋投影同语义); 脱离后=盖自身轨迹
    # 按**世界平移**换基 (与瓶同一平移 —— 保住"盖终点落在桌面"的不变量:
    # 若按脱离点连续性拼接, U24a 投影造成的座位高度差会把整段送放轨迹整体
    # 压低 ~5cm, 盖末行 z 掉破 D1 死线, 放音自检实锤)。脱离窗 8 行线性混合
    # 消拼接跳变 (该窗 conf 本来就低, 皮筋红档容差吸收)。
    cap_p = body_p + 0.18 * axis
    cap_q = body_q.copy()
    if k_sep < N - 1:
        shift_w = rest_b[:3] - body_raw_p[w0]          # 与瓶同一世界平移
        cap_free = cap_raw_p[w0:w1 + 1] + shift_w
        BL = min(8, N - 1 - k_sep)
        for j in range(BL):
            u = (j + 1) / BL
            cap_p[k_sep + j] = (1 - u) * cap_p[k_sep + j] + u * cap_free[k_sep + j]
        cap_p[k_sep + BL:] = cap_free[k_sep + BL:]
        dq_c = qmul(cap_raw_q[sep_src:w1 + 1],
                    np.tile(qconj(cap_raw_q[sep_src]), (N - k_sep, 1)))
        cap_q[k_sep:] = qmul(dq_c, np.tile(cap_q[k_sep], (N - k_sep, 1)))
        cap_q /= np.linalg.norm(cap_q, axis=1, keepdims=True)
    cap_p = smooth(cap_p, 1.5)
    # 盖不许低于"平躺在桌面"的中心高 (送放段重建噪声会往下扎)
    cap_p[:, 2] = np.maximum(cap_p[:, 2], TABLE_Z + 0.006)
    cap_q = smooth_quats(cap_q, 1.5)

    def _mk_ik(side):
        if rest and rest.get(f"anchor_T_{side}"):
            return ArmIK(side, anchor_link="arm_center",
                         anchor_T=np.asarray(rest[f"anchor_T_{side}"], float))
        return ArmIK(side, torso_deg={"torso_j1": 40.5196, "torso_j2": 73.6595,
                                      "torso_j3": 0.3896})

    # ---- 腕参考 (交互窗) ----
    # ⚠ 死通道零依赖 (2026-08-30): ref_qpos 的 wrist_pos 是上游静态填充的
    # 死数据 (clip32 腕 <0.6cm/盖 66cm), 本构建器只消费**活通道**:
    # 右腕四元数流 / 右手指流 / 物体轨迹+conf / 人手逐帧置信度。
    qr = np.load(os.path.join(take, "ref_qpos_right.npz"), allow_pickle=True)
    fin_names = [str(n) for n in qr["joint_names"]]
    assert all(n.startswith("right_") for n in fin_names), fin_names[:3]

    # ---- 右腕 = **盖 GraspPose 先验锚在盖行上** (2026-09-01, 与左手对称) ----
    # 旧法是几何构造 (盖心 + reach·螺轴 + 掌轴对准 + 人手指流), 没有抓取先验 ——
    # probe_pinch 实测三指落在盖系径向 6.4~11.2cm (盖半径 1.75cm), 合到 45° 仍
    # 零接触, 食/中指越合越往外: 缺的是"人手抓法 -> 这只手关节角"的那一层。
    # Screw27_cap 候选本来就是**右手**约定 (不镜像), 接触点半径 1.9cm 落在盖面上。
    _mc = trimesh.load(objs[CAP_ID]["mesh"], process=False, force="mesh")
    _zc = np.asarray(_mc.vertices)[:, 2]
    _cap_dz = float(0.5 * (_zc.min() + _zc.max()) - _zc.min())   # 物体系原点修正
    _cands = sorted(glob.glob(os.path.join(TC.PRIOR_CAP_DIR, "*.npz")))
    if TC.CAP_GRASP_PICK:
        _cands = [c for c in _cands
                  if os.path.basename(c).startswith(TC.CAP_GRASP_PICK)] or _cands
    assert _cands, f"没有盖抓取候选: {TC.PRIOR_CAP_DIR}"
    ik_r0 = _mk_ik("right")

    def _cap_track(cz, roll_deg=0.0, z_from=0, z_to=None):
        """候选 -> (腕位轨迹, 腕姿轨迹): 抓取位姿刚性锚在盖行上。

        ⚠ 这批候选的来源是 `bottle_cap_sharpa_wave_**left**` —— 是**左手**抓法,
        给右手用必须镜像 (与左手用 Screw27_body 右手约定时要镜像, 方向相反但
        同一变换): 位姿 (x,-y,z)/(w,-x,y,-z), 指值按名映射。不镜像直接套上去,
        实测指垫落在盖系径向 4.9~9.9cm (盖半径 1.75cm), 合到 45° 也碰不到。
        """
        gp = np.asarray(cz["grasp"], np.float64)[:3].copy()
        gq = np.asarray(cz["grasp"], np.float64)[3:7].copy()
        gp[1] = -gp[1]                                  # 镜像: 左手抓法 -> 右手
        gq = np.array([gq[0], -gq[1], gq[2], -gq[3]])
        gq /= np.linalg.norm(gq)
        gp[2] += _cap_dz
        gp += np.asarray(TC.CAP_GRASP_TRIM, np.float64)   # 闭环实测的对准量
        # 盖是旋转体: 整个抓取变换可绕盖轴转动而不改接触半径/高度。T2-12 后
        # 右手是在瓶已水平时才进场, 因而方位必须以该帧的机器人可达性来选，
        # 不能继续沿用直立首帧的方位。
        _a = np.radians(float(roll_deg))
        _qz = np.array([np.cos(_a / 2), 0.0, 0.0, np.sin(_a / 2)])
        gp = quat_to_R(_qz) @ gp
        gq = qmul(_qz[None], gq[None])[0]
        Pp = cap_p + qrot(cap_q, np.tile(gp, (N, 1)))
        Qq = qmul(cap_q, np.tile(gq / np.linalg.norm(gq), (N, 1)))
        Qq /= np.linalg.norm(Qq, axis=1, keepdims=True)
        # 净空窗口 = [z_from, z_to) = 启动行..脱盖行 —— 之前的行被 P1->f58
        # 重定向覆盖; 之后的行被 rigid-ride 携带段重建 (原始盖姿态是翻滚垃圾),
        # 两头算进去都会让所有滚转角不及格 (T2-23 前两版之误)。
        raw_min_z = float(Pp[z_from:z_to, 2].min())
        # 钳位 +2cm -> +5cm (T2-25 终版, 用户视频: 右臂/拳握指节在护送段轻微
        # 蹭桌 —— 腕距桌 2cm 时指节只剩指厚)。80° 倾角上限后抓取窗最低 0.917,
        # 该钳位实际只在护送段生效; 盖已在手里, 抬 3cm 仅轻微弯曲冻结偏移。
        Pp[:, 2] = np.maximum(Pp[:, 2], TABLE_Z + 0.05)
        return Pp, Qq, raw_min_z

    _best_c = None
    _ka = int(right_start_src - w0)
    _kb = int(right_grasp_src - w0)
    for _cf in _cands:
        _cz = np.load(_cf)
        # 手内"指向"方向 (腕->盖, 腕系), 滚转不变量: 用于桌面净空评分 (T2-23)。
        _gpm = np.asarray(_cz["grasp"], np.float64)[:3].copy()
        _gqm = np.asarray(_cz["grasp"], np.float64)[3:7].copy()
        _gpm[1] = -_gpm[1]
        _gqm = np.array([_gqm[0], -_gqm[1], _gqm[2], -_gqm[3]])
        _gqm /= np.linalg.norm(_gqm)
        _fdir_local = quat_to_R(_gqm).T @ (-_gpm)
        _fdir_local /= max(np.linalg.norm(_fdir_local), 1e-9)
        _roll_pin = os.environ.get("UNSCREW_CAP_ROLL", "")
        for _roll_deg in ([int(_roll_pin)] if _roll_pin
                          else range(-180, 180, 15)):
            _P, _Q, _rawz = _cap_track(_cz, _roll_deg, z_from=_ka,
                                       z_to=_kb + 1)
            # 护送段净空: 腕以脱盖行的世界系偏移刚性跟盖 (位置用重建, 可信),
            # 该段真实最低 z = min(盖位) + 脱盖行偏移的 z 分量。
            _esc_minz = (float(cap_p[_kb:, 2].min())
                         + float(_P[_kb, 2] - cap_p[_kb, 2]))
            _ss = ik_r0.solve_traj(_P[[_ka, _kb]], _Q[[_ka, _kb]],
                                    w_rot=0.25, n_restart=24)
            _pe2 = max(float(x["pos_err"]) for x in _ss)
            _re2 = max(float(x["rot_err"]) for x in _ss)
            _m0 = min(float(np.minimum(x["q"] - ik_r0.lower,
                                       ik_r0.upper - x["q"]).min()) for x in _ss)
            _st_ok = (_pe2 < 0.02 and _re2 < np.radians(10.0)
                      and _m0 > np.radians(3.0))
            # 桌面净空 (T2-23, 用户视频: 腕被 TABLE_Z+2cm 钳位兜住、三指插桌):
            # ① 腕轨迹真实 z 不得靠钳位救 (>=桌面+4cm);
            # ② 指向 (腕->盖) 在抓盖行不得朝下超过 ~20° —— 指尖比腕更低,
            #    朝下的滚转方位必然把下方手指插进桌面。
            _clear = int(min(_rawz, _esc_minz) >= TABLE_Z + 0.04)
            _fd_w = quat_to_R(_Q[_kb]) @ _fdir_local
            _fd_ok = int(float(_fd_w[2]) >= -0.35)
            # 找盖起点与水平平台末端都必须可达；可达里先比净空, 再比限位余量。
            _key = (1 if _st_ok else 0, _clear + _fd_ok, _m0,
                    -float(_pe2 + 0.25 * _re2))
            if os.environ.get("UNSCREW_SCAN_DEBUG"):
                print(f"[scan] {os.path.basename(_cf)} roll={_roll_deg:+4d} "
                      f"st={int(_st_ok)} clear={_clear} fd_z={_fd_w[2]:+.2f} "
                      f"margin={np.degrees(_m0):5.1f}° "
                      f"dz_kb={_P[_kb, 2] - cap_p[_kb, 2]:+.3f} "
                      f"err={_pe2 * 100:.1f}cm/{np.degrees(_re2):.0f}°")
            if _best_c is None or _key > _best_c[0]:
                _best_c = (_key, _cf, _cz, _pe2,
                           float(np.degrees(_re2)),
                           np.asarray(_ss[0]["q"], np.float64),
                           np.asarray(_ss[1]["q"], np.float64), _roll_deg)
    (_key, _cf, _capz, _e0, _r0d, _q_start_scan_r, _q_grasp_r,
     _grasp_roll_deg) = _best_c
    print(f"[v1] 右抓候选/绕盖轴扫描 ({len(_cands)}×24): 选中 "
          f"{os.path.basename(_cf)} roll={_grasp_roll_deg:+.0f}° | "
          f"水平 f{right_start_src}/f{right_grasp_src} 最坏 "
          f"{_e0 * 100:.2f}cm/{_r0d:.1f}° "
          f"限位余量 {np.degrees(_key[1]):.1f}° "
          f"({'可达' if _key[0] else '⚠ 不可达'}) | 盖系原点修正 "
          f"{_cap_dz * 100:+.2f}cm")
    wr_P, wr_Q, _rawz_pick = _cap_track(_capz, _grasp_roll_deg, z_from=_ka,
                                        z_to=_kb + 1)
    _esc_pick = (float(cap_p[_kb:, 2].min())
                 + float(wr_P[_kb, 2] - cap_p[_kb, 2]))
    print(f"[v1] 右腕桌面净空 (T2-23): 抓取窗最低 z = {_rawz_pick:.3f} | "
          f"护送段最低 z = {_esc_pick:.3f} (桌面 {TABLE_Z:.3f}; "
          f"任一 <{TABLE_Z + 0.04:.3f} 说明净空评分被迫妥协)")
    # 脱离后: 腕姿态**冻结**在脱离时刻, 位置保持与盖的世界系刚性偏移。
    # 理由: 盖脱手后由手携带 (下面的 rigid ride 让盖姿态跟手走), 而重建的盖
    # 自身翻滚 107° 是不可信的自转/翻滚 —— 腕跟着刚性翻会直接超出可达域
    # (实测全程可达率掉到 12%、姿态中位差 89°)。冻结姿态 + 位置跟盖 = 携带段
    # 平滑且可达, 盖的落地姿态由脱离时的握法决定 (物理上就该如此)。
    if k_sep < N - 1:
        _off_w = wr_P[k_sep] - cap_p[k_sep]
        wr_Q[k_sep:] = wr_Q[k_sep]
        wr_P[k_sep:] = cap_p[k_sep:] + _off_w
        # T2-28 放置摆转: 侧抓 (腕在盖轴下方, 水平自然拧) 的刚性偏移在放置段
        # 会把腕目标压到桌下、被 z 钳位截断 => 盖看起来自己沉出指间。人的动作
        # 是拧完翻腕把盖口朝下再放: 脱盖后绕盖轴把抓握方位限速摆到放置方位。
        # 盖是旋转体, 摆转对盖只是绕自身轴自转 (自转本就不可判), 物理无害;
        # 摆完的放置几何与顶抓方位完全一致 (已验证可行的形态)。
        _pl_roll = os.environ.get("UNSCREW_PLACE_ROLL", "")
        if _pl_roll:
            _sw_delta = np.radians(float(_pl_roll) - _grasp_roll_deg)
            _sw_axis = quat_to_R(cap_q[k_sep]) @ np.array([0.0, 0.0, 1.0])
            _sw_axis = _sw_axis / max(np.linalg.norm(_sw_axis), 1e-9)
            # smoothstep 峰值斜率 1.5 => 行数按峰值限速折算。默认 6°/行同自转
            # 漂移; 大角度摆转 (翻掌需 ~180°) 允许放宽 (20Hz 下 9°/行 ≈ 3.1rad/s)
            _sw_rate = np.radians(float(os.environ.get(
                "UNSCREW_SWING_RATE", "6.0")))
            _sw_n = int(np.ceil(1.5 * abs(_sw_delta) / _sw_rate)) + 1
            _sw_n = max(2, min(_sw_n, N - 2 - k_sep))
            _u = np.clip(np.arange(N - k_sep) / float(_sw_n), 0.0, 1.0)
            _ang = (_u * _u * (3.0 - 2.0 * _u)) * _sw_delta
            _h = 0.5 * _ang
            _qsw = np.concatenate(
                [np.cos(_h)[:, None], np.sin(_h)[:, None] * _sw_axis[None]], 1)
            wr_P[k_sep:] = cap_p[k_sep:] + qrot(
                _qsw, np.tile(_off_w, (N - k_sep, 1)))
            wr_Q[k_sep:] = qmul(_qsw, wr_Q[k_sep:])
            print(f"[v1] T2-28 放置摆转: roll {_grasp_roll_deg:+.0f}° -> "
                  f"{float(_pl_roll):+.0f}° (绕盖轴 {np.degrees(_sw_delta):+.0f}°"
                  f", {_sw_n} 行, 峰值 "
                  f"{np.degrees(1.5 * abs(_sw_delta)) / _sw_n:.1f}°/行)")
        # T2-28b 放置上翻: 绕盖轴滚转翻不了掌 (j7 行程 143° < 需要的 ~180°,
        # -230 实测搬运 IK 崩到 35%)。改绕"垂直于盖轴的水平轴"上翻: 盖口翻
        # 朝上、腕甩到盖上方 —— 主要走腕俯仰, 不吃前臂滚转行程。腕绕盖公转,
        # 盖位置路径不变 (仍锚在重建盖位上), 不引入新的盖-瓶/桌几何。
        _pl_tip = float(os.environ.get("UNSCREW_PLACE_TIP", "0"))
        if abs(_pl_tip) > 1e-6:
            _A = quat_to_R(cap_q[k_sep]) @ np.array([0.0, 0.0, 1.0])
            _U = np.cross(_A, np.array([0.0, 0.0, 1.0]))
            _U = _U / max(np.linalg.norm(_U), 1e-9)
            _tp_delta = np.radians(_pl_tip)
            _tp_rate = np.radians(float(os.environ.get(
                "UNSCREW_SWING_RATE", "6.0")))
            _tp_n = int(np.ceil(1.5 * abs(_tp_delta) / _tp_rate)) + 1
            _tp_n = max(2, min(_tp_n, N - 2 - k_sep))
            _u2 = np.clip(np.arange(N - k_sep) / float(_tp_n), 0.0, 1.0)
            _a2 = (_u2 * _u2 * (3.0 - 2.0 * _u2)) * _tp_delta
            _h2 = 0.5 * _a2
            _qtp = np.concatenate(
                [np.cos(_h2)[:, None], np.sin(_h2)[:, None] * _U[None]], 1)
            _offs = wr_P[k_sep:] - cap_p[k_sep:]
            wr_P[k_sep:] = cap_p[k_sep:] + qrot(_qtp, _offs)
            wr_Q[k_sep:] = qmul(_qtp, wr_Q[k_sep:])
            print(f"[v1] T2-28b 放置上翻: 绕⊥盖轴水平轴 {_pl_tip:+.0f}° "
                  f"({_tp_n} 行, 峰值 {np.degrees(1.5 * abs(_tp_delta)) / _tp_n:.1f}"
                  f"°/行) | 盖轴末端指向 z="
                  f"{(quat_to_R(qmul(_qtp[-1:], cap_q[k_sep:k_sep+1])[0]) @ np.array([0.,0.,1.]))[2]:+.2f}")
        # T2-28c 掌心朝下放置 (用户裁定: 放盖必须手背朝上)。几何硬约束: 该
        # 双指捏法掌法线 ⊥ 盖轴 (实测侧抓时掌朝上+0.87/盖口朝上时掌只能水平),
        # 故掌心朝下 ⟺ 盖躺倒(轴水平) —— 与重建终点 (盖侧躺 82°) 一致, 人类
        # 示范者就是这么放的。解"掌法线→正下"的最小旋转 (混合轴, 不像纯绕盖
        # 轴滚转全压 j7), 护送窗内 slerp 过去; 绕竖直轴的剩余自由度 ψ 扫 IK。
        _pl_palm = os.environ.get("UNSCREW_PLACE_PALM", "")
        if _pl_palm:
            # 腕系掌法线 = C_MC 局部 +X (FK 真值: 张开手指垫平面法线在腕系为
            # [1.00,0.04,-0.01]; 此前误用 +Y 导致"掌心朝下"实落在朝外偏下 25°)
            _n_loc = np.array([float(v) for v in os.environ.get(
                "UNSCREW_PALM_AXIS", "1,0,0").split(",")])
            _n_loc /= max(np.linalg.norm(_n_loc), 1e-9)
            _n_w = quat_to_R(wr_Q[k_sep]) @ _n_loc
            _tgt = np.array([0.0, 0.0, -1.0])
            _c = float(np.clip(np.dot(_n_w, _tgt), -1.0, 1.0))
            _axv = np.cross(_n_w, _tgt)
            _axn = np.linalg.norm(_axv)
            _axv = (_axv / _axn) if _axn > 1e-9 else np.array([1.0, 0.0, 0.0])
            _th = float(np.arccos(_c))
            # T2-28d (用户: 手朝外转多了): 沿同一测地线提前停 —— 掌不必转到
            # 正朝下, 留 UNSCREW_PLACE_PALM_DEG 度 (拇指不再垂直指桌); 盖由
            # 释放段自己落平 (末态已覆写为圆面朝下, 见 T2-13 块)。
            _th = max(0.0, _th - np.radians(float(os.environ.get(
                "UNSCREW_PLACE_PALM_DEG", "0"))))
            _best_sw = None
            for _psi in range(-60, 61, 15):
                _hp = 0.5 * np.radians(_psi)
                _qz2 = np.array([np.cos(_hp), 0.0, 0.0, np.sin(_hp)])
                _hh = 0.5 * _th
                _qa = np.r_[np.cos(_hh), np.sin(_hh) * _axv]
                _qR = qmul(_qz2[None], _qa[None])[0]
                _pe = cap_p[k_sep] + quat_to_R(_qR) @ _off_w
                _qe = qmul(_qR[None], wr_Q[k_sep:k_sep + 1])[0]
                _sv = ik_r0.solve_traj(_pe[None], _qe[None],
                                       w_rot=0.25, n_restart=16)[0]
                _mg = float(np.minimum(_sv["q"] - ik_r0.lower,
                                       ik_r0.upper - _sv["q"]).min())
                _ok2 = (_sv["pos_err"] < 0.02
                        and _sv["rot_err"] < np.radians(10.0))
                _key2 = (1 if _ok2 else 0, _mg,
                         -float(_sv["pos_err"] + 0.25 * _sv["rot_err"]))
                if _best_sw is None or _key2 > _best_sw[0]:
                    _best_sw = (_key2, _psi, _qR)
            _, _psi_pick, _qR = _best_sw
            _sw_rate2 = np.radians(float(os.environ.get(
                "UNSCREW_SWING_RATE", "6.0")))
            _qid = np.array([1.0, 0.0, 0.0, 0.0])
            _d_q = float(np.dot(_qid, _qR))
            _qRs = _qR if _d_q >= 0 else -_qR
            _ang3 = np.arccos(np.clip(abs(_d_q), -1.0, 1.0)) * 2.0
            _n2 = int(np.ceil(1.5 * _ang3 / _sw_rate2)) + 1
            _n2 = max(2, min(_n2, N - 2 - k_sep))
            _u3 = np.clip(np.arange(N - k_sep) / float(_n2), 0.0, 1.0)
            _u3 = _u3 * _u3 * (3.0 - 2.0 * _u3)
            _qsw3 = np.empty((N - k_sep, 4))
            for _j, _uu in enumerate(_u3):     # 单参数测地线 => 轴固定角度缩放
                _hh3 = 0.5 * _uu * _ang3
                _axr = _qRs[1:] / max(np.linalg.norm(_qRs[1:]), 1e-12)
                _qsw3[_j] = np.r_[np.cos(_hh3), np.sin(_hh3) * _axr]
            _offs3 = wr_P[k_sep:] - cap_p[k_sep:]
            wr_P[k_sep:] = cap_p[k_sep:] + qrot(_qsw3, _offs3)
            wr_Q[k_sep:] = qmul(_qsw3, wr_Q[k_sep:])
            # 掌心朝下时拇指/指尖垂在腕平面下 ~5cm, +5cm 通用钳位不够 —— 该
            # 窗抬到 +7cm (盖落桌后从 ~1-2cm 高度释放, 微降更物理)
            wr_P[k_sep:, 2] = np.maximum(wr_P[k_sep:, 2], TABLE_Z + 0.07)
            _n_end = quat_to_R(wr_Q[N - 1]) @ _n_loc
            _dz_end = float(wr_P[N - 1, 2] - cap_p[N - 1, 2])
            print(f"[v1] T2-28c 掌心朝下放置: 最小旋转 {np.degrees(_th):.0f}° "
                  f"+ ψ={_psi_pick:+d}° ({_n2} 行, 峰值 "
                  f"{np.degrees(1.5 * _th) / _n2:.1f}°/行) | 末行掌法线 z="
                  f"{_n_end[2]:+.2f} | 腕高于盖 {_dz_end * 100:+.1f}cm")
        # +2cm -> +5cm (T2-25 修正归因): 护送段拳握指节蹭桌的真凶就是这条钳位
        wr_P[:, 2] = np.maximum(wr_P[:, 2], TABLE_Z + 0.05)
        print(f"[v1] 右腕携带段: 姿态冻结于脱离行 {k_sep}, 位置跟盖 "
              f"(世界系偏移 {np.round(_off_w * 100, 1)}cm)")

    # ---- 右腕绕螺轴的自转角 = **规范自由度**, 解出来而不是照抄人腕 ----
    # 依据 (2026-08-30 用户裁定 + 数据事实):
    #   ① 盖是旋转体, 数据集里盖的**旋转**置信度低 (位置高), 判据侧 placed 只
    #      看轴倾角 (progress.py 用 _axis_tilt), 盖绕自身轴的自转不可观也不可判;
    #   ② 脱离前盖与瓶身同体 (cap_q 由瓶推导), 那段的自转 = 螺纹相位, **不动**;
    #      能动的只有"手怎么握" —— 一个常量抓握自转角 (刚性抓握);
    #   ③ 脱离后盖被手拿着, 盖的姿态跟着手走 (rigid ride), 自转角逐行限速漂移。
    # 不做这件事的代价 (实测): 人腕前臂自转随便 270°, 机器腕 j7 行程仅 143° ——
    # 右臂交互行 100/103 贴限位、对自己的腕目标中位差 44.7cm, 整段搬运冻死。
    ROLL_STEP = np.radians(6.0)          # 每行自转限速 (20Hz 下 ~2 rad/s)
    ik_r = _mk_ik("right")
    axis_n = axis / np.linalg.norm(axis, axis=1, keepdims=True)

    def _roll_q(q_rows, ax_rows, angs):
        h = 0.5 * np.asarray(angs, float)
        rq = np.concatenate([np.cos(h)[:, None], np.sin(h)[:, None] * ax_rows], 1)
        out = qmul(rq, q_rows)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def _score_rows(P_rows, Q_rows, n_restart=2):
        sols = ik_r.solve_traj(P_rows, Q_rows, w_rot=0.25, n_restart=n_restart)
        pe = np.array([sv["pos_err"] for sv in sols])
        re = np.array([sv["rot_err"] for sv in sols])
        qs = np.stack([sv["q"] for sv in sols])
        marg = np.minimum(qs - ik_r.lower, ik_r.upper - qs).min(axis=1)
        good = (pe < 0.02) & (re < np.radians(10.0)) & (marg > np.radians(3.0))
        return float(good.mean()), float(np.median(pe)), float(np.median(re)), sols

    ks1 = k_sep + 1
    # GraspPose 的常量绕盖轴方位已在 f35 扫描并刚性写入 _cap_track；这里的
    # roll 只表示脱离后的额外携带漂移，不能再重复加一次常量方位。
    yaw_g = 0.0
    roll = np.full(N, yaw_g)

    # ② 脱离后逐行限速漂移: 手与盖一起转 (不滑手), 盖的自转本就不可判
    q_seed = np.asarray(ik_r.solve_traj(
        wr_P[k_sep:k_sep + 1], wr_Q[k_sep:k_sep + 1],
        w_rot=0.25, n_restart=16)[-1]["q"], np.float64)
    drift = 0.0
    for k in range(ks1, N):
        best = None
        for d in (-2, -1, 0, 1, 2):
            cand = drift + d * ROLL_STEP
            Qk = _roll_q(wr_Q[k:k + 1], axis_n[k:k + 1], [cand])[0]
            r = ik_r.solve(wr_P[k], quat_to_R(Qk), q0=q_seed, iters=150,
                           w_rot=0.25)
            m = float(np.minimum(r["q"] - ik_r.lower, ik_r.upper - r["q"]).min())
            # 滞回 (T2-17): 盖是旋转体, 得分对 roll 平坦时纯 argmin 会数值噪声
            # 驱动随机游走 —— clip32 实测护送段累计漂移 336°, 视频里手带着盖
            # 乱翻, 指垫根本没法稳定捏住。挪一步 (6°) 必须换来 ≥3mm 位置误差
            # 量级的真实可达性改善; 真被限位/误差顶到时该罚项远小于收益。
            sc = (r["pos_err"] + 0.25 * r["rot_err"]
                  + max(0.0, np.radians(3.0) - m)
                  + 0.03 * abs(d) * ROLL_STEP)
            if best is None or sc < best[0]:
                best = (sc, r, cand, Qk)
        drift = best[2]
        wr_Q[k] = best[3]
        roll[k] = yaw_g + drift
        if best[1]["pos_err"] < 0.02:      # 失败解不许污染下一行的种子
            q_seed = np.asarray(best[1]["q"], np.float64)
    print(f"[v1] 右腕自转漂移 (脱离后 {N - ks1} 行, 限速 "
          f"{np.degrees(ROLL_STEP):.0f}°/行): 累计 "
          f"{np.degrees(roll[-1] - yaw_g):+.0f}°")

    # 盖脱离后姿态改为"跟着手走" (rigid ride): 位置仍用重建 (置信度高), 姿态
    # 由抓握变换从脱离行推出 —— 重建的盖自转本就不可信, 让它与手自洽更物理。
    grasp_rel = qmul(qconj(wr_Q[k_sep:k_sep + 1]), cap_q[k_sep:k_sep + 1])[0]
    cap_q[k_sep:] = qmul(wr_Q[k_sep:], np.tile(grasp_rel, (N - k_sep, 1)))
    cap_q /= np.linalg.norm(cap_q, axis=1, keepdims=True)
    _ct = np.degrees(np.arccos(np.clip(
        qrot(cap_q, np.tile([0.0, 0.0, 1.0], (N, 1)))[:, 2], -1, 1)))
    print(f"[v1] 盖姿态 rigid ride (脱离行 {k_sep} 起): 末行倾角 "
          f"{_ct[-1]:.0f}° (原重建 107°, 自转不可判故不追)")

    # ---- P1 起的完整右手数据轨迹 -----------------------------------
    # P0->P1 由 cuRobo；P1 之后全是 RL/data 轨迹。右手不被代码“钉死”：
    # 保留原始右掌从 f22(P1) 到 f60(到盖) 的整条曲线和时序。数据本身在
    # f35 前只有小幅运动，所以视觉上近似等待；这是演示的结果，不是硬冻结。
    # 原人手与机器人工作空间不同，因此用一个端点受约束相似变换重定向：
    #   raw(P1) -> 可达等待位，raw(f60) -> 盖 GraspPose；
    # 中间的弯曲、快慢和微动原样保留。姿态使用原始活的腕四元数相对轨迹，
    # 再沿同一数据进度将终点精确校正到 GraspPose。没有中途 cuRobo 段。
    _ks = int(right_start_src - w0)
    _kg = int(right_grasp_src - w0)
    _away = wr_P[_ks] - cap_p[_ks]
    _away /= max(np.linalg.norm(_away), 1e-9)
    _up = np.array([0.0, 0.0, 1.0])
    _mix = _away + _up
    _mix /= max(np.linalg.norm(_mix), 1e-9)
    _wait_specs = [(_away, 0.04), (_away, 0.06), (_away, 0.08),
                   (_mix, 0.06), (_mix, 0.08), (_up, 0.06)]
    _best_wait = None
    for _dir, _dist in _wait_specs:
        _wp = wr_P[_ks] + _dist * _dir
        _sv = ik_r0.solve_traj(_wp[None], wr_Q[_ks:_ks + 1],
                               w_rot=0.25, n_restart=24)[0]
        _mgw = float(np.minimum(_sv["q"] - ik_r0.lower,
                                ik_r0.upper - _sv["q"]).min())
        _ok = (_sv["pos_err"] < 0.02
               and _sv["rot_err"] < np.radians(10.0)
               and _mgw > np.radians(3.0))
        _keyw = (1 if _ok else 0, _mgw,
                 -float(_sv["pos_err"] + 0.25 * _sv["rot_err"]),
                 -abs(_dist - 0.06))
        if _best_wait is None or _keyw > _best_wait[0]:
            _best_wait = (_keyw, _wp, wr_Q[_ks].copy(),
                          np.asarray(_sv["q"], np.float64), _dir, _dist,
                          float(_sv["pos_err"]), float(_sv["rot_err"]))
    (_kw, right_wait_P, right_wait_Q, _q_station_r, _wait_dir, _wait_dist,
     _wait_pe, _wait_re) = _best_wait
    if not _kw[0]:
        raise ValueError("P1 右手等待位无可达 pregrasp；不能用坏 IK 继续")
    print(f"[v1] P1 右手等待位: 水平抓取位退 {_wait_dist * 100:.0f}cm "
          f"dir={np.round(_wait_dir, 2)} | {_wait_pe * 100:.2f}cm/"
          f"{np.degrees(_wait_re):.1f}° | 限位余量 {np.degrees(_kw[1]):.1f}°")

    _step = np.linalg.norm(np.diff(_palm_r[right_start_src:right_grasp_src + 1],
                                   axis=0), axis=1)
    _prog = np.r_[0.0, np.cumsum(_step)]
    if _prog[-1] <= 1e-9:
        _prog = np.linspace(0.0, 1.0, _kg - _ks + 1)
    else:
        _prog /= _prog[-1]
    _prog = _prog * _prog * (3.0 - 2.0 * _prog)
    right_motion_progress_src = np.zeros(N, np.float64)
    right_motion_progress_src[_ks:_kg + 1] = _prog
    right_motion_progress_src[_kg + 1:] = 1.0

    def _slerp_one(qa, qb, u):
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        if float(np.dot(qa, qb)) < 0:
            qb = -qb
        d = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
        th = float(np.arccos(d))
        if th < 1e-7:
            out = (1.0 - u) * qa + u * qb
        else:
            out = (np.sin((1.0 - u) * th) * qa + np.sin(u * th) * qb) / np.sin(th)
        return out / np.linalg.norm(out)

    _grasp_P, _grasp_Q = wr_P.copy(), wr_Q.copy()

    def _align_vec(a, b):
        """最小旋转 R，使 R@a 与b 同向（含反平行退化）。"""
        a, b = np.asarray(a, float), np.asarray(b, float)
        an, bn = np.linalg.norm(a), np.linalg.norm(b)
        if an < 1e-9 or bn < 1e-9:
            return np.eye(3)
        a, b = a / an, b / bn
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if c > 1.0 - 1e-10:
            return np.eye(3)
        if c < -1.0 + 1e-10:
            basis = np.eye(3)[int(np.argmin(np.abs(a)))]
            axis180 = np.cross(a, basis)
            axis180 /= np.linalg.norm(axis180)
            return 2.0 * np.outer(axis180, axis180) - np.eye(3)
        v = np.cross(a, b)
        K = np.array([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]])
        return np.eye(3) + K + K @ K * ((1.0 - c) / max(np.dot(v, v), 1e-12))

    # 位置: 从 P1 开始消费整条原始掌路径，而不是只取一个标量进度直线插值。
    _rawP = smooth(_palm_r[w0:right_grasp_src + 1], 1.0)
    _raw_d = _rawP[-1] - _rawP[0]
    _dst_d = _grasp_P[_kg] - right_wait_P
    if np.linalg.norm(_raw_d) < 1e-6 or np.linalg.norm(_dst_d) < 1e-6:
        raise ValueError("P1->右手到盖轨迹端点位移太小，无法做相似重定向")
    _right_map_R = _align_vec(_raw_d, _dst_d)
    _right_map_scale = float(np.linalg.norm(_dst_d) / np.linalg.norm(_raw_d))
    _mappedP = right_wait_P + _right_map_scale * (
        _right_map_R @ (_rawP - _rawP[0]).T).T
    _mappedP[0], _mappedP[-1] = right_wait_P, _grasp_P[_kg]

    # 姿态: 原始活腕的相对旋转 + 终点校正。校正权重用掌路径弧长，
    # 因而前段掌心几乎不动时也不会凭空旋转。
    _rawQ_all = cont(np.asarray(qr["wrist_quat_wxyz"], np.float64))
    _rawQ = _rawQ_all[w0:right_grasp_src + 1]
    _relQ = qmul(_rawQ, np.tile(qconj(_rawQ[:1]), (len(_rawQ), 1)))
    _provQ = qmul(_relQ, np.tile(right_wait_Q, (len(_rawQ), 1)))
    _provQ = cont(_provQ / np.linalg.norm(_provQ, axis=1, keepdims=True))
    _corr_end = qmul(_grasp_Q[_kg:_kg + 1], qconj(_provQ[-1:]))[0]
    _full_step = np.linalg.norm(np.diff(_rawP, axis=0), axis=1)
    _full_u = np.r_[0.0, np.cumsum(_full_step)]
    _full_u = (_full_u / _full_u[-1] if _full_u[-1] > 1e-9
               else np.linspace(0.0, 1.0, len(_rawP)))
    _full_u = _full_u * _full_u * (3.0 - 2.0 * _full_u)
    _mappedQ = np.empty_like(_provQ)
    _q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    for _j, _u in enumerate(_full_u):
        _qc = _slerp_one(_q_ident, _corr_end, float(_u))
        _mappedQ[_j] = qmul(_qc[None], _provQ[_j:_j + 1])[0]
    _mappedQ = cont(_mappedQ / np.linalg.norm(_mappedQ, axis=1, keepdims=True))
    _mappedQ[0], _mappedQ[-1] = right_wait_Q, _grasp_Q[_kg]

    wr_P[:_kg + 1] = _mappedP
    wr_Q[:_kg + 1] = _mappedQ
    _raw_arc = float(np.linalg.norm(np.diff(_rawP, axis=0), axis=1).sum())
    _map_arc = float(np.linalg.norm(np.diff(_mappedP, axis=0), axis=1).sum())
    print(f"[v1] P1 后右腕数据轨迹: f{w0}..f{right_grasp_src} 完整曲线重定向 "
          f"(scale={_right_map_scale:.3f}, raw/map arc={_raw_arc * 100:.1f}/"
          f"{_map_arc * 100:.1f}cm); f{right_start_src} 为数据明显启动点，后续随盖")

    # 左腕: GraspPose prior 镜像锚定 (2026-08-30 拍板: **零死数据依赖**)。
    # 旧法取静态腕点当握位偏移 —— 那正是死通道 (上游腕平移静态填充), 且"静态点
    # =首帧真实握位"只在旧 clip 验证过。换成: Screw27_body prior (与本批 CAD
    # 字节相同, screw27 任务实证) 的抓取位姿, 镜像到左手 (q'=(w,-x,y,-z),
    # p'=(x,-y,z), 指模板由 env 按名镜像), 锚在瓶行上随瓶走 —— 手物相对位姿
    # by construction 恒定 (框架 v2 的同一哲学)。绕瓶轴方位角是自由参数
    # (旋转体), 用 ArmIK 可达率扫出来 (Pour17 best_yaw 同源)。
    zp = np.load(TC.PRIOR_AUX)
    gp = np.asarray(zp["grasp"], np.float64)[:3]
    gq = np.asarray(zp["grasp"], np.float64)[3:7]
    # ★ 坐标系原点错配 (2026-08-31 实测揪出, 一直卡着 G1 的真凶):
    #   数据集的瓶 mesh z∈[0,0.197] —— 刚体原点在**瓶底**;
    #   prior 来自 Dexonomy (bottle_body/scale010), 它的物体系是**居中**的
    #   —— 抓取锚点 z=-2.0cm、接触点 -2.0~+7.1cm, 只有在瓶心系里才讲得通。
    #   直接搬过来 = 把左手锚在桌面以下 2cm; 贴桌钳位抬到桌上 2cm 后手掌仍
    #   插在桌里, 物理把整条手臂顶高 10cm -> 手在瓶顶上方抓空, 站位垫 0/5,
    #   G1 永远不成形 (旧验收凭据里"左垫 1 个接触"同一个病)。
    #   偏移从**本 clip 自己的 mesh** 现算 (bbox 中心), 换 clip 自动适配。
    _mb = trimesh.load(objs[BODY_ID]["mesh"], process=False, force="mesh")
    _zb = np.asarray(_mb.vertices)[:, 2]
    _frame_dz = float(0.5 * (_zb.min() + _zb.max()) - _zb.min())
    print(f"[v1] prior 物体系对齐: 瓶 mesh z∈[{_zb.min():.3f},{_zb.max():.3f}] "
          f"(原点在底), prior 居中 -> 锚点 z 补 {_frame_dz * 100:+.2f}cm")
    gp_m = np.array([gp[0], -gp[1], gp[2] + _frame_dz])
    gq_m = np.array([gq[0], -gq[1], gq[2], -gq[3]])
    gq_m /= np.linalg.norm(gq_m)

    def _left_track(yaw):
        qy = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        pl = (quat_to_R(qy) @ gp_m)
        # 径向收紧 TC.PRIOR_RADIAL_TRIM: 让指尖真正落在瓶面上 (见 task_config)
        _rh = np.array([pl[0], pl[1], 0.0])
        _rn = float(np.linalg.norm(_rh))
        if _rn > 1e-6:
            pl = pl - (TC.PRIOR_RADIAL_TRIM / _rn) * _rh
        qlg = qmul(qy[None], gq_m[None])[0]
        P = body_p + qrot(body_q, np.tile(pl, (N, 1)))
        Q = qmul(body_q, np.tile(qlg, (N, 1)))
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        P[:, 2] = np.maximum(P[:, 2], TABLE_Z + 0.02)
        return P, Q

    ik_l = _mk_ik("left")
    best = None
    for ydeg in range(0, 360, 15):
        P, Q = _left_track(np.radians(ydeg))
        # **站位行 (交互第 0 行) 单独出解并作为硬条件**: 那是左手合拢抓瓶的
        # 唯一时刻, 也是机器段 pregrasp 的锚。2026-08-30 实测教训: 只看全程
        # 平均可达率会选出"站位行差 5~19cm"的方位角 —— 手根本没到瓶上, G1
        # 三垫永远不成形 (零动作验收里左垫只有 1 个接触就是这么来的)。
        s0 = ik_l.solve_traj(P[:1], Q[:1], w_rot=0.25, n_restart=16)[0]
        m0 = float(np.minimum(s0["q"] - ik_l.lower, ik_l.upper - s0["q"]).min())
        st_ok = (s0["pos_err"] < 0.02 and s0["rot_err"] < np.radians(10.0)
                 and m0 > np.radians(3.0))
        sols = ik_l.solve_traj(P[::4], Q[::4], w_rot=0.25, n_restart=2)
        pe = np.array([sv["pos_err"] for sv in sols])
        re = np.array([sv["rot_err"] for sv in sols])
        qs = np.stack([sv["q"] for sv in sols])
        # 限位余量判据 (2026-08-30 cuRobo 排障): ArmIK 是钳限位的, 贴边解
        # (clip32 原 yaw*=240°: j7 钉死 -79°下限 33 行) 误差再小也过不了带
        # 限位余量的 cuRobo —— 机器段规划直接全灭。内点 ≥3° 才算达标。
        marg = np.minimum(qs - ik_l.lower, ik_l.upper - qs).min(axis=1)
        good = (pe < 0.02) & (re < np.radians(10.0)) & (marg > np.radians(3.0))
        score = float(good.mean())
        # Tie-break with the same position/rotation geometry used by ArmIK.
        cost = float(np.median(pe ** 2 + (0.25 * re) ** 2))
        key = (1 if st_ok else 0, score, -cost)
        if best is None or key > best[6]:
            best = (ydeg, score, cost, float(np.median(pe)),
                    float(np.median(re)), float(np.degrees(np.median(marg))),
                    key, float(s0["pos_err"]), float(np.degrees(s0["rot_err"])))
    yaw_l = np.radians(best[0])
    print(f"[v1] 左抓方位扫描: yaw*={best[0]}° 站位行 {best[7] * 100:.2f}cm/"
          f"{best[8]:.1f}° ({'可达' if best[6][0] else '⚠ 不可达'}) "
          f"可达率(含限位内点) {best[1] * 100:.0f}% 全行中位 "
          f"{best[3] * 100:.2f}cm/{np.degrees(best[4]):.1f}° 限位余量中位 "
          f"{best[5]:.1f}° (prior=Screw27_body 镜像)")
    wl_P, wl_Q = _left_track(yaw_l)

    # ---- [TASK] 交互段时间拉伸 (2026-09-01) ----
    # 演示 15fps, env 20Hz 播放 = 1.33× 实时。转瓶启动段 (row189~199) 左腕目标
    # 0.5s 转 147° (峰值 21.6°/行 = 432°/s); 零动作回放与训练后策略都在 row190
    # 起 3 步内把瓶甩飞 (D4)。PD 臂 + 冻结指形跟不上人手的一甩。
    # 按 UNSCREW_IA_STRETCH 倍数把交互段重采样到更多行: 位置线性、四元数 slerp,
    # **首末行精确保留** —— 规划摘要 (reference_planning_digest) 只锚首行 station
    # 与末行障碍位, 故已入库的 cuRobo 机器段仍有效, 不需重规划。
    # k_sep / frame_of_row / conf 列 / 人手指流 同步按 ia_src_t 映射重采样。
    # 默认 1.0 = 恒等, 不改变任何输出。
    _S = float(os.environ.get("UNSCREW_IA_STRETCH", "1.0"))
    ia_src_t = np.arange(N, dtype=float)          # 新行 -> 旧行 (浮点)
    if _S > 1.0 + 1e-9:
        N_new = int(round((N - 1) * _S)) + 1
        ia_src_t = np.linspace(0.0, N - 1, N_new)

        def _lin(A):
            A = np.asarray(A, float)
            return np.stack([np.interp(ia_src_t, np.arange(N), A[:, j])
                             for j in range(A.shape[1])], axis=1)

        def _slerp_rows(Q):
            Q = np.asarray(Q, float)
            i0 = np.clip(np.floor(ia_src_t).astype(int), 0, N - 1)
            i1 = np.clip(i0 + 1, 0, N - 1)
            u = (ia_src_t - i0)[:, None]
            a, b = Q[i0], Q[i1]
            b = np.where((a * b).sum(1, keepdims=True) < 0, -b, b)   # 最短弧
            d = np.clip((a * b).sum(1, keepdims=True), -1.0, 1.0)
            th = np.arccos(d)
            s = np.sin(th)
            safe = np.where(s > 1e-6, s, 1.0)
            wa = np.where(s > 1e-6, np.sin((1 - u) * th) / safe, 1 - u)
            wb = np.where(s > 1e-6, np.sin(u * th) / safe, u)
            out = wa * a + wb * b
            return out / np.linalg.norm(out, axis=1, keepdims=True)

        body_p, cap_p, wr_P, wl_P = _lin(body_p), _lin(cap_p), _lin(wr_P), _lin(wl_P)
        body_q, cap_q = _slerp_rows(body_q), _slerp_rows(cap_q)
        wr_Q, wl_Q = _slerp_rows(wr_Q), _slerp_rows(wl_Q)
        right_motion_progress_src = np.interp(
            ia_src_t, np.arange(N), right_motion_progress_src)
        _k_old, _N_old = k_sep, N
        k_sep = int(round(k_sep * (N_new - 1) / max(_N_old - 1, 1)))
        N = N_new
        print(f"[v1] 交互段时间拉伸 ×{_S:.2f}: {_N_old} -> {N} 行 "
              f"(k_sep {_k_old} -> {k_sep}); 首末行精确保留", flush=True)
    # 第一行连续源时钟达到原始 f35 的位置; 不用 frame_of_row 的四舍五入重复行
    # 决定边界, 避免把右手提前到 f34.67。
    right_start_k = int(np.searchsorted(
        w0 + ia_src_t, float(right_start_src), side="left"))
    right_start_k = min(max(right_start_k, 0), N - 1)
    assert w0 + ia_src_t[right_start_k] >= right_start_src - 1e-9
    right_grasp_k = int(np.searchsorted(
        w0 + ia_src_t, float(right_grasp_src), side="left"))
    right_grasp_k = min(max(right_grasp_k, right_start_k + 1), N - 1)
    print(f"[v1] T2-12 映射: f{right_start_src} -> 交互 k={right_start_k}; "
          f"f{right_grasp_src} -> k={right_grasp_k} "
          f"(连续源时钟 {w0 + ia_src_t[right_start_k]:.3f}/"
          f"{w0 + ia_src_t[right_grasp_k]:.3f})")
    ia_src_i = w0 + np.rint(ia_src_t).astype(int)   # 新行 -> 最近演示帧

    # ---- T2-13 脱盖后手-盖护送约束 ---------------------------------
    # FoundationPose 在遮挡最强的脱盖段把盖跟丢了：原重建的盖心
    # 从右手指间滑出 9cm，但接触标注仍说右手持盖。这不能交给
    # residual RL “自己学回来”：物体 leash/时钟会把错的盖行当监督。
    #
    # 脱盖到任务末段前，盖严格跟随右腕在 k_sep 捕获的 SE(3) 抓持变换；
    # 最后 0.6s 才平滑放到原数据的任务终点。这样不改变末态障碍和已有
    # cuRobo Retreat 的端点契约，同时把“持盖”与“放盖后可分离”分成两个
    # 可见阶段。手腕本身不改，故不引入新的臂 IK/机器段端点。
    _cap_end_p = cap_p[-1].copy()
    _cap_end_q = cap_q[-1].copy()
    # T2-28d 盖末态覆写 (用户裁定: 圆面朝下平放, 重建的侧躺 82° 弃用)。
    # 方位取离手中 (ride) 末姿态最近的扭转, 释放 12 行里盖自然落平。
    _cef = os.environ.get("UNSCREW_CAP_END_FLAT", "")
    if _cef:
        _t_ax = np.array([0.0, 0.0, -float(_cef)])   # 1: 盖+z 朝下
        _z_e = quat_to_R(_cap_end_q) @ np.array([0.0, 0.0, 1.0])
        _cf_c = float(np.clip(np.dot(_z_e, _t_ax), -1.0, 1.0))
        _cf_ax = np.cross(_z_e, _t_ax)
        _cf_n = np.linalg.norm(_cf_ax)
        _cf_ax = (_cf_ax / _cf_n) if _cf_n > 1e-9 else np.array([1.0, 0.0, 0.0])
        _cf_h = 0.5 * np.arccos(_cf_c)
        _q_al = np.r_[np.cos(_cf_h), np.sin(_cf_h) * _cf_ax]
        _cap_end_q = qmul(_q_al[None], _cap_end_q[None])[0]
        _cap_end_q /= np.linalg.norm(_cap_end_q)
        _cap_end_p[2] = TABLE_Z + float(os.environ.get(
            "UNSCREW_CAP_REST_H", "0.011"))
        print(f"[v1] T2-28d 盖末态覆写: 圆面朝下平放 (自 ride 末姿态扭转 "
              f"{np.degrees(2 * _cf_h):.0f}°), z={_cap_end_p[2]:.3f}")
    _R_sep = quat_to_R(wr_Q[k_sep])
    _cap_in_wr = _R_sep.T @ (cap_p[k_sep] - wr_P[k_sep])
    _cap_q_in_wr = qmul(qconj(wr_Q[k_sep:k_sep + 1]),
                        cap_q[k_sep:k_sep + 1])[0]
    _cap_hold_p = wr_P + qrot(
        wr_Q, np.tile(_cap_in_wr, (N, 1)))
    _cap_hold_q = qmul(
        wr_Q, np.tile(_cap_q_in_wr, (N, 1)))
    _cap_hold_q = cont(
        _cap_hold_q / np.linalg.norm(_cap_hold_q, axis=1, keepdims=True))
    # 横放盖的重建原点与网格底面不同义；一个短窗内腕姿会把
    # 刚性推导的盖原点压进桌面 1.6cm。只做单边桌面投影，并将该偏差
    # 纳入下面的 2cm 护送不变量，不允许“为了刚性”把物体塞进桌里。
    _cap_hold_p[:, 2] = np.maximum(_cap_hold_p[:, 2], TABLE_Z + 0.006)
    _place_rows = min(12, max(N - k_sep - 1, 1))       # 20Hz 下 0.6s
    cap_place_start_k = max(k_sep + 1, N - 1 - _place_rows)
    cap_p[k_sep:cap_place_start_k] = _cap_hold_p[k_sep:cap_place_start_k]
    cap_q[k_sep:cap_place_start_k] = _cap_hold_q[k_sep:cap_place_start_k]
    _place_u = np.linspace(0.0, 1.0, N - cap_place_start_k)
    _place_u = _place_u * _place_u * (3.0 - 2.0 * _place_u)
    for _j, _u in enumerate(_place_u):
        _k = cap_place_start_k + _j
        cap_p[_k] = ((1.0 - _u) * _cap_hold_p[_k]
                     + _u * _cap_end_p)
        cap_q[_k] = _slerp_one(_cap_hold_q[_k], _cap_end_q, float(_u))
    cap_p[-1], cap_q[-1] = _cap_end_p, _cap_end_q
    _hold_err = np.linalg.norm(
        cap_p[k_sep:cap_place_start_k]
        - _cap_hold_p[k_sep:cap_place_start_k], axis=1)
    _hold_err_max = float(_hold_err.max()) if len(_hold_err) else 0.0
    assert _hold_err_max < 1e-9
    assert np.linalg.norm(cap_p[-1] - _cap_end_p) < 1e-12
    assert abs(float(np.dot(cap_q[-1], _cap_end_q))) > 1.0 - 1e-10
    print(f"[v1] 右手-盖护送: k={k_sep}..{cap_place_start_k - 1} "
          f"SE(3) 刚性同行; k={cap_place_start_k}..{N - 1} "
          f"{len(_place_u) / 20:.2f}s 受控放下并张手 | 末态精确保持")

    # ---- ArmIK: 逐行热启 + 失败冻结 + 轻平滑后**重投影** ----
    # 2026-08-30 修掉的三个坑 (都是"母带看起来存在, 其实臂根本没到位"的来源):
    #   ① 失败解当下一行种子 => 一次解崩会顺着热启链把整段拖进限位角落;
    #   ② "最近达标行顶替" 会在分支之间跳 (顶替行与邻行不是同一 IK 分支);
    #   ③ 解完再 σ=1.5 高斯平滑: 跨分支平均出来的构型谁也不是 —— 实测左臂
    #      拧盖窗中位 0.14cm 的解, 平滑后对同一目标差 9.3cm/62.7°。
    # 现在: 热启保连续 -> 坏行冻结上一行 -> σ=0.5 轻平滑 -> 每行以平滑值为种子
    # 重投影一次, 只在不变差时采纳 (平滑与精度不再互相拆台)。
    def solve_side(side, P, Q, seed=None, tag=""):
        ik = _mk_ik(side)
        P = np.asarray(P, float)
        Q = np.asarray(Q, float)
        n = len(P)
        rng_s = np.random.default_rng(11)
        q = np.zeros((n, 7))
        pe = np.zeros(n)
        re = np.zeros(n)
        frozen = 0
        q_prev = None if seed is None else np.asarray(seed, float)
        branch = np.radians(25.0)

        def _err(qc, k):
            fp, fR = ik.fk(qc)
            ep = float(np.linalg.norm(fp - P[k]))
            er = float(np.arccos(np.clip(
                (np.trace(fR.T @ quat_to_R(Q[k])) - 1) * 0.5, -1, 1)))
            return ep, er

        # T2-25 肘/前臂离桌净空: 选支只看腕位姿误差时, 左臂在低位持瓶段会选
        # "肘下支"把前臂平贴桌面 (用户视频实测"机械臂本身有接触")。对肘/前臂
        # 连杆做 FK, 最低点低于桌面+5cm 按 2.0/m 计罚 (5cm 违规 = 10cm 腕误差
        # 量级), 首行在随机种子里就选到肘上支, 后续行靠 25° 邻域连续跟随。
        _pfx = "R" if side == "right" else "L"
        _arm_links = [f"{_pfx}_arm_l4", f"{_pfx}_arm_l5",
                      f"vega_1p_{_pfx}_arm_l6", f"{_pfx}_arm_l7"]

        def _arm_clear_pen(qc):
            # 只对右臂生效 (T2-25 终版): 左臂低位持瓶时前臂贴桌是演示本身的
            # 姿态, 任何净空罚都把左臂逼上肘上支极限 (实测两档罚参都是坏行
            # 0->37、限位余量 0°、8cm 不可达行 —— clip32 病灶的复刻)。前臂轻
            # 搁桌面物理无害; 右臂罚零成本 (余量 43°) 且防抓盖段指节插桌。
            if side != "right":
                return 0.0
            mz = min(float(ik.link_pose_world(lk, qc)[2, 3]) for lk in _arm_links)
            return 1.0 * max(0.0, TABLE_Z + 0.02 - mz)

        for k in range(n):
            # 目标逐行平滑，正确解也应留在上一行附近。把“原地保持”本身放进
            # 候选，并只接纳 25° 邻域内的 IK 解；否则随机重启会跳到远端冗余
            # 分支，后续 RATE 限速只会把一次跳变摊成多行甩臂。
            candidates = []
            if q_prev is not None:
                hp, hr = _err(q_prev, k)
                candidates.append((hp + 0.25 * hr + _arm_clear_pen(q_prev),
                                   hp, hr, q_prev.copy(), 0.0, True))
                local = np.clip(q_prev + rng_s.normal(0.0, 0.04, 7),
                                ik.lower, ik.upper)
                seeds = [q_prev, local, None]
            else:
                seeds = [None]
            if q_prev is None or k % 12 == 0:
                seeds += [rng_s.uniform(ik.lower, ik.upper) for _ in range(2)]
            for q0 in seeds:
                r = ik.solve(P[k], quat_to_R(Q[k]), q0=q0, iters=200,
                             w_rot=0.25)
                qr = np.asarray(r["q"], float)
                if not np.isfinite(qr).all():
                    continue
                dq = 0.0 if q_prev is None else float(np.abs(qr - q_prev).max())
                if q_prev is not None and dq > branch:
                    continue
                base = float(r["pos_err"] + 0.25 * r["rot_err"])
                candidates.append((base + 0.02 * dq + _arm_clear_pen(qr),
                                   float(r["pos_err"]), float(r["rot_err"]),
                                   qr, dq, False))
            assert candidates, f"{side} IK row{k}: no finite local candidate"
            _, pe[k], re[k], q[k], _, held = min(candidates, key=lambda x: x[0])
            frozen += int(held and k > 0)
            q_prev = q[k].copy()
        # 逐行限速: 冻结段重新锁定目标时会一次跳好几十度 (母带里就是"臂瞬移"),
        # PD 跟不上会把桌上的物体扫飞。坏行本来就不准, 把跳变摊到几行里换来
        # 可播放性; 好行几乎不受影响 (跳变本来就 <RATE)。
        RATE = np.radians(20.0)
        for k in range(1, n):
            d = q[k] - q[k - 1]
            m = np.abs(d).max()
            if m > RATE:
                q[k] = q[k - 1] + d * (RATE / m)
        # 限速已经改变了真实 FK；重投影的比较基线必须按限速后的 q 重算。
        for k in range(n):
            pe[k], re[k] = _err(q[k], k)
        qs = smooth(q, 0.5)
        for k in range(n):
            r = ik.solve(P[k], quat_to_R(Q[k]), q0=qs[k], iters=80, w_rot=0.25)
            # 只在**既更准又不跳分支**时采纳: 重投影可能落到另一支解上, 那会
            # 在母带里留下一行几十度的关节跳变 (PD 跟不上 = 把物体打飞)。
            near = np.abs(np.asarray(r["q"], float) - q[k]).max() < np.radians(20)
            if near and (r["pos_err"] + 0.25 * r["rot_err"]) <= (pe[k]
                                                                 + 0.25 * re[k]):
                q[k], pe[k], re[k] = r["q"], r["pos_err"], r["rot_err"]
        # 相邻两行各自重投影最多 20°，两者反向时净跳变仍可近 40°；二次限速
        # 后再按最终写盘构型重算误差，日志与视频/训练消费值保持一致。
        for k in range(1, n):
            d = q[k] - q[k - 1]
            m = np.abs(d).max()
            if m > RATE:
                q[k] = q[k - 1] + d * (RATE / m)
        for k in range(n):
            pe[k], re[k] = _err(q[k], k)
        good = (pe < 0.02) & (re < np.radians(10.0))
        marg = np.minimum(q - ik.lower, ik.upper - q).min(axis=1)
        jump = np.degrees(np.abs(np.diff(q, axis=0)).max()) if n > 1 else 0.0
        print(f"[v1] {side} IK{tag}: 位置<2cm且姿态<10° {good.mean() * 100:.0f}% "
              f"| 全行中位 {np.median(pe) * 100:.2f}cm/"
              f"{np.degrees(np.median(re)):.1f}° | 90分位 "
              f"{np.percentile(pe, 90) * 100:.2f}cm | 冻结 {frozen} 行 "
              f"| 限位余量中位 {np.degrees(np.median(marg)):.1f}° "
              f"(贴限<3° {int((marg < np.radians(3)).sum())} 行) "
              f"| 最大逐行跳变 {jump:.1f}°")
        return q, float(good.mean()), pe, re, marg

    # P1 是数据轨迹第 0 行：右手从这里就消费完整参考，不再把 f35 前的
    # 微动替换成硬冻结。专门扫描出的 P1 可达解作热启，保证从正确分支起步。
    # 右臂冗余分支以 f60 抓盖锚反向选定。前向从 P1 解会在 f39 附近遇到
    # 奇异区并跳到另一支；关节限速把该跳变摊成 7 行甩臂。f60 已由上面的
    # GraspPose 扫描给出可行解，从它反求到 P1 可在整段开始前就选对分支，
    # 不改任何腕目标/数据时序。f60 后再沿同一锚正向求搬运与放置。
    _qr0, _, _per0, _rer0, _mgr0 = solve_side(
        "right", wr_P[:right_grasp_k + 1][::-1],
        wr_Q[:right_grasp_k + 1][::-1], seed=_q_grasp_r,
        tag="(f60->P1反向选支探针)")
    _qr0, _per0, _rer0, _mgr0 = (
        _qr0[::-1], _per0[::-1], _rer0[::-1], _mgr0[::-1])
    # 同一连续分支在 P1 可能只能逼近最初单帧扫描出的等待腕位。用该分支
    # P1 构型的真实 FK 作为新的可达等待锚，并沿原始右手运动进度把修正平滑
    # 衰减到 f60 的 GraspPose（f60 端点完全不动）。f35 前 progress=0，修正
    # 是常量，所以原数据的近静止位移/时序也原样保留。
    _ikr_branch = _mk_ik("right")
    _bp, _bR = _ikr_branch.fk(_qr0[0])
    _bq = R_to_quat(_bR)
    _dp_branch = _bp - wr_P[0]
    _dq_branch = qmul(_bq[None], qconj(wr_Q[0:1]))[0]
    _dq_ang = 2.0 * np.arccos(np.clip(abs(_dq_branch[0]), 0.0, 1.0))
    if np.linalg.norm(_dp_branch) > 0.002 or _dq_ang > np.radians(2.0):
        _ident = np.array([1.0, 0.0, 0.0, 0.0])
        for _k in range(right_grasp_k + 1):
            _fade = 1.0 - float(right_motion_progress_src[_k])
            wr_P[_k] += _fade * _dp_branch
            _qc = _slerp_one(_ident, _dq_branch, _fade)
            wr_Q[_k] = qmul(_qc[None], wr_Q[_k:_k + 1])[0]
            wr_Q[_k] /= np.linalg.norm(wr_Q[_k])
        _map_arc = float(np.linalg.norm(
            np.diff(wr_P[:right_grasp_k + 1], axis=0), axis=1).sum())
        print(f"[v1] 右腕连续分支 P1 重锚: Δp="
              f"{np.round(_dp_branch * 100, 2)}cm ΔR="
              f"{np.degrees(_dq_ang):.1f}°; 修正在 f60 衰减为0")
        _qr0, _, _per0, _rer0, _mgr0 = solve_side(
            "right", wr_P[:right_grasp_k + 1][::-1],
            wr_Q[:right_grasp_k + 1][::-1], seed=_q_grasp_r,
            tag="(f60->P1反向选支重锚后)")
        _qr0, _per0, _rer0, _mgr0 = (
            _qr0[::-1], _per0[::-1], _rer0[::-1], _mgr0[::-1])
    _qr1, _, _per1, _rer1, _mgr1 = solve_side(
        "right", wr_P[right_grasp_k:], wr_Q[right_grasp_k:],
        seed=_qr0[-1], tag="(f60后搬运/放置)")
    q_r = np.concatenate([_qr0, _qr1[1:]])
    pe_r = np.concatenate([_per0, _per1[1:]])
    re_r = np.concatenate([_rer0, _rer1[1:]])
    mg_r = np.concatenate([_mgr0, _mgr1[1:]])
    ok_r = float(((pe_r < 0.02) & (re_r < np.radians(10.0))).mean())
    print(f"[v1] right IK(P1后完整数据轨迹, f60锚定): "
          f"位置<2cm且姿态<10° {ok_r * 100:.0f}% | "
          f"全行中位 {np.median(pe_r) * 100:.2f}cm/"
          f"{np.degrees(np.median(re_r)):.1f}° | 90分位 "
          f"{np.percentile(pe_r, 90) * 100:.2f}cm | 限位余量中位 "
          f"{np.degrees(np.median(mg_r)):.1f}°")
    q_l, ok_l, pe_l, re_l, mg_l = solve_side("left", wl_P, wl_Q)

    # InitialPose 单一来源。T2-12 起它不只是机器段起点, 还是右臂在瓶身水平前
    # 必须逐行精确保持的姿态, 因而在生成任何 pregrasp 候选前先取出来。
    if rest and rest.get("stance_arm14") is not None:
        st = np.asarray(rest["stance_arm14"], float)
        st_r, st_l = st[:7], st[7:14]
    else:
        st_r = np.radians([-45.0, 0, 0, -90.0, 0, 0, 0])
        st_l = np.radians([45.0, 0, 0, -90.0, 0, 0, 0])
    if rest and rest.get("stance_fin44") is not None:
        sf = np.asarray(rest["stance_fin44"], float)
        sf_r, sf_l = sf[:22], sf[22:44]
    else:
        sf_r = sf_l = np.zeros(22)

    # ---- 机器段 pregrasp 内点构型 (2026-08-30 cuRobo 排障定稿, 台账 T2-2) ----
    # 位姿 IK 规划对贴限锚不可用: ArmIK 钳限位出解 (j7 钉 -79° 也算达标),
    # cuRobo 带限位余量把贴边构型全拒 —— Approach 位姿模式全灭。机器段改
    # cspace 直达关节构型: 收缩限位 3° 的 ArmIK 解 station+净空 的最近内点,
    # 误差原样入档 (左臂 ~8cm/54° 是腕行程物理极限, 缝1+RL 消化;
    # 右 pregrasp 抬升 4cm —— 8cm 超可达域, 离线 cuRobo 实测 ≤5cm 才通)。
    # 净空**候选梯子**: P0->P1 的两条手臂都是 cuRobo 机器段。规划器
    # 逐个试，第一条
    # cspace 通的就用 —— 2026-08-31 的教训: 左抓锚修准之后 pregrasp 正好落在
    # 瓶壁上, 充气 1cm 的障碍把**目标构型**判碰, Approach 全灭; 而净空给多少
    # 才够是逐 clip 的几何问题, 手调标定跑不动 17 条的数据引擎。右手候选用
    # P1 构型向 P0 的关节空间内收，与左手各档成对提交联合规划。
    PRE_LADDER = [(TC.PRE_L_RADIAL, 0.00), (0.12, 0.06),
                  (0.16, 0.06), (0.20, 0.10),
                  (0.24, 0.14), (0.28, 0.18)]
    _m3 = np.radians(3.0)
    _radL = wl_P[0] - body_p[0]
    _radL[2] = 0.0
    _radL /= max(np.linalg.norm(_radL), 1e-9)
    _ikp = {sd: _mk_ik(sd) for sd in ("left", "right")}
    for _sd in ("left", "right"):
        _ikp[_sd].lower = _ikp[_sd].lower + _m3
        _ikp[_sd].upper = _ikp[_sd].upper - _m3
    pre_alts = {"left": [], "right": []}
    _pre_alpha = np.linspace(0.20, 0.90, len(PRE_LADDER))
    for (_lr, _ll), _a in zip(PRE_LADDER, _pre_alpha):
        _pp = wl_P[0] + _lr * _radL + [0.0, 0.0, _ll]
        _sp = _ikp["left"].solve_traj(np.asarray(_pp, float)[None],
                                      np.asarray(wl_Q[0], float)[None],
                                      w_rot=0.25, n_restart=24)[0]
        pre_alts["left"].append(np.asarray(_sp["q"], np.float64))
        pre_alts["right"].append(
            (1.0 - _a) * q_r[0] + _a * st_r)
        print(f"[v1] 开场左手 pregrasp (径向{_lr * 100:.0f}cm/抬{_ll * 100:.0f}cm): "
              f"距锚 {np.linalg.norm(_ikp['left'].fk(pre_alts['left'][-1])[0] - _pp) * 100:.2f}cm; "
              f"右臂 P1->P0 内收 {_a * 100:.0f}%")
    machine_pre = {"left": pre_alts["left"][0],
                   "right": pre_alts["right"][0]}
    pre_alts = {sd: np.stack(v) for sd, v in pre_alts.items()}

    # 任务终点 -> P0 也完整交给 cuRobo。交互末行手贴着瓶/盖，先向 P0
    # 取一组关节空间净空内点，cuRobo 在收缩目标物世界规划“末行->净空点”，
    # 再在完整障碍世界规划“净空点->P0”。不复用 Approach 倒放。
    _ret_alpha = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 0.90])
    retreat_pre_alts_r = np.stack([
        (1.0 - _a) * q_r[-1] + _a * st_r for _a in _ret_alpha])
    retreat_pre_alts_l = np.stack([
        (1.0 - _a) * q_l[-1] + _a * st_l for _a in _ret_alpha])
    for _a, _qrp, _qlp in zip(_ret_alpha, retreat_pre_alts_r,
                               retreat_pre_alts_l):
        _dr = np.degrees(np.abs(_qrp - q_r[-1]).max())
        _dl = np.degrees(np.abs(_qlp - q_l[-1]).max())
        print(f"[v1] 收尾双臂 cspace 净空: 向 P0 退 {_a * 100:.0f}% "
              f"(距任务末行 R{_dr:.1f}°/L{_dl:.1f}°)")

    # 手指行: 右手 = 活的人手流 (拧盖手法, 本批实证活通道, P-HYB 形状指引同源);
    # 左手 = prior 抓形模板 (与镜像腕位姿配套 —— 人手指流描述的是**人的**握法,
    # 锚在死腕点上, 与 prior 握位几何不配; β squeeze 由 env 前馈负责加压)
    # 右指形 = 候选的 grasp 模板 (与左手同法)。人手指流仍存进 human_right_f,
    # 供 P-HYB 的指形指引奖使用 —— 但**参考**必须是这只手能形成握的那组角。
    f_r_human = smooth(np.asarray(qr["finger_qpos"], float)[ia_src_i], 2.0)
    from rl_rebuild.correction.ref_builders.replay_grasp import (
        GENERIC_JOINT_ORDER)
    gfin = np.asarray(zp["grasp"], np.float64)[7:29]
    perm = [GENERIC_JOINT_ORDER.index(n) for n in fin_names]
    f_l = np.tile(gfin[perm], (N, 1))
    # 右手同法: 候选的 grasp 指值 (右手约定, 按名映射, 不镜像)
    _cap_grasp_fin = np.asarray(_capz["grasp"], np.float64)[7:29][perm].copy()
    # 站位行捏合量: 对准把拇/食指放到盖轴两侧后, 还差 ~1.8cm 跨距 (盖径 3.5cm)
    _curl = np.zeros(22)
    for _i, _n in enumerate(fin_names):
        if not _n.startswith(("right_thumb", "right_index", "right_middle")):
            continue
        if _n.endswith(("MCP_FE", "PIP", "IP")):
            _curl[_i] = 1.0
        elif _n.endswith("DIP"):
            _curl[_i] = 0.5
        elif _n == "right_thumb_CMC_AA":
            _curl[_i] = 1.0
    _cap_grasp_fin = _cap_grasp_fin + np.radians(TC.CAP_PINCH_DEG) * _curl
    # P1 时右手不再预先捏紧：在原数据到达“瓶身已水平且右掌
    # 开始动”的 f35 前，指形精确保持 P0 张开态 sf_r。f35 后才随
    # 原始右掌路径进度先形成 pre-shape，靠近后半段再合指，f60 精确到
    # GraspPose。这把“左手抓瓶”和“右手捏盖”从原来的同时合指改成
    # 由瓶身倾角触发的先后时序。人手 finger_qpos 活轨迹仍单独存档。
    _right_pre_fin = _cap_grasp_fin + 0.35 * (sf_r - _cap_grasp_fin)
    _right_move_u = np.asarray(right_motion_progress_src, np.float64).copy()
    _right_pre_u = np.clip(_right_move_u / 0.45, 0.0, 1.0)
    _right_pre_u = _right_pre_u * _right_pre_u * (3.0 - 2.0 * _right_pre_u)
    _right_close_u = np.clip((_right_move_u - 0.55) / 0.45, 0.0, 1.0)
    _right_close_u = _right_close_u * _right_close_u * (3.0 - 2.0 * _right_close_u)
    f_r = (sf_r[None]
           + _right_pre_u[:, None] * (_right_pre_fin - sf_r)[None]
           + _right_close_u[:, None]
           * (_cap_grasp_fin - _right_pre_fin)[None])
    # 最后的显式放盖窗同步张手；脱盖后到此窗之前一直保持
    # GraspPose + squeeze，不允许视频/训练里出现“手先走、盖自己飞”。
    _right_release_u = np.zeros(N, np.float64)
    _right_release_u[cap_place_start_k:] = _place_u
    f_r = ((1.0 - _right_release_u[:, None]) * f_r
           + _right_release_u[:, None] * sf_r[None])
    _right_grip_u = _right_close_u * (1.0 - _right_release_u)
    assert np.abs(f_r[:right_start_k + 1] - sf_r).max() < 1e-9, \
        "右手在瓶身水平触发前不得预合指"
    assert _right_grip_u[right_grasp_k] > 1.0 - 1e-9
    assert _right_grip_u[-1] < 1e-9
    # 右手 squeeze 增量随母带走 (env 的 βR 前馈读它): 候选自己的 squeeze-grasp,
    # 与左手对称。原来 env 里写死 beta_r*zeros(22), βR 提上去也不生效。
    sq_delta_r = (np.asarray(_capz["squeeze"], np.float64)[7:29][perm]
                  - np.asarray(_capz["grasp"], np.float64)[7:29][perm])
    _cap_sq = np.asarray(_capz["squeeze"], np.float64)[7:29][perm]
    print(f"[v1] 右指形 = P1 pre-shape -> f{right_grasp_src} grasp "
          f"(P1/grasp 中位 {np.degrees(np.median(f_r[0])):.1f}°/"
          f"{np.degrees(np.median(_cap_grasp_fin)):.1f}°); squeeze 增量中位 "
          f"{np.degrees(np.median(np.abs(_cap_sq - f_r[0]))):.1f}° (βR={TC.BETA_R})")

    # ---- 机器段 (占位: 关节 smoothstep; 正式训练前换规划轨迹, 见 docstring) ----

    def ramp(a, b, K):
        s = np.linspace(0, 1, K, endpoint=False)
        s = s * s * (3 - 2 * s)
        return a[None] * (1 - s)[:, None] + b[None] * s[:, None]

    def ramp_closed(a, b, K):
        """含首末端点的 smoothstep；用于必须精确交接的后置右手段。"""
        s = np.linspace(0, 1, K, endpoint=True)
        s = s * s * (3 - 2 * s)
        return a[None] * (1 - s)[:, None] + b[None] * s[:, None]

    # ⚠ 摘要必须按**本次要写出的**几何算, 不能读磁盘上的旧 v1: 否则改了
    # P1/任务末行/净空候选后，旧规划会被照旧剪进新母带。摘要只包含 cuRobo
    # 真正消费的两端、候选和末态障碍位姿；P1->任务末行的中间数据曲线不属于机器规划。
    _basis_now = {
        "source": np.ones(N, np.int8),
        "station_wr": np.r_[wr_P[0], wr_Q[0]],
        "station_wl": np.r_[wl_P[0], wl_Q[0]],
        "station_q_r": q_r[0],
        "station_q_l": q_l[0],
        "machine_pre_q_r": machine_pre["right"],
        "machine_pre_q_l": machine_pre["left"],
        "machine_pre_alts_r": pre_alts["right"],
        "machine_pre_alts_l": pre_alts["left"],
        "retreat_station_q_r": q_r[-1],
        "retreat_station_q_l": q_l[-1],
        "machine_retreat_pre_alts_r": retreat_pre_alts_r,
        "machine_retreat_pre_alts_l": retreat_pre_alts_l,
        "obj_pos_0": body_p, "obj_quat_0": body_q,
        "obj_pos_1": cap_p, "obj_quat_1": cap_q,
    }
    planning_basis = TC.reference_planning_digest(_basis_now)
    rest_md5 = TC.file_md5(args.rest_json)

    def _plan_usable(p, segment):
        # 规划**失败**的产物 (ok=False) 一律当作"没有规划": 否则 bootstrap 死循环
        # —— 要重建母带才能出新的规划目标, 而重建被上一次的失败产物拦住。
        if not os.path.isfile(p):
            return False
        try:
            with np.load(p, allow_pickle=True) as _pz:
                if not bool(_pz["ok"]):
                    print(f"[v1] ⚠ {os.path.basename(p)} 是失败产物 (ok=False), "
                          f"当作无规划处理 (机器段退占位)")
                    return False
                _got = {key: str(np.asarray(_pz[key]).item()) for key in
                        ("plan_clip", "plan_segment", "planning_basis_digest",
                         "rest_md5") if key in _pz.files}
                _want = {"plan_clip": TC.CLIP_ID, "plan_segment": segment,
                         "planning_basis_digest": planning_basis,
                         "rest_md5": rest_md5}
                if _got != _want:
                    print(f"[v1] ⚠ {os.path.basename(p)} 指纹已过期, 当作无规划: "
                          f"got={_got} want={_want}")
                    return False
        except Exception as _e:
            print(f"[v1] ⚠ {p} 不可读 ({type(_e).__name__}), 当作无规划")
            return False
        return True
    have_approach = _plan_usable(TC.APPROACH_NPZ, "approach")
    have_retreat = _plan_usable(TC.RETREAT_NPZ, "retreat")

    def _load_plan(npz_path, segment):
        """cuRobo worker 产物 -> (右臂行, 左臂行)。列按 joint_names 名取。"""
        pz = np.load(npz_path, allow_pickle=True)
        assert bool(pz["ok"]), f"{npz_path}: 规划失败产物 (ok=False)"   # 上游已筛
        provenance = {key: str(np.asarray(pz[key]).item()) for key in
                      ("plan_clip", "plan_segment", "planning_basis_digest",
                       "rest_md5") if key in pz.files}
        expected = {"plan_clip": TC.CLIP_ID, "plan_segment": segment,
                    "planning_basis_digest": planning_basis, "rest_md5": rest_md5}
        assert provenance == expected, (
            f"{npz_path}: 规划产物与当前母带/静置不匹配；重新运行 plan_machine_segs",
            provenance, expected)
        names = [str(n) for n in pz["joint_names"]]
        tr = np.asarray(pz["traj"], np.float64)
        assert np.isfinite(tr).all() and tr.ndim == 2 and len(tr) >= 2, npz_path
        ir = [names.index(f"R_arm_j{i}") for i in range(1, 8)]
        il = [names.index(f"L_arm_j{i}") for i in range(1, 8)]
        return tr[:, ir], tr[:, il]

    # Approach: cuRobo 规划产物优先 (碰撞检查过的可行解; plan_machine_segs.py 产出)
    if have_approach:
        app_r, app_l = _load_plan(TC.APPROACH_NPZ, "approach")
        n_app = len(app_r)
        print(f"[v1] Approach = cuRobo 规划 {TC.APPROACH_NPZ} ({n_app} 行)")
        assert np.abs(app_r[0] - st_r).max() < 0.06 \
            and np.abs(app_l[0] - st_l).max() < 0.06, \
            "Approach 首行必须是 P0"
        assert np.abs(app_r[-1] - q_r[0]).max() < np.radians(2.0) \
            and np.abs(app_l[-1] - q_l[0]).max() < np.radians(2.0), \
            "Approach 末行必须精确到 P1 双臂构型"
    else:
        print("[v1] ⚠ Approach = P0->P1 双臂 smoothstep 占位 "
              "(无碰撞背书; 先跑 "
              "A_Design/L1_Data/Motion_Planning/plan_machine_segs.py)")
        app_r = ramp_closed(st_r, q_r[0], APP_ROWS)
        app_l = ramp_closed(st_l, q_l[0], APP_ROWS)
        n_app = APP_ROWS
    # 手指时序 (2026-08-31 修, 冒烟实测 D2瓶倒):
    #   机器段**全程保持站姿指形** —— cuRobo 规划机器段时手指就锁在这一组,
    #   原来"approach 后 30% 先合拢"与规划的避障假设自相矛盾;
    #   合拢挪到缝1 的**后段**: 缝1 是从 pregrasp 净空横扫到抓握位的那 25 行,
    #   闭合的手一路扫过去会把瓶推倒 (净空从 5cm 提到 20cm 后必现)。
    #   现在: 缝1 前 40% 保持张开 (完成靠近), 后 60% 才合到抓握形 (人到位再合手)。
    app_fr = np.tile(sf_r, (n_app, 1))
    app_fl = np.tile(sf_l, (n_app, 1))
    # 缝1 = 从 pregrasp 净空**沿笛卡尔直线**进刀到抓握位姿 (2026-08-31 改)。
    # 原来是关节空间直线插值 —— 关节直线 ≠ 笛卡尔直线, 手在合拢途中划弧**扫穿
    # 瓶身**, 实测把瓶直接扫倒 (站位时瓶倾角 90°)。左抓锚还偏高 10cm 时反而
    # "躲开"了, 锚一修对就撞上。腕位姿线性插值 + 逐行 IK = 标准的沿抓取轴进刀。
    def _seam_cartesian(side, q_from, q_to, K):
        ik = _mk_ik(side)
        p0v, R0v = ik.fk(q_from)
        p1v, R1v = ik.fk(q_to)
        q0v, q1v = R_to_quat(R0v), R_to_quat(R1v)
        if float(np.dot(q0v, q1v)) < 0:
            q1v = -q1v
        # 两段进刀 (2026-08-31 实测定稿): 直线斜插会让**掌部**掠过瓶顶把瓶扫倒
        # (逐行探针: row96 起倾, 而此时指垫 0 接触、指力 0N —— 撞的是连杆不是指尖)。
        # 先在径向外侧降到抓握高度, 再**纯径向**平进 —— 标准 pregrasp 进刀。
        # 右手那种"正上方下压"的情形自动退化成单段垂直下降 (水平分量≈0)。
        _d = p0v - p1v
        _pm = p1v + np.array([_d[0], _d[1], 0.0])
        _leg1 = max(1, int(round(K * 0.4)))
        out = np.zeros((K, 7))
        seed = np.asarray(q_from, float)
        nbad = 0
        for i in range(K):
            a = (i + 1) / (K + 1)
            if i < _leg1:               # 第一段: 对高度 (径向不动)
                b = (i + 1) / _leg1
                pt = (1 - b) * p0v + b * _pm
            else:                       # 第二段: 纯径向平进
                b = (i + 1 - _leg1) / max(K - _leg1, 1)
                pt = (1 - b) * _pm + b * p1v
            qt = (1 - a) * q0v + a * q1v
            qt = qt / np.linalg.norm(qt)
            r = ik.solve(pt, quat_to_R(qt), q0=seed, iters=200, w_rot=0.25)
            if (r["pos_err"] < 0.02 and r["rot_err"] < np.radians(12)
                    and np.abs(np.asarray(r["q"], float) - seed).max()
                    < np.radians(25)):
                out[i] = r["q"]
                seed = np.asarray(r["q"], float)
            else:                       # 解不出来就退回关节直线的那一格
                out[i] = (1 - a) * np.asarray(q_from, float) + a * np.asarray(q_to, float)
                seed = out[i]
                nbad += 1
        print(f"[v1] 缝1 {side}: 笛卡尔进刀 {K} 行 (退回关节插值 {nbad} 行)")
        return out

    # 缝1 分两段 (2026-08-31 实测定稿): **先到位, 再合手**。
    #   前 70%: 笛卡尔进刀, 手保持站姿张开;
    #   后 30%: 手臂**停住**在抓握位姿, 只合手指 (含 β squeeze 渐入)。
    # 原来"边移动边合拢"实测在缝1 第 100 行起把瓶推倒 (105 行时左垫已 3 个、
    # 左指峰值 12N —— 手是抓上了, 但那是**推**不是**握**): 0.53kg 的自由瓶
    # 受一侧指力就倒。到位后再合, 指力才是对称的。
    # 两只手的抓法不同, 合手时序也必须不同 (2026-08-31 逐行实测定稿):
    #   左手 = 侧向环抱瓶身 -> **先径向进刀 (手张开), 到位后再合手**;
    #     边走边合会用一侧指力把 0.53kg 的自由瓶推倒 (实测 105 行左指峰值 12N)。
    #   右手 = 自上而下捏盖 -> **先在高处预合成捏握手型, 再下降** (pre-shape);
    #     张开手掌心朝下时"腕→指尖"约 20cm (旧 HAND_DROP=0.203 正是这个数),
    #     捏握时只有 15.7cm —— 张着手下降, 伸出的指尖正好压在盖顶上把瓶推倒
    #     (实测 row86 起倾而指垫 0 接触: 撞的是指节不是指腹)。
    def _sf_of(side):
        return sf_l if side == "left" else sf_r

    _kpre = max(1, int(SEAM1 * 0.25))   # ① 预张开 (手臂原地不动)
    _kclose = max(1, int(SEAM1 * 0.40))  # ③ 合拢 (手臂到位不动, 放慢=不推倒瓶)
    _kmov = SEAM1 - _kpre - _kclose    # ② 进刀行数
    # 顺序照 prior 自己的配方: **先预合成 grasp 杯状手型 (手臂不动) -> 杯状手
    # 进刀 -> 到位后才加 squeeze**。
    #   prior 的 pregrasp 阶梯就是"同一手型沿进刀轴后退 2cm", 指值与 grasp 只差
    #   0.5~5°, 从不摊平; 而站姿手型是全 0° 的**平手**, 比 grasp 大一圈 ——
    #   张着平手进刀, 伸出的指节会扫穿瓶身/压在盖顶上 (实测两只手各倒一次:
    #   右手 row86 压盖、左手 row99 扫瓶, 两次都是指垫 0 接触 = 撞的是指节)。
    #   squeeze 留到到位之后 (TC.SEAM_MOVE_FRAC), 边走边压会把自由瓶推倒。
    # 预张开手型 = 抓握手型沿**合拢方向**反向外推 (逐关节封顶 25°):
    # 杯口比物体粗一圈才进得去。用站姿(全 0° 平手)当预张开是不行的 —— 平手比
    # 抓握手型大得多, 右手会用伸直的指节压在盖顶上 (实测 row86 起瓶就倒)。
    # 机器段是否已经把臂送到**操作位置**? (2026-08-31 用户拍板的四段结构:
    # 初始位置 -> cuRobo 到操作位置 -> 操作 -> cuRobo 回初始位置)
    # cuRobo 的第二腿 (排除目标物体) 成功时, app_*[-1] 就是站位构型 —— 缝1
    # 于是退化成"手臂不动、只合手指", 手写的笛卡尔进刀 (碰倒瓶的来源) 退役。
    _at_station = (float(np.abs(app_l[-1] - q_l[0]).max()) < np.radians(2.0)
                   and float(np.abs(app_r[-1] - q_r[0]).max()) < np.radians(2.0))
    print(f"[v1] Approach 终点 {'= P1 (缝1 双臂不动)' if _at_station else '= 净空点 (缝1 需进刀)'}"
          f" | 距 P1 R{np.degrees(np.abs(app_r[-1] - q_r[0]).max()):.1f}°/"
          f"L{np.degrees(np.abs(app_l[-1] - q_l[0]).max()):.1f}°")
    _OPEN_K, _OPEN_CAP = 2.0, np.radians(25.0)
    _sqz_l = np.asarray(zp["squeeze"], np.float64)[7:29][perm]   # 左手合拢方向
    for _side, _qa, _qb, _ff in (("left", app_l[-1], q_l[0], f_l[0]),):
        _open = _ff - np.clip(_OPEN_K * (_sqz_l - _ff), -_OPEN_CAP, _OPEN_CAP)
        if _at_station:
            # 臂已在操作位置: 全程不动, 手指 先张开(短) -> 合拢(长)
            _kop = max(1, int(SEAM1 * 0.25))
            _arm = np.tile(_qb, (SEAM1, 1))
            _fin = np.concatenate([ramp(_sf_of(_side), _open, _kop),
                                   ramp(_open, _ff, SEAM1 - _kop)])
        else:
            _arm = np.concatenate([np.tile(_qa, (_kpre, 1)),
                                   _seam_cartesian(_side, _qa, _qb, _kmov),
                                   np.tile(_qb, (_kclose, 1))])
            _fin = np.concatenate([ramp(_sf_of(_side), _open, _kpre),
                                   np.tile(_open, (_kmov, 1)),
                                   ramp(_open, _ff, _kclose)])
        print(f"[v1] 缝1 {_side}: " + (
            f"臂不动, 手指 张开 {np.degrees(np.abs(_open - _ff)).mean():.1f}° "
            f"-> 合拢 (共 {SEAM1} 行)" if _at_station else
            f"预张开 {_kpre} 行 (张开量中位 "
            f"{np.degrees(np.abs(_open - _ff)).mean():.1f}°) -> 进刀 {_kmov} 行 "
            f"-> 合拢 {_kclose} 行"))
        s1_l, s1_fl = _arm, _fin
    # 右臂也已由 Approach 到 P1；缝1 不合右手，保持 P0 张开指形。
    # 进入交互后仍先等瓶身达到原数据给出的水平触发点，之后才随
    # 数据轨迹预成形/合指。
    if _at_station:
        s1_r = np.tile(q_r[0], (SEAM1, 1))
    else:
        s1_r = _seam_cartesian("right", app_r[-1], q_r[0], SEAM1)
    s1_fr = np.tile(sf_r, (SEAM1, 1))

    # ---- P1 -> 任务末行: 单一连续数据/RL 段 --------------------------
    # 不插行、不冻结物体、不加任何 cuRobo 子段。right_start_k 只是数据里
    # “右手明显启动”的诊断锚，不是正在运行时的硬门。
    assert len(q_r) == len(q_l) == len(body_p) == len(cap_p) == N
    assert 0 < right_start_k < right_grasp_k < N
    print(f"[v1] P1->任务末行 = 连续数据/RL 轨迹 {N} 行 | "
          f"右手明显启动 k={right_start_k} (f{right_start_src}), "
          f"到盖 k={right_grasp_k} (f{right_grasp_src}), k_sep={k_sep}")

    # Retreat: cuRobo cspace 规划产物优先 (物体已在终位, 躲避着回站姿)
    if have_retreat:
        ret_r, ret_l = _load_plan(TC.RETREAT_NPZ, "retreat")
        print(f"[v1] Retreat = cuRobo 规划 {TC.RETREAT_NPZ} ({len(ret_r)} 行)")
        assert np.abs(ret_r[0] - q_r[-1]).max() < np.radians(2.0) \
            and np.abs(ret_l[0] - q_l[-1]).max() < np.radians(2.0), \
            "Retreat 首行必须是数据任务实际末行"
        assert np.abs(ret_r[-1] - st_r).max() < 0.06 \
            and np.abs(ret_l[-1] - st_l).max() < 0.06, \
            "Retreat 末行须回到 P0 (progress G4 拿它当 InitialPose)"
        ret_r = np.concatenate([ret_r, st_r[None]])   # 末行=精确站姿 (判据口径)
        ret_l = np.concatenate([ret_l, st_l[None]])
    else:
        print("[v1] ⚠ Retreat = 任务末行->P0 smoothstep 占位 "
              "(无碰撞背书; 正式训练前必须生成 cuRobo 产物)")
        ret_r = ramp_closed(q_r[-1], st_r, RET_ROWS)
        ret_l = ramp_closed(q_l[-1], st_l, RET_ROWS)
    s2_r = ramp(q_r[-1], ret_r[0], SEAM2)
    s2_l = ramp(q_l[-1], ret_l[0], SEAM2)
    s2_fr, s2_fl = ramp(f_r[-1], sf_r, SEAM2), ramp(f_l[-1], sf_l, SEAM2)
    ret_fr = np.tile(sf_r, (len(ret_r), 1))
    ret_fl = np.tile(sf_l, (len(ret_l), 1))

    right_q = np.concatenate([app_r, s1_r, q_r, s2_r, ret_r])
    left_q = np.concatenate([app_l, s1_l, q_l, s2_l, ret_l])
    right_f = np.concatenate([app_fr, s1_fr, f_r, s2_fr, ret_fr])
    left_f = np.concatenate([app_fl, s1_fl, f_l, s2_fl, ret_fl])
    Tn = len(right_q)
    seg_lens = np.array([len(app_r), SEAM1, N, SEAM2, len(ret_r)], np.int64)
    assert seg_lens.sum() == Tn

    # ---- 物体列全链 ----
    n_pre, n_post = len(app_r) + SEAM1, SEAM2 + len(ret_r)
    def obj_cols(p_ia, q_ia, rest7):
        pre_p = np.tile(rest7[:3], (n_pre, 1))
        pre_q = np.tile(rest7[3:7], (n_pre, 1))
        post_p = np.tile(p_ia[-1], (n_post, 1))
        post_q = np.tile(q_ia[-1], (n_post, 1))
        return (np.concatenate([pre_p, p_ia, post_p]),
                np.concatenate([pre_q, q_ia, post_q]))
    bp_all, bq_all = obj_cols(body_p, body_q, rest_b)
    cp_all, cq_all = obj_cols(cap_p, cap_q, rest_c)

    # ---- conf 列 (物体逐帧真话版 + 人手置信度), 机器行 NaN ----
    ovm = np.load(os.path.join(take, "object_valid_measured.npz"), allow_pickle=True)
    hc = np.load(os.path.join(take, "hand_confidence.npz"), allow_pickle=True)
    def conf_col(arr):
        v = np.full(Tn, np.nan)
        v[n_pre:n_pre + N] = np.asarray(arr, float)[ia_src_i]
        return v
    cols = {
        "conf_pos_0": conf_col(ovm["conf_pos_per_frame"][bi]),
        "conf_rot_0": conf_col(ovm["conf_rot_per_frame"][bi]),
        "conf_pos_1": conf_col(ovm["conf_pos_per_frame"][ci]),
        "conf_rot_1": conf_col(ovm["conf_rot_per_frame"][ci]),
        "hand_conf_pos_l": conf_col(hc["conf_pos"][0]),
        "hand_conf_pos_r": conf_col(hc["conf_pos"][1]),
        "hand_conf_fin_l": conf_col(np.asarray(hc["conf_fingers"], float)[0].mean(1)),
        "hand_conf_fin_r": conf_col(np.asarray(hc["conf_fingers"], float)[1].mean(1)),
    }

    source = np.zeros(Tn, np.int8)
    source[n_pre:n_pre + N] = 1
    frame_of_row = np.full(Tn, -1, np.int32)
    frame_of_row[n_pre:n_pre + N] = ia_src_i
    human_right_f_all = right_f.copy()
    human_right_f_all[n_pre:n_pre + N] = f_r_human

    meta = {
        "task": "unscrew", "clip": TC.CLIP_ID, "take": take,
        "gen": "make_reference_v1_20260829",
        "windows": {"w0": w0, "w1": w1, "sep_src": sep_src,
                    "k_sep": k_sep},
        "right_timing": {
            "rule": "raw bottle tilt + robust right palm displacement",
            "right_start_src": right_start_src,
            "right_start_k": right_start_k,
            "right_grasp_src": right_grasp_src,
            "right_grasp_k": right_grasp_k,
            "bottle_tilt_deg": right_start_tilt,
            "right_palm_disp_mm": right_start_palm_disp * 1000.0,
            "tilt_threshold_deg": float(TC.RIGHT_START_TILT_DEG),
            "palm_threshold_mm": float(TC.RIGHT_START_PALM_DISP_M * 1000.0),
            "pre_start_motion": "preserved from raw P1 hand trajectory",
            "finger_open_until_k": int(right_start_k),
            "finger_rule": "P0-open until bottle-horizontal trigger; then pre-shape/grasp",
            "retarget": "endpoint-constrained similarity",
            "position_scale": _right_map_scale,
            "raw_arc_cm": _raw_arc * 100.0,
            "mapped_arc_cm": _map_arc * 100.0,
            "continuous_branch_reanchor_cm": (_dp_branch * 100.0).tolist(),
            "continuous_branch_reanchor_deg": float(np.degrees(_dq_ang)),
        },
        "hold_yaw_deg": hold_yaw,
        "gauge": {"right_grasp_roll_deg": float(_grasp_roll_deg),
                  "right_roll_drift_deg": float(np.degrees(roll[-1] - yaw_g)),
                  "bottle_spin_frozen_from_row": int(k_sep),
                  "cap_rigid_ride_from_row": int(k_sep),
                  "cap_place_start_row": int(cap_place_start_k),
                  "cap_place_end_row": int(N - 1),
                  "cap_hold_rule": "right-wrist SE(3) grasp then explicit controlled place"},
        "ik": {"right": {"ok": ok_r, "med_cm": float(np.median(pe_r) * 100),
                         "p90_cm": float(np.percentile(pe_r, 90) * 100),
                         "marg_med_deg": float(np.degrees(np.median(mg_r)))},
               "left": {"ok": ok_l, "med_cm": float(np.median(pe_l) * 100),
                        "p90_cm": float(np.percentile(pe_l, 90) * 100),
                        "marg_med_deg": float(np.degrees(np.median(mg_l)))}},
        "rest_source": "probe" if rest else "offline_estimate",
        "left_grasp": {"prior": TC.PRIOR_AUX, "mirror": "xz-plane",
                       "yaw_deg": float(np.degrees(yaw_l))},
        "dead_channel_free": "wrist_pos 静态填充死数据零依赖 (2026-08-30 拍板)",
        "ik_ok": {"right": ok_r, "left": ok_l},
        "obj_map": {"obj_0": f"{BODY_ID} (bottle, left hand, env.object)",
                    "obj_1": f"{CAP_ID} (cap, right hand, env.aux)"},
        "screw": {"turns": TC.SCREW_TURNS, "pitch_m": 0.00318,
                  "closed_offset_m": 0.18},
        "approach": ("curobo:" + TC.APPROACH_NPZ
                     if have_approach else "smoothstep占位"),
        "retreat": ("curobo:" + TC.RETREAT_NPZ
                    if have_retreat else "smoothstep占位"),
        "phase_contract": {
            "P0_to_P1": "cuRobo Approach",
            "P1_to_task_end": "continuous data reference + RL residual",
            "task_end_to_P0": "cuRobo Retreat",
        },
        "planning_basis_digest": planning_basis,
        "rest_md5": rest_md5,
        "notes": (f"重建15fps按 x{_S:.2f} 重采样到 env 20Hz 行轴; "
                  "P1后完整腕轨迹; f35前右指保持P0张开态; "
                  "脱盖后手盖护送至末段显式放下"),
    }
    out = dict(
        # 站位腕靶 (plan_machine_segs 的规划目标) + 交互腕靶全轨 (v2/诊断)
        station_wr=_basis_now["station_wr"],
        station_wl=_basis_now["station_wl"],
        station_q_r=_basis_now["station_q_r"],
        station_q_l=_basis_now["station_q_l"],
        sq_delta_r=sq_delta_r, cap_grasp_src=np.array(os.path.basename(_cf)),
        machine_pre_q_r=machine_pre["right"],
        machine_pre_q_l=machine_pre["left"],
        machine_pre_alts_r=pre_alts["right"],
        machine_pre_alts_l=pre_alts["left"],
        right_start_src=np.array(right_start_src, np.int32),
        right_start_k=np.array(right_start_k, np.int32),
        right_grasp_src=np.array(right_grasp_src, np.int32),
        right_grasp_k=np.array(right_grasp_k, np.int32),
        right_motion_progress=np.asarray(right_motion_progress_src, np.float64),
        right_grasp_progress=np.asarray(_right_grip_u, np.float64),
        retreat_station_q_r=_basis_now["retreat_station_q_r"],
        retreat_station_q_l=_basis_now["retreat_station_q_l"],
        machine_retreat_pre_alts_r=retreat_pre_alts_r,
        machine_retreat_pre_alts_l=retreat_pre_alts_l,
        wrist_tgt_r=np.concatenate([wr_P, wr_Q], axis=1),
        wrist_tgt_l=np.concatenate([wl_P, wl_Q], axis=1),
        human_right_f=human_right_f_all,
        right_q=right_q, left_q=left_q, right_f=right_f, left_f=left_f,
        obj_pos_0=bp_all, obj_quat_0=bq_all, obj_pos_1=cp_all, obj_quat_1=cq_all,
        source=source, frame_of_row=frame_of_row, seg_lens=seg_lens,
        seg_names=np.array(["approach", "seam1", "interact", "seam2", "retreat"]),
        fin_names=np.array(fin_names), meta=np.array(json.dumps(meta)),
        **cols)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp_out = f"{args.out}.tmp.{os.getpid()}"
    with open(tmp_out, "wb") as fh:
        np.savez(fh, **out)
    os.replace(tmp_out, args.out)
    with open(args.out, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()[:8]
    print(f"[v1] 已写 {args.out} md5={md5} 全链 {Tn} 行 "
          f"(app {len(app_r)}/seam1 {SEAM1}/ia {N}/seam2 {SEAM2}"
          f"/ret {len(ret_r)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
