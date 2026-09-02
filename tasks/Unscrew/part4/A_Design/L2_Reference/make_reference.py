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
5. 腕参考**死通道零依赖** (2026-08-30 拍板: 上游腕平移是静态填充死数据,
   clip32 复测腕 <0.6cm 而盖走 66cm —— 一律不用):
   右腕 = 盖心 + reach·螺轴 (活物体轨迹), 姿态 = 活四元数流 + U35 掌轴对准;
   左腕 = Screw27_body GraspPose **镜像**锚在瓶行上 (同 CAD 字节相同,
   绕瓶轴方位角 ArmIK 可达率扫描), 指形 = prior 抓形模板 + env squeeze 加压。
   ArmIK 逐行解臂 (anchor 用 env_rest.json 实测 arm_center, 缺省退 URDF 推导
   —— 有 ~cm 级系统差, v2 重铸时被增量空间锚消掉)
6. 机器段: 有 Motion_Planning/<clip>/{Approach,Retreat}.npz (cuRobo 产物,
   plan_machine_segs.py) 就剪规划行; 没有退回关节 smoothstep 占位 (⚠ 无碰撞
   背书, 只够冒烟)。缝1/缝2 永远做焊接斜坡桥接两套解的分支差。
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
    # 四元数序核验: 与 world_fused 的旋转矩阵交叉对账 (wxyz 应逐位一致)
    w = np.load(os.path.join(take, "world_fused.npz"), allow_pickle=True)
    M = np.asarray(w["object_ob_in_world_all"], np.float64)   # (2,T,4,4)
    q_chk = OP[bi, 0, 3:7]
    err_w = np.abs(quat_to_R(q_chk) - M[bi, 0, :3, :3]).max()
    err_x = np.abs(quat_to_R(np.r_[q_chk[3], q_chk[:3]]) - M[bi, 0, :3, :3]).max()
    assert err_w < err_x and err_w < 0.02, \
        f"replay obj_pose_all 四元数序存疑: wxyz误差{err_w:.3f} vs xyzw {err_x:.3f}"

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
    sep_src = int(hit[0]) if len(hit) else w1
    sep_src = int(np.clip(sep_src, w0 + 1, w1))
    k_sep = sep_src - w0
    print(f"[v1] 盖脱离帧: f{sep_src} (交互行 {k_sep}) | "
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
    # ★ 瓶滚转规范化 (Unscrew/17 台账 §8, 2026-09-01): 瓶绕自身轴的滚转在数据里**不可观**
    #   (回转体, rot_observability 轴向 0.01), 重建给的是任意值; 而右手候选只在滚转的
    #   ±20° 窗口里可用 (盖系抓姿刚性跟盖转)。这里给整段瓶行加一个**常量**局部滚转 Δ,
    #   让接触行的 盖→腕 方向落到离线筛选定档的方向 (近侧、水平: 台账 §8 的 1_47)。
    #   常量滚转不改轴摆动 (数据), 只定这个不可观自由度 —— 等价于选左手 yaw 的自由度。
    _roll_tgt = getattr(TC, "CAP_WRIST_DIR_WORLD", None)
    if _roll_tgt is not None and getattr(TC, "NATIVE_PRIORS", False):
        try:
            _ca0 = json.load(open(os.path.join(take, "contact_auto.json")))
            _iv0 = [iv for iv in _ca0["annotations"]["right"] if iv[1] >= w0]
            _kc0 = int(np.clip(int(_iv0[0][0]) - w0, 0, N - 1)) if _iv0 else 0
        except Exception:
            _kc0 = 0
        _cfs = sorted(glob.glob(os.path.join(TC.PRIOR_CAP_DIR, "*.npz")))
        if TC.CAP_GRASP_PICK:
            _cfs = [c for c in _cfs if os.path.basename(c).startswith(TC.CAP_GRASP_PICK)] or _cfs
        _gl = np.asarray(np.load(_cfs[0])["grasp"], np.float64)[:3]      # 盖 CAD 系腕位
        _dstar = np.asarray(_roll_tgt, np.float64)
        _dstar /= np.linalg.norm(_dstar)
        _best = None
        for _dd in range(0, 360, 2):
            _h = 0.5 * np.radians(_dd)
            _qz = np.array([np.cos(_h), 0.0, 0.0, np.sin(_h)])
            _qb = qmul(body_q[_kc0:_kc0 + 1], _qz[None])[0]
            _dw = qrot(_qb[None], _gl[None])[0]        # 盖→腕 (世界), 与盖心无关
            _sc = float(np.dot(_dw / np.linalg.norm(_dw), _dstar))
            if _best is None or _sc > _best[0]:
                _best = (_sc, _dd, _dw)
        _h = 0.5 * np.radians(_best[1])
        _qz = np.tile(np.array([np.cos(_h), 0.0, 0.0, np.sin(_h)]), (N, 1))
        body_q = qmul(body_q, _qz)
        body_q /= np.linalg.norm(body_q, axis=1, keepdims=True)
        axis = qrot(body_q, np.tile([0.0, 0.0, 1.0], (N, 1)))
        # 静置姿也同滚转 (机器段/复位用的是交互首行, 二者同源)
        rest_b = np.r_[rest_b[:3], qmul(rest_b[None, 3:7], _qz[:1])[0]]
        rest_c = np.r_[rest_c[:3], qmul(rest_c[None, 3:7], _qz[:1])[0]]
        print(f"[v1] 瓶滚转规范化: 常量局部滚转 {_best[1]}° -> 接触行 (行 {_kc0}) 盖→腕 "
              f"{np.round(_best[2], 3)} (目标方向 {np.round(_dstar, 2)}, cos={_best[0]:.3f})")
        hold_roll_deg = float(_best[1])
    else:
        hold_roll_deg = 0.0

    # 盖行: 脱离前=瓶推导 (盖钉在瓶顶, 与螺旋投影同语义); 脱离后=盖自身轨迹
    # 按**世界平移**换基 (与瓶同一平移 —— 保住"盖终点落在桌面"的不变量:
    # 若按脱离点连续性拼接, U24a 投影造成的座位高度差会把整段送放轨迹整体
    # 压低 ~5cm, 盖末行 z 掉破 D1 死线, 放音自检实锤)。脱离窗 8 行线性混合
    # 消拼接跳变 (该窗 conf 本来就低, 皮筋红档容差吸收)。
    # ★ U13 (2026-09-02 用户抓包"轨迹到桌下"): 桌面净空钳制升级为 Pour L2-6 官方口径 ——
    #   **网格最低点** ≥ 桌+2mm (原来只钳盖原点 z, 盖躺倒时网格底比原点低, 实测穿桌 1.4cm/19行;
    #   瓶末段也 -0.4cm)。9 帧平滑但不低于必要抬升。瓶钳制在盖行派生**之前**做,
    #   咬合段盖跟瓶一起抬, 螺旋刚性不变。
    def _table_lift(P9, Q9, mesh_path9, tag9):
        import trimesh as _tm9
        from rl_rebuild.correction.kinematics import quat_to_R as _q2R9
        _V9 = np.asarray(_tm9.load(mesh_path9, process=False, force="mesh").vertices)  # 全顶点: 采样会漏底圈极值 (U13.1)
        _lift9 = np.zeros(len(P9))
        for _r9 in range(len(P9)):
            _zm9 = (_V9 @ _q2R9(Q9[_r9]).T)[:, 2].min() + P9[_r9, 2]
            _lift9[_r9] = max(0.0, TABLE_Z + 0.002 - _zm9)
        _sm9 = np.convolve(np.pad(_lift9, 4, mode="edge"), np.ones(9) / 9, mode="valid")[:len(P9)]
        _lift9 = np.maximum(_sm9, _lift9)
        P9[:, 2] += _lift9
        _n9 = int((_lift9 > 1e-4).sum())
        print(f"[v1] {tag9} 桌面净空钳制(网格最低点): 触发 {_n9}/{len(P9)} 行 | 最大抬升 {_lift9.max()*100:.1f}cm")
        return P9
    body_p = _table_lift(body_p, body_q, objs[BODY_ID]["mesh"], "瓶")
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
    cap_p = _table_lift(cap_p, cap_q, objs[CAP_ID]["mesh"], "盖")

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

    def _cap_track(cz):
        """候选 -> (腕位轨迹, 腕姿轨迹): 抓取位姿刚性锚在盖行上。

        ⚠ 这批候选的来源是 `bottle_cap_sharpa_wave_**left**` —— 是**左手**抓法,
        给右手用必须镜像 (与左手用 Screw27_body 右手约定时要镜像, 方向相反但
        同一变换): 位姿 (x,-y,z)/(w,-x,y,-z), 指值按名映射。不镜像直接套上去,
        实测指垫落在盖系径向 4.9~9.9cm (盖半径 1.75cm), 合到 45° 也碰不到。
        """
        gp = np.asarray(cz["grasp"], np.float64)[:3].copy()
        gq = np.asarray(cz["grasp"], np.float64)[3:7].copy()
        # ★ 原生右手先验 (Unscrew/17: Dexonomy screw17_cap_right, 物体系 = 盖 CAD 系,
        #   装配态, 带 com_offset 键) —— **不镜像、不做原点修正**; 老 Screw27 候选
        #   (左手抓法, 居中物体系) 照旧镜像 + 补 bbox 中心。
        _native = "com_offset" in getattr(cz, "files", ())
        if not _native:
            gp[1] = -gp[1]                              # 镜像: 左手抓法 -> 右手
            gq = np.array([gq[0], -gq[1], gq[2], -gq[3]])
            gq /= np.linalg.norm(gq)
            gp[2] += _cap_dz
        gp += np.asarray(TC.CAP_GRASP_TRIM, np.float64)   # 闭环实测的对准量
        Pp = cap_p + qrot(cap_q, np.tile(gp, (N, 1)))
        Qq = qmul(cap_q, np.tile(gq / np.linalg.norm(gq), (N, 1)))
        Qq /= np.linalg.norm(Qq, axis=1, keepdims=True)
        Pp[:, 2] = np.maximum(Pp[:, 2], TABLE_Z + 0.02)
        return Pp, Qq

    _best_c = None
    for _cf in _cands:
        _cz = np.load(_cf)
        _P, _Q = _cap_track(_cz)
        _s0 = ik_r0.solve_traj(_P[:1], _Q[:1], w_rot=0.25, n_restart=16)[0]
        _m0 = float(np.minimum(_s0["q"] - ik_r0.lower,
                               ik_r0.upper - _s0["q"]).min())
        _st_ok = (_s0["pos_err"] < 0.02 and _s0["rot_err"] < np.radians(10.0)
                  and _m0 > np.radians(3.0))
        _sols = ik_r0.solve_traj(_P[::4], _Q[::4], w_rot=0.25, n_restart=2)
        _pe = np.array([v["pos_err"] for v in _sols])
        _re = np.array([v["rot_err"] for v in _sols])
        _qs = np.stack([v["q"] for v in _sols])
        _mg = np.minimum(_qs - ik_r0.lower, ik_r0.upper - _qs).min(axis=1)
        _good = (_pe < 0.02) & (_re < np.radians(10.0)) & (_mg > np.radians(3.0))
        _key = (1 if _st_ok else 0, float(_good.mean()),
                -float(np.median(_pe)))
        if _best_c is None or _key > _best_c[0]:
            _best_c = (_key, _cf, _cz, float(_s0["pos_err"]),
                       float(np.degrees(_s0["rot_err"])),
                       np.asarray(_s0["q"], np.float64))
    _key, _cf, _capz, _e0, _r0d, _q_station_r = _best_c
    print(f"[v1] 右抓候选扫描 ({len(_cands)} 个): 选中 "
          f"{os.path.basename(_cf)} 站位行 {_e0 * 100:.2f}cm/{_r0d:.1f}° "
          f"({'可达' if _key[0] else '⚠ 不可达'}) 拧盖窗可达率 {_key[1] * 100:.0f}% "
          f"| 盖系原点修正 {_cap_dz * 100:+.2f}cm")
    wr_P, wr_Q = _cap_track(_capz)
    # ★ U8 (2026-09-01 用户裁定): 右手**接触前原地等**, 盖由左手送过来 —— 数据实证
    #   f25→f35 盖走 16.4cm、右 knuckle 0.1cm、右腕朝向 0~1°。接触前的右腕参考 =
    #   人手接触起点行 (contact_auto 右 onset) 的盖上抓握位姿, 沿盖 PreGrasp 退让方向
    #   后退 PRE_R_WAIT_M, **静止**; 接触行起才跟盖 (开环行; env 内 U9 闭环伺服可覆盖)。
    #   人手 knuckle 代理只动 3cm, 与"静止等待"同义, 不再另建人手系换基。
    _k_contact = 0
    try:
        _ca = json.load(open(os.path.join(take, "contact_auto.json")))
        _iv = [iv for iv in _ca["annotations"]["right"] if iv[1] >= w0]
        if _iv:
            _k_contact = int(np.clip(int(_iv[0][0]) - w0, 0, N - 1))
    except Exception as _e:
        print(f"[v1] ⚠ contact_auto 右手区间不可读 ({_e}), 等待位退化为首行")
    _PRE_R_WAIT_M = float(getattr(TC, "PRE_R_WAIT_M", 0.04))
    if _k_contact > 0:
        _pg = np.asarray(_capz["pregrasp"], np.float64)
        _g0 = np.asarray(_capz["grasp"], np.float64)
        _kf = int(np.argmax(np.linalg.norm(_pg[:, :3] - _g0[:3], axis=1)))
        _back_local = _pg[_kf, :3] - _g0[:3]              # 盖系退让方向
        _bn = np.linalg.norm(_back_local)
        _back_local = _back_local / _bn * _PRE_R_WAIT_M if _bn > 1e-6 else np.zeros(3)
        _wait_p = wr_P[_k_contact] + qrot(cap_q[_k_contact:_k_contact + 1],
                                           _back_local[None])[0]
        _wait_p[2] = max(_wait_p[2], TABLE_Z + 0.03)
        wr_P[:_k_contact] = _wait_p
        wr_Q[:_k_contact] = wr_Q[_k_contact]
        print(f"[v1] 右腕接触前等待位 (U8): 行 0~{_k_contact - 1} 静止在接触行 "
              f"{_k_contact} (f{w0 + _k_contact}) 盖上抓握位姿沿 PreGrasp 方向退 "
              f"{_PRE_R_WAIT_M * 100:.0f}cm; 接触行起跟盖")
    # 脱离后: 腕姿态**冻结**在脱离时刻, 位置保持与盖的世界系刚性偏移。
    # 理由: 盖脱手后由手携带 (下面的 rigid ride 让盖姿态跟手走), 而重建的盖
    # 自身翻滚 107° 是不可信的自转/翻滚 —— 腕跟着刚性翻会直接超出可达域
    # (实测全程可达率掉到 12%、姿态中位差 89°)。冻结姿态 + 位置跟盖 = 携带段
    # 平滑且可达, 盖的落地姿态由脱离时的握法决定 (物理上就该如此)。
    if k_sep < N - 1:
        _off_w = wr_P[k_sep] - cap_p[k_sep]
        wr_Q[k_sep:] = wr_Q[k_sep]
        wr_P[k_sep:] = cap_p[k_sep:] + _off_w
        wr_P[:, 2] = np.maximum(wr_P[:, 2], TABLE_Z + 0.02)
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
    # ⚠ 抓握自转**不再扫描**: 腕姿现在由盖抓取先验确定 (T2-10), 自转不是自由
    # 参数了。这套扫描是几何构造时代的遗产 (那时腕姿=掌轴对准+人腕自转, 绕轴
    # 自转确实自由); 保留它会把先验的捏握姿态再转一个网格档 —— 2026-09-01 实测
    # 正好转了 60°, 站位腕位对得上而**姿态差 59.95°**, 指垫因此落到 r5~9cm。
    yaw_g = 0.0
    roll = np.full(N, yaw_g)

    # ② 脱离后逐行限速漂移: 手与盖一起转 (不滑手), 盖的自转本就不可判
    q_seed = np.asarray(ik_r.solve_traj(wr_P[:ks1], wr_Q[:ks1], w_rot=0.25,
                                        n_restart=2)[-1]["q"], np.float64)
    drift = 0.0
    for k in range(ks1, N):
        best = None
        for d in (-2, -1, 0, 1, 2):
            cand = drift + d * ROLL_STEP
            Qk = _roll_q(wr_Q[k:k + 1], axis_n[k:k + 1], [cand])[0]
            r = ik_r.solve(wr_P[k], quat_to_R(Qk), q0=q_seed, iters=150,
                           w_rot=0.25)
            m = float(np.minimum(r["q"] - ik_r.lower, ik_r.upper - r["q"]).min())
            sc = (r["pos_err"] + 0.25 * r["rot_err"]
                  + max(0.0, np.radians(3.0) - m))
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
    if "com_offset" in zp.files:
        # ★ 原生左手先验 (Unscrew/17: Dexonomy screw17_bottle_left, sharpa_wave_v2_left,
        #   物体系 = 瓶 CAD 系, 瓶底原点): 不镜像、不补原点 (台账 §8 离线 IK 已按此系筛过)
        gp_m = np.array(gp, np.float64)
        gq_m = np.array(gq, np.float64) / np.linalg.norm(gq)
        print("[v1] 左先验 = 原生左手 (不镜像/不补原点):", os.path.basename(TC.PRIOR_AUX))
    else:
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
        _pref = getattr(TC, "LEFT_AZ_PREF_DEG", None)
        if _pref is not None:
            # ★ U6/§8: 左抓方位优先落在人手接近方位 (数据定, 世界方位角), 容差内比可达率
            _az = np.degrees(np.arctan2(P[0, 1] - body_p[0, 1], P[0, 0] - body_p[0, 0]))
            _dpsi = abs((_az - _pref + 180) % 360 - 180)
            key = (1 if _dpsi <= TC.LEFT_YAW_PREF_TOL else 0, ) + key
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
        frozen_run = 0
        q_prev = None if seed is None else np.asarray(seed, float)
        for k in range(n):
            seeds = ([q_prev] if q_prev is not None else []) + [None]
            if q_prev is None or k % 12 == 0:
                seeds += [rng_s.uniform(ik.lower, ik.upper) for _ in range(2)]
            best = None
            for q0 in seeds:
                r = ik.solve(P[k], quat_to_R(Q[k]), q0=q0, iters=200,
                             w_rot=0.25)
                sc = r["pos_err"] + 0.25 * r["rot_err"]
                if best is None or sc < best[0]:
                    best = (sc, r)
            r = best[1]
            ok_k = (np.isfinite(r["q"]).all() and r["pos_err"] < 0.02
                    and r["rot_err"] < np.radians(10.0))
            if k == 0 and q_prev is not None and not ok_k:
                # 站位行是抓握的锚, 不能"因为是第一行"就无条件收下坏解:
                # 多试几个随机重启, 还不行就用种子本身 (扫描验证过的好解)。
                for _t in range(24):
                    _rr = ik.solve(P[k], quat_to_R(Q[k]),
                                   q0=rng_s.uniform(ik.lower, ik.upper),
                                   iters=250, w_rot=0.25)
                    if (_rr["pos_err"] < 0.02
                            and _rr["rot_err"] < np.radians(10.0)):
                        r, ok_k = _rr, True
                        break
                if not ok_k:
                    r = {"q": q_prev, "pos_err": 0.0, "rot_err": 0.0}
                    fp0, fR0 = ik.fk(q_prev)
                    r["pos_err"] = float(np.linalg.norm(fp0 - P[k]))
                    r["rot_err"] = float(np.arccos(np.clip(
                        (np.trace(fR0.T @ quat_to_R(Q[k])) - 1) * 0.5, -1, 1)))
                    ok_k = True
            if ok_k and q_prev is not None and frozen_run < 2 and k > 0:
                # 单行跳变上限 60°: 相邻行跳一支解 = PD 跟不上 (会把物体打飞)。
                # 但**连冻 2 行就放行** —— 否则一次抖动会把整条热启链锁死在
                # 冻结态 (实测把左臂 66% 打到 0%: 冻住后真解越离越远, 永不回来)。
                ok_k = bool(np.abs(np.asarray(r["q"], float) - q_prev).max()
                            < np.radians(60))
            if ok_k or q_prev is None:
                q[k] = r["q"]
                q_prev = q[k].copy()
                pe[k], re[k] = r["pos_err"], r["rot_err"]
                frozen_run = 0
            else:
                q[k] = q_prev           # 冻结: 不跳分支, 差多少如实入账
                frozen_run += 1
                fp, fR = ik.fk(q[k])
                pe[k] = float(np.linalg.norm(fp - P[k]))
                re[k] = float(np.arccos(np.clip(
                    (np.trace(fR.T @ quat_to_R(Q[k])) - 1) * 0.5, -1, 1)))
                frozen += 1
        # 逐行限速: 冻结段重新锁定目标时会一次跳好几十度 (母带里就是"臂瞬移"),
        # PD 跟不上会把桌上的物体扫飞。坏行本来就不准, 把跳变摊到几行里换来
        # 可播放性; 好行几乎不受影响 (跳变本来就 <RATE)。
        RATE = np.radians(20.0)
        for k in range(1, n):
            d = q[k] - q[k - 1]
            m = np.abs(d).max()
            if m > RATE:
                q[k] = q[k - 1] + d * (RATE / m)
        qs = smooth(q, 0.5)
        for k in range(n):
            r = ik.solve(P[k], quat_to_R(Q[k]), q0=qs[k], iters=80, w_rot=0.25)
            # 只在**既更准又不跳分支**时采纳: 重投影可能落到另一支解上, 那会
            # 在母带里留下一行几十度的关节跳变 (PD 跟不上 = 把物体打飞)。
            near = np.abs(np.asarray(r["q"], float) - q[k]).max() < np.radians(20)
            if near and (r["pos_err"] + 0.25 * r["rot_err"]) <= (pe[k]
                                                                 + 0.25 * re[k]):
                q[k], pe[k], re[k] = r["q"], r["pos_err"], r["rot_err"]
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

    # ★ 站位解按**限位余量最大**择优 (48 种子; 台账 §8; 在 U8 等待位/自转漂移之后, 对**最终站位** wr_P[0] 求): 单种子/最小误差解常落在贴限分支
    #   (右臂 j2 顶限), 之后整段热启被拖在坏分支里, 净空点向上 10cm 都抬不起来 (同分支
    #   5~7cm 解不出), 机器段只能跨分支 (缝1 甩 150°)。先把站位放进宽裕分支再热启。
    _rng0 = np.random.default_rng(5)
    _bestq = None
    for _i in range(48):
        _q0 = _q_station_r if _i == 0 else _rng0.uniform(ik_r0.lower, ik_r0.upper)
        _r = ik_r0.solve(wr_P[0], quat_to_R(wr_Q[0]), q0=_q0, iters=250, w_rot=0.25)
        if _r["pos_err"] < 0.01 and _r["rot_err"] < np.radians(8):
            _m = float(np.minimum(_r["q"] - ik_r0.lower, ik_r0.upper - _r["q"]).min())
            if _bestq is None or _m > _bestq[0]:
                _bestq = (_m, np.asarray(_r["q"], np.float64))
    if _bestq is not None:
        _m_old = float(np.minimum(_q_station_r - ik_r0.lower, ik_r0.upper - _q_station_r).min())
        print(f"[v1] 右站位解按限位余量择优: {np.degrees(_m_old):.1f}° -> {np.degrees(_bestq[0]):.1f}°")
        _q_station_r = _bestq[1]
    q_r, ok_r, pe_r, re_r, mg_r = solve_side("right", wr_P, wr_Q,
                                            seed=_q_station_r)
    q_l, ok_l, pe_l, re_l, mg_l = solve_side("left", wl_P, wl_Q)

    # ---- 机器段 pregrasp 内点构型 (2026-08-30 cuRobo 排障定稿, 台账 T2-2) ----
    # 位姿 IK 规划对贴限锚不可用: ArmIK 钳限位出解 (j7 钉 -79° 也算达标),
    # cuRobo 带限位余量把贴边构型全拒 —— Approach 位姿模式全灭。机器段改
    # cspace 直达关节构型: 收缩限位 3° 的 ArmIK 解 station+净空 的最近内点,
    # 误差原样入档 (左臂 ~8cm/54° 是腕行程物理极限, 缝1+RL 消化;
    # 右 pregrasp 抬升 4cm —— 8cm 超可达域, 离线 cuRobo 实测 ≤5cm 才通)。
    # 净空**候选梯子**: 由近及远 (左径向, 左抬升, 右抬升)。规划器逐个试, 第一条
    # cspace 通的就用 —— 2026-08-31 的教训: 左抓锚修准之后 pregrasp 正好落在
    # 瓶壁上, 充气 1cm 的障碍把**目标构型**判碰, Approach 全灭; 而净空给多少
    # 才够是逐 clip 的几何问题, 手调标定跑不动 17 条的数据引擎。右手抬升不超
    # 5cm (T2-2 实测 8cm 已超可达域)。
    PRE_LADDER = [(TC.PRE_L_RADIAL, 0.00, TC.PRE_R_LIFT),
                  (0.12, 0.06, 0.04), (0.16, 0.06, 0.05),
                  (0.20, 0.10, 0.05), (0.24, 0.14, 0.03),
                  (0.28, 0.18, 0.02)]
    if getattr(TC, "PRE_LADDER_OVERRIDE", None):
        PRE_LADDER = list(TC.PRE_LADDER_OVERRIDE)
        print(f"[v1] 机器段 pregrasp 净空梯 (任务覆写): {PRE_LADDER}")
    _m3 = np.radians(3.0)
    _radL = wl_P[0] - body_p[0]
    _radL[2] = 0.0
    _radL /= max(np.linalg.norm(_radL), 1e-9)
    _ikp = {sd: _mk_ik(sd) for sd in ("left", "right")}
    for _sd in ("left", "right"):
        _ikp[_sd].lower = _ikp[_sd].lower + _m3
        _ikp[_sd].upper = _ikp[_sd].upper - _m3
    pre_alts = {"left": [], "right": []}
    _q_station = {"left": np.asarray(q_l[0], np.float64), "right": np.asarray(q_r[0], np.float64)}
    for _lr, _ll, _rl in PRE_LADDER:
        for _side, _pp, _qq in (
                ("left", wl_P[0] + _lr * _radL + [0.0, 0.0, _ll], wl_Q[0]),
                ("right", wr_P[0] + [0.0, 0.0, _rl], wr_Q[0])):
            # ★ 2026-09-01 (Unscrew/17): 净空内点先用**站位解**热启, 只在同一 IK 分支里找
            #   —— 多重启取最优误差会跨分支 (实测右臂净空点距站位 151.6°, 缝1 9 行进刀
            #   要甩 17°/行, 肘会扫过桌/瓶). 同分支解不出来 (>2cm) 才退回多重启。
            _sp = _ikp[_side].solve(np.asarray(_pp, float), quat_to_R(np.asarray(_qq, float)),
                                    q0=_q_station[_side], iters=300, w_rot=0.25)
            _dq = float(np.degrees(np.abs(np.asarray(_sp["q"]) - _q_station[_side]).max()))
            if not (_sp["pos_err"] < 0.02 and _sp["rot_err"] < np.radians(12) and _dq < 90.0):
                _sp2 = _ikp[_side].solve_traj(np.asarray(_pp, float)[None],
                                              np.asarray(_qq, float)[None],
                                              w_rot=0.25, n_restart=24)[0]
                print(f"[v1]   ⚠ {_side} 净空点同分支解不出 (err {_sp['pos_err']*100:.1f}cm, 距站位 {_dq:.0f}°), 退回多重启 "
                      f"(距站位 {np.degrees(np.abs(np.asarray(_sp2['q']) - _q_station[_side]).max()):.0f}°)")
                _sp = _sp2
            pre_alts[_side].append(np.asarray(_sp["q"], np.float64))
        print(f"[v1] 机器段 pregrasp 候选 (L径向{_lr * 100:.0f}cm/抬{_ll * 100:.0f}cm, "
              f"R抬{_rl * 100:.0f}cm): 左距锚 "
              f"{np.linalg.norm(_ikp['left'].fk(pre_alts['left'][-1])[0] - (wl_P[0] + _lr * _radL + [0.0, 0.0, _ll])) * 100:.2f}cm "
              f"右距锚 "
              f"{np.linalg.norm(_ikp['right'].fk(pre_alts['right'][-1])[0] - (wr_P[0] + [0.0, 0.0, _rl])) * 100:.2f}cm")
    machine_pre = {sd: pre_alts[sd][0] for sd in ("left", "right")}
    pre_alts = {sd: np.stack(v) for sd, v in pre_alts.items()}

    # 手指行: 右手 = 活的人手流 (拧盖手法, 本批实证活通道, P-HYB 形状指引同源);
    # 左手 = prior 抓形模板 (与镜像腕位姿配套 —— 人手指流描述的是**人的**握法,
    # 锚在死腕点上, 与 prior 握位几何不配; β squeeze 由 env 前馈负责加压)
    # 右指形 = 候选的 grasp 模板 (与左手同法)。人手指流仍存进 human_right_f,
    # 供 P-HYB 的指形指引奖使用 —— 但**参考**必须是这只手能形成握的那组角。
    f_r_human = smooth(np.asarray(qr["finger_qpos"], float)[w0:w1 + 1], 2.0)
    from rl_rebuild.correction.ref_builders.replay_grasp import (
        GENERIC_JOINT_ORDER)
    gfin = np.asarray(zp["grasp"], np.float64)[7:29]
    perm = [GENERIC_JOINT_ORDER.index(n) for n in fin_names]
    f_l = np.tile(gfin[perm], (N, 1))
    # ★ Unscrew/17 (2026-09-01 09:00): 左站位额外捏合量 LEFT_CURL_DEG (与右手 CAP_PINCH_DEG 同性质)。
    #   Power_Sphere 原生先验在 Isaac 里五垫离瓶面 1~2cm (探针实测 4.2~5.4cm, 贴面 ~4.3),
    #   而它的 squeeze 层只有 0~4° (βL 放大无用), 径向平移对环抱抓法也无效 —— 缺的是手指弯曲量。
    _lcurl = float(getattr(TC, "LEFT_CURL_DEG", 0.0))
    if _lcurl:
        _cl = np.zeros(22)
        for _i, _n in enumerate(fin_names):
            if _n.endswith(("MCP_FE", "PIP", "IP")):
                _cl[_i] = 1.0
            elif _n.endswith("DIP"):
                _cl[_i] = 0.5
        f_l = f_l + np.radians(_lcurl) * _cl
        print(f"[v1] 左站位额外捏合 LEFT_CURL_DEG={_lcurl}° (MCP_FE/PIP/IP 满额, DIP 半额)")
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
    f_r = np.tile(_cap_grasp_fin, (N, 1))
    # 右手 squeeze 增量随母带走 (env 的 βR 前馈读它): 候选自己的 squeeze-grasp,
    # 与左手对称。原来 env 里写死 beta_r*zeros(22), βR 提上去也不生效。
    sq_delta_r = (np.asarray(_capz["squeeze"], np.float64)[7:29][perm]
                  - np.asarray(_capz["grasp"], np.float64)[7:29][perm])
    _cap_sq = np.asarray(_capz["squeeze"], np.float64)[7:29][perm]
    print(f"[v1] 右指形 = 候选 grasp 模板 (中位 "
          f"{np.degrees(np.median(f_r[0])):.1f}°); squeeze 增量中位 "
          f"{np.degrees(np.median(np.abs(_cap_sq - f_r[0]))):.1f}° (βR={TC.BETA_R})")

    # ---- 机器段 (占位: 关节 smoothstep; 正式训练前换规划轨迹, 见 docstring) ----
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

    # ★ U11 (2026-09-02 用户看活样机裁定): 右臂"在家" —— 机器段/缝1 保持站姿,
    #   交互行 0..k_contact 平滑飞向接触行抓姿 (瓶被左手放倒转过来的那段)。
    #   训练时钟 G2 前不走 ⟹ 右手起飞发生在左手认证抓稳之后, 正合 U6 时序;
    #   顺带消掉右臂净空梯/站位悬停 (§9 三发里右腕悬停离盖 21.6cm 的画面)。
    RIGHT_HOME = (os.environ.get("UNSCREW_RIGHT_HOME") or "1").strip() not in ("0",)
    if RIGHT_HOME:
        _kc = int(np.clip(_k_contact, 6, N - 1))
        _s9 = np.linspace(0, 1, _kc, endpoint=False)
        _s9 = _s9 * _s9 * (3 - 2 * _s9)
        q_r[:_kc] = st_r[None] * (1 - _s9)[:, None] + q_r[_kc][None] * _s9[:, None]
        machine_pre["right"] = st_r.copy()
        pre_alts["right"] = [st_r.copy() for _ in pre_alts["right"]]
        _jmp9 = float(np.degrees(np.abs(np.diff(q_r[:_kc + 1], axis=0)).max()))
        print(f"[v1] ★右臂在家 (U11): 机器段/缝1=站姿; 交互行 0~{_kc - 1} 平滑飞向"
              f"接触行 {_k_contact} (峰值行跳 {_jmp9:.1f}°/行)")

    def ramp(a, b, K):
        s = np.linspace(0, 1, K, endpoint=False)
        s = s * s * (3 - 2 * s)
        return a[None] * (1 - s)[:, None] + b[None] * s[:, None]

    def _plan_usable(p):
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
        except Exception as _e:
            print(f"[v1] ⚠ {p} 不可读 ({type(_e).__name__}), 当作无规划")
            return False
        return True

    # ⚠ 摘要必须按**本次要写出的**几何算, 不能读磁盘上的旧 v1: 否则改了站位/
    # pregrasp 之后, 旧规划会被照旧剪进新母带 (两者已经对不上), 而流程要等到
    # 下一次重建才发现 —— 2026-08-31 实测踩到 (左抓锚修正 10cm 那次)。
    _basis_now = {                      # 只喂交互段 (摘要本来就只取交互行)
        "source": np.ones(N, np.int8),
        "station_wr": np.r_[wr_P[0], wr_Q[0]],
        "station_wl": np.r_[wl_P[0], wl_Q[0]],
        "machine_pre_q_r": machine_pre["right"],
        "machine_pre_q_l": machine_pre["left"],
        "obj_pos_0": body_p, "obj_pos_1": cap_p,
    }
    have_approach = _plan_usable(TC.APPROACH_NPZ)
    have_retreat = _plan_usable(TC.RETREAT_NPZ)
    if os.environ.get("UNSCREW_NO_PLAN") == "1":     # 沙盒试验 (站位微调探针): 不剪规划行, 占位机器段
        have_approach = have_retreat = False
        print("[v1] ⚠ UNSCREW_NO_PLAN=1: 机器段占位 (只供站位几何探针, 不可训练)")
    have_machine_plan = have_approach or have_retreat
    planning_basis = (TC.reference_planning_digest(_basis_now)
                      if have_machine_plan else None)
    rest_md5 = TC.file_md5(args.rest_json)

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
    else:
        print("[v1] ⚠ Approach = smoothstep 占位 (无碰撞背书; 先跑 "
              "A_Design/L1_Data/Motion_Planning/plan_machine_segs.py)")
        app_r = ramp(st_r, q_r[0], APP_ROWS)
        app_l = ramp(st_l, q_l[0], APP_ROWS)
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
    _at_station = (float(np.abs(app_r[-1] - q_r[0]).max()) < np.radians(2.0)
                   and float(np.abs(app_l[-1] - q_l[0]).max()) < np.radians(2.0))
    print(f"[v1] 机器段终点 {'= 操作位置 (缝1 只合手指)' if _at_station else '= 净空点 (缝1 需进刀)'}"
          f" | 距站位 R{np.degrees(np.abs(app_r[-1] - q_r[0]).max()):.1f}° "
          f"L{np.degrees(np.abs(app_l[-1] - q_l[0]).max()):.1f}°")
    _OPEN_K, _OPEN_CAP = 2.0, np.radians(25.0)
    _sqz_l = np.asarray(zp["squeeze"], np.float64)[7:29][perm]   # 左手合拢方向
    for _side, _qa, _qb, _ff in (("left", app_l[-1], q_l[0], f_l[0]),
                                 ("right", app_r[-1], q_r[0], f_r[0])):
        if _side == "left":     # 有 squeeze prior: 沿合拢方向反向外推
            _open = _ff - np.clip(_OPEN_K * (_sqz_l - _ff), -_OPEN_CAP, _OPEN_CAP)
        else:                   # 右手无 squeeze prior: 朝站姿(张手)方向退 35%
            _open = _ff + 0.35 * (sf_r - _ff)
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
        if _side == "left":
            s1_l, s1_fl = _arm, _fin
        else:
            s1_r, s1_fr = _arm, _fin
    # ★ U12 (2026-09-02 用户裁定"左手不错, 剪进去"): 左臂接近段整体替换为
    #   build_left_approach 产物 (cuRobo 满障碍 + Dexonomy 六级梯 PCHIP, 合拢在成形段内
    #   缓慢完成 —— 实测自由瓶只倾 1.6°, 旧缝1 10 行快合拢会推到 28.6°, §10.5/10.6)。
    #   缝1 从"进刀+合拢"降级为"静持 + squeeze 渐入窗" (env 的 βL 前馈仍按 APP..IA0 斜坡)。
    _lap = os.environ.get("UNSCREW_LEFT_APPROACH_NPZ") or ""
    if _lap:
        _zl = np.load(_lap, allow_pickle=True)
        _fnl = [str(n) for n in _zl["fin_names"]]
        assert _fnl == fin_names, "LeftApproach fin_names 与母带列序不一致"
        app_l = np.asarray(_zl["left_q"], np.float64)
        app_fl = np.asarray(_zl["left_f"], np.float64)
        app_r = np.asarray(_zl["right_q"], np.float64)
        app_fr = np.asarray(_zl["right_f"], np.float64)
        assert np.degrees(np.abs(app_l[-1] - q_l[0]).max()) < 2.0, "LeftApproach 末行 ≠ 左站位"
        s1_l = np.tile(np.asarray(q_l[0], np.float64), (SEAM1, 1))
        s1_fl = np.tile(np.asarray(f_l[0], np.float64), (SEAM1, 1))
        s1_r = np.tile(np.asarray(app_r[-1], np.float64), (SEAM1, 1))   # 右臂在家静持
        s1_fr = ramp(np.asarray(app_fr[-1], np.float64), np.asarray(f_r[0], np.float64), SEAM1)
        print(f"[v1] ★左接近段替换 (U12): {os.path.basename(_lap)} {len(app_l)} 行 "
              f"(cuRobo+六级梯, 合拢在段内; 缝1={SEAM1} 行静持+squeeze 窗)")
    # Retreat: cuRobo cspace 规划产物优先 (物体已在终位, 躲避着回站姿)
    if have_retreat:
        ret_r, ret_l = _load_plan(TC.RETREAT_NPZ, "retreat")
        print(f"[v1] Retreat = cuRobo 规划 {TC.RETREAT_NPZ} ({len(ret_r)} 行)")
        assert np.abs(ret_r[-1] - st_r).max() < 0.06 and             np.abs(ret_l[-1] - st_l).max() < 0.06,             "Retreat 末行须回到站姿 (progress G4 拿它当 InitialPose)"
        ret_r = np.concatenate([ret_r, st_r[None]])   # 末行=精确站姿 (判据口径)
        ret_l = np.concatenate([ret_l, st_l[None]])
    else:
        ret_r = np.concatenate([np.tile(st_r, (RET_ROWS - 1, 1)), st_r[None]])
        ret_l = np.tile(st_l, (RET_ROWS, 1))
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
        v[n_pre:n_pre + N] = np.asarray(arr, float)[w0:w1 + 1]
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
    frame_of_row[n_pre:n_pre + N] = np.arange(w0, w1 + 1)

    meta = {
        "task": "unscrew", "clip": TC.CLIP_ID, "take": take,
        "gen": "make_reference_v1_20260829",
        "windows": {"w0": w0, "w1": w1, "sep_src": sep_src, "k_sep": k_sep},
        "hold_yaw_deg": hold_yaw,
        "bottle_roll_gauge_deg": hold_roll_deg,
        "right_wait": {"k_contact": int(_k_contact), "back_m": _PRE_R_WAIT_M,
                       "native_priors": bool(getattr(TC, "NATIVE_PRIORS", False))},
        "gauge": {"right_grasp_roll_deg": float(np.degrees(yaw_g)),
                  "right_roll_drift_deg": float(np.degrees(roll[-1] - yaw_g)),
                  "bottle_spin_frozen_from_row": int(k_sep),
                  "cap_rigid_ride_from_row": int(k_sep)},
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
        "planning_basis_digest": planning_basis,
        "rest_md5": rest_md5,
        "notes": "交互1行=1重建帧@15fps, "
                 "env 20Hz 播放 1.33x 实时 (Pour17 v1 同口径)",
    }
    out = dict(
        # 站位腕靶 (plan_machine_segs 的规划目标) + 交互腕靶全轨 (v2/诊断)
        station_wr=np.r_[wr_P[0], wr_Q[0]], station_wl=np.r_[wl_P[0], wl_Q[0]],
        sq_delta_r=sq_delta_r, cap_grasp_src=np.array(os.path.basename(_cf)),
        machine_pre_q_r=machine_pre["right"],
        machine_pre_q_l=machine_pre["left"],
        machine_pre_alts_r=pre_alts["right"],
        machine_pre_alts_l=pre_alts["left"],
        wrist_tgt_r=np.concatenate([wr_P, wr_Q], axis=1),
        wrist_tgt_l=np.concatenate([wl_P, wl_Q], axis=1),
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
