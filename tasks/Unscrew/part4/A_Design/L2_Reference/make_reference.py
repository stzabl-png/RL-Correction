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
CAP_CLEARANCE, HAND_DROP = 0.015, 0.203     # U35b: 盖顶净空 + 腕-指尖垂距(实测)
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
                    help="probe_rest.py 产物 (env 实测静置位/anchor_T/站姿); 缺省走 URDF 推导")
    ap.add_argument("--out", default=TC.REF_V1)
    args = ap.parse_args()
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
    hold_yaw = float(os.environ.get("UNSCREW_HOLD_YAW", "0"))
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
    else:
        import trimesh
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

    # ---- 腕参考 (交互窗) ----
    # ⚠ 死通道零依赖 (2026-08-30): ref_qpos 的 wrist_pos 是上游静态填充的
    # 死数据 (clip32 腕 <0.6cm/盖 66cm), 本构建器只消费**活通道**:
    # 右腕四元数流 / 右手指流 / 物体轨迹+conf / 人手逐帧置信度。
    qr = np.load(os.path.join(take, "ref_qpos_right.npz"), allow_pickle=True)
    fin_names = [str(n) for n in qr["joint_names"]]
    assert all(n.startswith("right_") for n in fin_names), fin_names[:3]

    # 右腕: U35b 对握锚 = 盖心 + reach·螺轴; U35 姿态 = 活流 + hand_z→-螺轴
    reach = 2 * TC.CAP_HALF_H + CAP_CLEARANCE + HAND_DROP
    wr_P = cap_p + reach * axis
    wr_P[:, 2] = np.maximum(wr_P[:, 2], TABLE_Z + 0.02)
    wr_Q = smooth_quats(np.asarray(qr["wrist_quat_wxyz"], float)[w0:w1 + 1])
    ez = np.tile([0.0, 0.0, 1.0], (N, 1))
    hz = qrot(wr_Q, ez)
    tgt = -axis / np.linalg.norm(axis, axis=1, keepdims=True)
    crx = np.cross(hz, tgt)
    ang = np.arctan2(np.linalg.norm(crx, axis=1), (hz * tgt).sum(1))
    ax_n = crx / np.maximum(np.linalg.norm(crx, axis=1, keepdims=True), 1e-9)
    ang[k_sep:], ax_n[k_sep:] = ang[k_sep], ax_n[k_sep]   # 脱离后冻结修正量
    half = 0.5 * ang
    fixq = np.concatenate([np.cos(half)[:, None], np.sin(half)[:, None] * ax_n], 1)
    wr_Q = qmul(fixq, wr_Q)
    wr_Q /= np.linalg.norm(wr_Q, axis=1, keepdims=True)

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
    gp_m = np.array([gp[0], -gp[1], gp[2]])
    gq_m = np.array([gq[0], -gq[1], gq[2], -gq[3]])
    gq_m /= np.linalg.norm(gq_m)

    def _left_track(yaw):
        qy = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        pl = (quat_to_R(qy) @ gp_m)
        qlg = qmul(qy[None], gq_m[None])[0]
        P = body_p + qrot(body_q, np.tile(pl, (N, 1)))
        Q = qmul(body_q, np.tile(qlg, (N, 1)))
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        P[:, 2] = np.maximum(P[:, 2], TABLE_Z + 0.02)
        return P, Q

    def _mk_ik(side):
        if rest and rest.get(f"anchor_T_{side}"):
            return ArmIK(side, anchor_link="arm_center",
                         anchor_T=np.asarray(rest[f"anchor_T_{side}"], float))
        return ArmIK(side, torso_deg={"torso_j1": 40.5196, "torso_j2": 73.6595,
                                      "torso_j3": 0.3896})

    ik_l = _mk_ik("left")
    best = None
    for ydeg in range(0, 360, 30):
        P, Q = _left_track(np.radians(ydeg))
        sols = ik_l.solve_traj(P[::4], Q[::4], w_rot=0.25, n_restart=2)
        pe = np.array([sv["pos_err"] for sv in sols])
        score = float((pe < 0.02).mean())
        med = float(np.median(pe))
        if best is None or (score, -med) > (best[1], -best[2]):
            best = (ydeg, score, med)
    yaw_l = np.radians(best[0])
    print(f"[v1] 左抓方位扫描: yaw*={best[0]}° 可达率 {best[1] * 100:.0f}% "
          f"中位 {best[2] * 100:.2f}cm (prior=Screw27_body 镜像)")
    wl_P, wl_Q = _left_track(yaw_l)

    # ---- ArmIK ----
    def solve_side(side, P, Q):
        if rest and rest.get(f"anchor_T_{side}"):
            ik = ArmIK(side, anchor_link="arm_center",
                       anchor_T=np.asarray(rest[f"anchor_T_{side}"], float))
        else:
            # ⚠ URDF 推导锚: 必须传新场景站姿 (kinematics 默认 45/90/0 是旧站姿,
            # 肩位差 ~10cm); 且仿真躯干会塌 (torso_j2 实测掉 ~6°) —— 有 probe
            # 实测 anchor_T 永远优先, 这条路只是让离线冒烟能跑起来.
            ik = ArmIK(side, torso_deg={"torso_j1": 40.5196,
                                        "torso_j2": 73.6595,
                                        "torso_j3": 0.3896})
        sols = ik.solve_traj(np.asarray(P, float), np.asarray(Q, float),
                             w_rot=0.25)
        q = np.stack([s["q"] for s in sols])
        pe = np.array([s["pos_err"] for s in sols])
        good = np.isfinite(q).all(1) & (pe < 0.02)
        bad = np.flatnonzero(~good)
        if len(bad):
            gd = np.flatnonzero(good)
            assert len(gd), f"{side} IK 全程不达标"
            q[bad] = q[gd[np.abs(gd[None] - bad[:, None]).argmin(1)]]
        print(f"[v1] {side} IK: 位置<2cm {good.mean() * 100:.0f}% | "
              f"中位 {np.median(pe[good]) * 100:.2f}cm | 顶替 {len(bad)} 行")
        return q, float(good.mean())

    q_r, ok_r = solve_side("right", wr_P, wr_Q)
    q_l, ok_l = solve_side("left", wl_P, wl_Q)
    q_r = smooth(q_r, 1.5)
    q_l = smooth(q_l, 1.5)

    # 手指行: 右手 = 活的人手流 (拧盖手法, 本批实证活通道, P-HYB 形状指引同源);
    # 左手 = prior 抓形模板 (与镜像腕位姿配套 —— 人手指流描述的是**人的**握法,
    # 锚在死腕点上, 与 prior 握位几何不配; β squeeze 由 env 前馈负责加压)
    f_r = smooth(np.asarray(qr["finger_qpos"], float)[w0:w1 + 1], 2.0)
    from rl_rebuild.correction.ref_builders.replay_grasp import (
        GENERIC_JOINT_ORDER)
    gfin = np.asarray(zp["grasp"], np.float64)[7:29]
    perm = [GENERIC_JOINT_ORDER.index(n) for n in fin_names]
    f_l = np.tile(gfin[perm], (N, 1))

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

    def ramp(a, b, K):
        s = np.linspace(0, 1, K, endpoint=False)
        s = s * s * (3 - 2 * s)
        return a[None] * (1 - s)[:, None] + b[None] * s[:, None]

    def _load_plan(npz_path):
        """cuRobo worker 产物 -> (右臂行, 左臂行)。列按 joint_names 名取。"""
        pz = np.load(npz_path, allow_pickle=True)
        assert bool(pz["ok"]), f"{npz_path}: 规划失败产物 (ok=False)"
        names = [str(n) for n in pz["joint_names"]]
        tr = np.asarray(pz["traj"], np.float64)
        ir = [names.index(f"R_arm_j{i}") for i in range(1, 8)]
        il = [names.index(f"L_arm_j{i}") for i in range(1, 8)]
        return tr[:, ir], tr[:, il]

    # Approach: cuRobo 规划产物优先 (碰撞检查过的可行解; plan_machine_segs.py 产出)
    if os.path.isfile(TC.APPROACH_NPZ):
        app_r, app_l = _load_plan(TC.APPROACH_NPZ)
        n_app = len(app_r)
        print(f"[v1] Approach = cuRobo 规划 {TC.APPROACH_NPZ} ({n_app} 行)")
    else:
        print("[v1] ⚠ Approach = smoothstep 占位 (无碰撞背书; 先跑 "
              "A_Design/L1_Data/Motion_Planning/plan_machine_segs.py)")
        app_r = ramp(st_r, q_r[0], APP_ROWS)
        app_l = ramp(st_l, q_l[0], APP_ROWS)
        n_app = APP_ROWS
    # 手指: approach 前 70% 保持站姿张开, 后 30% 合到交互首行 (合拢斜坡)
    k70 = int(n_app * 0.7)
    app_fr = np.concatenate([np.tile(sf_r, (k70, 1)),
                             ramp(sf_r, f_r[0], n_app - k70)])
    app_fl = np.concatenate([np.tile(sf_l, (k70, 1)),
                             ramp(sf_l, f_l[0], n_app - k70)])
    # 缝1 = 焊接斜坡: 规划终帧构型 -> 离线 IK 交互首行 (两套解不同分支时在此桥接)
    s1_r, s1_l = ramp(app_r[-1], q_r[0], SEAM1), ramp(app_l[-1], q_l[0], SEAM1)
    s1_fr, s1_fl = np.tile(f_r[0], (SEAM1, 1)), np.tile(f_l[0], (SEAM1, 1))
    # Retreat: cuRobo cspace 规划产物优先 (物体已在终位, 躲避着回站姿)
    if os.path.isfile(TC.RETREAT_NPZ):
        ret_r, ret_l = _load_plan(TC.RETREAT_NPZ)
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
                     if os.path.isfile(TC.APPROACH_NPZ) else "smoothstep占位"),
        "retreat": ("curobo:" + TC.RETREAT_NPZ
                    if os.path.isfile(TC.RETREAT_NPZ) else "smoothstep占位"),
        "notes": "交互1行=1重建帧@15fps, "
                 "env 20Hz 播放 1.33x 实时 (Pour17 v1 同口径)",
    }
    out = dict(
        # 站位腕靶 (plan_machine_segs 的规划目标) + 交互腕靶全轨 (v2/诊断)
        station_wr=np.r_[wr_P[0], wr_Q[0]], station_wl=np.r_[wl_P[0], wl_Q[0]],
        wrist_tgt_r=np.concatenate([wr_P, wr_Q], axis=1),
        wrist_tgt_l=np.concatenate([wl_P, wl_Q], axis=1),
        right_q=right_q, left_q=left_q, right_f=right_f, left_f=left_f,
        obj_pos_0=bp_all, obj_quat_0=bq_all, obj_pos_1=cp_all, obj_quat_1=cq_all,
        source=source, frame_of_row=frame_of_row, seg_lens=seg_lens,
        seg_names=np.array(["approach", "seam1", "interact", "seam2", "retreat"]),
        fin_names=np.array(fin_names), meta=np.array(json.dumps(meta)),
        **cols)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    md5 = hashlib.md5(open(args.out, "rb").read()).hexdigest()[:8]
    print(f"[v1] 已写 {args.out} md5={md5} 全链 {Tn} 行 "
          f"(app {APP_ROWS}/seam1 {SEAM1}/ia {N}/seam2 {SEAM2}/ret {RET_ROWS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
