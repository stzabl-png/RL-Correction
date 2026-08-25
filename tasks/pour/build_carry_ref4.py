"""CARRY4 构建器: 手当里程计 + 物体高置信帧当路标 (整段融合, 2026-08-24 用户裁定).

与 carry3 (build_carry_ref.py) 的唯一区别在**腕参考**:
  carry3: 腕 = 物体轨迹 ∘ T_rel*⁻¹ 逐帧反推 —— 低置信段物体是虚构 ⟹ 腕也是虚构;
  carry4: 腕 = 在物体高置信锚点处取 carry3 同款反推 (保住机器人抓姿契约),
          锚点之间播**人手腕的相对运动弧** (ARKit, conf=1.0, 覆盖全程含沙漠段),
          锚点闭合误差沿弧平摊 (实测 p50≈12°, 摊到锚距≥10帧 ⟹ <1.2°/帧).
物体目标行 (obj_pos/obj_quat) 与 carry3 **完全一致** —— 任务契约不变, 只换手段.

依据 (2026-08-24 实测): pour17 手部 retarget valid 142/142, 窗内逐帧 ≤9.1°/1.0cm;
手腕窗内峰值转角 117.8° ≈ demo 倾角 119° (RL 之前要自己发明的整臂重构就在这里).

用法:
    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pour.build_carry_ref4 \
        --out tasks/pour/carry4_pour17.npz
(无 Isaac, 纯 numpy+IK, 秒级)
"""
import argparse
import os

import numpy as np

from rl_rebuild.correction.kinematics import ArmIK

p = argparse.ArgumentParser()
p.add_argument("--rts_dir", default="/home/lyh/Project/Reconstruct_and_Retarget/"
                                    "results/pour_17_better/poseqa")
p.add_argument("--hand_dir", default="/home/lyh/Project/Reconstruct_and_Retarget/"
                                     "results/pour_17_better/retarget")
p.add_argument("--out", default="tasks/pour/carry4_pour17.npz")
p.add_argument("--grip_frames", type=int, default=40)
p.add_argument("--stride", type=int, default=1)
p.add_argument("--rot_cap_deg", type=float, default=8.0)
p.add_argument("--ms_gap", type=int, default=15)
p.add_argument("--conf", type=float, default=60.0)
p.add_argument("--mouth_assist", type=float, default=0.0,
               help=">0 = 参考辅助 (v9S, 2026-08-24 用户裁定): 倾角≥60°的窗内行把隐含"
                    "瓶口水平收向杯口, 目标口距从 60°/12cm 线性收至 93°+/该值(m)。"
                    "还的是重定向丢的 7cm demo 几何 (demo 实测 4cm@109°), 非虚构外力")
p.add_argument("--end_frame", type=int, default=124,
               help="截断帧: 物体结束移动处 (只学交互段, 撤退不进参考; -1=不截)。"
                    "pour17 实测: 瓶真实运动止于~120, 帧128-136=右手松开撤退, "
                    "杯尾段 conf=0 纯虚构")
p.add_argument("--table_z", type=float, default=0.85)
p.add_argument("--hand_clear", type=float, default=0.02,
               help="桌面净空钳制: 手最低点(抓姿指型)距桌面最小余量 (m); <=0 关闭")
p.add_argument("--prior_r", default="tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")
p.add_argument("--prior_l", default="tasks/pregrasp/priors/Pour17_cup_thumbfix.npz")
p.add_argument("--anchor_gap", type=int, default=10,
               help="路标锚最小间隔 (帧): 锚太密会把手/物分歧变成逐帧抖动")
args = p.parse_args()


def q2m(q):
    w, x, y, z = np.asarray(q, np.float64)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def m2q(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-6:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        q = np.zeros(4); q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4.0; q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
        return q / np.linalg.norm(q)
    return np.array([w, (R[2, 1] - R[1, 2])/(4*w), (R[0, 2] - R[2, 0])/(4*w),
                     (R[1, 0] - R[0, 1])/(4*w)])


def slerp(q0, q1, t):
    q0 = q0/np.linalg.norm(q0); q1 = q1/np.linalg.norm(q1)
    if np.dot(q0, q1) < 0: q1 = -q1
    d = np.clip(np.dot(q0, q1), -1, 1); th = np.arccos(d)
    if th < 1e-6: return q0
    return (np.sin((1-t)*th)*q0 + np.sin(t*th)*q1)/np.sin(th)


def qang(a, b):
    return np.degrees(2*np.arccos(min(1.0, abs(float(np.dot(
        a/np.linalg.norm(a), b/np.linalg.norm(b)))))))


def R_log(R):
    """SO(3) 对数映射 -> 轴角向量."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-8:
        return np.zeros(3)
    v = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
    return th / (2*np.sin(th)) * v


def R_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-10:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)


# ---------------- 场景静置摆位与抓姿 (与 carry3 同源: scene_snap) ----------------
snap = np.load("tasks/pour/scene_snap.npz", allow_pickle=True)
OBJ_P = np.asarray(snap["obj_pos"], np.float64)
OBJ_Q = np.asarray(snap["obj_quat"], np.float64)
GW_P = np.asarray(snap["grasp_pos"], np.float64)
GW_Q = np.asarray(snap["grasp_quat"], np.float64)
QARM0 = np.asarray(snap["q_grasp_arm"], np.float64)
ANCHOR_T = np.asarray(snap["anchor_T"], np.float64)

# ---------------- 重建物体轨迹 + 人手腕轨迹 (同一 recon_world) ----------------
RTS_IDX = {0: 1, 1: 0}          # 本文件序 [瓶, 杯] -> rts object 下标
HAND = {0: "right", 1: "left"}
T_rec, HI, HW = {}, {}, {}
for s in (0, 1):
    z = np.load(f"{args.rts_dir}/rts_pour_17_object_{RTS_IDX[s]}.npz",
                allow_pickle=True)
    T_rec[s] = np.asarray(z["object_ob_in_world_smooth"], np.float64)[::args.stride]
    HI[s] = ((np.asarray(z["conf_pos"]) >= args.conf)
             & (np.asarray(z["conf_rot"]) >= args.conf)
             & np.asarray(z["measurement_used"]).astype(bool)
             & (np.asarray(z["innov_pos_nis"]) < 5.0)
             & (np.asarray(z["innov_rot_nis"]) < 5.0))[::args.stride]
    h = np.load(f"{args.hand_dir}/ref_qpos_{HAND[s]}.npz", allow_pickle=True)
    hp = np.asarray(h["wrist_pos"], np.float64)[::args.stride]
    hq = np.asarray(h["wrist_quat_wxyz"], np.float64)[::args.stride]
    hv = np.asarray(h["valid"]).astype(bool)[::args.stride]
    assert str(h["frame_of_reference"]) == "recon_world", \
        f"手轨迹坐标系 {h['frame_of_reference']} != recon_world"
    HW[s] = dict(p=hp, R=[q2m(q) for q in hq], valid=hv)
N = len(T_rec[0])
assert len(HW[0]["p"]) == N, f"手 {len(HW[0]['p'])} 帧 != 物体 {N} 帧"
if 0 < args.end_frame < N:
    # 只学交互段: 物体开始动 -> 结束动; 撤退(松手回位)不进参考 (2026-08-24 用户裁定)
    for s in (0, 1):
        T_rec[s] = T_rec[s][:args.end_frame]
        HI[s] = HI[s][:args.end_frame]
        HW[s] = dict(p=HW[s]["p"][:args.end_frame], R=HW[s]["R"][:args.end_frame],
                     valid=HW[s]["valid"][:args.end_frame])
    N = args.end_frame
    assert N > 114, "截断点必须保住沙漠出口路标锚 (帧104-114)"
    print(f"[carry4] 截断 @帧{N}: 撤退段已剔除 (帧128-136右手撤退, 杯尾段conf=0虚构)")

# ---------------- 全局水平转角 (与 carry3 完全一致) ----------------
_v_rec = (T_rec[0][0][:2, 3] - T_rec[1][0][:2, 3])
_v_sim = (OBJ_P[0] - OBJ_P[1])[:2]
_yaw = float(np.arctan2(_v_sim[1], _v_sim[0]) - np.arctan2(_v_rec[1], _v_rec[0]))
Rz = np.array([[np.cos(_yaw), -np.sin(_yaw), 0], [np.sin(_yaw), np.cos(_yaw), 0],
               [0, 0, 1.0]])
print(f"[carry4] 全局转角 {np.degrees(_yaw):.1f}° | 帧数 {N} | 热身 {args.grip_frames}")

out = {}
for s, nm, side in ((1, "杯", "left"), (0, "瓶", "right")):
    R_rest = q2m(OBJ_Q[s]); p_rest = OBJ_P[s]
    R0inv = T_rec[s][0][:3, :3].T
    p0 = T_rec[s][0][:3, 3]
    Rw = q2m(GW_Q[s]); pw = GW_P[s]
    R_rel = Rw.T @ R_rest
    p_rel = Rw.T @ (p_rest - pw)

    # ---- ① 物体目标行 (与 carry3 逐帧一致) + 物体锚腕 W_obj ----
    obj_pos, obj_quat = [], []
    W_obj = []                       # (R, p) 逐帧
    for t in range(N):
        dR = Rz @ (T_rec[s][t][:3, :3] @ R0inv) @ Rz.T
        dp = Rz @ (T_rec[s][t][:3, 3] - p0)
        Ro = dR @ R_rest
        po = p_rest + dp
        obj_pos.append(po); obj_quat.append(m2q(Ro))
        Rwt = Ro @ R_rel.T
        W_obj.append((Rwt, po - Rwt @ p_rel))

    # ---- ② 路标锚集: 帧0 强制 + 高置信帧按最小间隔抽稀 ----
    anchors = [0]
    for t in np.flatnonzero(HI[s]):
        if t - anchors[-1] >= args.anchor_gap:
            anchors.append(int(t))
    # ---- ③ 手里程计 + 锚点闭合误差平摊 ----
    Hp, HR = HW[s]["p"], HW[s]["R"]
    W = [None] * N
    W[0] = W_obj[0]
    clo_deg, clo_cm = [], []

    def _odo_step(Wprev, t):
        """t-1 -> t 的人手相对 SE3 (yaw 共轭进场景) 作用在腕上."""
        dRh = Rz @ (HR[t] @ HR[t-1].T) @ Rz.T
        dph = Rz @ (Hp[t] - Hp[t-1])
        Rp, pp = Wprev
        # 姿态跟人手转, 位置跟人手移 —— 链式积分即"人手相对弧作用于入段腕位":
        #   R(t) = Rz·Rh(t)·Rh(m)ᵀ·Rzᵀ · R(m),  p(t) = p(m) + Rz·(Hp(t)-Hp(m))
        return (dRh @ Rp, pp + dph)

    seg_ends = anchors[1:] + [N - 1]
    seg_starts = anchors[:len(seg_ends)]
    for m, mp in zip(seg_starts, seg_ends):
        if mp <= m:
            continue
        # 纯里程计链
        Wo = {m: W[m]}
        for t in range(m + 1, mp + 1):
            Wo[t] = _odo_step(Wo[t-1], t)
        if mp in anchors[1:]:
            # 闭合误差 (世界系左误差), 沿段平摊
            Ro_e, po_e = Wo[mp]
            Ra, pa = W_obj[mp]
            eR = R_log(Ra @ Ro_e.T)
            ep = pa - po_e
            clo_deg.append(np.degrees(np.linalg.norm(eR)))
            clo_cm.append(np.linalg.norm(ep) * 100)
            _long = (mp - m) > 25            # 沙漠跨段: 前60%纯人手弧, 后40%收误差
            for t in range(m + 1, mp + 1):
                u = (t - m) / (mp - m)
                if _long:
                    u = 0.0 if u < 0.6 else (u - 0.6) / 0.4
                sm = u * u * (3 - 2 * u)          # smoothstep
                Rc = R_exp(sm * eR)
                Rt, pt = Wo[t]
                W[t] = (Rc @ Rt, pt + sm * ep)
        else:
            for t in range(m + 1, mp + 1):
                W[t] = Wo[t]                       # 尾段: 纯里程计

    # ---- ④ IK ----
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=ANCHOR_T)
    # ---- ④b 桌面净空钳制 (2026-08-24 用户裁定): 手最低点(抓姿指型) ≥ 桌面+余量 ----
    # 穿桌是"人手弧搬进仿真"的搬运误差, 不是人的动作 (人没穿过他家桌子);
    # 桌面 = 高度百分百已知、永远在线的 z 维路标, 与手/物锚同属融合的一员。
    # 实测: carry3 尾段就穿 -2.5cm (37行, v6 不访问所以没炸), carry4 -4.5cm。
    if args.hand_clear > 0:
        # 指型 = URDF 零位(伸直) —— 即"探得最低"的最坏情况, 净空口径保守安全
        # (squeeze 模板是 29 维 GENERIC 序, 离线换序不值得; 伸直已覆盖弯指包络)
        _fmap = {}
        _dn = {ik.ee_link}; _ch = True
        while _ch:
            _ch = False
            for _n, _j in ik.u.joints.items():
                if _j["parent"] in _dn and _j["child"] not in _dn:
                    _dn.add(_j["child"]); _ch = True
        _qd = ik._qdict(QARM0[s]); _qd.update(_fmap)
        _Tw0 = ik.u.link_pose(ik.ee_link, _qd, ik.base_T, ik.anchor_link)
        _offs = np.stack([(np.linalg.inv(_Tw0)
                           @ ik.u.link_pose(_L, _qd, ik.base_T, ik.anchor_link))[:3, 3]
                          for _L in _dn])
        _minz = np.array([(W[t][0] @ _offs.T).T[:, 2].min() + W[t][1][2]
                          for t in range(N)])
        _lift = np.maximum(0.0, args.table_z + args.hand_clear - _minz)
        _lift_s = np.convolve(np.pad(_lift, 4, mode="edge"),
                              np.ones(9) / 9.0, mode="valid")
        _lift_s = np.maximum(_lift_s, _lift)   # 平滑只做缓坡, 不许低于必要抬升
        for t in range(N):
            if _lift_s[t] > 0:
                W[t] = (W[t][0], W[t][1] + np.array([0.0, 0.0, _lift_s[t]]))
        print(f"[carry4] {nm} 桌面净空钳制: 触发 {int((_lift > 0).sum())}/{N} 行 | "
              f"最大抬升 {_lift.max()*100:.1f}cm | 钳后余量 ≥{args.hand_clear*100:.0f}cm")
    if s == 0 and args.mouth_assist > 0:
        # ---- ④b2 参考辅助: 高倾角行隐含瓶口水平收向杯口 (杯契约已在 out 里) ----
        def _topc(mesh, Rr):
            V = np.asarray([[float(x) for x in l.split()[1:4]]
                            for l in open(mesh) if l.startswith("v ")])
            upv = Rr.T @ np.array([0.0, 0.0, 1.0])
            h = V @ upv
            return V[h > h.max() - 0.01].mean(0)
        _mou = _topc("datasets/pour17/objects/object_1/object_mesh_scaled_final.obj",
                     q2m(OBJ_Q[0]))
        _ctp = _topc("datasets/pour17/objects/object_0/object_mesh_scaled_final.obj",
                     q2m(OBJ_Q[1]))
        _upb = q2m(OBJ_Q[0]).T @ np.array([0.0, 0.0, 1.0])
        CP_ = np.asarray(out["obj_pos_1"], np.float64)
        CQ_ = np.asarray(out["obj_quat_1"], np.float64)
        _G = args.grip_frames
        _shift = np.zeros((N, 2))
        for t in range(44, min(104, N)):
            Rwt, pwt = W[t]
            Ro = Rwt @ R_rel
            po = pwt + Rwt @ p_rel
            _ti = np.degrees(np.arccos(np.clip((Ro @ _upb)[2], -1, 1)))
            if _ti < 60.0:
                continue
            mw = po + Ro @ _mou
            cw = CP_[t + _G] + q2m(CQ_[t + _G]) @ _ctp
            dxy = (cw - mw)[:2]
            d = np.linalg.norm(dxy)
            d_tgt = 0.12 + (args.mouth_assist - 0.12) * min(max((_ti - 60) / 33.0, 0), 1)
            if d > d_tgt:
                _shift[t] = (d - d_tgt) * dxy / d
        # 平滑 (5行滑窗) 后施加
        k5 = np.ones(5) / 5.0
        for c in (0, 1):
            _shift[:, c] = np.convolve(np.pad(_shift[:, c], 2, mode="edge"),
                                       k5, mode="valid")
        _n_as = int((np.linalg.norm(_shift, axis=1) > 1e-4).sum())
        for t in range(N):
            if np.linalg.norm(_shift[t]) > 1e-4:
                Rt, pt = W[t]
                W[t] = (Rt, pt + np.array([_shift[t, 0], _shift[t, 1], 0.0]))
        print(f"[carry4] 瓶 参考辅助: 修正 {_n_as} 行 | 最大水平位移 "
              f"{np.linalg.norm(_shift, axis=1).max()*100:.1f}cm | 目标口距@93°+ = "
              f"{args.mouth_assist*100:.0f}cm")
    wrist_rows, errs, dpos, drot = [], [], [], []
    seed = QARM0[s].copy()
    for t in range(N):
        Rwt, pwt = W[t]
        r = ik.solve(pwt, Rwt, q0=seed, iters=150)
        seed = r["q"]
        wrist_rows.append(seed)
        errs.append(r["pos_err"])
        Ra, pa = W_obj[t]
        dpos.append(np.linalg.norm(pwt - pa) * 100)
        drot.append(np.degrees(np.linalg.norm(R_log(Rwt @ Ra.T))))
    # ---- ④c 窗内物体契约 = 手隐含位姿 (2026-08-24: "物体抢跑"破案) ----
    # 沙漠帧 [44,103] 的重建物体是虚构且**抢跑** (RTS 帧52尖刺提前倾, 峰@68 vs
    # 手真实峰@80): 32/103 行与手隐含倾角差 >20° (时钟门容差) ⟹ v9A 时钟会卡在
    # 行100 附近, 残差被迫重新发明倾角 (v8 冻结前馈老病复发)。
    # 修: 刚握假设下由手弧传播物体位姿 —— 沙漠段物体的最优估计本来就是
    # "可靠的手 + 刚握", 时钟剖面随之与前馈自洽; 锚点处与重建自然衔接。
    for t in range(44, 104):
        Rwt, pwt = W[t]
        Ro = Rwt @ R_rel
        po = pwt + Rwt @ p_rel
        obj_pos[t] = po
        obj_quat[t] = m2q(Ro)
    errs = np.asarray(errs); dpos = np.asarray(dpos); drot = np.asarray(drot)
    print(f"[carry4] {nm}({side}): 锚 {len(anchors)} 个 | 闭合误差 "
          f"p50 {np.median(clo_deg):.1f}°/{np.median(clo_cm):.1f}cm "
          f"max {max(clo_deg):.1f}°/{max(clo_cm):.1f}cm")
    print(f"[carry4] {nm}: IK 位置误差 中位 {np.median(errs)*100:.2f}cm | "
          f"p90 {np.percentile(errs, 90)*100:.2f}cm | max {errs.max()*100:.2f}cm "
          f"@帧{int(errs.argmax())} | >2cm 帧数 {(errs > 0.02).sum()}/{N}")
    for lo, hi, tag in ((0, 44, "可信前段"), (44, 104, "沙漠窗"), (104, N, "尾段")):
        print(f"[carry4] {nm} 腕 vs carry3口径(物体反推) {tag}: "
              f"pos p50 {np.median(dpos[lo:hi]):.1f}cm max {dpos[lo:hi].max():.1f}cm | "
              f"rot p50 {np.median(drot[lo:hi]):.1f}° max {drot[lo:hi].max():.1f}°")

    G = args.grip_frames
    rows = np.concatenate([np.tile(wrist_rows[0], (G, 1)), np.stack(wrist_rows)])
    key = "right_q" if side == "right" else "left_q"
    out[key] = rows.astype(np.float32)
    out[f"obj_pos_{s}"] = np.concatenate(
        [np.tile(obj_pos[0], (G, 1)), np.stack(obj_pos)]).astype(np.float32)
    out[f"obj_quat_{s}"] = np.concatenate(
        [np.tile(obj_quat[0], (G, 1)), np.stack(obj_quat)]).astype(np.float32)
    out[f"ik_err_{s}"] = errs.astype(np.float32)
    out[f"anchors_{s}"] = np.asarray([a + G for a in anchors], np.int64)

# ---------------- 转速限幅重采样 (较 carry3 多把**腕**转步也计入细分依据) ----------------
_cap = float(args.rot_cap_deg)
Q0 = out["obj_quat_0"][args.grip_frames:]; Q1 = out["obj_quat_1"][args.grip_frames:]
QR = out["right_q"][args.grip_frames:]; QL = out["left_q"][args.grip_frames:]
Nn = len(Q0)
sub = np.ones(Nn - 1, dtype=int)
for k in range(Nn - 1):
    st = max(qang(Q0[k], Q0[k+1]), qang(Q1[k], Q1[k+1]),
             float(np.degrees(np.abs(QR[k+1] - QR[k]).max())),
             float(np.degrees(np.abs(QL[k+1] - QL[k]).max())))
    sub[k] = max(1, int(np.ceil(st / _cap)))


def _resample(arr, is_quat):
    body = arr[args.grip_frames:]
    rows = [body[0]]
    for k in range(Nn - 1):
        for j in range(1, sub[k] + 1):
            t = j / sub[k]
            if is_quat:
                rows.append(slerp(np.asarray(body[k], np.float64),
                                  np.asarray(body[k+1], np.float64), t))
            else:
                rows.append((1 - t) * np.asarray(body[k], np.float64)
                            + t * np.asarray(body[k+1], np.float64))
    head = np.tile(rows[0], (args.grip_frames, 1))
    return np.concatenate([head, np.stack(rows)]).astype(np.float32)


orig2new = np.concatenate([[0], np.cumsum(sub)])
for key, isq in (("obj_pos_0", 0), ("obj_pos_1", 0), ("obj_quat_0", 1),
                 ("obj_quat_1", 1), ("right_q", 0), ("left_q", 0)):
    out[key] = _resample(out[key], bool(isq))
for s in (0, 1):
    out[f"anchors_{s}"] = (orig2new[out[f"anchors_{s}"] - args.grip_frames]
                           + args.grip_frames)
T_new = len(out["right_q"]) - args.grip_frames
print(f"[carry4] 限幅重采样: cap {_cap}°/帧 | {Nn} -> {T_new} 帧 "
      f"(插 {T_new - Nn} 帧, 细分最多 {int(sub.max())}x @原帧{int(sub.argmax())})")

# ---------------- 高置信里程碑 (与 carry3 完全一致) ----------------
_hi = None
for si, ri in ((0, RTS_IDX[0]), (1, RTS_IDX[1])):
    zc = np.load(f"{args.rts_dir}/rts_pour_17_object_{ri}.npz", allow_pickle=True)
    h = ((np.asarray(zc["conf_pos"]) >= args.conf)
         & (np.asarray(zc["conf_rot"]) >= args.conf)
         & np.asarray(zc["measurement_used"]).astype(bool)
         & (np.asarray(zc["innov_pos_nis"]) < 5.0)
         & (np.asarray(zc["innov_rot_nis"]) < 5.0))[::args.stride]
    _hi = h if _hi is None else (_hi & h)
trusted_new = orig2new[np.flatnonzero(_hi)]
ms = []
for anchor in range(args.ms_gap, T_new - 5, args.ms_gap):
    if len(trusted_new) == 0:
        break
    cand = trusted_new[np.argmin(np.abs(trusted_new - anchor))]
    if not ms or cand - ms[-1] >= args.ms_gap // 2:
        ms.append(int(cand))
ms = sorted(set(ms))
out["ms_idx"] = np.asarray(ms, np.int64) + args.grip_frames
print(f"[carry4] 高置信里程碑 {len(ms)} 个 @新帧 {ms}")

out["free_lo"] = np.int64(orig2new[44] + args.grip_frames)
out["free_hi"] = np.int64(orig2new[103] + args.grip_frames)
print(f"[carry4] 自由段行号 (recon帧[44,103] 映射): [{int(out['free_lo'])},"
      f"{int(out['free_hi'])}) —— 训练时 pour_free_lo/hi 用这个, 不是 carry3 的 93/140")
out["grip_frames"] = np.int64(args.grip_frames)
out["meta"] = np.array(f"carry4_pour17 手锚融合 yaw={np.degrees(_yaw):.1f} "
                       f"anchor_gap={args.anchor_gap} src={os.path.basename(args.hand_dir)}")
np.savez(args.out, **out)
print(f"[carry4] ✅ 写出 {args.out} | 路径长 {len(out['right_q'])} 行 (含热身)")
