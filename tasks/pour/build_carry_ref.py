"""Pour 携带参考构建器: 重建物体轨迹 → 场景物体目标 → 反推双腕 → IK → 双臂关节路径。

口径 = POUR_DESIGN §5 公约:
  物体目标  T_obj,sim(t) = [Rz·ΔR_rec(t)·Rzᵀ] 作用在场景静置位姿上 (支点=物体自己)
  腕目标    T_wrist(t) = T_obj,sim(t) · T_rel*⁻¹,  T_rel* = T_grasp_wrist⁻¹·T_obj,rest
输出 npz (curobo_ref 同格式: right_q/left_q + 物体目标序列), 供 PourCarryEnv 直接吃。

用法:
    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pour.build_carry_ref \
        --out tasks/pour/carry_pour17.npz
(无 Isaac, 纯 numpy+IK, 秒级)
"""
import argparse
import os

import numpy as np

from rl_rebuild.correction.kinematics import ArmIK, quat_to_R

p = argparse.ArgumentParser()
p.add_argument("--rts_dir", default="/home/lyh/Project/Reconstruct_and_Retarget/"
                                    "results/pour_17_better/poseqa")
p.add_argument("--out", default="tasks/pour/carry_pour17.npz")
p.add_argument("--grip_frames", type=int, default=40,
               help="路径头部重复首行的帧数 (抓稳热身窗, 时钟走这里=原地)")
p.add_argument("--stride", type=int, default=1)
p.add_argument("--rot_cap_deg", type=float, default=8.0,
               help="转速限幅 (度/帧): 超限帧间插 slerp 过渡帧摊平 (数据尖刺 64.8°/帧)")
p.add_argument("--ms_gap", type=int, default=15,
               help="高置信里程碑目标间隔 (重采样后帧数); 在可信约束下最大均匀")
p.add_argument("--conf", type=float, default=60.0)
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


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


# ---------------- 场景静置摆位与抓姿 (与训练 env 同一来源, 不起 Isaac) ----------------
# 物体摆位/抓姿世界位姿的权威 = env 构造; 这里离线复算会有口径漂移风险, 所以改为:
# 从一次 pose_pregrasp --headless 的对账输出取, 或直接读 env 快照。为今晚 MVP,
# 用 dbg 快照文件 (build 前先跑 tasks.pour.snap_scene 生成)。
snap = np.load("tasks/pour/scene_snap.npz", allow_pickle=True)
# scene_snap 键: obj_pos/obj_quat (2,·) 按 [瓶(A/右), 杯(B/左)]; grasp_pos/grasp_quat
# 同序 (真抓姿腕位姿); q_grasp_arm (2,7) 各臂抓姿 IK; anchor_T
OBJ_P = np.asarray(snap["obj_pos"], np.float64)      # (2,3) [瓶, 杯]
OBJ_Q = np.asarray(snap["obj_quat"], np.float64)     # (2,4)
GW_P = np.asarray(snap["grasp_pos"], np.float64)     # (2,3)
GW_Q = np.asarray(snap["grasp_quat"], np.float64)    # (2,4)
QARM0 = np.asarray(snap["q_grasp_arm"], np.float64)  # (2,7)
ANCHOR_T = np.asarray(snap["anchor_T"], np.float64)

# ---------------- 重建轨迹 (rts smooth) ----------------
# 身份公约: obj0=杯(左/B), obj1=瓶(右/A)。本文件序 = [瓶, 杯] = [rts1, rts0]
RTS_IDX = {0: 1, 1: 0}          # 本文件第 s 个 (0=瓶,1=杯) -> rts object 下标
T_rec = {}
for s in (0, 1):
    z = np.load(f"{args.rts_dir}/rts_pour_17_object_{RTS_IDX[s]}.npz", allow_pickle=True)
    T_rec[s] = np.asarray(z["object_ob_in_world_smooth"], np.float64)[::args.stride]
N = len(T_rec[0])

# 全局水平转角: 重建"杯→瓶" 对齐 场景"杯→瓶"
_v_rec = (T_rec[0][0][:2, 3] - T_rec[1][0][:2, 3])   # 杯→瓶 (本文件序: 0=瓶,1=杯)
_v_sim = (OBJ_P[0] - OBJ_P[1])[:2]                   # 杯位→瓶位 (场景)
_yaw = float(np.arctan2(_v_sim[1], _v_sim[0]) - np.arctan2(_v_rec[1], _v_rec[0]))
Rz = np.array([[np.cos(_yaw), -np.sin(_yaw), 0], [np.sin(_yaw), np.cos(_yaw), 0],
               [0, 0, 1.0]])
print(f"[carry] 全局转角 {np.degrees(_yaw):.1f}° | 帧数 {N} | 热身 {args.grip_frames}")

out = {}
for s, nm, side in ((0, "瓶", "right"), (1, "杯", "left")):
    R_rest = q2m(OBJ_Q[s]); p_rest = OBJ_P[s]
    R0inv = T_rec[s][0][:3, :3].T
    p0 = T_rec[s][0][:3, 3]
    # T_rel*: 抓姿腕 -> 物体静置 (规格给定, 不靠运行时快照)
    Rw = q2m(GW_Q[s]); pw = GW_P[s]
    R_rel = Rw.T @ R_rest
    p_rel = Rw.T @ (p_rest - pw)
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=ANCHOR_T)
    obj_pos, obj_quat, wrist_rows, errs = [], [], [], []
    seed = QARM0[s].copy()
    for t in range(N):
        dR = Rz @ (T_rec[s][t][:3, :3] @ R0inv) @ Rz.T
        dp = Rz @ (T_rec[s][t][:3, 3] - p0)
        Ro = dR @ R_rest
        po = p_rest + dp
        obj_pos.append(po); obj_quat.append(m2q(Ro))
        # 腕 = 物体 ∘ T_rel*⁻¹
        Rwt = Ro @ R_rel.T
        pwt = po - Rwt @ p_rel
        r = ik.solve(pwt, Rwt, q0=seed, iters=150)
        seed = r["q"]
        wrist_rows.append(seed)
        errs.append(r["pos_err"])
    errs = np.asarray(errs)
    print(f"[carry] {nm}({side}): IK 位置误差 中位 {np.median(errs)*100:.2f}cm | "
          f"p90 {np.percentile(errs, 90)*100:.2f}cm | max {errs.max()*100:.2f}cm "
          f"@帧{int(errs.argmax())} | >2cm 帧数 {(errs > 0.02).sum()}/{N}")
    G = args.grip_frames
    rows = np.concatenate([np.tile(wrist_rows[0], (G, 1)), np.stack(wrist_rows)])
    key = "right_q" if side == "right" else "left_q"
    out[key] = rows.astype(np.float32)
    out[f"obj_pos_{s}"] = np.concatenate(
        [np.tile(obj_pos[0], (G, 1)), np.stack(obj_pos)]).astype(np.float32)
    out[f"obj_quat_{s}"] = np.concatenate(
        [np.tile(obj_quat[0], (G, 1)), np.stack(obj_quat)]).astype(np.float32)
    out[f"ik_err_{s}"] = errs.astype(np.float32)

# ---------------- 转速限幅重采样 (2026-08-19 CARRY3): 尖刺摊平 ----------------
# 两侧共享时间轴: 每帧细分数 n_k = ceil(双物体最大转步 / cap), 同步插帧
_cap = float(args.rot_cap_deg)
Q0 = out["obj_quat_0"][args.grip_frames:]; Q1 = out["obj_quat_1"][args.grip_frames:]
Nn = len(Q0)
sub = np.ones(Nn - 1, dtype=int)
for k in range(Nn - 1):
    st = max(qang(Q0[k], Q0[k+1]), qang(Q1[k], Q1[k+1]))
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
orig2new = np.concatenate([[0], np.cumsum(sub)])   # 原帧号 -> 重采样帧号
for key, isq in (("obj_pos_0", 0), ("obj_pos_1", 0), ("obj_quat_0", 1),
                 ("obj_quat_1", 1), ("right_q", 0), ("left_q", 0)):
    out[key] = _resample(out[key], bool(isq))
T_new = len(out["right_q"]) - args.grip_frames
print(f"[carry] 限幅重采样: cap {_cap}°/帧 | {Nn} -> {T_new} 帧 "
      f"(插 {T_new - Nn} 帧, 细分最多 {int(sub.max())}x @原帧{int(sub.argmax())})")

# ---------------- 高置信里程碑 (可信约束下最大均匀) ----------------
_hi = None
for si, ri in ((0, RTS_IDX[0]), (1, RTS_IDX[1])):
    zc = np.load(f"{args.rts_dir}/rts_pour_17_object_{ri}.npz", allow_pickle=True)
    h = ((np.asarray(zc["conf_pos"]) >= args.conf)
         & (np.asarray(zc["conf_rot"]) >= args.conf)
         & np.asarray(zc["measurement_used"]).astype(bool)
         & (np.asarray(zc["innov_pos_nis"]) < 5.0)
         & (np.asarray(zc["innov_rot_nis"]) < 5.0))[::args.stride]
    _hi = h if _hi is None else (_hi & h)
trusted_new = orig2new[np.flatnonzero(_hi)]        # 可信原帧 -> 新帧号
ms = []
for anchor in range(args.ms_gap, T_new - 5, args.ms_gap):
    if len(trusted_new) == 0:
        break
    cand = trusted_new[np.argmin(np.abs(trusted_new - anchor))]
    if not ms or cand - ms[-1] >= args.ms_gap // 2:   # 防扎堆
        ms.append(int(cand))
ms = sorted(set(ms))
out["ms_idx"] = np.asarray(ms, np.int64) + args.grip_frames   # 全时间轴下标(含热身)
print(f"[carry] 高置信里程碑 {len(ms)} 个 @新帧 {ms} (目标间隔 {args.ms_gap}, "
      f"可信帧 {int(_hi.sum())}/{Nn}; 信任沙漠段自然留空)")

out["grip_frames"] = np.int64(args.grip_frames)
out["meta"] = np.array(f"carry_pour17 yaw={np.degrees(_yaw):.1f} stride={args.stride}")
np.savez(args.out, **out)
print(f"[carry] ✅ 写出 {args.out} | 路径长 {len(out['right_q'])} 行 (含热身)")
