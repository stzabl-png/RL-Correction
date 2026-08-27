"""Motion_Planning 拼装器 (L1-4, Isaac 无头一次性脚本)。

输入: curobo_stance2pregrasp.npz (cuRobo 联合规划, 站姿->双手掌心锚点 PreGrasp)
输出: Approach_pour17.npz  = cuRobo 段(手指张开) + 成形段(PreGrasp 6 级梯->GraspPose,
                             ArmIK 逐级解臂, 手指按 prior 逐行 [7:29] 走到合拢)
      Retreat_pour17.npz   = Approach 整条倒放 (完全一样的动作倒着做)

行格式: right_q/left_q (T,7) 臂关节 + right_f/left_f (T,22) 手指关节 + seg 表。
用法:
    cd /home/lyh/Project/RL_Correction
    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.Pour.17.A_Design.L1_Data.Motion_Planning.build_motion --headless
(目录含点号不能做包名 — 实际用 python <绝对路径> 直跑, 见 README)
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--prior_a", default="tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")
p.add_argument("--yaw_a", type=float, default=19.5)
p.add_argument("--prior_b", default="tasks/pregrasp/priors/Pour17_cup_thumbfix.npz")
p.add_argument("--yaw_b", type=float, default=90.0)
p.add_argument("--plan", default="tasks/Pour/17/A_Design/L1_Data/Motion_Planning/"
                                 "curobo_stance2pregrasp.npz")
p.add_argument("--out_dir", default="tasks/Pour/17/A_Design/L1_Data/Motion_Planning")
p.add_argument("--form_frames", type=int, default=12,
               help="成形段每级梯之间的插值帧数")
p.add_argument("--close_backoff", type=float, default=0.03,
               help="合拢终点指关节回退量(rad): 抵消运动学回放的微穿透驱动"
                    "(2026-08-27 杯在手里打转伪影), 0=完全合拢")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("build_motion")
app = AppLauncher(args).app

import os  # noqa: E402

import numpy as np  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import (  # noqa: E402
    GENERIC_JOINT_ORDER,
)
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402


def qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.prior_a, args.yaw_a, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()

A_name = cfg.hand_side
B_name = "left" if A_name == "right" else "right"
A_obj = E.obj_init_pos.cpu().numpy().astype(np.float64)
aux_off = getattr(E, "aux_rel_offset_np", None)
assert aux_off is not None, "没有第二个物体"
B_obj = A_obj + np.asarray(aux_off, np.float64)

# ---- 每侧: prior 的 pregrasp(6,29)+grasp(29) -> 物体系->世界系 -> ArmIK 臂行 ----
za = np.load(args.prior_a)
zb = np.load(args.prior_b)
PRI = {A_name: (za, args.yaw_a, A_obj), B_name: (zb, args.yaw_b, B_obj)}

jn = list(E.hand.joint_names)
plan = np.load(args.plan, allow_pickle=True)
arm_rows = {"right": np.asarray(plan["right_q"], np.float64),
            "left": np.asarray(plan["left_q"], np.float64)}
T_plan = len(arm_rows["right"])
assert len(arm_rows["left"]) == T_plan

# 手指张开位形 (逐侧, 与训练出生手型一致)
q_open = {}
for side in (A_name, B_name):
    E._resolve_joint_ids(force_side=side)
    q_open[side] = E.q_open.clone().cpu().numpy().astype(np.float64)
    hj = list(E.hand_jids)
    # 手指关节名顺序 (22): 以 GENERIC_JOINT_ORDER 换侧后的名字为准
    names = [n.replace("right_", f"{side}_") for n in GENERIC_JOINT_ORDER]
    idx_in_hand = [hj.index(jn.index(n)) for n in names]
    q_open[side] = q_open[side][idx_in_hand]          # 重排到 GENERIC 顺序
E._resolve_joint_ids(force_side=A_name)

form_arm, form_fin = {}, {}
for side in (A_name, B_name):
    z, yaw, obj_p = PRI[side]
    a = np.radians(yaw)
    oq = qmul(np.array([np.cos(a/2), 0, 0, np.sin(a/2)]),
              np.asarray(z["canon_rot"], np.float64))
    Ro = quat_to_R(oq)
    # 终点=完整 GraspPose (2026-08-27 定身版恢复: 查看器已每帧钳住物体,
    # 合拢不再驱动打转; 毫米级手模标定差将以网格重叠形式可见, 供穿模稽查)
    ladder = list(np.asarray(z["pregrasp"], np.float64))     # 远->近 6 级
    ladder.append(np.asarray(z["grasp"], np.float64))        # + 完整抓形
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=E._anchor_T)
    seed = arm_rows[side][-1].copy()                          # cuRobo 终点位形接续
    way_a, way_f = [seed], [q_open[side]]
    for li, row in enumerate(ladder):
        wp = Ro @ row[:3] + obj_p
        wq = qmul(oq, row[3:7])
        r = ik.solve(wp, quat_to_R(wq), q0=way_a[-1], iters=300)
        assert r["pos_err"] < 0.02, f"{side} 成形梯级{li} IK 误差 {r['pos_err']*100:.1f}cm"
        way_a.append(r["q"])
        way_f.append(row[7:29])
    # 丝滑化(2026-08-27 用户裁定): 全局三次样条穿过全部路点 + 整段单次缓入缓出,
    # 段间速度连续 —— 旧版逐级 smoothstep 每级起止速度归零 = 七次微顿挫
    # PCHIP 单调插值 (2026-08-27 验尸: CubicSpline 在级5->回退终点间过冲到
    # 更深合拢, f150 杯被夹飞 267°/s —— 三次样条过冲是我引入的 bug)
    from scipy.interpolate import PchipInterpolator
    T = max(2, int(args.form_frames)) * len(ladder)
    sway = np.arange(len(way_a), dtype=np.float64)
    tt = np.linspace(0.0, 1.0, T)
    u = (tt * tt * (3 - 2 * tt)) * (len(way_a) - 1)
    form_arm[side] = PchipInterpolator(sway, np.stack(way_a), axis=0)(u)
    form_fin[side] = PchipInterpolator(sway, np.stack(way_f), axis=0)(u)
    print(f"[build] {side} 成形段 {T} 行 (样条过 {len(way_a)} 路点, 单次缓入缓出)")

T_form = len(form_arm[A_name])
assert len(form_arm[B_name]) == T_form

# ---- 拼装 Approach: cuRobo 段(手指张开) + 成形段 ----
out = {}
for side in ("right", "left"):
    fin_plan = np.tile(q_open[side], (T_plan, 1))
    out[f"{side}_q"] = np.concatenate([arm_rows[side], form_arm[side]]).astype(np.float32)
    out[f"{side}_f"] = np.concatenate([fin_plan, form_fin[side]]).astype(np.float32)
out["fin_names"] = np.array(GENERIC_JOINT_ORDER, dtype=object)
out["seg_names"] = np.array(["curobo_stance2pregrasp", "form_pregrasp2grasp"], dtype=object)
out["seg_lens"] = np.array([T_plan, T_form], np.int64)
out["meta"] = (f"Approach_pour17: cuRobo({T_plan}) + 成形梯({T_form}) "
               f"新场景(桌87cm/新站姿) thumbfix priors yaw {args.yaw_a}/{args.yaw_b}")

ap = os.path.join(args.out_dir, "Approach_pour17.npz")
np.savez(ap, **out)
print(f"[build] Approach -> {ap} ({T_plan}+{T_form}={T_plan+T_form} 行)")

rev = {k: (v[::-1].copy() if getattr(v, "ndim", 0) >= 1 and len(v) == T_plan + T_form
           else v) for k, v in out.items()}
rev["seg_names"] = np.array(["form_grasp2pregrasp", "curobo_pregrasp2stance"], dtype=object)
rev["seg_lens"] = np.array([T_form, T_plan], np.int64)
rev["meta"] = "Retreat_pour17: Approach 整条倒放"
rp = os.path.join(args.out_dir, "Retreat_from_build_unused.npz")   # 正式 Retreat 已由用户设计版定稿, 勿覆盖
np.savez(rp, **rev)
print(f"[build] Retreat -> {rp}")

try:
    _slot.release()
except Exception:
    pass
app.close()
raise SystemExit(0)
