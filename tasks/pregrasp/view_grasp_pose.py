"""双手直接摆到各自 GraspPose —— GUI 检查脚本 (不规划, 只做 IK + 静置)。
(2026-08-17 用户已确认: 这两个 GraspPose 摆放没问题)

回答的问题: "GraspPose 摆放好不好?" —— 把两只手用 IK 直接钉到瓶/杯的 GraspPose 上,
肉眼看手和物体的相对关系。绿球 = 右手靶点(瓶), 红球 = 左手靶点(杯)。

用法:
    cd /home/lyh/Project/RL_Correction
    SHARPA_WANDB=0 PYTHONPATH=. /home/lyh/luhr/MagicSim/.venv/bin/python \
        -m tasks.pregrasp.view_grasp_pose

    --headless   自检模式: 只打印 IK 误差后退出
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--prior_a", default="tasks/pregrasp/priors/Pour17_bottle.npz")
p.add_argument("--yaw_a", type=float, default=19.5)
p.add_argument("--prior_b", default="tasks/pregrasp/priors/Pour17_cup.npz")
p.add_argument("--fingers", choices=["grasp", "open"], default="grasp",
               help="grasp=手指上GraspPose合拢位形(默认); open=旧行为只对齐腕")
p.add_argument("--yaw_b", type=float, default=180.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("view_grasp_pose")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

BAR = "=" * 70


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
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)

A_name = cfg.hand_side                                  # 右手 -> 瓶
B_name = "left" if A_name == "right" else "right"       # 左手 -> 杯
A_gp = E._grasp_pos_w.cpu().numpy().astype(np.float64)
A_gq = E._grasp_quat_w.cpu().numpy().astype(np.float64)
A_obj = E.obj_init_pos.cpu().numpy().astype(np.float64)
aux_off = getattr(E, "aux_rel_offset_np", None)
assert aux_off is not None, "没有第二个物体"
B_obj = A_obj + np.asarray(aux_off, np.float64)
zb = np.load(args.prior_b)
_a = np.radians(args.yaw_b)
oq_b = qmul(np.array([np.cos(_a/2), 0, 0, np.sin(_a/2)]),
            np.asarray(zb["canon_rot"], np.float64))
B_gp = quat_to_R(oq_b) @ np.asarray(zb["grasp"][:3], np.float64) + B_obj
B_gq = qmul(oq_b, np.asarray(zb["grasp"][3:7], np.float64))

# ---- 记录 A 侧关节索引, 再解析 B 侧的 ----
A_jids = list(E.arm_jids)
A_hjids = list(E.hand_jids)
A_open = E.q_open.clone()
E._resolve_joint_ids(force_side=B_name)
B_jids = list(E.arm_jids)
B_hjids = list(E.hand_jids)
B_open = E.q_open.clone()
E._resolve_joint_ids(force_side=A_name)          # 还原

# ---- 双侧 IK (锚点 = env 实测 arm_center, 与训练同口径) ----
print(f"\n{BAR}\n双手直接摆到 GraspPose | {args.clip}\n{BAR}")
sol = {}
for nm, side, gp, gq in ((f"右手->瓶", A_name, A_gp, A_gq),
                         (f"左手->杯", B_name, B_gp, B_gq)):
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=E._anchor_T)
    r = ik.solve(gp, quat_to_R(gq), iters=300)
    sol[side] = r["q"]
    print(f"  {nm}: IK {'✅' if r['ok'] else '❌'} 位置误差 {r['pos_err']*100:.2f}cm "
          f"| 靶点 {np.round(gp*100,1)} cm")

# ---- 摆上去并静置 ----
q = E.hand.data.default_joint_pos.clone()
q[:, A_jids] = torch.tensor(sol[A_name], dtype=torch.float32, device=E.device)
q[:, B_jids] = torch.tensor(sol[B_name], dtype=torch.float32, device=E.device)
if args.fingers == "grasp":
    # GraspPose 完整位形: 手指上 prior 的 grasp[7:29] (22关节, 含 thumbfix)
    from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
    za = np.load(args.prior_a)
    jn = list(E.hand.joint_names)
    for z_, side_ in ((za, A_name), (zb, B_name)):
        vals = np.asarray(z_["grasp"], np.float64)[7:29]
        for n_, v_ in zip(GENERIC_JOINT_ORDER, vals):
            q[:, jn.index(n_.replace("right_", f"{side_}_"))] = float(v_)
else:
    q[:, A_hjids] = A_open.unsqueeze(0)          # 手指张开(接近段的手型)
    q[:, B_hjids] = B_open.unsqueeze(0)
E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
E.hand.set_joint_position_target(q)
E.hand.write_data_to_sim()

E._draw_sphere("/World/TgtR", (A_gp if A_name == "right" else B_gp) + W,
               (0.15, 1.0, 0.15), 0.014)
E._draw_sphere("/World/TgtL", (B_gp if B_name == "left" else A_gp) + W,
               (1.0, 0.15, 0.15), 0.014)
print(f"\n[view] 绿球 = 右手靶点(瓶) | 红球 = 左手靶点(杯)")
print(f"[view] 检查点: 手座是否贴着球 / 手指是否环向物体 / 有没有穿桌穿物")
print(f"[view] 关窗口或 Ctrl-C 退出\n{BAR}")

if getattr(args, "headless", False):
    print("[view] headless 自检完成, 退出")
    app.close()
    raise SystemExit(0)

try:
    while app.is_running():
        E.hand.set_joint_position_target(q)
        E.hand.write_data_to_sim()
        E.sim.step(render=True)
except KeyboardInterrupt:
    pass
app.close()
