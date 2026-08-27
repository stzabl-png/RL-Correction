"""判据可达性体检 —— **把手臂钉在 GraspPose 上**, 看到位判据能不能满足。

## 为什么必须做这个

到位判据用的 `d_pos` 不是"腕到 GraspPose", 而是 **`_anchor_w()`(手上的锚点) 到
`_target_w()`(物体上的亲和点)**。这两者在"手正好摆在 GraspPose 上"时的距离**不一定是 0**。
若它本来就大于阈值 eps_pos, 那这道判据**在几何上不可能满足** —— 策略再怎么训都是 0%,
而 TB 上看起来只是"卡在某个距离"。

2026-08-17 的背景: Minimal_L 跑满 40M 步, 距离稳定在 **3.39~3.58cm** 不动,
`ep_rew/arrive` 全程 **0.00000**(那 100 分一次没拿到), 而 align 一直是正的
(靠近仍然净赚) ⟹ 不像"学不动", 像**撞墙**。本脚本就是去看那堵墙是不是判据自己砌的。

## 判读

    钉在 GraspPose 上时 d_pos ≈ 0        -> 判据可达, 卡住是学习问题
    钉在 GraspPose 上时 d_pos ≈ 3.4cm    -> **判据不可达**, 五条 run 全部作废,
                                            要改判据(或改 anchor/aff 的定义)

用法:
    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.diag_reachable --headless \\
        --clip Pour17_bottle --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz \\
        --prior_yaw 19.5
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--hold", type=int, default=60, help="钉住后静置多少步再量")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("diag_reachable")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.scene.num_envs = 4
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()

BAR = "=" * 72
print(f"\n{BAR}\n判据可达性体检: 把臂钉在 GraspPose 的 IK 解上\n{BAR}")

# ---- 把臂命令到 GraspPose 的 IK 解, 静置 ----
q = E.hand.data.default_joint_pos.clone()
q[:, E.arm_jids] = E.q_pregrasp.unsqueeze(0).expand(E.num_envs, 7)
q[:, E.hand_jids] = E.q_open.unsqueeze(0).expand(E.num_envs, len(E.hand_jids))
E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
E.hand.set_joint_position_target(q)
for _ in range(args.hold):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())

anc = E._anchor_w()
tgt = E._target_w()
d_pos = float((anc - tgt).norm(dim=1).mean())
wp = E.wrist_pos_w - E.scene.env_origins
gp = E._grasp_pos_w
d_wrist = float((wp - gp.unsqueeze(0)).norm(dim=1).mean()) if gp.dim() == 1 else \
    float((wp - gp).norm(dim=1).mean())

eps = float(cfg.eps_pos)
print(f"  判据阈值 eps_pos            = {eps*100:.2f} cm")
print(f"  **判据量的** d_pos          = {d_pos*100:.2f} cm  "
      f"(锚点 _anchor_w -> 亲和点 _target_w)")
print(f"  腕位置 vs GraspPose 腕位     = {d_wrist*100:.2f} cm  (这个才是'摆到位了没')")
print(f"  锚点相对腕的偏移 anchor_local = {np.round(E.anchor_local.cpu().numpy()*100, 2)} cm")
print(f"  亲和点相对物体 aff_local      = {np.round(E.aff_local.cpu().numpy()*100, 2)} cm")
print(BAR)
if d_pos <= eps:
    print(f"  ✅ 判据**可达**: 钉在 GraspPose 上时 d_pos {d_pos*100:.2f}cm ≤ 阈值 "
          f"{eps*100:.2f}cm\n     ⟹ 训练卡住是**学习问题**, 判据本身没错。")
else:
    print(f"  ❌ 判据**不可达**: 钉在 GraspPose 上时 d_pos {d_pos*100:.2f}cm > 阈值 "
          f"{eps*100:.2f}cm")
    print(f"     ⟹ 手摆到**正确位置**时判据依然不满足, 策略无论如何训都是 0%。")
    print(f"     ⟹ 要么把阈值放到 >{d_pos*100:.2f}cm, 要么让 d_pos 量'腕到 GraspPose'"
          f"而不是'锚点到亲和点'。")
print(BAR)
app.close()
