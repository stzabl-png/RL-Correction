"""标定 pen_wrench 的权重 —— 换机器人后这一项的**量纲变了**.

飞手:    wrench_norm = 0.5·(|F|/200 + |T|/5), F/T 是推浮动根的凭空 wrench
DexMate: effort_norm = 7 个臂关节 |τ|/τ_max 的均值, τ_max 用 URDF 真机值

reward.py 里 p_wrench = -w_wrench·x², w_wrench=0.02 是按**飞手**那个量级调的.
两边 x 的典型值差多少, w_wrench 就要反着差多少的平方 —— 否则省力惩罚会压过任务奖励
(策略学到"别动"), 或者形同虚设.

  $PY -m rl_rebuild.correction.calib_effort_penalty --robot flying
  $PY -m rl_rebuild.correction.calib_effort_penalty --robot dexmate
一个进程只能建一个 Isaac env, 两边要分别跑.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--robot", choices=("flying", "dexmate"), required=True)
p.add_argument("--clip", default="Grasp2")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--sigma", type=float, default=0.0,
               help=">0: 叠加高斯随机动作, 量的是**带探索噪声**时的用量 (更接近真实训练)")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_s = isaac_slot("calib_eff")
app = AppLauncher(a).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.registry import make_env  # noqa: E402
from rl_rebuild.correction.env.reward import RewardWeights  # noqa: E402

EnvCls, CfgCls = make_env(a.robot)
cfg = CfgCls()
clips.configure_cfg(cfg, a.clip)
cfg.scene.num_envs = a.num_envs
cfg.rsi_prob = 0.0
env = EnvCls(cfg)
env.reset()

xs = []
for k in range(env.ep_total * a.episodes):
    act = (torch.randn(env.num_envs, cfg.action_space, device=env.device) * a.sigma
           if a.sigma > 0 else torch.zeros(env.num_envs, cfg.action_space, device=env.device))
    env.step(act.clamp(-1, 1))
    if k >= cfg.settle_steps:
        xs.append(env.effort_norm.mean().item())
xs = np.array(xs)
w = RewardWeights().w_wrench
print("\n" + "=" * 78)
print(f"robot={a.robot}  clip={a.clip}  sigma={a.sigma}  ({len(xs)} 个控制步)")
print(f"  effort/wrench_norm:  中位 {np.median(xs):.4f}  均值 {xs.mean():.4f}  "
      f"95分位 {np.percentile(xs,95):.4f}  最大 {xs.max():.4f}")
print(f"  当前 w_wrench={w}  ->  每步惩罚 中位 {w*np.median(xs)**2:.5f}  "
      f"95分位 {w*np.percentile(xs,95)**2:.5f}")
print(f"CALIB_EFFORT {a.robot} {np.median(xs):.5f} {xs.mean():.5f} "
      f"{np.percentile(xs,95):.5f} {w*np.median(xs)**2:.6f}")
print("=" * 78)
print("用法: 两个 robot 各跑一次, 取 中位 的比值 r; 新 w_wrench = 0.02 / r²,")
print("      这样两边**每步惩罚的绝对值**一致, 省力项在总 reward 里的占比才不变.")
sys.stdout.flush()
env.close()
_s.release()
os._exit(0)
