"""策略腕轨迹形状体检 —— 回答"手为什么往身体方向走"。

逐步打印 (env 坐标系):
  · 离目标多远              —— 任务本身
  · 离**臂基座**多远        —— 变小 = 手在往身体方向缩回来
  · 腕的高度                —— 是否先抬起再压下
  · 本步位移与"指向目标"的夹角 —— 0°=直着去, >90°=这一步在背离目标

## 为什么要量策略自己的轨迹

`Minimal_dyn` **完全没用人手轨迹**(无参考/无模仿罚), 路是自己学出来的 ——
去查人手数据答不了"它为什么往回走"。而且 Grasp3 的人手轨迹本身是
"帧 0 就离物体 8.2cm、之后一路远离"(抓着往回搬), 根本不是接近段。

用法:
    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.diag_wrist_path --headless \\
      --clip Grasp3 --checkpoint "logs/Minimal_dyn/*/stage1_nn/eval_best.pth" \\
      --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz \\
      --prior_yaw 215 --minimal
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp3")
p.add_argument("--checkpoint", required=True)
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--minimal", action="store_true",
               help="被测 ckpt 是 --minimal 训的; 场景必须逐项对齐")
p.add_argument("--steps", type=int, default=250)
p.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("diag_wrist_path")
app = AppLauncher(args).app

import glob  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
if args.minimal:
    cfg.minimal_no_ff = True
    cfg.minimal_fixed_res = False        # 与训练时一致: 保留远松近紧
    cfg.eps_pos, cfg.eps_rot = 0.01, np.radians(15.0)
    cfg.eps_pos0, cfg.eps_rot0 = cfg.eps_pos, cfg.eps_rot
    cfg.w_imit0_approach, cfg.w_imit_ramp = 0.0, 0.0
cfg.scene.num_envs = args.num_envs
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
if args.minimal:                          # 起点/直接抓取比例要在构造后落定
    cfg.stance_prob, cfg.retract_ratio = 1.0, 1.0
    cfg.direct_grasp_prob = 0.0
E.gentle = 1.0
env = GymStyleEnvWrapper(E, clip_actions=cfg.clip_actions)

# ★ 推理走 record.py 里验证过的那套 (含**观测归一化** running_mean_std ——
#   自己手搓 forward 会漏掉它, 动作是错的)
ck = sorted(glob.glob(args.checkpoint))[-1] if "*" in args.checkpoint else args.checkpoint
with open(os.path.join(os.path.dirname(__file__), "ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent = PPO(env, output_dir="/tmp/.diag_wrist", create_output_dir=False,
            full_config=ConfigWrapper(agent_cfg, cfg, test=True))
print(f"\n[diag] ckpt = {ck}")
agent.restore_test(ck)
agent.set_eval()
obs_dict = env.reset()

base = np.asarray(E._anchor_T[:3, 3], float)      # 臂基座 (env 局部系)
print(f"[diag] 臂基座 {np.round(base*100,1)} cm | 目标 "
      f"{np.round(E._target_w().mean(dim=0).cpu().numpy()*100,1)} cm")
print(f"\n{'步':>4} {'离目标cm':>9} {'离臂基座cm':>11} {'腕高cm':>8} {'走向夹角°':>10}")

prev = None
rows = []
with torch.no_grad():
    for k in range(args.steps):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        mu = agent.model.act_inference(_inp)
        obs_dict, _, _, _ = env.step(torch.clamp(mu, -1.0, 1.0))
        w = (E.wrist_pos_w - E.scene.env_origins).mean(dim=0).cpu().numpy()
        t = E._target_w().mean(dim=0).cpu().numpy()
        d_t, d_b = np.linalg.norm(w - t) * 100, np.linalg.norm(w - base) * 100
        ang = float("nan")
        if prev is not None:
            mv, to = w - prev, t - prev
            if np.linalg.norm(mv) > 1e-5 and np.linalg.norm(to) > 1e-5:
                ang = np.degrees(np.arccos(np.clip(
                    mv @ to / (np.linalg.norm(mv) * np.linalg.norm(to)), -1, 1)))
        rows.append((k, d_t, d_b, w[2] * 100, ang))
        if k % 10 == 0:
            print(f"{k:4d} {d_t:9.2f} {d_b:11.2f} {w[2]*100:8.2f} {ang:10.1f}")
        prev = w

# ---- 汇总: 到底有没有"往回走" ----
d_b0 = rows[0][2]
back = [r for r in rows if r[2] < d_b0 - 1.0]        # 离基座比出发时近 >1cm
away = [r for r in rows if not np.isnan(r[4]) and r[4] > 90.0]
print(f"\n{'='*70}")
print(f"  出发时离臂基座 {d_b0:.1f}cm | 全程最小 {min(r[2] for r in rows):.1f}cm "
      f"| 最大 {max(r[2] for r in rows):.1f}cm")
print(f"  **手比出发时更靠近身体的步数 = {len(back)}/{len(rows)}**")
print(f"  **背离目标(夹角>90°)的步数   = {len(away)}/{len(rows)-1}**")
print(f"  离目标: 首 {rows[0][1]:.1f}cm -> 末 {rows[-1][1]:.1f}cm | 最小 "
      f"{min(r[1] for r in rows):.2f}cm")
print(f"{'='*70}")
app.close()
