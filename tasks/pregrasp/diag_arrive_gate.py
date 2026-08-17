"""到位判据逐条体检 —— 四个条件里到底是哪一条卡住。

背景(2026-08-16): 确定性策略在**所有**起点距离(12~83 帧)上成功率都是 0.0%, 而训练
TB 说 0.37~0.55。既然与起点无关, 嫌疑就落到判据本身:

    到位 = (位置差 < eps_pos) 且 (朝向差 < eps_rot) 且 (腕速 < switch_vel_max)
           且 以上连续保持 switch_hold 步

本脚本让确定性策略跑一遍, 把四条**分别**统计, 直接指出哪条从来没满足。
⚠ 不看成功率(已知是 0), 看的是各条件各自的达成率与最好值。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.diag_arrive_gate --headless \\
        --checkpoint <ckpt> --clip Grasp3 \\
        --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--clip", default="Grasp3")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--steps", type=int, default=260)
p.add_argument("--start_j0", type=int, default=-1,
               help=">=0: 起点钉在路径这一帧; 默认 -1 = 站姿(j0=0, 正式口径)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("arrive_gate")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only = True
cfg.action_space = 7
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.retract_start = True
cfg.stance_prob = 1.0            # 正式口径: 全部从站姿出发
cfg.retract_ratio = 1.0
cfg.scene.num_envs = args.num_envs
raw = GraspTaskEnv(cfg)
raw.gentle = 1.0
if hasattr(raw, "_eps_final"):
    cfg.eps_pos, cfg.eps_rot = raw._eps_final
env = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
agent = PPO(env, output_dir="/tmp/_gate", create_output_dir=False,
            full_config=ConfigWrapper(agent_cfg, cfg, test=True))
agent.restore_test(args.checkpoint)
agent.set_eval()

N = args.num_envs
dev = raw.device
best_p = torch.full((N,), 1e9, device=dev)      # 整回合最小位置差
best_r = torch.full((N,), 1e9, device=dev)
n_p = torch.zeros(N, device=dev)                # 各条件的达成步数
n_r = torch.zeros(N, device=dev)
n_v = torch.zeros(N, device=dev)
n_all = torch.zeros(N, device=dev)
best_run = torch.zeros(N, device=dev)           # 连续满足的最长步数
# 位置&朝向都达标那一刻的腕速 (若从未同时达标则留 nan)
v_at_pr = torch.full((N,), float("nan"), device=dev)

from rl_rebuild.correction.kinematics import ArmIK as _AIK
_IK = _AIK(cfg.hand_side, anchor_link="arm_center", anchor_T=raw._anchor_T)
_gp_np = raw._grasp_pos_w.cpu().numpy().astype(np.float64)
snap_dp = torch.full((N,), 1e9, device=dev)     # 迄今最近的位置差
snap_cmd = torch.full((N,), float("nan"), device=dev)   # 那一刻的命令离目标
snap_res = torch.zeros(N, device=dev)
snap_act = torch.zeros(N, device=dev)
snap_v = torch.zeros(N, device=dev)
obs = env.reset()
with torch.no_grad():
    for st in range(args.steps):
        _inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        if "pointcloud" in obs:
            _inp["pointcloud"] = obs["pointcloud"]
        obs, _, d, _ = env.step(torch.clamp(agent.model.act_inference(_inp), -1.0, 1.0))
        dp, dr, _ = raw._align_err()
        v = raw.wrist_linvel_w.norm(dim=1)
        ok_p, ok_r, ok_v = dp < cfg.eps_pos, dr < cfg.eps_rot, v < cfg.switch_vel_max
        best_p = torch.minimum(best_p, dp)
        best_r = torch.minimum(best_r, dr)
        n_p += ok_p.float()
        n_r += ok_r.float()
        n_v += ok_v.float()
        pr = ok_p & ok_r
        n_all += (pr & ok_v).float()
        v_at_pr = torch.where(pr & torch.isnan(v_at_pr), v, v_at_pr)
        best_run = torch.maximum(best_run, raw.switch_run.float())
        # ---- 为什么停在门口: 在**离目标最近那一刻**取样 ----
        # ⚠ 不能按"固定第 N 步"取 —— 回合上限 250 步而我跑 260 步, 末尾 6 步测到的是
        #   **刚复位的新回合**(实测 29.3cm = 站姿距离), 完全无意义。踩过一次。
        _better = dp < snap_dp
        if bool(_better.any()):
            _qc = raw.arm_tgt.cpu().numpy().astype(np.float64)
            _idx = torch.nonzero(_better, as_tuple=False).squeeze(-1)
            for _i in _idx[:64].tolist():          # 逐个 FK 很慢, 每步最多更新 64 个
                _fkp = _IK.fk(_qc[_i])[0]
                snap_cmd[_i] = float(np.linalg.norm(_fkp - _gp_np))
            snap_res[_better] = raw.res_step_cm[_better]
            snap_act[_better] = raw.actions_buf[:, :7].abs().mean(dim=1)[_better]
            snap_v[_better] = v[_better]
            snap_dp = torch.where(_better, dp, snap_dp)

f = lambda t: (t > 0).float().mean().item() * 100.0
print(f"\n{'=' * 78}")
print(f"[到位判据体检] {args.clip} | {N} env × {args.steps} 步 | 起点=站姿(正式口径)")
print(f"  判据: 位置<{cfg.eps_pos*100:.2f}cm 且 朝向<{np.degrees(cfg.eps_rot):.2f}° "
      f"且 腕速<{cfg.switch_vel_max*100:.0f}cm/s, 连续 {cfg.switch_hold} 步")
print("-" * 78)
print(f"  ① 位置达标过的 env : {f(n_p):5.1f}%   最好 {best_p.min()*100:.2f}cm "
      f"(中位 {best_p.median()*100:.2f}cm)")
print(f"  ② 朝向达标过的 env : {f(n_r):5.1f}%   最好 {np.degrees(best_r.min().item()):.2f}° "
      f"(中位 {np.degrees(best_r.median().item()):.2f}°)")
print(f"  ③ 腕速达标过的 env : {f(n_v):5.1f}%")
print(f"  ①②同时达标过的 env: {f((n_p > 0) & (n_r > 0)):5.1f}%")
print(f"  ①②③同时达标过    : {f(n_all):5.1f}%")
print(f"  连续满足最长步数    : max {best_run.max():.0f} 步 (需要 {cfg.switch_hold} 步)")
_mc = ~torch.isnan(snap_cmd)
if bool(_mc.any()):
    print("-" * 78)
    print("  [为什么停在门口] 在**离目标最近那一刻**取样:")
    print(f"    实际离目标 : {snap_dp[_mc].median()*100:6.2f} cm")
    print(f"    **命令**离目标 : {snap_cmd[_mc].median()*100:6.2f} cm   "
          f"← 与实际接近 ⟹ 不是跟踪误差; 明显更小 ⟹ 跟踪不上")
    print(f"    该刻残差用量 : {snap_res[_mc].median():6.3f} cm")
    print(f"    该刻动作幅度 : {snap_act[_mc].median():6.3f}  (满量程 1.0)")
    print(f"    该刻腕速     : {snap_v[_mc].median()*100:6.2f} cm/s")
_m = ~torch.isnan(v_at_pr)
if bool(_m.any()):
    print(f"  ①②达标那一刻的腕速 : 中位 {v_at_pr[_m].median()*100:.1f}cm/s "
          f"最小 {v_at_pr[_m].min()*100:.1f}cm/s (门槛 {cfg.switch_vel_max*100:.0f}cm/s)")
else:
    print("  ①②从未同时达标 —— 卡在精度, 不是速度")
print("-" * 78)
print("判读: ①②达标率高而③低 ⟹ 卡在**腕速**(到得了但停不下来);")
print("      ①或②本身就低    ⟹ 卡在**精度**(根本没走到那么准);")
print("      三条都高但连续步数<hold ⟹ 卡在**保持**(路过而非停住)。")
print(f"{'=' * 78}\n")
app.close()
