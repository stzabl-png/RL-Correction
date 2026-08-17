"""成功率 vs 起点距离 —— 分开"训练平均被近距起点撑着"和"整体都不行"。

背景(2026-08-16): 训练起点 `j0 ~ U(0, L-1)` 均匀铺满整条参考路径(平均离终点 42 帧),
而评测钉死 `j0=0`(离终点 83 帧, 最难那一档且全部)。于是 TB 成功率 0.37~0.55 而
确定性评测 0.00% —— 两个数根本不是一回事。本脚本把成功率按**起点距离**拆开来看。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.diag_start_sweep --headless \\
        --checkpoint logs/AppV2_Grasp3_L/<run>/stage1_nn/last.pth \\
        --clip Grasp3 --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz \\
        --prior_yaw 215
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
p.add_argument("--buckets", type=int, default=7, help="起点距离分几档")
p.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("start_sweep")
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
cfg.scene.num_envs = args.num_envs
raw = GraspTaskEnv(cfg)
raw.gentle = 1.0
if hasattr(raw, "_eps_final"):          # 用**最终**公差, 与正式评测同口径
    cfg.eps_pos, cfg.eps_rot = raw._eps_final
env = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
agent = PPO(env, output_dir="/tmp/_sweep", create_output_dir=False,
            full_config=ConfigWrapper(agent_cfg, cfg, test=True))
agent.restore_test(args.checkpoint)
agent.set_eval()

L = raw.retract_path.shape[0]
edges = np.linspace(0, L - 1, args.buckets + 1).round().astype(int)

print(f"\n{'=' * 76}")
print(f"[起点扫描] {args.clip} | 路径 {L} 帧 | 公差 {cfg.eps_pos*100:.2f}cm/"
      f"{np.degrees(cfg.eps_rot):.2f}° (正式口径) | 每档 {args.num_envs} env")
print(f"{'起点离终点(帧)':>16} {'成功率':>9} {'超时':>8} {'撞/碰终止':>10} {'到位中位步':>11}")

_orig = GraspTaskEnv._reset_idx
_pin = {"j0": 0}


def _patched(self, env_ids):                      # 把起点钉在指定档位
    _orig(self, env_ids)
    n = len(env_ids)
    t0 = torch.full((n,), int(_pin["j0"]), dtype=torch.long, device=self.device)
    q = self.hand.data.joint_pos[env_ids].clone()
    q[:, self.arm_jids] = self.retract_path[t0]
    self.hand.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    self.hand.set_joint_position_target(q, env_ids=env_ids)
    self.q_cmd[env_ids] = self.retract_path[t0]
    self.arm_tgt[env_ids] = self.retract_path[t0]
    self.ref_t[env_ids] = t0
    self.ref_q_prev[env_ids] = self.retract_path[t0]


GraspTaskEnv._reset_idx = _patched
rows = []
for b in range(args.buckets):
    _pin["j0"] = int(edges[b])
    obs = env.reset()
    done_once = torch.zeros(args.num_envs, dtype=torch.bool, device=raw.device)
    succ = to = hit = 0
    lens = []
    with torch.no_grad():
        for st in range(args.steps):
            _inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
            if "pointcloud" in obs:
                _inp["pointcloud"] = obs["pointcloud"]
            obs, _, d, _ = env.step(torch.clamp(agent.model.act_inference(_inp), -1.0, 1.0))
            d = d.bool() if torch.is_tensor(d) else torch.tensor(d, device=raw.device).bool()
            fresh = d & ~done_once
            if fresh.any():
                done_once |= d
                idx = torch.nonzero(fresh, as_tuple=False).squeeze(-1)
                ns = raw.succeeded[idx]
                succ += int(ns.sum())
                lens += [st] * int(ns.sum())
                to += int((raw._sig["timeout"][idx] & ~ns).sum()) if "timeout" in raw._sig else 0
                hit += int((raw._approach_hit[idx] & ~ns).sum())
            if bool(done_once.all()):
                break
    n = args.num_envs
    rows.append((L - 1 - edges[b], succ / n, to / n, hit / n,
                 float(np.median(lens)) if lens else float("nan")))
    print(f"{rows[-1][0]:>16d} {rows[-1][1]*100:>8.1f}% {rows[-1][2]*100:>7.1f}% "
          f"{rows[-1][3]*100:>9.1f}% {rows[-1][4]:>11.0f}")

print("-" * 76)
print("判读: 成功率随起点距离**陡降到 0** ⟹ 训练平均值被近距起点撑着, 远端没学会;")
print("      全程**平坦且低**       ⟹ 与起点距离无关, 是别的问题。")
print(f"{'=' * 76}\n")
app.close()
