"""Pour17 确定性评测 — 关探索噪声 (只用 mu), 全程 t0 口径 (考试分布).

  SHARPA_WANDB=0 PYTHONPATH=. $PY <task>/C_Wiring/eval_task.py \
      --checkpoint logs/Pour17_0/stage1_nn/last.pth --num_envs 256 --headless

成功 = M4 (终局); 逐关率 M1-M4 按"完成回合"精确计, 另报失败谱 (超时/env侧/进度机)。
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=512, help="总完成回合数(下限)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("pour17_eval")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_env as PE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
cfg = PE.build_cfg(num_envs=args.num_envs)
raw = PE.PourEnv(cfg)
raw.force_entry = [0]                       # 考试分布 = 全程 t0
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_task.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir="/tmp/pour17_eval", full_config=ConfigWrapper(agent_cfg, {}, test=True),
            create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

obs = env.reset()
N = args.num_envs
done_n = 0
ms_cnt = {1: 0, 2: 0, 3: 0, 4: 0}
fail_spec = {"timeout": 0, "fail": 0}
with torch.no_grad():
    while done_n < args.episodes:
        inp = {"obs": agent.running_mean_std(obs["obs"]),
               "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        d = dones.nonzero(as_tuple=False).squeeze(1)
        if len(d):
            # dones 时 PB 状态尚未被 reset 覆盖? reset 在 step 内已发生 —— 用 TB 口径
            done_n += len(d)
if hasattr(raw.PB, "pop_rates"):
    # 评测期间 PB.reset_idx 已在 env 内累计逐关率
    rates = raw.PB.pop_rates()
    print(f"[eval] 完成回合≈{done_n} | " +
          " ".join(f"{k}={v:.3f}" for k, v in rates.items()))
print(f"[eval] ★Success(M4) = {rates['sr/gate4']:.3f}")
try:
    _slot.release()
except Exception:
    pass
app.close()
