"""Deterministic fixed-start evaluation; acceptance requires >=512 episodes, >=50%."""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--out", required=True, help="JSON path under project logs/")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.num_envs >= 512, "acceptance evaluation requires at least 512 episodes"

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_eval")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
out = os.path.abspath(args.out)
assert os.path.commonpath([os.path.join(ROOT, "logs"), out]) == os.path.join(ROOT, "logs")
raw = SE.SweepEnv(SE.build_cfg(args.num_envs)); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(os.path.dirname(__file__), "ppo_sweep.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir=os.path.join(ROOT, "logs", "_eval_tmp"),
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint); agent.set_eval()
obs = env.reset(); pending = torch.ones(args.num_envs, dtype=torch.bool, device=raw.device)
success = torch.zeros_like(pending); lengths = torch.zeros(args.num_envs, device=raw.device)
with torch.no_grad():
    while bool(pending.any()):
        mu = agent.model.act_inference({"obs": agent.running_mean_std(obs["obs"]),
                                        "priv_info": obs["priv_info"]}).clamp(-1, 1)
        obs, reward, done, info = env.step(mu)
        lengths[pending] += 1
        first = pending & done
        success[first] = raw._tick_out["success"][first]
        pending[first] = False
rate = float(success.float().mean())
report = {"checkpoint": os.path.abspath(args.checkpoint), "episodes": args.num_envs,
          "successes": int(success.sum()), "success_rate": rate,
          "mean_length": float(lengths.mean()), "acceptance_threshold": 0.50,
          "accepted": bool(rate >= 0.50)}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f: json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
try: _slot.release()
except Exception: pass
app.close()
if rate < 0.50: raise SystemExit(2)
