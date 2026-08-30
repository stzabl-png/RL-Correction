"""Sweep2 actor-BC warm start followed by pure on-policy PPO."""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--name", default="Sweep2_fixed_seed42")
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--expert", required=True, help="successful expert NPZ under logs/")
p.add_argument("--bc_epochs", type=int, default=200)
p.add_argument("--max_agent_steps", type=int, default=None)
p.add_argument("--load_path", default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.headless, "training must be headless"

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot(f"sweep2_train_{args.name}")
app = AppLauncher(args).app

import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from bc_warmup import warm_actor  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
expert = os.path.abspath(args.expert)
assert os.path.commonpath([os.path.join(ROOT, "logs"), expert]) == os.path.join(ROOT, "logs")
cfg = SE.build_cfg(args.num_envs); cfg.seed = args.seed
raw = SE.SweepEnv(cfg); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(os.path.dirname(__file__), "ppo_sweep.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["seed"] = args.seed
acfg["algorithm"]["experiment_name"] = args.name
acfg["algorithm"]["num_actors"] = args.num_envs
acfg["algorithm"]["minibatch_size"] = min(
    acfg["algorithm"]["minibatch_size"], 32 * args.num_envs)
if args.max_agent_steps: acfg["algorithm"]["max_agent_steps"] = args.max_agent_steps
log_dir = os.path.join(ROOT, "logs", args.name); os.makedirs(log_dir, exist_ok=True)
class SweepPPO(PPO):
    def __init__(self, *a, raw_env=None, **kw):
        super().__init__(*a, **kw); self._raw = raw_env

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        for key, value in self._raw.pop_rates().items():
            self.writer.add_scalar(key, value, self.agent_steps)


agent = SweepPPO(env, output_dir=log_dir, full_config=ConfigWrapper(acfg, {}), raw_env=raw)
if args.load_path:
    agent.restore_train(args.load_path)
else:
    mse = warm_actor(agent, expert, epochs=args.bc_epochs)
    agent.save(os.path.join(log_dir, "stage1_nn", "bc_warm"))
    with open(os.path.join(log_dir, "bc_summary.txt"), "w") as f:
        f.write(f"expert={expert}\nepochs={args.bc_epochs}\nfinal_mse={mse}\n")
    print("[train_sweep] BC complete; all following samples/updates are pure on-policy PPO")
agent.train()
try: _slot.release()
except Exception: pass
app.close()
