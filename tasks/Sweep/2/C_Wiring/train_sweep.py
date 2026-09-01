"""Sweep2 actor-BC warm start followed by pure on-policy PPO."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--name", default="Sweep2_fixed_seed42")
p.add_argument("--artifact_prefix", default="",
               help="artifact stem, e.g. Sweep2__20260830_policy")
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--expert25", required=True, help="25 mm transition NPZ under logs/")
p.add_argument("--expert40", required=True, help="40 mm near-success transition")
p.add_argument("--expert_full", required=True, help="old 15M fully-inside transition")
p.add_argument("--failure", required=True, help="canonical failure transition NPZ under logs/")
p.add_argument("--actor_epochs", type=int, default=300)
p.add_argument("--critic_epochs", type=int, default=200)
p.add_argument("--diag_every_steps", type=int, default=3_000_000)
p.add_argument("--no_autorec", action="store_true")
p.add_argument("--max_agent_steps", type=int, default=None)
p.add_argument("--load_path", default=None)
p.add_argument("--initial_agent_steps", type=int, default=0,
               help="resume-only global step count stored outside legacy checkpoints")
p.add_argument("--initial_epoch", type=int, default=0,
               help="resume-only completed PPO epoch count")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.headless, "training must be headless"

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot(f"sweep2_train_{args.name}")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from bc_warmup import warm_actor_critic  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
expert_paths = [os.path.abspath(p) for p in (
    args.expert25, args.expert40, args.expert_full, args.failure)]
for path in expert_paths:
    assert os.path.commonpath([os.path.join(ROOT, "logs"), path]) == os.path.join(ROOT, "logs")
    assert os.path.isfile(path), path
cfg = SE.build_cfg(args.num_envs); cfg.seed = args.seed
raw = SE.SweepEnv(cfg); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(os.path.dirname(__file__), "ppo_sweep.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["seed"] = args.seed
acfg["algorithm"]["experiment_name"] = args.name
acfg["algorithm"]["num_actors"] = args.num_envs
acfg["algorithm"]["save_frequency"] = 0
acfg["algorithm"]["save_best_after"] = 2**31 - 1
acfg["algorithm"]["minibatch_size"] = min(
    acfg["algorithm"]["minibatch_size"], 32 * args.num_envs)
if args.max_agent_steps: acfg["algorithm"]["max_agent_steps"] = args.max_agent_steps
log_dir = os.path.join(ROOT, "logs", args.name); os.makedirs(log_dir, exist_ok=True)
if args.artifact_prefix:
    artifact_prefix = args.artifact_prefix
else:
    match = re.search(r"(20\d{6})", args.name)
    artifact_prefix = f"Sweep2__{match.group(1)}_policy" if match else f"{args.name}_policy"
assert artifact_prefix == os.path.basename(artifact_prefix), artifact_prefix
checkpoints_root = os.path.join(ROOT, "logs", "checkpoints")
os.makedirs(checkpoints_root, exist_ok=True)

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

world = {
    "schema": 2, "task": "Sweep2_fixed_cube_fullinside", "seed": args.seed,
    "policy_io": {"obs_dim": SE.OBS_DIM, "priv_dim": SE.PRIV_DIM,
                  "act_dim": SE.ACT_DIM},
    "time": {"control_dt_s": 0.05,
             "scripted_prelude_steps": SE.SCRIPTED_PRELUDE_STEPS},
    "cube_start_world_m": list(SE.SWEEP2_FIXED_CUBE_START),
    "success": {"definition": "fully_inside",
                "whole_cube_inside": True, "mouth_clearance_margin_m": 0.0,
                "immediate_termination": True,
                "recording_only_freeze_seconds": 2.0,
                "final_eval_episodes": 512, "required_rate": 0.50},
    "transitions": [{"path": p, "sha256": _sha256(p)} for p in expert_paths],
    "reference": {
        "path": SE.REFERENCE, "sha256": _sha256(SE.REFERENCE)},
    "dustpan_asset": {},
}
_pan_asset = SE.clips.clip_entry("Sweep2_broom")["secondary"]["mesh"]
world["dustpan_asset"] = {"path": _pan_asset, "sha256": _sha256(_pan_asset)}
with open(os.path.join(log_dir, "world.json"), "w") as f:
    json.dump(world, f, indent=2)
class SweepPPO(PPO):
    def __init__(self, *a, raw_env=None, **kw):
        super().__init__(*a, **kw); self._raw = raw_env
        self._diag_every = int(args.diag_every_steps)
        self._next_diag = ((int(args.initial_agent_steps) // self._diag_every) + 1) * self._diag_every
    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        rates = self._raw.pop_rates()
        for key, value in rates.items():
            self.writer.add_scalar(key, value, self.agent_steps)
        with open(os.path.join(self.output_dir, "progress_steps.txt"), "w") as f:
            f.write(str(int(self.agent_steps)))
        while self._diag_every > 0 and self.agent_steps >= self._next_diag:
            tag = f"{self._next_diag // 1_000_000:04d}M"
            node_dir = os.path.join(checkpoints_root, f"{artifact_prefix}_{tag}")
            os.makedirs(node_dir, exist_ok=True)
            checkpoint = os.path.join(node_dir, "checkpoint")
            self.save(checkpoint)
            payload = {
                "schema": 1, "target_step": self._next_diag,
                "actual_agent_steps": int(self.agent_steps),
                "epoch": int(self.epoch_num), "checkpoint": checkpoint + ".pth",
                "rates": rates,
                "losses": {
                    "actor": float(torch.mean(torch.stack(a_losses))),
                    "critic": float(torch.mean(torch.stack(c_losses))),
                    "entropy": float(torch.mean(torch.stack(entropies))),
                    "kl": float(torch.mean(torch.stack(kls))),
                },
            }
            target = os.path.join(node_dir, "metrics.json")
            temp = target + ".tmp"
            with open(temp, "w") as f: json.dump(payload, f, indent=2)
            os.replace(temp, target)
            print(f"[3M诊断] checkpoint+JSON 已保存: {tag} "
                  f"(actual={self.agent_steps})", flush=True)
            self._next_diag += self._diag_every
        # Autorecord owns pause.request while its Isaac recorder is pending.
        # Yield only at this clean epoch boundary, after checkpoint durability.
        _slot.yield_if_paused()


agent = SweepPPO(env, output_dir=log_dir, full_config=ConfigWrapper(acfg, {}), raw_env=raw)
if args.load_path:
    agent.restore_train(args.load_path)
    agent.initial_agent_steps = int(args.initial_agent_steps)
    agent.epoch_num = int(args.initial_epoch)
else:
    summary = warm_actor_critic(
        agent, *expert_paths, actor_epochs=args.actor_epochs,
        critic_epochs=args.critic_epochs)
    bc_dir = os.path.join(checkpoints_root, f"{artifact_prefix}_BC")
    os.makedirs(bc_dir, exist_ok=True)
    agent.save(os.path.join(bc_dir, "checkpoint"))
    with open(os.path.join(log_dir, "bc_summary.txt"), "w") as f:
        f.write(
            f"expert25={expert_paths[0]}\nexpert40={expert_paths[1]}\n"
            f"failure={expert_paths[2]}\nactor_epochs={args.actor_epochs}\n"
            f"critic_epochs={args.critic_epochs}\nsummary={summary}\n")
    print("[train_sweep] actor/critic warmup complete; all following samples/updates are pure on-policy PPO")
if not args.no_autorec:
    monitor = os.path.join(os.path.dirname(__file__), "autorecord_sweep.sh")
    subprocess.Popen(
        ["bash", monitor, checkpoints_root, os.path.join(ROOT, "outputs_video"),
         artifact_prefix, sys.executable, str(os.getpid())],
        stdout=open(os.path.join(log_dir, "autorecord.log"), "a"),
        stderr=subprocess.STDOUT)
    print("[train_sweep] 3M checkpoint/data/video monitor started", flush=True)
agent.train()
try: _slot.release()
except Exception: pass
sys.stdout.flush()
# Isaac Sim 5.1 can hang during plugin teardown after all checkpoints and logs
# are durable.  Other project entrypoints use the same clean process boundary.
os._exit(0)
