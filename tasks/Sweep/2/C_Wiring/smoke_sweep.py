"""Short finite-state smoke test for zero or random residual actions."""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--steps", type=int, default=64)
p.add_argument("--random", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_smoke")
app = AppLauncher(args).app
import torch  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
raw = SE.SweepEnv(SE.build_cfg(args.num_envs)); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
obs = env.reset()
for step in range(args.steps):
    action = (0.1 * torch.randn(args.num_envs, SE.ACT_DIM, device=raw.device)
              if args.random else torch.zeros(args.num_envs, SE.ACT_DIM, device=raw.device))
    obs, reward, done, info = env.step(action)
    assert torch.isfinite(obs["obs"]).all() and torch.isfinite(reward).all()
print(f"[smoke_sweep] PASS envs={args.num_envs} steps={args.steps} random={args.random} "
      f"row_max={int(raw.row.max())} gates={raw.progress.gates.any(0).int().tolist()}")
try: _slot.release()
except Exception: pass
app.close()
