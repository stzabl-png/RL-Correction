"""Minimal PPO entry point for a staged static reconstruction.

This verifies that the reconstructed asset and object-only placement contract
can be consumed by the existing Step4 PPO stack.  It does not claim grasping
success and does not replay the video wrist trajectory as a robot command.
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="task1_static_smoke")
parser.add_argument("--name", default="static_phone_smoke")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--max_agent_steps", type=int, default=8192)
parser.add_argument("--output_root", default="logs/static_reconstruction")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("static_reconstruction_train")
app = AppLauncher(args).app

from datetime import datetime  # noqa: E402

import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402


def main() -> int:
    if args.num_envs < 1 or args.horizon < 1 or args.max_agent_steps < 1:
        raise ValueError("num_envs, horizon, and max_agent_steps must be positive")
    entry = clips.clip_entry(args.clip)
    if entry.get("source") != "static_reconstruction":
        raise ValueError(f"{args.clip!r} is not a static_reconstruction clip")

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "ppo.yaml"), encoding="utf-8") as handle:
        agent_cfg = yaml.safe_load(handle)

    env_cfg = DexmateCorrectionEnvCfg()
    clips.configure_cfg(env_cfg, args.clip)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.rsi_prob = 0.0
    env_cfg.grasp_only = False
    env_cfg.show_dexmate = False
    env_cfg.show_human_traj = False

    batch_size = args.num_envs * args.horizon
    algorithm = agent_cfg["algorithm"]
    agent_cfg["seed"] = args.seed
    agent_cfg["device"] = args.device
    algorithm["experiment_name"] = args.name
    algorithm["num_actors"] = args.num_envs
    algorithm["horizon_length"] = args.horizon
    algorithm["minibatch_size"] = min(args.num_envs * 8, batch_size)
    algorithm["max_agent_steps"] = args.max_agent_steps

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join(args.output_root, args.name, stamp))
    print(
        f"[static-train] clip={args.clip} envs={args.num_envs} "
        f"horizon={args.horizon} max_steps={args.max_agent_steps} "
        f"device={args.device} log_dir={log_dir}",
        flush=True,
    )

    env_raw = DexmateCorrectionEnv(env_cfg)
    env = GymStyleEnvWrapper(env_raw, clip_actions=env_cfg.clip_actions)
    agent = PPO(env, output_dir=log_dir,
                full_config=ConfigWrapper(agent_cfg, env_cfg))
    agent.epoch_hook = _slot.yield_if_paused
    try:
        agent.train()
        writer = getattr(agent, "writer", None)
        if writer is not None:
            writer.flush()
            writer.close()
        print(f"[static-train] completed: {log_dir}", flush=True)
    except Exception as error:
        print(f"[static-train] FAILED: {error}", file=sys.stderr, flush=True)
        env.close()
        os._exit(1)
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
