# PHASE-1 DATA CURATION (2026-07-07): no-policy settle test. For one object, run pinch-replay
# resets and then ZERO actions (delta control -> hold targets). Measures the fraction of reset
# poses that physically SEAT the object (retained K steps after settle without termination).
# Objects with low seat rates have defective reset data (train_102 class) -> curate, don't train.
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--eval_steps", type=int, default=900)
parser.add_argument("--retain_steps", type=int, default=100, help="active steps to count as seated")
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--contact_thresh", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym, torch
from isaaclab.envs import DirectRLEnvCfg
import rl_rebuild.tasks.inhand_rotate
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.randomize_pd_gains = False; env_cfg.randomize_friction = False
    env_cfg.randomize_com = False; env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    N, dev = uenv.num_envs, uenv.device
    zero = torch.zeros(N, uenv.num_actions if hasattr(uenv, "num_actions") else 22, device=dev)

    started = torch.zeros(N, dtype=torch.bool, device=dev)
    age = torch.zeros(N, device=dev)          # active steps so far this episode
    total, seated, dropped_early = 0, 0, 0
    env.reset()
    with torch.no_grad():
        for t in range(args_cli.eval_steps):
            obs, r, done, info = env.step(zero)
            done = done.bool()
            active = (uenv._gx_settle == 0) if hasattr(uenv, "_gx_settle") else \
                torch.ones(N, dtype=torch.bool, device=dev)
            started |= active
            age += active.float()
            reached = age >= args_cli.retain_steps
            n_eng = (uenv.last_contacts > args_cli.contact_thresh).sum(-1)
            # count episodes as they END or as they REACH the retain mark (whichever first)
            if bool(reached.any()):
                m = reached & started
                seated += int(m.sum())          # survived K active steps under zero policy
                total += int(m.sum())
                age[m] = -1e9                   # count once
            if done.any():
                em = done & started & (age > 0)  # ended before reaching mark
                dropped_early += int(em.sum())
                total += int(em.sum())
                age[done] = 0; started[done] = False
    rate = seated / max(total, 1)
    print(f"[settle] episodes={total} seated(>= {args_cli.retain_steps} steps)={seated} "
          f"dropped_early={dropped_early} SEAT_RATE={rate:.3f}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
