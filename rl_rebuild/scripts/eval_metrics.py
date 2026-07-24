# Bounded evaluation: load a checkpoint, run N steps, report rotation/hold metrics.
# Based on play.py but exits after --eval_steps instead of looping forever.

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate an RL agent (bounded).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache", type=str, default=None)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--eval_steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import os
import torch

from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.algo.padapt.padapt import ProprioAdapt
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper

from isaaclab.envs import DirectRLEnvCfg
import rl_rebuild.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    # eval config: full gravity, no DR (mirrors play.py)
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, -9.81)
    env_cfg.gravity_curriculum = False
    env_cfg.grasp_cache_path = args_cli.cache if args_cli.cache is not None else env_cfg.grasp_cache_path
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    log_dir = os.path.abspath(os.path.join("logs", "eval_tmp"))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config, create_output_dir=False)
    print(f"[INFO]: Loading checkpoint from: {agent_cfg['load_path']}")
    agent.restore_test(agent_cfg["load_path"])
    agent.set_eval()

    rms = agent.running_mean_std
    model = agent.model
    is_padapt = agent_cfg["algo"] == "ProprioAdapt"
    sa = getattr(agent, "sa_mean_std", None)

    obs_dict = env.reset()
    N = args_cli.eval_steps
    sum_yaw = 0.0
    sum_abs_yaw = 0.0
    sum_drop = 0.0
    sum_rotate_reward = 0.0
    sum_reset_upper = 0.0
    n_dones = 0
    with torch.no_grad():
        for t in range(N):
            if is_padapt:
                input_dict = {"obs": rms(obs_dict["obs"]), "proprio_hist": sa(obs_dict["proprio_hist"])}
            else:
                input_dict = {"obs": rms(obs_dict["obs"]), "priv_info": obs_dict["priv_info"]}
            if "pointcloud" in obs_dict:
                input_dict["pointcloud"] = obs_dict["pointcloud"]
            mu = model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = env.step(mu)
            sum_yaw += float(info.get("yaw", 0.0))
            sum_abs_yaw += abs(float(info.get("yaw", 0.0)))
            sum_rotate_reward += float(info.get("rotate_reward", 0.0))
            sum_reset_upper += float(info.get("height_reset_upper", 0.0))
            sum_drop += float(info.get("height_reset_lower", 0.0))
            n_dones += int(done.sum().item())
            if (t + 1) % 100 == 0:
                print(f"step {t+1}/{N}: yaw_angvel={info.get('yaw',0):.4f} "
                      f"rotate_rew={info.get('rotate_reward',0):.4f} "
                      f"drop_frac={info.get('height_reset_lower',0):.5f} "
                      f"fly_frac={info.get('height_reset_upper',0):.5f}")

    print("\n==== EVAL SUMMARY ====")
    print(f"task={args_cli.task} num_envs={args_cli.num_envs} steps={N} ckpt={agent_cfg['load_path']}")
    print(f"mean yaw angvel (about z, rad/s):       {sum_yaw / N:.4f}")
    print(f"mean |yaw| angvel (rad/s):              {sum_abs_yaw / N:.4f}")
    print(f"mean rotate_reward (clipped angvel.z):  {sum_rotate_reward / N:.4f}")
    print(f"mean per-step drop frac (below floor):  {sum_drop / N:.5f}")
    print(f"mean per-step fly frac  (above ceil):   {sum_reset_upper / N:.5f}")
    print(f"total resets over {N} steps x {args_cli.num_envs} envs: {n_dones}")
    print("======================")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
