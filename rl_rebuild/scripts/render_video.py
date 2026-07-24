# Render a trained policy to an mp4 (close-up viewer). Bounded; exits after --video_length steps.
import argparse, sys, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render an RL agent to video.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--gravity_z", type=float, default=-9.81, help="gravity (eval at the policy's trained g)")
parser.add_argument("--settle_skip", type=int, default=0, help="run this many warmup steps before recording")
parser.add_argument("--out_dir", type=str, default="videos")
parser.add_argument("--eye", type=str, default="0.45,0.45,0.78", help="camera eye (env-relative), comma-sep")
parser.add_argument("--lookat", type=str, default="-0.05,0.0,0.60", help="camera lookat (env-relative), comma-sep")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.algo.padapt.padapt import ProprioAdapt
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
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
    # eval config: full gravity, no DR (mirror play.py)
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z)
    env_cfg.gravity_curriculum = False
    # close-up viewer framed on env 0 (the cylinder lives at ~z=0.62, hand wrist ~z=0.5)
    _eye = tuple(float(x) for x in args_cli.eye.split(","))
    _lookat = tuple(float(x) for x in args_cli.lookat.split(","))
    env_cfg.viewer = ViewerCfg(eye=_eye, lookat=_lookat,
                               origin_type="env", env_index=0, resolution=(720, 540))
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base_env = gym.wrappers.RecordVideo(
        base_env, video_folder=args_cli.out_dir, name_prefix=f"render_{args_cli.task}",
        step_trigger=lambda s: s == 0, video_length=args_cli.video_length, disable_logger=True,
    )
    env = GymStyleEnvWrapper(base_env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/render_tmp", full_config=config, create_output_dir=False)
    print(f"[INFO]: Loading checkpoint from: {agent_cfg['load_path']}")
    agent.restore_test(agent_cfg["load_path"])
    agent.set_eval()

    rms = agent.running_mean_std
    model = agent.model
    is_padapt = agent_cfg["algo"] == "ProprioAdapt"
    sa = getattr(agent, "sa_mean_std", None)

    obs_dict = env.reset()
    total = args_cli.settle_skip + args_cli.video_length + 5
    with torch.no_grad():
        for t in range(total):
            if is_padapt:
                input_dict = {"obs": rms(obs_dict["obs"]), "proprio_hist": sa(obs_dict["proprio_hist"])}
            else:
                input_dict = {"obs": rms(obs_dict["obs"]), "priv_info": obs_dict["priv_info"]}
            if "pointcloud" in obs_dict:
                input_dict["pointcloud"] = obs_dict["pointcloud"]
            mu = model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = env.step(mu)
    env.close()
    print(f"[INFO] video written under: {os.path.abspath(args_cli.out_dir)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
