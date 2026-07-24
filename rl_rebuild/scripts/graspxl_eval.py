# Settle-aware eval for the GraspXL env: measures held + rotation over the ACTIVE phase only
# (envs whose replay settle has finished), at full gravity.
import argparse, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-v0")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--eval_steps", type=int, default=900)
parser.add_argument("--gravity_z", type=float, default=-9.81, help="eval gravity (z). default full gravity")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym, torch
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
    env_cfg.seed = args_cli.seed; agent_cfg["seed"] = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = env_cfg.sim.device
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path
    # eval: full gravity, no DR
    env_cfg.randomize_pd_gains = False; env_cfg.randomize_friction = True
    env_cfg.randomize_com = False; env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/gxeval", full_config=config, create_output_dir=False)
    print(f"[INFO] loading {args_cli.load_path}")
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model

    obs = env.reset()
    orient = bool(getattr(uenv, "_gx_orient", False))
    drop_disp = float(getattr(uenv, "_gx_drop_disp", 0.10))
    n_act = 0; sum_yaw = 0.0; sum_held = 0.0
    with torch.no_grad():
        for t in range(args_cli.eval_steps):
            inp = {"obs": rms(obs["obs"]), "priv_info": obs["priv_info"]}
            if "pointcloud" in obs:
                inp["pointcloud"] = obs["pointcloud"]
            mu = torch.clamp(model.act_inference(inp), -1.0, 1.0)
            obs, r, done, info = env.step(mu)
            active = (uenv._gx_settle == 0) if hasattr(uenv, "_gx_settle") else torch.ones(uenv.num_envs, dtype=torch.bool, device=uenv.device)
            # rotation about the (possibly rotated) reward axis = object angular velocity . rot_axis
            yaw = (uenv.object.data.root_ang_vel_w * uenv.rot_axis).sum(-1)
            objpos = uenv.object.data.root_pos_w - uenv.scene.env_origins
            if orient:  # orientation-invariant: held = small displacement from the (rotated) grasp pose
                held = torch.norm(objpos - uenv.object_default_pose[:, :3], dim=-1) < drop_disp
            else:
                held = objpos[:, 2] > uenv.reset_height_lower
            a = active.float()
            denom = a.sum().clamp_min(1.0)
            sum_yaw += float((yaw.abs() * a).sum() / denom)
            sum_held += float((held.float() * a).sum() / denom)
            n_act += 1
    print("\n==== GraspXL ACTIVE-PHASE EVAL (full gravity) ====")
    print(f"task={args_cli.task} ckpt={args_cli.load_path} steps={n_act}")
    print(f"held fraction (active phase):     {sum_held / n_act:.4f}")
    print(f"mean |yaw| rate about +Z (rad/s): {sum_yaw / n_act:.4f}")
    print("==================================================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
