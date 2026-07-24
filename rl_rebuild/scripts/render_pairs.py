# Render CONSISTENT (pose, clip) PAIRS for a policy (ZERO TRAINING; eval only).
# User spec 2026-07-02: "final pose + how that behaves under normal gravity, 3 pairs per reward".
# Each pair k is ONE env-0 episode under fixed eval conditions:
#   pairK_clip.mp4  — the episode's ACTIVE phase (settle trimmed), 20 fps
#   pairK_pose.png  — the LAST active frame of that same episode (the clip's final frame)
# so the pose and the clip can never disagree (same env, same rollout, same instant).
# Conditions: --gravity_z (default -9.81), SHARPA_EVAL_CLEAN=1 recommended (mechanisms off),
# DR off, canonical palm-up reset (SHARPA_ORIENT_CAP=0), deterministic policy (mean action).
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--pairs", type=int, default=3)
parser.add_argument("--max_steps", type=int, default=1600)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--orient_cap", type=float, default=0.0)
parser.add_argument("--min_frames", type=int, default=15, help="segments shorter than this are skipped")
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--eye", type=str, default="0.45,0.45,0.78")
parser.add_argument("--lookat", type=str, default="-0.05,0.0,0.60")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
os.environ["SHARPA_ORIENT_CAP"] = str(args_cli.orient_cap)
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym, torch
import imageio.v2 as imageio

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
    env_cfg.seed = args_cli.seed; agent_cfg["seed"] = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = env_cfg.sim.device
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path
    env_cfg.randomize_pd_gains = False; env_cfg.randomize_friction = False
    env_cfg.randomize_com = False; env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0.0, 0.0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    if hasattr(env_cfg, "rgv2_gravity_curriculum"):
        env_cfg.rgv2_gravity_curriculum = False
    _eye = tuple(float(x) for x in args_cli.eye.split(","))
    _lookat = tuple(float(x) for x in args_cli.lookat.split(","))
    env_cfg.viewer = ViewerCfg(eye=_eye, lookat=_lookat, origin_type="env", env_index=0, resolution=(720, 540))
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    os.makedirs(args_cli.out_dir, exist_ok=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/render_pairs", full_config=config, create_output_dir=False)
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    print(f"[pairs] task={args_cli.task} ckpt={args_cli.load_path} gravity={args_cli.gravity_z} "
          f"clean={os.environ.get('SHARPA_EVAL_CLEAN','0')} pairs={args_cli.pairs}")

    obs = env.reset()
    frames, saved = [], 0
    with torch.no_grad():
        for t in range(args_cli.max_steps):
            inp = {"obs": rms(obs["obs"]), "priv_info": obs["priv_info"]}
            if "pointcloud" in obs:
                inp["pointcloud"] = obs["pointcloud"]
            mu = torch.clamp(model.act_inference(inp), -1.0, 1.0)
            obs, r, done, info = env.step(mu)
            settle0 = int(uenv._gx_settle[0]) if hasattr(uenv, "_gx_settle") else 0
            if settle0 == 0:
                frames.append(uenv.render())
            if bool(done[0]):
                if len(frames) >= args_cli.min_frames:
                    saved += 1
                    clip = os.path.join(args_cli.out_dir, f"pair{saved}_clip.mp4")
                    pose = os.path.join(args_cli.out_dir, f"pair{saved}_pose.png")
                    imageio.mimwrite(clip, frames, fps=args_cli.fps)
                    imageio.imwrite(pose, frames[-1])
                    print(f"[pairs] pair{saved}: {len(frames)} active frames "
                          f"({len(frames)/args_cli.fps:.1f}s) -> {clip} + pose")
                else:
                    print(f"[pairs] skipped a {len(frames)}-frame segment (<{args_cli.min_frames})")
                frames = []
                if saved >= args_cli.pairs:
                    break
    if saved < args_cli.pairs and len(frames) >= args_cli.min_frames:
        saved += 1
        imageio.mimwrite(os.path.join(args_cli.out_dir, f"pair{saved}_clip.mp4"), frames, fps=args_cli.fps)
        imageio.imwrite(os.path.join(args_cli.out_dir, f"pair{saved}_pose.png"), frames[-1])
        print(f"[pairs] pair{saved}: {len(frames)} active frames (timeout-truncated)")
    print(f"[pairs] DONE: {saved}/{args_cli.pairs} pairs -> {os.path.abspath(args_cli.out_dir)}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
