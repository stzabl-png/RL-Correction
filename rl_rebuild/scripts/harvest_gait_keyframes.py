# Harvest a MULTI-KEYFRAME reset cache from the trained CYLINDER band-rotation policy.
#
# WHY: the finger-gaiting sweep (sharpa_wave_graspxl_bandgait.py) wants resets that span the WHOLE rotation
# cycle, not just the single canonical pinch in cache/sharpa_grasp_linspace_0.5-0.5-1.npy. We roll out the
# BandRot cylinder policy and, every K steps, snapshot each (held) env's FULL grasp state into the SAME
# 29-column format the cylinder grasp-cache loader expects, so the BandGait env can load it unmodified.
#
# CACHE FORMAT (must match sharpa_wave_env.py):
#   loader     :117  np.load(f"{grasp_cache_path}_{scale_range0}-{scale_range1}-{scale_range2}.npy")
#              -> cylinder: cache/sharpa_grasp_linspace_0.5-0.5-1.npy  shape (50000, 29) float32
#   reset use  :360-406  row = [ hand_dof_pos[:22] , object_pos_local[22:25] , object_quat_wxyz[25:29] ]
#       dof_pos            = row[:, :22]     -> hand.write_joint_state_to_sim (ALL 22 joints, articulation order)
#       object_pos (local) = row[:, 22:25]  -> + scene.env_origins  (so we store root_pos_w - env_origins)
#       object_quat (wxyz) = row[:, 25:29]  -> object root quat
#   With scale_range[2]==1: bucket_grasp == n_rows, bucket_env == num_envs; each env samples ONE random row.
#
# OUTPUT: cache/cylinder_gait_keyframes_0.5-0.5-1.npy  (base path 'cache/cylinder_gait_keyframes' matches
#   SharpaWaveBandGaitCylinderCfg.grasp_cache_path; the loader re-appends the '_a-b-c.npy' scale suffix).
#
# Harness (app launch, hydra cfg, env make, PPO restore, running-mean-std, act_inference rollout) is copied
# from scripts/contact_switch_probe.py / orient_diag_eval.py. This script ONLY reads the env; it writes a .npy.
#
# RUN (GPU; do NOT run on the CPU verify box):
#   SHARPA_WANDB=0 <sharpa-python> rl_rebuild/scripts/harvest_gait_keyframes.py \
#       --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-Cylinder-v1 \
#       --load_path logs/debug/2026-07-01_04-29-45/stage1_nn/last.pth \
#       --num_envs 128 --eval_steps 600 --snapshot_every 40 --gravity_z -9.81 --headless
import argparse, os, sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Harvest a multi-keyframe reset cache from the band-rotation policy.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-Cylinder-v1")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--load_path", type=str, default="logs/debug/2026-07-01_04-29-45/stage1_nn/last.pth")
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--eval_steps", type=int, default=600)
parser.add_argument("--snapshot_every", type=int, default=40, help="K: snapshot every K steps")
parser.add_argument("--warmup", type=int, default=40, help="skip snapshots before this step (let the roll get going)")
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--held_disp", type=float, default=0.04, help="only snapshot envs whose object is within this "
                                                                  "displacement of its default pose (well held)")
parser.add_argument("--target_states", type=int, default=2000, help="if more valid states are collected, "
                                                                    "randomly subsample down to this many")
parser.add_argument("--out_base", type=str, default="cache/cylinder_gait_keyframes",
                    help="base path; the '_a-b-c.npy' scale suffix is appended to match the loader")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym, torch
import numpy as np
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
    # eval config: fixed gravity, no PD/com/mass DR (mirror the diag/probe scripts); keep friction DR
    env_cfg.randomize_pd_gains = False; env_cfg.randomize_friction = True
    env_cfg.randomize_com = False; env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/harvest", full_config=config, create_output_dir=False)
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N = uenv.num_envs
    n_dof = uenv.hand.data.joint_pos.shape[1]
    sr = uenv.cfg.scale_range
    out_path = f"{args_cli.out_base}_{sr[0]}-{sr[1]}-{sr[2]}.npy"
    print(f"[HARVEST] task={args_cli.task} ckpt={args_cli.load_path} num_envs={N} n_dof={n_dof} "
          f"gravity_z={args_cli.gravity_z} K={args_cli.snapshot_every} -> {out_path}")
    assert n_dof == 22, f"expected 22 hand joints (29-col cache: 22 qpos + 3 pos + 4 quat); got {n_dof}"

    rows = []                                            # collected (29,) keyframes
    obs = env.reset()
    with torch.no_grad():
        for t in range(args_cli.eval_steps):
            inp = {"obs": rms(obs["obs"]), "priv_info": obs["priv_info"]}
            if "pointcloud" in obs:
                inp["pointcloud"] = obs["pointcloud"]
            mu = torch.clamp(model.act_inference(inp), -1.0, 1.0)
            obs, r, done, info = env.step(mu)
            done = done.bool()
            take = (t >= args_cli.warmup) and (t % args_cli.snapshot_every == 0)
            if not take:
                continue
            # FULL grasp state in the loader's 29-col layout, env-LOCAL object pose
            qpos = uenv.hand.data.joint_pos.clone()                              # (N,22) articulation order
            obj_pos = (uenv.object.data.root_pos_w - uenv.scene.env_origins).clone()  # (N,3) env-local
            obj_quat = uenv.object.data.root_quat_w.clone()                       # (N,4) wxyz
            state = torch.cat([qpos, obj_pos, obj_quat], dim=-1)                  # (N,29)
            # keep only WELL-HELD, non-reset envs (skip dropped / mid-teleport states)
            disp = torch.norm(obj_pos - uenv.object_default_pose[:, :3], dim=-1)
            keep = (disp < args_cli.held_disp) & (~done)
            k = keep.nonzero(as_tuple=False).squeeze(-1)
            if k.numel() > 0:
                rows.append(state[k].cpu().numpy())
            print(f"[HARVEST] step {t:4d}: snapshot kept {int(keep.sum())}/{N} held  "
                  f"(total so far {sum(x.shape[0] for x in rows)})")

    env.close()
    if not rows:
        raise RuntimeError("harvested 0 keyframes -- policy may be dropping the object; check load_path/gravity")
    arr = np.concatenate(rows, axis=0).astype(np.float32)                         # (M,29)
    if arr.shape[0] > args_cli.target_states:
        idx = np.random.default_rng(args_cli.seed).choice(arr.shape[0], size=args_cli.target_states, replace=False)
        arr = arr[idx]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr)
    print(f"[HARVEST] wrote {arr.shape} float32 -> {out_path}  "
          f"(bucket_grasp={arr.shape[0]} with scale_range[2]={sr[2]}; each env samples one row at reset)")


if __name__ == "__main__":
    main()
    app.close()
