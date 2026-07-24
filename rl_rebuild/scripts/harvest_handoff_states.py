# Harvest HANDOFF states (ZERO TRAINING; eval only) — D2 Stage 0 of
# notes/experiment_playbooks/regrip_contact_switch_designs_2026-07-01.md (robotics-rl-expert repo).
#
# Question: perturb-v1 reaches a sustained >=3-fingertip grasp of the smooth ball in only ~1.3% of
# episodes (d0a probe, 2026-07-01). Are those rare states PHYSICALLY STABLE grasps (=> the >=3-finger
# regrip target is viable and the problem is incentive), or transient flukes (=> physical ceiling)?
#
# This script rolls out a policy on its own task (default: AdjustHold-Perturb-v1, pinch replay reset,
# max perturbation at fixed -9.81) and, per episode, the FIRST time the d0a-style detector confirms
#   (a) the episode started pinch-like  (min #fingers over the first --early_steps active steps <= --pinch_max)
#   (b) n_engaged >= --enclose_min sustained for --sustain_steps consecutive active steps
# it snapshots that env's grasp as a 29-vector [22 hand-joint pos | 3 obj pos (env-rel) | 4 obj quat (wxyz)]
# — the exact cache format of cache/graspxl_enclosing_*.npy — into --cache_out, plus a sibling
# *_meta.npy (M,4): [n_engaged_at_capture, active_step_at_capture, env_id, episode_ordinal].
#
# Validate the produced cache with d0c_enclosing_grasp_validate.py --cache <cache_out> (zero-action /
# small-squeeze hold at full gravity). Harness mirrors d0a_pinch_regrasp_probe.py verbatim; the state
# read mirrors extract_adjusted_poses.py. No env/config/checkpoint is modified.
import argparse, os, sys, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustHold-Perturb-v1")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--eval_steps", type=int, default=2400)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--orient_cap", type=float, default=0.0)
# detector (same defaults as the d0a run that measured the 1.3%)
parser.add_argument("--contact_thresh", type=float, default=0.1, help="per-finger force (N) counted as contact")
parser.add_argument("--pinch_max", type=int, default=2, help="<= this many fingers in the early window = pinch start")
parser.add_argument("--enclose_min", type=int, default=3, help=">= this many fingers = target grasp")
parser.add_argument("--sustain_steps", type=int, default=10, help="target grasp must hold this many active steps")
parser.add_argument("--early_steps", type=int, default=20, help="active-step window that measures the pinch start")
parser.add_argument("--min_active", type=int, default=30, help="ignore episodes shorter than this (episode count)")
parser.add_argument("--cache_out", type=str, default=None,
                    help="output .npy (default: cache/graspxl_handoff_<objid>.npy)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
os.environ["SHARPA_ORIENT_CAP"] = str(args_cli.orient_cap)
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
    # deterministic harvest: DR off, fixed gravity, curriculum off (mirrors d0a / extract)
    env_cfg.randomize_pd_gains = False; env_cfg.randomize_friction = False
    env_cfg.randomize_com = False; env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/harvest_handoff", full_config=config, create_output_dir=False)
    objid = getattr(env_cfg, "graspxl_object_id", "obj")
    cache_out = args_cli.cache_out or f"cache/graspxl_handoff_{objid}.npy"
    meta_out = os.path.splitext(cache_out)[0] + "_meta.npy"
    print(f"[harvest] loading {args_cli.load_path}; task={args_cli.task}; gravity_z={args_cli.gravity_z}; "
          f"detector: pinch<={args_cli.pinch_max} (first {args_cli.early_steps} active) -> "
          f">={args_cli.enclose_min} fingers sustained {args_cli.sustain_steps}; -> {cache_out}")
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N = uenv.num_envs
    dev = uenv.device

    # per-env episode state (same semantics as d0a) -------------------------
    ep_act = torch.zeros(N, device=dev, dtype=torch.long)
    early_min = torch.full((N,), 99, device=dev, dtype=torch.long)
    enc_run = torch.zeros(N, device=dev, dtype=torch.long)
    harvested_ep = torch.zeros(N, device=dev, dtype=torch.bool)   # one snapshot per episode
    ep_ord = torch.zeros(N, device=dev, dtype=torch.long)          # episode ordinal per env

    n_episodes = 0            # completed episodes with active >= min_active
    n_pinch_eps = 0           # ...of which started pinch-like
    rows, meta = [], []

    obs = env.reset()
    with torch.no_grad():
        for t in range(args_cli.eval_steps):
            inp = {"obs": rms(obs["obs"]), "priv_info": obs["priv_info"]}
            if "pointcloud" in obs:
                inp["pointcloud"] = obs["pointcloud"]
            mu = torch.clamp(model.act_inference(inp), -1.0, 1.0)
            obs, r, done, info = env.step(mu)
            done = done.bool()

            settle = uenv._gx_settle
            active = settle == 0
            valid = active & (~done)
            n_fing = (uenv.last_contacts > args_cli.contact_thresh).sum(dim=1).long()
            ai = ep_act

            if valid.any():
                in_early = valid & (ai < args_cli.early_steps)
                early_min = torch.where(in_early & (n_fing < early_min), n_fing, early_min)
                is_enc = valid & (n_fing >= args_cli.enclose_min)
                enc_run = torch.where(is_enc, enc_run + 1, torch.where(valid, torch.zeros_like(enc_run), enc_run))

                # ---- harvest: first sustained crossing, pinch-start episodes only ----
                fire = (valid & (~harvested_ep) & (enc_run >= args_cli.sustain_steps)
                        & (ai >= args_cli.early_steps) & (early_min <= args_cli.pinch_max))
                if fire.any():
                    if hasattr(uenv, "_refresh_lab"):
                        uenv._refresh_lab()
                    dof = uenv.hand.data.joint_pos
                    if dof.shape[1] != 22:
                        dof = dof[:, uenv.actuated_dof_indices]
                    state = torch.cat([dof, uenv.object_pos, uenv.object_rot], dim=-1)  # (N,29)
                    ids = fire.nonzero(as_tuple=False).squeeze(-1)
                    rows.append(state[ids].detach().cpu().clone())
                    meta.append(torch.stack([n_fing[ids].float(), ai[ids].float(),
                                             ids.float(), ep_ord[ids].float()], dim=-1).cpu().clone())
                    harvested_ep[ids] = True

                ep_act = ep_act + valid.long()

            if done.any():
                fin = done.nonzero(as_tuple=False).squeeze(-1)
                long_enough = ep_act[fin] >= args_cli.min_active
                n_episodes += int(long_enough.sum())
                n_pinch_eps += int((long_enough & (early_min[fin] <= args_cli.pinch_max)).sum())
                ep_act[done] = 0; early_min[done] = 99; enc_run[done] = 0
                harvested_ep[done] = False; ep_ord[done] += 1

            if (t + 1) % 400 == 0:
                print(f"[harvest] step {t+1}/{args_cli.eval_steps}: episodes~{n_episodes}, harvested={sum(r.shape[0] for r in rows)}")

    M = sum(r.shape[0] for r in rows)
    print("\n==== HANDOFF-STATE HARVEST ====")
    print(f"task={args_cli.task}  ckpt={args_cli.load_path}")
    print(f"episodes completed (active>={args_cli.min_active}): {n_episodes}  (pinch-start: {n_pinch_eps})")
    print(f"HARVESTED sustained >= {args_cli.enclose_min}-finger states: {M}  "
          f"(yield {100.0*M/max(n_pinch_eps,1):.2f}% of pinch-start episodes)")
    if M > 0:
        cache_np = torch.cat(rows, 0).numpy().astype(np.float32)
        meta_np = torch.cat(meta, 0).numpy().astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(cache_out)), exist_ok=True)
        np.save(cache_out, cache_np); np.save(meta_out, meta_np)
        n_at = meta_np[:, 0]
        print(f"saved {cache_np.shape} -> {os.path.abspath(cache_out)}  (+ meta {meta_np.shape})")
        print(f"n_engaged at capture: mean={n_at.mean():.2f}  hist " +
              " ".join(f"{k}:{int((n_at==k).sum())}" for k in range(6)))
        print(f"capture active-step: mean={meta_np[:,1].mean():.0f}  p10={np.percentile(meta_np[:,1],10):.0f}  "
              f"p90={np.percentile(meta_np[:,1],90):.0f}")
        print(f"NEXT: validate with d0c_enclosing_grasp_validate.py --cache {cache_out} (squeeze 0 and ~0.15)")
    else:
        print("NO states harvested -> at this yield the >=3-finger target is not reachable by this policy; "
              "the D2 Stage-B decomposition has no seed and the verdict falls to the training designs (D1/D3/D4).")
    print("===============================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
