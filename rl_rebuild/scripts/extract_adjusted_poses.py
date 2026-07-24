# Extract + render LEARNED ADJUSTED GRASPS from an adjustment-only policy (eval only; no training).
#
# Given a checkpoint trained on the ADJUSTMENT-ONLY task (readiness-Phi reward, NO rotation reward, e.g.
# Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustOnly-FixedG-v0), this script:
#   (a) rolls the policy out FROM THE PINCH (the task's default GraspXL replay reset) and, at the end of the
#       adjustment horizon, saves each env's FINAL grasp state as a 29-vector
#         [ 22 hand-joint pos (articulation order) | 3 object pos (env-relative) | 4 object quat (wxyz) ]
#       -- the SAME format as cache/graspxl_enclosing_*.npy -- to a .npy cache (--num_envs states).
#   (b) renders --num_poses of those final grasps to PNG images under --out_dir (each env's real achieved
#       grasp is transplanted into env 0, which the viewer frames, then rendered).
#   (c) optionally (--write_mp4) writes a short mp4 of ONE adjustment rollout (env 0, pinch -> adjusted).
#   Also computes + prints each saved pose's rotation-readiness Phi (reuses the env's _readiness_phi()).
#
# Harness (app launch, hydra cfg, env make, PPO restore, running-mean-std, act_inference) mirrors
# d0a_pinch_regrasp_probe.py / render_video.py; the (N,29) state read + Phi mirror d0c_enclosing_grasp_validate.py.
# No env/config/checkpoint is modified. The produced cache is meant to feed the rotation-from-cache probe
# (see the header of rl_rebuild/scripts/d0c_enclosing_grasp_validate.py and the AdjustRotate-Diverse cfg's
# enclosing_cache_path).
import argparse, os, sys, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Extract + render learned adjusted grasps from an adjustment-only policy.")
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustOnly-FixedG-v0")
parser.add_argument("--load_path", type=str, required=True, help="adjustment-only checkpoint (.pth)")
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--seed", type=int, default=123)
# --- cache (K = number of saved states) / outputs ---
parser.add_argument("--num_envs", type=int, default=256, help="K: number of final grasps saved to the cache")
parser.add_argument("--num_poses", type=int, default=5, help="how many of the saved grasps to render as PNG")
parser.add_argument("--cache_out", type=str, default=None,
                    help="output .npy path for the (K,29) adjusted-grasp cache "
                         "(default: cache/graspxl_adjusted_<objid>.npy)")
parser.add_argument("--out_dir", type=str, default="results/adjusted_poses", help="dir for PNG stills (+ mp4)")
# --- rollout / physics ---
parser.add_argument("--active_steps", type=int, default=200,
                    help="control steps to run AFTER the replay settle finishes (the adjustment horizon)")
parser.add_argument("--gravity_z", type=float, default=-4.9, help="fixed gravity during extraction (match training)")
parser.add_argument("--orient_cap", type=float, default=-1.0,
                    help="if >=0, force the SE(3) orientation cap (rad) via SHARPA_ORIENT_CAP; <0 = leave the "
                         "task default (the trained cap; AdjustOnly-FixedG stays at graspxl_orient_start~0.3)")
parser.add_argument("--contact_thresh", type=float, default=0.1, help="per-finger force (N) counted as engaged (Phi)")
# --- optional video ---
parser.add_argument("--write_mp4", action="store_true", help="also write a short mp4 of env 0's adjustment rollout")
parser.add_argument("--video_fps", type=int, default=20, help="mp4 fps (control rate ~20 Hz at dt=1/240, dec=12)")
parser.add_argument("--eye", type=str, default="0.45,0.45,0.78", help="camera eye (env-relative), comma-sep")
parser.add_argument("--lookat", type=str, default="-0.05,0.0,0.60", help="camera lookat (env-relative), comma-sep")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.orient_cap >= 0.0:
    os.environ["SHARPA_ORIENT_CAP"] = str(args_cli.orient_cap)
args_cli.enable_cameras = True   # required for rgb_array render (PNG + mp4)
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import gymnasium as gym, torch
import numpy as np
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
    env_cfg.seed = args_cli.seed
    agent_cfg["seed"] = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = env_cfg.sim.device
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path
    # deterministic extraction: DR off, fixed gravity, curriculum off (we want the policy's learned grasp,
    # not a robustness sweep). Mirrors d0a/d0c.
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0.0, 0.0, args_cli.gravity_z)
    env_cfg.gravity_curriculum = False
    # close-up viewer framed on env 0 (same framing as render_video.py)
    _eye = tuple(float(x) for x in args_cli.eye.split(","))
    _lookat = tuple(float(x) for x in args_cli.lookat.split(","))
    env_cfg.viewer = ViewerCfg(eye=_eye, lookat=_lookat, origin_type="env", env_index=0, resolution=(720, 540))
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    objid = getattr(env_cfg, "graspxl_object_id", "obj")
    cache_out = args_cli.cache_out or f"cache/graspxl_adjusted_{objid}.npy"
    os.makedirs(args_cli.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(cache_out)), exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped
    dev = uenv.device
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/extract_tmp", full_config=config, create_output_dir=False)
    print(f"[extract] loading {args_cli.load_path}; task={args_cli.task}; gravity_z={args_cli.gravity_z}; "
          f"K(num_envs)={args_cli.num_envs}; num_poses={args_cli.num_poses}")
    agent.restore_test(args_cli.load_path)
    agent.set_eval()
    rms, model = agent.running_mean_std, agent.model

    N = uenv.num_envs
    assert args_cli.num_poses <= N, f"--num_poses ({args_cli.num_poses}) must be <= --num_envs ({N})"
    settle_total = int(getattr(uenv, "_gx_settle_total", 0))     # replay settle length (0 if cache-mode)
    total_steps = settle_total + int(args_cli.active_steps)
    orient_cur = float(getattr(uenv, "_gx_orient_cur", 0.0))
    print(f"[extract] settle_total={settle_total}; active_steps={args_cli.active_steps}; "
          f"total_rollout_steps={total_steps}; orient_cur={orient_cur:.3f} rad")

    frames = []  # env-0 frames for the optional mp4
    obs = env.reset()
    with torch.no_grad():
        for t in range(total_steps):
            inp = {"obs": rms(obs["obs"]), "priv_info": obs["priv_info"]}
            if "pointcloud" in obs:
                inp["pointcloud"] = obs["pointcloud"]
            mu = torch.clamp(model.act_inference(inp), -1.0, 1.0)
            obs, r, done, info = env.step(mu)
            if args_cli.write_mp4:
                frames.append(uenv.render())

    # ---- read the FINAL grasp state of every env (29-vec, same layout as cache/graspxl_enclosing_*.npy) ----
    uenv._refresh_lab()
    dof = uenv.hand.data.joint_pos.clone()                       # (N, num_joints) articulation order
    if dof.shape[1] != 22:
        dof = dof[:, uenv.actuated_dof_indices]                  # defensive: keep the 22 actuated joints
    assert dof.shape[1] == 22, f"expected 22 hand joints for the cache, got {dof.shape[1]}"
    obj_pos_rel = uenv.object_pos.clone()                        # (N,3) env-relative
    obj_quat = uenv.object_rot.clone()                           # (N,4) wxyz
    final = torch.cat([dof, obj_pos_rel, obj_quat], dim=-1)      # (N,29)

    # readiness Phi of the achieved grasps (ACTIVE metric: uses the live final grip) -- reuse env formula
    if hasattr(uenv, "_readiness_phi"):
        phi = uenv._readiness_phi()                              # (N,)
    else:
        phi = _readiness_phi_fallback(uenv, args_cli.contact_thresh)
    n_eng = (uenv.last_contacts > args_cli.contact_thresh).float().sum(-1)  # #engaged fingers per env

    # keep only ACTIVE envs (replay settle finished, i.e. under RL control) so every cached row is a genuine
    # learned adjusted grasp -- NOT a fresh pinch from an env that just drop-reset, and not a settling frame.
    settle = getattr(uenv, "_gx_settle", None)
    active = (settle == 0) if settle is not None else torch.ones(N, dtype=torch.bool, device=dev)
    sel = active.nonzero(as_tuple=False).squeeze(-1)
    if sel.numel() == 0:
        print("[extract] WARNING: no active envs at capture -> saving ALL envs (increase --active_steps)")
        sel = torch.arange(N, device=dev)
    n_poses = min(args_cli.num_poses, int(sel.numel()))

    cache_np = final[sel].detach().cpu().numpy().astype(np.float32)
    np.save(cache_out, cache_np)
    phi_c = phi[sel].detach().cpu()
    n_eng_c = n_eng[sel].detach().cpu()
    print(f"\n==== ADJUSTED-GRASP CACHE SAVED ====")
    print(f"  {cache_np.shape} -> {os.path.abspath(cache_out)}   (active {int(sel.numel())}/{N} envs kept)")
    print(f"  format: [22 hand-joint pos | 3 obj pos (env-rel) | 4 obj quat] (matches graspxl_enclosing_*.npy)")
    print(f"  readiness Phi over the {int(sel.numel())} saved grasps: mean={phi_c.mean():.3f}  "
          f"median={phi_c.median():.3f}  p10={phi_c.quantile(0.1):.3f}  p90={phi_c.quantile(0.9):.3f}")
    print(f"  #engaged fingers (>{args_cli.contact_thresh}N): mean={n_eng_c.mean().item():.2f}")
    print(f"  per-pose Phi (first {n_poses}): " +
          "  ".join(f"[{j}] Phi={float(phi_c[j]):.3f} n={int(n_eng_c[j].item())}" for j in range(n_poses)))

    # ---- capture the full state of each env so we can transplant it into env 0 for a correct render ----
    hand_root = uenv.hand.data.root_state_w.clone()             # (N,13) world [pos, quat, linvel, angvel]
    joint_pos_all = uenv.hand.data.joint_pos.clone()
    origins = uenv.scene.env_origins                            # (N,3)
    env0 = torch.tensor([0], device=dev, dtype=torch.long)

    print(f"\n==== RENDERING {n_poses} ADJUSTED POSES -> {os.path.abspath(args_cli.out_dir)} ====")
    for j in range(n_poses):
        e = int(sel[j].item())                                     # a genuine active env
        # transplant env e's exact final state (hand root + joints + object) into env 0 (the framed env)
        hs = hand_root[e:e + 1].clone()
        hs[:, 0:3] = hs[:, 0:3] - origins[e:e + 1] + origins[0:1]   # re-seat to env-0 origin
        hs[:, 7:13] = 0.0                                            # zero velocities
        uenv.hand.write_root_state_to_sim(hs, env0)
        jp = joint_pos_all[e:e + 1]
        uenv.hand.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env0)
        uenv.hand.set_joint_position_target(jp, env_ids=env0)
        uenv.prev_targets[0] = jp[0]
        uenv.cur_targets[0] = jp[0]
        root = torch.zeros(1, 7, device=dev)
        root[:, :3] = obj_pos_rel[e:e + 1] + origins[0:1]
        root[:, 3:7] = obj_quat[e:e + 1]
        uenv.object.write_root_pose_to_sim(root, env0)
        uenv.object.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env0)
        uenv.sim.forward()
        uenv.render()                                               # warm the annotator (first frame may be blank)
        frame = uenv.render()
        png = os.path.join(args_cli.out_dir, f"adjusted_pose_{j:02d}_phi{float(phi_c[j]):.2f}.png")
        imageio.imwrite(png, frame)
        print(f"  [{j}] env={e} Phi={float(phi_c[j]):.3f}  ->  {png}")

    # ---- optional mp4 of the single adjustment rollout (env 0) ----
    if args_cli.write_mp4 and frames:
        mp4 = os.path.join(args_cli.out_dir, f"adjust_rollout_{objid}.mp4")
        imageio.mimwrite(mp4, [f for f in frames if f is not None and f.size > 0], fps=args_cli.video_fps)
        print(f"\n[extract] adjustment rollout mp4 ({len(frames)} frames @ {args_cli.video_fps} fps) -> "
              f"{os.path.abspath(mp4)}")

    print("\n[extract] DONE. Feed the cache to the rotation-from-cache probe to rotate FROM the learned "
          "adjusted grasps (vs FROM the pinch). See this repo's extract/rotate command templates.")
    env.close()


def _readiness_phi_fallback(uenv, thresh):
    """Exact copy of SharpaWaveGraspXLAdjustRotateEnv._readiness_phi, in case the loaded env lacks it."""
    center = uenv.object.data.root_pos_w.unsqueeze(1)
    tips = uenv.hand.data.body_link_state_w[:, uenv.elastomer_ids, :3]
    rel = tips - center
    k = uenv.rot_axis
    k = k.view(1, 1, 3) if k.dim() == 1 else k.unsqueeze(1)
    rel_perp = rel - (rel * k).sum(-1, keepdim=True) * k
    d = rel_perp / rel_perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    eng = (uenv.last_contacts > thresh).float()
    n = eng.sum(-1)
    coverage = ((n - 1.0) / 4.0).clamp(0.0, 1.0)
    mean_d = (d * eng.unsqueeze(-1)).sum(1) / n.clamp_min(1.0).unsqueeze(-1)
    spread = torch.where(n >= 2.0, 1.0 - mean_d.norm(dim=-1), torch.zeros_like(n))
    return 0.5 * spread + 0.5 * coverage


if __name__ == "__main__":
    main()
    app.close()
