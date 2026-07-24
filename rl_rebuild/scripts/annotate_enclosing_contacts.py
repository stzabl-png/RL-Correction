# Annotate the enclosing grasp cache with per-row per-finger CONTACT IDENTITY vectors (ZERO TRAINING).
# One-time prerequisite for the R-D3 GOALGRASP regrip variant (sharpa_wave_graspxl_goalgrasp.py).
#
# For every row of cache/graspxl_enclosing_<obj>.npy (29-vec = 22 dof + 3 obj pos env-rel + 4 quat wxyz):
# write the state into a batch of envs (d0c_enclosing_grasp_validate.py's manual-drive pattern verbatim:
# rewards/dones neutered, real env.step for correct contact sensing), hold with a small active flexion
# squeeze for --settle_steps at full gravity, then record each elastomer's MAJORITY-VOTE engagement over
# the last 10 steps -> (M,5) {0,1} saved to <cache>_contacts.npy.
#
# Also prints calibration stats for the GOALGRASP reward: the L1 joint-distance distribution from the
# perturb-final poses (238-row proxy for the post-regrip hand pose) to the >=3-finger goal rows, with
# suggested gg_sigma_q / gg_eps_q values. No env/config/cache-input is modified.
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-GraspXL-v0")
parser.add_argument("--cache", type=str, default="cache/graspxl_enclosing_002aa1853c974f3a9565e85f5e09a515.npy")
parser.add_argument("--ref_cache", type=str, default="cache/graspxl_adjusted_perturb_002aa1853c974f3a9565e85f5e09a515.npy",
                    help="(K,29) reference poses for the distance calibration printout")
parser.add_argument("--out", type=str, default=None, help="output .npy (default: <cache>_contacts.npy)")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--settle_steps", type=int, default=40)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--squeeze", type=float, default=0.15, help="rad of active flexion squeeze (reproduces the grip)")
parser.add_argument("--contact_thresh", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import carb
import gymnasium as gym, torch
import numpy as np
from isaaclab.envs import DirectRLEnvCfg
import rl_rebuild.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    cache = np.load(args_cli.cache).astype(np.float32)
    assert cache.ndim == 2 and cache.shape[1] == 29, f"expected (M,29), got {cache.shape}"
    M = cache.shape[0]
    N = min(args_cli.num_envs, M)
    out_path = args_cli.out or (os.path.splitext(args_cli.cache)[0] + "_contacts.npy")

    env_cfg.scene.num_envs = N
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.gravity_curriculum = False
    env_cfg.sim.gravity = (0.0, 0.0, args_cli.gravity_z)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    dev = uenv.device
    cache_t = torch.from_numpy(cache).to(dev)

    env.reset()
    all_ids = uenv.hand._ALL_INDICES
    _zeros_b = torch.zeros(N, dtype=torch.bool, device=dev)
    uenv._get_rewards = (lambda: torch.zeros(N, device=dev))          # neuter grasp-env reward/reset
    uenv._get_dones = (lambda: (_zeros_b.clone(), _zeros_b.clone()))  # never done
    uenv.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, float(args_cli.gravity_z)))

    jn = uenv.hand.joint_names
    flex = torch.tensor([1.0 if (any(t in n for t in ("_FE", "_PIP", "_DIP", "_IP")) or n.endswith("pinky_CMC"))
                         else 0.0 for n in jn], device=dev)
    za = torch.zeros(N, uenv.cfg.action_space, device=dev)

    contacts = np.zeros((M, 5), dtype=np.float32)
    n_batches = (M + N - 1) // N
    for b in range(n_batches):
        lo, hi = b * N, min((b + 1) * N, M)
        rows = cache_t[lo:hi]
        m = rows.shape[0]
        # pad the batch to N by repeating the last row (padded envs are ignored on read-back)
        if m < N:
            rows = torch.cat([rows, rows[-1:].expand(N - m, -1)], dim=0)
        dofs = rows[:, :22].clone()
        obj_pos_rel = rows[:, 22:25]
        obj_quat = rows[:, 25:29]
        target_sq = torch.clamp(dofs + args_cli.squeeze * flex.unsqueeze(0),
                                uenv.hand_dof_lower_limits, uenv.hand_dof_upper_limits)
        uenv.hand.write_joint_state_to_sim(dofs, torch.zeros_like(dofs), env_ids=all_ids)
        uenv.hand.set_joint_position_target(target_sq, env_ids=all_ids)
        uenv.prev_targets[:] = target_sq
        uenv.cur_targets[:] = target_sq
        root = torch.zeros(N, 7, device=dev)
        root[:, :3] = obj_pos_rel + uenv.scene.env_origins
        root[:, 3:7] = obj_quat
        uenv.object.write_root_pose_to_sim(root, all_ids)
        uenv.object.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev), all_ids)
        uenv.sim.forward()
        eng_hist = []
        for t in range(args_cli.settle_steps):
            uenv.prev_targets[:] = target_sq
            uenv.step(za)
            uenv._refresh_lab()
            if t >= args_cli.settle_steps - 10:
                eng_hist.append((uenv.last_contacts > args_cli.contact_thresh).float())
        eng = (torch.stack(eng_hist).mean(0) > 0.5).float()          # majority vote over the last 10 steps
        contacts[lo:hi] = eng[:m].cpu().numpy()
        print(f"[annotate] batch {b+1}/{n_batches}: rows {lo}:{hi}  "
              f"mean n_contact={eng[:m].sum(-1).mean().item():.2f}")

    np.save(out_path, contacts)
    ns = contacts.sum(-1)
    hist = {k: int((ns == k).sum()) for k in range(6)}
    print("\n==== ENCLOSING-CACHE CONTACT ANNOTATION ====")
    print(f"cache={args_cli.cache} (M={M})  squeeze={args_cli.squeeze}  gravity={args_cli.gravity_z}")
    print(f"saved -> {os.path.abspath(out_path)}  (M,5) {{0,1}}")
    print("contact-sum histogram: " + "  ".join(f"{k}:{hist[k]}" for k in range(6)))
    keep = (ns >= 3) & (ns <= 5)
    print(f"goal-pool rows with 3-5 contacts: {int(keep.sum())} ({100.0*keep.mean():.1f}%)")

    # ---- gg_sigma_q / gg_eps_q calibration: L1 joint distance from post-regrip reference poses ----
    if os.path.isfile(args_cli.ref_cache) and keep.sum() > 0:
        ref = np.load(args_cli.ref_cache).astype(np.float32)[:, :22]           # (K,22)
        goals = cache[keep, :22]                                               # (P,22)
        k = min(len(ref), 64)
        d = np.abs(ref[:k, None, :] - goals[None, :, :]).sum(-1)               # (k,P) L1 rad
        q10, q50, q90 = np.percentile(d, [10, 50, 90])
        print(f"L1 joint distance ref->goal-pool: p10={q10:.2f}  p50={q50:.2f}  p90={q90:.2f} rad")
        print(f"SUGGESTED cfg values: gg_sigma_q ~= {q50/1.5:.2f}  (median distance -> reward ~0.22), "
              f"gg_eps_q ~= {max(q10/2.0, 0.5):.2f}")
        print("Update SharpaWaveGraspXLRegripD3GoalGraspSphereCfg if these differ a lot from the defaults.")
    print("============================================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
