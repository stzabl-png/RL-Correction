# D0c -- ENCLOSING grasp-cache validation (ZERO TRAINING; physics + geometry only).
#
# Given an enclosing grasp cache (cache/graspxl_enclosing_<objid>.npy, 29-vec = 22 hand-joint pos + 3 obj
# pos(env-rel) + 4 obj quat), confirm it is a usable ROTATABLE reset state:
#   (a) STABLE  -- reset to the cached grasp, hold the grasp pose with ZERO policy action at FULL gravity
#                  (0,0,-9.81) for ~200 control steps; measure held fraction / drop rate (drop = object
#                  displaced > --drop_thresh m from its cached seat).
#   (b) ROTATION-READY -- compute the rotation-readiness potential Phi (the exact formula from
#                  sharpa_wave_graspxl_adjustrotate.py::_readiness_phi: 0.5*spread + 0.5*coverage from
#                  per-finger contact + fingertip dirs _|_ rot_axis). Enclosing grasp -> Phi >= ~0.7;
#                  the imported pinch was ~0.0-0.2.
#
# It drives the sim MANUALLY (no env.step) so the grasp-gen env's gravity-rotation / auto-save never run.
# No env/config/checkpoint/dataset is modified.
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-GraspXL-v0")
parser.add_argument("--cache", type=str, required=True, help="path to the enclosing grasp cache .npy (N,29)")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--hold_steps", type=int, default=200, help="control steps to hold at full gravity")
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--drop_thresh", type=float, default=0.10, help="object displacement (m) counted as a drop")
parser.add_argument("--contact_thresh", type=float, default=0.1, help="per-finger force (N) counted as engaged")
parser.add_argument("--phi_min", type=float, default=0.7, help="readiness target for an enclosing grasp")
parser.add_argument("--squeeze", type=float, default=0.0, help="rad of active finger-flexion squeeze on the held "
                    "pose (0=passive zero-action hold). _readiness_phi is an ACTIVE-rollout metric; a small "
                    "squeeze reproduces the policy's grip so the enclosing contacts (and Phi) actually register.")
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
    assert cache.ndim == 2 and cache.shape[1] == 29, f"expected (M,29) cache, got {cache.shape}"
    M = cache.shape[0]
    N = min(args_cli.num_envs, M)
    env_cfg.scene.num_envs = N
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # clean physical test: no DR (keep deterministic friction from the cfg), no curriculum
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.gravity_curriculum = False
    env_cfg.sim.gravity = (0.0, 0.0, args_cli.gravity_z)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    dev = uenv.device
    dec = uenv.cfg.decimation
    rot_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).view(1, 1, 3)

    cache_t = torch.from_numpy(cache).to(dev)
    torch.manual_seed(args_cli.seed)
    idx = torch.randint(0, M, (N,), device=dev)
    sel = cache_t[idx]                                   # (N,29)
    dofs = sel[:, :22].clone()                           # hand joint pos (articulation order)
    obj_pos_rel = sel[:, 22:25].clone()                  # env-relative object position
    obj_quat = sel[:, 25:29].clone()                     # object quat (wxyz)
    # active-squeeze target: close the FLEXION joints (FE/PIP/DIP/IP + pinky_CMC) by --squeeze rad past the
    # saved pose so the fingers actively press the object (the AA joints are left alone).
    jn = uenv.hand.joint_names
    flex = torch.tensor([1.0 if (any(t in n for t in ("_FE", "_PIP", "_DIP", "_IP")) or n.endswith("pinky_CMC"))
                         else 0.0 for n in jn], device=dev)
    print(f"[D0c] squeeze={args_cli.squeeze} rad on flexion joints: {[n for n, f in zip(jn, flex.tolist()) if f]}")

    env.reset()                                          # init scene/buffers (grasp-env reset; cache empty -> harmless)
    all_ids = uenv.hand._ALL_INDICES

    # Neuter the grasp env's gravity-rotation / cond-reset (_get_rewards) and force never-done (_get_dones)
    # so we drive the REAL env.step pipeline (correct contact-sensor + last_contacts updates) but gravity
    # stays at -z, no env is ever reset, and the grasp-gen auto-save never fires (cache file untouched).
    _zeros_b = torch.zeros(N, dtype=torch.bool, device=dev)
    uenv._get_rewards = (lambda: torch.zeros(N, device=dev))
    uenv._get_dones = (lambda: (_zeros_b.clone(), _zeros_b.clone()))

    # --- write the cached enclosing grasp as the reset state ---
    target_sq = torch.clamp(dofs + args_cli.squeeze * flex.unsqueeze(0),
                            uenv.hand_dof_lower_limits, uenv.hand_dof_upper_limits)   # active grip target
    dof_vel = torch.zeros_like(dofs)
    uenv.hand.write_joint_state_to_sim(dofs, dof_vel, env_ids=all_ids)               # joints start AT the saved pose
    uenv.hand.set_joint_position_target(target_sq, env_ids=all_ids)
    uenv.prev_targets[:] = target_sq
    uenv.cur_targets[:] = target_sq
    root = torch.zeros(N, 7, device=dev)
    root[:, :3] = obj_pos_rel + uenv.scene.env_origins
    root[:, 3:7] = obj_quat
    uenv.object.write_root_pose_to_sim(root, all_ids)
    uenv.object.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev), all_ids)
    uenv.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, float(args_cli.gravity_z)))
    uenv.sim.forward()
    seat = obj_pos_rel.clone()                           # reference seat (env-relative)

    def finger_force(i):
        # object-FILTERED contact force for sensor i (force_matrix_w[:,0,0,:]) -- the EXACT signal cond2 uses.
        return uenv._contact_sensor[i].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)

    def phi_from(last_contacts):
        center = uenv.object.data.root_pos_w.unsqueeze(1)                       # (N,1,3) world
        tips = uenv.hand.data.body_link_state_w[:, uenv.elastomer_ids, :3]      # (N,5,3) world
        rel = tips - center
        rel_perp = rel - (rel * rot_axis).sum(-1, keepdim=True) * rot_axis
        d = rel_perp / rel_perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        eng = (last_contacts > args_cli.contact_thresh).float()                 # 5 elastomers (matches _readiness_phi)
        n = eng.sum(-1)
        coverage = ((n - 1.0) / 4.0).clamp(0.0, 1.0)
        mean_d = (d * eng.unsqueeze(-1)).sum(1) / n.clamp_min(1.0).unsqueeze(-1)
        spread = torch.where(n >= 2.0, 1.0 - mean_d.norm(dim=-1), torch.zeros(N, device=dev))
        return 0.5 * spread + 0.5 * coverage, n

    # --- hold at full gravity via the REAL env.step (zero action; target = active-squeeze grip) ---
    za = torch.zeros(N, uenv.cfg.action_space, device=dev)
    max_disp = torch.zeros(N, device=dev)
    phi_hist, n_hist, c10_hist = [], [], []
    for t in range(args_cli.hold_steps):
        uenv.prev_targets[:] = target_sq                            # maintain the grip target (zero action keeps cur=prev)
        uenv.step(za)                                                # real pipeline (correct sensor/last_contacts updates)
        uenv._refresh_lab()
        disp = torch.norm(uenv.object_pos - seat, dim=-1)
        max_disp = torch.maximum(max_disp, disp)
        phi, n = phi_from(uenv.last_contacts)                        # Phi from last_contacts (== _readiness_phi)
        c10 = torch.stack([(finger_force(i) > 0.5).float() for i in range(10)], dim=1).sum(-1)  # cond2-style (10 sensors)
        phi_hist.append(phi); n_hist.append(n); c10_hist.append(c10)

    disp_final = torch.norm(uenv.object_pos - seat, dim=-1)
    zdrop = (seat[:, 2] - uenv.object_pos[:, 2])                     # +ve = sagged downward
    held_final = (disp_final <= args_cli.drop_thresh)
    never_dropped = (max_disp <= args_cli.drop_thresh)
    phi_t = torch.stack(phi_hist); n_t = torch.stack(n_hist).float(); c10_t = torch.stack(c10_hist).float()
    phi_early = phi_t[2:7].mean(0); phi_last = phi_t[-20:].mean(0)    # t~=0 (as-loaded) vs settled
    n_early = n_t[2:7].mean(0); n_last = n_t[-20:].mean(0)
    c10_early = c10_t[2:7].mean(0); c10_last = c10_t[-20:].mean(0)

    print("\n==== D0c ENCLOSING-GRASP VALIDATION ====")
    print(f"task={args_cli.task}")
    print(f"cache={args_cli.cache}  (M={M} states; validated N={N})")
    print(f"gravity_z={args_cli.gravity_z}  hold_steps={args_cli.hold_steps} (={args_cli.hold_steps*uenv.step_dt:.1f}s)  "
          f"drop_thresh={args_cli.drop_thresh} m  squeeze={args_cli.squeeze} rad")
    print("-- (a) STABILITY (zero action, full gravity) --")
    print(f"  held fraction (final disp <= {args_cli.drop_thresh}m):   {held_final.float().mean().item():.3f}")
    print(f"  never-dropped fraction (max disp <= thresh):    {never_dropped.float().mean().item():.3f}")
    print(f"  drop rate (final):                              {(1.0-held_final.float().mean()).item():.3f}")
    print(f"  object displacement (m): mean={disp_final.mean().item():.4f}  median={disp_final.median().item():.4f}  "
          f"p90={disp_final.quantile(0.9).item():.4f}  max={disp_final.max().item():.4f}")
    print(f"  z-sag (m, held envs):    mean={zdrop[held_final].mean().item() if held_final.any() else float('nan'):.4f}")
    print("-- (b) ROTATION-READINESS  Phi = 0.5*spread + 0.5*coverage  (5-elastomer contact _|_ rot_axis) --")
    print(f"  Phi @load (t~0, steps 2-6): mean={phi_early.mean().item():.3f}  median={phi_early.median().item():.3f}  "
          f"frac>={args_cli.phi_min}: {(phi_early >= args_cli.phi_min).float().mean().item():.3f}")
    print(f"  Phi @settled (last 20):     mean={phi_last.mean().item():.3f}  median={phi_last.median().item():.3f}  "
          f"frac>={args_cli.phi_min}: {(phi_last >= args_cli.phi_min).float().mean().item():.3f}  "
          f"p10={phi_last.quantile(0.1).item():.3f}  p90={phi_last.quantile(0.9).item():.3f}")
    print(f"  #elastomers engaged (>{args_cli.contact_thresh}N): @load mean={n_early.mean().item():.2f}  "
          f"@settled mean={n_last.mean().item():.2f}")
    print(f"  #contacts cond2-style (>0.5N over 10 sensors): @load mean={c10_early.mean().item():.2f}  "
          f"@settled mean={c10_last.mean().item():.2f}")
    hist = {k: int((torch.round(n_early) == k).sum().item()) for k in range(6)}
    print(f"  #elastomers-engaged @load histogram: " + "  ".join(f"{k}:{hist[k]}" for k in range(6)))
    good = held_final & (phi_last >= args_cli.phi_min)
    print(f"  HELD & Phi(settled)>={args_cli.phi_min} (usable rotatable reset): {good.float().mean().item():.3f}")
    print("VERDICT: enclosing grasp confirmed if held fraction high AND Phi >= ~0.7 (pinch baseline ~0.0-0.2). "
          "Phi@load = the as-saved grasp; Phi@settled = after re-settling under constant -z.")
    print("========================================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
