# ADJUST-CYCLE DIAGNOSTIC (ZERO TRAINING, 2026-07-07): tests whether the policy is a FEEDFORWARD
# primitive ("grab once, twist once, hold") or a CLOSED-LOOP CYCLIC controller (rotate -> sense
# degradation -> re-seat -> rotate again). Two questions, one run:
#   (A) ONE-TIME vs CYCLIC: over an episode, count "rotation bursts" (contiguous supra-threshold
#       spin runs) and contact-SWITCH events (a finger changing contact state). One-time+stereotyped
#       => ~1 burst, switch rate collapses to ~0 after the initial grab. Cyclic => many bursts,
#       sustained switching, cum-rotation is a STAIRCASE not a single step.
#   (B) OPEN- vs CLOSED-LOOP: mid-episode, inject a velocity KICK into HALF the envs (perturbed);
#       the other half are controls. If perturbed envs DROP and never recover held (while controls
#       continue) => no re-adjustment capability (open-loop-ish). If they RECOVER held and resume
#       rotation => closed-loop re-adjustment exists.
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--eval_steps", type=int, default=600)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--contact_thresh", type=float, default=0.15)
parser.add_argument("--min_fingers", type=int, default=2)
parser.add_argument("--held_disp", type=float, default=0.05)
parser.add_argument("--burst_thresh", type=float, default=0.30, help="rad/s: supra = 'rotating'")
parser.add_argument("--perturb_at", type=int, default=200, help="active-step to inject the kick")
parser.add_argument("--kick", type=float, default=0.35, help="m/s lateral velocity impulse")
parser.add_argument("--recover_win", type=int, default=200, help="steps to watch recovery")
parser.add_argument("--tag", type=str, default="")
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
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
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
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    if hasattr(env_cfg, "rgv2_gravity_curriculum"):
        env_cfg.rgv2_gravity_curriculum = False
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/adjcyc", full_config=config, create_output_dir=False)
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N, dev = uenv.num_envs, uenv.device
    ct = args_cli.contact_thresh
    # env split: even idx = control, odd idx = perturbed
    is_pert = (torch.arange(N, device=dev) % 2 == 1)
    print(f"[adjcyc]{args_cli.tag} ckpt={os.path.basename(args_cli.load_path)} "
          f"kick={args_cli.kick} @active-step {args_cli.perturb_at}")

    seat = torch.zeros(N, 3, device=dev); started = torch.zeros(N, dtype=torch.bool, device=dev)
    age = torch.zeros(N, device=dev)                      # active steps this episode
    prev_bit = torch.zeros(N, 5, device=dev)
    in_burst = torch.zeros(N, dtype=torch.bool, device=dev)
    ep_bursts = torch.zeros(N, device=dev)               # rotation-burst count
    ep_switch = torch.zeros(N, device=dev)               # contact-switch events (early window)
    ep_switch_late = torch.zeros(N, device=dev)          # switches AFTER first burst ends
    ep_first_burst_rot = torch.zeros(N, device=dev)      # net rot during the 1st burst
    ep_total_rot = torch.zeros(N, device=dev)
    seen_first_burst_end = torch.zeros(N, dtype=torch.bool, device=dev)
    kicked = torch.zeros(N, dtype=torch.bool, device=dev)
    # recovery tracking: for perturbed envs, held-frac in the window AFTER their kick
    post_held = torch.zeros(N, device=dev); post_steps = torch.zeros(N, device=dev)
    post_rot = torch.zeros(N, device=dev)
    R = {"bursts": [], "sw_early": [], "sw_late": [], "firstfrac": [], "pert_recov": [], "pert_rot": [],
         "ctrl_recov": [], "ctrl_rot": [], "is_pert": []}

    obs = env.reset()
    prev_q = uenv.object.data.root_quat_w.clone()
    with torch.no_grad():
        for t in range(args_cli.eval_steps):
            inp = {"obs": rms(obs["obs"]), "priv_info": obs["priv_info"]}
            if "pointcloud" in obs:
                inp["pointcloud"] = obs["pointcloud"]
            mu = torch.clamp(model.act_inference(inp), -1.0, 1.0)
            obs, r, done, info = env.step(mu)
            done = done.bool()
            active = (uenv._gx_settle == 0) if hasattr(uenv, "_gx_settle") else \
                torch.ones(N, dtype=torch.bool, device=dev)
            newly = active & (~started)
            if bool(newly.any()):
                seat[newly] = uenv.object_pos[newly]; started |= active
            age += active.float()
            q = uenv.object.data.root_quat_w
            dth = (axis_angle_from_quat(quat_mul(q, quat_conjugate(prev_q))) * uenv.rot_axis).sum(-1)
            prev_q = q.clone()
            dt = uenv.step_dt
            w = dth / dt
            n_eng = (uenv.last_contacts > ct).sum(-1)
            bit = (uenv.last_contacts > ct).float()
            disp = (uenv.object_pos - seat).norm(dim=-1)
            valid = active & (~done) & started
            held = valid & (n_eng >= args_cli.min_fingers) & (disp < args_cli.held_disp)
            # --- contact-switch events (finger flips contact state) ---
            sw = ((bit != prev_bit).float().sum(-1)) * valid.float()
            prev_bit = bit
            # --- rotation bursts ---
            rotating = (w.abs() > args_cli.burst_thresh) & held
            rising = rotating & (~in_burst)
            ep_bursts += rising.float()
            falling = (~rotating) & in_burst
            seen_first_burst_end |= (falling & (ep_bursts >= 1))
            in_burst = rotating
            ep_total_rot += dth * held.float()
            ep_first_burst_rot += dth * held.float() * (ep_bursts <= 1).float()
            ep_switch += sw * (~seen_first_burst_end).float()
            ep_switch_late += sw * seen_first_burst_end.float()
            # --- inject the kick when a PERTURBED env reaches perturb_at ---
            fire = is_pert & (~kicked) & valid & (age.long() == args_cli.perturb_at)
            if bool(fire.any()):
                fid = fire.nonzero(as_tuple=False).squeeze(-1)
                vel = uenv.object.data.root_vel_w.clone()[fid]
                ang = torch.rand(len(fid), device=dev) * 6.2832
                vel[:, 0] += args_cli.kick * torch.cos(ang)
                vel[:, 1] += args_cli.kick * torch.sin(ang)
                uenv.object.write_root_velocity_to_sim(vel, fid)
                kicked[fid] = True
            # --- also mark a matched control window (control envs at perturb_at) for baseline recovery ---
            ctrl_fire = (~is_pert) & (~kicked) & valid & (age.long() == args_cli.perturb_at)
            kicked[ctrl_fire] = True   # reuse 'kicked' as "window-open" flag for both arms
            # accumulate post-window held + rotation for any env whose window is open
            win_open = kicked & valid & (age.long() > args_cli.perturb_at) & \
                       (age.long() <= args_cli.perturb_at + args_cli.recover_win)
            post_held += held.float() * win_open.float()
            post_steps += win_open.float()
            post_rot += dth * held.float() * win_open.float()
            # --- episode bookkeeping ---
            if done.any():
                fin = done.nonzero(as_tuple=False).squeeze(-1)
                for e in fin.tolist():
                    if int(age[e]) >= args_cli.perturb_at + 30:  # lived long enough to matter
                        tot = float(ep_total_rot[e]); fb = float(ep_first_burst_rot[e])
                        R["bursts"].append(float(ep_bursts[e]))
                        R["sw_early"].append(float(ep_switch[e]))
                        R["sw_late"].append(float(ep_switch_late[e]))
                        R["firstfrac"].append(abs(fb) / (abs(tot) + 1e-6))
                        R["is_pert"].append(bool(is_pert[e]))
                        rec = float(post_held[e]) / max(float(post_steps[e]), 1.0)
                        if bool(is_pert[e]):
                            R["pert_recov"].append(rec); R["pert_rot"].append(float(post_rot[e]))
                        else:
                            R["ctrl_recov"].append(rec); R["ctrl_rot"].append(float(post_rot[e]))
                for z in (age, ep_bursts, ep_switch, ep_switch_late, ep_first_burst_rot, ep_total_rot,
                          post_held, post_steps, post_rot):
                    z[done] = 0
                in_burst[done] = False; started[done] = False; seen_first_burst_end[done] = False
                kicked[done] = False; prev_bit[done] = 0

    def m(x): return float(np.mean(x)) if len(x) else float("nan")
    n = len(R["bursts"])
    print(f"\n==== ADJUST-CYCLE DIAGNOSTIC ({n} episodes) ====")
    print(f"(A) ONE-TIME vs CYCLIC:")
    print(f"    rotation bursts / episode:      mean={m(R['bursts']):.2f}   (1 => one-time; >3 => cyclic)")
    print(f"    net rot in FIRST burst / total: mean={m(R['firstfrac']):.2f}   (~1.0 => one-time)")
    print(f"    contact switches EARLY(pre-1stburst-end): {m(R['sw_early']):.1f}")
    print(f"    contact switches LATE (after):            {m(R['sw_late']):.1f}   (~0 => stereotyped hold; high => cyclic re-grasp)")
    print(f"(B) OPEN vs CLOSED loop (perturbation recovery):")
    print(f"    CONTROL   held-frac in window: {m(R['ctrl_recov']):.3f}   rot: {m(R['ctrl_rot']):+.3f} rad")
    print(f"    PERTURBED held-frac in window: {m(R['pert_recov']):.3f}   rot: {m(R['pert_rot']):+.3f} rad")
    cr, pr = m(R['ctrl_recov']), m(R['pert_recov'])
    if pr == pr and cr == cr:
        ratio = pr / (cr + 1e-6)
        print(f"    RECOVERY RATIO (perturbed/control held): {ratio:.2f}   "
              f"(~1 => re-adjusts & recovers=CLOSED-LOOP; <0.4 => drops after kick=OPEN-LOOP)")
    print("=" * 46)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
