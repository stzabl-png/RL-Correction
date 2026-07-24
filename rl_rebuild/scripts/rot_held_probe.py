# HELD-GATED rotation probe (ZERO TRAINING) — closes the "escape-spin" metric trap (2026-07-03):
# a falling/rolling ball registers angular velocity without being held, inflating signed-yaw metrics.
# This probe counts rotation ONLY on steps where the object is genuinely held:
#   held := n_engaged >= --min_fingers  AND  displacement from episode-start seat < --held_disp.
# Reports held-gated signed/|omega| about the rot axis, held-time fraction, held cumulative rotation
# per episode, and per-finger duty over HELD steps only. Everything else mirrors contact_switch_probe.
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
parser.add_argument("--min_active", type=int, default=30)
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
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/rot_held", full_config=config, create_output_dir=False)
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N, dev = uenv.num_envs, uenv.device
    print(f"[rot-held] {args_cli.task} ckpt={args_cli.load_path} g={args_cli.gravity_z} "
          f"held := n>={args_cli.min_fingers} & disp<{args_cli.held_disp}")

    seat = torch.zeros(N, 3, device=dev); started = torch.zeros(N, dtype=torch.bool, device=dev)
    ep_act = torch.zeros(N, device=dev); ep_held = torch.zeros(N, device=dev)
    ep_cum_held = torch.zeros(N, device=dev)
    ep_cum_phys = torch.zeros(N, device=dev)   # PHYSICAL net rotation: all valid steps, held or not
    duty = torch.zeros(N, 5, device=dev)
    R = {k: [] for k in ("act", "held_frac", "cum_held", "rate_held")}
    sum_w_held, sum_absw_held, held_steps_total = 0.0, 0.0, 0.0
    # BRAKING CURVE (Track C, 2026-07-03): per-step signed/|omega| binned by instantaneous n_contact
    bk_w = torch.zeros(6, device=dev); bk_absw = torch.zeros(6, device=dev); bk_n = torch.zeros(6, device=dev)

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
            q = uenv.object.data.root_quat_w
            aa = axis_angle_from_quat(quat_mul(q, quat_conjugate(prev_q)))
            dth = (aa * uenv.rot_axis).sum(-1)
            prev_q = q.clone()
            n_eng = (uenv.last_contacts > args_cli.contact_thresh).sum(-1)
            disp = (uenv.object_pos - seat).norm(dim=-1)
            valid = active & (~done) & started
            held = valid & (n_eng >= args_cli.min_fingers) & (disp < args_cli.held_disp)
            nb = n_eng.clamp(0, 5).long()
            vw = dth / dt if (dt := uenv.step_dt) else dth
            for b in range(6):
                m = valid & (nb == b)
                if bool(m.any()):
                    bk_w[b] += vw[m].sum(); bk_absw[b] += vw[m].abs().sum(); bk_n[b] += m.sum()
            ep_act += valid.float(); ep_held += held.float()
            ep_cum_held += dth * held.float()
            ep_cum_phys += dth * valid.float()
            duty += (uenv.last_contacts > args_cli.contact_thresh).float() * held.float().unsqueeze(-1)
            sum_w_held += float((dth * held.float()).sum())
            sum_absw_held += float((dth.abs() * held.float()).sum())
            held_steps_total += float(held.float().sum())
            if done.any():
                fin = done.nonzero(as_tuple=False).squeeze(-1)
                for e in fin.tolist():
                    if int(ep_act[e]) >= args_cli.min_active:
                        a = float(ep_act[e]); h = float(ep_held[e])
                        R["act"].append(a); R["held_frac"].append(h / a)
                        R["cum_held"].append(float(ep_cum_held[e]))
                        R.setdefault("cum_phys", []).append(float(ep_cum_phys[e]))
                        R["rate_held"].append(float(ep_cum_held[e]) / (h * uenv.step_dt) if h > 5 else 0.0)
                ep_act[done] = 0; ep_held[done] = 0; ep_cum_held[done] = 0
                ep_cum_phys[done] = 0
                started[done] = False

    n = len(R["act"])
    dt = uenv.step_dt
    print("\n==== HELD-GATED ROTATION PROBE ====")
    print(f"episodes scored: {n}")
    if n:
        a = {k: np.array(v) for k, v in R.items()}
        print(f"held-time fraction:            mean={a['held_frac'].mean():.3f}  median={np.median(a['held_frac']):.3f}")
        print(f"HELD signed rate (rad/s):      global={sum_w_held/max(held_steps_total*dt,1e-9):+.4f}  "
              f"per-ep median={np.median(a['rate_held']):+.4f}")
        print(f"HELD |rate| (rad/s):           global={sum_absw_held/max(held_steps_total*dt,1e-9):.4f}")
        print(f"HELD cumulative rot per ep:    mean={a['cum_held'].mean():+.3f} rad  "
              f"p90(|.|)={np.percentile(np.abs(a['cum_held']),90):.3f}")
        if "cum_phys" in R and len(R["cum_phys"]) == n:
            cp = np.array(R["cum_phys"])
            print(f"PHYSICAL net rot per ep:       mean={cp.mean():+.3f} rad  "
                  f"p90(|.|)={np.percentile(np.abs(cp),90):.3f}  (held+unheld; the eye-visible number)")
        if held_steps_total > 0:
            d = (duty.sum(0) / max(held_steps_total, 1.0)).cpu().numpy()
            print("per-finger duty over HELD steps: " +
                  "/".join(f"{x:.2f}" for x in d) + "  (T/I/M/R/P)")
    print("BRAKING CURVE (per-step omega about rot axis, binned by instantaneous n_contact):")
    for b in range(6):
        cnt = float(bk_n[b])
        if cnt >= 50:
            print(f"  n={b}: steps={int(cnt):7d}  signed={float(bk_w[b])/cnt:+.4f} rad/s  "
                  f"|omega|={float(bk_absw[b])/cnt:.4f} rad/s")
        else:
            print(f"  n={b}: steps={int(cnt):7d}  (below 50-step floor, skipped)")
    print(f"VERDICT: in-hand rotation requires BOTH held-frac high AND |held signed rate| >> 0.06 "
          f"(the historical wall). Escape-spin shows high unheld rates but ~zero held rotation.")
    print("===================================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
