# SNAPSHOT PROBER (2026-07-04): anti-churn harvest — train long, probe EVERY saved checkpoint,
# keep the argmax. One sim startup, N checkpoint rolls (restore_test between rolls). Prints a
# per-ckpt table of held-frac / held signed rate (median) / held cum rad per ep / episodes.
# Metrics mirror rot_held_probe (held := n>=min_fingers & disp<held_disp, active-gated, min_active).
import argparse, os, sys, glob
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--ckpt_dir", type=str, required=True, help="stage1_nn dir; probes ep_*.pth + last.pth + best.pth")
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--eval_steps", type=int, default=450)
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
    env_cfg.randomize_pd_gains = False; env_cfg.randomize_friction = False
    env_cfg.randomize_com = False; env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    if hasattr(env_cfg, "rgv2_gravity_curriculum"):
        env_cfg.rgv2_gravity_curriculum = False

    cks = sorted(glob.glob(os.path.join(args_cli.ckpt_dir, "ep_*.pth")))
    for extra in ("best.pth", "last.pth"):
        p = os.path.join(args_cli.ckpt_dir, extra)
        if os.path.isfile(p):
            cks.append(p)
    if not cks:
        print(f"[snap] no checkpoints in {args_cli.ckpt_dir}"); return
    agent_cfg["load_path"] = cks[0]
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/snap", full_config=config, create_output_dir=False)
    N, dev = uenv.num_envs, uenv.device
    print(f"[snap] {args_cli.task} dir={args_cli.ckpt_dir} n_ckpt={len(cks)} "
          f"envs={N} steps={args_cli.eval_steps} g={args_cli.gravity_z}")
    print("| ckpt | eps | held-frac | held rate med | held cum/ep | cum p90 |")
    print("|---|---|---|---|---|---|")

    for ck in cks:
        agent.restore_test(ck); agent.set_eval()
        rms, model = agent.running_mean_std, agent.model
        seat = torch.zeros(N, 3, device=dev); started = torch.zeros(N, dtype=torch.bool, device=dev)
        ep_act = torch.zeros(N, device=dev); ep_held = torch.zeros(N, device=dev)
        ep_cum = torch.zeros(N, device=dev)
        HF, RH, CM = [], [], []
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
                ep_act += valid.float(); ep_held += held.float()
                ep_cum += dth * held.float()
                if done.any():
                    fin = done.nonzero(as_tuple=False).squeeze(-1)
                    for e in fin.tolist():
                        a = float(ep_act[e])
                        if a >= args_cli.min_active:
                            h = float(ep_held[e])
                            HF.append(h / a)
                            RH.append(float(ep_cum[e]) / (h * uenv.step_dt) if h > 5 else 0.0)
                            CM.append(float(ep_cum[e]))
                    ep_act[done] = 0; ep_held[done] = 0; ep_cum[done] = 0
                    started[done] = False
        tag = os.path.basename(ck).replace(".pth", "")
        if HF:
            print(f"| {tag} | {len(HF)} | {np.mean(HF):.3f} | {np.median(RH):+.3f} | "
                  f"{np.mean(CM):+.3f} | {np.percentile(np.abs(CM), 90):.3f} |", flush=True)
        else:
            print(f"| {tag} | 0 | - | - | - | - |", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
