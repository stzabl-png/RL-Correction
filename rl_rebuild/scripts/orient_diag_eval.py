# Diagnostic eval for the SE(3)-orientation in-hand-rotation policy.
# Distinguishes SUSTAINED DIRECTIONAL rotation from SHAKING by measuring the per-episode CUMULATIVE
# SIGNED rotation about the (tilted, per-env) rot_axis: it grows ~linearly for real rotation and stays
# ~0 for back-and-forth jitter. Forces the orientation cap (default full SO(3)) so the eval shows the
# real capability, and supports a zero-action control.
import argparse, os, sys, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-OrientCurr-v0")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--eval_steps", type=int, default=1500)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--orient_cap", type=float, default=math.pi)
parser.add_argument("--zero_action", action="store_true", help="feed zero actions (passive-roll control)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# force the orientation cap before the env is built (env reads SHARPA_ORIENT_CAP at __init__)
os.environ["SHARPA_ORIENT_CAP"] = str(args_cli.orient_cap)
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import math
import gymnasium as gym, torch
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.algo.padapt.padapt import ProprioAdapt
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.math import axis_angle_from_quat, quat_mul, quat_conjugate
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
    env_cfg.randomize_pd_gains = False; env_cfg.randomize_friction = True
    env_cfg.randomize_com = False; env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/odiag", full_config=config, create_output_dir=False)
    print(f"[INFO] loading {args_cli.load_path}; orient_cur={float(uenv._gx_orient_cur):.3f} rad; "
          f"zero_action={args_cli.zero_action}; gravity_z={args_cli.gravity_z}")
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N = uenv.num_envs
    dt = uenv.cfg.sim.dt * uenv.cfg.decimation
    drop_disp = float(getattr(uenv, "_gx_drop_disp", 0.10))

    obs = env.reset()
    prev_q = uenv.object.data.root_quat_w.clone()
    cum = torch.zeros(N, device=uenv.device)          # per-env signed cumulative rotation (current episode)
    asteps = torch.zeros(N, device=uenv.device)        # per-env active steps (current episode)
    done_cum, done_steps = [], []                      # completed-episode cumulative + active-step counts
    sum_abs = 0.0; sum_signed = 0.0; sum_held = 0.0; nstep = 0
    # SUSTAINED check: |cumulative| net rotation at active-time milestones (linear growth => sustained;
    # saturating/short => one-time-then-drop). Bins are active-step counts.
    milestones = [50, 100, 200, 300, 400, 500]
    ms_cum = {m: [] for m in milestones}
    with torch.no_grad():
        for t in range(args_cli.eval_steps):
            if args_cli.zero_action:
                mu = torch.zeros(N, uenv.cfg.action_space, device=uenv.device)
            else:
                inp = {"obs": rms(obs["obs"]), "priv_info": obs["priv_info"]}
                if "pointcloud" in obs:
                    inp["pointcloud"] = obs["pointcloud"]
                mu = torch.clamp(model.act_inference(inp), -1.0, 1.0)
            obs, r, done, info = env.step(mu)
            done = done.bool()
            settle = uenv._gx_settle
            active = settle == 0
            q = uenv.object.data.root_quat_w
            aa = axis_angle_from_quat(quat_mul(q, quat_conjugate(prev_q)))    # rotation vector this step
            dtheta = (aa * uenv.rot_axis).sum(-1)                              # signed about rot_axis
            valid = active & (~done)                                          # exclude the reset-teleport step
            cum = cum + dtheta * valid.float()
            asteps = asteps + valid.float()
            ai = asteps.long()
            for m in milestones:                                # snapshot |cum| as each env crosses a milestone
                hit = valid & (ai == m)
                if hit.any():
                    ms_cum[m].extend(cum[hit].abs().tolist())
            # active-phase stats
            objpos = uenv.object.data.root_pos_w - uenv.scene.env_origins
            held = torch.norm(objpos - uenv.object_default_pose[:, :3], dim=-1) < drop_disp
            a = active.float(); denom = a.sum().clamp_min(1.0)
            sum_abs += float((dtheta.abs() / dt * a).sum() / denom)
            sum_signed += float((dtheta / dt * a).sum() / denom)
            sum_held += float((held.float() * a).sum() / denom); nstep += 1
            # record completed episodes
            if done.any():
                for e in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    if asteps[e] > 30:                                        # ignore stubs
                        done_cum.append(float(cum[e])); done_steps.append(float(asteps[e]))
                cum[done] = 0.0; asteps[done] = 0.0
            prev_q = q.clone()

    import numpy as np
    dc = np.array(done_cum) if done_cum else np.array([0.0])
    ds = np.array(done_steps) if done_steps else np.array([1.0])
    mean_cum = float(dc.mean()); mean_abscum = float(np.abs(dc).mean())
    mean_secs = float((ds * dt).mean())
    print("\n==== ORIENT DIAGNOSTIC (full SO(3) if cap=pi) ====")
    print(f"task={args_cli.task} ckpt={args_cli.load_path} episodes={len(done_cum)} cap={float(uenv._gx_orient_cur):.3f}")
    print(f"held fraction (active):                 {sum_held / nstep:.4f}")
    print(f"mean |yaw| about rot_axis (rad/s):      {sum_abs / nstep:.4f}")
    print(f"mean SIGNED yaw about rot_axis (rad/s): {sum_signed / nstep:.4f}   (<<|yaw| => shaking)")
    print(f"mean per-episode CUMULATIVE signed rot: {mean_cum:.3f} rad  ({mean_cum/(2*math.pi):.2f} rev) over ~{mean_secs:.1f}s active")
    print(f"mean per-episode |cumulative| signed:   {mean_abscum:.3f} rad   (net directional magnitude)")
    full = int((ds >= 270).sum()); print(f"sustained episodes (active>=270 steps ~13.5s): {full}/{len(done_cum)} "
                                          f"({100.0*full/max(len(done_cum),1):.0f}%)  [low => one-time/drop]")
    print("|cumulative| net rotation vs active-time milestone (LINEAR growth => SUSTAINED; flat => one-time):")
    for m in milestones:
        if ms_cum[m]:
            arr = np.array(ms_cum[m]); print(f"   active {m:4d} steps (~{m*dt:4.1f}s): |cum|={arr.mean():.2f} rad "
                                             f"({arr.mean()/(2*math.pi):.2f} rev)  n={len(arr)}")
    print(f"INTERPRET: signed≈|yaw| AND |cum| grows ~linearly to many rev => SUSTAINED directional rotation.")
    print("==================================================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
