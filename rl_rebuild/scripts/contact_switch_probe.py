# Contact-switch probe: does the trained hand rotate/hold via GENUINE FINGER GAITING
# (contact switching -- the SET of engaged fingers changes cyclically: fingers release, reposition,
# re-contact) or via FIXED-CONTACT rolling/rocking (the same fingers stay engaged throughout while the
# object still net-rotates)?
#
# Adapted verbatim from orient_diag_eval.py's harness (app launch, hydra cfg, env make, PPO restore,
# running-mean-std, act_inference rollout, zero-action control, signed-rotation-about-rot_axis). It ADDS
# per-finger contact binarisation + a contact-transition counter. It does NOT modify any env/cfg/ckpt.
#
# Contact field:  uenv.last_contacts  (N,5) per-finger contact force
#   defined  rl_rebuild/tasks/inhand_rotate/sharpa_wave_env.py:128
#   updated  rl_rebuild/tasks/inhand_rotate/sharpa_wave_env.py:460 (binary) / :467 (smooth)
#   engaged := last_contacts > contact_thresh (default 0.1 N == the reward's phi_contact_thresh)
# Rotation axis:  uenv.rot_axis  (N,3)   rl_rebuild/tasks/inhand_rotate/sharpa_wave_env.py:123
#
# GraspXL-only attrs are GUARDED so this also runs on the plain cylinder env
# (SharpaWaveBandRotCylinderEnv has NEITHER _gx_settle NOR _gx_orient_cur):
#   getattr(uenv,'_gx_settle',None)  -> if None, treat ALL steps as active.
#   getattr(uenv,'_gx_orient_cur',0.0)
import argparse, os, sys, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-Cylinder-v1")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--eval_steps", type=int, default=600)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--orient_cap", type=float, default=math.pi)
parser.add_argument("--zero_action", action="store_true", help="feed zero actions (passive-roll control)")
parser.add_argument("--contact_thresh", type=float, default=0.1, help="per-finger force (N) counted as engaged")
parser.add_argument("--min_active", type=int, default=30, help="ignore episodes shorter than this many active steps")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# force the orientation cap before the env is built (GraspXL envs read SHARPA_ORIENT_CAP at __init__;
# the plain cylinder env ignores it harmlessly)
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
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/cswitch", full_config=config, create_output_dir=False)
    orient_cur = float(getattr(uenv, "_gx_orient_cur", 0.0))     # GUARD: cylinder env has no _gx_orient_cur
    print(f"[CSWITCH] loading {args_cli.load_path}; task={args_cli.task}; zero_action={args_cli.zero_action}; "
          f"gravity_z={args_cli.gravity_z}; orient_cur={orient_cur:.3f} rad; contact_thresh={args_cli.contact_thresh}N")
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N = uenv.num_envs
    dt = uenv.cfg.sim.dt * uenv.cfg.decimation
    n_fing = uenv.last_contacts.shape[1]                          # 5 fingers

    # ---- global (over-active-steps) accumulators ----
    valid_steps = 0                                               # #(env,step) pairs that are active & not-reset
    sw_pairs = 0                                                  # #(env,step) pairs with a comparable previous step
    neng_sum = 0.0                                               # sum of n_engaged over valid steps
    perfinger_sum = torch.zeros(n_fing, device=uenv.device)      # per-finger engaged count over valid steps
    sw_sum = 0.0                                                 # sum of #fingers-flipped over comparable steps
    sw_ge1 = 0                                                   # #comparable steps with >=1 flip
    sum_abs = 0.0; sum_signed = 0.0                              # yaw about rot_axis (rad, pre-/dt scaling)

    # ---- per-episode buffers ----
    cum = torch.zeros(N, device=uenv.device)                    # signed cumulative rotation (rad), this episode
    ep_sw = torch.zeros(N, device=uenv.device)                 # total finger-flips this episode
    ep_act = torch.zeros(N, device=uenv.device)               # active steps this episode
    done_cum, done_sw, done_steps = [], [], []                  # completed-episode records

    obs = env.reset()
    prev_q = uenv.object.data.root_quat_w.clone()
    prev_eng = (uenv.last_contacts > args_cli.contact_thresh)    # (N,5) bool
    prev_valid = torch.zeros(N, dtype=torch.bool, device=uenv.device)
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

            # active mask: GraspXL settle-countdown if present, else ALL steps active (cylinder env)
            settle = getattr(uenv, "_gx_settle", None)
            active = (settle == 0) if settle is not None else torch.ones(N, dtype=torch.bool, device=uenv.device)
            valid = active & (~done)                             # exclude the reset-teleport step

            # signed rotation about rot_axis this step
            q = uenv.object.data.root_quat_w
            aa = axis_angle_from_quat(quat_mul(q, quat_conjugate(prev_q)))
            dtheta = (aa * uenv.rot_axis).sum(-1)                # (N,)

            # per-finger engaged mask + transitions vs previous step
            eng = (uenv.last_contacts > args_cli.contact_thresh)          # (N,5) bool
            flips = (eng != prev_eng).float().sum(-1)                     # (N,) #fingers flipped vs prev step
            sw_ok = valid & prev_valid                                    # compare only within one continuous episode segment

            # ---- accumulate global stats ----
            vf = valid.float()
            valid_steps += int(valid.sum())
            neng_sum += float((eng.float().sum(-1) * vf).sum())
            perfinger_sum += (eng.float() * vf.unsqueeze(-1)).sum(0)
            sum_abs += float((dtheta.abs() * vf).sum())
            sum_signed += float((dtheta * vf).sum())
            swf = sw_ok.float()
            sw_pairs += int(sw_ok.sum())
            sw_sum += float((flips * swf).sum())
            sw_ge1 += int(((flips >= 1) & sw_ok).sum())

            # ---- accumulate per-episode ----
            cum = cum + dtheta * vf
            ep_sw = ep_sw + flips * swf
            ep_act = ep_act + vf

            if done.any():
                for e in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    if int(ep_act[e]) >= args_cli.min_active:
                        done_cum.append(float(cum[e])); done_sw.append(float(ep_sw[e]))
                        done_steps.append(float(ep_act[e]))
                cum[done] = 0.0; ep_sw[done] = 0.0; ep_act[done] = 0.0

            prev_q = q.clone(); prev_eng = eng.clone(); prev_valid = valid.clone()

    # ---- report ----
    vs = max(valid_steps, 1); sp = max(sw_pairs, 1)
    mean_neng = neng_sum / vs
    perfinger_frac = (perfinger_sum / vs).tolist()
    switch_rate = sw_sum / sp                                     # mean #fingers flipped per active step
    frac_ge1 = sw_ge1 / sp                                        # fraction of active steps with >=1 flip
    dc = np.array(done_cum) if done_cum else np.array([0.0])
    dsw = np.array(done_sw) if done_sw else np.array([0.0])
    ds = np.array(done_steps) if done_steps else np.array([1.0])
    fnames = ["thumb", "index", "middle", "ring", "pinky"]

    print("\n==== CONTACT-SWITCH PROBE ====")
    print(f"task={args_cli.task}")
    print(f"ckpt={args_cli.load_path}  zero_action={args_cli.zero_action}  contact_thresh={args_cli.contact_thresh}N")
    print(f"active steps counted={valid_steps}  comparable(switch) steps={sw_pairs}  episodes(active>={args_cli.min_active})={len(done_cum)}")
    print("-- CONTACT USE --")
    print(f"mean n_engaged (fingers w/ force>thr):  {mean_neng:.3f}")
    print("per-finger engaged-fraction:            " + "  ".join(f"{n}:{f:.3f}" for n, f in zip(fnames, perfinger_frac)))
    print("-- CONTACT SWITCHING (gaiting signal) --")
    print(f"CONTACT-SWITCH RATE (#fingers flipped/active step): {switch_rate:.4f}   (~0 => fixed contacts; higher => gaiting)")
    print(f"fraction of active steps with >=1 switch:           {frac_ge1:.4f}")
    print(f"mean per-episode total switches:                    {dsw.mean():.2f}  (over ~{ds.mean():.0f} active steps ~{ds.mean()*dt:.1f}s)")
    print("-- ROTATION (correlate with switching) --")
    print(f"mean |yaw| about rot_axis (rad/s):      {sum_abs / vs / dt:.4f}")
    print(f"mean SIGNED yaw about rot_axis (rad/s): {sum_signed / vs / dt:.4f}   (<<|yaw| => shaking, not net rotation)")
    print(f"mean per-episode CUMULATIVE signed rot: {dc.mean():.3f} rad  ({dc.mean()/(2*math.pi):.2f} rev)")
    print(f"mean per-episode |cumulative| signed:   {np.abs(dc).mean():.3f} rad")
    print("INTERPRET: net-rotating (|cum| grows) WITH switch-rate ~0 => FIXED-CONTACT rolling/rocking (no gaiting).")
    print("           switch-rate clearly >0 with fingers cycling on/off => GENUINE FINGER GAITING.")
    print("==============================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
