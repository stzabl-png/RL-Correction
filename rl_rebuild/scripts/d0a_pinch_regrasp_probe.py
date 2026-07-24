# D0a — Pinch-start regrasp probe (ZERO TRAINING; eval only).
#
# Question (from notes/experiment_playbooks/adjust_then_rotate_from_nonready_grasp_plan.md, v2 §0.3):
#   Starting from the GraspXL *pinch* grasp, does an existing rotation-trained policy ever exhibit a
#   within-episode migration of the contact distribution from PINCH-LIKE (<= --pinch_max fingers in
#   contact) to ENCLOSING-LIKE (>= --enclose_min fingers, sustained >= --sustain_steps)? If >=1 episode
#   out of ~200 shows this, regrasp is physically possible -> proceed with the reverse-curriculum plan.
#   If 0/200, the conversion is not discoverable by this policy -> supply a reference trajectory or
#   pivot to the staged two-policy baseline.
#
# This script mirrors orient_diag_eval.py's harness verbatim (app launch, hydra cfg, env make, PPO
# restore, running-mean-std, act_inference, zero-action control) and ONLY adds per-finger contact
# logging + the migration detector. It does not modify any env/config/checkpoint.
#
# Always run a --zero_action control alongside the policy run: any migration under zero action is
# settling, not policy-driven regrasp.
import argparse, os, sys, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=200)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-PCWM-v0")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--eval_steps", type=int, default=1200)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--orient_cap", type=float, default=math.pi)
parser.add_argument("--zero_action", action="store_true", help="feed zero actions (settling control)")
# contact / migration definition
parser.add_argument("--contact_thresh", type=float, default=0.1, help="per-finger force (N) counted as contact")
parser.add_argument("--pinch_max", type=int, default=2, help="<= this many fingers = pinch-like (early)")
parser.add_argument("--enclose_min", type=int, default=4, help=">= this many fingers = enclosing-like")
parser.add_argument("--sustain_steps", type=int, default=10, help="enclosing must hold this many active steps")
parser.add_argument("--early_steps", type=int, default=20, help="active-step window used to measure the pinch start")
parser.add_argument("--min_active", type=int, default=30, help="ignore episodes shorter than this many active steps")
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
    # deterministic probe: all DR off (we want the physical answer, not robustness)
    env_cfg.randomize_pd_gains = False; env_cfg.randomize_friction = False
    env_cfg.randomize_com = False; env_cfg.randomize_mass = False
    env_cfg.sim.gravity = (0, 0, args_cli.gravity_z); env_cfg.gravity_curriculum = False
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/d0a", full_config=config, create_output_dir=False)
    orient_cur = float(getattr(uenv, "_gx_orient_cur", 0.0))
    print(f"[D0a] loading {args_cli.load_path}; task={args_cli.task}; zero_action={args_cli.zero_action}; "
          f"gravity_z={args_cli.gravity_z}; orient_cur={orient_cur:.3f}; "
          f"contact_thresh={args_cli.contact_thresh}N; pinch<={args_cli.pinch_max} -> enclose>={args_cli.enclose_min} "
          f"(sustain {args_cli.sustain_steps} steps)")
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N = uenv.num_envs
    dt = uenv.cfg.sim.dt * uenv.cfg.decimation
    drop_disp = float(getattr(uenv, "_gx_drop_disp", 0.10))
    n_fingers_total = uenv.last_contacts.shape[1]
    t5 = int(round(5.0 / dt)); t10 = int(round(10.0 / dt))   # active-step indices for the t=5s / t=10s snapshots

    # per-env episode state -------------------------------------------------
    ep_act = torch.zeros(N, device=uenv.device, dtype=torch.long)      # active steps this episode
    early_min = torch.full((N,), 99, device=uenv.device, dtype=torch.long)  # min #fingers in early window
    ever_max = torch.zeros(N, device=uenv.device, dtype=torch.long)    # max #fingers seen (active)
    enc_run = torch.zeros(N, device=uenv.device, dtype=torch.long)     # current consecutive enclosing run
    reached = torch.zeros(N, device=uenv.device, dtype=torch.bool)     # ever sustained enclosing
    snap_t0 = torch.full((N,), -1, device=uenv.device, dtype=torch.long)
    snap_t5 = torch.full((N,), -1, device=uenv.device, dtype=torch.long)
    snap_t10 = torch.full((N,), -1, device=uenv.device, dtype=torch.long)
    cum = torch.zeros(N, device=uenv.device)                           # signed cumulative rotation (rad)

    # completed-episode records --------------------------------------------
    rec_mig, rec_early, rec_max, rec_cum = [], [], [], []
    h_t0, h_t5, h_t10 = [], [], []

    obs = env.reset()
    prev_q = uenv.object.data.root_quat_w.clone()
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
            aa = axis_angle_from_quat(quat_mul(q, quat_conjugate(prev_q)))
            dtheta = (aa * uenv.rot_axis).sum(-1)
            valid = active & (~done)                       # exclude reset-teleport step
            cum = cum + dtheta * valid.float()

            # per-finger contact -> number of engaged fingers
            n_fing = (uenv.last_contacts > args_cli.contact_thresh).sum(dim=1).long()   # (N,)
            ai = ep_act                                    # active-step index BEFORE increment

            vmask = valid
            if vmask.any():
                # early-window minimum (the "pinch start" measurement)
                in_early = vmask & (ai < args_cli.early_steps)
                early_min = torch.where(in_early & (n_fing < early_min), n_fing, early_min)
                # running max + sustained-enclosing detector
                ever_max = torch.where(vmask & (n_fing > ever_max), n_fing, ever_max)
                is_enc = vmask & (n_fing >= args_cli.enclose_min)
                enc_run = torch.where(is_enc, enc_run + 1, torch.where(vmask, torch.zeros_like(enc_run), enc_run))
                reached = reached | (enc_run >= args_cli.sustain_steps)
                # snapshots at active t=0 / t=5s / t=10s
                snap_t0 = torch.where(vmask & (ai == 0), n_fing, snap_t0)
                snap_t5 = torch.where(vmask & (ai == t5), n_fing, snap_t5)
                snap_t10 = torch.where(vmask & (ai == t10), n_fing, snap_t10)
                ep_act = ep_act + vmask.long()

            if done.any():
                for e in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    if int(ep_act[e]) >= args_cli.min_active:
                        mig = (int(early_min[e]) <= args_cli.pinch_max) and bool(reached[e])
                        rec_mig.append(1 if mig else 0)
                        rec_early.append(int(early_min[e]))
                        rec_max.append(int(ever_max[e]))
                        rec_cum.append(float(cum[e]))
                        if int(snap_t0[e]) >= 0:  h_t0.append(int(snap_t0[e]))
                        if int(snap_t5[e]) >= 0:  h_t5.append(int(snap_t5[e]))
                        if int(snap_t10[e]) >= 0: h_t10.append(int(snap_t10[e]))
                # reset per-env state for the finished envs
                ep_act[done] = 0; early_min[done] = 99; ever_max[done] = 0
                enc_run[done] = 0; reached[done] = False
                snap_t0[done] = -1; snap_t5[done] = -1; snap_t10[done] = -1
                cum[done] = 0.0
            prev_q = q.clone()

    # ---- report ----------------------------------------------------------
    def hist(vals):
        a = np.array(vals) if vals else np.array([], dtype=int)
        return {k: int((a == k).sum()) for k in range(n_fingers_total + 1)}, len(a)

    nep = len(rec_mig)
    nmig = int(sum(rec_mig))
    mig_with_rot = int(sum(1 for m, c in zip(rec_mig, rec_cum) if m and abs(c) > 0.5))  # migration AND >0.5 rad net
    print("\n==== D0a PINCH-START REGRASP PROBE ====")
    print(f"task={args_cli.task}  ckpt={args_cli.load_path}  zero_action={args_cli.zero_action}")
    print(f"episodes scored (active>={args_cli.min_active}): {nep}")
    print(f"MIGRATION episodes (early<= {args_cli.pinch_max} fingers AND later>= {args_cli.enclose_min} "
          f"sustained {args_cli.sustain_steps} steps): {nmig}/{nep}  ({100.0*nmig/max(nep,1):.1f}%)")
    print(f"   ...of which ALSO net-rotated >0.5 rad in-episode: {mig_with_rot}/{nep}")
    if rec_early:
        print(f"early-window min #fingers:  mean={np.mean(rec_early):.2f}  "
              f"(pinch-like fraction <= {args_cli.pinch_max}: {np.mean(np.array(rec_early)<=args_cli.pinch_max):.2f})")
        print(f"episode max #fingers:       mean={np.mean(rec_max):.2f}  "
              f"(reached >= {args_cli.enclose_min} fraction: {np.mean(np.array(rec_max)>=args_cli.enclose_min):.2f})")
    for label, vals in [("t=0   (active start)", h_t0), (f"t={5}s  active", h_t5), (f"t={10}s active", h_t10)]:
        h, n = hist(vals)
        print(f"#fingers-in-contact @ {label:18s} (n={n}): " + "  ".join(f"{k}:{h[k]}" for k in range(n_fingers_total + 1)))
    print("DECISION: >=1 migration episode => regrasp is physically reachable (proceed). "
          "0 migrations (and 0 under zero-action) => pivot to reference-trajectory or staged baseline.")
    print("=======================================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
