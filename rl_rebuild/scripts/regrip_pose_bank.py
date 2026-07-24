# CLOSED-LOOP pose bank (ZERO TRAINING): roll a regrip policy, capture each env's LAST GOOD grasp
# with the user-specified signals, rank, and export the top-K diverse poses as single-row caches for
# rotation testing (SHARPA_POSE_CACHE + BandRot-PoseBank-v1).
#
# Captured per pose (user spec 2026-07-03): height change (z - z_start), per-finger 3D force VECTORS
# and CONTACT POSITIONS (force distribution in space), per-finger force RATIOS, plus n_engaged and a
# spread score (entropy of the force-ratio distribution, /log5 -> [0,1]).
# "Last good" = the most recent active step where the env was held (not sinking) with n_eng>=2.
# Selection (diversity on purpose): (1) max spread score, (2) max height (z-z_start), (3) max n_eng;
# ties broken by spread. Exports: <out>/pose{1,2,3}.npy (1,29), <out>/bank.npz (all), <out>/report.md.
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--orient_cap", type=float, default=0.0)
parser.add_argument("--topk", type=int, default=3)
parser.add_argument("--n_min", type=int, default=2, help="min engaged fingers for a snapshot")
parser.add_argument("--eng_thresh", type=float, default=0.15, help="per-finger force (N) counted engaged")
parser.add_argument("--sink_margin", type=float, default=0.03)
parser.add_argument("--use_ema", action="store_true",
                    help="count engagement from last_contacts (EMA, matches probes) instead of raw force-matrix norms")
parser.add_argument("--prefer_max_n", action="store_true",
                    help="keep each env's HIGHEST-n snapshot instead of its last good one")
parser.add_argument("--capture_degrading", action="store_true",
                    help="snapshot RECENTLY-HELD states degraded to n in {2,3} within margins "
                         "(recoverable-loss reset family for the rotate->adjust->rotate cycle)")
parser.add_argument("--min_omega", type=float, default=0.0,
                    help="if >0, only snapshot DRIVE states: signed omega about rot_axis >= this "
                         "(rad/s), and rank pose1 by omega (reverse-curriculum harvesting)")
parser.add_argument("--out", type=str, required=True)
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
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
import rl_rebuild.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

_EPS = 1e-9


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
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/pose_bank", full_config=config, create_output_dir=False)
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N, dev = uenv.num_envs, uenv.device
    os.makedirs(args_cli.out, exist_ok=True)
    print(f"[bank] {args_cli.task} ckpt={args_cli.load_path} g={args_cli.gravity_z}")

    zstart = torch.zeros(N, device=dev); started = torch.zeros(N, dtype=torch.bool, device=dev)
    good = torch.zeros(N, dtype=torch.bool, device=dev)          # env has >=1 good snapshot
    S_state = torch.zeros(N, 29, device=dev)
    S_F = torch.zeros(N, 5, 3, device=dev)
    S_P = torch.zeros(N, 5, 3, device=dev)
    S_dz = torch.zeros(N, device=dev)
    S_n = torch.zeros(N, device=dev)
    S_w = torch.zeros(N, device=dev)

    held_ema = torch.zeros(N, device=dev)
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
            z = uenv.object_pos[:, 2]
            newly = active & (~started)
            if bool(newly.any()):
                zstart[newly] = z[newly]; started |= active
            F = torch.stack([uenv._contact_sensor[i].data.force_matrix_w[:, 0, 0, :] for i in range(5)], dim=1)
            P = torch.cat([uenv._contact_sensor[i].data.contact_pos_w[:, 0, 0, :].unsqueeze(1) for i in range(5)], dim=1)
            mag = F.norm(dim=-1)
            eng_src = uenv.last_contacts if args_cli.use_ema else mag
            n_eng = (eng_src > args_cli.eng_thresh).float().sum(-1)
            held = started & (z > zstart - args_cli.sink_margin)  # not sinking
            snap = active & (~done) & held & (n_eng >= float(args_cli.n_min))
            q_now = uenv.object.data.root_quat_w
            aa = axis_angle_from_quat(quat_mul(q_now, quat_conjugate(prev_q)))
            vw = (aa * uenv.rot_axis).sum(-1) / max(uenv.step_dt, 1e-9)
            prev_q = q_now.clone()
            if args_cli.capture_degrading:
                held_now = held & (n_eng >= 4.0)
                held_ema = 0.95 * held_ema + 0.05 * held_now.float()
                snap = (active & (~done) & held & (held_ema > 0.3)
                        & (n_eng >= 2.0) & (n_eng <= 3.0))        # recently solid, now degraded
            elif args_cli.min_omega > 0.0:
                snap = snap & (vw >= args_cli.min_omega)          # DRIVE states only (commanded dir)
                snap = snap & (vw >= S_w)                         # keep each env's fastest
            elif args_cli.prefer_max_n:
                snap = snap & (n_eng >= S_n)                      # only replace with >= support
            if bool(snap.any()):
                dof = uenv.hand.data.joint_pos
                if dof.shape[1] != 22:
                    dof = dof[:, uenv.actuated_dof_indices]
                state = torch.cat([dof, uenv.object_pos, uenv.object_rot], dim=-1)
                m = snap
                S_state[m] = state[m]; S_F[m] = F[m]; S_P[m] = P[m] - uenv.object.data.root_pos_w[m].unsqueeze(1)
                S_dz[m] = (z - zstart)[m]
                S_n[m] = n_eng[m]
                S_w[m] = vw[m]
                good |= m

    idx = good.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        print(f"[bank] NO good snapshots captured — policy never held with n>={args_cli.n_min}. Nothing exported.")
        env.close(); return
    st = S_state[idx].cpu().numpy(); Fv = S_F[idx].cpu().numpy(); Pv = S_P[idx].cpu().numpy()
    dz = S_dz[idx].cpu().numpy(); wv = S_w[idx].cpu().numpy()
    mag = np.linalg.norm(Fv, axis=-1)                             # (M,5)
    tot = mag.sum(-1) + _EPS
    lam = mag / tot[:, None]                                      # per-finger force ratios
    plam = np.clip(lam, _EPS, 1.0)
    spread = (-(plam * np.log(plam)).sum(-1)) / np.log(5.0)       # [0,1]
    n_eng = (mag > 0.15).sum(-1)

    # diverse top-K: best spread, best height, best n_eng (dedup; fill by spread)
    order = {"spread": np.argsort(-spread), "height": np.argsort(-dz), "neng": np.lexsort((-spread, -n_eng))}
    if args_cli.min_omega > 0.0:
        order = {"omega": np.argsort(-wv), "neng": np.lexsort((-wv, -n_eng)), "spread": np.argsort(-spread)}
    picks, used = [], set()
    for key in order:
        for j in order[key]:
            if int(j) not in used:
                picks.append((key, int(j))); used.add(int(j)); break
    k = 0
    while len(picks) < args_cli.topk and k < len(order["spread"]):
        j = int(order["spread"][k]); k += 1
        if j not in used:
            picks.append(("fill", j)); used.add(j)

    rep = ["# Pose bank report", f"policy: {args_cli.load_path}", f"task: {args_cli.task}",
           f"good envs: {len(idx)}/{N}", "",
           "| pose | picked-for | dz (m) | n_eng | omega (rad/s) | spread | ratios T/I/M/R/P | |F| per finger (N) |",
           "|---|---|---|---|---|---|---|---|"]
    for i, (why, j) in enumerate(picks, 1):
        np.save(os.path.join(args_cli.out, f"pose{i}.npy"), st[j:j+1].astype(np.float32))
        rep.append(f"| pose{i} | {why} | {dz[j]:+.4f} | {int(n_eng[j])} | {wv[j]:+.2f} | {spread[j]:.3f} | "
                   + "/".join(f"{v:.2f}" for v in lam[j]) + " | "
                   + "/".join(f"{v:.2f}" for v in mag[j]) + " |")
        print(f"[bank] pose{i} ({why}): dz={dz[j]:+.4f} n={int(n_eng[j])} omega={wv[j]:+.2f} "
              f"spread={spread[j]:.3f} ratios={np.round(lam[j],2)}")
    np.savez(os.path.join(args_cli.out, "bank.npz"), state=st, F=Fv, P=Pv, dz=dz, lam=lam,
             spread=spread, n_eng=n_eng, omega=wv)
    rep += ["", "Contact positions (object-relative, m) for picked poses:"]
    for i, (_, j) in enumerate(picks, 1):
        rep.append(f"- pose{i}: " + "; ".join(
            f"{f}=({Pv[j,fi,0]:+.3f},{Pv[j,fi,1]:+.3f},{Pv[j,fi,2]:+.3f})"
            for fi, f in enumerate(["T", "I", "M", "R", "P"]) if mag[j, fi] > 0.15))
    open(os.path.join(args_cli.out, "report.md"), "w").write("\n".join(rep) + "\n")
    print(f"[bank] exported {len(picks)} poses + bank.npz + report.md -> {os.path.abspath(args_cli.out)}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
