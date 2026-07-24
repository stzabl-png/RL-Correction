# LOAD-LEDGER PROBE (ZERO TRAINING) — the pre-registered eval for the REGRIP v2 sweep.
# Design doc: robotics-rl-expert/notes/reward_design/regrip_v2_load_bearing_designs_2026-07-02.md
#
# Runs a policy on its own v2 task and aggregates PER-EPISODE the chassis' load-bearing machinery
# (sharpa_wave_regripv2_common.py buffers — the probe measures exactly what the rewards measure):
#   * failure terminations (sink below z_start-3cm / height band) vs timeouts
#   * palm-clear survival: episodes with >=200 active steps (~10 s) and clear-fraction >= 0.95
#   * min per-finger cumulative load share (the maximin ledger), median/p90
#   * completed verified handovers per episode + distinct fingers with >=1
#   * switch-rate on the debounced engagement (fingers flipped / active step)
#   * wrench residual (||sum F + m g + F_ext|| / m|g|) median
# Launch with SHARPA_EVAL_CLEAN=1 (runner does) so perturbation/mechanisms/curriculum are OFF and
# every design is compared under identical clean physics. Works on ball and cylinder v2 tasks.
# Also runnable on NON-v2 tasks (e.g. the failed v1 g2_scon ckpt) via --shadow: a self-contained
# shadow ledger (same formulas) is computed here instead of reading env buffers.
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--load_path", type=str, required=True)
parser.add_argument("--algorithm", type=str, default=None)
parser.add_argument("--eval_steps", type=int, default=600)
parser.add_argument("--gravity_z", type=float, default=-9.81)
parser.add_argument("--orient_cap", type=float, default=0.0)
parser.add_argument("--zero_action", action="store_true")
parser.add_argument("--shadow", action="store_true",
                    help="compute a self-contained ledger (for non-v2 tasks); v2 buffers ignored")
parser.add_argument("--drop_below", type=float, default=0.03)
parser.add_argument("--clear_margin", type=float, default=0.015)
parser.add_argument("--min_active", type=int, default=30)
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
    agent = eval(agent_cfg["algo"])(env, output_dir="logs/ledger_probe", full_config=config, create_output_dir=False)
    print(f"[ledger] {args_cli.task} ckpt={args_cli.load_path} zero_action={args_cli.zero_action} "
          f"gravity={args_cli.gravity_z} clean={os.environ.get('SHARPA_EVAL_CLEAN','0')}")
    agent.restore_test(args_cli.load_path); agent.set_eval()
    rms, model = agent.running_mean_std, agent.model
    N, dev = uenv.num_envs, uenv.device
    is_v2 = hasattr(uenv, "_v2_L") and not args_cli.shadow

    # shadow ledger state (used when the task lacks the v2 chassis)
    zstart = torch.zeros(N, device=dev); started = torch.zeros(N, dtype=torch.bool, device=dev)
    L = torch.zeros(N, 5, device=dev); eng_state = torch.zeros(N, 5, device=dev)
    # per-episode accumulators (both modes)
    ep_act = torch.zeros(N, device=dev); ep_clear = torch.zeros(N, device=dev)
    ep_flip = torch.zeros(N, device=dev); prev_eng = torch.zeros(N, 5, device=dev)
    ep_resid = torch.zeros(N, device=dev)
    # completed-episode records
    R = {k: [] for k in ("act", "clear_frac", "min_share", "handovers", "distinct", "switch", "resid",
                         "fail", "timeout")}

    obs = env.reset()
    with torch.no_grad():
        for t in range(args_cli.eval_steps):
            if args_cli.zero_action:
                mu = torch.zeros(N, uenv.cfg.action_space, device=dev)
            else:
                inp = {"obs": rms(obs["obs"]), "priv_info": obs["priv_info"]}
                if "pointcloud" in obs:
                    inp["pointcloud"] = obs["pointcloud"]
                mu = torch.clamp(model.act_inference(inp), -1.0, 1.0)
            obs, r, done, info = env.step(mu)
            done = done.bool()
            active = (uenv._gx_settle == 0) if hasattr(uenv, "_gx_settle") else \
                torch.ones(N, dtype=torch.bool, device=dev)
            z = uenv.object_pos[:, 2]
            if is_v2:
                zs, st = uenv._v2_zstart, uenv._v2_started
                eng = uenv._v2_eng
            else:
                newly = active & (~started)
                zstart[newly] = z[newly]; started |= active
                zs, st = zstart, started
                mag = uenv.last_contacts
                eng_state = torch.where(mag > 0.15, torch.ones_like(eng_state),
                                        torch.where(mag < 0.05, torch.zeros_like(eng_state), eng_state))
                eng = eng_state
                L += mag * eng * (active & st & (z >= zs - args_cli.clear_margin)).float().unsqueeze(-1)
            valid = active & (~done)
            clear = st & (z >= zs - args_cli.clear_margin)
            ep_clear += (clear & valid).float()
            ep_flip += ((eng - prev_eng).abs().sum(-1) * valid.float())
            prev_eng = eng.clone()
            # wrench residual (shadow-computable from magnitudes is meaningless; only track for v2)
            if is_v2 and uenv._v2_cache:
                ep_resid += uenv._v2_cache["resid"].clamp(0, 3) * valid.float()
            ep_act += valid.float()

            if done.any():
                fin = done.nonzero(as_tuple=False).squeeze(-1)
                for e in fin.tolist():
                    if int(ep_act[e]) < args_cli.min_active:
                        continue
                    a = float(ep_act[e])
                    if is_v2:
                        Le = uenv._v2_L[e]; ho = uenv._v2_handover_done[e]
                        fail = bool(uenv._v2_fail_now[e])
                    else:
                        Le = L[e]; ho = torch.zeros(5, device=dev)
                        fail = bool(z[e] < zs[e] - args_cli.drop_below)
                    Ls = float(Le.sum())
                    R["act"].append(a)
                    R["clear_frac"].append(float(ep_clear[e]) / a)
                    R["min_share"].append(float(Le.min()) / Ls if Ls > 1e-9 else 0.0)
                    R["handovers"].append(float(ho.sum()))
                    R["distinct"].append(float((ho > 0.5).sum()))
                    R["switch"].append(float(ep_flip[e]) / a)
                    R["resid"].append(float(ep_resid[e]) / a)
                    R["fail"].append(1.0 if fail else 0.0)
                    R["timeout"].append(0.0 if fail else 1.0)
                ep_act[done] = 0; ep_clear[done] = 0; ep_flip[done] = 0; ep_resid[done] = 0
                prev_eng[done] = 0
                if not is_v2:
                    L[done] = 0; started[done] = False; eng_state[done] = 0

    def q(v, p):
        return float(np.percentile(np.array(v), p)) if v else float("nan")

    n = len(R["act"])
    a = {k: np.array(v) for k, v in R.items()}
    surv = (a["act"] >= 200) & (a["clear_frac"] >= 0.95) if n else np.array([])
    print("\n==== LOAD-LEDGER PROBE ====")
    print(f"task={args_cli.task}")
    print(f"episodes scored (active>={args_cli.min_active}): {n}")
    if n:
        print(f"FAIL-terminated: {a['fail'].mean():.2f}   timeout: {a['timeout'].mean():.2f}")
        print(f"PALM-CLEAR SURVIVAL (>=200 active steps AND clear>=0.95): {surv.mean():.3f}")
        print(f"clear fraction:  mean={a['clear_frac'].mean():.3f}  median={q(R['clear_frac'],50):.3f}")
        print(f"MIN per-finger cumulative load share: median={q(R['min_share'],50):.4f}  "
              f"p90={q(R['min_share'],90):.4f}   (all-five-support iff > 0; target >= 0.05)")
        print(f"verified handovers/ep: mean={a['handovers'].mean():.2f}  p90={q(R['handovers'],90):.1f}  "
              f"distinct fingers w/ >=1: mean={a['distinct'].mean():.2f}")
        print(f"switch-rate (debounced flips/active step): mean={a['switch'].mean():.4f}")
        print(f"wrench residual (v2 only): median={q(R['resid'],50):.3f}  (balanced < 0.3)")
    print("PASS bar: survival>=0.2 AND min_share median>=0.05 AND handovers>=2 (design doc).")
    print("===========================")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
