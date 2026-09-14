"""Clean/3 Stage-2 冒烟: 零动作 + 随机动作 各一回合 (N env), 核 obs/act 维、奖励有限、锁存/认证/时钟/A4/死线逻辑与台账。

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Clean/3/C_Wiring/smoke_task.py --headless [--num_envs 8] [--release_row 50]
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--release_row", type=int, default=None)
p.add_argument("--modes", default="zero,random")
p.add_argument("--force_cert_at", type=int, default=-1, help="冒烟专用: 放手后第 N 行强制置 certified (绕过认证, 验证时钟/重锚/A4 链)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
for k, v in TC.PHYS.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("clean3_smoke_task")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402
import clean_task_env as CE  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

raw = CE.CleanTaskEnv(CE.build_cfg(args.num_envs))
if args.release_row is not None:
    raw.release_row_cur = int(args.release_row)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
N = args.num_envs
for mode in args.modes.split(","):
    obs = env.reset()
    assert obs["obs"].shape == (N, CE.OBS_DIM), obs["obs"].shape
    assert obs["priv_info"].shape == (N, CE.PRIV_DIM), obs["priv_info"].shape
    rew_sum = torch.zeros(N, device=raw.device); done_at = torch.full((N,), -1, device=raw.device)
    marks = set([0, TC.K_CLOSE, raw.release_row_cur - 1, raw.release_row_cur + 2, raw.release_row_cur + 12, raw.release_row_cur + 30,
                 raw.release_row_cur + 61, raw.release_row_cur + 61 + 100, raw.release_row_cur + 61 + 300, raw.T_EP - 1])
    for step in range(raw.T_EP + 2):
        a = (torch.zeros(N, CE.ACT_DIM, device=raw.device) if mode == "zero"
             else 0.3 * torch.randn(N, CE.ACT_DIM, device=raw.device))
        if args.force_cert_at >= 0 and step == raw.release_row_cur + args.force_cert_at:
            raw.certified[:] = True
        obs, rew, done, info = env.step(a)
        assert torch.isfinite(obs["obs"]).all() and torch.isfinite(rew).all(), step
        rew_sum += rew
        done_at = torch.where((done > 0) & (done_at < 0), torch.full_like(done_at, step), done_at)
        t = raw._tick
        if step in marks or (step % 100 == 0 and step > raw.release_row_cur + 61):
            i = 0
            print(f"[smoke:{mode}] step {step:3d} rel={int(t['released'][i])} latch={int(raw.latched[i])} cert={int(raw.certified[i])} "
                  f"k={int(raw.k[i])} gate={int(raw.gate_ok[i])} e_n={float(t['e_n'][i])*100:.2f}cm e_xy={float(t['e_xy'][i])*100:.2f}cm "
                  f"dp L/R {float(t['dp']['left'][i])*100:.2f}/{float(t['dp']['right'][i])*100:.2f}cm dr {np.degrees(float(t['dr']['left'][i])):.1f}/{np.degrees(float(t['dr']['right'][i])):.1f}° "
                  f"contact={int(t['sig']['contact'][i])} gap={float(t['sig']['gap_min'][i])*1000:.1f}mm cov={float(t['out']['coverage'][i]):.2f} "
                  f"trav={float(t['out']['travel'][i])*100:.0f}cm tilt={np.degrees(float(t['sig']['plate_tilt'][i])):.1f}° died={int(raw.died[i])}/{int(raw.die_kind[i])} "
                  f"re_dx={float(raw._re_dx[i])*100:.2f}cm dq={np.degrees(float(raw.dq_re[i].abs().max())):.1f}° "
                  f"rew={float(rew[i]):.3f}", flush=True)
        if (done_at >= 0).all():
            break
    rates = raw.pop_rates()
    keys = ("sr/cert", "sr/success", "sr/clock_done", "prog/clock_frac", "prog/gate_frac", "term/die", "term/die_rel", "term/die_tilt",
            "term/die_plate_dev", "term/die_drop", "term/die_table", "term/cert_timeout", "hold/relp_max_plate_cm", "hold/relp_max_sponge_cm",
            "hold/relrot_max_plate_deg", "hold/relrot_max_sponge_deg", "task/coverage", "task/travel_cm", "task/contact_frac", "shape/cross_frac",
            "ep_rew/adv", "ep_rew/leash", "ep_rew/bonus", "ep/len")
    print(f"[smoke:{mode}] ep_rew mean={float(rew_sum.mean()):.2f} done_at={done_at.cpu().tolist()}")
    print(f"[smoke:{mode}] rates: " + " ".join(f"{k.split('/')[1]}={rates.get(k, float('nan')):.3f}" for k in keys), flush=True)
print(f"[smoke_task] PASS envs={N} T_EP={raw.T_EP} obs={CE.OBS_DIM} act={CE.ACT_DIM}", flush=True)
sys.stdout.flush()
try: _slot.release()
except Exception: pass
os._exit(0)
