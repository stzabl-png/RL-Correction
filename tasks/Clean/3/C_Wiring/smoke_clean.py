"""Clean/3 Stage-1 抓稳段 冒烟: 零动作 + 随机动作 各一回合, 核 obs/act 维、奖励有限、钉住/放手/认证/掉落逻辑与台账。

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Clean/3/C_Wiring/smoke_clean.py --headless [--num_envs 8]
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--release_row", type=int, default=None, help="覆写课程起点 (默认 RELEASE_MAX)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
for k, v in TC.PHYS.items():                 # 物理规矩必须在 import env 之前
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("clean3_smoke")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402
import clean_env as CE  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

raw = CE.CleanHoldEnv(CE.build_cfg(args.num_envs))
if args.release_row is not None:
    raw.release_row_cur = int(args.release_row)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
ok_all = True
for mode in ("zero", "random"):
    obs = env.reset()
    assert obs["obs"].shape == (args.num_envs, CE.OBS_DIM), obs["obs"].shape
    assert obs["priv_info"].shape == (args.num_envs, CE.PRIV_DIM), obs["priv_info"].shape
    rew_sum = torch.zeros(args.num_envs, device=raw.device); done_at = torch.full((args.num_envs,), -1, device=raw.device)
    for step in range(TC.T_EP + 2):
        a = (torch.zeros(args.num_envs, CE.ACT_DIM, device=raw.device) if mode == "zero"
             else 0.3 * torch.randn(args.num_envs, CE.ACT_DIM, device=raw.device))
        obs, rew, done, info = env.step(a)
        assert torch.isfinite(obs["obs"]).all() and torch.isfinite(rew).all(), step
        rew_sum += rew
        done_at = torch.where((done > 0) & (done_at < 0), torch.full_like(done_at, step), done_at)
        t = raw._tick
        if step in (0, TC.K_CLOSE, raw.release_row_cur - 1, raw.release_row_cur + 1, raw.release_row_cur + 12, TC.T_EP - 1):
            F = {s: t["F"][s][0].cpu().numpy().round(1).tolist() for s in ("left", "right")}
            print(f"[smoke:{mode}] step {step:3d} released={int(t['released'][0])} "
                  f"dev L {t['dp']['left'][0]*100:.2f}cm/{np.degrees(float(t['dr']['left'][0])):.1f}° "
                  f"R {t['dp']['right'][0]*100:.2f}cm/{np.degrees(float(t['dr']['right'][0])):.1f}° "
                  f"within={int(t['within'][0])} plate_ok={int(t['plate_ok'][0])} sponge_ok={int(t['sponge_ok'][0])} "
                  f"cert={int(raw.cert[0])} dropped={int(raw.dropped[0])} F_L={F['left']} F_R={F['right']} rew={float(rew[0]):.3f}", flush=True)
    rates = raw.pop_rates()
    print(f"[smoke:{mode}] ep_rew mean={float(rew_sum.mean()):.2f} done_at={done_at.cpu().tolist()} rates={ {k: round(v, 3) for k, v in rates.items()} }", flush=True)
print(f"[smoke_clean] PASS envs={args.num_envs} T_EP={TC.T_EP} obs={CE.OBS_DIM} act={CE.ACT_DIM}", flush=True)
sys.stdout.flush()
try: _slot.release()
except Exception: pass
os._exit(0)
