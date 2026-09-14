"""零动作逐步门诊断: env0 每步打印 row/gates/broom_dist/moved/entered/fully_inside/rel_speed (2026-09-14, 175 §8)."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=8); p.add_argument("--steps", type=int, default=90); p.add_argument("--method", default="wo_human")
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot; _slot = isaac_slot("sweep_gates")
app = AppLauncher(args).app
import torch, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SE = importlib.import_module("sweep_grip_env" if os.environ.get("SWEEP_VARIANT") == "grip" else "sweep_env")
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
raw = SE.SweepEnv(SE.build_cfg(args.num_envs, ablation_method=args.method)); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
print(f"[gates] contact_row={raw.contact_row} T={raw.T} prelude={getattr(SE,'SCRIPTED_PRELUDE_STEPS','?')}", flush=True)
obs = env.reset(); N = args.num_envs
with torch.no_grad():
    for t in range(args.steps):
        obs, rew, dones, infos = env.step(torch.zeros(N, SE.ACT_DIM, device=raw.device))
        o = raw._tick_out; sig = o["signals"]; g = o["gates"][0].int().tolist(); c = sig["cube_pan"][0].cpu().numpy() * 100
        print(f"t={t:3d} row={int(raw.row[0]):3d} gates={g} near={int(o.get('broom_near', sig.get('broom_near', torch.zeros(N)))[0]) if 'broom_near' in o or 'broom_near' in sig else '?'} "
              f"bd={float(sig['broom_dist'][0])*100:4.1f}cm moved={float(sig['moved'][0])*100:4.1f} ent={int(sig['entered'][0])} full={int(sig['fully_inside'][0])} "
              f"cube=({c[0]:+5.1f},{c[1]:+4.1f},{c[2]:+5.1f}) v={float(sig['rel_speed'][0])*100:4.1f}cm/s succ={int(o['success'][0])} done={int(dones.reshape(-1)[0])}", flush=True)
        if t >= 3 and bool(dones.reshape(-1)[0]): break
gsum = raw._tick_out["gates"].int().sum(0).tolist(); print(f"[gates] {N} env 各门达成数 g1-4={gsum}", flush=True)
sys.stdout.flush(); os._exit(0)
