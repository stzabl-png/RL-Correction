"""为什么 pin_release 的回合 ~15 步就结束: 逐步打印终止原因 (零动作)."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=8); p.add_argument("--steps", type=int, default=140)
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot; _slot = isaac_slot("sweep_pin_probe")
app = AppLauncher(args).app
import torch, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SE = importlib.import_module("sweep_grip_env")
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
raw = SE.SweepEnv(SE.build_cfg(args.num_envs)); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
obs = env.reset()
for t in range(args.steps):
    obs, rew, dones, infos = env.step(torch.zeros(args.num_envs, SE.ACT_DIM, device=raw.device))
    o = raw._tick_out; sig = o["signals"]; g = o.get("grip", {})
    cube_z = raw.cube.data.root_pos_w[:, 2] - raw.cfg.table_top_z
    if t < 12 or t % 10 == 0 or bool(dones.reshape(-1).any()):
        print(f"t={t:3d} row={int(raw.row.max())} elen={int(raw.episode_length_buf.max())} | mouth_clr mm min={float(sig['mouth_clearance'].min())*1000:+.1f} med={float(sig['mouth_clearance'].median())*1000:+.1f} | "
              f"cube_z-table mm min={float(cube_z.min())*1000:+.1f} | pan_tilt max={float(sig['pan_tilt'].max())*57.3:.1f}° | "
              f"term={int(o['terminated'].sum())} succ={int(o['success'].sum())} released={int(raw.released.sum())} died={int(raw.died.sum())} "
              f"dp_r max={float(g['dp_r'].max())*100:.2f}cm dr_r={float(g['dr_r'].max())*57.3:.1f}°", flush=True)
sys.stdout.flush(); os._exit(0)
