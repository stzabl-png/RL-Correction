"""逐 env 打印: 簸箕实际位姿 vs 母带该行位姿, 口沿离桌, 倾角 (找出为什么钉到母带位姿还会 failed)."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=8); p.add_argument("--steps", type=int, default=6)
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot; _slot = isaac_slot("sweep_pin_probe2")
app = AppLauncher(args).app
import torch, importlib, numpy as np
from isaaclab.utils.math import quat_apply
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SE = importlib.import_module("sweep_grip_env")
from sweep_env import _qangle
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
raw = SE.SweepEnv(SE.build_cfg(args.num_envs)); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
obs = env.reset()
def dump(tag):
    org = raw.scene.env_origins; row = raw.row.clamp(max=raw.T-1)
    pp = raw.aux.data.root_pos_w - org; pq = raw.aux.data.root_quat_w
    rp = raw.ref_pos[0][row]; rq = raw.ref_quat[0][row]
    lip = torch.tensor([list(SE.SPEC.lip_local[0]), list(SE.SPEC.lip_local[1])], device=raw.device)
    up = quat_apply(pq, torch.tensor([0.,1.,0.], device=raw.device).expand(raw.num_envs,3))
    for e in range(raw.num_envs):
        lw = quat_apply(pq[e].expand(2,4), lip) + pp[e]
        tilt = float(torch.acos(up[e,2].clamp(-1,1)))*57.3
        print(f"  {tag} env{e} row={int(row[e])} elen={int(raw.episode_length_buf[e])} | 簸箕 实际-母带: {float(torch.linalg.vector_norm(pp[e]-rp[e]))*1000:6.1f}mm {float(_qangle(pq[e:e+1],rq[e:e+1]))*57.3:5.1f}° | "
              f"口沿两角离桌 {float(lw[0,2])-raw.cfg.table_top_z:+.4f}/{float(lw[1,2])-raw.cfg.table_top_z:+.4f} m | 倾角 {tilt:5.1f}° | 实际z {float(pp[e,2]):.4f} 母带z {float(rp[e,2]):.4f}", flush=True)
dump("reset后")
for t in range(args.steps):
    obs, rew, dones, infos = env.step(torch.zeros(args.num_envs, SE.ACT_DIM, device=raw.device))
    o = raw._tick_out
    print(f"t={t} term={o['terminated'].int().tolist()} mouth_clr={[round(float(v)*1000,1) for v in o['signals']['mouth_clearance']]}", flush=True)
    if t < 2: dump(f"t={t}")
sys.stdout.flush(); os._exit(0)
