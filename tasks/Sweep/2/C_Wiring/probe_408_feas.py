"""408 数据上方块到底能不能进斗: 零动作(纯母带) 与 给定 ckpt 两种模式, 逐步记方块在簸箕系的 (x,z), entered, 行号."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", default=""); p.add_argument("--num_envs", type=int, default=8); p.add_argument("--steps", type=int, default=520); p.add_argument("--method", default="wo_human")
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot; _slot = isaac_slot("sweep_408_feas")
app = AppLauncher(args).app
import torch, importlib, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SE = importlib.import_module("sweep_grip_env" if os.environ.get("SWEEP_VARIANT") == "grip" else "sweep_env")
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
raw = SE.SweepEnv(SE.build_cfg(args.num_envs, ablation_method=args.method)); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
agent = None
if args.checkpoint:
    from rl_rebuild.algo.ppo.ppo import PPO
    from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
    with open(os.path.join(os.path.dirname(__file__), "ppo_sweep.yaml")) as f: acfg = yaml.safe_load(f)
    acfg["algorithm"]["num_actors"] = args.num_envs
    agent = PPO(env, output_dir="/tmp/sweep_408_feas", full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
    agent.restore_test(args.checkpoint); agent.set_eval()
G = raw.geometry
print(f"[feas] mode={'ckpt' if agent else 'zero'} mouth_z={G.pan_mouth_z} inside_z_min={G.pan_inside_z_min} half_w={G.pan_half_width} y[{G.pan_center_y_min},{G.pan_center_y_max}] contact_row={raw.contact_row}", flush=True)
obs = env.reset(); N = args.num_envs
zmin = np.full(N, 9.9); ent = np.zeros(N, bool); full = np.zeros(N, bool); succ = np.zeros(N, bool); rowmax = np.zeros(N, int); done_at = np.full(N, -1)
with torch.no_grad():
    for t in range(args.steps):
        if agent is None: act = torch.zeros(N, SE.ACT_DIM, device=raw.device)
        else:
            inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}; act = torch.clamp(agent.model.act_inference(inp), -1, 1)
        obs, rew, dones, infos = env.step(act)
        o = raw._tick_out; sig = o["signals"]; c = sig["cube_pan"].cpu().numpy(); d = dones.reshape(-1).bool().cpu().numpy()
        for e in range(N):
            if done_at[e] >= 0: continue
            zmin[e] = min(zmin[e], c[e, 2]); ent[e] |= bool(sig["entered"][e]); full[e] |= bool(sig["fully_inside"][e]); succ[e] |= bool(o["success"][e]); rowmax[e] = int(raw.row[e])
            if d[e]: done_at[e] = t
        if t % 40 == 0 or t < 3:
            e = 0
            print(f"t={t:3d} row={int(raw.row[e])} cube_pan x={c[e,0]*100:+5.1f} y={c[e,1]*100:+5.1f} z={c[e,2]*100:+5.1f}cm | broom_dist {float(sig['broom_dist'][e])*100:4.1f}cm moved {float(sig['moved'][e])*100:4.1f}cm | mouth_clr {float(sig['mouth_clearance'][e])*1000:+.1f}mm | entered={int(sig['entered'][e])} done={int(d[e])}", flush=True)
        if (done_at >= 0).all(): break
print(f"[feas] entered {int(ent.sum())}/{N} fully_inside {int(full.sum())}/{N} success {int(succ.sum())}/{N} early_fail {int(((done_at>=0)&~succ).sum())}/{N} | 方块最深 pan_z 中位 {np.median(zmin)*100:.1f}cm (口沿 {G.pan_mouth_z*100:.1f}, 起点 {(G.pan_mouth_z+G.start_outside)*100:.1f}) | 行号最大 中位 {np.median(rowmax):.0f}/{raw.T} | 回合结束步 {done_at.tolist()}")
sys.stdout.flush(); os._exit(0)
