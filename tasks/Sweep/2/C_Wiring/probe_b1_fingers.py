"""B1 (手指+全程焊) 是不是用手指直接扒方块: 确定性 rollout, 逐步记 手指链节离方块最近距离 vs 刷毛面离方块距离."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--num_envs", type=int, default=16); p.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot; _slot = isaac_slot("sweep_b1_probe")
app = AppLauncher(args).app
import torch, importlib, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SE = importlib.import_module("sweep_grip_env")
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
raw = SE.SweepEnv(SE.build_cfg(args.num_envs, ablation_method="full")); env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(os.path.dirname(__file__), "ppo_sweep.yaml")) as f: acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir="/tmp/sweep_b1_probe", full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint); agent.set_eval()
bn = list(raw.hand.body_names)
fing = [i for i, n in enumerate(bn) if n.startswith("right_") and any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))]
lfing = [i for i, n in enumerate(bn) if n.startswith("left_") and any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))]
print(f"[b1probe] 右手指链节 {len(fing)} 个, 左手指链节 {len(lfing)} 个", flush=True)
obs = env.reset()
N = args.num_envs; done_at = np.full(N, -1); first_move = np.full(N, -1); fmin_at_move = np.full(N, np.nan); bmin_at_move = np.full(N, np.nan)
fin_touch_steps = np.zeros(N); moved_steps = np.zeros(N); succ = np.zeros(N, bool)
lf_touch_steps = np.zeros(N); pan_disp_succ = np.full(N, np.nan); lf_at_succ = np.full(N, np.nan); bd_at_succ = np.full(N, np.nan); pan_speed_succ = np.full(N, np.nan)
pan0 = (raw.aux.data.root_pos_w - raw.scene.env_origins).clone()
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        act = torch.clamp(agent.model.act_inference(inp), -1, 1)
        obs, rew, dones, infos = env.step(act)
        o = raw._tick_out; sig = o["signals"]
        cube = raw.cube.data.root_pos_w
        fp = raw.hand.data.body_pos_w[:, fing]                      # (N,F,3)
        fdist = torch.linalg.vector_norm(fp - cube[:, None, :], dim=-1).amin(1)   # 最近手指链节
        lfp = raw.hand.data.body_pos_w[:, lfing]
        lfdist = torch.linalg.vector_norm(lfp - cube[:, None, :], dim=-1).amin(1)
        pan_disp = torch.linalg.vector_norm((raw.aux.data.root_pos_w - raw.scene.env_origins) - pan0, dim=-1)
        bdist = sig["broom_dist"]                                    # 刷毛面点云最近
        moved = sig["moved"] > raw.geometry.moved_gate
        d = dones.reshape(-1).bool().cpu().numpy()
        for e in range(N):
            if done_at[e] >= 0: continue
            if bool(moved[e]) and first_move[e] < 0:
                first_move[e] = t; fmin_at_move[e] = float(fdist[e]); bmin_at_move[e] = float(bdist[e])
            if bool(moved[e]): moved_steps[e] += 1; fin_touch_steps[e] += float(fdist[e] < 0.03); lf_touch_steps[e] += float(lfdist[e] < 0.03)
            if d[e]:
                done_at[e] = t; succ[e] = bool(o["success"][e])
                if succ[e]: pan_disp_succ[e] = float(pan_disp[e]); lf_at_succ[e] = float(lfdist[e]); bd_at_succ[e] = float(bdist[e]); pan_speed_succ[e] = float(sig["pan_lin_speed"][e])
        if (done_at >= 0).all(): break
print(f"[b1probe] 成功 {int(succ.sum())}/{N} | 首次移动方块的步 中位 {np.nanmedian(first_move[first_move>=0]) if (first_move>=0).any() else 'nan'}")
print(f"[b1probe] 方块开始动的那一步: 最近手指链节离方块 中位 {np.nanmedian(fmin_at_move)*100:.1f}cm | 刷毛面离方块 中位 {np.nanmedian(bmin_at_move)*100:.1f}cm")
print(f"[b1probe] 方块移动期间 手指链节 <3cm 的步占比 中位 {np.nanmedian(np.where(moved_steps>0, fin_touch_steps/np.maximum(moved_steps,1), np.nan)):.2f}")
print(f"[b1probe] 成功回合: 簸箕离起点位移 中位 {np.nanmedian(pan_disp_succ)*100:.1f}cm | 成功时 左手指离方块 {np.nanmedian(lf_at_succ)*100:.1f}cm | 刷面离方块 {np.nanmedian(bd_at_succ)*100:.1f}cm | 簸箕线速 {np.nanmedian(pan_speed_succ):.3f}m/s")
print(f"[b1probe] 方块移动期间 左手指 <3cm 的步占比 中位 {np.nanmedian(np.where(moved_steps>0, lf_touch_steps/np.maximum(moved_steps,1), np.nan)):.2f}")
for e in range(N):
    print(f"   env{e}: succ={int(succ[e])} done@{done_at[e]} 首动@{first_move[e]} 右指{fmin_at_move[e]*100:5.1f}cm 刷面{bmin_at_move[e]*100:5.1f}cm 右指<3cm{(fin_touch_steps[e]/max(moved_steps[e],1)):.2f} 左指<3cm{(lf_touch_steps[e]/max(moved_steps[e],1)):.2f} 成功时:簸箕位移{pan_disp_succ[e]*100:5.1f}cm 左指{lf_at_succ[e]*100:5.1f}cm 刷面{bd_at_succ[e]*100:5.1f}cm")
sys.stdout.flush(); os._exit(0)
