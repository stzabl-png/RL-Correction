"""死因分解 (Unscrew/17): 策略 (确定性 mu, 或 --zero 零动作) N env 从 t0 跑; 钩住 _reset_idx 在重置前抓每个 env 的死状态:
步/行/env 侧 fail_code/PB 三条死线实测 (瓶倾角 vs 60°, 瓶离参考 vs 35cm, 掉落)/手最低点离桌 (D5 5mm)/左垫数/G 链. 末尾按死因计数."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--checkpoint", default=""); p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--steps", type=int, default=420); p.add_argument("--zero", action="store_true")
AppLauncher.add_app_launcher_args(p); args = p.parse_args()
app = AppLauncher(args).app
import numpy as np, torch, yaml
from isaaclab.utils.math import quat_apply
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "part4", "C_Wiring"))
import task_env as PE
from progress import D2_PRE_TILT, D3_DEV, D1_DROP, D5_BELOW_TABLE, TABLE_Z
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "part4", "C_Wiring")
N = args.num_envs
cfg = PE.build_cfg(num_envs=N); raw = PE.UnscrewEnv(cfg); raw.force_entry = [0] * N
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
agent = None
if not args.zero:
    agent_cfg = yaml.safe_load(open(os.path.join(_HERE, "ppo_task.yaml"))); agent_cfg["algorithm"]["num_actors"] = N
    agent = PPO(env, output_dir="/tmp/unscrew_deathprobe", full_config=ConfigWrapper(agent_cfg, {}, test=True), create_output_dir=False)
    agent.restore_test(args.checkpoint); agent.set_eval()
PB = raw.PB; TABLE_Z = float(TABLE_Z); deaths = []; step_now = [0]
CODES = {0: "无(PB侧)", 10: "D1瓶掉", 11: "D1盖掉", 20: "D2瓶倒", 21: "D2盖倒", 30: "D3瓶偏", 31: "D3盖偏", 4: "D4滑移", 5: "D5插桌", 6: "D6互碰"}
_orig = raw._reset_idx
def _hooked(env_ids):
    ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
    if ids and step_now[0] > 0:
        org = raw.scene.env_origins
        bot = raw.object.data.root_state_w[:, :7].clone(); bot[:, :3] -= org
        cap = raw.aux.data.root_state_w[:, :7].clone(); cap[:, :3] -= org
        k = PB.k.clamp(max=PB.N_ROW - 1)
        ref0 = PB.ref_obj[0][k]; ref1 = PB.ref_obj[1][k]
        upw = quat_apply(bot[:, 3:7], PB.up[0].unsqueeze(0).expand(N, 3))
        tilt = torch.rad2deg(torch.acos((upw[:, 2] / upw.norm(dim=1).clamp(min=1e-9)).clamp(-1, 1)))
        hz = raw.hand.data.body_pos_w[:, raw.hand_bids, 2].min(dim=1).values - org[:, 2]
        f = raw._pads_f().norm(dim=-1); npl = (f[:, :5] > 0.5).sum(1); npr = (f[:, 5:] > 0.5).sum(1)
        for i in ids:
            fc = int(raw.fail_code[i]); row = int(raw.row[i]); kk = int(PB.k[i])
            dev0 = float((bot[i, :3] - ref0[i, :3]).norm()); dev1 = float((cap[i, :3] - ref1[i, :3]).norm())
            g = [int(PB.g1[i]), int(PB.g2[i]), int(PB.g3[i]), int(PB.g4[i])]
            why = CODES.get(fc, str(fc))
            if fc == 0:
                if float(tilt[i]) > np.degrees(D2_PRE_TILT) and not PB.g2[i]: why = "PB:瓶倒(G2前)"
                elif dev0 > D3_DEV: why = "PB:瓶离参考>35cm"
                elif dev1 > D3_DEV: why = "PB:盖离参考>35cm"
                elif float(bot[i, 2]) < TABLE_Z - D1_DROP or float(cap[i, 2]) < TABLE_Z - D1_DROP: why = "PB:掉落"
                elif step_now[0] >= raw.D7 - 1: why = "D7超时"
                elif g[3]: why = "G4成功"
                else: why = "?(PB其他/放置后)"
            deaths.append((step_now[0], row, kk, fc, why, float(tilt[i]), dev0, dev1, float(hz[i]) - TABLE_Z, int(npl[i]), int(npr[i]), g))
            print(f"[death] t{step_now[0]:3d} env{i:2d} 行{row:3d}(交互k{kk:3d}) {why:14s} | 瓶倾 {float(tilt[i]):5.1f}° 瓶离参考 {dev0*100:5.1f}cm 盖离参考 {dev1*100:5.1f}cm 瓶z-桌 {(float(bot[i,2])-TABLE_Z)*100:+5.1f}cm | 手最低-桌 {(float(hz[i])-TABLE_Z)*100:+5.1f}cm | 垫 L{int(npl[i])}/R{int(npr[i])} | G{g}", flush=True)
    return _orig(env_ids)
raw._reset_idx = _hooked
obs = env.reset()
with torch.no_grad():
    for t in range(1, args.steps + 1):
        step_now[0] = t
        if agent is None:
            act = torch.zeros(N, int(getattr(raw, "num_actions", 0) or raw.cfg.action_space), device=raw.device)
        else:
            inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
            act = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, rew, dones, infos = env.step(act)
        if t % 20 == 0:
            f = raw._pads_f().norm(dim=-1); npl = (f[:, :5] > 0.5).sum(1)
            print(f"[t{t:3d}] 行中位 {int(raw.row.float().median())} 活 {int((~PB.done).sum())} 左垫≥3: {int((npl>=3).sum())}/{N} G1 {int(PB.g1.sum())} G2 {int(PB.g2.sum())}", flush=True)
from collections import Counter
print(f"[death] 共 {len(deaths)} 次终止 | 死因: {dict(Counter(d[4] for d in deaths))}")
if deaths:
    rows = np.array([d[1] for d in deaths]); print(f"[death] 死亡行 中位 {int(np.median(rows))} 四分位 {int(np.percentile(rows,25))}~{int(np.percentile(rows,75))} | 死时左垫中位 {int(np.median([d[9] for d in deaths]))} | 手最低-桌 中位 {np.median([d[8] for d in deaths])*100:+.1f}cm | 瓶倾中位 {np.median([d[5] for d in deaths]):.0f}°")
print("[death] done", flush=True)
app.close(); os._exit(0)
