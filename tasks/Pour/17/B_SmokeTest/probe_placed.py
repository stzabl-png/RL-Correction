"""placed 探针 (L5-32, 2026-08-31): 回合末瓶子离原位到底多远?

★ 由来: AB1 的 13M ckpt 在新判据下 `放回(placed) = 0.000`, **一条都没有**,
主判据 `G3_pour ∧ placed` 因此全线为 0。查判据发现一个结构性嫌疑:

    placed 要求  距 rest <= 3cm         (M3_POS)
    皮筋容忍     绿3 / 黄5 / **红 8cm**  (LEASH_POS)

⟹ **红档下"完美跟随参考"(皮筋一分不扣)也可能离 rest 8cm, 过不了 3cm 的 placed。**
训练信号的容忍度比判据松 2.7 倍 —— 这不是策略笨。

但"结构上可能"不等于"实际就是"。本探针量真实分布, 把它变成一个数:
  · 若末态距 rest 集中在 3~8cm  ⟹ 假设成立, 松判据或加奖励就能解
  · 若集中在 20cm+              ⟹ 策略根本没往回走, 是另一回事(得查参考跟随)

用法: POUR_REF_NPZ=... POUR_VARIANT=... [POUR_CONF_FLAT=1] POUR_IGNORE_WORLD=1 \
      python probe_placed.py --checkpoint <ckpt> --num_envs 128 --steps 1400 --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=1400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "C_Wiring"))
import pour_env as PE  # noqa: E402
sys.path.insert(0, os.path.join(_HERE, "..", "A_Design", "L3_Learning"))
import progress_batch as PBM  # noqa: E402
from progress import M3_POS, M3_ROT, LEASH_POS  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

cfg = PE.build_cfg(num_envs=args.num_envs)
raw = PE.PourEnv(cfg)
raw.force_entry = [0]
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "..", "C_Wiring", "ppo_pour.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir="/tmp/pour17_placed",
            full_config=ConfigWrapper(agent_cfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

PB = raw.PB
rest_b = PB.rest[1][:3].clone()
rest_c = PB.rest[0][:3].clone()
recs = []          # 每个回合结束时: (瓶距rest, 瓶倾角, 杯距rest, 时钟比例, g3)
best = torch.full((args.num_envs,), 1e9)   # 回合内**最接近** rest 的距离
mr = torch.zeros(args.num_envs)     # 回合内 m3_run 的峰值 (差几步就 placed)
jn = torch.zeros(args.num_envs)     # 位置与倾角**同时**达标的总步数
# ★★ 必须用**这一步之前**的快照。第一版在 env.step() 之后读 `_read_objs()`,
#    而 done 触发时环境**已经自动 reset 过了** —— 读到的是复位后的状态, 于是
#    "末态距 rest" 全是 0.00cm、倾角恒 2.00°、时钟恒 0.00、G3 达成恒 0。
#    那不是"策略完美放回", 是**根本没测到**。
#    (纪律: 恒 0 的诊断量先怀疑"没被写"。这次正是。)
obs = env.reset()
with torch.no_grad():
    for _ in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        cup0, bot0 = raw._read_objs()                      # ← step 之前
        db0 = (bot0[:, :3] - rest_b.unsqueeze(0)).norm(dim=1).cpu()
        R0 = PBM._q2R(bot0[:, 3:7])
        v0 = torch.einsum("nij,j->ni", R0, PB.up)
        tl0 = torch.rad2deg(torch.acos(
            (v0[:, 2] / v0.norm(dim=1).clamp(min=1e-9)).clamp(-1, 1))).cpu()
        dc0 = (cup0[:, :3] - rest_c.unsqueeze(0)).norm(dim=1).cpu()
        kf0 = (PB.k.float() / max(PB.N_ROW - 1, 1)).cpu()
        g30 = PB.g3.clone().cpu()
        # ★用判据**自己**的 hold 计数器: m3_run 是"placed 条件已连续成立几步",
        #   需要 M3_HOLD=15。它直接回答"差几步", 比任何间接量都可靠。
        #   (回合内最近距离 那个量是废的 —— t0 出生时瓶子就在 rest, 恒为 0。)
        mr = torch.maximum(mr, PB.m3_run.cpu())
        # 联合近失: 位置与倾角**同时**满足各自阈值的步数
        # ★必须 gate 在 g3 之后。不 gate 的话, t0 出生时瓶子本来就在 rest 且竖直,
        #   会白送 250+ 步"同时达标", 让这个量看起来与 m3_run=0 自相矛盾。
        #   (2026-08-31 实测: 不 gate 时 100% 回合报 252~540 步, 全是开局那段。)
        jt = ((db0 <= M3_POS) & (tl0 <= float(np.degrees(M3_ROT))) & g30).float()
        jn = jn + jt
        best = torch.minimum(best, db0)
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        fin = (dones.bool().cpu() if torch.is_tensor(dones)
               else torch.zeros(args.num_envs, dtype=torch.bool))
        for i in fin.nonzero(as_tuple=False).squeeze(1).tolist():
            recs.append((float(db0[i]), float(tl0[i]), float(dc0[i]),
                         float(kf0[i]), float(mr[i]), bool(g30[i]), float(jn[i])))
            best[i] = 1e9
            mr[i] = 0.0
            jn[i] = 0.0

print("\n" + "=" * 78)
print("[placed] 回合末瓶子离原位有多远 —— 为什么 placed 恒为 0")
print("=" * 78)
if not recs:
    print("  ⚠ 没采到任何回合结束 —— 步数不够或环境不终止。**不是结论, 是没数据。**")
else:
    a = np.array([[r[0], r[1], r[2], r[3], r[4], r[6]] for r in recs])
    g3 = np.array([r[5] for r in recs])
    print(f"  回合数 {len(recs)} (其中 G3 达成 {int(g3.sum())})")
    print()
    print("  %-22s %8s %8s %8s %8s" % ("量", "P10", "中位", "P90", "均值"))
    for nm, col, sc in (("末态 瓶距rest (cm)", 0, 100), ("末态 瓶倾角 (°)", 1, 1),
                        ("末态 杯距rest (cm)", 2, 100), ("时钟走完比例", 3, 1),
                        ("★m3_run 峰值 (需15)", 4, 1),
                        ("★位置+倾角同时达标步数", 5, 1)):
        c = a[:, col] * sc
        print("  %-22s %8.2f %8.2f %8.2f %8.2f"
              % (nm, np.percentile(c, 10), np.median(c), np.percentile(c, 90), c.mean()))
    print()
    print(f"  判据线: placed 需 距rest <= {M3_POS*100:.0f}cm 且 倾角 <= "
          f"{np.degrees(M3_ROT):.0f}°  (还要保持 15 步)")
    print(f"  皮筋线: 绿{LEASH_POS[2]*100:.0f} / 黄{LEASH_POS[1]*100:.0f} / "
          f"红{LEASH_POS[0]*100:.0f}cm")
    print()
    for thr in (0.03, 0.05, 0.08, 0.12, 0.20):
        print(f"    末态距rest <= {thr*100:4.0f}cm 的回合占比: "
              f"{100*(a[:,0]<=thr).mean():5.1f}%")
    print()
    print(f"    ★m3_run 峰值 >= 15 (即 placed 成立) 的回合: "
          f"{100*(a[:,4]>=15).mean():5.1f}%")
    for t in (1, 5, 10, 14):
        print(f"      峰值 >= {t:2d} 步: {100*(a[:,4]>=t).mean():5.1f}%")
    print(f"    ★位置与倾角**同时**达标过(哪怕一步)的回合: "
          f"{100*(a[:,5]>=1).mean():5.1f}%")
    print()
    print("  ★判读:")
    print("    · 末态集中在 3~8cm ⟹ 皮筋容忍(红8cm) > 判据(3cm) 的结构性问题坐实,")
    print("      松判据或给 placed 加奖励就能解。")
    print("    · 末态 20cm+       ⟹ 策略根本没往回走, 是参考跟随的问题, 另查。")
    print("    · '曾经到过'远高于'末态' ⟹ 瓶子回去过又被推开, 是 hold 15 步的问题。")
print("=" * 78, flush=True)
app.close()
