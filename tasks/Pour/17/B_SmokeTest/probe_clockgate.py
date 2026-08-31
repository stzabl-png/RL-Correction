"""时钟门探针 (L5-32, 2026-08-31 用户提出): G2 之后为什么接不上物体轨迹?

★ 由来: 用户看录像发现 base_oh / noise1x_oh / noise3x_oh 三条**同一个死法** ——
"G2 之后接不上物体轨迹进行训练"。而 base_o 有完整倒水、走完轨迹。

查代码发现一处结构性不对称:
    皮筋:   dvec = 物体 − ref[k] − lb      ← **减掉**换基偏移 lb
    时钟门: ok   = ‖物体 − ref[k]‖ <= gp    ← **不减** lb
`lb` 是 G1 抓住那一刻捕获的"物体相对参考的偏移"(progress_batch: 药⑤ G1 起换基)。
⟹ 皮筋永远原谅这个偏移, 时钟门完全不认。策略可以"皮筋满分、时钟一步不动"。

但这不解释为什么只有 HYB+红档 死(红档 gp=8cm 反而**更宽**)。所以本探针**量**,不猜:
在 G2 已达成、时钟尚未推进的那些 (env,步) 上记录
    ① ‖物体−ref[k]‖        时钟门看到的距离
    ② ‖物体−ref[k]−lb‖      皮筋看到的距离
    ③ ‖lb‖                 换基偏移本身
    ④ gp / tier            当前门宽与档位
    ⑤ 朝向差 / GATE_ROT     朝向那一项过不过
并分别统计"位置项挡住"与"朝向项挡住"的占比 —— 直接指出是哪一项卡死了时钟。

用法: POUR_REF_NPZ=... POUR_VARIANT=... POUR_IGNORE_WORLD=1 \
      python probe_clockgate.py --checkpoint <ckpt> --num_envs 128 --steps 900 --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=900)
# ★出生点必须可选。第一版写死 force_entry=[0](只从 t0 出生), 于是对
#   "连 G2 都到不了"的 ckpt 一个样本都采不到 —— 而用户在录像里看到的
#   "G2 之后接不上轨迹"恰恰发生在 **RSI 从 g2 出生**的那些回合。
#   探针没覆盖到的地方, 不能当作"没发生"。
parser.add_argument("--entry", type=int, default=0,
                    help="RSI 出生点序号 (启动横幅会打印出生表)")
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
from progress import GATE_POS, RED_GATE_POS, GATE_ROT  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

cfg = PE.build_cfg(num_envs=args.num_envs)
raw = PE.PourEnv(cfg)
raw.force_entry = [args.entry]
print(f"[clockgate] 出生点序号={args.entry} 出生表={[e[4] for e in raw.entries]}",
      flush=True)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "..", "C_Wiring", "ppo_pour.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir="/tmp/pour17_clock",
            full_config=ConfigWrapper(agent_cfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()
PB = raw.PB

rec = []      # (d_clock, d_leash, |lb|, gp, tier, rotdiff, pos_ok, rot_ok, kfrac)
n_g2 = 0
obs = env.reset()
with torch.no_grad():
    for _ in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        cup, bot = raw._read_objs()
        k = PB.k.clamp(max=PB.N_ROW - 1)
        tier = PB.tmix[k]
        gp = torch.where(tier > 0, torch.full_like(tier, 1, dtype=torch.float) * GATE_POS,
                         torch.full_like(tier, 1, dtype=torch.float) * RED_GATE_POS)
        # 与 progress_batch 的时钟门**逐字同源**
        dcl = torch.zeros(args.num_envs, device=bot.device)
        dle = torch.zeros(args.num_envs, device=bot.device)
        lbn = torch.zeros(args.num_envs, device=bot.device)
        rotd = torch.zeros(args.num_envs, device=bot.device)
        for oi, act in ((0, cup), (1, bot)):
            ref = PB.ref_obj[oi][k]
            d0 = (act[:, :3] - ref[:, :3]).norm(dim=1)
            d1 = (act[:, :3] - ref[:, :3]
                  - PB.lb[oi] * PB.lb_set.float().unsqueeze(1)).norm(dim=1)
            dcl = torch.maximum(dcl, d0)
            dle = torch.maximum(dle, d1)
            lbn = torch.maximum(lbn, PB.lb[oi].norm(dim=1))
            rq = PBM._qang(act[:, 3:7] / act[:, 3:7].norm(dim=1, keepdim=True)
                           .clamp(min=1e-9), ref[:, 3:7])
            rotd = torch.maximum(rotd, rq)
        # 只记"G2 已达成 且 时钟还没走完 且 还活着"的时刻
        m = PB.g2 & (PB.k < PB.N_ROW - 1) & (~PB.done)
        n_g2 = max(n_g2, int(PB.g2.sum()))
        for i in m.nonzero(as_tuple=False).squeeze(1).tolist():
            rok = bool(rotd[i] <= GATE_ROT) or int(tier[i]) == 0   # 红档 rot 禁入
            rec.append((float(dcl[i]), float(dle[i]), float(lbn[i]), float(gp[i]),
                        int(tier[i]), float(np.degrees(rotd[i].item())),
                        bool(dcl[i] <= gp[i]), rok,
                        float(PB.k[i]) / max(PB.N_ROW - 1, 1)))
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))

print("\n" + "=" * 78)
print("[clockgate] G2 之后, 时钟为什么不走")
print("=" * 78)
if not rec:
    print(f"  ⚠ 没采到任何 G2 已达成的时刻 (峰值 g2={n_g2}/{args.num_envs})。")
    print("     **不是结论, 是没数据** —— 该 ckpt 连 G2 都到不了, 说明问题更靠前。")
else:
    a = np.array([[r[0], r[1], r[2], r[3], r[5], r[8]] for r in rec])
    pos_ok = np.array([r[6] for r in rec]); rot_ok = np.array([r[7] for r in rec])
    tiers = np.array([r[4] for r in rec])
    print(f"  样本 {len(rec)} 个 (env,步) · 峰值 g2 = {n_g2}/{args.num_envs}")
    print()
    print("  %-24s %8s %8s %8s %8s" % ("量", "P10", "中位", "P90", "均值"))
    for nm, c, sc in (("★时钟门看到的距离(cm)", 0, 100), ("皮筋看到的距离(cm)", 1, 100),
                      ("★换基偏移 |lb| (cm)", 2, 100), ("当前门宽 gp (cm)", 3, 100),
                      ("朝向差 (°)", 4, 1), ("时钟已走比例", 5, 1)):
        v = a[:, c] * sc
        print("  %-24s %8.2f %8.2f %8.2f %8.2f"
              % (nm, np.percentile(v, 10), np.median(v), np.percentile(v, 90), v.mean()))
    print()
    print(f"  档位分布: 绿{int((tiers==2).sum())} 黄{int((tiers==1).sum())} 红{int((tiers==0).sum())}")
    print(f"  门槛: 绿/黄 {GATE_POS*100:.0f}cm · 红 {RED_GATE_POS*100:.0f}cm · "
          f"朝向 {np.degrees(GATE_ROT):.0f}°")
    print()
    print("  ★ 到底是哪一项挡住时钟:")
    print(f"     位置项过     {100*pos_ok.mean():5.1f}%")
    print(f"     朝向项过     {100*rot_ok.mean():5.1f}%")
    print(f"     两项都过     {100*(pos_ok & rot_ok).mean():5.1f}%   ← 时钟能推进的比例")
    print(f"     只被位置挡   {100*((~pos_ok) & rot_ok).mean():5.1f}%")
    print(f"     只被朝向挡   {100*(pos_ok & (~rot_ok)).mean():5.1f}%")
    print(f"     两项都挡     {100*((~pos_ok) & (~rot_ok)).mean():5.1f}%")
    print()
    print("  ★ 换基不对称的实际后果 (皮筋减 lb, 时钟门不减):")
    would = (a[:, 1] <= a[:, 3])       # 若时钟门也减 lb, 位置项会不会过
    print(f"     现在位置项过 {100*pos_ok.mean():5.1f}%  →  若时钟门也减 lb 则 {100*would.mean():5.1f}%")
    print(f"     差值 {100*(would.mean()-pos_ok.mean()):+5.1f} 个百分点 —— "
          f"这就是这处不对称卡掉的量")
print("=" * 78, flush=True)
app.close()
