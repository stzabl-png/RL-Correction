"""倒水几何抽检 (L5-31, 2026-08-31 用户提出): G3 过关的那一刻, 瓶口到底在杯口的哪个方位?

★ 由来: 用户看录像发现"更多是瓶子倾倒后与杯子差不多高度的**左右关系**, 而不是
倒水的**上下关系**"。查判据坐实了这个怀疑:
    m2_now = (tilt >= 90°) & (‖瓶口 − 杯口‖ <= 0.12)
`mouth_gate` 是一个 **3D 欧氏距离**, **完全不区分上下** —— 瓶口在杯口正上方 10cm、
正下方 10cm、正左边 10cm, 判据眼里一模一样。策略完全可以学出"把瓶子横过来杵在
杯子旁边": 倾角 90° ✓ 距离 6cm ✓ 保持 25 步 ✓, 判据全绿而水一滴进不去。

★ 原始重建(人做的那次)在倾角>=90° 的 33 行里实测:
    dz = +1.25cm (瓶口在杯口上方, 84.8% 的帧 dz>0) · 水平 5.62cm · 3D 6.02cm
  ⟹ 真实动作是"瓶口略高于杯口、水平凑近", 水平偏移是竖直高差的 4.5 倍。
  判据阈值 12cm 比真实几何(5~9cm)宽了一倍。

本探针不改判据, 只**量策略实际学出来的几何**, 与上面那组人类基准对照。
输出: 满足 m2_now 的那些 (env,步) 上的 dz / 水平距 / 3D距 / 倾角 分布。

用法:
  POUR_REF_NPZ=... POUR_VARIANT=... [POUR_CONF_FLAT=1] \
  python probe_pour_geometry.py --checkpoint <ckpt> --num_envs 128 --steps 1200 --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=1200)
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
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

cfg = PE.build_cfg(num_envs=args.num_envs)
raw = PE.PourEnv(cfg)
raw.force_entry = [0]                       # t0 口径, 与正式评测一致
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "..", "C_Wiring", "ppo_pour.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir="/tmp/pour17_geom",
            full_config=ConfigWrapper(agent_cfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

PB = raw.PB
rec = []          # (dz, horiz, dist3d, tilt_deg)  —— 只记 m2_now 成立的 (env,步)
n_g3 = 0
obs = env.reset()
with torch.no_grad():
    for _ in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        cup, bot = raw._read_objs()
        # 与 progress_batch 里 G3 的算法**逐字同源**(抄错就白量)
        R1 = PBM._q2R(bot[:, 3:7])
        v = torch.einsum("nij,j->ni", R1, PB.up)
        tilt = torch.acos((v[:, 2] / v.norm(dim=1).clamp(min=1e-9)).clamp(-1, 1))
        mb = bot[:, :3] + torch.einsum("nij,j->ni", R1, PB.mb)
        R0 = PBM._q2R(cup[:, 3:7])
        mc = cup[:, :3] + torch.einsum("nij,j->ni", R0, PB.mc)
        d = mb - mc
        ok = (tilt >= float(np.radians(90))) & (d.norm(dim=1) <= PB.mgate)
        idx = ok.nonzero(as_tuple=False).squeeze(1)
        for i in idx.tolist():
            rec.append((float(d[i, 2]), float(d[i, :2].norm()),
                        float(d[i].norm()), float(np.degrees(tilt[i].item()))))
        n_g3 = int(PB.g3.sum())

print("\n" + "=" * 78)
print("[geom] 倒水几何抽检 —— G3 判据成立(倾角>=90° 且 3D距<=12cm)的那些瞬间")
print("=" * 78)
if not rec:
    print("  ⚠ 没有采到任何满足 m2_now 的样本 —— 该 ckpt 在本次 rollout 里没倒成。")
    print("     这**不是**通过, 是没数据。")
else:
    a = np.array(rec)
    print(f"  样本数 {len(a)} 个 (env,步) · 当前 g3 已达成 {n_g3}/{args.num_envs} env")
    print()
    print("  %-10s %9s %9s %9s %9s" % ("量", "均值", "P10", "中位", "P90"))
    for nm, col, sc in (("dz (cm)", 0, 100), ("水平 (cm)", 1, 100),
                        ("3D距 (cm)", 2, 100), ("倾角 (°)", 3, 1)):
        c = a[:, col] * sc
        print("  %-10s %9.2f %9.2f %9.2f %9.2f"
              % (nm, c.mean(), np.percentile(c, 10), np.median(c), np.percentile(c, 90)))
    print()
    print("  ★ dz>0 (瓶口在杯口**上方**) 占比: %.1f%%" % (100 * (a[:, 0] > 0).mean()))
    print("  ★ 水平距 / |dz| 比值 中位: %.2f  (比值越大 = 越像'横杵在旁边')"
          % np.median(np.abs(a[:, 1]) / np.maximum(np.abs(a[:, 0]), 1e-4)))
    print()
    print("  ── 人类基准(v1 原始重建, 倾角>=90° 的 33 行) ──")
    print("     dz +1.25cm · 水平 5.62cm · 3D 6.02cm · dz>0 占 84.8% · 比值 4.5")
    print()
    print("  判读: dz>0 占比明显低于 84.8%, 或水平/|dz| 比值明显大于 4.5,")
    print("        ⟹ 策略学的是'横杵'而非'悬在上方倒', 该 ckpt 的 G3 数字不可当倒水读。")
print("=" * 78, flush=True)
app.close()
