"""关节限位诊断: 倒水段"绕一大圈"是不是被 joint limit 逼出来的。
逐步记录 右臂7关节 的 实测q / 参考q / 残差 / 距上下限余量, 找饱和关节。"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=700)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
os.environ["POUR_NO_D6"] = "1"
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("jlim")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

_CW = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "C_Wiring"))
sys.path.insert(0, _CW)
import pour_env as PE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

cfg = PE.build_cfg(num_envs=1)
raw = PE.PourEnv(cfg)
raw.force_entry = [0]
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_CW, "ppo_pour.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = 1
agent = PPO(env, output_dir="/tmp/p17jlim",
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

lim = raw.hand.data.soft_joint_pos_limits[0][raw.map_ids_t[:7]].cpu().numpy()
jn = [raw.hand.joint_names[i] for i in raw.map_ids_t[:7].cpu().tolist()]
print("[jlim] 右臂关节与软限位(度):")
for i, n in enumerate(jn):
    print(f"  {n}: [{np.degrees(lim[i,0]):+7.1f}, {np.degrees(lim[i,1]):+7.1f}]")

obs = env.reset()
rows = []
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]),
               "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        q = raw.hand.data.joint_pos[0, raw.map_ids_t[:7]].cpu().numpy()
        r = int(raw.row[0])
        ref = raw.ref58[min(r, raw.T_ROW - 1), :7].cpu().numpy()
        res = raw.cum_res[0, :7].cpu().numpy()
        rows.append((t, r, int(raw.PB.k[0]), q.copy(), ref.copy(), res.copy()))
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        if bool(dones[0]):
            break

print(f"\n[jlim] 采到 {len(rows)} 步")
print("\n===== 倒水段逐步 (每20步) : 实测q(度) | 离下限 | 离上限 =====")
for t, r, k, q, ref, res in rows:
    if r < 190 or t % 20 != 0:
        continue
    lo = np.degrees(q - lim[:, 0])
    hi = np.degrees(lim[:, 1] - q)
    mn = min(lo.min(), hi.min())
    tag = "★顶限位" if mn < 3.0 else ""
    print(f"t{t:3d} row={r:3d} k={k:3d} q=" +
          " ".join(f"{np.degrees(x):+6.0f}" for x in q) +
          f" | 最小余量={mn:5.1f}° {tag}")

print("\n===== 各关节全程统计 =====")
Q = np.array([x[3] for x in rows])
REF = np.array([x[4] for x in rows])
RES = np.array([x[5] for x in rows])
ia = np.array([x[1] >= 190 for x in rows])
print(f"{'关节':14s}{'实测范围(度)':>20s}{'参考范围(度)':>20s}"
      f"{'最小余量':>10s}{'残差峰(度)':>11s}")
for i, n in enumerate(jn):
    qi, ri = Q[ia, i], REF[ia, i]
    lo = np.degrees(qi - lim[i, 0]).min()
    hi = np.degrees(lim[i, 1] - qi).min()
    print(f"{n:14s}{np.degrees(qi.min()):+8.0f}~{np.degrees(qi.max()):+7.0f}"
          f"{np.degrees(ri.min()):+11.0f}~{np.degrees(ri.max()):+7.0f}"
          f"{min(lo, hi):10.1f}{np.degrees(np.abs(RES[ia, i]).max()):11.1f}")
print("\n[jlim] 判读: 最小余量 <3° = 该关节顶到软限位; 残差峰接近 ±0.08rad(4.6°)"
      " = 残差被界限钳住")
print("[jlim] 完毕", flush=True)
try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
app.close()
os._exit(0)
