"""D4/D5 分离诊断 (L5-9 教训版: 全部读数逐步取, 不在终止后读)。
每步自算 滑移量/手部最低点/物体高度, 终止那一步取"前一步"的快照。"""
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
_slot = isaac_slot("d45")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

_CW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring")
sys.path.insert(0, os.path.abspath(_CW))
import pour_env as PE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

N = 8
cfg = PE.build_cfg(num_envs=N)
raw = PE.PourEnv(cfg)
raw.force_entry = [0] * N
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(os.path.abspath(_CW), "ppo_pour.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = N
agent = PPO(env, output_dir="/tmp/p17d45",
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

obs = env.reset()
prev = [None] * N          # 上一步快照 (终止时用它, 不读重置后的值)
res = {}
peakz = torch.zeros(N)
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]),
               "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        # ---- 终止前先采本步快照 ----
        dR = (raw.hand.data.body_pos_w[:, raw.wid["R"]]
              - raw.object.data.root_pos_w).norm(dim=1)
        dL = (raw.hand.data.body_pos_w[:, raw.wid["L"]]
              - raw.aux.data.root_pos_w).norm(dim=1)
        d0 = torch.nan_to_num(raw.grasp_d0, nan=0.0)
        slipR = (dR - d0[:, 0]).abs()
        slipL = (dL - d0[:, 1]).abs()
        handz = raw.hand.data.body_pos_w[:, raw.hand_bids, 2].min(dim=1).values
        org = raw.scene.env_origins
        bz = raw.object.data.root_pos_w[:, 2] - org[:, 2]
        peakz = torch.maximum(peakz, bz.cpu())
        snap = [(float(slipR[i]), float(slipL[i]), float(handz[i]),
                 float(bz[i]), int(raw.PB.g2[i]), int(raw.PB.k[i]), int(raw.row[i]))
                for i in range(N)]
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        o = raw._tick_out
        for i in range(N):
            if i in res or not bool(dones[i]):
                continue
            sr, sl, hz, bzz, g2, k, row = snap[i]
            if bool(o["timeout"][i]):
                why = "超时"
            elif bool(o["fail_env"][i]):
                d4 = max(sr, sl) > PE.D4_SLIP
                d5 = hz < PE.TABLE_Z - 0.005
                why = ("D4滑移" if d4 and not d5 else "D5撞桌" if d5 and not d4
                       else "D4+D5同时" if d4 and d5 else "env侧(两者都不满足?)")
            else:
                why = "进度机(D1/D2/D3/D8)"
            res[i] = (why, t, row, k, g2, sr * 100, sl * 100, hz, bzz)
        if len(res) == N:
            break

print("\n===== D4/D5 分离诊断 (读数全部取自终止前一步) =====")
for i in sorted(res):
    why, t, row, k, g2, sr, sl, hz, bzz = res[i]
    print(f"env{i} {why:14s} @t{t:3d} row={row:3d} 时钟k={k:3d} G2={g2} "
          f"右滑={sr:5.1f}cm 左滑={sl:5.1f}cm 手最低z={hz:.3f} 瓶z={bzz:.3f}")
print(f"[d45] 瓶高峰值(全程) = {[round(float(x), 3) for x in peakz]}")
print(f"[d45] 参考: D4滑移线={PE.D4_SLIP*100:.0f}cm  桌面z={PE.TABLE_Z:.3f}  "
      f"母带交互段瓶应升到 ~1.06")
print("[d45] 完毕", flush=True)
try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
app.close()
os._exit(0)
