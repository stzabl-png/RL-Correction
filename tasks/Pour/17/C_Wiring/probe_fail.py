"""死因探针: 载 ckpt, 各进入桶各 4 env 确定性跑, 分类终局 (env侧/进度机/超时/M4)
+ 死亡行号 + ms1 状态。不动训练。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=700)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("pour17_probe")
app = AppLauncher(args).app
import torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pour_env as PE
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

N = 16
cfg = PE.build_cfg(num_envs=N)
raw = PE.PourEnv(cfg)
labels = [e[4] for e in raw.entries]
buckets = [0, labels.index("seam1"), labels.index("green@87"),
           labels.index("seam2_ret")]
raw.force_entry = [buckets[i % 4] for i in range(N)]   # 每桶4个
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "ppo_pour.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = N
agent = PPO(env, output_dir="/tmp/p17probe", full_config=ConfigWrapper(acfg, {}, test=True),
            create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()
obs = env.reset()
res = {}
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]),
               "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        o = raw._tick_out
        for i in range(N):
            if i in res or not bool(dones[i]):
                continue
            kind = ("M4" if bool(raw.PB.ms4[i]) else
                    "timeout" if bool(o["timeout"][i]) else
                    "env侧(D4/D5)" if bool(o["fail_env"][i]) else "进度机(D1/D2/D3/D8)")
            res[i] = (labels[raw.force_entry[i % len(raw.force_entry)]] if False
                      else labels[buckets[i % 4]], kind, t, int(raw.row[i]),
                      bool(raw.PB.ms1[i]))
        if len(res) == N:
            break
print("\n===== 死因探针 =====")
for i in sorted(res):
    b, kind, t, row, m1 = res[i]
    print(f"env{i:2d} [{b:10s}] {kind:16s} @t{t:3d} row={row:3d} M1={int(m1)}")
from collections import Counter
print("汇总:", Counter((b, k) for _, (b, k, *_ ) in
                       sorted((i, v) for i, v in res.items())))
_slot.release() if hasattr(_slot, "release") else None
app.close()
