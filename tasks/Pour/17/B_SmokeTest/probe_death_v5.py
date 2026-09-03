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
import world_fingerprint as _WF  # noqa: E402
_WF.restore_physics_env(_WF.world_json_of(args.checkpoint))  # L5-34: 先还原再建环境
import pour_env as PE
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

N = 16
cfg = PE.build_cfg(num_envs=N)
raw = PE.PourEnv(cfg)
labels = [e[4] for e in raw.entries]
# L5-9: 出生点名已换代 (t0/g1/g2/g3/ret); 全员 t0 = 确定性口径的死因分布
buckets = [0] * 4
raw.force_entry = [0] * N
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "C_Wiring", "ppo_pour.yaml")) as f:
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
            _cup, _bot = o["cup"], o["bot"]
            _bz = float(_bot[i, 2]); _cz = float(_cup[i, 2])
            _dr = float((raw.hand.data.body_pos_w[i, raw.wid["R"]]
                         - raw.object.data.root_pos_w[i]).norm())
            kind = ("G4成功" if bool(raw.PB.g4[i]) else
                    "timeout" if bool(o["timeout"][i]) else
                    "env侧(D4滑移/D5撞桌)" if bool(o["fail_env"][i])
                    else "进度机(D1掉落/D2倾倒/D3甩飞/D8扰动)")
            kind += f" 瓶z={_bz:.3f} 杯z={_cz:.3f} 右手距={_dr*100:.1f}cm"
            res[i] = (labels[raw.force_entry[i % len(raw.force_entry)]] if False
                      else labels[buckets[i % 4]], kind, t, int(raw.row[i]),
                      bool(raw.PB.g2[i]))
        if len(res) == N:
            break
print("\n===== 死因探针 =====")
for i in sorted(res):
    b, kind, t, row, m1 = res[i]
    print(f"env{i:2d} {kind} @t{t:3d} row={row:3d} G2={int(m1)}")
from collections import Counter
print("汇总:", Counter((b, k) for _, (b, k, *_ ) in
                       sorted((i, v) for i, v in res.items())))
_slot.release() if hasattr(_slot, "release") else None
app.close()
