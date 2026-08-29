"""抓握显微探针: 载 ckpt, 同场跑 确定性(mu) vs 随机(sigma) 两组,
逐步打印双手垫数/垫压/G1-G2/认证相位 —— 判"噪声城堡 vs 真本事"。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=420)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
os.environ["POUR_NO_D6"] = "1"
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("grip")
app = AppLauncher(args).app
import numpy as np, torch, yaml
_HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring")
sys.path.insert(0, os.path.abspath(_HERE))
import pour_env as PE
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

N = 8                      # 前4=确定性, 后4=随机
cfg = PE.build_cfg(num_envs=N)
raw = PE.PourEnv(cfg)
raw.force_entry = [0] * N
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_pour.yaml")) as f:
    ac = yaml.safe_load(f)
ac["algorithm"]["num_actors"] = N
agent = PPO(env, output_dir="/tmp/pour17_grip",
            full_config=ConfigWrapper(ac, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()
obs = env.reset()
det = torch.zeros(N, dtype=torch.bool, device=raw.device)
det[:4] = True
pk_r = torch.zeros(N, dtype=torch.long)     # 各env右垫峰值
pk_l = torch.zeros(N, dtype=torch.long)
g1_at = {i: None for i in range(N)}
catt = torch.zeros(N, dtype=torch.long)
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]),
               "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        a = mu.clone()
        if hasattr(agent.model, "a2c_network"):
            pass
        noise = torch.randn_like(mu) * 0.1        # 随机组: 加策略量级噪声
        a = torch.where(det.unsqueeze(1), mu, mu + noise)
        obs, rew, dones, infos = env.step(torch.clamp(a, -1.0, 1.0))
        f = raw._pads_f().norm(dim=-1)
        nr = (f[:, :5] > 0.5).sum(dim=1).cpu()
        nl = (f[:, 5:] > 0.5).sum(dim=1).cpu()
        pk_r = torch.maximum(pk_r, nr); pk_l = torch.maximum(pk_l, nl)
        catt = torch.maximum(catt, raw.PB.cert_try.cpu())
        for i in range(N):
            if g1_at[i] is None and bool(raw.PB.g1[i]):
                g1_at[i] = t
        if t % 60 == 0:
            fr = f[:, :5].sum(dim=1); fl = f[:, 5:].sum(dim=1)
            print(f"[grip] t{t:3d} 确定性组 垫R={nr[:4].tolist()} 垫L={nl[:4].tolist()} "
                  f"压R={[round(float(x),1) for x in fr[:4]]} | 随机组 垫R={nr[4:].tolist()} "
                  f"垫L={nl[4:].tolist()}", flush=True)
print("\n===== 结论 =====", flush=True)
for tag, sl in (("确定性(mu)", slice(0, 4)), ("随机(mu+0.1)", slice(4, 8))):
    g1n = sum(1 for i in range(*sl.indices(N)) if g1_at[i] is not None)
    g2n = int(raw.PB.g2[sl].sum())
    print(f"[grip] {tag}: G1达成 {g1n}/4 | G2达成 {g2n}/4 | "
          f"右垫峰值={pk_r[sl].tolist()} 左垫峰值={pk_l[sl].tolist()} "
          f"认证尝试={catt[sl].tolist()}", flush=True)
print("[grip] 完毕", flush=True)
try: _slot.release()
except Exception: pass
sys.stdout.flush()
app.close(); os._exit(0)
