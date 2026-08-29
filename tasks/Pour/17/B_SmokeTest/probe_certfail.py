"""认证失败拆解: 在每次认证判决的那一刻, 把五个条件逐条打印出来。
G2 判据 = 瓶升>=5mm & 杯升>=5mm & 右手物相对位移<8mm & 左<8mm & 双手各>=3垫
(全部发生在站位行 190, 与交互段无关)"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=420)
p.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
os.environ["POUR_NO_D6"] = "1"
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("certfail")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

_CW = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "C_Wiring"))
sys.path.insert(0, _CW)
sys.path.insert(0, os.path.abspath(os.path.join(_CW, "..", "A_Design", "L3_Learning")))
import pour_env as PE  # noqa: E402
from progress import CERT_HOLD, CERT_RISE, CERT_SLIP  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

N = 4
cfg = PE.build_cfg(num_envs=N)
raw = PE.PourEnv(cfg)
raw.force_entry = [0] * N
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_CW, "ppo_pour.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = N
agent = PPO(env, output_dir="/tmp/p17cf",
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

obs = env.reset()
PB = raw.PB
n_judge = 0
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]),
               "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        # 判决发生在 phase==2 且 cert_t 将达到 CERT_HOLD 的那一步之后;
        # 我们在 step 之前判断"下一步是否判决", 用当前量近似判决时刻的量
        judging = ((PB.cert_phase == 2) & (PB.cert_t >= CERT_HOLD - 1))
        if bool(judging.any()):
            org = raw.scene.env_origins
            cup, bot = raw._read_objs()
            wr = raw.hand.data.body_pos_w[:, raw.wid["R"]] - org
            wl = raw.hand.data.body_pos_w[:, raw.wid["L"]] - org
            f = raw._pads_f().norm(dim=-1)
            npr = (f[:, :5] > 0.5).sum(dim=1)
            npl = (f[:, 5:] > 0.5).sum(dim=1)
            rise_b = bot[:, 2] - PB.cert_z0[:, 1]
            rise_c = cup[:, 2] - PB.cert_z0[:, 0]
            rel_r = ((wr - bot[:, :3]) - PB.cert_rel0["right"]).norm(dim=1)
            rel_l = ((wl - cup[:, :3]) - PB.cert_rel0["left"]).norm(dim=1)
            for i in range(N):
                if not bool(judging[i]):
                    continue
                n_judge += 1
                c = [float(rise_b[i]) >= CERT_RISE, float(rise_c[i]) >= CERT_RISE,
                     float(rel_r[i]) < CERT_SLIP, float(rel_l[i]) < CERT_SLIP,
                     int(npr[i]) >= 3 and int(npl[i]) >= 3]
                bad = [n for n, ok in zip(["瓶没升", "杯没升", "右滑超", "左滑超",
                                           "垫不足"], c) if not ok]
                print(f"[cf]{args.tag} t{t:3d} env{i} 第{int(PB.cert_try[i])+1}次 "
                      f"瓶升={float(rise_b[i])*1000:+6.1f}mm 杯升={float(rise_c[i])*1000:+6.1f}mm "
                      f"右滑={float(rel_r[i])*1000:5.1f}mm 左滑={float(rel_l[i])*1000:5.1f}mm "
                      f"垫R={int(npr[i])} 垫L={int(npl[i])} -> "
                      f"{'✅过' if not bad else '❌ ' + '/'.join(bad)}", flush=True)
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        if bool(dones.all()):
            break
print(f"[cf]{args.tag} 判决次数={n_judge} G2达成={int(PB.g2.sum())}/{N} "
      f"G1达成={int(PB.g1.sum())}/{N}", flush=True)
print("[cf] 完毕", flush=True)
try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
app.close()
os._exit(0)
