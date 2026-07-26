"""确定性评测 — 关掉探索噪声 (只用 mu), 大批量跑完整回合, 数真实成功率.

训练期 TensorBoard 里的 success_rate_t0 有两个噪声源: (a) 动作带 sigma 采样噪声,
(b) 分母是"这一步恰好 reset 的 env 数". 本脚本两个都消除:
所有 env 同时 reset、同时截断, 于是 reset 那一拍的 log 就是全体 env 的成功率.

  .venv/bin/python -m rl_rebuild.correction.eval_policy \
      --checkpoint logs/.../stage1_nn/last.pth --num_envs 2048 --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--zero_action", action="store_true", help="不载策略, 零残差基线对照")
parser.add_argument("--robot", type=str, default="flying", choices=("flying", "dexmate"),
                    help="必须与训练时一致 —— 动作/观测维度不同, 载错会直接维度不匹配")
parser.add_argument("--clip", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--episodes", type=int, default=2, help="连跑几个完整回合")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# GPU 独占槽位: 同一时刻只允许一个 Isaac 进程占 GPU (见 utils/gpu_guard.py).
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("eval")

app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.registry import make_env  # noqa: E402

EnvCls, EnvCfgCls = make_env(args.robot)
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "agents/ppo_correction.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

# clip: 显式给, 否则从 checkpoint 路径 correction_<clip>_<tag> 推断
if args.clip is None:
    assert args.checkpoint, "--zero_action 时必须显式给 --clip"
    _exp = os.path.basename(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(args.checkpoint)))))
    _rest = _exp[len("correction_"):] if _exp.startswith("correction_") else _exp
    args.clip = next((k for k in sorted(clips.CLIPS, key=len, reverse=True)
                      if _rest == k or _rest.startswith(k + "_")), None)
    assert args.clip, f"无法从 {_exp} 推断 clip, 请显式传 --clip"
    print(f"[eval] 自动推断 clip={args.clip}")

env_cfg = EnvCfgCls()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.scene.num_envs = args.num_envs
raw = EnvCls(env_cfg)
env = GymStyleEnvWrapper(raw, clip_actions=env_cfg.clip_actions)

agent = None
if not args.zero_action:
    agent = PPO(env, output_dir="/tmp/_eval_tmp",
                full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
                create_output_dir=False)
    print(f"[eval] loading {args.checkpoint}")
    agent.restore_test(args.checkpoint)
    agent.set_eval()
    for k, v in agent.model.state_dict().items():
        if "sigma" in k:
            print(f"[eval] 训练得到的 sigma mean={torch.exp(v).mean():.4f} "
                  f"(评测不使用, 只跑确定性 mu)")

N, dev = raw.num_envs, raw.device
print(f"\n[eval] {'零残差基线' if args.zero_action else '确定性 mu 策略'} | "
      f"clip={args.clip} | {N} env × {args.episodes} 回合 × {raw.ep_total} 步")

obs_dict = env.reset()
results = []
with torch.no_grad():
    for ep in range(args.episodes):
        lift_max = torch.zeros(N, device=dev)
        contact_sum = torch.zeros(N, device=dev)
        # 逐指 × 分阶段接触计数 (5 指 × 3 阶段: 抓取窗口 / 抬升 / hold)
        finger_phase = torch.zeros(3, 5, device=dev)
        phase_steps = torch.zeros(3, device=dev)
        first_contact = torch.full((N, 5), float('nan'), device=dev)
        for t in range(raw.ep_total):
            if agent is None:
                act = torch.zeros(N, env_cfg.action_space, device=dev)   # 28(飞手) / 29(DexMate)
            else:
                _inp = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"],
                }
                if "pointcloud" in obs_dict:          # 点云策略需要 PointNet 分支
                    _inp["pointcloud"] = obs_dict["pointcloud"]
                act = agent.model.act_inference(_inp)
            obs_dict, r, done, info = env.step(torch.clamp(act, -1.0, 1.0))
            op = raw.object.data.root_pos_w - raw.scene.env_origins
            lift_max = torch.maximum(lift_max, op[:, 2] - raw.obj_rest_z)
            tc = raw._tip_contacts()                     # (N,5) 0/1
            contact_sum += tc.sum(dim=1)
            # 阶段: 0=抓取窗口(含静置) 1=硬编码抬升 2=hold
            ph = 0 if t < raw.lift_step0 else (1 if t < raw.hold_step0 else 2)
            finger_phase[ph] += tc.mean(dim=0)
            phase_steps[ph] += 1
            newc = torch.isnan(first_contact) & (tc > 0)
            first_contact[newc] = float(t)
        log = raw.extras.get("log", {})
        sr = log.get("success_rate_t0", log.get("success_rate", float("nan")))
        results.append(sr)
        print(f"  回合 {ep+1}: 成功率 {sr*100:5.1f}%  |  "
              f"最大抬升 均值 {lift_max.mean()*100:5.2f}cm 中位 {lift_max.median()*100:5.2f}cm  |  "
              f"抬升≥8cm 的 env {int((lift_max >= 0.08).sum())}/{N}  |  "
              f"平均指尖接触数 {contact_sum.mean()/raw.ep_total:.2f}/5")
        if ep == args.episodes - 1:
            names = [n.replace('right_', '').replace('_elastomer', '')
                     for n in env_cfg.fingertip_bodies]
            fp = (finger_phase / phase_steps[:, None].clamp(min=1)).cpu().numpy()
            print(f"\n  ── 逐指 elastomer 接触率 (该阶段内平均有多少比例的 env 该指在接触) ──")
            print(f"  {'阶段':<14s}" + ''.join(f'{n:>9s}' for n in names))
            for i, pn in enumerate(['抓取窗口(80步)', '硬编码抬升(20)', 'hold(40步)']):
                print(f"  {pn:<14s}" + ''.join(f'{fp[i][j]:>9.3f}' for j in range(5)))
            fc = first_contact.cpu().numpy()
            print(f"  {'首次接触步':<14s}" + ''.join(
                f'{np.nanmedian(fc[:, j]):>9.1f}' if np.isfinite(np.nanmedian(fc[:, j])) else f'{"从未":>9s}'
                for j in range(5)) + f"   (静置 0-19, 抓取 20-79, 抬升 80-99, hold 100-139)")

print(f"\n{'='*64}")
print(f"  成功率 (确定性, {N} env × {args.episodes} 回合): "
      f"{np.mean(results)*100:.2f}%  各回合 {[round(x*100,1) for x in results]}")
print(f"  判据: hold 段内 抬升≥{env_cfg.success_lift_height*100:.0f}cm 且 ≥2 指尖接触, "
      f"占 hold 步数 ≥{env_cfg.success_hold_frac*100:.0f}%")
print(f"{'='*64}")
env.close()
app.close()
