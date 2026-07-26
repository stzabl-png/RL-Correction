"""M0 验收: 零残差回放 — 手是否开环跟得住参考, 物体被动发生什么.

  .venv-isaac/bin/python -m rl_rebuild.correction.m0_replay --num_envs 4 --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default="clip11")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--grasp_prior", action="store_true")
parser.add_argument("--curobo_guide", action="store_true")
parser.add_argument("--rsi_prob", type=float, default=None, help="覆盖 RSI 概率 (评测用 0)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# GPU 独占槽位: 同一时刻只允许一个 Isaac 进程占 GPU (见 utils/gpu_guard.py).
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("replay")

app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul  # noqa: E402
from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402

cfg = SharpaCorrectionEnvCfg()
clips.configure_cfg(cfg, args.clip)
cfg.use_grasp_prior = args.grasp_prior
cfg.use_curobo_guide = args.curobo_guide
if args.rsi_prob is not None:
    cfg.rsi_prob = args.rsi_prob
cfg.scene.num_envs = args.num_envs
env = SharpaCorrectionEnv(cfg)
env.reset()

N, dev = env.num_envs, env.device
zero = torch.zeros(N, 28, device=dev)
log = {k: [] for k in ["wrist_pos", "wrist_rot", "finger", "obj_pos", "obj_z"]}

rew_sum = torch.zeros(N, device=dev)
with torch.inference_mode():
    for step in range(env.ep_total):
        _, rew, term, trunc, extras = env.step(zero)
        rew_sum += rew
        t = env._ref_t()
        origins = env.scene.env_origins
        wp = env.hand.data.root_pos_w - origins
        wq = env.hand.data.root_quat_w
        log["wrist_pos"].append((wp - env.ref_wrist_pos[t]).norm(dim=1).cpu().numpy())
        q_err = quat_mul(env.ref_wrist_quat[t], quat_conjugate(wq))
        q_err = q_err * torch.sign(q_err[:, 0:1])   # 双覆盖归正
        log["wrist_rot"].append(axis_angle_from_quat(q_err).norm(dim=1).cpu().numpy())
        log["finger"].append((env.hand.data.joint_pos - env.ref_finger[t])
                             .abs().mean(dim=1).cpu().numpy())
        op = env.object.data.root_pos_w - origins
        log["obj_pos"].append((op - env.ref_obj_pos[t]).norm(dim=1).cpu().numpy())
        log["obj_z"].append(op[:, 2].cpu().numpy())

L, t0, hold = env.L, env.t0, env.cfg.hold_steps
settle = env.cfg.settle_steps
curves = {k: np.stack(v) for k, v in log.items()}          # (T, N)
inter0 = settle + env.du.ref.interaction_seg[0] - t0        # 交互段在 episode 里的起点
phases = {
    "静置段": slice(0, settle),
    "接近段": slice(settle, inter0),
    "交互段": slice(inter0, settle + L - t0),
    "hold段": slice(settle + L - t0, settle + L - t0 + hold),
}
print(f"\n== M0 零残差回放: {N} env × {env.ep_total} 步 @20Hz "
      f"(t0={t0}, L={L}, hold={hold}) ==")
print(f"{'指标':<12}" + "".join(f"{p:>16}" for p in phases))
for k, unit, s in [("wrist_pos", "cm", 100), ("wrist_rot", "deg", 57.3),
                   ("finger", "deg", 57.3), ("obj_pos", "cm", 100)]:
    row = [(f"{curves[k][sl].mean() * s:6.1f}/{curves[k][sl].max() * s:<6.1f}"
            if curves[k][sl].size else "   -  ")
           for sl in phases.values()]
    print(f"{k:<12}" + "".join(f"{r:>16}" for r in row) + f"  (均值/最大 {unit})")
obj_z = curves["obj_z"]
print(f"\n物体高度: 起始 {obj_z[0].mean():.3f}m  最高 {obj_z.max():.3f}m  "
      f"结束 {obj_z[-1].mean():.3f}m  (桌面 {env.cfg.table_top_z}m)")
lifted = obj_z.max(axis=0) - obj_z[0] > 0.10
print(f"被动抬升 >10cm 的 env: {int(lifted.sum())}/{N} "
      f"(零残差下不指望成功, 这里只是记录基线)")
print(f"\n零残差 episode return: {rew_sum.mean().item():.2f} ± {rew_sum.std().item():.2f}")
if "log" in env.extras:
    print("episode 结算日志 (reset 时写入):")
    for k, v in sorted(env.extras["log"].items()):
        print(f"  {k:<24} {v:+.4f}")

_out = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "m0_curves.npz"))  # 项目相对, 跨机通用
os.makedirs(os.path.dirname(_out), exist_ok=True)
np.savez(_out, **curves)
print(f"曲线已存 {_out}")
env.close()
app.close()
