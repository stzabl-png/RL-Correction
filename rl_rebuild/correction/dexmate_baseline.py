"""零残差 baseline: DexMate 只跟 q_ref, 残差恒为 0, 看能不能抓起来.

这是换机器人之后的**第一个真信号**, 同时回答两件事:
  ① 执行器增益够不够 —— USD 里那套是给遥操作调的(要跟手要软), 不一定适合 RL 跟踪
  ② 换成真机械臂之后这条参考轨迹还抓不抓得起来

飞手侧的同类结论 (phys_ablation, 2026-07-21): 零残差回放能抓, 53% 时间 ≥2 指接触,
抬 6.8cm. 这里就是拿 DexMate 复刻这个对照.

  $PY -m rl_rebuild.correction.dexmate_baseline --clip Grasp2 --num_envs 64
  $PY -m rl_rebuild.correction.dexmate_baseline --clip Grasp2 --video out.mp4   # 录一段看
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--episodes", type=int, default=2)
p.add_argument("--video", default=None, help="录一个 env 的画面到 mp4")
p.add_argument("--eye", default="0.9,0.9,1.35")
p.add_argument("--lookat", default="0.0,0.0,0.90")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
if args.video:
    args.enable_cameras = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("baseline")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402

cfg = DexmateCorrectionEnvCfg()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = args.num_envs
cfg.rsi_prob = 0.0                    # 从头抓, 不要 RSI 抬高数字
if args.video:
    cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                           lookat=tuple(float(v) for v in args.lookat.split(",")),
                           origin_type="world", resolution=(1280, 720))

env = DexmateCorrectionEnv(cfg, render_mode="rgb_array" if args.video else None)
obs, _ = env.reset()
print(f"[baseline] obs {obs['policy'].shape} | 动作 {env.cfg.action_space} | "
      f"episode {env.ep_total} 步 | {args.num_envs} env × {args.episodes} 回合")

zero = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
n_steps = env.ep_total * args.episodes
frames = []
# 逐步统计
trk, tor, cont, lift = [], [], [], []
succ = []
for k in range(n_steps):
    obs, rew, term, trunc, info = env.step(zero)
    # 臂跟踪误差 (关节空间 -> 末端): 直接量末端离参考腕位多远
    t = env._ref_t()
    ee = env.wrist_pos_w - env.scene.env_origins
    trk.append(float((ee - env.ref_wrist_pos[t]).norm(dim=1).mean()) )
    tor.append(float(env.arm_torque_norm.max(dim=1).values.mean()))
    cont.append(float((env._tip_contacts().sum(dim=1) >= 2).float().mean()))
    lift.append(float((env.object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                       - env.obj_rest_z).mean()))
    if "log" in info and "success_rate_t0" in info["log"]:
        succ.append(info["log"]["success_rate_t0"])
    if args.video:
        img = env.render()
        if img is not None:
            frames.append(np.asarray(img))

trk, tor, cont, lift = map(np.array, (trk, tor, cont, lift))
settle = cfg.settle_steps
print("\n" + "=" * 70)
print(f"① 臂跟踪 (末端到参考腕位):  中位 {np.median(trk)*100:5.2f}cm   "
      f"95分位 {np.percentile(trk,95)*100:5.2f}cm   最大 {trk.max()*100:5.2f}cm")
print(f"   静置后(第{settle}步起):     中位 {np.median(trk[settle:])*100:5.2f}cm")
print(f"② 关节力矩 (归一化, 越接近1越顶到电机上限):")
print(f"   中位 {np.median(tor):.3f}   95分位 {np.percentile(tor,95):.3f}   最大 {tor.max():.3f}")
print(f"③ ≥2 指接触的时间占比:      {cont[settle:].mean()*100:5.1f}%")
print(f"④ 物体抬升:                 峰值 {lift.max()*100:5.2f}cm   末值 {lift[-1]*100:5.2f}cm")
if succ:
    print(f"⑤ 成功率(t0):               {np.mean(succ)*100:5.1f}%  ({len(succ)} 次回合结算)")
else:
    print("⑤ 成功率: 本次没有回合结算 (--episodes 调大)")
print("=" * 70)
print("判读: ① >3cm 说明增益跟不上参考, 要调 arm_stiffness;")
print("      ② 长期 >0.9 说明电机顶死 (可能撞到桌子/自碰撞);")
print("      ③④ 与飞手 phys_ablation 的 53% / 6.8cm 对照.")

if frames:
    import imageio.v2 as iio
    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    iio.mimwrite(args.video, frames, fps=20, macro_block_size=1)
    a = np.stack(frames[:5])
    print(f"\n视频 -> {args.video}  {len(frames)} 帧  画面标准差 {a.std():.1f}"
          f"{' (⚠ 疑似纯色)' if a.std() < 5 else ''}")

sys.stdout.flush()
env.close()
_slot.release()
os._exit(0)          # Isaac 关闭流程会挂住, 数已经出完了
