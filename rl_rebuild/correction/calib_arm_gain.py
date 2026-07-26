"""臂增益标定: 扫 kp/kd 缩放, 看哪一档能真正跟住参考轨迹.

为什么需要标定 (不能直接用 USD 的值):
  USD 里的 stiffness (2449~113) 是**度制**, IsaacLab 转成弧度后 ×57.3
  -> 实际 kp = 140313~6497 Nm/rad. 配上 URDF 的真实力矩上限 (150/80/25 Nm),
  0.06° 的误差就要 158Nm 已经超限 —— 整条臂全程跑在力矩饱和里做 bang-bang 控制,
  稳态停在"重力与被夹住的力矩打平"的地方 (实测末端恒偏 7.4cm).
  那套增益是给**遥操作**调的 (要跟手、要软), 不是给 RL 位置跟踪的.

判据 (按重要性):
  ① 末端跟踪误差  —— 参考轨迹跟不跟得住, 这是主指标
  ② 力矩饱和占比  —— 长期饱和 = 实际在做 bang-bang, 残差动作会失去意义
  ③ 关节速度      —— 增益太高会激起抖动

  $PY -m rl_rebuild.correction.calib_arm_gain --clip Grasp2
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--scale", type=float, required=True,
               help="kp 缩放. ⚠ 一个进程只能建一个 Isaac env, 多档要外层脚本循环")
p.add_argument("--damp_ratio", type=float, default=1.0,
               help="kd 缩放 = kp 缩放 × 此值 (1.0 = 同比例缩)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("calib")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402

s = args.scale
if True:
    cfg = DexmateCorrectionEnvCfg()
    clips.configure_cfg(cfg, args.clip)
    cfg.scene.num_envs = args.num_envs
    cfg.rsi_prob = 0.0
    cfg.arm_stiffness_scale = s
    cfg.arm_damping_scale = s * args.damp_ratio
    cfg.apply_gains()                   # ⚠ 必须显式重填: __post_init__ 早就跑完了
    cfg.term_arm_err = 1e9              # 标定期间不因追不上而提前终止, 否则数据不完整

    env = DexmateCorrectionEnv(cfg)
    env.reset()
    zero = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    ee_err, q_err, sat, qd = [], [], [], []
    for k in range(env.ep_total):
        env.step(zero)
        if k < cfg.settle_steps:        # 静置段是从 reset 位姿收敛的瞬态, 不计入
            continue
        t = env._ref_t()
        ee = env.wrist_pos_w - env.scene.env_origins
        ee_err.append(float((ee - env.ref_wrist_pos[t]).norm(dim=1).mean()))
        q_err.append(float((env.arm_q - env.q_ref[t]).abs().max(dim=1).values.mean()))
        sat.append(float((env.arm_torque_norm >= 0.99).float().mean()))
        qd.append(float(env.arm_qd.abs().max(dim=1).values.mean()))
    ee_err, q_err, sat, qd = map(np.array, (ee_err, q_err, sat, qd))
    r = (s, np.median(ee_err) * 100, np.percentile(ee_err, 95) * 100,
         np.degrees(np.median(q_err)), sat.mean() * 100, float(qd.max()))
    print(f"CALIB_RESULT {r[0]:g} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f} {r[4]:.2f} {r[5]:.3f}")
    print(f"[calib] ×{s:<7g} 末端误差 中位 {r[1]:6.2f}cm  95分位 {r[2]:6.2f}cm  "
          f"关节误差中位 {r[3]:5.2f}°  饱和 {r[4]:5.1f}%  最大关节速度 {r[5]:.2f}rad/s")
sys.stdout.flush()
_slot.release()
os._exit(0)
