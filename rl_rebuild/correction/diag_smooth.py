"""量化动作平滑度: 逐相位看关节速度 / 加速度 / 加加速度(jerk).

"卡顿"在数值上就是**加速度突变**. 这个脚本定位它出现在哪个相位边界.
"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
from rl_rebuild.utils.gpu_guard import isaac_slot
_s = isaac_slot("smooth"); app = AppLauncher(a).app
import numpy as np, torch
from rl_rebuild.correction import clips
from rl_rebuild.correction.env.registry import make_env
E, C = make_env("dexmate"); cfg = C(); clips.configure_cfg(cfg, a.clip)
cfg.scene.num_envs = a.num_envs; cfg.rsi_prob = 0.0
env = E(cfg); env.reset()
z = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
hz = cfg.target_hz
tgt, act, ph = [], [], []
for k in range(env.ep_total):
    env.step(z)
    tgt.append(env.arm_tgt[0].cpu().numpy())          # 下发的目标
    act.append(env.arm_q[0].cpu().numpy())            # 实际关节角
    ph.append("静置" if k < cfg.settle_steps else
              "接近" if k < cfg.settle_steps + getattr(env, "home_steps", 0) else
              "抓取" if k < env.lift_step0 else "抬升" if k < env.hold_step0 else "hold")
tgt, act, ph = np.array(tgt), np.array(act), np.array(ph)

def deriv(x, n):
    for _ in range(n): x = np.diff(x, axis=0) * hz
    return x
print("\n" + "=" * 88)
print("目标轨迹(策略看到的参考)的光滑度 —— 卡顿 = 加速度突变")
print(f"{'相位':<6}{'速度 max':>12}{'加速度 max':>14}{'jerk max':>14}")
for name in ["静置", "接近", "抓取", "抬升", "hold"]:
    m = ph == name
    if m.sum() < 4: continue
    seg = tgt[m]
    v, ac, j = deriv(seg, 1), deriv(seg, 2), deriv(seg, 3)
    print(f"{name:<6}{np.abs(v).max():>10.3f}r/s{np.abs(ac).max():>11.1f}r/s²"
          f"{np.abs(j).max():>11.0f}r/s³")
# 相位边界处的加速度跳变 (真正让人看到"卡"的地方)
print(f"\n各相位边界前后 3 步的加速度 (跳变越大越卡):")
acc = deriv(tgt, 2)
bnd = [(i, ph[i], ph[i+1]) for i in range(len(ph)-1) if ph[i] != ph[i+1]]
for i, a0, a1 in bnd:
    lo, hi = max(i-3, 0), min(i+4, len(acc))
    print(f"  {a0}->{a1} @步{i}: {np.round(np.abs(acc[lo:hi]).max(axis=1), 2).tolist()}")
print(f"\n实际关节的最大加速度 {np.abs(deriv(act,2)).max():.1f} r/s²  "
      f"(目标 {np.abs(deriv(tgt,2)).max():.1f})")
sys.stdout.flush(); env.close(); _s.release(); os._exit(0)
