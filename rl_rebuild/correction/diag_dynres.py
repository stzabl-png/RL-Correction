"""看动态残差在一条 episode 里长什么样 —— 距离、缩放、实际残差幅度随时间的变化."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--sigma", type=float, default=0.3, help="随机动作幅度, 模拟探索")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
from rl_rebuild.utils.gpu_guard import isaac_slot
_s = isaac_slot("dynres"); app = AppLauncher(a).app
import numpy as np, torch
from rl_rebuild.correction import clips
from rl_rebuild.correction.env.registry import make_env
EnvCls, CfgCls = make_env("dexmate")
cfg = CfgCls(); clips.configure_cfg(cfg, a.clip)
cfg.scene.num_envs = a.num_envs; cfg.rsi_prob = 0.0
env = EnvCls(cfg); env.reset()

rows = []
for k in range(env.ep_total):
    act = torch.randn(env.num_envs, cfg.action_space, device=env.device) * a.sigma
    env.step(act.clamp(-1, 1))
    sA, sF, u = env._dyn_res_scale()
    d = env._palm_obj_dist()
    # 实际每步扰动量: 归一化动作 × 界 × 缩放
    arm_amp = (env.res_scale[:7] * sA.mean()).cpu().numpy()      # rad
    fin_amp = float(env.res_scale[7:].mean() * sF.mean())
    ph = ("静置" if k < cfg.settle_steps else
          "抓取" if k < env.lift_step0 else "抬升" if k < env.hold_step0 else "hold")
    rows.append((k, ph, float(d.mean()), float(u.mean()), float(sA.mean()),
                 float(sF.mean()), float(np.degrees(arm_amp).max()), float(np.degrees(fin_amp))))

print("\n" + "=" * 96)
print(f"{'相位':<6}{'掌心离表面':>12}{'u':>7}{'臂缩放':>9}{'指缩放':>9}"
      f"{'臂最大残差':>12}{'指残差':>10}")
import itertools
for ph, grp in itertools.groupby(rows, key=lambda r: r[1]):
    g = list(grp)
    m = lambda i: np.mean([x[i] for x in g])
    print(f"{ph:<6}{m(2)*100:>10.2f}cm{m(3):>7.2f}{m(4):>9.2f}{m(5):>9.2f}"
          f"{m(6):>10.2f}°{m(7):>9.2f}°")
print("=" * 96)
print("期望: 静置/接近段 u≈1 -> 臂放开(3×) 手指几乎不动(0.2×);")
print("      抓取段 u→0 -> 臂收到标定值(1×) 手指放开到 finger_residual_max(1×)")
d0 = np.array([r[2] for r in rows])
print(f"\n掌心到物体表面: 最小 {d0.min()*100:.2f}cm  最大 {d0.max()*100:.2f}cm  "
      f"(d_near={cfg.dyn_d_near*100:.0f}cm d_far={cfg.dyn_d_far*100:.0f}cm)")
u0 = np.array([r[3] for r in rows])
print(f"u 的分布: <0.05 的步数 {int((u0<0.05).sum())}/{len(u0)}, "
      f">0.95 的 {int((u0>0.95).sum())}/{len(u0)}, 中间过渡 {int(((u0>=0.05)&(u0<=0.95)).sum())}")
if (u0 > 0.95).all():
    print("⚠ u 全程贴 1 -> 手从没进入 d_far 以内, d_far 设小了或摆放不对")
if (u0 < 0.05).all():
    print("⚠ u 全程贴 0 -> 手一直在 d_near 内, 动态残差等于没开")
sys.stdout.flush(); env.close(); _s.release(); os._exit(0)
