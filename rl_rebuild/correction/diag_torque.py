"""力矩饱和归因: 哪个关节、在哪个相位饱和, 以及饱和会不会吃掉残差动作.

饱和的后果不是"力矩大", 而是**策略的残差在那些时刻失效** —— 电机已经顶满,
再叠一个 Δq 目标也出不了更多力. 所以要看的是"饱和落在任务的哪一段".
"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
from rl_rebuild.utils.gpu_guard import isaac_slot
_s = isaac_slot("diagtorque"); app = AppLauncher(a).app
import numpy as np, torch
from rl_rebuild.correction import clips
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg

cfg = DexmateCorrectionEnvCfg(); clips.configure_cfg(cfg, a.clip)
cfg.scene.num_envs = a.num_envs; cfg.rsi_prob = 0.0
env = DexmateCorrectionEnv(cfg); env.reset()
z = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

pv = env.hand.root_physx_view
try:
    print(f"[solver] 位置迭代 = {pv.get_solver_position_iteration_counts()[0].item()} "
          f"(USD 请求 32, SimulationCfg 上限 {cfg.sim.physx.max_position_iteration_count})")
    print(f"[solver] 速度迭代 = {pv.get_solver_velocity_iteration_counts()[0].item()}")
except Exception as e:
    print(f"[solver] 读不到迭代数: {e}")

sat, tau, qd_ref, ph = [], [], [], []
for k in range(env.ep_total):
    env.step(z)
    t = env._ref_t()
    sat.append((env.arm_torque_norm >= 0.99).float().mean(0).cpu().numpy())
    tau.append(env.arm_torque_norm.mean(0).cpu().numpy())
    t1 = (t + 1).clamp(max=env.L - 1)
    qd_ref.append(float((env.q_ref[t1] - env.q_ref[t]).abs().max(dim=1).values.mean()) * 20)
    ph.append("静置" if k < cfg.settle_steps else
              ("抓取" if k < env.lift_step0 else ("抬升" if k < env.hold_step0 else "hold")))
sat = np.array(sat); tau = np.array(tau); qd_ref = np.array(qd_ref); ph = np.array(ph)

print(f"\n{'关节':<10}" + "".join(f"{s:>10}" for s in ["静置", "抓取", "抬升", "hold", "全程"]))
for j in range(7):
    r = [sat[ph == s, j].mean() * 100 for s in ["静置", "抓取", "抬升", "hold"]]
    print(f"  j{j+1} 饱和%  " + "".join(f"{v:>9.1f}%" for v in r) + f"{sat[:,j].mean()*100:>9.1f}%")
print(f"{'':10}" + "".join(f"{s:>10}" for s in ["静置", "抓取", "抬升", "hold", "全程"]))
for j in range(7):
    r = [tau[ph == s, j].mean() for s in ["静置", "抓取", "抬升", "hold"]]
    print(f"  j{j+1} 均值   " + "".join(f"{v:>10.3f}" for v in r) + f"{tau[:,j].mean():>10.3f}")

any_sat = (sat > 0).any(axis=1)
print(f"\n任一关节饱和的步数占比: {any_sat.mean()*100:.1f}%")
for s in ["静置", "抓取", "抬升", "hold"]:
    m = ph == s
    print(f"  {s}: {any_sat[m].mean()*100:5.1f}%   参考关节速度均值 {qd_ref[m].mean():.3f} rad/s")
c = np.corrcoef(qd_ref, sat.max(axis=1))[0, 1]
print(f"\n饱和 与 参考关节速度 的相关系数 = {c:+.3f}  "
      f"({'快速段引起' if c > 0.3 else '与速度无关 -> 是持续负载(重力/接触)'})")
sys.stdout.flush(); env.close(); _s.release(); os._exit(0)
