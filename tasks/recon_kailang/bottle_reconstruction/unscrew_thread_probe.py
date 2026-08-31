"""U40 真实螺纹副物理探针 (H40.0, 训前必过).

对盖直接施加已知世界系轴向扭矩, 逐段验证静锁/解锁/稳态转速/回锁:

  A 无扭矩静置        -> 不转, 保持锁定, |tau_ema| < 5 mN·m
  B 0.02 N·m (阈下)   -> 保持锁定不转, tau_ema 校准到 20 mN·m ±25%
  C 0.06 N·m (阈上)   -> 解锁, 稳态 ω = (0.06-0.015)/0.03 = 1.5 rad/s ±20%
  D 撤扭矩            -> ω 衰减回锁, 角度保持不回退
  E 0.06 N·m 持续     -> 拧满 270° 脱扣 (release)

  OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. \
      CUDA_VISIBLE_DEVICES=1 $PY -u \
      -m tasks.recon_kailang.bottle_reconstruction.unscrew_thread_probe --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="screw_unscrew_cap1_task_real")
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_thread_probe")
app = AppLauncher(args).app

import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.unscrew_env import (  # noqa: E402
    UnscrewTaskCfg,
    UnscrewTaskEnv,
)

env_cfg = UnscrewTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.scene.num_envs = args.num_envs

base = UnscrewTaskEnv(env_cfg, render_mode=None)
env = GymStyleEnvWrapper(base, clip_actions=env_cfg.clip_actions)
raw = env.unwrapped
spec = raw.screw_spec
assert spec.breakaway_torque_nm is not None, "探针需要 real 螺纹 clip"
raw._thread_tau_contact_gate = False   # 外加扭矩审计: 旁路 U40d 接触真值门
N, dev = raw.num_envs, raw.device
zero_act = torch.zeros(N, env_cfg.action_space, device=dev)
fz = torch.zeros(N, 1, 3, device=dev)

failures: list[str] = []


def check(name: str, ok: bool, detail: str):
    tag = "PASS" if ok else "FAIL"
    print(f"[probe] {tag} {name}: {detail}")
    if not ok:
        failures.append(name)


def set_torque(nm: float):
    t = torch.zeros(N, 1, 3, device=dev)
    t[:, 0, 2] = nm
    raw.cap.set_external_force_and_torque(fz, t, body_ids=[0], is_global=True)


def run(steps: int):
    angles, omegas, taus, locked, contact = [], [], [], [], []
    for _ in range(steps):
        _, _, done, _ = env.step(zero_act)
        if bool(done.any()):
            print("[probe] ⚠ 段内发生回合终止/重置, 该段作废重跑")
            return None
        angles.append(raw.screw_angle.clone())
        omegas.append(raw.screw_omega.clone())
        taus.append(raw.screw_tau_ema.clone())
        locked.append(raw.screw_locked.clone())
        contact.append(raw._cap_contacts().sum(dim=1).clone())
    return (torch.stack(angles), torch.stack(omegas), torch.stack(taus),
            torch.stack(locked), torch.stack(contact))


def fresh():
    set_torque(0.0)
    return env.reset()


print(f"[probe] ep_total={raw.ep_total} spec: breakaway={spec.breakaway_torque_nm} "
      f"kinetic={spec.kinetic_torque_nm} viscous={spec.viscous_nms} "
      f"I_eff={spec.inertia_eff_kgm2}")

# ---- A: 无扭矩静置 -----------------------------------------------------
fresh()
out = run(40)
assert out is not None, "段 A 内不该有终止"
ang, om, tau, lk, ct = out
check("A.静置不转", float(ang[-1].abs().max()) < 1e-4,
      f"末角 {torch.rad2deg(ang[-1]).abs().max():.4f}° (指尖接触步占比 "
      f"{(ct > 0).float().mean():.2f})")
check("A.保持锁定", bool(lk.all()), f"全程锁定={bool(lk.all())}")
check("A.静置零力矩", float(tau[-1].abs().max()) < 0.005,
      f"tau_ema={1000 * tau[-1].abs().max():.2f} mN·m")

# ---- B: 阈下 0.02 N·m --------------------------------------------------
fresh()
set_torque(0.02)
out = run(60)
assert out is not None, "段 B 内不该有终止"
ang, om, tau, lk, ct = out
check("B.阈下锁死", bool(lk.all()) and float(ang[-1].abs().max()) < 1e-4,
      f"全程锁定={bool(lk.all())} 末角 {torch.rad2deg(ang[-1]).abs().max():.4f}°")
t_meas = float(tau[-1].mean())
check("B.估计器校准", 0.015 < t_meas < 0.025,
      f"外加 20 mN·m, 估计 {1000 * t_meas:.1f} mN·m")

# ---- C: 阈上 0.06 N·m + D: 撤扭矩回锁 ---------------------------------
# 1.5 rad/s 下 45 步 (2.25s) ≈ 190° < 270°, 不会撞上脱扣终止.
fresh()
set_torque(0.06)
out = run(45)
assert out is not None, "段 C 内不该有终止"
ang, om, tau, lk, ct = out
check("C.阈上解锁", bool((~lk).any()), "曾解锁" if bool((~lk).any()) else "从未解锁")
w_ss = float(om[-25:].mean())
w_ref = (0.06 - spec.kinetic_torque_nm) / spec.viscous_nms
check("C.稳态转速", abs(w_ss - w_ref) < 0.2 * w_ref,
      f"ω_ss={w_ss:.2f} rad/s (期望 {w_ref:.2f} ±20%)")
ang_c = raw.screw_angle.clone()
set_torque(0.0)
out = run(40)
assert out is not None, "段 D 内不该有终止"
ang, om, tau, lk, ct = out
check("D.撤力回锁", bool(lk[-1].all()), f"locked={lk[-1].tolist()}")
check("D.角度保持", float((raw.screw_angle - ang_c).abs().max()) < 0.15,
      f"漂移 {torch.rad2deg((raw.screw_angle - ang_c)).abs().max():.2f}° "
      f"(衰减滑行属预期, 不回退即可)")
check("D.不回退", float((raw.screw_angle - ang_c).min()) > -1e-4,
      f"min Δ {float((raw.screw_angle - ang_c).min()):.5f} rad")

# ---- E: 持续阈上扭矩拧到脱扣 ------------------------------------------
fresh()
set_torque(0.06)
released = False
for k in range(12):
    out = run(30)
    if out is None:   # 脱扣后 v1 环境会以 release 终止 -> done 即成功信号
        released = True
        break
    if bool((~raw.screw_engaged).any()):
        released = True
        break
check("E.拧满脱扣", released,
      f"angle={torch.rad2deg(raw.screw_angle).max():.0f}° "
      f"engaged={raw.screw_engaged.tolist()}")
set_torque(0.0)

print(f"\n[probe] {'全部通过 ✅' if not failures else '失败: ' + ', '.join(failures)}")
env.close()
app.close()
raise SystemExit(1 if failures else 0)
