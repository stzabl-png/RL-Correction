"""U40 真实螺纹副物理探针 (H40.0, 训练前必过 —— 老方法线 LEDGER_unscrew 移植)。

为什么要这条探针: 旧口径的"接触门 + ω 阻尼"是假摩擦替身 —— 有接触就白给转动,
**碰一下瓶盖它就自己转开/脱落**。换成真实摩擦副之后, "捏得紧才拧得动"是由
PhysX 摩擦锥 + 解析螺纹阻力共同裁决的, 必须先在**已知外加扭矩**下逐段验明,
否则训练里读到的每一个 release 都可能是数值幻影 (老台账连续三次尸检)。

对盖直接施加已知世界系轴向扭矩 (旁路 U40d 接触真值门), 手停在母带首行不参与:

  A 无扭矩静置        -> 不转, 保持锁定, |tau_ema| < 5 mN·m
  B breakaway/2       -> 保持锁定不转, tau_ema 校准到外加值 ±25%
  C 1.5×breakaway     -> 解锁, 稳态 ω=(τ-kinetic)/viscous ±20%
  D 撤扭矩            -> 衰减回锁, 角度保持不回退
  E 持续 1.5×         -> 拧满 turns 脱扣 (release)

  UNSCREW_CLIP=32 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. \
      CUDA_VISIBLE_DEVICES=0 $PY -u \
      tasks/Unscrew/part4/B_SmokeTest/probe_thread.py --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("unscrew_thread_probe")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
import task_env as PE  # noqa: E402

N = int(args.num_envs)
cfg = PE.build_cfg(num_envs=N)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0] * N
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
spec = E.screw_spec
assert spec.breakaway_torque_nm is not None, \
    "clip 未带 U40 真实螺纹参数 (breakaway_torque_nm=None => 还是假摩擦口径)"
# 外加扭矩审计: 旁路接触真值门 (手根本不碰盖, 门会把外加扭矩也掐掉)
E._thread_tau_contact_gate = False
BRK = float(spec.breakaway_torque_nm)
FZ = torch.zeros(N, 1, 3, device=dev)
# 瓶身在探针里被逐子步钉住: 反作用扭矩会把桌上的轻瓶旋起来, 污染相对角速度
# (老方法线实测 B 段 1.6× 偏差), 而本探针要测的是螺纹本身。
bot_pose0 = torch.cat([E.object.data.root_pos_w.clone(),
                       E.object.data.root_quat_w.clone()], dim=1)
bot_zero = torch.zeros(N, 6, device=dev)
failures = []


def check(name, ok, detail):
    print(f"[thread] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)


def set_torque(nm):
    t = torch.zeros(N, 1, 3, device=dev)
    axis = E.object.data.root_quat_w                      # 世界系螺轴 = 瓶 z
    from isaaclab.utils.math import quat_apply
    t[:, 0, :] = nm * quat_apply(
        axis, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3))
    E.aux.set_external_force_and_torque(FZ, t, body_ids=[0], is_global=True)


def run(steps):
    """手停在母带首行, 只推物理; 返回逐步 (角度, ω, τ_ema, locked)。"""
    ang, om, tau, lk = [], [], [], []
    tgt = E.ref58[0].unsqueeze(0).expand(N, -1)
    for _ in range(steps):
        full = E.hand.data.joint_pos.clone()
        full[:, E.map_ids_t] = tgt
        E.hand.set_joint_position_target(full)
        for _ in range(DECI):
            E.object.write_root_pose_to_sim(bot_pose0)
            E.object.write_root_velocity_to_sim(bot_zero)
            E._SA.apply_screw(E)
            E.scene.write_data_to_sim()
            E.sim.step(render=False)
            E.scene.update(E.sim.get_physics_dt())
        ang.append(E.screw_angle.clone())
        om.append(E.screw_omega.clone())
        tau.append(E.screw_tau_ema.clone())
        lk.append(E.screw_locked.clone())
    return (torch.stack(ang), torch.stack(om), torch.stack(tau), torch.stack(lk))


def fresh():
    set_torque(0.0)
    E.force_entry = [0] * N
    E.reset()


print(f"[thread] spec: breakaway={BRK} kinetic={spec.kinetic_torque_nm} "
      f"viscous={spec.viscous_nms} I_eff={spec.inertia_eff_kgm2} "
      f"turns={spec.turns} dt={E.sim.get_physics_dt():.5f} deci={DECI}",
      flush=True)

# ---- A: 无扭矩静置 ----
fresh()
ang, om, tau, lk = run(30)
check("A.静置不转", float(ang[-1].abs().max()) < 1e-4,
      f"末角 {float(torch.rad2deg(ang[-1]).abs().max()):.4f}°")
check("A.保持锁定", bool(lk.all()), f"全程锁定={bool(lk.all())}")
check("A.静置零力矩", float(tau[-1].abs().max()) < 0.005,
      f"tau_ema={1000 * float(tau[-1].abs().max()):.2f} mN·m")

# ---- B: 阈下 (breakaway/2) ----
fresh()
set_torque(0.5 * BRK)
ang, om, tau, lk = run(50)
check("B.阈下锁死", bool(lk.all()) and float(ang[-1].abs().max()) < 1e-4,
      f"全程锁定={bool(lk.all())} 末角 "
      f"{float(torch.rad2deg(ang[-1]).abs().max()):.4f}°")
t_meas = float(tau[-1].mean())
check("B.估计器校准", abs(t_meas - 0.5 * BRK) < 0.25 * BRK,
      f"外加 {1000 * 0.5 * BRK:.0f} mN·m, 估计 {1000 * t_meas:.1f} mN·m")

# ---- C: 阈上 (1.5×breakaway) + D: 撤扭矩回锁 ----
fresh()
TAU_C = 1.5 * BRK
set_torque(TAU_C)
ang, om, tau, lk = run(40)
check("C.阈上解锁", bool((~lk).any()), "曾解锁" if bool((~lk).any()) else "从未解锁")
w_ss = float(om[-20:].mean())
w_ref = (TAU_C - spec.kinetic_torque_nm) / spec.viscous_nms
check("C.稳态转速", abs(w_ss - w_ref) < 0.25 * w_ref,
      f"ω_ss={w_ss:.2f} rad/s (解析 {w_ref:.2f} ±25%)")
ang_c = E.screw_angle.clone()
set_torque(0.0)
ang, om, tau, lk = run(30)
check("D.撤力回锁", bool(lk[-1].all()), f"locked={lk[-1].tolist()}")
check("D.不回退", float((E.screw_angle - ang_c).min()) > -1e-4,
      f"min Δ {float((E.screw_angle - ang_c).min()):.5f} rad")
check("D.角度保持", float((E.screw_angle - ang_c).abs().max()) < 0.25,
      f"衰减滑行 {float(torch.rad2deg(E.screw_angle - ang_c).abs().max()):.1f}° "
      f"(不回退即可)")

# ---- E: 持续阈上扭矩拧到脱扣 ----
fresh()
set_torque(TAU_C)
released = False
for _ in range(14):
    run(25)
    if bool((~E.screw_engaged).any()):
        released = True
        break
check("E.拧满脱扣", released,
      f"angle={float(torch.rad2deg(E.screw_angle).max()):.0f}° "
      f"(满 {360 * spec.turns:.0f}°) engaged={E.screw_engaged.tolist()}")
set_torque(0.0)

print(f"[thread] {'★全部通过 ✅' if not failures else '❌ 失败: ' + ', '.join(failures)}",
      flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(1 if failures else 0)
