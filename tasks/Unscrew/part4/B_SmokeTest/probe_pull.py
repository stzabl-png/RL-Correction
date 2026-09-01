"""拔出物理探针 (Unscrew/17 拔盖变体, 台账 §2; 训练前必过 —— 取代 probe_thread 的地位)。

detach_mode=pull: 螺纹永远锁死, 唯一脱扣通路 = 沿螺轴向外的持续拉力 ≥ breakaway_pull_n。
对盖直接施加已知世界系轴向力 (旁路接触真值门), 手停在母带首行不参与, 瓶身逐子步钉住:

  A 无外力静置          -> 不脱扣, |pull_ema| < 0.2N, 螺纹角恒 0
  B 零接触门            -> 门开着时外力 1.5×阈值 也**不**能脱扣 (真值门封死幻影)
  C 阈下 (0.5×)         -> 不脱扣, 估计器校准到外加值 ±30%
  D 阈上 (1.5×) 持续    -> 脱扣 (screw_engaged=False), 脱扣后盖质量还原实物 (3g)
  E 向内推 (−1.5×)      -> 不脱扣 (只认向外)

  UNSCREW_CLIP=17 UNSCREW_DETACH=pull SHARPA_WANDB=0 PYTHONPATH=. $PY -u \
      tasks/Unscrew/part4/B_SmokeTest/probe_pull.py --headless
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
_slot = isaac_slot("unscrew_pull_probe")
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
assert spec.detach_mode == "pull", f"clip 不是拔出模式 (detach_mode={spec.detach_mode}); 加 UNSCREW_DETACH=pull"
PULL = float(spec.breakaway_pull_n)
TZ = torch.zeros(N, 1, 3, device=dev)
bot_pose0 = torch.cat([E.object.data.root_pos_w.clone(),
                       E.object.data.root_quat_w.clone()], dim=1)
bot_zero = torch.zeros(N, 6, device=dev)
failures = []


def check(name, ok, detail):
    print(f"[pull] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    if not ok:
        failures.append(name)


def set_force(newton):
    from isaaclab.utils.math import quat_apply
    f = torch.zeros(N, 1, 3, device=dev)
    f[:, 0, :] = newton * quat_apply(
        E.object.data.root_quat_w,
        torch.tensor([0.0, 0.0, 1.0], device=dev).expand(N, 3))   # +螺轴 = 向外
    E.aux.set_external_force_and_torque(f, TZ, body_ids=[0], is_global=True)


def run(steps):
    ema, eng, ang = [], [], []
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
        ema.append(E.screw_pull_ema.clone())
        eng.append(E.screw_engaged.clone())
        ang.append(E.screw_angle.clone())
    return torch.stack(ema), torch.stack(eng), torch.stack(ang)


def fresh(gate):
    set_force(0.0)
    E._thread_tau_contact_gate = gate
    E.force_entry = [0] * N
    E.reset()


print(f"[pull] spec: breakaway_pull={PULL}N m_eff={spec.mass_eff_kg}kg dwell={spec.unlock_dwell_s}s "
      f"ema={spec.torque_ema_s}s dt={E.sim.get_physics_dt():.5f} deci={DECI}", flush=True)

# ---- A: 静置 ----
fresh(gate=False)
ema, eng, ang = run(30)
check("A.静置不脱扣", bool(eng.all()), f"engaged={eng[-1].tolist()}")
check("A.静置零拉力", float(ema[-1].abs().max()) < 0.2, f"pull_ema={float(ema[-1].abs().max()):.3f}N")
check("A.螺纹角恒0", float(ang.abs().max()) < 1e-5, f"max angle {float(torch.rad2deg(ang.abs().max())):.4f}°")

# ---- B: 真值门开着 (手不碰盖) + 阈上外力 -> 不得脱扣 ----
fresh(gate=True)
set_force(1.5 * PULL)
ema, eng, ang = run(40)
check("B.零接触门封幻影", bool(eng.all()) and float(ema.abs().max()) < 0.2,
      f"engaged={eng[-1].tolist()} pull_ema max {float(ema.abs().max()):.3f}N (门应归零)")

# ---- C: 阈下 (旁路门) ----
fresh(gate=False)
set_force(0.5 * PULL)
ema, eng, ang = run(50)
check("C.阈下不脱扣", bool(eng.all()), f"engaged={eng[-1].tolist()}")
f_meas = float(ema[-10:].mean())
check("C.估计器校准", abs(f_meas - 0.5 * PULL) < 0.30 * PULL,
      f"外加 {0.5 * PULL:.2f}N, 估计 {f_meas:.2f}N")

# ---- D: 阈上持续 -> 脱扣 + 质量还原 ----
fresh(gate=False)
set_force(1.5 * PULL)
released = False
for _ in range(8):
    ema, eng, ang = run(20)
    if bool((~E.screw_engaged).any()):
        released = True
        break
check("D.阈上脱扣", released, f"engaged={E.screw_engaged.tolist()} pull_ema={[round(v, 2) for v in E.screw_pull_ema.tolist()]}N")
m_now = E.aux.root_physx_view.get_masses()[:, 0].cpu().numpy()
check("D.脱扣后质量还原", released and bool(np.all(m_now < 0.05)),
      f"masses={np.round(m_now, 4).tolist()} kg (实物 0.003, 咬合期 {spec.mass_eff_kg})")
set_force(0.0)

# ---- E: 向内推 -> 不脱扣 ----
fresh(gate=False)
set_force(-1.5 * PULL)
ema, eng, ang = run(60)
check("E.向内推不脱扣", bool(eng.all()), f"engaged={eng[-1].tolist()} pull_ema={float(ema[-1].mean()):.2f}N")
set_force(0.0)

print(f"[pull] {'★全部通过 ✅' if not failures else '❌ 失败: ' + ', '.join(failures)}", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(1 if failures else 0)
