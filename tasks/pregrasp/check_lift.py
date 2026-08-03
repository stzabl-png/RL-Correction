"""查微抬升的关节斜坡 `q_lift` 是不是真的把腕抬起来了。

背景：`_load_grasp_prior` 会按抓握位姿**重解**抬升轨迹：
    for i in range(lift_steps+1):  tgt = gp + [0,0,lift_height*i/lift_steps]
                                   r = ik.solve(tgt, ...);  if r["ok"] or i==0: qw = r["q"]
IK 在某一级解不出来时 `qw` 保持上一级 —— 于是 `q_lift` **静默**退化成常量，
验证段的腕根本不上升，物体自然跟不上，成功率恒 0 而**没有任何报错**。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.check_lift --headless \\
        --clip Grasp3 --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215

判读：第 `verify_lvl` 级(通常 4)的末端 z 应 ≈ +10mm。若 ≈0 就是上面那个静默失败。
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp3")
p.add_argument("--grasp_prior", default="")
p.add_argument("--prior_yaw", type=float, default=-1)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot                    # noqa: E402
_s = isaac_slot("check_lift")
app = AppLauncher(a).app

import numpy as np                                                   # noqa: E402

from rl_rebuild.correction import clips                              # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK                   # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior       # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv                          # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, a.clip)
apply_grasp_prior(cfg, a.grasp_prior, a.prior_yaw)
cfg.scene.num_envs = 4
env = GraspTaskEnv(cfg)

ik = ArmIK(cfg.hand_side, anchor_link="arm_center", anchor_T=env._anchor_T)
ql = env.q_lift.cpu().numpy().astype(np.float64)


def legacy_ramp():
    """复现 2026-08-02 之前的算法: 默认 pos_tol=5e-3(5mm) + 解不出就沿用上一级.

    用于回答"历史上那些 100% 的 run, 它们的抬升斜坡是好的吗" —— 若旧算法在某个
    候选上恰好能抬满, 那条 run 就有效; 抬不满的话它根本不可能训到 100%.
    """
    gp = env._grasp_pos_w.cpu().numpy().astype(np.float64)
    gq = env._grasp_quat_w.cpu().numpy().astype(np.float64)
    from rl_rebuild.correction.kinematics import quat_to_R
    out, qw = [], env._prior_q_grasp.cpu().numpy().astype(np.float64)
    for i in range(cfg.lift_steps + 1):
        tgt = gp + np.array([0.0, 0.0, cfg.lift_height * i / max(cfg.lift_steps, 1)])
        r = ik.solve(tgt, quat_to_R(gq), q0=qw)          # 旧: 默认 pos_tol / iters
        if r["ok"] or i == 0:
            qw = r["q"].copy()
        out.append(qw.copy())
    return np.stack(out)


qlg = legacy_ramp()
print(f"\nq_lift 形状 {ql.shape} | verify 用前 {env.verify_lvl:.0f} 级 "
      f"(每级名义 {cfg.lift_height / cfg.lift_steps * 1000:.1f}mm)")
p0, _ = ik.fk(ql[0])
p0g, _ = ik.fk(qlg[0])
print(f"{'级':>3s} {'末端z 修复后(mm)':>18s} {'末端z 旧算法(mm)':>18s}")
for i in range(min(9, len(ql))):
    zi = (ik.fk(ql[i])[0][2] - p0[2]) * 1000
    zg = (ik.fk(qlg[i])[0][2] - p0g[2]) * 1000
    mark = "  ← verify 用这级" if i == int(env.verify_lvl) else ""
    print(f"{i:3d} {zi:18.2f} {zg:18.2f}{mark}")
_l = int(env.verify_lvl)
print(f"\n旧算法在第 {_l} 级抬 {(ik.fk(qlg[_l])[0][2]-p0g[2])*1000:.2f}mm "
      f"(要求 {cfg.verify_lift_m*1000:.0f}mm) -> "
      f"{'✅ 旧 run 有效' if (ik.fk(qlg[_l])[0][2]-p0g[2]) >= 0.9*cfg.verify_lift_m else '❌ 旧算法在这个候选上也是坏的'}")
print("\n判读: 第 4 级末端 z 应 ≈ +10mm。若 ≈0 -> IK 在抬升方向失败, q_lift 退化成常量,")
print("      验证段腕根本没抬起来 -> 成功率必然恒 0 (且**无任何报错**)")
env.close()
app.close()
