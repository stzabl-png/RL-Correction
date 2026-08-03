"""接近段自检: 零动作 = 沿人手参考轨迹走? (验证前馈增量接对了没)

全部 env 从 t=0 起步 (direct_grasp_prob=0, approach_t0_max=0), 发零动作跑完整个接近窗口,
逐步打印 腕位移 / 参考位移 / 跟踪差 / d_pos / d_rot.

判读:
  跟踪差 < ~1cm            -> 前馈接对了 (零动作复现人手的形状与速度剖面)
  跟踪差 与 参考位移 同量级 -> 前馈**没接上**, 手在原地不动 (dexmate_env.py:384-392 那个坑)
  末端 d_pos / d_rot        -> 策略必须自己走完的缺口 (人手轨迹的系统偏差, 见 PLAN §7.7)

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.check_ff --headless \
        --clip Grasp3 --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/31_7.npz \
        --prior_yaw 215
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--clip", default="Grasp3")
p.add_argument("--grasp_prior", default=""); p.add_argument("--prior_yaw", type=float, default=-1)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot          # noqa: E402
_s = isaac_slot("check_ff"); app = AppLauncher(a).app
import numpy as np, torch                                   # noqa: E402
from rl_rebuild.correction import clips                     # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior   # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv                 # noqa: E402
cfg = GraspTaskCfg(); clips.configure_cfg(cfg, a.clip)
apply_grasp_prior(cfg, a.grasp_prior, a.prior_yaw, approach=True)
cfg.scene.num_envs = 16; cfg.direct_grasp_prob = 0.0; cfg.approach_t0_max = 0.0
cfg.obj_jitter_xy = 0.0
env = GraspTaskEnv(cfg); env.gentle = 1.0
env.reset()
z = torch.zeros((16, cfg.action_space), device=env.device)
org = env.scene.env_origins
print(f"\n{'步':>3s} {'φ':>5s} {'相位':>6s} {'腕位移':>8s} {'参考位移':>9s} "
      f"{'跟踪差':>7s} {'d_pos':>7s} {'d_rot':>7s} {'臂离桌':>7s} {'臂↔躯干':>8s}")
print(f"{'':3s} {'':5s} {'':6s} {'(cm)':>8s} {'(cm)':>9s} {'(cm)':>7s} {'(cm)':>7s} "
      f"{'(°)':>7s} {'(cm)':>7s} {'(cm)':>8s}")
w0 = None
gaps = []
for i in range(env.gs + 22):
    env.step(z)
    s = env._sig
    w = (env.wrist_pos_w - org)[0]
    if w0 is None:
        w0 = w.clone(); r0 = env.ref_wrist_pos[env.ref_t[0]].clone()
    if i % 4 == 0 or i == env.gs:
        dw = (w - w0).norm().item() * 100
        dr = (env.ref_wrist_pos[env.ref_t[0]] - r0).norm().item() * 100
        print(f"{i:3d} {env._phi()[0]:5.2f} {int(env.task_phase[0]):6d} {dw:8.2f} {dr:9.2f} "
              f"{abs(dw-dr):7.2f} {s['d_pos'][0]*100:7.2f} "
              f"{np.degrees(s['d_rot'][0].item()):7.1f} "
              f"{s['arm_gap'].min()*100:7.2f} {s['self_gap'].min()*100:8.2f}")
        gaps.append((float(s['arm_gap'].min()*100), float(s['self_gap'].min()*100)))
n_app = int((env.task_phase == 0).sum())
print(f"\n结束: 仍在接近相位 {n_app}/16 | 已切到抓取 {16-n_app}/16")
g = np.array(gaps)
print(f"\n★ P0.3 阈值标定 (整条接近段的最小间隙, 全部 env):")
print(f"   臂连杆离桌面   最小 {g[:,0].min():6.2f}cm  中位 {np.median(g[:,0]):6.2f}cm")
print(f"   臂↔躯干/另一臂 最小 {g[:,1].min():6.2f}cm  中位 {np.median(g[:,1]):6.2f}cm")
print(f"   ⟹ 阈值必须定在最小值**之下**, 否则会去罚参考轨迹自己")
print(f"平均 d_pos {env._sig['d_pos'].mean()*100:.2f}cm  d_rot "
      f"{np.degrees(env._sig['d_rot'].mean().item()):.1f}°  (阈值 "
      f"{cfg.eps_pos*100:.2f}cm / {np.degrees(cfg.eps_rot):.2f}°)")
env.close(); app.close()
