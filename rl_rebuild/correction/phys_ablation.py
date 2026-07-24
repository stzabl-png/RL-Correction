"""物理/控制器消融 — 定位"同一条 cuRobo 轨迹在 OCIR 验证器成功、在 RL env 失败"的原因.

两个正交开关:
  --kinematic   手根改成运动学搬运 (复刻 OCIR: 逐子步改写根位姿, 无根动力学),
                否则用 RL env 原本的 wrench-PD 浮动根.
  --phys_align  把物体/关节物理参数对齐 OCIR 验证器 (depenetration/速度上限/
                求解器迭代/contact-rest offset).

零残差回放 pp0_anchor 参考轨迹, 看物体是否被抬起 9.3cm (OCIR 验证器的结果).

  .venv/bin/python -m rl_rebuild.correction.phys_ablation --clip pp0_anchor \
      --rsi_prob 0 --num_envs 4 --headless [--kinematic] [--phys_align]
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default="pp0_anchor")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--rsi_prob", type=float, default=0.0)
parser.add_argument("--kinematic", action="store_true", help="手根运动学搬运 (复刻 OCIR)")
parser.add_argument("--phys_align", action="store_true", help="物理参数对齐 OCIR 验证器")
parser.add_argument("--action_noise", type=float, default=0.0,
                    help="逐步高斯动作噪声 std (归一化动作单位); 0.1 = sigma_floor")
parser.add_argument("--tag", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg  # noqa: E402


class KinematicHandEnv(SharpaCorrectionEnv):
    """手根运动学搬运: 每个物理子步把根位姿直接写成目标位姿 (无 wrench-PD).

    复刻 OCIR 验证器的 "kinematic anchor transport" —— 手是固定基座 articulation,
    锚点跟随 wrapper Xform, 整手刚性瞬移, 接触不会把手顶开.
    """

    def _apply_action(self) -> None:
        self.hand.set_joint_position_target(self.finger_tgt)
        pose = torch.cat([self.wrist_tgt_pos, self.wrist_tgt_quat], dim=1)
        self.hand.write_root_pose_to_sim(pose)
        self.wrench_norm.zero_()


def align_physics(cfg: SharpaCorrectionEnvCfg) -> dict:
    """把物理参数调成 OCIR 验证器同款 (simulate_grasp_traj.py build_object / setup)."""
    changes = {}
    # --- 物体: 弹射上限 + 求解器迭代 + 接触偏移 ---
    o = cfg.object_cfg.spawn
    changes["obj_depen"] = (o.rigid_props.max_depenetration_velocity, 2.0)
    o.rigid_props.max_depenetration_velocity = 2.0
    o.rigid_props.max_linear_velocity = 1.5            # OCIR: 防"西瓜籽"弹射
    o.rigid_props.max_angular_velocity = 4.0 * 180.0 / np.pi
    changes["obj_solver"] = ((o.rigid_props.solver_position_iteration_count,
                              o.rigid_props.solver_velocity_iteration_count), (16, 2))
    o.rigid_props.solver_position_iteration_count = 16
    o.rigid_props.solver_velocity_iteration_count = 2
    changes["obj_offset"] = ((o.collision_props.contact_offset,
                              o.collision_props.rest_offset), (0.004, 0.001))
    o.collision_props.contact_offset = 0.004
    o.collision_props.rest_offset = 0.001
    # --- 手: 求解器迭代 + 接触偏移 + 弹射上限 ---
    r = cfg.robot_cfg.spawn
    changes["hand_solver"] = ((r.articulation_props.solver_position_iteration_count,
                              r.articulation_props.solver_velocity_iteration_count), (20, 10))
    r.articulation_props.solver_position_iteration_count = 20
    r.articulation_props.solver_velocity_iteration_count = 10
    changes["hand_offset"] = ((r.collision_props.contact_offset,
                              r.collision_props.rest_offset), (0.004, 0.001))
    r.collision_props.contact_offset = 0.004
    r.collision_props.rest_offset = 0.001
    changes["hand_depen"] = (r.rigid_props.max_depenetration_velocity, 2.0)
    r.rigid_props.max_depenetration_velocity = 2.0
    return changes


cfg = SharpaCorrectionEnvCfg()
clips.configure_cfg(cfg, args.clip)
cfg.rsi_prob = args.rsi_prob
cfg.scene.num_envs = args.num_envs
mode = f"{'KIN' if args.kinematic else 'WRENCH'}+{'ALIGNED' if args.phys_align else 'ORIG'}"
if args.phys_align:
    ch = align_physics(cfg)
    print(f"[ablation] 物理已对齐 OCIR: {json.dumps({k: str(v) for k, v in ch.items()}, ensure_ascii=False)}")
print(f"[ablation] ===== 模式 {mode} | clip={args.clip} rsi_prob={args.rsi_prob} =====")

EnvCls = KinematicHandEnv if args.kinematic else SharpaCorrectionEnv
env = EnvCls(cfg)
env.reset()

N, dev = env.num_envs, env.device
zero = torch.zeros(N, 28, device=dev)
rec = {k: [] for k in ["obj_z", "obj_xy_drift", "n_contact", "wrist_err", "obj_speed"]}
rew_sum = torch.zeros(N, device=dev)
term_any = torch.zeros(N, dtype=torch.bool, device=dev)

with torch.inference_mode():
    for step in range(env.ep_total):
        act = zero if args.action_noise <= 0 else (
            torch.randn(N, 28, device=dev) * args.action_noise).clamp(-1, 1)
        _, rew, term, trunc, _ = env.step(act)
        rew_sum += rew
        term_any |= term
        t = env._ref_t()
        origins = env.scene.env_origins
        op = env.object.data.root_pos_w - origins
        wp = env.hand.data.root_pos_w - origins
        rec["obj_z"].append(op[:, 2].cpu().numpy())
        rec["obj_xy_drift"].append((op - env.ref_obj_pos[t])[:, :2].norm(dim=1).cpu().numpy())
        rec["n_contact"].append(env._tip_contacts().sum(dim=1).cpu().numpy())
        rec["wrist_err"].append((wp - env.ref_wrist_pos[t]).norm(dim=1).cpu().numpy())
        rec["obj_speed"].append(env.object.data.root_lin_vel_w.norm(dim=1).cpu().numpy())

c = {k: np.stack(v) for k, v in rec.items()}          # (T,N)
z0 = c["obj_z"][0]
lift = c["obj_z"].max(axis=0) - z0
print("\n" + "=" * 74)
print(f"结果 [{mode}]  clip={args.clip}  {N} env × {env.ep_total} 步")
print("=" * 74)
print(f"  物体起始 z        {z0.mean():.4f} m  (桌面 {env.cfg.table_top_z})")
print(f"  物体最大抬升      {lift.mean() * 100:6.2f} cm  (各env: {np.round(lift * 100, 1).tolist()})")
print(f"  物体末端抬升      {(c['obj_z'][-1] - z0).mean() * 100:6.2f} cm")
print(f"  >5cm 抬起的 env    {int((lift > 0.05).sum())}/{N}      >9cm: {int((lift > 0.09).sum())}/{N}")
print(f"  OCIR 验证器基准    9.30 cm")
print(f"  ------")
print(f"  物体水平漂移 最大 {c['obj_xy_drift'].max() * 100:6.2f} cm  末端 {c['obj_xy_drift'][-1].mean() * 100:6.2f} cm")
print(f"     (OCIR 基准 1.28 cm; RL 终止阈值 {env.cfg.term_obj_div * 100:.0f} cm)")
print(f"  物体最大速度      {c['obj_speed'].max():6.2f} m/s  (OCIR 上限 1.5)")
print(f"  指尖接触数 均值   {c['n_contact'].mean():6.2f} / 5   最大 {int(c['n_contact'].max())}")
print(f"  接触>=2 的时间占比 {(c['n_contact'] >= 2).mean() * 100:5.1f}%")
print(f"  腕跟踪误差 均值   {c['wrist_err'].mean() * 100:6.2f} cm  最大 {c['wrist_err'].max() * 100:6.2f} cm")
print(f"  提前终止的 env     {int(term_any.sum())}/{N}")
print(f"  零残差 return     {rew_sum.mean().item():+.2f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "data", f"ablation_{'kin' if args.kinematic else 'wrench'}"
                           f"_{'aligned' if args.phys_align else 'orig'}{args.tag}.npz")
out = os.path.abspath(out)
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, **c, lift=lift, mode=mode)
print(f"  曲线已存 {out}")
env.close()
app.close()
