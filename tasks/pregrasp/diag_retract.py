"""退避式起始状态族的可行性扫描 —— "从 GraspPose 沿接近轴往后退" 能不能当 RSI 的起点族。

动机 (2026-08-16 与用户讨论): RSI 需要的是"一族有难度序的起始状态", **不是**"一条人手
轨迹"。若把起点族定义成 `IK(腕位 - d·û)`, 则:
  ① d 天然是课程参数 (d→0 直接抓 / d→D 从头做);
  ② 往**远离物体**的方向退 ⟹ 构造上无碰撞;
  ③ 把退避过程倒放就是可行路径 ⟹ 任务永远可解。
本脚本验证前提: 这一族**IK 有解**且**间隙随 d 单调变好**。若某个 d 之后 IK 就崩,
可用课程范围就受限, 方案要改。

扫三种接近轴 û, 因为选哪个方向本身是设计决策:
  human  = 人手轨迹在交互帧的入射方向 (视频可信的那部分)
  radial = 从物体中心指向腕位 (纯几何)
  palm   = 抓姿掌心法向 (GraspPose 自带)

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.diag_retract --headless \\
        --clip Pour17_bottle --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz \\
        --prior_yaw 19.5 --stance_prefix 60
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--stance_prefix", type=int, default=60)
p.add_argument("--dmax", type=float, default=0.30, help="最大退避距离 (m)")
p.add_argument("--step", type=float, default=0.02, help="扫描步长 (m)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("diag_retract")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.stance_prefix_frames = args.stance_prefix
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()
dev = E.device
org = E.scene.env_origins[0].cpu().numpy().astype(np.float64)

# ---- 抓握腕位姿 (env 局部系) ----
gp = E._grasp_pos_w.cpu().numpy().astype(np.float64)
gq = E._grasp_quat_w.cpu().numpy().astype(np.float64)
obj = E.obj_init_pos.cpu().numpy().astype(np.float64)
K = int(args.stance_prefix)
gs = int(E.grasp_start)

# ---- 三种接近轴 (单位向量, 指向"退避"方向 = 远离物体) ----
axes = {}
_w = E.ref_wrist_pos.cpu().numpy().astype(np.float64)
_i0, _i1 = max(gs - 10, 0), min(gs, len(_w) - 1)
_v = _w[_i1] - _w[_i0]
if np.linalg.norm(_v) > 1e-6:
    axes["human"] = -_v / np.linalg.norm(_v)          # 入射方向的反向 = 退避
_r = gp - obj
if np.linalg.norm(_r[:2]) > 1e-6:
    axes["radial"] = _r / np.linalg.norm(_r)
_pz = quat_to_R(gq) @ np.array([0.0, 0.0, 1.0])       # 掌心法向 (手系 z)
if float(_pz @ (gp - obj)) < 0:
    _pz = -_pz                                        # 取指向远离物体的一侧
axes["palm"] = _pz / np.linalg.norm(_pz)

ik = ArmIK(cfg.hand_side, anchor_link="arm_center", anchor_T=E._anchor_T)
Rg = quat_to_R(gq)
q_seed = E._prior_q_grasp.cpu().numpy().astype(np.float64)

# ---- 物体表面点 (间隙用) ----
objp = None
if E.obj_points is not None:
    _P = E.obj_points.cpu().numpy().astype(np.float64)
    objp = (quat_to_R(E.obj_init_quat.cpu().numpy().astype(np.float64)) @ _P.T).T + obj

bn = list(E.hand.body_names)
side = cfg.hand_side
hand_bids = [i for i, n in enumerate(bn) if n.startswith(f"{side}_")]
ztab = float(cfg.table_top_z)

print(f"\n{'=' * 90}")
print(f"[退避族扫描] clip={args.clip} | 抓握腕位 {np.round(gp * 100, 1).tolist()}cm "
      f"| 物体 {np.round(obj * 100, 1).tolist()}cm | 桌面 z={ztab:.3f}")
for k, v in axes.items():
    print(f"  轴 {k:<7s} = {np.round(v, 3).tolist()}  (仰角 {np.degrees(np.arcsin(v[2])):+.1f}°)")
print(f"{'=' * 90}")

for name, u in axes.items():
    print(f"\n--- 退避轴 = {name} " + "-" * 60)
    print(f"{'d(cm)':>6} {'IK':>4} {'位置误差cm':>10} {'臂外壳->桌cm':>13} "
          f"{'手->物体cm':>11} {'腕->物体cm':>11}")
    qw = q_seed
    for d in np.arange(0.0, args.dmax + 1e-9, args.step):
        tgt = gp + d * u
        r = ik.solve(tgt, Rg, q0=qw, iters=200)
        if r["ok"]:
            qw = r["q"]
        # 把臂摆过去, 量间隙
        q_all = E.hand.data.joint_pos[0].clone()
        q_all[E.arm_jids] = torch.tensor(r["q"], dtype=torch.float32, device=dev)
        q_all[E.hand_jids] = E.q_open           # 退避段手是张开的
        E.hand.write_joint_state_to_sim(q_all.unsqueeze(0),
                                        torch.zeros_like(q_all).unsqueeze(0))
        E.sim.step(render=False)
        E.hand.update(E.sim.get_physics_dt())
        shell = E._shell_clearance_np(ik, r["q"]) if E.shell_pts is not None else float("nan")
        hp = E.hand.data.body_pos_w[0, hand_bids].cpu().numpy().astype(np.float64) - org
        d_obj = (np.linalg.norm(hp[:, None, :] - objp[None], axis=-1).min()
                 if objp is not None else float("nan"))
        d_wr = float(np.linalg.norm(tgt - obj))
        print(f"{d * 100:>6.0f} {'ok' if r['ok'] else '**X**':>4} "
              f"{r['pos_err'] * 100:>10.2f} {shell * 100:>13.2f} "
              f"{d_obj * 100:>11.2f} {d_wr * 100:>11.2f}")

print(f"\n{'=' * 90}")
print("判读: IK 全程 ok 且 臂外壳/手->物体 随 d 单调变大 ⟹ 退避族可用作 RSI 起点族;")
print("      某个 d 之后 IK 崩 ⟹ 课程范围只能到那里, 更远的起点得另想办法。")
print(f"{'=' * 90}\n")
app.close()
