"""把手静态摆到 Dexonomy GraspPose 上看几何 (只渲染不跑物理, 类比 view_ref).

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.view_prior \
      --clip Grasp5 --grasp_prior tasks/pregrasp/priors/Grasp5.npz

  --pose grasp    (默认) 臂=抓握腕位 IK 解, 指=Dexonomy 抓握构型
  --pose pregrasp 臂=预抓位 IK 解, 指=张开
⚠ 手是被"摆"上去的, 不是物理撑住的 —— 看的是接触几何落点, 不是力.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp5")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--pose", default="grasp", choices=("grasp", "pregrasp"))
p.add_argument("--eye", default="0.55,-0.85,1.45")
p.add_argument("--lookat", default="-0.10,-0.08,0.92")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("viewprior")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.grasp_prior_npz = args.grasp_prior
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0             # 看精确摆姿, 不抖
cfg.closure_init_max = 0.0
cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                       lookat=tuple(float(v) for v in args.lookat.split(",")),
                       origin_type="world", resolution=(1600, 900))
E = GraspTaskEnv(cfg)
E.reset()

W = E.scene.env_origins[0]
# 物体钉到精确初始位 (loader 已做 yaw 旋转)
pose = torch.cat([E.obj_init_pos + W, E.obj_init_quat]).unsqueeze(0)
E.object.write_root_pose_to_sim(pose)
E.object.write_root_velocity_to_sim(torch.zeros(1, 6, device=E.device))

# 手摆姿
q = E.hand.data.default_joint_pos.clone()
if args.pose == "grasp":
    q[:, E.arm_jids] = E._prior_q_grasp
    q[:, E.hand_jids] = E.q_close        # Dexonomy 抓握手型
else:
    q[:, E.arm_jids] = E.q_pregrasp
    q[:, E.hand_jids] = E.q_open
E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
E.hand.set_joint_position_target(q)
E.hand.write_data_to_sim()
E.sim.step(render=False)             # 推一步物理让 body 位姿刷新 (读数用)
E.hand.update(E.sim.get_physics_dt())
E.object.update(E.sim.get_physics_dt())

pad_d = E._pad_dists()[0] * 1000
target = E._target_w()[0] - W
print("\n" + "=" * 70)
print(f"pose={args.pose} | 物体 {np.round((E.obj_init_pos).cpu().numpy(),4).tolist()}"
      f" | 对齐目标(接触质心) {np.round(target.cpu().numpy(),4).tolist()}")
print(f"五垫(elastomer 原点)到物体表面距离 mm: "
      f"{[round(float(v),1) for v in pad_d]}")
print("=" * 70)
print("[view_prior] 转视角看接触落点; 关窗口或 Ctrl-C 退出")
while app.is_running():
    E.sim.render()
E.close()
app.close()
