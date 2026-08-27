"""物理栈对齐探针: dump 运行期关节增益 + 零动作被动轨迹。

两台机器各跑一次, diff 输出 —— 增益不同 = 找到改写层; 增益同而轨迹不同 = 积分器/求解器层。
  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.probe_gains --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default="Grasp3")
parser.add_argument("--grasp_prior", type=str,
                    default="tasks/pregrasp/priors/Grasp3_candidates/8_5.npz")
parser.add_argument("--prior_yaw", type=float, default=215.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("probe")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only = True
cfg.action_space = 7
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.minimal_no_ff = True
cfg.stance_prob, cfg.retract_ratio = 1.0, 1.0
cfg.direct_grasp_prob = 0.0
cfg.scene.num_envs = 4
env = GraspTaskEnv(cfg)

h = env.hand
jn = list(h.joint_names)
d = h.data
print(f"[probe] sim dt={env.physics_dt} decimation={getattr(env.cfg,'decimation','?')} "
      f"control dt={env.step_dt}")
_arm = [i for i, n in enumerate(jn) if n.startswith("R_arm_j")]
for tag, t in (("stiffness", d.joint_stiffness), ("damping", d.joint_damping),
               ("armature", d.joint_armature), ("friction", d.joint_friction),
               ("vel_limit", d.joint_velocity_limits),
               ("effort_limit", d.joint_effort_limits)):
    try:
        v = t[0, _arm].cpu().numpy()
        print(f"[probe] R臂 {tag:12s} {np.round(v, 3)}")
    except Exception as e:
        print(f"[probe] R臂 {tag:12s} 取不到: {type(e).__name__}")

env.reset()
za = torch.zeros(4, cfg.action_space, device=env.device)
for t in range(120):
    env.step(za)
    if t % 20 == 0:
        w = env.wrist_pos_w[0].cpu().numpy()
        q = h.data.joint_pos[0, _arm].cpu().numpy()
        print(f"[passive] 步{t:3d} 腕 {np.round(w*100,2)}cm | R臂q {np.round(q,4)}")
env.close()
app.close()
