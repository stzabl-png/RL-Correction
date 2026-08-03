"""站姿快照: 按给定关节角摆好机器人, 离屏渲染存 png (headless 调姿用).

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.snap_pose --headless \
      --set "L_arm_j2=45,R_arm_j2=-45" --out /tmp/pose.png

--set 覆盖在默认站姿之上 (度; 未提到的关节保持默认). 物理静置 30 步后拍照.
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp5")
p.add_argument("--set", default="", help='如 "L_arm_j2=45,R_arm_j2=-45" (度)')
p.add_argument("--out", required=True)
p.add_argument("--eye", default="0.75,-1.0,1.6")
p.add_argument("--lookat", default="-0.2,0.0,0.95")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("snap")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                       lookat=tuple(float(v) for v in args.lookat.split(",")),
                       origin_type="world", resolution=(1280, 800))
E = GraspTaskEnv(cfg, render_mode="rgb_array")
E.reset()

q = E.hand.data.default_joint_pos.clone()
jn = list(E.hand.joint_names)
for kv in filter(None, args.set.split(",")):
    n, v = kv.split("=")
    n = n.strip()
    assert n in jn, f"没有关节 {n} (可用: 臂 [LR]_arm_j1..7)"
    q[:, jn.index(n)] = float(np.radians(float(v)))
    print(f"[snap] {n} = {v}°")
E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
E.hand.set_joint_position_target(q)
E.hand.write_data_to_sim()
for _ in range(30):
    E.sim.step(render=False)
E.hand.update(E.sim.get_physics_dt())
frame = None
for _ in range(40):                 # RTX 离屏渲染需要预热多帧, 否则拍出来是黑的
    E.sim.render()
    f = E.render()
    if f is not None:
        frame = f
import imageio  # noqa: E402
imageio.imwrite(args.out, frame)
print(f"[snap] -> {args.out}")
E.close()
app.close()
