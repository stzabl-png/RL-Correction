"""env 静置对账探针 —— make_reference.py 换基的**实测**输入 (CHECKLIST 顺序第一站)。

  UNSCREW_CLIP=32 SHARPA_WANDB=0 PYTHONPATH=. $PY \
      tasks/Unscrew/part4/C_Wiring/probe_rest.py --headless

产出 A_Design/L2_Reference/<clip>/env_rest.json:
  body_pose/cap_pose   settle 后瓶/盖静置位姿 (env origin 相对, wxyz)
  anchor_T_right/left  实测 arm_center 世界位姿 4x4 (离线 ArmIK 锚;
                       URDF 推导锚有 ~cm 级系统差: 仿真躯干会塌 ~6°)
  stance_arm14/fin44   default_joint_pos 的 [R臂7,L臂7] / [R指22,L指22]
                       (母带机器段端点 = 真实站姿, 不是配置值)

为什么必须实测: 框架冒烟 A 项要求母带静置位与 env <5mm —— 只有 env 自己知道
摆放机制 (listen-to-hand 锚点 + upright 投影 + 贴桌) 最终把瓶摆在哪。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--settle", type=int, default=60)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("unscrew_probe_rest")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402
from rl_rebuild.correction.kinematics import quat_to_R  # noqa: E402

# 母带可能还不存在 (先 probe 后 make_reference): 给 env 一个只建场景的降级路径
_have_ref = os.path.isfile(PE.MASTER)
cfg = PE.build_cfg(num_envs=1)
if _have_ref:
    E = PE.UnscrewEnv(cfg)
    E.force_entry = [0]
    E.reset()
else:
    from tasks.pregrasp.env import GraspTaskEnv
    print("[probe_rest] 母带缺席 -> 基类场景 (只测静置/锚, 不放音)")
    cfg.observation_space = 8   # 基类自算, 占位
    cfg.action_space = 29
    E = GraspTaskEnv(cfg)
    E.reset()
for _ in range(args.settle):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())

org = E.scene.env_origins[0].cpu().numpy()
body = np.concatenate([E.object.data.root_pos_w[0].cpu().numpy() - org,
                       E.object.data.root_quat_w[0].cpu().numpy()])
cap = np.concatenate([E.aux.data.root_pos_w[0].cpu().numpy() - org,
                      E.aux.data.root_quat_w[0].cpu().numpy()])
bn = list(E.hand.body_names)
ac = bn.index("arm_center")
aT = np.eye(4)
aT[:3, :3] = quat_to_R(E.hand.data.body_quat_w[0, ac].cpu().numpy())
aT[:3, 3] = E.hand.data.body_pos_w[0, ac].cpu().numpy() - org
jn = list(E.hand.joint_names)
dq = E.hand.data.default_joint_pos[0].cpu().numpy()
arm14 = [float(dq[jn.index(f"{P}_arm_j{i}")]) for P in ("R", "L")
         for i in range(1, 8)]
# 手指列序 = ref_qpos joint_names (母带 fin_names 同源)
qr = np.load(os.path.join(TC.TAKE_DIR, "ref_qpos_right.npz"), allow_pickle=True)
fins = [str(n) for n in qr["joint_names"]]
fin44 = [float(dq[jn.index(n.replace("right_", f"{s}_"))])
         for s in ("right", "left") for n in fins]
out = {"clip": TC.CLIP_ID,
       "body_pose": [float(v) for v in body],
       "cap_pose": [float(v) for v in cap],
       "anchor_T_right": aT.tolist(),      # arm_center 是躯干中点, 双臂同锚
       "anchor_T_left": aT.tolist(),
       "stance_arm14": arm14, "stance_fin44": fin44,
       "settle_steps": args.settle}
os.makedirs(os.path.dirname(TC.REST_JSON), exist_ok=True)
json.dump(out, open(TC.REST_JSON, "w"), indent=1)
tilt = np.degrees(np.arccos(np.clip(
    quat_to_R(body[3:7])[2, 2], -1, 1)))
print(f"[probe_rest] -> {TC.REST_JSON}")
print(f"[probe_rest] 瓶静置 pos={np.round(body[:3], 4).tolist()} 倾角 {tilt:.1f}° "
      f"| 盖 pos={np.round(cap[:3], 4).tolist()} "
      f"| 盖-瓶 z 差 {(cap[2] - body[2]) * 100:.1f}cm (期望 ~18)")
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
