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

os.environ["POUR_NO_D6"] = "1"  # 静置测量不需要跨臂碰撞探针
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402
from rl_rebuild.correction.kinematics import quat_to_R  # noqa: E402

# 静置测量必须独立于母带：UnscrewEnv.reset() 会用母带首行重置物体，
# 若在已有 bootstrap v1 时走它，会把“实测”变成对占位母带的循环复述。
from tasks.pregrasp.env import GraspTaskEnv
cfg = PE.build_cfg(num_envs=1)
cfg.observation_space = 8   # 基类自算, 占位
cfg.action_space = 29
# build_cfg 的垫序是 前5左+后5右 (10 个); 基类 _pad_signals 按 5 指算
# (finger_active 形状 5)。UnscrewEnv._setup_scene 会自己切片, 但本探针
# 走基类, 要在建场景前切掉右垫 —— 静置测量不消费任何接触信号。
cfg.contact_sensors = list(cfg.contact_sensors)[:5]
E = GraspTaskEnv(cfg)
# 探针不消费观测/动作, 只借场景静置; 观测宽度随基类开关浮动 (实测 167),
# 占位 8 过不了 _check_obs_dim —— 直接豁免, 别追着开关改数
E._check_obs_dim = lambda *_a, **_k: None
E.reset()
print("[probe_rest] 基类场景独立测量（不读取母带）", flush=True)
for _ in range(args.settle):
    # probe_rest bypasses DirectRLEnv.step(), so _apply_action() does not run.
    # Keep the analytic screw constraint alive explicitly; otherwise the cap
    # falls through the collision-filtered bottle and a bogus rest file is
    # produced with cap and bottle on the same table plane.
    E._SA.apply_screw(E, integrate_angle=False)
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
E._SA.apply_screw(E, integrate_angle=False)

org = E.scene.env_origins[0].cpu().numpy()
body = np.concatenate([E.object.data.root_pos_w[0].cpu().numpy() - org,
                       E.object.data.root_quat_w[0].cpu().numpy()])
cap = np.concatenate([E.aux.data.root_pos_w[0].cpu().numpy() - org,
                      E.aux.data.root_quat_w[0].cpu().numpy()])
axis = quat_to_R(body[3:7])[:, 2]
rel = cap[:3] - body[:3]
axial = float(np.dot(rel, axis))
radial = float(np.linalg.norm(rel - axial * axis))
expected_axial = float(E.screw_spec.closed_offset_m)
assert abs(axial - expected_axial) < 0.005, (
    f"瓶盖未保持合拢: axial={axial:.4f}m, expected={expected_axial:.4f}m")
assert radial < 0.005, f"瓶盖偏离螺轴: radial={radial:.4f}m"
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
tmp_json = TC.REST_JSON + ".tmp"
with open(tmp_json, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
os.replace(tmp_json, TC.REST_JSON)
tilt = np.degrees(np.arccos(np.clip(
    quat_to_R(body[3:7])[2, 2], -1, 1)))
print(f"[probe_rest] -> {TC.REST_JSON}")
print(f"[probe_rest] 瓶静置 pos={np.round(body[:3], 4).tolist()} 倾角 {tilt:.1f}° "
      f"| 盖 pos={np.round(cap[:3], 4).tolist()} "
      f"| 轴向差 {axial * 100:.1f}cm (期望 {expected_axial * 100:.1f}) "
      f"| 径向差 {radial * 100:.2f}cm")
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
