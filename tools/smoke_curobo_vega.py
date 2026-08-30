"""cuRobo vega_1p_sharpa 配置冒烟 (不开 Isaac): 建规划器 -> FK -> 一次小位姿规划。

  ~/miniforge3/envs/isaac/bin/python tools/smoke_curobo_vega.py

判据:
  A 规划器可建 (锁死 躯干/头/手指/底座, 只留双臂 14 DoF)
  B FK: 新站姿下 right_hand_C_MC 位置合理 (相对底座, |p| < 1.2m, z > 0)
  C plan_pose: 右腕从站姿到 "FK点 + 5cm 前移" 可解, 终点误差 < 1cm
worker 的调用形状 (MotionPlannerCfg.create/plan_pose/lock_joints) 与
curobo_plan_worker 逐项同款 —— 本测过 = worker 的 cuRobo 侧就绪。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YML = os.path.join(REPO, "datasets", "vega_urdf", "vega_1p_sharpa_curobo.yml")

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose  # noqa: E402

TOOL = ["right_hand_C_MC", "left_hand_C_MC"]
STANCE = {"torso_j1": np.radians(40.5196), "torso_j2": np.radians(73.6595),
          "torso_j3": np.radians(0.3896),
          "L_arm_j1": np.radians(45.0), "R_arm_j1": np.radians(-45.0),
          "L_arm_j4": np.radians(-90.0), "R_arm_j4": np.radians(-90.0),
          "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0}

raw = yaml.safe_load(open(YML))
kin = raw["robot_cfg"]["kinematics"]
kin["tool_frames"] = list(TOOL)
# 锁死 躯干/头/手指 (worker 同款: lock_joints 值 = start_joints)
lj = dict(kin.get("lock_joints") or {})
for n in ("torso_j1", "torso_j2", "torso_j3", "head_j1", "head_j2", "head_j3"):
    lj[n] = float(STANCE[n])
cs_joints = (kin.get("cspace") or {}).get("joint_names") or []
for n in cs_joints:
    if n.startswith(("right_", "left_")):
        lj[n] = 0.0
kin["lock_joints"] = lj
print(f"[smoke] A 建规划器: 锁死 {len(lj)} 关节 ...", flush=True)
cfg = MotionPlannerCfg.create(
    robot=raw["robot_cfg"],
    scene_model=[{"cuboid": {}, "mesh": {}}],
    device_cfg=DeviceCfg(device="cuda:0", dtype=torch.float32),
    self_collision_check=True, max_batch_size=1, multi_env=True,
    max_goalset=1, collision_cache={"cuboid": 10, "mesh": 500},
    num_trajopt_seeds=4, num_ik_seeds=32, use_cuda_graph=False,
    optimizer_collision_activation_distance=0.015)
P = MotionPlanner(cfg)
P.warmup(enable_graph=False, num_warmup_iterations=1)
pj = list(P.joint_names)
print(f"[smoke] A ✅ 活动关节 {len(pj)}: {pj}")
assert all(("arm_j" in n) for n in pj), f"应只剩双臂: {pj}"

q0 = P.default_joint_state.position.clone()
for i, n in enumerate(pj):
    if n in STANCE:
        q0[i] = STANCE[n]
cur = JointState.from_position(q0.unsqueeze(0), joint_names=pj)
ks = P.compute_kinematics(cur)
pr = ks.tool_poses.get_link_pose("right_hand_C_MC", make_contiguous=True)
p0 = pr.position.view(-1)[:3].cpu().numpy()
print(f"[smoke] B FK right_hand_C_MC (底座系) = {np.round(p0, 3).tolist()}")
# 合理域: 手在躯干右前方, 离地 0.7~1.5m (底座在地面, 桌面 0.87)
assert 0.1 < p0[0] < 0.9 and -0.8 < p0[1] < 0.1 and 0.7 < p0[2] < 1.5, \
    "FK 位置离谱, 查 base_link/锁角"

tgt = p0 + np.array([0.05, 0.0, 0.0])
pd = {}
for f in TOOL:
    pp = ks.tool_poses.get_link_pose(f, make_contiguous=True)
    pos = pp.position.view(-1, 3).contiguous()
    qq = pp.quaternion.view(-1, 4).contiguous()
    if f == "right_hand_C_MC":
        pos = torch.tensor(tgt, dtype=torch.float32,
                           device=pos.device).view(1, 3)
    pd[f] = Pose(position=pos, quaternion=qq)
g = GoalToolPose.from_poses(pd, ordered_tool_frames=TOOL, num_goalset=1)
t0 = time.time()
r = P.plan_pose(g, cur, max_attempts=20, enable_graph_attempt=2)
ok = r is not None and bool(getattr(r, "success",
                                    torch.tensor([False])).flatten()[0])
print(f"[smoke] C plan_pose: {'✅' if ok else '❌'} ({time.time() - t0:.1f}s)")
assert ok, "空世界 5cm 平移都解不了 —— 查配置"
p_ = r.js_solution.position
while p_.dim() > 2:
    p_ = p_.squeeze(0)
# worker._take 同款: js_solution 的关节序是全 cspace, 按名取回规划器 14 关节列
nm = list(getattr(r.js_solution, "joint_names", None) or pj)
idx = [nm.index(n) for n in pj]
p_ = p_[:, idx]
end = JointState.from_position(p_[-1:].contiguous().to(q0.device),
                               joint_names=pj)
ke = P.compute_kinematics(end)
pe = ke.tool_poses.get_link_pose("right_hand_C_MC", make_contiguous=True)
err = np.linalg.norm(pe.position.view(-1)[:3].cpu().numpy() - tgt)
print(f"[smoke] C 终点误差 {err * 1000:.1f}mm | 轨迹 {len(p_)} 行")
assert err < 0.01, "终点误差超 1cm"
print("[smoke] ★ cuRobo vega_1p_sharpa 全通 —— worker 侧就绪")
