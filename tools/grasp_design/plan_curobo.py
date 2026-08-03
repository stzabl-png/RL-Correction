"""用 cuRobo v2 规划整条 pick-transport-place 手臂轨迹.

坐标: 设计构型在 **env 桌面局部系** (桌面 z=0.85, 机器人底座在 x=-0.5).
      cuRobo 在 **机器人底座系**, 所以 p_curobo = p_env + [0.5, 0, 0].
R_ee 与 right_hand_C_MC 已验证**完全重合**(单位变换), 腕位姿可直接当工具目标.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
SP = "/home/lyh/Project/RL_Correction/tools/grasp_design"

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene, Cuboid
from curobo.types import GoalToolPose, JointState, Pose
from curobo.config_io import load_yaml, join_path
from curobo.content import get_robot_configs_path

BASE_OFF = np.array([0.5, 0.0, 0.0])          # env -> curobo base
TZ = 0.85
# 搬运剖面: 抓起后**只**上升 LIFT, 水平 MOVE, 再原样下降 LIFT 把物体放回桌面高度后松开.
# 下降量必须 == 上升量, 否则松手时物体悬在半空自由落体 (旧版 LIFT=20/DROP=10 就是这样).
# MOVE 为什么是 14 而不是重建数据里的 20: 保持腕姿态做纯平移时, 腕滚转 R_arm_j7 在
# 约 15cm 处走完行程 —— 实测 14cm 时 IK 残差 1.8mm 干净可达, 16cm 起 6.1mm, 20cm 26.6mm.
LIFT, MOVE = 0.10, 0.14
APPROACH_UP = 0.12

GRIP = os.environ.get("GRIP_NPZ", "grip_cfg_v6.npz")
g = np.load(f"{SP}/{GRIP}", allow_pickle=True)
print(f"构型文件: {GRIP}")
T_grip = g["wrist_T"]
q_open, q_grip = g["q_open"], g["q_grip"]
JN = [str(s) for s in g["joint_names"]]
OBJ = g["obj"]
wq = g["wrist_quat"]                          # wxyz, 全程保持不变 (纯平移搬运)

# 搬运方向: 重建里物体 [0.338,-0.074,-0.447] -> [0.457,-0.189,-0.357]
d = np.array([0.457 - 0.338, -0.189 - (-0.074)])
DIR = d / np.linalg.norm(d)
print(f"搬运水平方向 (env xy) = {np.round(DIR,4).tolist()}  (重建位移方位角 "
      f"{np.degrees(np.arctan2(DIR[1],DIR[0])):.1f}°)")

W = T_grip[:3, 3]
MV = np.append(DIR * MOVE, 0.0)
WP = {
    "P1_接近(物体正上方)": W + [0, 0, APPROACH_UP],
    "P2_抓取位(五指张开就位)": W,
    "P3_提起10cm": W + [0, 0, LIFT],
    "P4_水平20cm": W + [0, 0, LIFT] + MV,
    "P5_下降10cm(松开)": W + MV,
    "P6_抬手撤离": W + [0, 0, LIFT] + MV,
}
print("\n路点 (env 桌面局部系, 手基座/R_ee 位置):")
for k, v in WP.items():
    print(f"  {k:<26}{np.round(v,4).tolist()}   离桌 {(v[2]-TZ)*100:6.2f}cm")
obj_end = OBJ[:2] + DIR * MOVE
drop_h = (WP["P5_下降10cm(松开)"][2] - W[2]) * 100
print(f"\n松开时物体中心 xy = {np.round(obj_end,4).tolist()}, "
      f"离桌 {drop_h + (OBJ[2]-TZ)*100:.1f}cm (= 抓取时的离桌高度, 自由落体 {drop_h:.1f}cm)")

# ---------- 机器人配置: 锁躯干到训练 env 站姿, 锁手指到张开构型 ----------
cfg_d = load_yaml(join_path(get_robot_configs_path(), "magicsim_vega1p_sharpa.yml"))
lk = cfg_d["robot_cfg"]["kinematics"].setdefault("lock_joints", {})
lk.update({"torso_j1": np.deg2rad(45.0), "torso_j2": np.deg2rad(90.0),
           "torso_j3": 0.0, "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0})
for n, v in zip(JN, q_open):                  # 张开构型 = 更宽的包络, 保守
    lk[n] = float(v)
print(f"\n锁定 {len(lk)} 个关节 (躯干 45/90/0°, 头 0, 右手=张开构型, 左手=0)")

scene = Scene(cuboid=[
    Cuboid(name="table", dims=[1.2, 1.2, 0.04],
           pose=[*(np.array([0.0, 0.0, TZ - 0.02]) + BASE_OFF), 1, 0, 0, 0]),
])
cfg = MotionPlannerCfg.create(robot=cfg_d, scene_model=scene, num_ik_seeds=64,
                              num_trajopt_seeds=8, position_tolerance=0.005,
                              orientation_tolerance=0.05,
                              anchor_ik_seeds_to_current_state=True)
planner = MotionPlanner(cfg)
planner.warmup(enable_graph=True, num_warmup_iterations=3)
print(f"\n规划关节: {planner.joint_names}")
print(f"工具系: {planner.tool_frames}")

dev = "cuda"
tf = list(planner.tool_frames)
# ⚠ cuRobo 的 default_joint_state 与 env 的默认站姿**左右是反的**:
#   cuRobo R_arm_j1=+45/j4=0 (平举, 肘伸直);  env R_arm_j1=-45/j4=-90 (肘弯).
#   直接拿 cuRobo 默认当起点, 回放里右臂就会平举着出场, 和左臂不对称.
from rl_rebuild.correction.kinematics import DEFAULT_ARM_DEG  # noqa: E402
q0 = torch.zeros(1, len(planner.joint_names), device=dev)
for _i, _n in enumerate(planner.joint_names):
    q0[0, _i] = np.deg2rad(DEFAULT_ARM_DEG.get(_n, 0.0))
js0 = JointState.from_position(q0, joint_names=planner.joint_names)
_q0d = np.degrees(q0[0].detach().cpu().numpy())
print(f"起始站姿(env 默认, deg) = "
      f"{ {n: round(float(_q0d[i]), 1) for i, n in enumerate(planner.joint_names)} }")


def tool_pose_at(js, frame):
    p = planner.compute_kinematics(js).tool_poses.get_link_pose(frame, make_contiguous=True)
    return (p.position.view(-1, 3)[0].detach().cpu().numpy(),
            p.quaternion.view(-1, 4)[0].detach().cpu().numpy())


R_now, _ = tool_pose_at(js0, "R_ee")
print(f"右手 R_ee 起始 (base系) {np.round(R_now,4).tolist()} "
      f"-> env系 {np.round(R_now - BASE_OFF,4).tolist()}")


def goal(p_env, js_cur):
    """R_ee 追目标; 其余工具系(L_ee 等)填**当前 FK**, 免得成为 NaN 目标.
    (vega 双臂同时给硬目标会让 MG 失败, 见 repro_vega_fail.py)"""
    pd = {}
    for f in tf:
        if f == "R_ee":
            pd[f] = Pose(
                position=torch.tensor([list(np.asarray(p_env) + BASE_OFF)],
                                      device=dev, dtype=torch.float32),
                quaternion=torch.tensor([list(wq)], device=dev, dtype=torch.float32))
        else:
            p, q = tool_pose_at(js_cur, f)
            pd[f] = Pose(position=torch.tensor([list(p)], device=dev, dtype=torch.float32),
                         quaternion=torch.tensor([list(q)], device=dev, dtype=torch.float32))
    return GoalToolPose.from_poses(pd, ordered_tool_frames=tf, num_goalset=1)


dt = planner.trajopt_solver.config.interpolation_dt

# 抓取位的关节解 = 构型求解的设计变量本身 (FK 精确吻合), 用于受控下降段的插值终点.
ARM_Q = g["arm_q"]
ARM_JN = [str(s) for s in g["arm_joints"]]
RIDX = [planner.joint_names.index(n) for n in ARM_JN]
print(f"抓取位右臂关节(deg) = {np.round(np.degrees(ARM_Q),2).tolist()}")

# 只有 P1 (从站姿伸到物体上方) 需要 cuRobo 的避障; 之后全是**贴着桌面的直线搬运**:
#   - 抓取高度附近手指离桌仅 ~1cm, 落在 cuRobo 碰撞球+激活距离内, MG 必判撞桌
#     (cuRobo 自己的 grasp planning 也是"最后一段关掉手指碰撞"来处理这一点);
#   - cuRobo 每段都是静止到静止, 段边界速度归零, 多段拼起来看着就是"一顿一顿".
# 所以提起/平移/下降/撤离四段一律走**自研 IK 沿直线逐点求解**, 姿态全程锁 wq (纯平移搬运).
PLAN = [("P1_接近(物体正上方)", WP["P1_接近(物体正上方)"], "curobo"),
        ("P2_下降到抓取位", None, "interp_down"),
        ("P3_提起10cm", WP["P3_提起10cm"], "line"),
        ("P4_水平20cm", WP["P4_水平20cm"], "line"),
        ("P5_下降10cm(松开)", WP["P5_下降10cm(松开)"], "line"),
        ("P6_抬手撤离", WP["P6_抬手撤离"], "line"),
        ("P7_回初始位姿", None, "interp_home")]

from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
_ikA = ArmIK("right", base_pos=(-0.5, 0.0, 0.0))
_R_grip = quat_to_R(np.asarray(wq, dtype=np.float64))

segs, js, ok_all = [], js0, True
for name, p, mode in PLAN:
    if mode == "interp_down":
        q_a = js.position[0].clone()
        q_b = q_a.clone()
        q_b[RIDX] = torch.tensor(ARM_Q, device=dev, dtype=q_b.dtype)
        n_i = 40
        w = torch.linspace(0, 1, n_i, device=dev, dtype=q_a.dtype)
        w = (w * w * (3 - 2 * w)).unsqueeze(1)
        qtraj = (1 - w) * q_a.unsqueeze(0) + w * q_b.unsqueeze(0)
        segs.append((name, qtraj.detach().cpu().numpy()))
        js = JointState.from_position(qtraj[-1:].clone(), joint_names=planner.joint_names)
        got = tool_pose_at(js, "R_ee")[0] - BASE_OFF
        print(f"  ✅ {name:<26} {n_i:>4} 点 / {n_i*dt:5.2f}s   [关节插值] 末端误差 "
              f"{np.linalg.norm(got - W)*1000:5.1f} mm")
        continue
    if mode == "line":
        # 从当前 R_ee 位置沿**直线**走到 p, 逐点 IK (上一点热启动), 单一 smoothstep
        # 速度剖面 —— 一段到底, 中途不停, 姿态锁 _R_grip.
        base = js.position[0].clone()
        q_cur = base[RIDX].detach().cpu().numpy().astype(float)
        p_from = tool_pose_at(js, "R_ee")[0] - BASE_OFF
        p_to = np.asarray(p, dtype=float)
        n_i = max(int(round(np.linalg.norm(p_to - p_from) / 0.0025)), 20)
        sflat = np.linspace(0, 1, n_i)
        sm = sflat * sflat * (3 - 2 * sflat)
        qs, devs = [], []
        for u_ in sm:
            r = _ikA.solve(p_from + (p_to - p_from) * float(u_), _R_grip,
                           q0=q_cur, pos_tol=0.002, rot_tol=np.deg2rad(5))
            q_cur = r["q"]
            qs.append(q_cur.copy())
            devs.append(r["pos_err"])
        qtraj = base.unsqueeze(0).repeat(n_i, 1)
        qtraj[:, RIDX] = torch.tensor(np.array(qs), device=dev, dtype=qtraj.dtype)
        segs.append((name, qtraj.detach().cpu().numpy()))
        js = JointState.from_position(qtraj[-1:].clone(), joint_names=planner.joint_names)
        got = tool_pose_at(js, "R_ee")[0] - BASE_OFF
        print(f"  ✅ {name:<26} {n_i:>4} 点 / {n_i*dt:5.2f}s   [直线IK] "
              f"末端误差 {np.linalg.norm(got - p_to)*1000:5.1f} mm  "
              f"沿途IK残差 最大 {max(devs)*1000:.2f} mm")
        if max(devs) > 0.005:
            print(f"     ⚠ 直线中途 IK 残差 >5mm, 这段不是干净的直线")
        continue
    if mode == "interp_home":
        # 松手撤离后回到 env reset 站姿 (= 轨迹首帧 q0), 全关节 smoothstep 插值.
        # 撤离段已经把手抬离物体 LIFT, 这里是自由空间; 是否穿桌由 export_traj.py
        # 的全链 FK 核验兜底 (它报"手上最低 link 离桌"的全程最小值).
        q_a = js.position[0].clone()
        q_b = q0[0].clone()
        n_i = 60
        w = torch.linspace(0, 1, n_i, device=dev, dtype=q_a.dtype)
        w = (w * w * (3 - 2 * w)).unsqueeze(1)
        qtraj = (1 - w) * q_a.unsqueeze(0) + w * q_b.unsqueeze(0)
        segs.append((name, qtraj.detach().cpu().numpy()))
        js = JointState.from_position(qtraj[-1:].clone(), joint_names=planner.joint_names)
        back = float(torch.abs(js.position[0] - q0[0]).max())
        print(f"  ✅ {name:<26} {n_i:>4} 点 / {n_i*dt:5.2f}s   [关节插值回站姿] "
              f"末态与初始站姿最大关节差 {np.degrees(back):.3f}°")
        continue
    res = planner.plan_pose(goal(p, js), js)
    if res is None or not bool(res.success.any()):
        print(f"  ❌ {name}: 规划失败")
        ok_all = False
        break
    traj = res.get_interpolated_plan()
    tn = list(getattr(traj, "joint_names", None) or planner.joint_names)
    Qfull = traj.position.reshape(-1, len(tn))
    qtraj = Qfull[:, [tn.index(nm) for nm in planner.joint_names]].contiguous()
    n = qtraj.shape[0]
    segs.append((name, qtraj.detach().cpu().numpy()))
    js = JointState.from_position(qtraj[-1:].clone(), joint_names=planner.joint_names)
    got = tool_pose_at(js, "R_ee")[0] - BASE_OFF
    print(f"  ✅ {name:<26} {n:>4} 点 / {n*dt:5.2f}s   末端误差 "
          f"{np.linalg.norm(got - p)*1000:5.1f} mm")

if ok_all:
    tot = sum(s[1].shape[0] for s in segs)
    print(f"\n全段成功: {len(segs)} 段, 共 {tot} 点, {tot*dt:.2f}s (dt={dt*1000:.1f}ms)")
    np.savez(f"{SP}/arm_traj.npz", dt=dt, joint_names=np.array(planner.joint_names),
             q_home=q0[0].detach().cpu().numpy(),
             obj_start=OBJ, obj_end=np.append(obj_end, OBJ[2]), lift=LIFT, move=MOVE,
             **{f"seg{i}_{n.split('_')[0]}": q for i, (n, q) in enumerate(segs)})
    print(f"-> {SP}/arm_traj.npz")
