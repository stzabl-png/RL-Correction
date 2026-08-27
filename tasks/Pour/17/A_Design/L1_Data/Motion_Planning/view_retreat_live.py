"""Retreat 现场版 (2026-08-27 用户裁定): **照抄静止 GraspPose 脚本**的场景与摆位
(用户 GUI 验证过无碰撞), 在它实现出来的状态上原地加 Retreat:

    静止合拢定格 -> 按 Enter -> 臂沿各自接近方向 cuRobo 退 5cm,
    手指在前 45 帧内绷直(每指根部关节钉住不动) -> 归位定格等下一次 Enter

退避靶 = **实测**腕位 + 5cm×(PreGrasp-Grasp 单位方向); cuRobo 起点 = 实测臂位形。
不经过任何离线 npz —— 状态与静止脚本逐字节同源。
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--prior_a", default="tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")
p.add_argument("--yaw_a", type=float, default=19.5)
p.add_argument("--prior_b", default="tasks/pregrasp/priors/Pour17_cup_thumbfix.npz")
p.add_argument("--yaw_b", type=float, default=90.0)
p.add_argument("--retreat_cm", type=float, default=5.0)
p.add_argument("--extra_cm", type=float, default=10.0,
               help="安全腿(径向)额外退距cm; 0=跳过(合并为单次后撤)")
p.add_argument("--fin_frames", type=int, default=45)
p.add_argument("--act_dist", type=float, default=0.015)
p.add_argument("--fps", type=float, default=30.0)
p.add_argument("--order", choices=["blend", "seq"], default="blend",
               help="blend=绷直与后退并行; seq=先绷直(臂定格)再后退")
p.add_argument("--retreat_dir", choices=["approach", "palm2wrist", "mix"], default="approach",
               help="approach=沿PreGrasp-Grasp接近轴反向; palm2wrist=沿掌心->手腕"
                    "的轴向抽手(实测四指PP质心->臂末端ee)")
p.add_argument("--save_npz", default="tasks/Pour/17/A_Design/L1_Data/Motion_Planning/"
                                     "Retreat_pour17.npz",
               help="规划成功后把动作行存到这里 (与 view_motion 同格式); 空串=不存")
p.add_argument("--selftest", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("view_retreat_live")
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
import select  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import (  # noqa: E402
    GENERIC_JOINT_ORDER,
)
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

BAR = "=" * 70


def qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


# ============================ 以下与 view_grasp_pose.py 逐段同源 ============================
cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.prior_a, args.yaw_a, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()

# ---- Initial Pose 基准 (对称待命站姿): 臂关节 / 腕位姿 / 手指张开, 全部实测 ----
_bn0 = list(E.hand.body_names)
_jn0 = list(E.hand.joint_names)
_W0 = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
_dq0 = E.hand.data.default_joint_pos[0].cpu().numpy().astype(np.float64)
stance_arm = {s: np.array([_dq0[_jn0.index(f"{P}_arm_j{i}")] for i in range(1, 8)])
              for s, P in (("right", "R"), ("left", "L"))}
stance_wrist = {}
for s in ("right", "left"):
    _bi = _bn0.index(f"{s}_hand_C_MC")
    stance_wrist[s] = (
        E.hand.data.body_pos_w[0, _bi].cpu().numpy().astype(np.float64) - _W0,
        E.hand.data.body_quat_w[0, _bi].cpu().numpy().astype(np.float64))
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER as _GJ0
fin_stance = {s: np.array([_dq0[_jn0.index(n.replace("right_", f"{s}_"))]
                           for n in _GJ0]) for s in ("right", "left")}

A_name = cfg.hand_side
B_name = "left" if A_name == "right" else "right"
A_gp = E._grasp_pos_w.cpu().numpy().astype(np.float64)
A_gq = E._grasp_quat_w.cpu().numpy().astype(np.float64)
A_obj = E.obj_init_pos.cpu().numpy().astype(np.float64)
aux_off = getattr(E, "aux_rel_offset_np", None)
assert aux_off is not None, "没有第二个物体"
B_obj = A_obj + np.asarray(aux_off, np.float64)
zb = np.load(args.prior_b)
_a = np.radians(args.yaw_b)
oq_b = qmul(np.array([np.cos(_a/2), 0, 0, np.sin(_a/2)]),
            np.asarray(zb["canon_rot"], np.float64))
B_gp = quat_to_R(oq_b) @ np.asarray(zb["grasp"][:3], np.float64) + B_obj
B_gq = qmul(oq_b, np.asarray(zb["grasp"][3:7], np.float64))

A_jids = list(E.arm_jids)
A_hjids = list(E.hand_jids)
E._resolve_joint_ids(force_side=B_name)
B_jids = list(E.arm_jids)
E._resolve_joint_ids(force_side=A_name)

print(f"\n{BAR}\n静止 GraspPose 同源摆位 | {args.clip}\n{BAR}")
sol = {}
for nm, side, gp, gq in ((f"右手->瓶", A_name, A_gp, A_gq),
                         (f"左手->杯", B_name, B_gp, B_gq)):
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=E._anchor_T)
    r = ik.solve(gp, quat_to_R(gq), iters=300)
    sol[side] = r["q"]
    print(f"  {nm}: IK {'✅' if r['ok'] else '❌'} 位置误差 {r['pos_err']*100:.2f}cm")

jn = list(E.hand.joint_names)
za = np.load(args.prior_a)
q = E.hand.data.default_joint_pos.clone()
q[:, A_jids] = torch.tensor(sol[A_name], dtype=torch.float32, device=E.device)
q[:, B_jids] = torch.tensor(sol[B_name], dtype=torch.float32, device=E.device)
fin_grasp = {}
for z_, side_ in ((za, A_name), (zb, B_name)):
    vals = np.asarray(z_["grasp"], np.float64)[7:29]
    fin_grasp[side_] = vals
    for n_, v_ in zip(GENERIC_JOINT_ORDER, vals):
        q[:, jn.index(n_.replace("right_", f"{side_}_"))] = float(v_)
E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
E.hand.set_joint_position_target(q)
E.hand.write_data_to_sim()
# ============================ 同源摆位到此为止 ============================

# ---- 退避方向 (PreGrasp0 - Grasp, 用与摆位同一条链) + 实测起点 ----
_bn = list(E.hand.body_names)
for _ in range(5):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
real_w = {}
for side in (A_name, B_name):
    real_w[side] = (E.hand.data.body_pos_w[0, _bn.index(f"{side}_hand_C_MC")]
                    .cpu().numpy().astype(np.float64) - W)
u_dir, tgt = {}, {}
for side, z_, yaw_, obj_p, gq_ in ((A_name, za, args.yaw_a, A_obj, A_gq),
                                   (B_name, zb, args.yaw_b, B_obj, B_gq)):
    _aa = np.radians(yaw_)
    oq_ = qmul(np.array([np.cos(_aa/2), 0, 0, np.sin(_aa/2)]),
               np.asarray(z_["canon_rot"], np.float64))
    pre0 = quat_to_R(oq_) @ np.asarray(z_["pregrasp"], np.float64)[0][:3] + obj_p
    g_ = quat_to_R(oq_) @ np.asarray(z_["grasp"], np.float64)[:3] + obj_p
    u = pre0 - g_
    u = u / max(np.linalg.norm(u), 1e-9)
    u_dir[side] = u

if args.retreat_dir in ("palm2wrist", "mix"):
    # 掌心(四指PP质心) -> 手腕(臂末端 ee) 的轴向 = 抽手方向 (在抓握位形下实测)
    for side_ in (A_name, B_name):
        P_ = "R" if side_ == "right" else "L"
        palm_c = np.mean([E.hand.data.body_pos_w[0, _bn.index(f"{side_}_{f_}_PP")]
                          .cpu().numpy().astype(np.float64) - W
                          for f_ in ("index", "middle", "ring", "pinky")], axis=0)
        wrist = (E.hand.data.body_pos_w[0, _bn.index(f"{P_}_ee")]
                 .cpu().numpy().astype(np.float64) - W)
        v_ = wrist - palm_c
        u_dir[side_] = v_ / max(np.linalg.norm(v_), 1e-9)
        if args.retreat_dir == "mix":
            # 掌轴+径向合成: 左手掌轴与杯面相切(2026-08-27 分诊实锤,
            # 纯掌轴退10cm仍被杯判碰), 叠加径向分量保证每cm都真离物
            _objc0 = A_obj if side_ == A_name else B_obj
            _vr = wrist - _objc0
            _vr = _vr / max(np.linalg.norm(_vr), 1e-9)
            _vm = u_dir[side_] + _vr
            u_dir[side_] = _vm / max(np.linalg.norm(_vm), 1e-9)
        print(f"[live] {side_}: 退避方向({args.retreat_dir}) {np.round(u_dir[side_],3)}")

for side in (A_name, B_name):
    gq_ = A_gq if side == A_name else B_gq
    tgt[side] = (real_w[side] + (args.retreat_cm / 100.0) * u_dir[side], gq_)
    print(f"[live] {side}: 实测腕 {np.round(real_w[side]*100,1)}cm -> "
          f"退避靶 {np.round(tgt[side][0]*100,1)}cm ({args.retreat_dir})")

# ---- 手指绷直位形 (根部钉抓握值) ----
_ROOT = {"thumb_CMC_FE", "thumb_CMC_AA", "index_MCP_FE", "index_MCP_AA",
         "middle_MCP_FE", "middle_MCP_AA", "ring_MCP_FE", "ring_MCP_AA",
         "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA"}
_lim = E.hand.root_physx_view.get_dof_limits()[0].cpu().numpy().astype(np.float64)
fin_straight = {}
for side in (A_name, B_name):
    row = []
    for gi, n_ in enumerate(GENERIC_JOINT_ORDER):
        nm = n_.replace("right_", f"{side}_")
        if n_.replace("right_", "") in _ROOT:
            row.append(float(fin_grasp[side][gi]))
        else:
            row.append(float(np.clip(0.0, _lim[jn.index(nm), 0], _lim[jn.index(nm), 1])))
    fin_straight[side] = np.array(row)

# ---- cuRobo: 实测臂位形 -> 退避靶 (物体排除/桌保留) ----
_root_p = (E.hand.data.root_pos_w[0].cpu().numpy().astype(np.float64) - W)
_sx, _sy, _sz = cfg.table_size
_start = {n: float(v) for n, v in zip(
    jn, E.hand.data.default_joint_pos[0].cpu().numpy().astype(float))}
for side in (A_name, B_name):
    P = "R" if side == "right" else "L"
    for i in range(7):
        _start[f"{P}_arm_j{i+1}"] = float(sol[side][i])
    for n_, v_ in zip(GENERIC_JOINT_ORDER, fin_straight[side]):
        _start[n_.replace("right_", f"{side}_")] = float(v_)
_start.update({"torso_j1": 0.7072, "torso_j2": 1.2856, "torso_j3": 0.0068,
               "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0})
_goals = {f"{s}_hand_C_MC": {"pos": (tgt[s][0] + W - _root_p).tolist(),
                             "quat": tgt[s][1].tolist()} for s in (A_name, B_name)}
_tmp = tempfile.mkdtemp(prefix="curobo_live_")
_tj, _out = os.path.join(_tmp, "targets.json"), os.path.join(_tmp, "plan.npz")
with open(_tj, "w") as f:
    json.dump({
        "table_pose": [float(W[0] - _root_p[0]), float(W[1] - _root_p[1]),
                       float(W[2] + cfg.table_top_z - _sz / 2 - _root_p[2])],
        "table_dims": [float(_sx), float(_sy), float(_sz)],
        "objects": [],
        "start_joints": _start,
        "lock_joints": ["torso_j1", "torso_j2", "torso_j3",
                        "head_j1", "head_j2", "head_j3"]
                       + [n for n in jn if n.startswith(("right_", "left_"))],
        "tool_frames": [f"{A_name}_hand_C_MC", f"{B_name}_hand_C_MC"],
        "hand_targets": {},
        "goals": _goals,
        "pregrasp_goals": _goals,
    }, f)
print(f"[live] cuRobo 规划中 (实测起点 -> 各退 {args.retreat_cm}cm) ...")
subprocess.run([sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
                "--targets", _tj, "--out", _out, "--act_dist", str(args.act_dist),
                "--joint", "1", "--left_short_cm", "0.0", "--table_pad", "0.0",
                "--exclude_objects", "1"],
               cwd=os.getcwd(), env=dict(os.environ, PYTHONPATH=os.getcwd()),
               timeout=2400)
assert os.path.exists(_out), "cuRobo 子进程没有产出"
_z = np.load(_out, allow_pickle=True)
assert bool(_z["ok"]), f"规划失败 (卡在 {_z.get('failed_frame')})"
traj = _z["traj"]
_pjn = [str(x) for x in _z["joint_names"]]
arm_cols = {s: [_pjn.index(f"{'R' if s == 'right' else 'L'}_arm_j{i}")
                for i in range(1, 8)] for s in ("right", "left")}
T2 = traj.shape[0]
K = min(max(2, int(args.fin_frames)), T2)
rows = {}
if args.order == "seq":
    # 先绷直(臂定格 K 帧) 再后退(T2 帧, 指保持绷直)
    for s in ("right", "left"):
        tt = np.linspace(0.0, 1.0, K)
        ss = tt * tt * (3 - 2 * tt)
        f1 = (1 - ss)[:, None] * fin_grasp[s] + ss[:, None] * fin_straight[s]
        a1 = np.tile(sol[s], (K, 1))
        rows[f"{s}_q"] = np.concatenate([a1, traj[:, arm_cols[s]]]).astype(np.float32)
        rows[f"{s}_f"] = np.concatenate(
            [f1, np.tile(fin_straight[s], (T2, 1))]).astype(np.float32)
    T2 = K + T2
    print(f"[live] ✅ seq 版: 绷直{K} + 后退{traj.shape[0]} = {T2} 帧")
else:
    for s in ("right", "left"):
        tt = np.clip(np.arange(T2) / max(K - 1, 1), 0.0, 1.0)
        ss = tt * tt * (3 - 2 * tt)
        rows[f"{s}_q"] = traj[:, arm_cols[s]].astype(np.float32)
        rows[f"{s}_f"] = ((1 - ss)[:, None] * fin_grasp[s]
                          + ss[:, None] * fin_straight[s]).astype(np.float32)
    print(f"[live] ✅ blend 版: 规划 {T2} 帧, 指绷直窗前 {K} 帧")
# ---- 终章: cuRobo 从退避终点规划回 Initial Pose, 指渐变到对称张开 ----
_start2 = dict(_start)
for side in (A_name, B_name):
    P = "R" if side == "right" else "L"
    for i in range(7):
        _start2[f"{P}_arm_j{i+1}"] = float(traj[-1, arm_cols[side][i]])
# cspace 目标 = dexmate_joints 定义的对称待命站姿 (实测 default_joint_pos, 关节空间直达)
_cs = {}
for s in ("right", "left"):
    P = "R" if s == "right" else "L"
    for i in range(7):
        _cs[f"{P}_arm_j{i+1}"] = float(stance_arm[s][i])
# ★终章必须带物体障碍 (2026-08-27 用户抓包: objects=[] 让回程穿瓶)。
#   退避段起点贴物必须排除物体; 终章起点已退开 5cm, 全障碍无起点冲突
_e2 = clips.clip_entry(args.clip)
_sec2 = _e2.get("secondary") or {}
_objs2 = []
for _nm2, _mesh2, _ob2 in (("obj_primary", _e2["mesh"], E.object),
                           ("obj_secondary", _sec2.get("mesh"), getattr(E, "aux", None))):
    if _mesh2 and _ob2 is not None:
        _st2 = _ob2.data.root_state_w[0].cpu().numpy().astype(np.float64)
        _objs2.append({"name": _nm2, "mesh": _mesh2,
                       "pos": ((_st2[:3] - W) + W - _root_p).tolist(),
                       "quat": _st2[3:7].tolist()})
print(f"[live] 终章障碍: 桌 + 物体×{len(_objs2)} (实测位姿)")
_tj2, _out2 = os.path.join(_tmp, "targets2.json"), os.path.join(_tmp, "plan2.npz")
with open(_tj2, "w") as f:
    json.dump({
        "table_pose": json.load(open(_tj))["table_pose"],
        "table_dims": [float(_sx), float(_sy), float(_sz)],
        "objects": _objs2,
        "start_joints": _start2,
        "lock_joints": ["torso_j1", "torso_j2", "torso_j3",
                        "head_j1", "head_j2", "head_j3"]
                       + [n for n in jn if n.startswith(("right_", "left_"))],
        "tool_frames": [f"{A_name}_hand_C_MC", f"{B_name}_hand_C_MC"],
        "hand_targets": {},
        "cspace_goal": _cs,
        "goals": {},
        "pregrasp_goals": {},
    }, f)
# ---- 终章B1: 沿掌轴再退 5cm 拉开安全距 (空物体障碍, 与退避段同性质) ----
if args.extra_cm > 0:
    # ★B1 方向定案(2026-08-27 分诊全链): 左手掌轴与杯面近似相切, 沿掌轴退 15cm
    #   指尖仍贴面滑行不脱离激活区(逐物体探针: 杯有责/瓶清白)。安全腿改**径向**
    #   (物心→退避终点腕), 保证每 cm 都是真离开
    _objc = {A_name: A_obj, B_name: B_obj}
    _usafe = {}
    for s in (A_name, B_name):
        _v = tgt[s][0] - _objc[s]
        _usafe[s] = _v / max(np.linalg.norm(_v), 1e-9)
        print(f"[live] {s}: B1 径向撤离方向 {np.round(_usafe[s],3)}")
    _goalsB1 = {f"{s}_hand_C_MC": {"pos": (tgt[s][0] + args.extra_cm / 100.0 * _usafe[s] + W - _root_p).tolist(),
                                   "quat": tgt[s][1].tolist()} for s in (A_name, B_name)}
    _tjB1, _outB1 = os.path.join(_tmp, "targetsB1.json"), os.path.join(_tmp, "planB1.npz")
    with open(_tjB1, "w") as f:
        json.dump({
            "table_pose": json.load(open(_tj))["table_pose"],
            "table_dims": [float(_sx), float(_sy), float(_sz)],
            "objects": [],
            "start_joints": _start2,
            "lock_joints": ["torso_j1", "torso_j2", "torso_j3",
                            "head_j1", "head_j2", "head_j3"]
                           + [n for n in jn if n.startswith(("right_", "left_"))],
            "tool_frames": [f"{A_name}_hand_C_MC", f"{B_name}_hand_C_MC"],
            "hand_targets": {},
            "goals": _goalsB1,
            "pregrasp_goals": _goalsB1,
        }, f)
    print("[live] cuRobo 终章B1 (再退5cm拉开安全距) ...")
    subprocess.run([sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
                    "--targets", _tjB1, "--out", _outB1, "--act_dist", str(args.act_dist),
                    "--joint", "1", "--left_short_cm", "0.0", "--table_pad", "0.0",
                    "--exclude_objects", "1"],
                   cwd=os.getcwd(), env=dict(os.environ, PYTHONPATH=os.getcwd()),
                   timeout=2400)
    assert os.path.exists(_outB1), "终章B1 没有产出"
    _zB1 = np.load(_outB1, allow_pickle=True)
    assert bool(_zB1["ok"]), "终章B1 规划失败"
    trajB1 = _zB1["traj"]
    _pjnB1 = [str(x) for x in _zB1["joint_names"]]
    colsB1 = {s: [_pjnB1.index(f"{'R' if s == 'right' else 'L'}_arm_j{i}")
                  for i in range(1, 8)] for s in ("right", "left")}
    TB1 = trajB1.shape[0]
    for s in ("right", "left"):
        rows[f"{s}_q"] = np.concatenate(
            [rows[f"{s}_q"], trajB1[:, colsB1[s]]]).astype(np.float32)
        rows[f"{s}_f"] = np.concatenate(
            [rows[f"{s}_f"], np.tile(fin_straight[s], (TB1, 1))]).astype(np.float32)
    T2 = T2 + TB1
    print(f"[live] ✅ 终章B1 {TB1} 帧 (总退距 {args.retreat_cm + args.extra_cm:.0f}cm)")
    for side in (A_name, B_name):
        P = "R" if side == "right" else "L"
        for i in range(7):
            _start2[f"{P}_arm_j{i+1}"] = float(trajB1[-1, colsB1[side][i]])
    # ★2026-08-27 验尸: 终章 JSON 在 B1 之前就已写盘, _start2 更新曾是死代码
    #   (分诊指纹: 三个不同退距的腕位逐位相同) —— B1 后必须重写 start_joints
    with open(_tj2) as _fj2:
        _T2j = json.load(_fj2)
    _T2j["start_joints"] = _start2
    with open(_tj2, "w") as _fj2:
        json.dump(_T2j, _fj2)

print("[live] cuRobo 规划终章 (安全距起点 -> Initial Pose) ...")
subprocess.run([sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
                # 2026-08-27 用户定案: 激活距=官方默认 0.010, 不充气,
                # 余量靠 15cm 退距的真实几何
                "--targets", _tj2, "--out", _out2, "--act_dist", "0.010",
                "--attempts", "40",
                "--joint", "1", "--left_short_cm", "0.0", "--table_pad", "0.0",
                "--exclude_objects", "0"],
               cwd=os.getcwd(), env=dict(os.environ, PYTHONPATH=os.getcwd()),
               timeout=2400)
assert os.path.exists(_out2), "终章规划没有产出"
_z2 = np.load(_out2, allow_pickle=True)
assert bool(_z2["ok"]), f"终章规划失败 (卡在 {_z2.get('failed_frame')})"
traj2 = _z2["traj"]
_pjn2 = [str(x) for x in _z2["joint_names"]]
cols2 = {s: [_pjn2.index(f"{'R' if s == 'right' else 'L'}_arm_j{i}")
             for i in range(1, 8)] for s in ("right", "left")}
T3 = traj2.shape[0]
for s in ("right", "left"):
    a3 = traj2[:, cols2[s]]
    # 指: 钉根绷直 -> 对称张开 (随终章渐变); cspace 终帧即精确站姿, 无需贴合段
    tt3 = np.linspace(0.0, 1.0, T3)
    ss3 = (tt3 * tt3 * (3 - 2 * tt3))[:, None]
    f3 = (1 - ss3) * fin_straight[s] + ss3 * fin_stance[s]
    rows[f"{s}_q"] = np.concatenate([rows[f"{s}_q"], a3]).astype(np.float32)
    rows[f"{s}_f"] = np.concatenate([rows[f"{s}_f"], f3]).astype(np.float32)
    _jerr = np.degrees(np.abs(a3[-1] - stance_arm[s]).max())
    print(f"[live] {s} 终帧 vs 站姿 最大关节差 {_jerr:.2f}°")
T2 = T2 + T3
print(f"[live] ✅ cspace 终章 {T3} 帧, 全程共 {T2} 帧")

if args.save_npz and (args.order == "seq" or args.retreat_dir != "approach"):
    _sfx = ("_seq" if args.order == "seq" else "") +         ({"palm2wrist": "_p2w", "mix": "_mix"}.get(args.retreat_dir, ""))
    args.save_npz = args.save_npz.replace(".npz", f"{_sfx}.npz")
if args.save_npz:
    sv = dict(rows)
    sv["fin_names"] = np.array(GENERIC_JOINT_ORDER, dtype=object)
    sv["seg_names"] = np.array(["blend_straighten_retreat"], dtype=object)
    sv["seg_lens"] = np.array([T2], np.int64)
    sv["meta"] = ("Retreat_pour17 现场版: 静止GraspPose同源摆位实测起点 + "
                  f"cuRobo退{args.retreat_cm}cm({T2}帧) 指前{K}帧绷直根部钉住")
    np.savez(args.save_npz, **sv)
    print(f"[live] 已存 {args.save_npz}")

# ---- 回合制播放: 定格 -> Enter -> 播 -> 归位 ----
_objs0 = []
for _ob in [E.object] + ([E.aux] if getattr(E, "aux", None) is not None else []):
    _objs0.append((_ob, _ob.data.root_state_w.clone()))
q0 = q.clone()
dt = 1.0 / max(args.fps, 1.0)


def _apply_row(t):
    for s in ("right", "left"):
        P = "R" if s == "right" else "L"
        for i in range(7):
            q[:, jn.index(f"{P}_arm_j{i+1}")] = float(rows[f"{s}_q"][t, i])
        for n_, v_ in zip(GENERIC_JOINT_ORDER, rows[f"{s}_f"][t]):
            q[:, jn.index(n_.replace("right_", f"{s}_"))] = float(v_)
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.write_data_to_sim()


try:
    rounds = 0
    while True:
        q[:] = q0
        E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
        E.hand.write_data_to_sim()
        for _ob, _st in _objs0:
            _ob.write_root_pose_to_sim(_st[:, :7])
            _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))
        if args.selftest:
            for t in range(min(T2, args.selftest)):
                _apply_row(t)
                E.sim.step(render=False)
            print("[live] selftest 完成")
            break
        print("[live] 已定格在静止 GraspPose —— 按 Enter 播放 Retreat", flush=True)
        while True:
            E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            E.hand.write_data_to_sim()
            E.sim.step(render=True)
            _r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if _r:
                sys.stdin.readline()
                break
        for t in range(T2):
            _apply_row(t)
            E.sim.step(render=True)
            time.sleep(dt)
        rounds += 1
        print(f"[live] 第 {rounds} 遍放完, 归位定格", flush=True)
        if not app.is_running():
            break
except KeyboardInterrupt:
    pass
try:
    _slot.release()
except Exception:
    pass
app.close()
raise SystemExit(0)
