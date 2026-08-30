"""Unscrew 机器段 cuRobo 规划驱动 (照 pregrasp_suite 的 targets->worker 模式)。

  # Approach: 站姿 -> (双臂联合, 满障碍) -> 站位腕靶 (v1 母带里的 station_wr/wl)
  UNSCREW_CLIP=32 SHARPA_WANDB=0 PYTHONPATH=. $PY \\
      tasks/Unscrew/part4/A_Design/L1_Data/Motion_Planning/plan_machine_segs.py \\
      --headless
  # Retreat: 交互末行构型 -> (cspace, 物体钉在终位当障碍) -> 站姿
  ... plan_machine_segs.py --retreat --headless

产出 A_Design/L1_Data/Motion_Planning/<clip>/{Approach,Retreat}.npz;
之后**重跑 make_reference.py** 把规划行剪进母带 (它检测到产物自动消费)。

依赖: MagicSim 定制版 cuRobo (curobo.motion_planner API) 已 pip 装进本解释器,
且 MAGICSIM_ROOT / CUROBO_ROBOT_YML 指到位 —— 见 curobo_plan_worker.py 头注。
worker 在**干净子进程**里跑 (Isaac 起来后同进程 import cuRobo 会崩在 warp)。

顺序 (循环依赖的解法): probe_rest -> make_reference(占位机器段) -> 本脚本
-> make_reference(消费规划) -> build_reference(v2)。station 腕靶只依赖交互段
几何, 占位机器段不污染它。
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--retreat", action="store_true", help="规划 Retreat (默认 Approach)")
p.add_argument("--act_dist", type=float, default=0.015)
p.add_argument("--attempts", type=int, default=20)
p.add_argument("--obj_inflate", type=float, default=0.01,
               help="物体障碍充气 (2026-08-20 用户裁定的净空配方)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("unscrew_plan")
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "C_Wiring"))
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402

os.environ.setdefault("POUR_NO_D6", "1")
cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0]
E.reset()
for _ in range(40):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
    E._SA.apply_screw(E)

W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
hand = E.hand
jn = list(hand.joint_names)
_root_p = hand.data.root_pos_w[0].cpu().numpy().astype(np.float64) - W
_e = clips.clip_entry(TC.CLIP)
_sec = _e["secondary"]
_sx, _sy, _sz = cfg.table_size
z1 = np.load(TC.REF_V1, allow_pickle=True)
rows_h = np.where(np.asarray(z1["source"]) == 1)[0]

# 物体障碍 (env 系 -> 底座系只减 _root_p, 见 pregrasp_suite 的 2026-08-18 坑注)
bot_p = (E.object.data.root_pos_w[0].cpu().numpy() - W).astype(np.float64)
bot_q = E.object.data.root_quat_w[0].cpu().numpy().astype(np.float64)
cap_p = (E.aux.data.root_pos_w[0].cpu().numpy() - W).astype(np.float64)
cap_q = E.aux.data.root_quat_w[0].cpu().numpy().astype(np.float64)
if args.retreat:
    # Retreat: 物体钉在**母带终位** (盖在桌上, 瓶回位) —— 撤退要躲的就是成果
    bot_end = np.asarray(z1["obj_pos_0"], np.float64)[rows_h][-1]
    bot_q = np.asarray(z1["obj_quat_0"], np.float64)[rows_h][-1]
    cap_end = np.asarray(z1["obj_pos_1"], np.float64)[rows_h][-1]
    cap_q = np.asarray(z1["obj_quat_1"], np.float64)[rows_h][-1]
    bot_p, cap_p = bot_end, cap_end
objs = [{"name": "obj_primary", "mesh": _e["mesh"],
         "pos": (bot_p - _root_p).tolist(), "quat": bot_q.tolist()},
        {"name": "obj_secondary", "mesh": _sec["mesh"],
         "pos": (cap_p - _root_p).tolist(), "quat": cap_q.tolist()}]

# start_joints: 站姿 (Approach) / 交互末行 (Retreat); 躯干锁死角 = 新场景站姿
# (fixedtorso USD 烘的就是它; 旧 pregrasp_suite 的 45/90/0 是旧世界值, 勿抄)
dq = hand.data.default_joint_pos[0].cpu().numpy().astype(float)
start = {n: float(v) for n, v in zip(jn, dq)}
if args.retreat:
    for i in range(1, 8):
        start[f"R_arm_j{i}"] = float(np.asarray(z1["right_q"])[rows_h][-1][i - 1])
        start[f"L_arm_j{i}"] = float(np.asarray(z1["left_q"])[rows_h][-1][i - 1])
start.update({"torso_j1": float(np.radians(40.5196)),
              "torso_j2": float(np.radians(73.6595)),
              "torso_j3": float(np.radians(0.3896)),
              "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0})
lock = (["torso_j1", "torso_j2", "torso_j3", "head_j1", "head_j2", "head_j3"]
        + [n for n in jn if n.startswith(("right_", "left_"))])

T = {"table_pose": [float(-_root_p[0]), float(-_root_p[1]),
                    float(cfg.table_top_z - _sz / 2 - _root_p[2])],
     "table_dims": [float(_sx), float(_sy), float(_sz)],
     "objects": objs, "start_joints": start, "lock_joints": lock,
     "tool_frames": ["right_hand_C_MC", "left_hand_C_MC"]}

if args.retreat:
    # ---- Retreat: cspace 直达站姿 (终帧=目标关节, 全程避障, 无贴合抖动) ----
    T["cspace_goal"] = {f"{P}_arm_j{i}": float(dq[jn.index(f"{P}_arm_j{i}")])
                        for P in ("R", "L") for i in range(1, 8)}
    out_npz = TC.RETREAT_NPZ
else:
    # ---- Approach: 双臂联合位姿规划到站位腕靶 (v1 母带的数据推导目标) ----
    st_r = np.asarray(z1["station_wr"], np.float64)   # (7,) env 系
    st_l = np.asarray(z1["station_wl"], np.float64)
    # pregrasp: 右手 = 靶上方 8cm (顶抓沿螺轴退); 左手 = 沿瓶轴径向外退 8cm
    rad = st_l[:3] - bot_p
    rad[2] = 0.0
    rad = rad / max(np.linalg.norm(rad), 1e-9)
    T["goals"] = {
        "right_hand_C_MC": {"pos": (st_r[:3] - _root_p).tolist(),
                            "quat": st_r[3:7].tolist()},
        "left_hand_C_MC": {"pos": (st_l[:3] - _root_p).tolist(),
                           "quat": st_l[3:7].tolist()}}
    T["pregrasp_goals"] = {
        "right_hand_C_MC": {"pos": (st_r[:3] + [0, 0, 0.08] - _root_p).tolist(),
                            "quat": st_r[3:7].tolist()},
        "left_hand_C_MC": {"pos": (st_l[:3] + 0.08 * rad - _root_p).tolist(),
                           "quat": st_l[3:7].tolist()}}
    T["hand_targets"] = {"right_hand_C_MC": "obj_secondary",
                         "left_hand_C_MC": "obj_primary"}
    out_npz = TC.APPROACH_NPZ

os.makedirs(os.path.dirname(out_npz), exist_ok=True)
_tmp = tempfile.mkdtemp(prefix="unscrew_plan_")
_tgt = os.path.join(_tmp, "targets.json")
with open(_tgt, "w") as f:
    json.dump(T, f)
cmd = [sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
       "--targets", _tgt, "--out", out_npz,
       "--act_dist", str(args.act_dist), "--attempts", str(args.attempts),
       "--obj_inflate", str(args.obj_inflate)]
if not args.retreat:
    cmd += ["--joint", "1"]
print(f"[plan] worker: {' '.join(cmd)}", flush=True)
subprocess.run(cmd, cwd=os.getcwd(),
               env=dict(os.environ, PYTHONPATH=os.getcwd()), timeout=2400)
_z = np.load(out_npz, allow_pickle=True)
if not bool(_z["ok"]):
    print(f"[plan] ❌ 失败: {_z.get('failed_frame')}", flush=True)
    try:
        _slot.release()
    except Exception:
        pass
    os._exit(1)
print(f"[plan] ✅ {out_npz}: {len(_z['traj'])} 行 | "
      f"记得重跑 make_reference.py 剪进母带", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
