"""Unscrew 机器段 cuRobo 规划驱动 (照 pregrasp_suite 的 targets->worker 模式)。

  # Approach: 站姿 -> (cspace, 满障碍) -> pregrasp 内点构型 (v1 machine_pre_q_*)
  UNSCREW_CLIP=32 SHARPA_WANDB=0 PYTHONPATH=. $PY \\
      tasks/Unscrew/part4/A_Design/L1_Data/Motion_Planning/plan_machine_segs.py \\
      --headless
  # Retreat: 交互末行构型 -> (cspace, 物体钉在终位当障碍) -> 站姿
  ... plan_machine_segs.py --retreat --headless

产出 A_Design/L1_Data/Motion_Planning/<clip>/{Approach,Retreat}.npz;
之后**重跑 make_reference.py** 把规划行剪进母带 (它检测到产物自动消费)。

依赖: NVlabs/curobo 新版主线已安装；机器人配置默认使用仓库自带生成品，也可由
CUROBO_ROBOT_YML 覆写（MAGICSIM_ROOT 仅保留兼容）—— 见 curobo_plan_worker.py。
worker 在**干净子进程**里跑 (Isaac 起来后同进程 import cuRobo 会崩在 warp)。

顺序 (循环依赖的解法): probe_rest -> make_reference(占位机器段) -> 本脚本
-> make_reference(消费规划) -> build_reference(v2)。station 腕靶只依赖交互段
几何, 占位机器段不污染它。
"""
from __future__ import annotations

import argparse
import time

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
    E._SA.apply_screw(E, integrate_angle=False)
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
E._SA.apply_screw(E, integrate_angle=False)

W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
hand = E.hand
jn = list(hand.joint_names)
_root_p = hand.data.root_pos_w[0].cpu().numpy().astype(np.float64) - W
_e = clips.clip_entry(TC.CLIP)
_sec = _e["secondary"]
_sx, _sy, _sz = cfg.table_size
z1 = np.load(TC.REF_V1, allow_pickle=True)
rows_h = np.where(np.asarray(z1["source"]) == 1)[0]
planning_basis = TC.reference_planning_digest(TC.REF_V1)
rest_md5 = TC.file_md5(TC.REST_JSON)
assert rest_md5 is not None, "缺 env_rest.json；先运行 probe_rest.py"
segment_name = "retreat" if args.retreat else "approach"


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
    # 起点 = machine_pre 净空内点构型, **不是**交互末行 (T2-2):
    # 交互末行的手贴着瓶/盖/桌面 —— 碰撞检查规划器必判 "start in collision"
    # (2026-08-30 实测: 仅桌/仅瓶/仅杯世界起点微动全 ❌, 空世界 ✅)。
    # "松手撤离" (交互末行 -> pre) 由缝2 数据斜坡负责, 与缝1 的"贴近合拢"
    # 对称; cuRobo 只管 pre -> 站姿 的机器段 (物体钉母带终位当障碍)。
    # 用 **Approach 实际选中的那一档**净空 (两段必须首尾相接; Approach 换了
    # 候选而 Retreat 还用 0 档, 撤退起点就不是机器段真正停的地方)
    _pre_idx = 0
    if os.path.isfile(TC.APPROACH_NPZ):
        with np.load(TC.APPROACH_NPZ, allow_pickle=True) as _za:
            if "pre_idx" in _za.files:
                _pre_idx = int(_za["pre_idx"])
    if "machine_pre_alts_r" in z1.files:
        _qpre_r = np.asarray(z1["machine_pre_alts_r"], np.float64)[_pre_idx]
        _qpre_l = np.asarray(z1["machine_pre_alts_l"], np.float64)[_pre_idx]
    else:
        _qpre_r = np.asarray(z1["machine_pre_q_r"], np.float64)
        _qpre_l = np.asarray(z1["machine_pre_q_l"], np.float64)
    print(f"[plan] Retreat 起点 = pregrasp 候选 #{_pre_idx} (随 Approach)",
          flush=True)
    for i in range(1, 8):
        start[f"R_arm_j{i}"] = float(_qpre_r[i - 1])
        start[f"L_arm_j{i}"] = float(_qpre_l[i - 1])
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
    # ---- Approach: cspace 直达 pregrasp 内点构型 (2026-08-30 排障定稿) ----
    # 旧的双臂位姿模式对本任务全灭: 左腕镜像锚姿态超 j7 行程, ArmIK 钳限位
    # 出"达标"解, cuRobo 带限位余量把贴边构型全部拒收 (台账 T2-2 八轮二分)。
    # 现在目标 = make_reference 里收缩限位 3° ArmIK 解出的内点构型
    # (machine_pre_q_*), plan_cspace 全程避障且终帧=目标关节, 离线预演
    # (桌+物体充气1cm) 已全通。
    if "machine_pre_alts_r" in z1.files:
        alts_r = np.asarray(z1["machine_pre_alts_r"], np.float64)
        alts_l = np.asarray(z1["machine_pre_alts_l"], np.float64)
    else:                       # 旧母带: 只有一档
        alts_r = np.asarray(z1["machine_pre_q_r"], np.float64)[None]
        alts_l = np.asarray(z1["machine_pre_q_l"], np.float64)[None]
    T["cspace_goal"] = {}
    for i in range(1, 8):
        T["cspace_goal"][f"R_arm_j{i}"] = float(alts_r[0][i - 1])
        T["cspace_goal"][f"L_arm_j{i}"] = float(alts_l[0][i - 1])
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
# Approach/Retreat 现都走 cspace_goal (worker 按 targets 里键自动分支)
# Approach 还要逐档试 pregrasp 净空: 目标构型本身被判碰是本任务的常见死因
# (左抓锚一准, pregrasp 就贴在瓶壁上), 净空该给多少是逐 clip 的几何问题 ——
# 由近及远试, 第一条通的记进产物 (Retreat 起点跟着用同一档)。
n_alt = 1 if args.retreat else len(alts_r)
plan_payload, pre_idx = None, 0
for _k in range(n_alt):
    if not args.retreat:
        for i in range(1, 8):
            T["cspace_goal"][f"R_arm_j{i}"] = float(alts_r[_k][i - 1])
            T["cspace_goal"][f"L_arm_j{i}"] = float(alts_l[_k][i - 1])
        with open(_tgt, "w") as f:
            json.dump(T, f)
        print(f"[plan] === pregrasp 候选 #{_k}/{n_alt - 1} ===", flush=True)
    print(f"[plan] worker: {' '.join(cmd)}", flush=True)
    started_ns = time.time_ns()
    run = subprocess.run(cmd, cwd=os.getcwd(),
                         env=dict(os.environ, PYTHONPATH=os.getcwd()),
                         timeout=2400)
    fresh = (os.path.isfile(out_npz)
             and os.stat(out_npz).st_mtime_ns >= started_ns)
    if run.returncode != 0 or not fresh:
        print(f"[plan] ❌ worker 未产出本次结果: returncode={run.returncode} "
              f"fresh={fresh} out={out_npz}", flush=True)
        try:
            _slot.release()
        except Exception:
            pass
        app.close(); os._exit(1)
    with np.load(out_npz, allow_pickle=True) as _z:
        payload = {key: _z[key] for key in _z.files}
    if bool(payload["ok"]):
        plan_payload, pre_idx = payload, _k
        break
    print(f"[plan] 候选 #{_k} 失败: {payload.get('failed_frame')}"
          f"{' —— 换下一档净空' if _k + 1 < n_alt else ''}", flush=True)
if plan_payload is None and args.retreat:
    # 反向兜底: 同一个世界里改规划"站姿 -> pregrasp", 再**时间翻转**。
    # cspace 路径是几何量, 翻转后依然合法; 实测 Retreat 方向 (pregrasp -> 站姿)
    # 连空世界都解不出来, 而反方向 (Approach 用的就是这一对构型) 一次就通 ——
    # 是求解方向的问题, 不是碰撞问题 (两端微动探针都 ✅)。
    print("[plan] Retreat 正向失败 -> 反向规划 + 时间翻转 兜底", flush=True)
    _fwd_start = dict(T["start_joints"])
    _fwd_goal = dict(T["cspace_goal"])
    T["start_joints"] = dict(_fwd_start)
    for _k, _v in _fwd_goal.items():
        T["start_joints"][_k] = _v          # 起点 = 站姿
    T["cspace_goal"] = {_k: float(_fwd_start[_k]) for _k in _fwd_goal}
    with open(_tgt, "w") as f:
        json.dump(T, f)
    started_ns = time.time_ns()
    run = subprocess.run(cmd, cwd=os.getcwd(),
                         env=dict(os.environ, PYTHONPATH=os.getcwd()),
                         timeout=2400)
    if run.returncode == 0 and os.path.isfile(out_npz) \
            and os.stat(out_npz).st_mtime_ns >= started_ns:
        with np.load(out_npz, allow_pickle=True) as _z:
            _pl = {key: _z[key] for key in _z.files}
        if bool(_pl["ok"]):
            _pl["traj"] = np.ascontiguousarray(
                np.asarray(_pl["traj"], np.float64)[::-1])
            _pl["reversed_plan"] = np.array(True)
            plan_payload, pre_idx = _pl, 0
            print("[plan] ✅ 反向兜底成功 (轨迹已时间翻转)", flush=True)
if plan_payload is None and args.retreat and os.path.isfile(TC.APPROACH_NPZ):
    # 末级兜底: 直接复用 Approach 的路径**倒放**。
    # 两段的世界只差"盖已放到桌上", 而撤退是从 pregrasp 向上/向后离开, 与桌面
    # 上的盖不在同一空间; Approach 那条路已带满障碍碰撞背书 (含瓶)。cuRobo 在
    # pregrasp->站姿 这个方向上正反都解不出来 (空世界也失败 = 求解问题不是碰撞),
    # 与其退回无背书的 smoothstep 占位, 不如复用有背书的反向路径 —— 但**如实
    # 标注来源** (derived_from=approach_reversed), 指纹/凭据里看得见。
    with np.load(TC.APPROACH_NPZ, allow_pickle=True) as _za:
        _ap = {key: _za[key] for key in _za.files}
    if bool(_ap.get("ok", False)):
        _ap["traj"] = np.ascontiguousarray(
            np.asarray(_ap["traj"], np.float64)[::-1])
        _ap["derived_from"] = np.array("approach_reversed")
        plan_payload = _ap
        pre_idx = int(_ap.get("pre_idx", 0))
        print("[plan] ⚠ 复用 Approach 路径倒放作为 Retreat "
              "(来源已标注; 撤退世界少了桌上的盖这一项障碍)", flush=True)
if plan_payload is None:
    print(f"[plan] ❌ {n_alt} 档净空全部失败", flush=True)
    try:
        _slot.release()
    except Exception:
        pass
    os._exit(1)
plan_payload.update(
    pre_idx=np.array(pre_idx),
    plan_clip=np.array(TC.CLIP_ID), plan_segment=np.array(segment_name),
    planning_basis_digest=np.array(planning_basis),
    rest_md5=np.array(rest_md5))
tmp_plan = f"{out_npz}.tmp.{os.getpid()}"
with open(tmp_plan, "wb") as fh:
    np.savez(fh, **plan_payload)
os.replace(tmp_plan, out_npz)
print(f"[plan] ✅ {out_npz}: {len(plan_payload['traj'])} 行 | "
      f"记得重跑 make_reference.py 剪进母带", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
