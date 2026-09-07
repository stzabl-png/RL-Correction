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
# 机器段规划必须以刚生成的 v1 为唯一母带。目录里可能还残留旧 v2；若先让
# task_env 自动选择它，场景长度/物体终态就会与本轮 P0/P1 规划摘要不一致。
os.environ["POUR_REF_NPZ"] = TC.REF_V1
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
# 第一腿不从贴物的任务端点起步：Approach 使用 P1 前的净空点；
# Retreat 使用任务末行后的净空点。贴物短腿稍后在收缩目标物世界
# 中另行规划，所以 P0->P1 和任务末行->P0 的所有臂运动仍都是 cuRobo。
if args.retreat:
    alts_r = np.asarray(z1["machine_retreat_pre_alts_r"], np.float64)
    alts_l = np.asarray(z1["machine_retreat_pre_alts_l"], np.float64)
else:
    alts_r = np.asarray(z1["machine_pre_alts_r"], np.float64)
    alts_l = np.asarray(z1["machine_pre_alts_l"], np.float64)
assert alts_r.shape == alts_l.shape and alts_r.ndim == 2 and alts_r.shape[1] == 7
if args.retreat:
    for i in range(1, 8):
        start[f"R_arm_j{i}"] = float(alts_r[0, i - 1])
        start[f"L_arm_j{i}"] = float(alts_l[0, i - 1])
start.update({"torso_j1": float(np.radians(40.5196)),
              "torso_j2": float(np.radians(73.6595)),
              "torso_j3": float(np.radians(0.3896)),
              "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0})
lock = (["torso_j1", "torso_j2", "torso_j3", "head_j1", "head_j2", "head_j3"]
        + [n for n in jn if n.startswith(("right_", "left_"))])

# 操作位置 = 母带交互首行的双臂关节角 (机器段真正的终点 —— 见下方两腿说明)
_rows_ia = np.flatnonzero(np.asarray(z1["source"], np.int8) == 1)
if args.retreat:
    _station_r = np.asarray(z1["right_q"], np.float64)[_rows_ia[-1]]
    _station_l = np.asarray(z1["left_q"], np.float64)[_rows_ia[-1]]
else:
    _station_r = np.asarray(z1["right_q"], np.float64)[_rows_ia[0]]
    _station_l = np.asarray(z1["left_q"], np.float64)[_rows_ia[0]]
STATION_GOAL = {}
for i in range(1, 8):
    STATION_GOAL[f"R_arm_j{i}"] = float(_station_r[i - 1])
    STATION_GOAL[f"L_arm_j{i}"] = float(_station_l[i - 1])

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
# python.sh 依靠 PYTHONPATH 里的 Isaac pip_prebundle 提供 numpy/torch/warp。旧代码把
# PYTHONPATH 整个覆盖成 cwd，于是“干净子进程”连 numpy 都 import 不了。
# 正确做法是在现有 Isaac 路径前追加仓库，不修改其他环境。
_worker_env = dict(os.environ)
_local_curobo = os.path.join(os.getcwd(), ".omc", "curobo")
_local_curobo_deps = os.path.join(os.getcwd(), ".omc", "curobo_deps")
_worker_env["PYTHONPATH"] = os.pathsep.join(
    p for p in (os.getcwd(), _local_curobo if os.path.isdir(_local_curobo) else "",
                _local_curobo_deps if os.path.isdir(_local_curobo_deps) else "",
                os.environ.get("PYTHONPATH", "")) if p)
# Approach/Retreat 现都走 cspace_goal (worker 按 targets 里键自动分支)
# Approach/Retreat 都逐档试自己的净空候选。这两组候选分别锚在 P1
# 和任务末行，不再共用索引，也不会把 Approach 路径倒放成 Retreat。
n_alt = len(alts_r)
plan_payload, pre_idx = None, 0
for _k in range(n_alt):
    if args.retreat:
        for i in range(1, 8):
            T["start_joints"][f"R_arm_j{i}"] = float(alts_r[_k][i - 1])
            T["start_joints"][f"L_arm_j{i}"] = float(alts_l[_k][i - 1])
    else:
        for i in range(1, 8):
            T["cspace_goal"][f"R_arm_j{i}"] = float(alts_r[_k][i - 1])
            T["cspace_goal"][f"L_arm_j{i}"] = float(alts_l[_k][i - 1])
    with open(_tgt, "w") as f:
        json.dump(T, f)
    print(f"[plan] === {'Retreat' if args.retreat else 'Approach'} 净空候选 "
          f"#{_k}/{n_alt - 1} ===", flush=True)
    print(f"[plan] worker: {' '.join(cmd)}", flush=True)
    started_ns = time.time_ns()
    run = subprocess.run(cmd, cwd=os.getcwd(), env=_worker_env,
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
    # cspace 路径是几何量，同一个 Retreat 末态障碍世界中做 P0->净空点
    # 再时间翻转，仍是合法的净空点->P0。这不是复用 Approach。
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
    run = subprocess.run(cmd, cwd=os.getcwd(), env=_worker_env,
                         timeout=2400)
    if run.returncode == 0 and os.path.isfile(out_npz) \
            and os.stat(out_npz).st_mtime_ns >= started_ns:
        with np.load(out_npz, allow_pickle=True) as _z:
            _pl = {key: _z[key] for key in _z.files}
        if bool(_pl["ok"]):
            _pl["traj"] = np.ascontiguousarray(
                np.asarray(_pl["traj"], np.float64)[::-1])
            _pl["reversed_plan"] = np.array(True)
            plan_payload, pre_idx = _pl, n_alt - 1
            print("[plan] ✅ 反向兜底成功 (轨迹已时间翻转)", flush=True)
if plan_payload is None:
    print(f"[plan] ❌ {n_alt} 档净空全部失败", flush=True)
    try:
        _slot.release()
    except Exception:
        pass
    os._exit(1)
_qpre_r = np.asarray(alts_r[pre_idx], np.float64)
_qpre_l = np.asarray(alts_l[pre_idx], np.float64)
# ================= 第二腿: 净空点 <-> 数据边界 =================
# Approach 的数据边界是 P1；Retreat 的数据边界是任务实际末行。这两个构型
# 都可能贴着目标物，在完整障碍世界会被判 goal/start in collision；因此用
# 收缩目标物的世界单独规划这条短腿，再与完整障碍世界的第一腿拼接。
_leg2_ok = False
if not bool(np.asarray(plan_payload.get("derived_from", ""))
            == "approach_reversed"):
    _T2 = dict(T)
    if args.retreat:      # 操作位置 -> 净空点 (撤退的第一腿)
        _s2 = dict(T["start_joints"])
        _s2.update(STATION_GOAL)
        _T2["start_joints"] = _s2
        _T2["cspace_goal"] = {}
        for i in range(1, 8):
            _T2["cspace_goal"][f"R_arm_j{i}"] = float(_qpre_r[i - 1])
            _T2["cspace_goal"][f"L_arm_j{i}"] = float(_qpre_l[i - 1])
    else:                 # 净空点 -> 操作位置 (接近的第二腿)
        _s2 = dict(T["start_joints"])
        for i in range(1, 8):
            _s2[f"R_arm_j{i}"] = float(alts_r[pre_idx][i - 1])
            _s2[f"L_arm_j{i}"] = float(alts_l[pre_idx][i - 1])
        _T2["start_joints"] = _s2
        _T2["cspace_goal"] = dict(STATION_GOAL)
    _tgt2 = os.path.join(_tmp, "targets_leg2.json")
    with open(_tgt2, "w") as f:
        json.dump(_T2, f)
    _out2 = os.path.join(_tmp, "leg2.npz")   # 临时目录: 别在产物目录里留垃圾
                                            # (失败的中间产物混进仓库会误导)
    _cmd2 = [sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
             "--targets", _tgt2, "--out", _out2,
             "--act_dist", str(args.act_dist), "--attempts", str(args.attempts),
             # 物体**收缩** 1.2cm 而不是排除: 贴着抓的目标位形合法, 但路径仍会
             # 绕开物体本体 (整个排除 = 这一腿完全没有避障, 等于退回手写进刀)。
             # 桌面单独排除: 操作位置离桌只有 ~8cm, cuRobo 的手部碰撞球比实物
             # 保守会把目标判碰, 而这一腿只在站位高度附近平移。
             # 撤退腿按物体收缩 (T2-26): 起点只被贴指的**盖**判碰 => 盖 -2cm
             # 解锁起点; 瓶保持 -1.2cm —— 全局 -2cm 实测让规划器从瓶边穿过
             # (瓶 r3.25cm 只剩 1.25cm, 用户视频抓到右掌扫瓶)。
             "--obj_inflate", "-0.012",
             *(["--obj_inflate_map",
                os.environ.get("UNSCREW_RET_INFLATE_MAP",
                               '{"obj_secondary": -0.02}')]
               if args.retreat else []),
             "--exclude_table", "1",
             # 兜底 (T2-27): 撤退近腿起点被贴指物体判碰且收缩救不动时,
             # UNSCREW_RET_EXCLUDE_OBJS=1 整体排除物体 —— 该腿是抬手离开的
             # 81 行短腿, 长路径避障仍由第一腿的完整障碍世界背书。
             *(["--exclude_objects", "1"]
               if args.retreat and os.environ.get("UNSCREW_RET_EXCLUDE_OBJS")
               else [])]
    print(f"[plan] === 第二腿: {'操作位置->净空点' if args.retreat else '净空点->操作位置'}"
          f" (目标物体排除出碰撞世界, 桌子仍是障碍) ===", flush=True)
    _r2 = subprocess.run(_cmd2, cwd=os.getcwd(), env=_worker_env,
                         timeout=2400)
    if _r2.returncode == 0 and os.path.isfile(_out2):
        with np.load(_out2, allow_pickle=True) as _z2:
            _p2 = {k: _z2[k] for k in _z2.files}
        if bool(_p2["ok"]):
            _t1 = np.asarray(plan_payload["traj"], np.float64)
            _t2 = np.asarray(_p2["traj"], np.float64)
            plan_payload["traj"] = np.ascontiguousarray(
                np.concatenate([_t1, _t2[1:]] if not args.retreat
                               else [_t2, _t1[1:]], axis=0))
            plan_payload["leg2_rows"] = np.array(len(_t2) - 1)
            _leg2_ok = True
            print(f"[plan] ✅ 第二腿 {len(_t2)} 行, 拼接后共 "
                  f"{len(plan_payload['traj'])} 行 —— 机器段终点=操作位置",
                  flush=True)
        else:
            print(f"[plan] ❌ 第二腿失败 ({_p2.get('failed_frame')}): "
                  "不能满足完整 cuRobo 边界", flush=True)
    else:
        print("[plan] ❌ 第二腿 worker 未产出: 不能满足完整 cuRobo 边界",
              flush=True)
if not _leg2_ok:
    plan_payload["ok"] = np.array(False)
    plan_payload["failed_frame"] = np.array("boundary_leg")
plan_payload.update(
    leg2_ok=np.array(_leg2_ok),
    pre_idx=np.array(pre_idx),
    plan_clip=np.array(TC.CLIP_ID), plan_segment=np.array(segment_name),
    planning_basis_digest=np.array(planning_basis),
    rest_md5=np.array(rest_md5))
tmp_plan = f"{out_npz}.tmp.{os.getpid()}"
with open(tmp_plan, "wb") as fh:
    np.savez(fh, **plan_payload)
os.replace(tmp_plan, out_npz)
print(f"[plan] {'✅' if _leg2_ok else '❌'} {out_npz}: "
      f"{len(plan_payload['traj'])} 行 | "
      + ("记得重跑 make_reference.py 剪进母带" if _leg2_ok
         else "仅留 ok=False 诊断产物，不得用于母带"), flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0 if _leg2_ok else 1)
