"""用户设计版 Retreat (2026-08-27):
① 双手从 GraspPose 出发, 手指全部 弯曲->绷直 (臂定格, K 帧 smoothstep)
② 臂沿各自接近方向后退 5cm, 全程 cuRobo 规划驱动 (物体排除/桌保留, leg2 先例)

输出 Retreat_pour17.npz (旧自动版归档为 Retreat_pour17_v1auto.npz)。
Isaac 实测验收: 终帧摆进仿真, hand_C_MC 距退避靶点 <2.5cm。
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
p.add_argument("--fin_frames", type=int, default=45, help="手指绷直段帧数")
p.add_argument("--act_dist", type=float, default=0.015)
p.add_argument("--out_dir", default="tasks/Pour/17/A_Design/L1_Data/Motion_Planning")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("design_retreat")
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import (  # noqa: E402
    GENERIC_JOINT_ORDER,
)
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402


def qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.prior_a, args.yaw_a, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()
# ★ 先静置 90 步取**沉降后**的实测物体位姿 —— 出生位=贴桌+2mm 悬空, 沉降 2mm+微漂
#   会让贴身环抱抓(杯 1_Large_Diameter, 毫米级间隙)从"轻触"变"楔死"
#   (2026-08-27 验尸: 杯被左手楔住匀速拖倒, f82 过 60°)
for _ in range(90):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())

A_name = cfg.hand_side
B_name = "left" if A_name == "right" else "right"
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
_bst = E.object.data.root_state_w[0].cpu().numpy().astype(np.float64)
_cst = E.aux.data.root_state_w[0].cpu().numpy().astype(np.float64)
A_obj, A_oq = _bst[:3] - W, _bst[3:7]
B_obj, B_oq = _cst[:3] - W, _cst[3:7]
print(f"[design] 沉降后实测: 瓶 {np.round(A_obj*100,2)}cm | 杯 {np.round(B_obj*100,2)}cm")

za, zb = np.load(args.prior_a), np.load(args.prior_b)
# ★姿态链定案(2026-08-27 300步握持探针): 姿态用 canon∘yaw(用户两次目检的抓区,
#   有真实间隙), 位置用沉降实测(高度对齐)。沉降 quat 虽"忠实"却落在重建网格
#   会互穿的区域(差48.5°), 按 Enter 前杯即弹飞; 出生位链+沉降高=1.31°纹丝不动
def _chain(z_, yaw_):
    _aa = np.radians(yaw_)
    return qmul(np.array([np.cos(_aa/2), 0, 0, np.sin(_aa/2)]),
                np.asarray(z_["canon_rot"], np.float64))
# ★逐侧配方(2026-08-27 v4 复检教训: 过度对称是错的, 按各自证据配):
#   右(瓶) = 沉降实测姿态 —— v3 下瓶全程 0.0°, 换 canon 链反而带倒 12°
#   左(杯) = canon∘yaw90 —— 沉降姿态差 48.5° 落在互穿网格区, canon 链握持 1.31° 稳
PRI = {A_name: (za, A_oq, A_obj),
       B_name: (zb, _chain(zb, args.yaw_b), B_obj)}

# ---- 每侧: GraspPose 按**沉降实测位姿**解到世界系 / 接近方向 / 退避靶点 / 抓握臂位形 ----
grasp_w, tgt_w, arm_grasp, fin_grasp = {}, {}, {}, {}
for side in (A_name, B_name):
    z, oq, obj_p = PRI[side]
    oq = np.asarray(oq, np.float64)
    Ro = quat_to_R(oq)
    g = np.asarray(z["grasp"], np.float64)
    gp, gq = Ro @ g[:3] + obj_p, qmul(oq, g[3:7])
    pre0 = Ro @ np.asarray(z["pregrasp"], np.float64)[0][:3] + obj_p
    u = pre0 - gp
    u = u / max(np.linalg.norm(u), 1e-9)               # 接近方向的反向 = 退避方向
    tp = gp + (args.retreat_cm / 100.0) * u
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=E._anchor_T)
    r = ik.solve(gp, quat_to_R(gq), iters=300)
    assert r["pos_err"] < 0.01, f"{side} GraspPose IK 误差 {r['pos_err']*100:.2f}cm"
    grasp_w[side], tgt_w[side] = (gp, gq), (tp, gq)
    arm_grasp[side], fin_grasp[side] = r["q"], g[7:29]
    print(f"[design] {side}: Grasp {np.round(gp*100,1)}cm -> 退避 {np.round(tp*100,1)}cm "
          f"(方向 {np.round(u,3)}, {args.retreat_cm}cm)")

# ---- 手指绷直位形: 远端关节回 0, **每指根部关节钉在抓握值不动** (2026-08-27 用户设计) ----
_ROOT = {"thumb_CMC_FE", "thumb_CMC_AA", "index_MCP_FE", "index_MCP_AA",
         "middle_MCP_FE", "middle_MCP_AA", "ring_MCP_FE", "ring_MCP_AA",
         "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA"}
jn = list(E.hand.joint_names)
_lim = E.hand.root_physx_view.get_dof_limits()[0].cpu().numpy().astype(np.float64)
fin_straight = {}
for side in (A_name, B_name):
    names = [n.replace("right_", f"{side}_") for n in GENERIC_JOINT_ORDER]
    row = []
    for gi, n in enumerate(names):
        if n.split(f"{side}_", 1)[1] in _ROOT:
            row.append(float(np.asarray(PRI[side][0]["grasp"], np.float64)[7 + gi]))
        else:
            row.append(float(np.clip(0.0, _lim[jn.index(n), 0], _lim[jn.index(n), 1])))
    fin_straight[side] = np.array(row)

# ---- targets JSON (schema 与 view_curobo_plan 一致; 物体排除/桌保留) ----
_root_p = (E.hand.data.root_pos_w[0].cpu().numpy().astype(np.float64) - W)
_sx, _sy, _sz = cfg.table_size
_start = {n: float(v) for n, v in zip(
    jn, E.hand.data.default_joint_pos[0].cpu().numpy().astype(float))}
for side in (A_name, B_name):
    P = "R" if side == "right" else "L"
    for i in range(7):
        _start[f"{P}_arm_j{i+1}"] = float(arm_grasp[side][i])
    for n, v in zip([n_.replace("right_", f"{side}_") for n_ in GENERIC_JOINT_ORDER],
                    fin_straight[side]):
        _start[n] = float(v)
_start.update({"torso_j1": 0.7072, "torso_j2": 1.2856, "torso_j3": 0.0068,
               "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0})
_goals = {f"{s}_hand_C_MC": {"pos": (tgt_w[s][0] + W - _root_p).tolist(),
                             "quat": tgt_w[s][1].tolist()} for s in (A_name, B_name)}
_tmp = tempfile.mkdtemp(prefix="curobo_retreat_")
_tgt, _out = os.path.join(_tmp, "targets.json"), os.path.join(_tmp, "plan.npz")
with open(_tgt, "w") as f:
    json.dump({
        "table_pose": [float(W[0] - _root_p[0]), float(W[1] - _root_p[1]),
                       float(W[2] + cfg.table_top_z - _sz / 2 - _root_p[2])],
        "table_dims": [float(_sx), float(_sy), float(_sz)],
        "objects": [],                                  # 物体排除/桌保留 (leg2 先例)
        "start_joints": _start,
        "lock_joints": ["torso_j1", "torso_j2", "torso_j3",
                        "head_j1", "head_j2", "head_j3"]
                       + [n for n in jn if n.startswith(("right_", "left_"))],
        "tool_frames": [f"{A_name}_hand_C_MC", f"{B_name}_hand_C_MC"],
        "hand_targets": {},
        "goals": _goals,
        "pregrasp_goals": _goals,
    }, f)
print(f"[plan] 起 cuRobo 子进程: GraspPose -> 各自后退 {args.retreat_cm}cm (双手联合)")
subprocess.run([sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
                "--targets", _tgt, "--out", _out, "--act_dist", str(args.act_dist),
                "--joint", "1", "--left_short_cm", "0.0", "--table_pad", "0.0",
                "--exclude_objects", "1"],
               cwd=os.getcwd(), env=dict(os.environ, PYTHONPATH=os.getcwd()),
               timeout=2400)
assert os.path.exists(_out), "cuRobo 子进程没有产出"
_z = np.load(_out, allow_pickle=True)
assert bool(_z["ok"]), f"规划失败 (卡在 {_z.get('failed_frame')})"
traj = _z["traj"]
_jn = [str(x) for x in _z["joint_names"]]
print(f"[plan] ✅ {traj.shape[0]} 帧")

# ---- 拼装: ①指绷直段(臂定格) + ②cuRobo 退避段(指绷直) ----
arm_cols = {s: [_jn.index(f"{'R' if s == 'right' else 'L'}_arm_j{i}")
                for i in range(1, 8)] for s in ("right", "left")}
T2 = traj.shape[0]
K = min(max(2, int(args.fin_frames)), T2)
out = {}
# 最小修订(2026-08-27 用户裁定): 绷直与后退**并行** —— 原地绷直会刮蹭贴身杯抓
# (接触设计1-2mm), 第一圈就翻; 臂从第0帧起退, 指在前K帧内伸直, 接触即刻解除
for s in ("right", "left"):
    tt = np.clip(np.arange(T2) / max(K - 1, 1), 0.0, 1.0)
    ss = tt * tt * (3 - 2 * tt)
    out[f"{s}_q"] = traj[:, arm_cols[s]].astype(np.float32)
    out[f"{s}_f"] = ((1 - ss)[:, None] * fin_grasp[s]
                     + ss[:, None] * fin_straight[s]).astype(np.float32)
out["fin_names"] = np.array(GENERIC_JOINT_ORDER, dtype=object)
out["seg_names"] = np.array(["blend_straighten_retreat"], dtype=object)
out["seg_lens"] = np.array([T2], np.int64)
out["meta"] = (f"Retreat_pour17 并行版: cuRobo 退 {args.retreat_cm}cm({T2}帧), "
               f"指在前{K}帧绷直(根部钉住) 新场景")

# ---- Isaac 实测验收: 终帧摆进仿真, 量 hand_C_MC 距退避靶点 ----
qf = E.hand.data.default_joint_pos.clone()
for s in ("right", "left"):
    P = "R" if s == "right" else "L"
    for i in range(7):
        qf[:, jn.index(f"{P}_arm_j{i+1}")] = float(out[f"{s}_q"][-1, i])
    for n, v in zip([n_.replace("right_", f"{s}_") for n_ in GENERIC_JOINT_ORDER],
                    out[f"{s}_f"][-1]):
        qf[:, jn.index(n)] = float(v)
E.hand.write_joint_state_to_sim(qf, torch.zeros_like(qf))
E.hand.set_joint_position_target(qf)
for _ in range(5):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
_bn = list(E.hand.body_names)
_fail = False
for s in (A_name, B_name):
    real = E.hand.data.body_pos_w[0, _bn.index(f"{s}_hand_C_MC")].cpu().numpy() - W
    err = np.linalg.norm(real - tgt_w[s][0]) * 100
    ok = err < 2.5
    _fail |= not ok
    print(f"[isaac验收] {'✅' if ok else '❌'} {s} 终帧实测 {np.round(real*100,1)}cm "
          f"vs 退避靶 {np.round(tgt_w[s][0]*100,1)}cm | 差 {err:.2f}cm")
assert not _fail, "验收失败 —— 不要看回放, 先修坐标"

old = os.path.join(args.out_dir, "Retreat_pour17.npz")
if os.path.exists(old):
    shutil.move(old, os.path.join(args.out_dir, "Retreat_pour17_v1auto.npz"))
    print("[save] 旧自动版已归档 -> Retreat_pour17_v1auto.npz")
np.savez(old, **out)
print(f"[save] Retreat -> {old} ({T2} 行, 指绷直窗前{K}帧)")

try:
    _slot.release()
except Exception:
    pass
app.close()
raise SystemExit(0)
