"""Sweep408 握持探针 (照 Clean/3 的 probe_hold 同款).

  SHARPA_WANDB=0 $PY tasks/Sweep/408/C_Wiring/probe_hold.py --headless [--num_envs 4] [--hold_s 3]
  GUI 目检:  ... --render --kinematic            (不加 --headless)

与 Clean 版的差别 (2026-09-11):
  --traj: human=手走人手腕轨/物体听手 · obj=物体走重建物轨/手被带着走 · ref=直接放母带。
  其余同款: 物体按 GraspPose 的 T_oh 钉在手上; --kinematic 时不开物理**并关掉物体碰撞**
  (Clean 实测不关碰撞时物理子步会按穿插把物体踢开 0.9cm/9°)。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--hold_s", type=float, default=3.0)
p.add_argument("--f0", type=int, default=0, help="人手轨迹起始帧")
p.add_argument("--f1", type=int, default=-1, help="结束帧 (-1=末帧)")
p.add_argument("--traj", choices=("human", "obj", "ref"), default="human",
               help="human=手走人手重定向腕轨迹, 物体听手; obj=物体走重建物轨, 手被物体带着走 "
                    "(手 = 物体 ∘ T_oh, 母带的约定); ref=直接放母带 (臂关节/指姿/物体位姿全取自母带)")
p.add_argument("--tape", default="tasks/Sweep/408/A_Design/L2_Reference/sweep408_reference_v1.npz")
p.add_argument("--kinematic", action="store_true", help="不开物理: 手物钉住 + 关物体碰撞 (纯目检)")
p.add_argument("--render", action="store_true", help="GUI 目检: 每步渲染")
p.add_argument("--fps", type=float, default=15.0)
p.add_argument("--broom_mass", type=float, default=0.15)
p.add_argument("--pan_mass", type=float, default=0.20)
p.add_argument("--mu", type=float, default=1.0)
p.add_argument("--prior_yaw", type=float, default=0.0,
               help="物体 yaw(度); <0 = 让 env 自搜可达带 (诊断用, 起手就握住的任务里桌面 yaw 无意义)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

# ★ 物理规矩必须在 import correction_env 之前 (PHYS_RULE 是导入期 setdefault)
os.environ.setdefault("SHARPA_WANDB", "0")
os.environ["POUR_OBJ_MASS"] = str(args.broom_mass)
os.environ["POUR_OBJ_FRIC"] = str(args.mu)
os.environ["POUR_PAD_FRIC"] = str(args.mu)
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep408_probe")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

_REPO = "/home/lyh/Project/RL_Correction"
RECON = "/home/lyh/Project/RL_Correction/datasets/sweep408"
CLIP = "Sweep408_broom"
PRIOR = {"right": f"{_REPO}/tasks/pregrasp/priors/Sweep408_broom.npz",
         "left":  f"{_REPO}/tasks/pregrasp/priors/Sweep408_dustpan.npz"}
OID = {"right": 0, "left": 1}          # object_0=扫把(右), object_1=簸箕(左)


def _norm(q):
    return np.asarray(q, np.float64) / max(np.linalg.norm(q), 1e-9)


def R_to_q(R):
    m = np.asarray(R, np.float64); t = np.trace(m)
    if t > 0:
        sq = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * sq, (m[2, 1] - m[1, 2]) / sq, (m[0, 2] - m[2, 0]) / sq, (m[1, 0] - m[0, 1]) / sq])
    else:
        i = int(np.argmax(np.diag(m))); j, k = (i + 1) % 3, (i + 2) % 3
        sq = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = np.zeros(4); q[0] = (m[k, j] - m[j, k]) / sq; q[i + 1] = 0.25 * sq
        q[j + 1] = (m[j, i] + m[i, j]) / sq; q[k + 1] = (m[k, i] + m[i, k]) / sq
    return _norm(q)


# ---- 建环境 (借 clip 起 DexMate + 两物体) ----
cfg = GraspTaskCfg()
clips.configure_cfg(cfg, CLIP)
cfg.approach_only = True
cfg.fixed_attached_tools = True
cfg.canon_rest_override = True   # 扫把最大支撑面 ≠ Dexonomy 静置面(上轴差 94.3°), 必须按后者摆
apply_grasp_prior(cfg, PRIOR["right"], float(args.prior_yaw), approach=True)

cfg.scene.num_envs = int(args.num_envs)
cfg.obj_jitter_xy = 0.0
cfg.observation_space = 145      # 探针不训练, 只需与基类实际拼出的宽度对上 (accumulate_finger 22 + dyn_res 1)
E = GraspTaskEnv(cfg)
E.reset()
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
DEV = E.device

if args.kinematic:
    import omni.usd
    from pxr import Usd, UsdPhysics
    _stage = omni.usd.get_context().get_stage()
    _n = 0
    for _ei in range(cfg.scene.num_envs):
        for _nm in ("Object", "Aux"):
            _root = _stage.GetPrimAtPath(f"/World/envs/env_{_ei}/{_nm}")
            # ★ 必须穿过**实例代理**: 物体 USD 是 instanceable 的, 真正的 collider 挂在
            #   /Props 原型里; 默认 PrimRange 不进去, 只会关到顶层那 1 个, 碰撞其实还开着。
            #   后果不是'物体被踢开'而是直接炸: 手物重度穿插 -> PhysX GPU 非法访存
            #   (integrateCoreParallel fail / error 700), 整个 app 挂掉。
            for _pr in Usd.PrimRange(_root, Usd.TraverseInstanceProxies()):
                if _pr.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(_pr).GetCollisionEnabledAttr().Set(False); _n += 1
    print(f"[Sweep408Probe] kinematic: 物体碰撞已关 ×{_n} collider", flush=True)

# ---- 轨迹来源 + 先验 ----
REF, PR = {}, {}
for side in ("right", "left"):
    z = np.load(f"{RECON}/ref_qpos_{side}.npz", allow_pickle=True)
    REF[side] = dict(wp=np.asarray(z["wrist_pos"], np.float64), wq=np.asarray(z["wrist_quat_wxyz"], np.float64),
                     fq=np.asarray(z["finger_qpos"], np.float64), names=[str(x) for x in z["joint_names"]])
    PR[side] = np.asarray(np.load(PRIOR[side])["grasp"], np.float64)
NF = len(REF["right"]["wp"])
f0 = max(0, args.f0); f1 = NF - 1 if args.f1 < 0 else min(args.f1, NF - 1)

TAPE = None
if args.traj == "ref":
    _tp = args.tape if os.path.isabs(args.tape) else os.path.join(_REPO, args.tape)
    TAPE = np.load(_tp, allow_pickle=True)
    NF = len(TAPE["right_q"])
    f0 = max(0, args.f0); f1 = NF - 1 if args.f1 < 0 else min(args.f1, NF - 1)
    print(f"[Sweep408Probe] 母带 {_tp}: {NF} 行 @{float(TAPE['control_hz']):.0f}Hz | "
          f"scene_yaw={float(TAPE['scene_yaw_deg']):.0f}° 中心={np.round(TAPE['scene_center_xy'], 3)} | "
          f"压深 {float(TAPE['press_depth_m'])*1000:.0f}mm 簸箕离桌 {float(TAPE['pan_lip_gap_m'])*1000:.0f}mm", flush=True)
    print(f"[Sweep408Probe] 母带 ik_report: {str(TAPE['ik_report'])}", flush=True)

# ---- traj=obj: 物体走重建物轨, 腕位姿由 手 = 物体 ∘ T_oh 推出来 ----
# 重建物轨在 **raw 网格系**, 入库网格是 Dexonomy 输入系 —— 关系精确为
#   v_in = s·v_raw − com     (s=1.30/1.25, com=info/simplified.json 的 com_offset;
#                             2026-09-11 实测残差 0.000mm, 补把手只加顶点不改坐标)
# 入库网格原点 = 缩放后网格的 COM, 对应 raw 系的点 com/s。所以物体位姿取
#   R_obj = R_rec ,  p_obj = t_rec + R_rec·(com/s)
# 即"让该物理点走重建轨迹, 物体绕它按 r2 尺度放大"。物体**不缩回** raw 尺度:
# 放大是 r2 交付有意为之 (重建的 sam3d_scale 偏小), 且 GraspPose 就是按放大后合的。
OBJ = {}
if args.traj == "obj":
    import json as _json
    _DX = {"right": "p4t408_broom_r2", "left": "p4t408_dustpan_r2"}
    _S = {"right": 1.30, "left": 1.25}
    for side in ("right", "left"):
        _T = np.asarray(np.load(f"{RECON}/poseqa/rts_sweep_dustpan_408_object_{OID[side]}.npz")
                        ["object_ob_in_world_smooth"], np.float64)
        _com = np.asarray(_json.load(open(
            f"/home/lyh/Project/Dexonomy/assets/object/custom/processed_data/{_DX[side]}"
            f"/info/simplified.json"))["com_offset"], np.float64)
        _c = _com / _S[side]
        OBJ[side] = np.stack([np.concatenate([_T[t][:3, 3] + _T[t][:3, :3] @ _c,
                                              R_to_q(_T[t][:3, :3])]) for t in range(len(_T))])
        assert len(OBJ[side]) == NF, (len(OBJ[side]), NF)

_bn = list(E.hand.body_names)
_rest = (E.hand.data.body_pos_w[0, _bn.index("right_hand_C_MC")] - E.scene.env_origins[0]).cpu().numpy().astype(np.float64)


def _wrist_target(side, t):
    """该帧腕(hand_C_MC)在**重建世界系**的目标位姿 (未加整体平移)."""
    if args.traj == "human":
        return REF[side]["wp"][t], _norm(REF[side]["wq"][t])
    op, oq = OBJ[side][t][:3], _norm(OBJ[side][t][3:7])      # 物体位姿
    Ro = quat_to_R(oq)                                       # 手 = 物体 ∘ T_oh
    return op + Ro @ PR[side][:3], R_to_q(Ro @ quat_to_R(_norm(PR[side][3:7])))


_shift = (np.zeros(3) if args.traj == "ref"
          else _rest - _wrist_target("right", f0)[0])
print(f"[Sweep408Probe] traj={args.traj} | 轨迹整体平移 {np.round(_shift, 3)} m "
      + ("(母带已在机器人场景系, 不再平移)" if args.traj == "ref" else f"(右腕 f{f0} -> 机器人静置腕位)"), flush=True)

IK = {s: ArmIK(s, anchor_link="arm_center", anchor_T=E._anchor_T) for s in ("right", "left")}
# 母带里存了造带时用的 anchor; 与本 env 实测 anchor 不一致的话臂角全错, 必须当场叫出来
if TAPE is not None and "anchor_T" in TAPE.files:
    _da = np.abs(np.asarray(TAPE["anchor_T"], np.float64) - E._anchor_T).max()
    print(f"[Sweep408Probe] anchor 核对: 母带 vs env 实测 最大分量差 {_da*1000:.3f}mm", flush=True)
    assert _da < 2e-3, f"母带 anchor 与 env 实测差 {_da*1000:.1f}mm —— 臂角会整体错位, 先统一再放"
JID = {s: [E.hand.joint_names.index(f"{'R' if s == 'right' else 'L'}_arm_j{i}") for i in range(1, 8)] for s in IK}
FIN = {s: [(E.hand.joint_names.index(n.replace("right_", f"{s}_", 1)), k)
           for k, n in enumerate(GENERIC_JOINT_ORDER) if n.replace("right_", f"{s}_", 1) in E.hand.joint_names]
       for s in ("right", "left")}

ARM = {s: {} for s in IK}
if args.traj == "ref":
    for s in ("right", "left"):
        q = np.asarray(TAPE[f"{s}_q"], np.float64)
        for t in range(f0, f1 + 1):
            ARM[s][t] = q[t]
    print(f"[Sweep408Probe] 臂角直接取自母带 (未做 IK); 指姿取自母带 {'right_f'} / {'left_f'}", flush=True)
else:
    for s in ("right", "left"):
        seed, perr, rerr = None, [], []
        for t in range(f0, f1 + 1):
            _p, _q = _wrist_target(s, t)
            r = IK[s].solve(_p + _shift + W, quat_to_R(_q), q0=seed, iters=150)
            seed = r["q"]; ARM[s][t] = np.asarray(r["q"], np.float64)
            perr.append(r["pos_err"]); rerr.append(r["rot_err"])
        perr, rerr = np.array(perr), np.degrees(np.array(rerr))
        print(f"[Sweep408Probe] {s} 臂 IK {len(ARM[s])}帧 | 位置误差 中位{np.median(perr)*100:.1f}cm "
              f"max{perr.max()*100:.1f}cm | >5cm {int((perr > 0.05).sum())}帧 | "
              f"姿态误差 中位{np.median(rerr):.1f}° max{rerr.max():.1f}°", flush=True)

# ---- traj=obj 专属体检: 手物相对关系实际被 IK 残差撑开多少 ----
if args.traj == "obj":
    for s in ("right", "left"):
        dp, dr = [], []
        Roh = quat_to_R(_norm(PR[s][3:7]))
        for t in range(f0, f1 + 1):
            hp, hR = IK[s].fk(ARM[s][t])
            op = OBJ[s][t][:3] + _shift + W
            oR = quat_to_R(_norm(OBJ[s][t][3:7]))
            dp.append(np.linalg.norm(oR.T @ (hp - op) - PR[s][:3]))
            _d = Roh.T @ (oR.T @ hR)
            dr.append(np.degrees(np.arccos(np.clip((np.trace(_d) - 1) / 2, -1, 1))))
        dp, dr = np.array(dp) * 100, np.array(dr)
        print(f"[Sweep408Probe] {s} 手物关系漂移(相对先验): 位置 中位{np.median(dp):.2f}cm "
              f"p95 {np.percentile(dp, 95):.2f}cm max {dp.max():.2f}cm | "
              f"姿态 中位{np.median(dr):.1f}° max {dr.max():.1f}°", flush=True)

ART = {"right": E.object, "left": E.aux}


def hand_pose(side, t):
    """手 hand_C_MC 的世界位姿, 由 ArmIK 正解给 —— **不读仿真**。

    读 `E.hand.data.body_pos_w` 需要先 `sim.step()` 刷新物理视图, 而 --kinematic 下
    推物理正是崩因: 闸已报"臂外壳离桌余量 -2.7cm"(手臂扎进桌子), 一步进去 PhysX
    就 solveStaticBlock/integrateCoreParallel fail -> CUDA error 700 整个 app 挂掉。
    IK 用的就是 env 实测的 anchor_T, 正解与仿真同源, 目检不需要物理。
    """
    hp, hR = IK[side].fk(ARM[side][t])          # IK 目标本来就带 +W, 所以这是世界系
    return hp, R_to_q(hR)


def set_frame(t):
    q = E.hand.data.default_joint_pos.clone()
    for s in ("right", "left"):
        q[:, JID[s]] = torch.tensor(ARM[s][t], dtype=torch.float32, device=DEV)
        _f = np.asarray(TAPE[f"{s}_f"], np.float64)[t] if TAPE is not None else PR[s][7:29]
        for ji, k in FIN[s]:                       # 指姿: ref=母带指行, 否则 GraspPose 抓握姿
            q[:, ji] = float(_f[k])
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.set_joint_position_target(q)
    E.hand.write_data_to_sim()
    # human: 物体 = 手 ∘ inv(T_oh) (物体听手)  |  obj: 物体走自己的重建轨迹, 手已由 IK 跟上
    out = {}
    for s in ("right", "left"):
        hp_np, hq_np = hand_pose(s, t)
        if args.traj == "ref":
            _oi = OID[s]
            p_np = np.asarray(TAPE[f"obj_pos_{_oi}"], np.float64)[t] + W
            q_np = _norm(np.asarray(TAPE[f"obj_quat_{_oi}"], np.float64)[t])
        elif args.traj == "obj":
            p_np = OBJ[s][t][:3] + _shift + W
            q_np = _norm(OBJ[s][t][3:7])
        else:
            _Rh = quat_to_R(hq_np)
            _Ro = _Rh @ quat_to_R(_norm(PR[s][3:7])).T
            p_np = hp_np - _Ro @ PR[s][:3]
            q_np = R_to_q(_Ro)
        p_obj = torch.tensor(p_np, dtype=torch.float32, device=DEV).expand(cfg.scene.num_envs, 3)
        q_obj = torch.tensor(q_np, dtype=torch.float32, device=DEV).expand(cfg.scene.num_envs, 4)
        root = torch.cat([p_obj, q_obj, torch.zeros(cfg.scene.num_envs, 6, device=DEV)], 1)
        ART[s].write_root_state_to_sim(root)
        out[s] = (hp_np, p_np)
    E.scene.write_data_to_sim()
    if not args.kinematic:                        # 物理模式才推步; 目检模式纯摆位
        E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())
    return out


POSE0 = set_frame(f0)
print(f"[Sweep408Probe] 定格@f{f0} | 帧[{f0},{f1}] | traj={args.traj} kinematic={args.kinematic}", flush=True)
for s in ("right", "left"):
    hp, op = POSE0[s]
    print(f"[Sweep408Probe] {s}: 手 {np.round(hp - W, 3)} | 物 {np.round(op - W, 3)} | "
          f"手物距 {np.linalg.norm(hp - op)*100:.1f}cm (先验 {np.linalg.norm(PR[s][:3])*100:.1f}cm)", flush=True)

if args.render:
    import select
    print("[Sweep408Probe] 按 Enter 播放 (Ctrl-C 退出)", flush=True)
    while True:
        set_frame(f0); E.sim.render()
        if select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline(); break
    dt = 1.0 / max(args.fps, 1e-6)
    for t in range(f0, f1 + 1):
        set_frame(t); E.sim.render(); time.sleep(dt)
        if not app.is_running():
            break
else:
    for t in range(f0, min(f1, f0 + int(args.hold_s * args.fps)) + 1):
        set_frame(t)
    print("[Sweep408Probe] selftest 完成", flush=True)

try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
