"""L2 拼接查看器 v2 (2026-08-27 用户手动指导拼接):

初始状态 = L1 静止 GraspPose (双手 IK 合拢在瓶/杯上, 与 view_grasppose 同源摆位)。
物体轨迹 = replay_world.npz 原始位姿 (无conf无RTS), **逐物体换基**:
    T_i'(t) = T_scene_i(静置) ∘ inv(T_raw_i(窗起)) ∘ T_raw_i(t)
即每个物体的轨迹接在它自己的场景静置位上 (杯窗起 f7, 瓶窗起 f23)。
灰色折线 = 各物体全窗位置轨迹, 常显。按 Enter 物体沿灰线走 (双手保持 GraspPose 定格)。
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
p.add_argument("--lo_cup", type=int, default=7, help="杯运动窗起 (原始口径实测)")
p.add_argument("--lo_bot", type=int, default=23, help="瓶运动窗起")
p.add_argument("--hi", type=int, default=141)
p.add_argument("--fps", type=float, default=15.0)
p.add_argument("--source", choices=["rts", "raw"], default="rts",
               help="rts=RTS平滑轨迹+置信度着色(绿≥hi/黄/红<lo); raw=原始灰线")
p.add_argument("--conf_hi", type=float, default=70.0)
p.add_argument("--hand_align", choices=["first2grasp", "nearest"], default="first2grasp",
               help="first2grasp=窗首帧焊在GraspPose上,只借人手增量运动(2026-08-27终试); "
                    "nearest=旧法(轨迹随物体搬,过渡到最近帧)")
p.add_argument("--conf_lo", type=float, default=40.0)
p.add_argument("--selftest", type=int, default=0)
p.add_argument("--export", default="", help="导出参考npz(母带)路径; 空=不导")
p.add_argument("--record", default="", help="录制全链mp4路径; 空=不录 (需--enable_cameras)")
p.add_argument("--obj_shift", default="0,0", help="★L5-37 位置泛化: 物体静置位平移 dx,dy (m, env系, 瓶杯同移)。"
                                                  "approach 混入平移(站姿起0→抓握全量)后逐行ArmIK重解; interact/seam/retreat 跟着换基。")
p.add_argument("--approach_npz", default="",
               help="★治本(2026-09-03): 给该位置的**真 cuRobo 重规划**接近段 (build_motion "
                    "--obj_shift 产出 Approach_pour17_pos<k>.npz)。给了就直接加载它当接近段, "
                    "**跳过 approach 臂形变**(逐帧IK拟合是落手变差的根因); 撤离段仍走原带形变; "
                    "interact/物体换基照旧。空=旧法(形变原始 Approach_pour17.npz)。")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
# ★强制 RTX 材质管线 (照搬 recon record_scene.py:26): 不带 --enable_cameras 时交互视口
#   播放中不渲 UsdPreviewSurface 纹理(显黑), 与 pour17 无材质裸 mesh 表现不同。2026-09-05。
args.enable_cameras = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("view_raw_window")
app = AppLauncher(args).app

import select  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
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


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


# ================= 静止 GraspPose 摆位 (与 view_grasp_pose 逐段同源) =================
cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.prior_a, args.yaw_a, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()
try:                                            # U15 (2026-09-02): SAM3D 真实纹理
    from rl_rebuild.correction.texture_objects import apply_textures
    apply_textures(E, tex_dir=os.path.abspath("datasets/pour31/cache/textures"),
                   names=(("object", "bottle", "瓶"), ("aux", "cup", "杯")))
except Exception as _te:
    print(f"[U15纹理] 跳过 ({_te})", flush=True)
for _ in range(90):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())

# ================= ★L5-37 位置泛化: 物体静置位平移 =================
_dxy = [float(v) for v in args.obj_shift.split(",")]
SHIFT = np.array([_dxy[0], _dxy[1], 0.0], np.float64)
if np.linalg.norm(SHIFT) > 1e-9:
    # 物理搬移场景物体到 rest+SHIFT (让 _objs0 捕获/interact换基/渲染都用新位)
    for _ob in (E.object, E.aux):
        _st = _ob.data.root_state_w.clone()
        _st[:, :3] = _st[:, :3] + torch.tensor(SHIFT, dtype=torch.float32, device=E.device)
        _ob.write_root_pose_to_sim(_st[:, :7])
        _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))
    for _ in range(10):
        E.scene.write_data_to_sim(); E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())
    print(f"[pos] ★物体静置位平移 dx={SHIFT[0]:+.3f} dy={SHIFT[1]:+.3f} m (env系)", flush=True)

A_name = cfg.hand_side
B_name = "left" if A_name == "right" else "right"
# 抓握位/物体位随平移 (E._grasp_pos_w/obj_init_pos 是 env 初始化缓存的原位, 手动加 SHIFT)
A_gp = E._grasp_pos_w.cpu().numpy().astype(np.float64) + SHIFT
A_gq = E._grasp_quat_w.cpu().numpy().astype(np.float64)
A_obj = E.obj_init_pos.cpu().numpy().astype(np.float64) + SHIFT
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
E._resolve_joint_ids(force_side=B_name)
B_jids = list(E.arm_jids)
E._resolve_joint_ids(force_side=A_name)

sol = {}
for nm, side, gp, gq in (("右手->瓶", A_name, A_gp, A_gq),
                         ("左手->杯", B_name, B_gp, B_gq)):
    ik = ArmIK(side, anchor_link="arm_center", anchor_T=E._anchor_T)
    r = ik.solve(gp, quat_to_R(gq), iters=300)
    sol[side] = r["q"]
    print(f"  {nm}: IK {'✅' if r['ok'] else '❌'} 位置误差 {r['pos_err']*100:.2f}cm")

jn = list(E.hand.joint_names)
za = np.load(args.prior_a)
q = E.hand.data.default_joint_pos.clone()
q[:, A_jids] = torch.tensor(sol[A_name], dtype=torch.float32, device=E.device)
q[:, B_jids] = torch.tensor(sol[B_name], dtype=torch.float32, device=E.device)
for z_, side_ in ((za, A_name), (zb, B_name)):
    for n_, v_ in zip(GENERIC_JOINT_ORDER, np.asarray(z_["grasp"], np.float64)[7:29]):
        q[:, jn.index(n_.replace("right_", f"{side_}_"))] = float(v_)
E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
E.hand.write_data_to_sim()

# ---- ② 机器人可穿透 (2026-08-27 用户指令): 关掉手+臂全部碰撞体,
#      物体沿重建轨迹走时直接穿过机器人, 不发生物理打架 ----
import omni.usd
from pxr import UsdPhysics
_stage = omni.usd.get_context().get_stage()
_COLS = []
for _prim in _stage.Traverse():
    _pth = str(_prim.GetPath())
    if "/Robot" in _pth and _prim.HasAPI(UsdPhysics.CollisionAPI):
        _COLS.append(UsdPhysics.CollisionAPI(_prim).CreateCollisionEnabledAttr())

def set_collision(on):
    for _a in _COLS:
        _a.Set(bool(on))
    print(f"[拼接] 机器人碰撞: {'开' if on else '关'} ({len(_COLS)} 体)", flush=True)

set_collision(True)   # 定格/Approach 段真实物理

# ================= 物体轨迹: 原始位姿逐物体换基到场景静置位 =================
z = np.load("/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/pour/31/replay_world.npz", allow_pickle=True)
OP = np.asarray(z["obj_pose_all"], np.float64)  # ★pour31: recon obj0=杯/obj1=瓶 已合 tape 约定(obj_0=杯) -> 不swap(同pour17)
CONF = {0: None, 1: None}
if args.source == "rts":
    def _m2q(M):
        R = M[:3, :3]
        w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        if w < 1e-6:
            i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
            q = np.zeros(4)
            q[1 + i] = s / 4
            q[1 + j] = (R[j, i] + R[i, j]) / s
            q[1 + k] = (R[k, i] + R[i, k]) / s
            q[0] = (R[k, j] - R[j, k]) / s
            return q / np.linalg.norm(q)
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    for _oi in (0, 1):
        _zr = np.load(f"/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/pour/31/poseqa/"
                      f"rts_pour_31_object_{_oi}.npz", allow_pickle=True)
        _M = np.asarray(_zr["object_ob_in_world_smooth"], np.float64)
        _n = len(_M)
        _pq = np.zeros((_n, 7))
        for _ti in range(_n):
            _pq[_ti, :3] = _M[_ti, :3, 3]
            _pq[_ti, 3:7] = _m2q(_M[_ti])
        _tgt = _oi              # ★pour31 不swap: rts_object_0=杯->OP[0], object_1=瓶->OP[1]
        OP[_tgt, :_n] = _pq
        CONF[_tgt] = np.minimum(np.asarray(_zr["conf_pos"], np.float64),
                                np.asarray(_zr["conf_rot"], np.float64))
    print("[拼接] 轨迹源=RTS平滑 | conf=min(pos,rot) 档位: "
          f"绿≥{args.conf_hi:.0f} 黄 红<{args.conf_lo:.0f}")
hi = int(args.hi)
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)

_objs0 = []
for _ob in (E.object, E.aux):
    _objs0.append((_ob, _ob.data.root_state_w.clone()))

traj = {}                       # oi -> (T,7) 换基后世界位姿
OFF = {}                        # oi -> 平移偏移 (p_scene - p_raw@lo)
for oi, ob, lo_i, tag in ((1, E.object, int(args.lo_bot), "瓶"),
                          (0, E.aux, int(args.lo_cup), "杯")):
    st = ob.data.root_state_w[0].cpu().numpy().astype(np.float64)
    p_s, q_s = st[:3], st[3:7]
    p0, q0 = OP[oi, lo_i, :3], OP[oi, lo_i, 3:7] / np.linalg.norm(OP[oi, lo_i, 3:7])
    # ③ 换基定案(2026-08-27): 场景物体为对接抓姿转过 yaw, 全刚体换基会把这个
    #    yaw 转进轨迹走向(老毛病)。改: 位置**纯平移**对接(轨迹方向原样),
    #    旋转 = 世界系增量 dq(t)=q_raw(t)∘inv(q_raw(lo)) 叠到场景朝向上
    OFF[oi] = p_s - p0
    out = np.zeros((hi + 1, 7))
    for t in range(hi + 1):
        pt, qt = OP[oi, t, :3], OP[oi, t, 3:7] / np.linalg.norm(OP[oi, t, 3:7])
        out[t, :3] = p_s + (pt - p0)
        _dq = qmul(qt, qconj(q0))
        _qn = qmul(_dq, q_s)
        out[t, 3:7] = _qn / np.linalg.norm(_qn)
    # 桌面净空钳制 (carry4 先例同款, 2026-08-27 实测: 瓶穿桌3.3cm/32帧 杯2.7cm/68帧,
    # 集中在放回段 —— 重建桌高≠场景桌高): 逐帧网格最低点不足 桌面+2mm 者抬升, 9帧平滑
    import trimesh as _tm
    _e0 = clips.clip_entry(args.clip)
    _mp0 = _e0["mesh"] if oi == 1 else (_e0.get("secondary") or {}).get("mesh")
    _V0 = _tm.load(_mp0, force="mesh").vertices
    _V0 = _V0[np.random.RandomState(0).choice(len(_V0), 600, replace=False)]
    _TT0 = float(cfg.table_top_z)
    _lift = np.zeros(hi + 1)
    for _t3 in range(lo_i, hi + 1):
        _zmin = (_V0 @ quat_to_R(out[_t3, 3:7]).T)[:, 2].min() + out[_t3, 2]
        _lift[_t3] = max(0.0, _TT0 + 0.002 - _zmin)
    _lift = np.convolve(np.pad(_lift, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    out[:, 2] += _lift[:hi + 1]
    _ntrig = int(np.sum(_lift[lo_i:hi + 1] > 1e-4))
    print(f"[拼接] {tag} 桌面净空钳制: 触发 {_ntrig}/{hi - lo_i + 1} 帧 | "
          f"最大抬升 {_lift.max()*100:.1f}cm")
    traj[oi] = out
    print(f"[拼接] {tag}: 平移换基@f{lo_i} -> 场景静置位 (朝向=原始增量) | 窗 [{lo_i},{hi}]")
    # 轨迹线: raw=灰; rts=按逐帧置信度分段着色 (同档连续段各一条)
    if CONF[oi] is None:
        E._draw_curve(f"/World/RawTraj_{tag}", out[lo_i:, :3] + W,
                      (0.55, 0.55, 0.55), 0.003)
    else:
        _c = CONF[oi]
        def _tier(v):
            return 2 if v >= args.conf_hi else (1 if v >= args.conf_lo else 0)
        _COL = {2: (0.15, 0.85, 0.15), 1: (0.95, 0.85, 0.15), 0: (0.95, 0.15, 0.15)}
        _s = lo_i
        _seg = 0
        for _t2 in range(lo_i + 1, hi + 2):
            if _t2 > hi or _tier(_c[_t2]) != _tier(_c[_s]):
                _pts = out[_s:min(_t2, hi) + 1, :3] + W
                if len(_pts) >= 2:
                    E._draw_curve(f"/World/ConfTraj_{tag}_{_seg}", _pts,
                                  _COL[_tier(_c[_s])], 0.004)
                    _seg += 1
                _s = _t2
        _nlo = int(np.sum(_c[lo_i:hi+1] < args.conf_lo))
        _nhi = int(np.sum(_c[lo_i:hi+1] >= args.conf_hi))
        print(f"[拼接] {tag} 置信分布: 绿{_nhi} 黄{hi-lo_i+1-_nhi-_nlo} 红{_nlo} 帧, 分段{_seg}条")

lo_all = min(int(args.lo_bot), int(args.lo_cup))

# ================= 人手轨迹 (2026-08-27 用户指导): 蓝线 + 机械手跟随 =================
# 腕位姿 = retarget ref_qpos (重建世界) + 各自物体的同款平移换基 (保持手物相对位置);
# 过渡目标 = 轨迹上离 GraspPose 最近的一帧 t*, 30帧关节空间平滑过去, 之后逐帧IK跟随
HW = {}
for side, fn, oi, lo_s in ((A_name, "ref_qpos_right.npz", 1, int(args.lo_bot)),
                           (B_name, "ref_qpos_left.npz", 0, int(args.lo_cup))):
    zh = np.load(f"/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/pour/31/{fn}", allow_pickle=True)  # ★pour31 人手 ref_qpos(按手right=瓶/left=杯, 不swap)
    _wpr = np.asarray(zh["wrist_pos"], np.float64)[:hi + 1]
    _wqr = np.asarray(zh["wrist_quat_wxyz"], np.float64)[:hi + 1]
    _wqr = _wqr / np.linalg.norm(_wqr, axis=1, keepdims=True)
    if args.hand_align == "first2grasp":
        # 终试: 窗首帧位姿 := GraspPose 腕位姿; 之后 = 人手世界系增量叠加
        # (人比机器人小 => 借"运动"不借"位置", 起点尺寸差归零)
        gp_s0 = A_gp if side == A_name else B_gp
        gq_s0 = A_gq if side == A_name else B_gq
        p0h, q0h = _wpr[lo_s].copy(), _wqr[lo_s].copy()
        wp = np.zeros_like(_wpr)
        wq = np.zeros_like(_wqr)
        for _t6 in range(hi + 1):
            wp[_t6] = gp_s0 + (_wpr[_t6] - p0h)
            _dq6 = qmul(_wqr[_t6], qconj(q0h))
            _q6 = qmul(_dq6, gq_s0)
            wq[_t6] = _q6 / np.linalg.norm(_q6)
        print(f"[拼接] {side} 手轨迹: 窗首帧f{lo_s}已焊在GraspPose (增量模式)")
    else:
        wp = _wpr + OFF[oi]
        wq = _wqr
    # 人手净空钳制 v2 (2026-08-27 升级): 基准=FK 全连杆手最低点(抓姿指型),
    # 官方 hand_lowest_world 口径; 平滑用 carry4 技巧(缓坡但不低于必要抬升)
    from rl_rebuild.correction.ref_builders.replay_grasp import hand_lowest_world
    _gf = np.asarray((za if side == A_name else zb)["grasp"], np.float64)[7:29]
    _TTh = float(cfg.table_top_z) + 0.02
    _low = np.array([hand_lowest_world(wp[_t4], wq[_t4], side, _gf)
                     for _t4 in range(hi + 1)])
    _lifth = np.maximum(0.0, _TTh - _low)
    _ls = np.convolve(np.pad(_lifth, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    _ls = np.maximum(_ls[:hi + 1], _lifth)
    wp[:, 2] += _ls
    _nh = int(np.sum(_ls[lo_s:hi + 1] > 1e-4))
    print(f"[拼接] {side} 人手净空钳制v2(FK手最低点): 触发 {_nh}/{hi - lo_s + 1} 帧 "
          f"| 最大抬升 {_ls.max()*100:.1f}cm")
    gp_s = A_gp if side == A_name else B_gp
    if args.hand_align == "first2grasp":
        tstar = lo_s
        print(f"[拼接] {side} 人手轨迹: 窗[{lo_s},{hi}] | t*=窗首帧{lo_s} (焊接距离0)")
    else:
        d = np.linalg.norm(wp[lo_s:hi + 1] - gp_s, axis=1)
        tstar = lo_s + int(np.argmin(d))
        print(f"[拼接] {side} 人手轨迹: 窗[{lo_s},{hi}] | 最近帧 t*={tstar} "
              f"(距GraspPose {d.min()*100:.1f}cm)")
    E._draw_curve(f"/World/HandTraj_{side}", wp[lo_s:] + W, (0.25, 0.45, 1.0), 0.003)
    ikf = ArmIK(side, anchor_link="arm_center", anchor_T=E._anchor_T)
    fq = {}
    # ★interact IK 种子: 有真 cuRobo approach 时从**其末帧(我们的臂 IK 分支)**起,
    #   否则用静态 GraspPose fresh IK。前者保证 interact 与 approach 同臂分支(不同分支
    #   实测右臂差 1.4rad/81°, seam 会甩臂), seam1 混合幅度≈0。2026-09-05。
    if args.approach_npz:
        _zap_seed = np.load(args.approach_npz, allow_pickle=True)
        seed = np.asarray(_zap_seed[f"{side}_q"][-1], np.float64)
    else:
        seed = np.asarray(sol[side], np.float64)
    for tt in range(tstar, hi + 1):
        r = ikf.solve(wp[tt], quat_to_R(wq[tt] / np.linalg.norm(wq[tt])),
                      q0=seed, iters=150)
        seed = r["q"]
        fq[tt] = np.asarray(r["q"], np.float64)
    print(f"[拼接] {side} 跟随段 IK {len(fq)} 帧就绪")
    # 手臂离桌核验: 跟随全程各臂连杆原点最低 z (原点在杆内, 余量按 3cm 口径看)
    _amin, _abad = 1e9, 0
    for _t5, _q5 in fq.items():
        _qd5 = ikf._qdict(np.asarray(_q5, np.float64))
        for _L5 in ikf.u.parent_joint:
            if "_arm_l" in _L5:
                _z5 = ikf.u.link_pose(_L5, _qd5, ikf.base_T, ikf.anchor_link)[2, 3]
                if _z5 < _amin:
                    _amin = _z5
                if _z5 < float(cfg.table_top_z) + 0.03:
                    _abad += 1
                    break
    print(f"[拼接] {side} 臂离桌核验: 连杆原点最低 {(_amin - float(cfg.table_top_z))*100:+.1f}cm"
          f" | 低于桌面+3cm 的帧 {_abad}/{len(fq)}")
    HW[side] = {"tstar": tstar, "fq": fq, "lo": lo_s}

JIDS = {A_name: A_jids, B_name: B_jids}
# ---- 全链编舞 (2026-08-27 总装): Approach(物理) -> 缝 -> 交互(钳位) -> 缝 -> Retreat(物理)
_APPROACH_REAL = bool(args.approach_npz)
ZAP = np.load(args.approach_npz if _APPROACH_REAL else
              "tasks/Pour/31/A_Design/L1_Data/Motion_Planning/Approach_pour31.npz",
              allow_pickle=True)
if _APPROACH_REAL:
    print(f"[pos] ★接近段=真 cuRobo 重规划 {args.approach_npz} (跳过臂形变)", flush=True)
ZRT = np.load("tasks/Pour/31/A_Design/L1_Data/Motion_Planning/Retreat_approach_rev.npz",
              allow_pickle=True)
TA, TR = len(ZAP["right_q"]), len(ZRT["right_q"])
_finN = [str(n) for n in ZAP["fin_names"]]

# ================= ★L5-37: approach/retreat 臂关节按平移重解 (dp_make_variants 同款混入) =================
#   转移段(cuRobo)权重 0→1(站姿钉住), 成形段恒 1(全量到新抓握位); 逐行 ArmIK 重解; 手指行不动。
#   retreat = approach 倒放, 用倒序权重。IK 超差抛错(该位置不合法, 换偏移量)。
_ZQ = {"ZAP": ZAP, "ZRT": ZRT}
if _APPROACH_REAL:
    _ZQ = {"ZRT": ZRT}      # 接近段已是真 cuRobo, 不形变; 只形变撤离段
_ARM_SHIFTED = {"ZAP": {}, "ZRT": {}}
if np.linalg.norm(SHIFT) > 1e-9:
    from rl_rebuild.correction.kinematics import ArmIK as _ArmIK
    _ikp = {s: _ArmIK(s, anchor_link="arm_center", anchor_T=E._anchor_T)
            for s in ("right", "left")}
    for _znm, _z in _ZQ.items():
        _segl = [int(v) for v in _z["seg_lens"]]
        _Tn = len(_z["right_q"])
        _w = np.ones(_Tn)
        if _znm == "ZAP":     # approach: 转移段 0→1
            _w[:_segl[0]] = np.linspace(0.0, 1.0, _segl[0])
        else:                 # retreat = approach 倒放: 成形段(前)恒1, 转移段(后)1→0
            _w[-_segl[-1]:] = np.linspace(1.0, 0.0, _segl[-1])
        _maxerr = 0.0
        for _s in ("right", "left"):
            _q_prev = np.asarray(_z[f"{_s}_q"][0], np.float64)
            for _t in range(_Tn):
                _q0 = np.asarray(_z[f"{_s}_q"][_t], np.float64)
                _pw, _Rw = _ikp[_s].fk(_q0)
                _pt = _pw + _w[_t] * SHIFT          # 纯平移混入 (yaw 不动)
                _r = _ikp[_s].solve(_pt, _Rw, q0=_q_prev, iters=80)
                _maxerr = max(_maxerr, _r["pos_err"])
                assert _r["pos_err"] < 0.02, \
                    f"{_znm} {_s} 行{_t} IK 误差 {_r['pos_err']*100:.1f}cm — 该位置超工作区, 换 --obj_shift"
                _ARM_SHIFTED[_znm][(_s, _t)] = np.asarray(_r["q"], np.float64)
                _q_prev = _ARM_SHIFTED[_znm][(_s, _t)]
        print(f"[pos] {_znm} 臂重解 {_Tn}行×2臂 | 最大IK误差 {_maxerr*100:.2f}cm", flush=True)


def q_from_npz(z_, t_):
    _qq = E.hand.data.default_joint_pos.clone()
    _znm = "ZAP" if z_ is ZAP else ("ZRT" if z_ is ZRT else None)
    for s_, P_ in (("right", "R"), ("left", "L")):
        _use_shift = (np.linalg.norm(SHIFT) > 1e-9 and _znm is not None
                      and not (_APPROACH_REAL and _znm == "ZAP"))
        _aq = _ARM_SHIFTED[_znm][(s_, t_)] if _use_shift else z_[f"{s_}_q"][t_]
        for i_ in range(7):
            _qq[:, jn.index(f"{P_}_arm_j{i_+1}")] = float(_aq[i_])
        for n_, v_ in zip(_finN, z_[f"{s_}_f"][t_]):
            _qq[:, jn.index(n_.replace("right_", f"{s_}_"))] = float(v_)
    return _qq


def q_interact(g):
    _qq = q.clone()
    for side in (A_name, B_name):
        _qq[:, JIDS[side]] = torch.tensor(arm_at(side, g, None),
                                          dtype=torch.float32, device=E.device)
    return _qq


def write_q(_qq):
    E.hand.write_joint_state_to_sim(_qq, torch.zeros_like(_qq))
    E.hand.write_data_to_sim()


def rest_objs():
    for _ob, _st in _objs0:
        _ob.write_root_pose_to_sim(_st[:, :7])
        _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))


def arm_at(side, g, k_pre=None):
    hw = HW[side]
    if g < hw["tstar"]:
        return hw["fq"][hw["tstar"]]
    return hw["fq"][min(g, hi)]


def set_hands(g=None, k_pre=None):
    for side in (A_name, B_name):
        q[:, JIDS[side]] = torch.tensor(arm_at(side, g if g is not None else lo_all,
                                               k_pre), dtype=torch.float32,
                                        device=E.device)
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.write_data_to_sim()


def set_objs(t):
    for oi, ob in ((1, E.object), (0, E.aux)):
        row = traj[oi][t]
        st = torch.tensor(row, dtype=torch.float32, device=E.device).unsqueeze(0)
        ob.write_root_pose_to_sim(st)
        ob.write_root_velocity_to_sim(torch.zeros(1, 6, device=E.device))


def hold_hands():
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.write_data_to_sim()


dt = 1.0 / max(args.fps, 1.0)
BL1, BL2 = 25, 25     # 缝帧数(含合拢/松手, 在钳位区完成)


_ON_STEP = [None]


def step(render, sleep):
    E.scene.write_data_to_sim()
    E.sim.step(render=render)
    E.scene.update(E.sim.get_physics_dt())
    if _ON_STEP[0] is not None:
        _ON_STEP[0]()
    if sleep:
        time.sleep(dt)


def play_chain(render, abbr=False, on_step=None):
    """abbr=True: 各段抽帧冒烟 (自检)."""
    _ON_STEP[0] = on_step
    def rng(n):
        return range(0, n, max(1, n // 10) if abbr else 1)
    # -- Approach: 完整帧序, 碰撞开, 物体逐帧钳位 --
    #    (2026-08-27 定案: 与用户验收过的单独版 view_motion 逐位同体制 ——
    #     单独版"不打转"的机制就是定身钳位, 自由物理与铁手回放天生互斥)
    A_END = TA
    set_collision(True)
    for t_ in rng(A_END):
        write_q(q_from_npz(ZAP, t_))
        rest_objs()
        step(render, not abbr)
    # -- 缝1: 关碰撞, 物体开始钳位, 手关节平滑到交互起始位形 --
    set_collision(False)
    _qa = q_from_npz(ZAP, A_END - 1)[0].cpu().numpy()
    _qb = q_interact(lo_all)[0].cpu().numpy()
    print(f"[seam1] approach末 vs interact始 关节最大差 {np.abs(_qa - _qb).max():.4f} rad "
          f"(修前~1.4=81°; 种子改后应≈0)", flush=True)
    for k_ in rng(BL1):
        s_ = k_ / max(BL1 - 1, 1)
        s_ = s_ * s_ * (3 - 2 * s_)
        _qm = torch.tensor((1 - s_) * _qa + s_ * _qb, dtype=torch.float32,
                           device=E.device).unsqueeze(0)
        set_objs(lo_all)
        write_q(_qm)
        step(render, not abbr)
    # -- 交互窗: 免碰撞 + 轨迹钳位 (保持现状) --
    for t_ in (range(lo_all, hi + 1, max(1, (hi - lo_all) // 10)) if abbr
               else range(lo_all, hi + 1)):
        set_objs(t_)
        write_q(q_interact(t_))
        step(render, not abbr)
    # -- 缝2: 物体归位(仍钳), 手平滑到 Retreat 首帧 --
    _qc = q_interact(hi)[0].cpu().numpy()
    R_START = 0                       # Retreat 完整帧序
    _qd = q_from_npz(ZRT, R_START)[0].cpu().numpy()
    for k_ in rng(BL2):
        s_ = k_ / max(BL2 - 1, 1)
        s_ = s_ * s_ * (3 - 2 * s_)
        _qm = torch.tensor((1 - s_) * _qc + s_ * _qd, dtype=torch.float32,
                           device=E.device).unsqueeze(0)
        rest_objs()
        write_q(_qm)
        step(render, not abbr)
    # -- Retreat: 完整帧序, 碰撞开, 物体逐帧钳位 (同单独版体制) --
    set_collision(True)
    for t_ in (range(R_START, TR, max(1, (TR - R_START) // 10)) if abbr
               else range(R_START, TR)):
        write_q(q_from_npz(ZRT, t_))
        rest_objs()
        step(render, not abbr)


# ================= 导出母带 (2026-08-27 批): source + frame_of_row + conf =================
if args.export:
    def _pull(qv):
        r = {}
        for s_, P_ in (("right", "R"), ("left", "L")):
            r[f"{s_}_q"] = [float(qv[jn.index(f"{P_}_arm_j{i_}")]) for i_ in range(1, 8)]
            r[f"{s_}_f"] = [float(qv[jn.index(n_.replace("right_", f"{s_}_"))])
                            for n_ in _finN]
        return r
    _rest = {oi: _objs0[i][1][0, :7].cpu().numpy().astype(np.float64)
             for i, oi in ((0, 1), (1, 0))}   # _objs0=[object(瓶=1), aux(杯=0)]
    _zrf = {oi: np.load(f"/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/pour/31/poseqa/"
                        f"rts_pour_31_object_{oi}.npz", allow_pickle=True)  # ★pour31 不swap: tape obj_0=杯<-rts_object_0, obj_1=瓶<-rts_object_1
            for oi in (0, 1)}
    rowsQ, rowsO, src, for_, segn, segl = [], [], [], [], [], []
    def _phase(qvs, objs, s_, frames, name):
        for k_, qv in enumerate(qvs):
            rowsQ.append(_pull(qv))
            rowsO.append(objs[k_])
            src.append(s_)
            for_.append(frames[k_])
        segn.append(name)
        segl.append(len(qvs))
    _restO = {0: _rest[0], 1: _rest[1]}
    # A
    _phase([q_from_npz(ZAP, t_)[0].cpu().numpy() for t_ in range(TA)],
           [dict(_restO) for _ in range(TA)], 0, [-1] * TA, "approach")
    # 缝1
    _qa0 = q_from_npz(ZAP, TA - 1)[0].cpu().numpy()
    _qb0 = q_interact(lo_all)[0].cpu().numpy()
    BL1_, BL2_ = 25, 25
    _sm = lambda k, n: (lambda x: x * x * (3 - 2 * x))(k / max(n - 1, 1))
    _phase([(1 - _sm(k, BL1_)) * _qa0 + _sm(k, BL1_) * _qb0 for k in range(BL1_)],
           [dict(_restO) for _ in range(BL1_)], 0, [-1] * BL1_, "seam1")
    # 交互
    _phase([q_interact(g)[0].cpu().numpy() for g in range(lo_all, hi + 1)],
           [{0: traj[0][g], 1: traj[1][g]} for g in range(lo_all, hi + 1)],
           1, list(range(lo_all, hi + 1)), "interact")
    # 缝2
    _qc0 = q_interact(hi)[0].cpu().numpy()
    _qd0 = q_from_npz(ZRT, 0)[0].cpu().numpy()
    _phase([(1 - _sm(k, BL2_)) * _qc0 + _sm(k, BL2_) * _qd0 for k in range(BL2_)],
           [dict(_restO) for _ in range(BL2_)], 0, [-1] * BL2_, "seam2")
    # R
    _phase([q_from_npz(ZRT, t_)[0].cpu().numpy() for t_ in range(TR)],
           [dict(_restO) for _ in range(TR)], 0, [-1] * TR, "retreat")
    T_all = len(rowsQ)
    out = {k: np.array([r[k] for r in rowsQ], np.float32)
           for k in ("right_q", "left_q", "right_f", "left_f")}
    for oi in (0, 1):
        out[f"obj_pos_{oi}"] = np.array([np.asarray(o[oi])[:3] for o in rowsO], np.float32)
        out[f"obj_quat_{oi}"] = np.array([np.asarray(o[oi])[3:7] for o in rowsO], np.float32)
        _cp = np.asarray(_zrf[oi]["conf_pos"], np.float64)
        _cr = np.asarray(_zrf[oi]["conf_rot"], np.float64)
        out[f"conf_pos_{oi}"] = np.array([_cp[f_] if f_ >= 0 else np.nan for f_ in for_],
                                         np.float32)
        out[f"conf_rot_{oi}"] = np.array([_cr[f_] if f_ >= 0 else np.nan for f_ in for_],
                                         np.float32)
    out["source"] = np.array(src, np.int8)              # 0=machine 1=human
    out["frame_of_row"] = np.array(for_, np.int32)      # -1=机器行
    out["seg_names"] = np.array(segn, dtype=object)
    out["seg_lens"] = np.array(segl, np.int64)
    out["fin_names"] = np.array(GENERIC_JOINT_ORDER, dtype=object)
    out["meta"] = (f"pour31_reference_v1: approach(真cuRobo Approach_pour31)+seam1(25)+interact(手=首帧焊GraspPose增量,"
                   "物=RTS+换基+双净空钳制)+seam2(25)+retreat(Approach倒放); obj_0=杯 obj_1=瓶(recon同向,不swap); "
                   f"交互窗[{int(args.lo_bot)}瓶/{int(args.lo_cup)}杯/{int(args.hi)}]; "
                   f"瓶抓握=10_Power_Disk 0_4 yaw={args.yaw_a}(B角=对齐人手接近侧203°, 视频yaw326); "
                   f"杯=1_Large_Diameter 3_32 yaw={args.yaw_b}(B角, 视频yaw179); 2026-09-06 用户目检定")
    np.savez(args.export, **out)
    print(f"[母带] {args.export} | {T_all} 行 | 段 {list(zip(segn, segl))} | "
          f"human行 {int(np.sum(out['source']==1))}")

# ================= 录像 =================
if args.record:
    import imageio
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom
    _st2 = omni.usd.get_context().get_stage()
    _camp = UsdGeom.Camera.Define(_st2, "/World/RecCam")
    _camp.CreateFocalLengthAttr().Set(16.0)
    _m = Gf.Matrix4d()
    _m.SetLookAt(Gf.Vec3d(0.85, -1.15, 1.60), Gf.Vec3d(-0.15, 0.10, 0.95),
                 Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(_camp).AddTransformOp().Set(_m.GetInverse())
    _rp = rep.create.render_product("/World/RecCam", (1280, 720))
    _annot = rep.AnnotatorRegistry.get_annotator("rgb")
    _annot.attach(_rp)
    _frames = []
    def _grab():
        _d = _annot.get_data()
        if _d is not None and getattr(_d, "size", 0):
            _frames.append(np.asarray(_d)[..., :3].astype(np.uint8))
    set_collision(True)
    rest_objs()
    write_q(q_from_npz(ZAP, 0))
    for _ in range(5):
        E.scene.write_data_to_sim(); E.sim.step(render=True)
        E.scene.update(E.sim.get_physics_dt())
    play_chain(render=True, abbr=False, on_step=_grab)
    imageio.mimsave(args.record, _frames, fps=15)
    print(f"[录像] {args.record} | {len(_frames)} 帧 @15fps")

if args.export or args.record:
    try:
        _slot.release()
    except Exception:
        pass
    app.close()
    raise SystemExit(0)

try:
    while True:
        set_collision(True)
        rest_objs()
        write_q(q_from_npz(ZAP, 0))
        if args.selftest:
            play_chain(render=False, abbr=True)
            print("[拼接] 全链 selftest 完成")
            break
        print("[拼接] 定格@InitialPose | 全链: Approach(物理)->交互(钳位)->Retreat(物理)"
              " —— 按 Enter 播放", flush=True)
        while True:
            rest_objs()
            write_q(q_from_npz(ZAP, 0))
            E.sim.step(render=True)
            _r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if _r:
                sys.stdin.readline()
                break
        play_chain(render=True)
        print("[拼接] 全链放完, 归位 (Ctrl-C 退出)", flush=True)
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
