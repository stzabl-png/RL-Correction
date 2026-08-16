"""Pour17 参考轨迹 GUI 回放: 站姿 -> 人手轨迹起点 -> 各自走到交互开始帧停下。

**双手各走各的**: 右臂跟瓶(`Pour17_bottle`)的参考, 左臂跟杯(`Pour17_cup`)的参考,
两者都从 `cfg.dexmate_joints` 的对称待命站姿出发, 关节空间 smoothstep 插值 K 帧
到各自轨迹的第 0 帧 (与训练里 `--stance_prefix K` 同一套做法), 然后按参考逐帧走到
各自的交互开始帧 (gs) **停住**。回车放下一次。

⚠ 这是**运动学回放**, 不跑物理: 直接写关节状态, 手不会被物体挡住, 也不会有接触力。
   看的是"参考轨迹把手带到哪儿", 不是"抓不抓得住"。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.view_pour_ref \
        --stance_prefix 60

    --fps N          回放速度 (默认 20 = 参考本身的频率)
    (默认) 走到 gs 后停住等回车; --loop 则自动循环
    --only right|left  只放一只手 (另一只留在站姿)
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle", help="场景来源 clip (主物体+aux 都在)")
p.add_argument("--second_clip", default="Pour17_cup", help="第二只手的参考来源 clip")
p.add_argument("--mode", default="ref", choices=("ref", "prior", "retract"),
               help="ref=跟人手轨迹走到交互开始帧; prior=从站姿直接摆到各自 GraspPose; "
                    "retract=站姿->退避族最远点->沿 radial 轴推进到 GraspPose")
p.add_argument("--dmax_k", type=float, default=1.5,
               help="retract: 退避距离 = 这个倍数 × (GraspPose 手座->物体中心距离), 自标定")
p.add_argument("--napp", type=int, default=60, help="retract: 接近段帧数")
p.add_argument("--grasp_prior", default="tasks/pregrasp/priors/Pour17_bottle.npz",
               help="mode=prior: 主手(瓶)的 GraspPose")
p.add_argument("--second_grasp_prior", default="tasks/pregrasp/priors/Pour17_cup.npz",
               help="mode=prior: 副手(杯)的 GraspPose, 摆到 aux 物体上")
p.add_argument("--prior_yaw", type=float, default=19.5, help="mode=prior: 主物体 yaw (度)")
p.add_argument("--stance_prefix", type=int, default=60)
p.add_argument("--fps", type=float, default=20.0)
p.add_argument("--only", default="both", choices=("both", "right", "left"))
p.add_argument("--loop", action="store_true", help="不等回车, 自动循环")
p.add_argument("--measure", action="store_true",
               help="不开 GUI, 逐帧量 臂/手->桌面 与 手->物体 的最小间隙, 打表后退出")
p.add_argument("--selftest", type=int, default=0,
               help=">0: 只放这么多帧然后退出 (headless 自检用, 不等回车)")
p.add_argument("--eye", default="0.95,-1.15,1.55")
p.add_argument("--lookat", default="0.10,-0.20,0.92")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("view_pour_ref")
app = AppLauncher(args).app

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import load_replay_grasp  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

# ---------------------------------------------------------------- 场景
cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
cfg.closure_init_max = 0.0
# ⚠ 不开 cfg.approach / cfg.stance_prefix_frames: 那条路径 assert 必须配 GraspPose
#   prior, 而一配 prior, 物体就改按 Dexonomy 规范姿态摆 —— 那就不是"参考轨迹的摆放"了。
#   本工具要看的正是 ref_builder 摆放(相机锚定 + 物体听手), 所以站姿前缀**两只手都由
#   本脚本自己拼**, 用与 env 里 stance_prefix 完全相同的关节空间 smoothstep。
cfg.approach = False
cfg.stance_prefix_frames = 0
if args.mode in ("prior", "retract"):
    # ⚠ 接了 prior, **主物体**就改按 Dexonomy 规范姿态 + 钉死 yaw 摆 —— 这正是
    #   GraspPose 自己定义的那套摆放, 要看抓姿几何就该用它 (ref 模式则相反, 用
    #   ref_builder 的相机锚定+物体听手)。两种模式看到的物体位置本来就不同。
    cfg.grasp_prior_npz = args.grasp_prior
    if args.prior_yaw >= 0:
        cfg.prior_yaw_deg = float(args.prior_yaw)
cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                       lookat=tuple(float(v) for v in args.lookat.split(",")),
                       origin_type="world", resolution=(1600, 900))
E = GraspTaskEnv(cfg)
E.reset()
dev = E.device
K = int(args.stance_prefix)

# ---------------------------------------------------------------- 主手 (场景 clip 的交互手)
prim = cfg.hand_side                                   # "right" (Pour17_bottle)
q_prim = E.q_ref.clone()                               # (L,7) 原始参考 (无前缀)
gs_prim = int(E.grasp_start) + K                       # 前缀把 gs 顺延 K 帧
fin_prim = E.ref_finger.clone()                        # (L0,22) USD 序, 无前缀
L0_prim = fin_prim.shape[0]
prim_jids, prim_hids = E.arm_jids, E.hand_jids


def _prefix(q, jids):
    """站姿 -> 轨迹第 0 帧, 关节空间 smoothstep K 帧 (与 env.stance_prefix 同式)。"""
    if K <= 0:
        return q
    st = E.hand.data.default_joint_pos[0, jids].to(dev)
    u = torch.linspace(0.0, 1.0, K + 1, device=dev)[:-1]
    u = u * u * (3.0 - 2.0 * u)
    return torch.cat([st + u.unsqueeze(1) * (q[0] - st), q], 0)


q_prim = _prefix(q_prim, prim_jids)


# ---------------------------------------------------------------- mode=prior 的目标位形
if args.mode in ("prior", "retract"):
    # 主手: 臂 = 抓握腕位的 IK 解, 指 = Dexonomy 抓握构型 (env 加载 prior 时已解好)
    # retract 模式也要这些量: 它的推进终点就是 GraspPose
    q_goal_prim = E._prior_q_grasp.clone()               # (7,)
    f_goal_prim = E.q_close.clone()                      # (22,) USD 序
    _gp_p = E._grasp_pos_w.cpu().numpy().astype(np.float64)     # 手座抓握位 (env 局部)
    _gq_p = E._grasp_quat_w.cpu().numpy().astype(np.float64)
    _obj_p = E.obj_init_pos.cpu().numpy().astype(np.float64)

    def _qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                         w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])

# ---------------------------------------------------------------- 第二只手
sec = "left" if prim == "right" else "right"
q_sec = fin_sec = None
gs_sec = 0
if args.only == "both" and args.mode in ("prior", "retract"):
    # 副手的 GraspPose 摆到 **aux 物体**(杯)上: prior 里的抓姿是**物体系**的,
    # 用 aux 在世界里的实际位姿把它变换过去, 再解 IK。
    assert getattr(E, "aux", None) is not None, "本 clip 没有 aux 物体, 副手无处可摆"
    jn = list(E.hand.joint_names)
    P2 = "R" if sec == "right" else "L"
    sec_hnames = [n for n in jn if n.startswith(f"{sec}_")]
    sec_hids = torch.tensor([jn.index(n) for n in sec_hnames], dtype=torch.long, device=dev)
    sec_jids = torch.tensor([jn.index(f"{P2}_arm_j{i}") for i in range(1, 8)],
                            dtype=torch.long, device=dev)
    _W0 = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
    z2 = np.load(args.second_grasp_prior)
    g2 = np.asarray(z2["grasp"], np.float64).copy()      # (29,) = pos3 + quat4 + 22 关节
    # 垫↔接触零位校准: 与 env 同一套决策 —— **按 clip 注册表记死**, 不在这里另立判据
    # (env 里那条的来龙去脉见 DESIGN_LOOP §2.24)。
    _e2 = clips.clip_entry(args.second_clip)
    if bool(_e2.get("pad_contact_calib", True)):
        from rl_rebuild.correction.ref_builders.replay_grasp import (
            GENERIC_JOINT_ORDER as _GJO, _urdf as _urdf_f)
        from tasks.pregrasp.env import FINGERS as _FG
        _u = _urdf_f()
        _qd = {n.replace("right_", f"{sec}_"): float(v) for n, v in zip(_GJO, g2[7:29])}
        _Th = np.eye(4)
        _Th[:3, :3], _Th[:3, 3] = quat_to_R(g2[3:7]), g2[:3]
        _pads = np.stack([_u.link_pose(f"{sec}_{f}_elastomer", _qd, _Th,
                                       f"{sec}_hand_C_MC")[:3, 3] for f in _FG])
        _cts = np.asarray(z2["contact_pos"], np.float64)
        _d2 = np.linalg.norm(_pads[:, None, :] - _cts[None], axis=-1)
        _own = _d2.argmin(axis=0)
        _act = np.array([bool(((_own == i) & (_d2[i] < 0.05)).any()) for i in range(5)])
        _sh = (_cts[_d2.argmin(axis=1)][_act] - _pads[_act]).mean(axis=0)
        if np.linalg.norm(_sh) > 0.008:
            g2[:3] += _sh
            print(f"[{sec}] 垫↔接触零位校准: 腕位平移 {np.round(_sh*100,2).tolist()}cm "
                  f"(|Δ|={np.linalg.norm(_sh)*100:.2f}cm) | 参与指 {int(_act.sum())}/5")
    # 物体系 -> 世界 (aux 的实际摆放)
    _ap = E.aux.data.root_pos_w[0].cpu().numpy().astype(np.float64) - _W0
    _aq = E.aux.data.root_quat_w[0].cpu().numpy().astype(np.float64)
    _wp = quat_to_R(_aq) @ g2[:3] + _ap
    _wq = _qmul(_aq, g2[3:7])
    ik2 = ArmIK(sec, anchor_link="arm_center", anchor_T=E._anchor_T)
    _r = ik2.solve(_wp, quat_to_R(_wq), q0=E.hand.data.default_joint_pos[0, sec_jids]
                   .cpu().numpy().astype(np.float64), iters=200)
    print(f"[{sec}] GraspPose IK: ok={_r['ok']} 位置误差 {_r['pos_err']*100:.2f}cm"
          + ("" if _r["ok"] else "   ⚠ 够不到 —— 这个抓姿在当前摆放下不可达"))
    q_goal_sec = torch.tensor(_r["q"], dtype=torch.float32, device=dev)
    _gp_s, _gq_s = _wp.copy(), _wq.copy()
    _obj_s = _ap.copy()
    _ik_s = ik2
    from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER as _G
    gen2 = [n.replace("right_", f"{sec}_") for n in _G]
    perm2 = [gen2.index(n) for n in sec_hnames]   # GENERIC -> USD 序 (漏了整手串位)
    f_goal_sec = torch.tensor(g2[7:29][perm2], dtype=torch.float32, device=dev)
elif args.only == "both":
    e2 = clips.clip_entry(args.second_clip)
    du2 = load_replay_grasp(
        e2["npz"], e2["mesh"], usd_path=e2.get("usd", ""), clip_id=args.second_clip,
        hand=sec, semantics=e2.get("semantics"),
        scene_layout_json=e2.get("scene_layout_json"),
        affordance_npz=e2.get("affordance"))
    r2 = du2.ref            # load_replay_grasp 返回 DataUnit, 参考在 .ref 里
    # 与主手同一个 arm_center 锚 (env 建场景时实测出来的), 保证两臂在同一个基座系里
    ik2 = ArmIK(sec, anchor_link="arm_center", anchor_T=E._anchor_T)
    import rl_rebuild.correction.frames as F
    J2 = r2.mano_joints.astype(np.float64)
    sols = ik2.solve_traj(J2[:, 0], F.sharpa_base_quat_from_joints(J2, sec))
    q2 = np.stack([s["q"] for s in sols])
    ok2 = np.array([s["ok"] for s in sols])
    bad = np.flatnonzero(~ok2 | ~np.isfinite(q2).all(1))
    if len(bad):                                        # 解不出的帧用最近可达帧顶上
        good = np.flatnonzero(ok2 & np.isfinite(q2).all(1))
        assert len(good), f"{sec} 手整条 clip 都解不出 IK"
        q2[bad] = q2[good[np.abs(good[None] - bad[:, None]).argmin(1)]]
    print(f"[{sec}] IK 全程可达 {ok2.mean()*100:.1f}% | 顶替 {len(bad)} 帧 | "
          f"参考 {len(q2)} 帧")
    q_sec = torch.tensor(q2, dtype=torch.float32, device=dev)
    # 手指 + 关节 id: **与 env 的 _resolve_joint_ids/_generic_perm 完全同一套做法**
    # (手指名前缀 = 手侧; 臂关节名 = R_arm_j1..7 / L_arm_j1..7)。
    # 漏了这张 perm 就是整手串位 —— 2026-07-30 踩过, B 组整条 run 作废。
    jn = list(E.hand.joint_names)
    P2 = "R" if sec == "right" else "L"
    sec_hnames = [n for n in jn if n.startswith(f"{sec}_")]
    sec_hids = torch.tensor([jn.index(n) for n in sec_hnames],
                            dtype=torch.long, device=dev)
    gen2 = [n.replace("right_", f"{sec}_") for n in r2.finger_names] \
        if not r2.finger_names[0].startswith(sec) else list(r2.finger_names)
    perm2 = [gen2.index(n) for n in sec_hnames]
    fin_sec = torch.tensor(r2.human_finger[:, perm2], dtype=torch.float32, device=dev)
    sec_arm = [f"{P2}_arm_j{i}" for i in range(1, 8)]
    miss = [n for n in sec_arm if n not in jn]
    assert not miss, f"关节表里找不到 {miss}"
    sec_jids = torch.tensor([jn.index(n) for n in sec_arm], dtype=torch.long, device=dev)
    gs_sec = int(r2.interaction_seg[0]) + K   # env 也是从 interaction_seg[0] 取 gs

# ------------------------------------------------- retract: 退避族 (radial 轴)
if args.mode == "retract":
    # û = normalize(GraspPose 手座 - 物体中心)。沿它退 = 顺着"物体->手"的连线往外走,
    # 于是"离物体的距离"与 d 严格 1:1 ⟹ 退得越远越安全是**构造保证**, 不靠 RL 学。
    # (另两个候选轴实测都不成立: human 轴逐 clip 差 80° 且瓶子那条是往下走的;
    #  palm 轴退 6cm 手-物间隙原地不动。见 diag_retract.py)
    from rl_rebuild.correction.kinematics import ArmIK as _AIK

    def _family(gp0, gq0, obj0, jids, ik, seed, tag):
        u = gp0 - obj0
        u = u / np.linalg.norm(u)
        dg = float(np.linalg.norm(gp0 - obj0))
        D = args.dmax_k * dg
        R0 = quat_to_R(gq0)
        qs, qw, nbad = [], seed.copy(), 0
        for i in range(args.napp + 1):
            d = D * (1.0 - i / max(args.napp, 1))          # 从最远推进到 d=0
            r = ik.solve(gp0 + d * u, R0, q0=qw, iters=200)
            if r["ok"]:
                qw = r["q"]
            else:
                nbad += 1
            qs.append(qw.copy())
        print(f"[retract:{tag}] 手座->物体中心 d_g={dg*100:.1f}cm | 退避 D={D*100:.1f}cm "
              f"({args.dmax_k}×d_g) | 仰角 {np.degrees(np.arcsin(u[2])):+.1f}° | "
              f"IK 失败 {nbad}/{args.napp+1} 帧")
        return torch.tensor(np.stack(qs), dtype=torch.float32, device=dev)

    _ikp = _AIK(cfg.hand_side, anchor_link="arm_center", anchor_T=E._anchor_T)
    app_p = _family(_gp_p, _gq_p, _obj_p, prim_jids, _ikp,
                    E._prior_q_grasp.cpu().numpy().astype(np.float64), prim)
    app_s = (_family(_gp_s, _gq_s, _obj_s, None, _ik_s,
                     q_goal_sec.cpu().numpy().astype(np.float64), sec)
             if q_goal_sec is not None else None)

# ------------------------------------------------- 统一成逐帧数组 (两种模式共用 put)
def _ramp(cur, goal, n):
    """站姿 -> 目标, 关节空间 smoothstep n 帧, 再补一帧终点。"""
    u = torch.linspace(0.0, 1.0, n + 1, device=dev)
    u = u * u * (3.0 - 2.0 * u)
    return cur + u.unsqueeze(1) * (goal - cur)


q_stance = E.hand.data.default_joint_pos[0].clone()
if args.mode == "retract":
    # ① 站姿 -> 退避族最远点 (K 帧, 关节空间 smoothstep, 与 stance_prefix 同款)
    # ② 沿 radial 轴推进 D -> 0 (napp 帧)
    # 手指: 接近段张开, 最后 25% 才合到 GraspPose 手型 (真任务就是这么做的)
    def _fin_ramp(open_q, close_q, n):
        w = torch.linspace(0.0, 1.0, n + 1, device=dev)
        w = ((w - 0.75) / 0.25).clamp(0.0, 1.0)          # 前 75% 全张开
        w = w * w * (3.0 - 2.0 * w)
        return open_q + w.unsqueeze(1) * (close_q - open_q)

    traj_p_arm = torch.cat([_ramp(q_stance[prim_jids], app_p[0], K)[:K], app_p], 0)
    traj_p_fin = torch.cat([_ramp(q_stance[prim_hids], E.q_open, K)[:K],
                            _fin_ramp(E.q_open, f_goal_prim, args.napp)], 0)
    gs_prim = K + args.napp
    if app_s is not None:
        traj_s_arm = torch.cat([_ramp(q_stance[sec_jids], app_s[0], K)[:K], app_s], 0)
        _open_s = q_stance[sec_hids] * 0.0 + E.q_open    # 副手同一套张开模板 (USD 序一致)
        traj_s_fin = torch.cat([_ramp(q_stance[sec_hids], _open_s, K)[:K],
                                _fin_ramp(_open_s, f_goal_sec, args.napp)], 0)
        gs_sec = K + args.napp
    else:
        traj_s_arm = traj_s_fin = None
    tag_p = tag_s = "GraspPose(沿 radial 推进)"
elif args.mode == "prior":
    HOLD = max(K, 1)
    traj_p_arm = _ramp(q_stance[prim_jids], q_goal_prim, HOLD)
    traj_p_fin = _ramp(q_stance[prim_hids], f_goal_prim, HOLD)
    gs_prim = HOLD
    if q_sec is not None or args.only == "both":
        traj_s_arm = _ramp(q_stance[sec_jids], q_goal_sec, HOLD)
        traj_s_fin = _ramp(q_stance[sec_hids], f_goal_sec, HOLD)
        gs_sec = HOLD
    else:
        traj_s_arm = traj_s_fin = None
    tag_p, tag_s = "GraspPose", "GraspPose"
else:
    # ref 模式: 臂已带站姿前缀; 手指参考没有前缀, 前 K 帧从站姿手型插到参考第 0 帧
    traj_p_arm = q_prim
    traj_p_fin = torch.cat([_ramp(q_stance[prim_hids], fin_prim[0], K)[:K], fin_prim], 0)
    if q_sec is not None:
        q_sec = _prefix(q_sec, sec_jids)
        traj_s_arm = q_sec
        traj_s_fin = torch.cat([_ramp(q_stance[sec_hids], fin_sec[0], K)[:K], fin_sec], 0)
    else:
        traj_s_arm = traj_s_fin = None
    tag_p, tag_s = "交互开始帧", "交互开始帧"

print(f"\n{'='*70}")
print(f"[回放] 模式={args.mode} | 主手={prim} -> {tag_p} 第 {gs_prim} 帧停")
if traj_s_arm is not None:
    print(f"[回放] 副手={sec} -> {tag_s} 第 {gs_sec} 帧停")
print(f"[回放] 共放 {max(gs_prim, gs_sec)} 帧 @ {args.fps}fps ≈ "
      f"{max(gs_prim, gs_sec)/args.fps:.1f}s;  运动学回放, 不跑物理")
print(f"{'='*70}\n")


def put(t: int) -> None:
    """把两只手摆到第 t 帧 (各自 clamp 在自己的终点上)。"""
    q = E.hand.data.joint_pos[0].clone()
    tp = min(t, gs_prim)
    q[prim_jids] = traj_p_arm[min(tp, len(traj_p_arm) - 1)]
    q[prim_hids] = traj_p_fin[min(tp, len(traj_p_fin) - 1)]
    if traj_s_arm is not None:
        ts = min(t, gs_sec)
        q[sec_jids] = traj_s_arm[min(ts, len(traj_s_arm) - 1)]
        q[sec_hids] = traj_s_fin[min(ts, len(traj_s_fin) - 1)]
    E.hand.write_joint_state_to_sim(q.unsqueeze(0), torch.zeros_like(q).unsqueeze(0))
    # ⚠ 物体每帧钉回初始位: sim.step 会跑物理, 而这里的手是**瞬移**上去的 ——
    #   不钉的话去穿透冲量会把物体踢飞(view_prior 里踩过同一个坑)。
    E.object.write_root_pose_to_sim(_obj_pose)
    E.object.write_root_velocity_to_sim(_obj_zero)
    if _aux is not None:
        _aux.write_root_pose_to_sim(_aux_pose)
        _aux.write_root_velocity_to_sim(_obj_zero)
    E.sim.step(render=True)
    E.scene.update(0.0)


_W = E.scene.env_origins[0]
_obj_pose = torch.cat([E.obj_init_pos + _W, E.obj_init_quat]).unsqueeze(0)
_obj_zero = torch.zeros(1, 6, device=dev)
_aux = getattr(E, "aux", None)
_aux_pose = (torch.cat([_aux.data.root_pos_w[0], _aux.data.root_quat_w[0]])
             .unsqueeze(0).clone() if _aux is not None else None)

T = max(gs_prim, gs_sec)
dt = 1.0 / max(args.fps, 1e-3)
if args.measure:
    # ---- 逐帧间隙普查: 这条接近路径在"抓稳之前"有没有撞桌/撞物体 ----
    # 口径说明(重要, 别混):
    #   桌面间隙 = **臂连杆外壳采样点**(48~96 点/节, 从 URDF 碰撞网格采的)的最低 z
    #              减桌面高度 —— 这是真实外壳口径, 不是连杆原点。手部连杆另算(原点口径)。
    #   物体间隙 = 手部所有连杆**原点**到物体表面采样点的最近距离。原点口径**偏乐观**
    #              (连杆本身有半径), 所以这里同时给指垫口径(_pad_dist_normal, 准确)。
    import os as _os
    _sp = np.load(_os.path.join(_os.path.dirname(__file__), "arm_shell_points.npz"))
    bn = list(E.hand.body_names)
    sh_b, sh_p = [], []
    for k in _sp.files:                      # **两条臂都要**, env 只装了交互臂那半
        if k in bn:
            sh_b.append(bn.index(k))
            sh_p.append(np.asarray(_sp[k], np.float32))
    _P = max(len(x) for x in sh_p)
    sh_pts = torch.tensor(np.stack([np.concatenate([x, np.repeat(x[:1], _P - len(x), 0)])
                                    for x in sh_p]), dtype=torch.float32, device=dev)
    sh_b = torch.tensor(sh_b, dtype=torch.long, device=dev)
    # ⚠ 必须**按手分开**: 左手贴近杯子是它的任务, 不是碰撞。真正的碰撞是**交叉项**
    #   (右手 -> 杯 / 左手 -> 瓶) 和 每只手对桌面。混在一起算最小值会把"正常接近"
    #   误报成"撞物体" —— 第一版我就是这么算的, 数字全是 0.1cm 却毫无意义。
    hb = {h: torch.tensor([i for i, n in enumerate(bn) if n.startswith(f"{h}_")],
                          dtype=torch.long, device=dev) for h in ("left", "right")}
    shb = {h: torch.tensor([i for i in sh_b.tolist()
                            if bn[i].replace("vega_1p_", "").startswith(
                                "R" if h == "right" else "L")],
                           dtype=torch.long, device=dev) for h in ("left", "right")}
    shp = {h: sh_pts[[sh_b.tolist().index(i) for i in shb[h].tolist()]]
           for h in ("left", "right")}
    ztab = float(E.cfg.table_top_z)
    from isaaclab.utils.math import quat_apply as _qa

    def _surf(asset):
        """物体表面采样点 -> 世界系 (M,3)。"""
        pl = E.obj_points
        if pl is None:
            return None
        M = pl.shape[0]
        return _qa(asset.data.root_quat_w[0:1].expand(M, 4), pl) \
            + asset.data.root_pos_w[0:1]

    def _shell_gap(h, org):
        """该侧臂**外壳采样点**最低 z 离桌面的余量 (m)。"""
        if len(shb[h]) == 0:
            return float("nan")
        bp = E.hand.data.body_pos_w[0, shb[h]]
        bq = E.hand.data.body_quat_w[0, shb[h]]
        n = shp[h].shape[1]
        w = _qa(bq.unsqueeze(1).expand(-1, n, 4).reshape(-1, 4),
                shp[h].reshape(-1, 3)).view(len(shb[h]), n, 3) + bp.unsqueeze(1)
        return float((w[..., 2] - org[2]).min()) - ztab

    # 谁抓谁: 主物体归交互手(瓶×右), aux 归另一只手(杯×左)
    own = {prim: "obj", sec: "aux"}
    rows = []
    for t in range(T + 1):
        put(t)
        org = E.scene.env_origins[0]
        so, sa = _surf(E.object), (_surf(_aux) if _aux is not None else None)
        r = [t, _shell_gap("right", org) * 100, _shell_gap("left", org) * 100]
        for h in ("right", "left"):
            hp = E.hand.data.body_pos_w[0, hb[h]]
            r.append(float(hp[:, 2].min() - org[2] - ztab) * 100)       # 手->桌
            tgt = so if own[h] == "obj" else sa
            oth = sa if own[h] == "obj" else so
            r.append(float(torch.cdist(hp, tgt).min()) * 100 if tgt is not None else float("nan"))
            r.append(float(torch.cdist(hp, oth).min()) * 100 if oth is not None else float("nan"))
        rows.append(r)

    import numpy as _np
    A = _np.array(rows)
    print(f"\n{'='*84}")
    print(f"[间隙普查] mode={args.mode}  {len(rows)} 帧 | 主物体=瓶(归{prim}手) aux=杯(归{sec}手)")
    print("  口径: 臂=连杆外壳采样点(真实外壳); 手=连杆原点(**偏乐观**, 连杆本身还有半径)")
    cols = ["帧", "R臂->桌", "L臂->桌",
            f"{prim[0].upper()}手->桌", f"{prim[0].upper()}手->自己的物",
            f"{prim[0].upper()}手->别人的物",
            f"{sec[0].upper()}手->桌", f"{sec[0].upper()}手->自己的物",
            f"{sec[0].upper()}手->别人的物"]
    print("".join(f"{c:>13s}" for c in cols))
    for r in A[::max(len(A) // 12, 1)]:
        print(f"{int(r[0]):>13d}" + "".join(f"{v:>13.2f}" for v in r[1:]))
    print("-" * 84)
    for j, nm in list(enumerate(cols))[1:]:
        col = A[:, j]
        if _np.isnan(col).all():
            continue
        k = int(_np.nanargmin(col))
        flag = ("  ⛔ 穿了" if col[k] < 0 else "  ⚠ <1cm" if col[k] < 1.0 else "")
        print(f"  最小 {nm:<16s} {col[k]:>7.2f} cm  @ 第 {int(A[k,0])} 帧{flag}")
    print(f"{'='*84}\n")
    app.close()
    raise SystemExit(0)

if args.selftest:                       # headless 自检: 放几帧确认不炸, 直接退出
    for t in range(0, T + 1, max(T // max(args.selftest, 1), 1)):
        put(t)
    print(f"[selftest] {args.selftest} 个采样帧放完, 未报错。"
          f" 主手 gs={gs_prim} 副手 gs={gs_sec} 总帧 {T}")
    app.close()
    raise SystemExit(0)

run = 0
while app.is_running():
    run += 1
    put(0)
    t0 = time.time()
    for t in range(T + 1):
        put(t)
        while time.time() - t0 < (t + 1) * dt:
            time.sleep(0.001)
        if not app.is_running():
            break
    # 停在交互开始帧
    print(f"[第 {run} 次] 已停在交互开始帧 (主手 {gs_prim} / 副手 {gs_sec})。")
    if args.loop:
        for _ in range(20):
            put(T)
        continue
    try:
        input("按 Enter 再放一次 (Ctrl-C 退出) ... ")
    except (EOFError, KeyboardInterrupt):
        break
app.close()
