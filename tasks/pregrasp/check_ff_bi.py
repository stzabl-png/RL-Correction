"""双臂前馈体检: 零动作跑完接近窗口, 双侧逐段打印 d_pos/d_rot/腕高。

回答两个问题 (2026-08-18, L4 右手 6.8cm 之谜 + L5 左手高位):
  1. 纯前馈(零残差)终点的 R 侧 d_pos —— 若 ≈ 训练里的停摆值(6.8cm),
     则策略只是躺在前馈终点, 缺口是参考设计(预抓后退)自带的;
  2. 参考路径上 L 腕的世界系高度剖面 —— 若参考本身不高, 录像里的
     "头顶落下"就全是残差学出来的停机坪。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.check_ff_bi --headless \
        --clip Pour17_bottle --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz \
        --prior_yaw 19.5 --prior_b tasks/pregrasp/priors/Pour17_cup.npz \
        --prior_b_yaw 180 --curobo_ref tasks/pregrasp/priors/curobo_pour17_joint_ref.npz \
        --approach --approach_only --minimal
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--prior_b", required=True)
p.add_argument("--prior_b_yaw", type=float, default=-1.0)
p.add_argument("--curobo_ref", required=True)
p.add_argument("--ff_freeze_cm", type=float, default=5.0)
p.add_argument("--ff_pull", type=float, default=0.08)
p.add_argument("--approach", action="store_true")
p.add_argument("--approach_only", action="store_true")
p.add_argument("--minimal", action="store_true")
p.add_argument("--l5", action="store_true")
p.add_argument("--dyn_far", type=float, default=-1.0)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--video", default="",
               help="非空 = 把零动作回放渲染成 mp4 存到该目录 (无 GUI 看效果)")
p.add_argument("--ref_stride", type=int, default=2,
               help="前馈播放步幅: 2=默认(2倍速), 1=逐帧(半速, 排查跟踪滞后用)")
p.add_argument("--pregrasp29", action="store_true",
               help="PreGrasp29 任务口径: 靶点=掌心PreGrasp (配 RL_HAND_JOINTS=1)")
p.add_argument("--phase2", action="store_true",
               help="到位后播 PreGrasp→GraspPose 名义斜坡 (验证 Phase2 稳定性)")
p.add_argument("--dump6d", default="",
               help="非空=逐步记录双物体 6D 位姿(pos+quat)+倾角+ref_t 到该 npz "
                    "(掌部接触/纯倾斜/回弹这类垫传感器盲区事件全可见)")
p.add_argument("--aag", action="store_true",
               help="AAG 口径: 指参考逐行(fin_ref_npz=curobo_ref) + 预算/豁免同 train"
                    " —— 零动作=臂指全按编舞走, 用于量'参考自己何时碰物'")
p.add_argument("--arm_abs", action="store_true",
               help="方案C: 臂绝对参考+有界残差 (零动作下与差分版应逐位等价, 用于对拍)")
p.add_argument("--arm_abs_dev_deg", type=float, default=2.86)
p.add_argument("--fin_cart", action="store_true",
               help="与 train 的 --fin_cart 同义: g2 指判据换成逐指笛卡尔 <fin_cart_tol")
p.add_argument("--fin_gate", action="store_true",
               help="与 train 的 --fin_gate 同义: 到位前指参考钳在 fin_hold_row")
p.add_argument("--eps_pos_cm", type=float, default=0.0,
               help="到位位置阈值 (cm), 必须与训练同值 (AAG 训练=1.35); 0=用默认 1.0")
p.add_argument("--native", action="store_true",
               help="用 v2 原生双臂底座 (bimanual_native_env) —— 新旧底座回归对拍用")
p.add_argument("--loop", action="store_true",
               help="GUI 循环回放: 不渲视频, 在 Isaac 窗口里无限循环零动作回放 "
                    "(配合不带 --headless 使用)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.video:
    args.enable_cameras = True                 # 离屏渲染必需

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("check_ff_bi")
app = AppLauncher(args).app

import numpy as _np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp import bimanual as BM  # noqa: E402
if args.native:
    from tasks.pregrasp.bimanual_native_env import BimanualNativeEnv as BimanualApproachEnv  # noqa: E402,N813
    print("[check_ff_bi] ★ v2 原生底座 (bimanual_native_env)")
else:
    from tasks.pregrasp.bimanual_env import BimanualApproachEnv  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402

# ---- 场景组装: 与 record.py/train.py 逐项一致 ----
env_cfg = GraspTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
if args.approach_only:
    env_cfg.approach_only = True
    env_cfg.action_space = 7
if args.pregrasp29:
    # ⚠ 必须在 apply_grasp_prior 之前 (观测宽度按 approach_only/pregrasp29 算死):
    #   approach_only 语义 + 29 维动作(由 RL_HAND_JOINTS=1 提供, 不强制 7)
    env_cfg.approach_only = True
    env_cfg.pregrasp29 = True
    if args.phase2:
        env_cfg.pregrasp_phase2 = True
apply_grasp_prior(env_cfg, args.grasp_prior, args.prior_yaw, approach=args.approach)
if args.approach:
    env_cfg.direct_grasp_prob = 0.0
    env_cfg.approach_t0_max = 0.0
if args.minimal:
    env_cfg.minimal_no_ff = True
    env_cfg.minimal_fixed_res = False
    env_cfg.eps_pos, env_cfg.eps_rot = 0.01, _np.radians(15.0)
    env_cfg.eps_pos0, env_cfg.eps_rot0 = env_cfg.eps_pos, env_cfg.eps_rot
    env_cfg.w_imit0_approach = 0.0
    env_cfg.w_imit_ramp = 0.0
    env_cfg.stance_prob, env_cfg.retract_ratio = 1.0, 1.0
    env_cfg.direct_grasp_prob = 0.0
if args.l5:
    env_cfg.l5_couple = True
    env_cfg.direct_grasp_prob = 0.0
if args.dyn_far > 0:
    env_cfg.dyn_arm_far = float(args.dyn_far)
env_cfg.curobo_ref_npz = os.path.abspath(args.curobo_ref)
if args.aag:
    env_cfg.fin_ref_npz = os.path.abspath(args.curobo_ref)
    env_cfg.approach_extra0 = 400
    env_cfg.near_exempt_m = 0.07
    # ★ 2026-08-23 修: 原来只设了 fin_ref_npz 却没设 fin_ref_track ⟹ env.py:1473
    # 的播放分支恒 False, **指参考根本不播**, 手指全程走通用合拢模板(标量 0=张开)。
    # 于是这个工具的帮助("零动作=臂指全按编舞走")是假的, 它其实只体检了**臂**,
    # 用它的"零接触"去论证参考抓不住是错的 —— 参考的 close/squeeze 压根没上场。
    env_cfg.fin_ref_track = True
    env_cfg.fin_prog_gate = bool(args.fin_gate)      # 与 train 的 --fin_gate 同名同义
if args.fin_cart:
    env_cfg.fin_cart = True
if args.arm_abs:
    env_cfg.arm_abs_res = True
    env_cfg.arm_abs_dev = float(args.arm_abs_dev_deg) * _np.pi / 180.0
if args.eps_pos_cm > 0:
    # 到位阈值必须与训练同值: 默认硬编码 1.0cm, 而 AAG 训练用 1.35cm。
    # 前馈落点正好 1.00/1.09cm ⟹ 用默认值两手都闩不上, 指参考被 fin_prog_gate
    # 永久钳在 fin_hold_row, close/squeeze 永远不放行 (第一次体检的假象来源)。
    env_cfg.eps_pos = float(args.eps_pos_cm) / 100.0
    env_cfg.eps_pos0 = env_cfg.eps_pos
env_cfg.curobo_ref_stride = int(args.ref_stride)
env_cfg.minimal_no_ff = False
env_cfg.curobo_ff_freeze_cm = args.ff_freeze_cm
env_cfg.curobo_ff_pull = args.ff_pull
env_cfg.prior_b_npz = os.path.abspath(args.prior_b)
env_cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
_a1, _o1 = env_cfg.action_space, env_cfg.observation_space
env_cfg.action_space = 2 * _a1
env_cfg.observation_space = 2 * _o1
env_cfg._obs_single = _o1
env_cfg.scene.num_envs = args.num_envs
env_cfg.obj_jitter_xy = 0.0

if args.video:
    import gymnasium as gym
    from isaaclab.envs import ViewerCfg
    env_cfg.viewer = ViewerCfg(eye=(0.55, -0.85, 1.45), lookat=(-0.10, -0.08, 0.92),
                               origin_type="env", env_index=0, resolution=(720, 540))
    env = BimanualApproachEnv(env_cfg, render_mode="rgb_array")
else:
    env = BimanualApproachEnv(env_cfg)
env.gentle = 1.0
stepper = env
if args.video:
    import os as _os
    _os.makedirs(args.video, exist_ok=True)
    stepper = gym.wrappers.RecordVideo(
        env, video_folder=args.video, name_prefix="ff_zero",
        step_trigger=lambda s: s == 0, video_length=env.ep_total,
        disable_logger=True)
stepper.reset()
z = torch.zeros((args.num_envs, env_cfg.action_space), device=env.device)

def side_stats(side):
    with BM.use_side(env, side):
        d = (env._anchor_w() - env._target_w()).norm(dim=1)
        wz = env.wrist_pos_w[:, 2]
    return float(d.mean()) * 100, float(wz.mean())

_best = {}   # 每侧记"离目标最近时刻"的腕世界位 (env0, 减 origin) —— 供跨工具对位
_tipmin = {}  # 每侧全程指尖最低高度 (m, 世界系) + 发生步

_bn_all = list(env.hand.body_names)
_tips = {s: [i for i, n in enumerate(_bn_all)
             if n.startswith(f"{s}_") and n.endswith("_DP")]
         for s in ("right", "left")}

def snap_best(tag, side):
    with BM.use_side(env, side):
        d = float((env._anchor_w() - env._target_w()).norm(dim=1)[0])
        wp = (env._anchor_w()[0] - env.scene.env_origins[0]).cpu().numpy().copy()
        dv = (env._anchor_w() - env._target_w())[0].cpu().numpy().copy()
        _dpw, _drw, _ = env._align_err()
        rec = (d, wp, dv, float(_dpw[0]), float(_drw[0]))
    if tag not in _best or d < _best[tag][0]:
        _best[tag] = rec

_park = {}   # 步 200 (停靠期) 快照: 同 snap_best 字段

def snap_tip(tag, side_name, step):
    z0 = float(env.hand.data.body_pos_w[0, _tips[side_name], 2].min())
    if tag not in _tipmin or z0 < _tipmin[tag][0]:
        _tipmin[tag] = (z0, step)

n = env.ep_total
_6d = [] if args.dump6d else None


def _obj6d(side):
    with BM.use_side(env, side):
        op = (env.object.data.root_pos_w
              - env.scene.env_origins)[0].cpu().numpy().copy()
        oq = env.object.data.root_quat_w[0].cpu().numpy().copy()
        rt = int(env.ref_t[0])
    return op, oq, rt


print(f"\n[check_ff_bi] 零动作 x {n} 步 | A={env._A_name}(先动) B={env._B_name}(后动)")
print(f"{'步':>4s} {'R_dpos(cm)':>11s} {'R_腕z(m)':>9s} {'L_dpos(cm)':>11s} {'L_腕z(m)':>9s}")
if args.loop:
    print("[check_ff_bi] GUI 循环回放中 (Ctrl+C 退出) ...")
    i = 0
    while app.is_running():
        stepper.step(z)
        if i % 10 == 0:
            ra, za = side_stats(env._A)
            rb, zb = side_stats(env._B)
            print(f"{i:>4d} {ra:>11.2f} {za:>9.3f} {rb:>11.2f} {zb:>9.3f}")
        i += 1
    app.close()
    raise SystemExit(0)
for i in range(n):
    stepper.step(z)
    if _6d is not None:
        _pa, _qa, _ra = _obj6d(env._A)
        _pb, _qb, _rb = _obj6d(env._B)
        _6d.append((_pa, _qa, _pb, _qb, _ra))
    _lg = env.extras.get("log") or {}
    _tf = {k: round(float(v), 2) for k, v in _lg.items()
           if "/term/" in str(k) and float(v) > 0}
    if _tf:
        print(f"[终因] 步{i}: {_tf}")
    snap_best("R", env._A)
    snap_best("L", env._B)
    snap_tip("R", env._A_name, i)
    snap_tip("L", env._B_name, i)
    if args.aag:
        # 逐步接触/物体扰动记录: 找"参考自己哪一行碰物"
        for _tg, _sd in (("R", env._A), ("L", env._B)):
            with BM.use_side(env, _sd):
                _F = torch.cat([s_.data.force_matrix_w.view(env.num_envs, 1, 3)
                                for s_ in env._contact_sensors], dim=1)
                _npd = int((_F.norm(dim=-1) > 0.5)[0].sum())
                _od = float((env.object.data.root_pos_w[0]
                             - env.scene.env_origins[0]
                             - env.obj_start_pos[0]).norm()) * 100
                _rt = int(env.ref_t[0])
            if _npd > 0 or _od > 0.5:
                # 逐垫点名 (顺序 = correction_env_cfg.fingertip_bodies)
                _mag = _F.norm(dim=-1)[0].cpu().numpy()
                _nm5 = ("拇", "食", "中", "无", "小")
                _on = " ".join(f"{_nm5[_j]}{_mag[_j]:.1f}N" if _mag[_j] > 0.5
                               else f"{_nm5[_j]}·" for _j in range(5))
                with BM.use_side(env, _sd):
                    _pd = env._pad_dists()[0].cpu().numpy() * 100   # cm, 含+2.4cm系统偏置
                _gap = " ".join(f"{_nm5[_j]}{_pd[_j]:.1f}" for _j in range(5))
                with BM.use_side(env, _sd):
                    _g2d = float(getattr(env, "_g2_d", torch.zeros(1))[0]) * 100
                    _g2f = _np.degrees(float(getattr(env, "_g2_fe",
                                                     torch.zeros(1))[0])) \
                        if getattr(env.cfg, "fin_cart", False) is False else \
                        float(getattr(env, "_g2_fe", torch.zeros(1))[0]) * 100
                    _g2r = int(getattr(env, "_g2_run", torch.zeros(1))[0])
                    _g2k = bool(getattr(env, "_g2_done", torch.zeros(1, dtype=torch.bool))[0])
                    _arr = bool(env.arrived[0])
                print(f"[接触] 步{i:03d} {_tg} ref_t={_rt} 垫数={_npd} "
                      f"[{_on}] 物体位移={_od:.2f}cm || g2: 腕→真抓姿 {_g2d:.2f}cm "
                      f"指最差 {_g2f:.2f} arrived={_arr} run={_g2r} done={_g2k}")
    if i == 200:
        for _t2, _s2 in (("R", env._A), ("L", env._B)):
            with BM.use_side(env, _s2):
                _d2 = float((env._anchor_w() - env._target_w()).norm(dim=1)[0])
                _dv2 = (env._anchor_w() - env._target_w())[0].cpu().numpy().copy()
                _dp2, _dr2, _ = env._align_err()
            _park[_t2] = (_d2, _dv2, float(_dp2[0]), float(_dr2[0]))
    if i % 10 == 0 or i == n - 1:
        ra, za = side_stats(env._A)
        rb, zb = side_stats(env._B)
        print(f"{i:>4d} {ra:>11.2f} {za:>9.3f} {rb:>11.2f} {zb:>9.3f}")
for _t, (_d, _wp, _dv, _dpw, _drw) in _best.items():
    print(f"[末位] {_t} 腕世界位(最近时刻) {_np.round(_wp, 4).tolist()} m "
          f"| 锚距 {_d*100:.2f}cm 向量 xyz=({_dv[0]*100:+.2f},{_dv[1]*100:+.2f},"
          f"{_dv[2]*100:+.2f})cm | 腕口径 {_dpw*100:.2f}cm/{_np.degrees(_drw):.1f}°")
for _t, (_d, _dv, _dpw, _drw) in _park.items():
    print(f"[停靠@200] {_t}: 锚距 {_d*100:.2f}cm 向量 xyz=({_dv[0]*100:+.2f},"
          f"{_dv[1]*100:+.2f},{_dv[2]*100:+.2f})cm | 腕口径 {_dpw*100:.2f}cm/"
          f"{_np.degrees(_drw):.1f}°")
if args.phase2:
    for _tag, _side in (("R", env._A), ("L", env._B)):
        with BM.use_side(env, _side):
            _t2 = int(env._p2_t[0]) if getattr(env, "_p2_t", None) is not None else -1
            _ar = bool(env.arrived[0])
            _sc = bool(env.succeeded[0])
        print(f"[phase2] {_tag} arrived={_ar} 斜坡进度 {_t2} succeeded={_sc}")
_tz = float(getattr(env.cfg, "table_top_z", 0.85))
for _t, (_z0, _st) in _tipmin.items():
    print(f"[指尖] {_t} 全程最低 {_z0:.4f}m = 桌面上方 {(_z0-_tz)*100:+.1f}cm "
          f"(@步 {_st}) {'⚠ 已触桌' if _z0 <= _tz else ''}")
ra, za = side_stats(env._A)
rb, zb = side_stats(env._B)
print(f"\n[末端] R d_pos={ra:.2f}cm  L d_pos={rb:.2f}cm "
      f"(训练里 L4 的 R 停摆值=6.8cm —— 若相近, 缺口=参考自带)")
# ---- 详测 (2026-08-22): 终态偏差**向量** + 双口径 + 物体相对方位 ----
for _tag, _side in (("R", env._A), ("L", env._B)):
    with BM.use_side(env, _side):
        _dv = (env._anchor_w() - env._target_w())[0].cpu().numpy()
        _dp, _dr, _ = env._align_err()
        _op = env.object.data.root_pos_w[0].cpu().numpy()
        _wp = env.wrist_pos_w[0].cpu().numpy()
        _ov = _op - _wp
    print(f"[详测] {_tag}: 锚→靶向量 xyz=({_dv[0]*100:+.2f},{_dv[1]*100:+.2f},"
          f"{_dv[2]*100:+.2f})cm |锚口径|={_np.linalg.norm(_dv)*100:.2f}cm"
          f" | 腕口径(arrive判据) d_pos={float(_dp[0])*100:.2f}cm"
          f" d_rot={_np.degrees(float(_dr[0])):.1f}°"
          f" | 腕→物体 xyz=({_ov[0]*100:+.1f},{_ov[1]*100:+.1f},{_ov[2]*100:+.1f})cm")
if args.video:
    stepper.close()          # RecordVideo 在 close 时才落盘 mp4
    print(f"[video] 已存 {args.video}/ff_zero-step-0.mp4")
if _6d is not None:
    _np.savez_compressed(
        args.dump6d,
        A_pos=_np.stack([x[0] for x in _6d]),
        A_quat=_np.stack([x[1] for x in _6d]),
        B_pos=_np.stack([x[2] for x in _6d]),
        B_quat=_np.stack([x[3] for x in _6d]),
        ref_t=_np.array([x[4] for x in _6d]),
        A_name=str(env._A_name), B_name=str(env._B_name))
    print(f"[dump6d] 双物体 6D 位姿已存 {args.dump6d}: {len(_6d)} 步")
# 活干完立刻放槽位 —— Isaac 退出常挂死, 不放会堵住下一个排队者 (2026-08-18 N 次实测)
try:
    _slot.release()
    print("[check_ff_bi] GPU 槽位已主动释放")
except Exception:
    pass
import sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
os._exit(0)          # Isaac 关闭流程常挂死 (2026-08-22 两次实测), 硬退出保住输出
app.close()
