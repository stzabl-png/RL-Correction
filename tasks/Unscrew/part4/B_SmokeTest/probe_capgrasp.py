"""盖抓取候选**逐个实测**: 15 个候选一次会话里全试, 直接量"能不能捏住"。

背景 (2026-09-01): 右手一直没有抓取重定向 (βR=0, 指形直接用人手指流), 结果
右手全程碰不到盖 -> 真实螺纹副下扭矩传不进去 -> 拧转收入恒 0, 训练学不到东西。
仓库里其实有 `Screw27_cap_candidates` (15 个, 来源 bottle_cap_sharpa_wave_left,
给右手用要镜像)。选哪个不能靠"IK 可达率"猜 —— 可达 ≠ 捏得住; 得**量接触**。

本探针不走母带: 逐候选自己解腕 IK、套该候选的指形、把瓶钉住 (排除缝1 碰倒瓶
那个未解项的干扰), 保持若干步后报告:
    右垫接触数 / 三指接触数 / 峰值指力 / 三指在盖系的 (径向, 轴向)
盖 r=1.75cm h=1.7cm —— 径向≈2cm、轴向落在 0~1.7cm 才是真捏住。

  UNSCREW_CLIP=32 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. \
      CUDA_VISIBLE_DEVICES=0 $PY tasks/Unscrew/part4/B_SmokeTest/probe_capgrasp.py --headless
"""
import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--hold", type=int, default=20, help="每个候选保持步数")
p.add_argument("--squeeze", type=float, default=1.0, help="候选 squeeze 的倍数")
p.add_argument("--pick", type=str, default="", help="只试这个候选 (前缀匹配)")
p.add_argument("--ztrim", type=str, default="0",
               help="沿盖轴下压量扫描 (m, 逗号分隔): 先验的腕高对这只手偏高时用")
p.add_argument("--curl", type=str, default="",
               help="对准之后再逐级捏合 (度, 逗号分隔): 拇/食/中屈曲 —— 对准只把"
                    "拇指与食指放到盖轴两侧, 跨距仍大于盖径时要靠合拢补上")
p.add_argument("--fit", type=int, default=0,
               help=">0 = 闭环对准迭代次数: 量到指垫实际位置后, 把腕**平移**到"
                    "'拇指与食指指垫中点落在盖心、盖半高', 重解 IK 再量。"
                    "不依赖先验的腕高假设, 用的是这只手的实测几何。")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("unscrew_probe_capgrasp")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ["POUR_SQUEEZE_FF"] = "1"
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import (  # noqa: E402
    GENERIC_JOINT_ORDER)

cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0]
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
IA0 = E.IA0
ik_r = ArmIK("right", anchor_link="arm_center",
             anchor_T=TC.rest_anchor_T("right"))
qr = np.load(os.path.join(TC.TAKE_DIR, "ref_qpos_right.npz"), allow_pickle=True)
FIN = [str(n) for n in qr["joint_names"]]
PERM = [GENERIC_JOINT_ORDER.index(n) for n in FIN]

# 盖静置位姿 (母带交互首行) + 物体系原点修正
z1 = np.load(TC.REF_V1, allow_pickle=True)
cap_p0 = np.asarray(z1["obj_pos_1"], np.float64)[
    np.flatnonzero(np.asarray(z1["source"], np.int8) == 1)[0]]
cap_q0 = np.asarray(z1["obj_quat_1"], np.float64)[
    np.flatnonzero(np.asarray(z1["source"], np.int8) == 1)[0]]
Rc0 = quat_to_R(cap_q0)
import trimesh  # noqa: E402
import json  # noqa: E402
lay = json.load(open(os.path.join(TC.TAKE_DIR, "scene_layout.json")))["objects"]
CAP_ID = sorted(lay, key=lambda o: -max(lay[o]["extent_cm"]))[-1]
_zc = np.asarray(trimesh.load(lay[CAP_ID]["mesh"], process=False,
                              force="mesh").vertices)[:, 2]
CAP_DZ = float(0.5 * (_zc.min() + _zc.max()) - _zc.min())
print(f"[capgrasp] 盖 {np.round(cap_p0, 3)} | mesh z[{_zc.min():.3f},{_zc.max():.3f}] "
      f"原点修正 {CAP_DZ * 100:+.2f}cm", flush=True)

_pin_pose = torch.cat([E.object.data.root_pos_w.clone(),
                       E.object.data.root_quat_w.clone()], dim=1)
_pin_vel = torch.zeros(1, 6, device=dev)
org = E.scene.env_origins[0].cpu().numpy()
base = E.ref58[IA0].clone()


def hold(q_arm_r, fin_r, steps):
    tgt = base.clone()
    tgt[0:7] = torch.tensor(q_arm_r, dtype=torch.float32, device=dev)
    tgt[14:36] = torch.tensor(fin_r, dtype=torch.float32, device=dev)
    full = E.hand.data.joint_pos.clone()
    full[0, E.map_ids_t] = tgt
    for _ in range(steps):
        E.hand.set_joint_position_target(full)
        for _ in range(DECI):
            E.object.write_root_pose_to_sim(_pin_pose)
            E.object.write_root_velocity_to_sim(_pin_vel)
            E._SA.apply_screw(E)
            E.scene.write_data_to_sim()
            E.sim.step(render=False)
            E.scene.update(E.sim.get_physics_dt())


print("\n候选                 IK残差      右垫 三指 峰值力 | 三指(盖系 径向/轴向 cm)",
      flush=True)
rows = []
_files = sorted(glob.glob(os.path.join(TC.PRIOR_CAP_DIR, "*.npz")))
if args.pick:
    _files = [f for f in _files
              if os.path.basename(f).startswith(args.pick)] or _files
_trims = [float(x) for x in args.ztrim.split(",")]
for cf, _zt in ((c, t) for c in _files for t in _trims):
    cz = np.load(cf)
    g = np.asarray(cz["grasp"], np.float64)
    sq = np.asarray(cz["squeeze"], np.float64)
    gp, gq = g[:3].copy(), g[3:7].copy()
    gp[1] = -gp[1]                                  # 左手抓法 -> 右手镜像
    gq = np.array([gq[0], -gq[1], gq[2], -gq[3]])
    gq /= np.linalg.norm(gq)
    gp[2] += CAP_DZ - _zt
    tp = cap_p0 + Rc0 @ gp
    tR = Rc0 @ quat_to_R(gq)
    r = ik_r.solve_traj(tp[None], np.array([[1.0, 0, 0, 0]]), w_rot=0.0,
                        n_restart=1)[0] if False else None
    best = None
    for t in range(12):
        q0 = (np.asarray(z1["right_q"], np.float64)[IA0] if t == 0
              else np.random.default_rng(t).uniform(ik_r.lower, ik_r.upper))
        s = ik_r.solve(tp, tR, q0=q0, iters=250, w_rot=0.25)
        sc = s["pos_err"] + 0.25 * s["rot_err"]
        if best is None or sc < best[0]:
            best = (sc, s)
    s = best[1]
    fin = np.asarray(g[7:29])[PERM] + args.squeeze * (
        np.asarray(sq[7:29])[PERM] - np.asarray(g[7:29])[PERM])
    hold(np.asarray(s["q"], np.float64), fin, args.hold)
    # ---- 闭环对准: 量 -> 平移 -> 重解 -> 再量 ----
    for _it in range(args.fit):
        _cap = E.aux.data.root_pos_w[0].cpu().numpy() - org
        _Rc = quat_to_R(E.aux.data.root_quat_w[0].cpu().numpy())
        _pads = []
        for _i in list(PE.SCREW_TRIAD)[:2]:          # 拇指 + 食指 (对握的两端)
            _w = E.hand.data.body_pos_w[0, E._pad_bids[5 + _i]].cpu().numpy() - org
            _pads.append(_w)
        _mid = 0.5 * (_pads[0] + _pads[1])
        _goal = _cap + _Rc @ np.array([0.0, 0.0, 0.5 * (_zc.max() - _zc.min())])
        _delta = _goal - _mid                        # 世界系平移量
        tp = tp + _delta
        _b = None
        for t in range(8):
            q0 = (np.asarray(s["q"], np.float64) if t == 0
                  else np.random.default_rng(100 + t).uniform(ik_r.lower, ik_r.upper))
            _s2 = ik_r.solve(tp, tR, q0=q0, iters=250, w_rot=0.25)
            _sc = _s2["pos_err"] + 0.25 * _s2["rot_err"]
            if _b is None or _sc < _b[0]:
                _b = (_sc, _s2)
        s = _b[1]
        hold(np.asarray(s["q"], np.float64), fin, args.hold)
        _ach = E.hand.data.body_pos_w[0, E.wid["R"]].cpu().numpy() - org
        print(f"    对准#{_it + 1}: 平移 {np.round(_delta * 100, 1)}cm -> IK "
              f"{s['pos_err'] * 100:.2f}cm | 实际腕 vs 指令腕 差 "
              f"{np.linalg.norm(_ach - tp) * 100:5.2f}cm | 垫R"
              f"{int((E._pads_f().norm(dim=-1)[0][5:] > 0.5).sum())}/5", flush=True)
    # ---- 对准之后逐级捏合 ----
    if args.curl:
        CURL = np.zeros(22)
        for _i, _n in enumerate(FIN):
            if not _n.startswith(("right_thumb", "right_index", "right_middle")):
                continue
            if _n.endswith(("MCP_FE", "PIP", "IP")):
                CURL[_i] = 1.0
            elif _n.endswith("DIP"):
                CURL[_i] = 0.5
            elif _n == "right_thumb_CMC_AA":
                CURL[_i] = 1.0
        for _d in [float(x) for x in args.curl.split(",")]:
            hold(np.asarray(s["q"], np.float64),
                 fin + np.radians(_d) * CURL, args.hold)
            _f = E._pads_f().norm(dim=-1)[0]
            _cap2 = E.aux.data.root_pos_w[0].cpu().numpy() - org
            _Rc2 = quat_to_R(E.aux.data.root_quat_w[0].cpu().numpy())
            _s2 = []
            for _i in list(PE.SCREW_TRIAD):
                _w = E.hand.data.body_pos_w[0, E._pad_bids[5 + _i]].cpu().numpy() - org
                _l = _Rc2.T @ (_w - _cap2)
                _s2.append(f"{np.linalg.norm(_l[:2]) * 100:4.1f}/{_l[2] * 100:+5.1f}")
            print(f"    捏合 {_d:4.0f}°: 垫R{int((_f[5:] > 0.5).sum())}/5 三指"
                  f"{int((_f[5:][list(PE.SCREW_TRIAD)] > 0.5).sum())}/3 峰值"
                  f"{float(_f[5:].max()):6.2f}N | " + "  ".join(_s2), flush=True)
    f = E._pads_f().norm(dim=-1)[0]
    n_pad = int((f[5:] > 0.5).sum())
    n_tri = int((f[5:][list(PE.SCREW_TRIAD)] > 0.5).sum())
    cap = E.aux.data.root_pos_w[0].cpu().numpy() - org
    Rc = quat_to_R(E.aux.data.root_quat_w[0].cpu().numpy())
    pp = []
    for i in list(PE.SCREW_TRIAD):
        w = E.hand.data.body_pos_w[0, E._pad_bids[5 + i]].cpu().numpy() - org
        l = Rc.T @ (w - cap)
        pp.append(f"{np.linalg.norm(l[:2]) * 100:4.1f}/{l[2] * 100:+5.1f}")
    print(f"{os.path.basename(cf)[:16]:16s} z-{_zt * 100:4.1f}cm "
          f"{s['pos_err'] * 100:5.2f}cm/"
          f"{np.degrees(s['rot_err']):4.1f}°  {n_pad}/5  {n_tri}/3 "
          f"{float(f[5:].max()):6.2f}N | " + "  ".join(pp), flush=True)
    rows.append((n_tri, n_pad, -float(s["pos_err"]),
                 f"{os.path.basename(cf)}@z-{_zt * 100:.1f}cm"))
rows.sort(reverse=True)
print(f"\n[capgrasp] ★接触最好的三个: "
      + " | ".join(f"{r[3]}({r[0]}三指/{r[1]}垫)" for r in rows[:3]), flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
