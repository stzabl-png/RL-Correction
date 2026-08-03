"""候选轨迹设计 + 离线可行性核验 (纯 numpy, 不开 Isaac).

在 camera 锚定的摆放上, 造一条分段参考:
  approach  重建腕路径, 平滑把末端拉到 PreGrasp 悬停位姿
  descend   悬停 -> 合拢中心落到 affordance 中心 (下降 hover_gap)
  close     腕冻结, 手指 open->closed
  lift      抬 10cm
再核验: 手不穿桌 / IK 可达 / 关节连续 / 静态力矩撑得住.
"""
import contextlib, io, sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
from rl_rebuild.correction import clips, place_camera as PC
from rl_rebuild.correction.ref_builders.replay_grasp import (
    _affordance_target, grasp_center_local, hand_lowest_world,
    GENERIC_OPEN, GENERIC_CLOSED)
from rl_rebuild.correction.kinematics import robot_anchors, quat_to_R
from rl_rebuild.correction import check_holdable as CH

CLIP = sys.argv[1] if len(sys.argv) > 1 else "Grasp2"
TZ, CLEAR, HOVER = 0.85, 0.015, 0.03
N_DOWN, N_CLOSE, N_LIFT = 20, 30, 25

cfg = SimpleNamespace(clip_name=CLIP, target_hz=20.0, table_top_z=TZ, clearance=CLEAR,
                      freeze_wrist=False, hover_gap=None, anchor_mode="camera")
with contextlib.redirect_stdout(io.StringIO()):
    du = clips.load_data_unit(cfg)
r = du.ref
e = clips.clip_entry(CLIP)
hand = clips.interact_hand(CLIP)
wp = r.track_wrist[:, :3].astype(np.float64)
wq = r.track_wrist[:, 3:7].astype(np.float64)
gs = min(int(r.interaction_seg[0]), len(wp) - 1)
obj = du.object_init_pose[:3].astype(np.float64)
oq = r.track_object[0, 3:7].astype(np.float64)
aff_c = _affordance_target(e["affordance"]) if e.get("affordance") else np.zeros(3)
aff_w = PC.affordance_world(obj, oq, aff_c)
gc = grasp_center_local(hand)
pg = PC.pregrasp_hover_pose(aff_w, wq[gs], gc, HOVER)
q_gs = wq[gs]

print(f"clip={CLIP} hand={hand} L={len(wp)} gs={gs}")
print(f"物体 {np.round(obj,4)}  affordance 世界 {np.round(aff_w,4)} (离桌 {(aff_w[2]-TZ)*100:.2f}cm)")
print(f"PreGrasp 悬停腕 {np.round(pg,4)}   参考腕@gs {np.round(wp[gs],4)}  "
      f"缺口 {np.linalg.norm(wp[gs]-pg)*100:.2f}cm")

# ---------- 设计轨迹 ----------
# approach: 重建路径 + 平滑偏移 ramp, 让 t=gs 落在 pg
delta = pg - wp[gs]
u = np.clip(np.arange(gs + 1) / max(gs, 1), 0, 1)
u = u * u * (3 - 2 * u)                                   # smoothstep
appr_p = wp[:gs + 1] + u[:, None] * delta[None]
appr_q = wq[:gs + 1].copy()

# descend: 合拢中心从 aff+hover 降到 aff
down_p = np.stack([pg + np.array([0, 0, -HOVER * s])
                   for s in (np.arange(1, N_DOWN + 1) / N_DOWN)])
pg_contact = down_p[-1]
close_p = np.repeat(pg_contact[None], N_CLOSE, 0)
lift_p = np.stack([pg_contact + np.array([0, 0, 0.10 * s])
                   for s in (np.arange(1, N_LIFT + 1) / N_LIFT)])

P = np.concatenate([appr_p, down_p, close_p, lift_p])
Q = np.concatenate([appr_q, np.repeat(q_gs[None], N_DOWN + N_CLOSE + N_LIFT, 0)])
seg = dict(approach=(0, len(appr_p)), descend=(len(appr_p), len(appr_p) + N_DOWN),
           close=(len(appr_p) + N_DOWN, len(appr_p) + N_DOWN + N_CLOSE),
           lift=(len(appr_p) + N_DOWN + N_CLOSE, len(P)))
print(f"\n设计轨迹 {len(P)} 帧: " + "  ".join(f"{k}[{a},{b})" for k, (a, b) in seg.items()))

# 手指: 张开 -> 合拢
fing = np.repeat(GENERIC_OPEN[None].astype(np.float64), len(P), 0)
ca, cb = seg["close"]
uu = np.clip((np.arange(len(P)) - ca) / max(cb - ca, 1), 0, 1)
uu = (uu * uu * (3 - 2 * uu))[:, None]
fing = (1 - uu) * GENERIC_OPEN[None] + uu * GENERIC_CLOSED[None]

# ---------- 核验 1: 穿桌 ----------
lo = np.array([hand_lowest_world(P[t], Q[t], hand, qvec=fing[t]) for t in range(len(P))])
print(f"\n[1] 手最低点−桌面: 全程最小 {(lo.min()-TZ)*100:+.2f}cm  "
      f"穿桌帧 {(lo < TZ).sum()}/{len(P)}")
for k, (a, b) in seg.items():
    print(f"      {k:<9} 最低 {(lo[a:b].min()-TZ)*100:+6.2f}cm  穿桌 {(lo[a:b] < TZ).sum()}/{b-a}")

# 合拢中心实际落点
gc_w = np.array([P[t] + quat_to_R(Q[t]) @ grasp_center_local(hand, fing[t])
                 for t in (seg["descend"][1] - 1, seg["close"][1] - 1)])
print(f"    合拢中心 @接触 {np.round(gc_w[0],4)}  @合拢完成 {np.round(gc_w[1],4)}  "
      f"affordance {np.round(aff_w,4)}  距离 {np.linalg.norm(gc_w[1]-aff_w)*100:.2f}cm")

# ---------- 核验 2: IK ----------
A = robot_anchors()
ik = A["ik"][hand]
sols = ik.solve_traj(P, Q, pos_tol=0.005, rot_tol=np.deg2rad(15))
ok = np.array([s["ok"] for s in sols])
pe = np.array([s["pos_err"] for s in sols])
re_ = np.array([s["rot_err"] for s in sols])
qs = np.stack([s["q"] for s in sols])
jump = np.degrees(np.abs(np.diff(qs, axis=0)).max(axis=1))
print(f"\n[2] IK 位+姿可达 {ok.mean()*100:.1f}%   仅位置(<5mm) {(pe<0.005).mean()*100:.1f}%   "
      f"位置误差 中位 {np.median(pe)*100:.2f}cm 最大 {pe.max()*100:.2f}cm")
print(f"    姿态误差 中位 {np.degrees(np.median(re_)):.1f}° 最大 {np.degrees(re_.max()):.1f}°   "
      f"关节跳变最大 {jump.max():.1f}°")
for k, (a, b) in seg.items():
    print(f"      {k:<9} 位+姿 {ok[a:b].mean()*100:5.1f}%  仅位置 {(pe[a:b]<0.005).mean()*100:5.1f}%"
          f"  最大位置误差 {pe[a:b].max()*100:5.2f}cm")
sh = A["shoulders"][hand]
print(f"    肩距: 最大 {np.linalg.norm(P-sh,axis=1).max():.3f}m (臂展 ~0.755m)")

# ---------- 核验 3: 静态力矩 ----------
masses = CH.link_masses()
tau = np.stack([CH.gravity_torque(ik, qs[t], masses) for t in range(0, len(P), 3)])
lim = np.array([CH.ARM_EFFORT[i + 1] for i in range(tau.shape[1])])
ratio = np.abs(tau) / lim
print(f"\n[3] 静态重力矩/上限: 最大 {ratio.max()*100:.1f}%  "
      f"(各关节最大 {np.round(ratio.max(0)*100,1).tolist()})")
print(f"    超限帧 {(ratio.max(1) > 1).sum()}/{len(tau)}")
np.save("/home/lyh/Project/RL_Correction/tools/grasp_design/traj_P.npy", P)
