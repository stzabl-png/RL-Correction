"""408 母带 → Sweep2 env 母带格式 (2026-09-13, 台账 §9).

Sweep2 的 env 读 [obj_0 = 簸箕(左), obj_1 = 扫把(右)] + 方块四键 + 人手臂 q; 408 母带的索引约定相反
(obj_0 = 扫把), 且没有方块。本脚本:
  ① 交换索引 (obj_pos/quat, confidence, scene_pose)
  ② 方块: cube_start_w 用 Sweep2 的 _pan_cube_start 公式 (簸箕第 0 行位姿 + 408 几何常量);
     contact_row = 刷毛面首次进入 cube_half+6mm (= env 的 broom_near) 的行;
     brush_contact_local = 该行离方块最近的刷毛面点 (网格局部系); nominal_brush_cube_distance_m 同行距离
  ③ 人手臂 q: ref_qpos_{right,left}.npz 的腕位姿 (recon_world = 场景系, table 0.87 同) 经与物轨
     **同一个** register(yaw, cx, cy, origin_xy) 平移后, 重采样到母带行, ArmIK 连续解 (pos_tol 0.2mm)。
     只有 `full` 消融用它 (wo_human/wo_conf 的 w_hand=0)。
用法: PYTHONPATH=. python tasks/Sweep/408/A_Design/L2_Reference/to_sweep2_format.py [--tape v3.npz] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot, Slerp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tasks/Sweep/2/A_Design/L3_Learning"))
sys.path.insert(0, os.path.join(ROOT, "tasks/Sweep/2/C_Wiring"))
from rl_rebuild.correction.kinematics import ArmIK  # noqa: E402
from task_spec import SWEEP408  # noqa: E402

TAKE = os.environ.get("SWEEP_TAKE", "408")           # 2026-09-13: 175 复用 (几何 spec 仍取 SWEEP408: 同一批实物)
DATA = os.path.join(ROOT, "datasets", f"sweep{TAKE}")
DEXO = "/home/lyh/Project/Dexonomy/assets/object/custom/processed_data"

p = argparse.ArgumentParser()
p.add_argument("--tape", default=os.path.join(HERE, "sweep408_reference_v3.npz"))
p.add_argument("--out", default=os.path.join(HERE, "sweep408_sweep2fmt_v2.npz"))
p.add_argument("--no_human_ik", action="store_true", help="跳过人手臂 IK, human_*_q 用 right_q/left_q 顶替 (只跑 wo_human/wo_conf 时够用)")
args = p.parse_args()

z = np.load(args.tape, allow_pickle=True)
G = SWEEP408.geometry
TABLE_Z = float(z["table_z"])
T = len(z["right_q"])


def q2R(q):
    return Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix()


# ---------------- ① 索引交换 ----------------
out = {k: np.asarray(z[k]) for k in ("right_q", "left_q", "right_f", "left_f", "fin_names",
                                     "control_hz", "source_fps", "source_frame", "scene_yaw_deg",
                                     "ik_report", "anchor_T", "table_z")}
for a, b in ((0, 1), (1, 0)):                       # 新 obj_a = 408 obj_b
    out[f"obj_pos_{a}"] = np.asarray(z[f"obj_pos_{b}"], np.float32)
    out[f"obj_quat_{a}"] = np.asarray(z[f"obj_quat_{b}"], np.float32)
    c = np.asarray(z[f"confidence_{b}"], np.float32)
    # ★ Sweep2 母带的 confidence 是百分制 (25~91), sweep_env 读入时 /100; 408 build_reference 存的是 0~1。
    #   v1 (2026-09-13) 原样复制 → 环境里 conf≈0.003~0.01 → w_hand=0.8 全行 / 追踪容差 8cm / 无转角罚 (台账 §9.13)。
    out[f"confidence_{a}"] = c * 100.0 if float(c.max()) <= 1.5 else c
    out[f"scene_pose_{a}"] = np.asarray(z[f"scene_pose_{b}"])
pan_p, pan_q = out["obj_pos_0"].astype(np.float64), out["obj_quat_0"].astype(np.float64)
br_p, br_q = out["obj_pos_1"].astype(np.float64), out["obj_quat_1"].astype(np.float64)
print(f"[fmt] 母带 {os.path.basename(args.tape)} {T} 行; 索引已交换: obj_0=簸箕 obj_1=扫把")

# ---------------- ② 方块 ----------------
R0 = q2R(pan_q[0])
local = np.array([0.0, 0.0, G.pan_mouth_z + G.start_outside])
base = pan_p[0] + R0 @ local
up = R0[:, 1]
assert abs(up[2]) > 0.5, up
target_z = TABLE_Z + G.cube_half + 0.0005
local[1] = (target_z - base[2]) / up[2]
cube_start = pan_p[0] + R0 @ local
print(f"[cube] 起点 (簸箕系 x=0, z=mouth+{G.start_outside*100:.1f}cm): 世界 {np.round(cube_start, 4).tolist()} "
      f"离桌 {(cube_start[2]-TABLE_Z)*1000:.1f}mm (簸箕局部 y={local[1]*1000:+.1f}mm)")

Vb = np.asarray(trimesh.load(os.path.join(DATA, "objects/object_0/object_mesh_scaled_final.obj"),
                             process=False).vertices, np.float64)
mask = (Vb[:, 1] < SWEEP408.bristle_y_max) & (Vb[:, 2] > SWEEP408.bristle_z_min) & (Vb[:, 2] < SWEEP408.bristle_z_max)
face = Vb[mask]
rng = np.random.RandomState(0)
face_s = face[rng.choice(len(face), min(4096, len(face)), replace=False)]
near_th = G.cube_half + 0.006
dmin = np.zeros(T); imin = np.zeros(T, int)
for t in range(T):
    W = face_s @ q2R(br_q[t]).T + br_p[t]
    d = np.linalg.norm(W - cube_start, axis=1)
    imin[t] = int(np.argmin(d)); dmin[t] = d[imin[t]]
hit = np.flatnonzero(dmin < near_th)
contact_row = int(hit[0]) if len(hit) else int(np.argmin(dmin))
brush_contact_local = face_s[imin[contact_row]]
print(f"[cube] 刷毛面 {len(face)} 点 (抽 {len(face_s)}); 刷面到方块起点距离: 行0 {dmin[0]*100:.1f}cm, 最小 {dmin.min()*100:.2f}cm@行{int(np.argmin(dmin))}; "
      f"broom_near(<{near_th*100:.2f}cm) 首行 = **{contact_row}** ({'命中' if len(hit) else '⚠ 从未进入阈值, 取最近行'}); "
      f"接触点局部 {np.round(brush_contact_local*100, 2).tolist()}cm; 命中行占比 {len(hit)/T:.2f}")
out["cube_start_w"] = cube_start.astype(np.float32)
out["contact_row"] = np.int32(contact_row)
out["brush_contact_local"] = brush_contact_local.astype(np.float32)
out["nominal_brush_cube_distance_m"] = np.float32(dmin[contact_row])

# ---------------- ③ 人手臂 q ----------------
report = {}
if args.no_human_ik:
    out["human_right_q"] = out["right_q"]; out["human_left_q"] = out["left_q"]
    report = {"source": "copied_from_reference (no_human_ik)"}
    print("[human] 跳过 IK: human_*_q = right_q/left_q 顶替 (仅供 wo_human/wo_conf)")
else:
    # 与 build_reference 同一套 origin_xy: 扫把 RTS 轨 despike+lowpass 后的首帧 xy
    sys.argv = [sys.argv[0]]
    rts = np.load(os.path.join(DATA, "poseqa", f"rts_sweep_dustpan_{TAKE}_object_0.npz"))
    Tm = np.asarray(rts["object_ob_in_world_smooth"], np.float64)
    com = np.asarray(json.load(open(f"{DEXO}/p4t408_broom_r2/info/simplified.json"))["com_offset"], np.float64) / 1.30
    Rr = Tm[:, :3, :3]; pr = Tm[:, :3, 3] + np.einsum("nij,j->ni", Rr, com)
    # despike (同阈值 8°/30mm) + lowpass win5, 只为拿 origin_xy —— 与构带器逐位同算
    rot = Rot.from_matrix(Rr); n = len(Rr)
    d_ang = np.zeros(n); d_pos = np.zeros(n)
    d_ang[1:] = (rot[:-1].inv() * rot[1:]).magnitude(); d_pos[1:] = np.linalg.norm(np.diff(pr, axis=0), axis=1)
    bad = (d_ang > np.radians(8.0)) | (d_pos > 0.030); bad[0] = False
    gi = np.flatnonzero(~bad)
    p2 = np.stack([np.interp(np.arange(n), gi, pr[gi, k]) for k in range(3)], 1)
    h = 2; P = np.stack([np.convolve(np.r_[[p2[0, k]]*h, p2[:, k], [p2[-1, k]]*h], np.ones(5)/5, "valid") for k in range(3)], 1)
    origin_xy = P[0, :2].copy()
    cx, cy = [float(v) for v in z["scene_center_xy"]]; yaw = float(z["scene_yaw_deg"])
    a = np.radians(yaw); Rw = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
    anchor_T = np.asarray(z["anchor_T"], np.float64)
    sf = np.asarray(z["source_frame"], np.float64)
    for side, key in (("right", "human_right_q"), ("left", "human_left_q")):
        h5 = np.load(os.path.join(DATA, f"ref_qpos_{side}.npz"), allow_pickle=True)
        wp = np.asarray(h5["wrist_pos"], np.float64); wq = np.asarray(h5["wrist_quat_wxyz"], np.float64)
        assert len(wp) == n, (len(wp), n)
        wp = np.einsum("ij,nj->ni", Rw, wp - np.r_[origin_xy, 0.0]) + np.r_[cx, cy, 0.0]
        Rh = np.einsum("ij,njk->nik", Rw, Rot.from_quat(np.c_[wq[:, 1:], wq[:, :1]]).as_matrix())
        # 重采样到母带行 (source_frame 是每行对应的源帧号, 可为小数)
        t_src = np.clip(sf, 0, n - 1)
        Ps = np.stack([np.interp(t_src, np.arange(n), wp[:, k]) for k in range(3)], 1)
        Rs = Slerp(np.arange(n), Rot.from_matrix(Rh))(t_src).as_matrix()
        ik = ArmIK(side, anchor_link="arm_center", anchor_T=anchor_T)
        q = ik.q_default.copy(); Q = np.zeros((T, 7)); pe = np.zeros(T); re = np.zeros(T)
        for t in range(T):
            r = ik.solve(Ps[t], Rs[t], q0=q, iters=200, pos_tol=2e-4, rot_tol=5e-3)
            q = r["q"]; Q[t] = q; pe[t] = r["pos_err"]; re[t] = r["rot_err"]
        Q = np.clip(Q, ik.lower + 1e-6, ik.upper - 1e-6)
        out[key] = Q.astype(np.float32)
        report[side] = dict(pos_med_cm=float(np.median(pe)*100), pos_max_cm=float(pe.max()*100),
                            pos_p90_cm=float(np.percentile(pe, 90)*100), rot_max_deg=float(np.degrees(re.max())),
                            ok_ratio=float((pe < 0.02).mean()),
                            joint_step_max_deg=float(np.degrees(np.abs(np.diff(Q, axis=0)).max())))
        print(f"[human] {side:5s} 腕→臂 IK: 位置 中位 {report[side]['pos_med_cm']:.2f}cm p90 {report[side]['pos_p90_cm']:.2f} max {report[side]['pos_max_cm']:.2f}cm | "
              f"<2cm 占比 {report[side]['ok_ratio']*100:.1f}% | 关节单步 max {report[side]['joint_step_max_deg']:.2f}°")
    report["origin_xy"] = origin_xy.tolist(); report["register"] = dict(yaw=yaw, cx=cx, cy=cy)
out["human_ik_report"] = np.asarray(json.dumps(report))
out["meta"] = np.asarray(f"Sweep{TAKE} 母带转 Sweep2 格式 (2026-09-13): 索引交换 obj_0=簸箕/obj_1=扫把; 方块起点按 _pan_cube_start(408几何 mouth_z={G.pan_mouth_z} start_outside={G.start_outside}); "
                         f"contact_row={contact_row}; 人手臂 q={'IK自 ref_qpos 腕位姿' if not args.no_human_ik else 'right_q/left_q 顶替'}; 源 {os.path.basename(args.tape)}")
np.savez(args.out, **out)
import hashlib
print(f"[fmt] → {args.out}  sha256 {hashlib.sha256(open(args.out,'rb').read()).hexdigest()[:16]}  键: {sorted(out)}")
