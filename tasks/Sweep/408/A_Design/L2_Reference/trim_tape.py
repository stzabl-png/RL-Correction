"""把 Sweep2 格式的 408 母带裁到 [row0,row1) 并把方块摆到指定的簸箕系 (x,z) (2026-09-13, 台账 §9.15).
动机: 408 人手示范里唯一朝口的两笔在行 165~180 / 203~228; 方块摆在口前起点会被早段横扫推歪 (§9.9)。
用法: python trim_tape.py --src sweep408_sweep2fmt_v2.npz --row0 163 --cube_xz 2.5 11.3 --out sweep408_sweep2fmt_v3_r163.npz
"""
import argparse, hashlib, os, sys
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../../../.."))
sys.path.insert(0, os.path.join(ROOT, "tasks/Sweep/2/A_Design/L3_Learning")); sys.path.insert(0, os.path.join(ROOT, "tasks/Sweep/2/C_Wiring"))
from task_spec import SWEEP408
p = argparse.ArgumentParser(); p.add_argument("--src", default=os.path.join(HERE, "sweep408_sweep2fmt_v2.npz"))
p.add_argument("--row0", type=int, required=True); p.add_argument("--row1", type=int, default=-1)
p.add_argument("--cube_xz", type=float, nargs=2, required=True, help="簸箕系 (x,z) cm, 按新第 0 行的簸箕位姿")
p.add_argument("--out", required=True); a = p.parse_args()
z = np.load(a.src, allow_pickle=True); T0 = len(z["right_q"]); r1 = T0 if a.row1 < 0 else a.row1
out = {k: z[k] for k in z.files}
for k in ("right_q", "left_q", "right_f", "left_f", "obj_pos_0", "obj_pos_1", "obj_quat_0", "obj_quat_1",
          "confidence_0", "confidence_1", "source_frame", "human_right_q", "human_left_q"):
    out[k] = np.asarray(z[k])[a.row0:r1]
T = r1 - a.row0; G = SWEEP408.geometry; TABLE_Z = float(z["table_z"])
q2R = lambda q: Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix()
pan_p, pan_q = out["obj_pos_0"].astype(np.float64), out["obj_quat_0"].astype(np.float64)
br_p, br_q = out["obj_pos_1"].astype(np.float64), out["obj_quat_1"].astype(np.float64)
R0 = q2R(pan_q[0]); local = np.array([a.cube_xz[0] / 100.0, 0.0, a.cube_xz[1] / 100.0])
base = pan_p[0] + R0 @ local; up = R0[:, 1]; assert abs(up[2]) > 0.5
local[1] = (TABLE_Z + G.cube_half + 0.0005 - base[2]) / up[2]; cube_start = pan_p[0] + R0 @ local
Vb = np.asarray(trimesh.load(os.path.join(ROOT, "datasets/sweep408/objects/object_0/object_mesh_scaled_final.obj"), process=False).vertices, np.float64)
face = Vb[(Vb[:, 1] < SWEEP408.bristle_y_max) & (Vb[:, 2] > SWEEP408.bristle_z_min) & (Vb[:, 2] < SWEEP408.bristle_z_max)]
face_s = face[np.random.RandomState(0).choice(len(face), min(4096, len(face)), replace=False)]
near_th = G.cube_half + 0.006; dmin = np.zeros(T); imin = np.zeros(T, int)
for t in range(T):
    d = np.linalg.norm(face_s @ q2R(br_q[t]).T + br_p[t] - cube_start, axis=1); imin[t] = int(np.argmin(d)); dmin[t] = d[imin[t]]
hit = np.flatnonzero(dmin < near_th); contact_row = int(hit[0]) if len(hit) else int(np.argmin(dmin))
out["cube_start_w"] = cube_start.astype(np.float32); out["contact_row"] = np.int32(contact_row)
out["brush_contact_local"] = face_s[imin[contact_row]].astype(np.float32); out["nominal_brush_cube_distance_m"] = np.float32(dmin[contact_row])
out["meta"] = np.asarray(f"{str(z['meta'])} | trim_tape: rows [{a.row0},{r1}) of {T0}, cube pan-frame (x,z)=({a.cube_xz[0]},{a.cube_xz[1]})cm, contact_row={contact_row}")
np.savez(a.out, **out)
print(f"[trim] {os.path.basename(a.src)} 行[{a.row0},{r1}) → {T} 行; 方块 簸箕系 ({a.cube_xz[0]},{a.cube_xz[1]})cm 世界 {np.round(cube_start,4).tolist()} 离桌 {(cube_start[2]-TABLE_Z)*1000:.1f}mm; "
      f"刷面→方块: 行0 {dmin[0]*100:.1f}cm 最小 {dmin.min()*100:.2f}cm@行{int(np.argmin(dmin))}; contact_row={contact_row} ({'命中' if len(hit) else '⚠未命中'}) 命中行占比 {len(hit)/T:.2f}; "
      f"→ {a.out} sha {hashlib.sha256(open(a.out,'rb').read()).hexdigest()[:16]}")
