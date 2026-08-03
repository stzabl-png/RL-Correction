"""Dexonomy GraspPose -> 本任务 prior 文件 (物体输入系 = 重建 mesh 系).

用**系统 python3** 跑 (Dexonomy npy 是 numpy2 pickle, MagicSim venv 的 numpy1 读不了):

  python3 tasks/pregrasp/make_prior.py \
      --grasp_npy /home/lyh/Project/Dexonomy/output/pp5_sharpa_wave/grasp_data/fingertip_mid/pp5/tabletop/scale010/43_8_grasp.npy \
      --info_json /home/lyh/Project/Dexonomy/assets/object/custom/processed_data/pp5/info/simplified.json \
      --out tasks/pregrasp/priors/Grasp5.npz

约定 (2026-07-30 用 FK+表面距离自检确认, 见台账):
  qpos 布局 [pos3, quat4(wxyz), 指22(GENERIC_JOINT_ORDER 同序)];
  规范系 = 输入 mesh 系旋转 canonical_from_input_rot (无平移); 输出统一转回**输入系**.
输出 npz 全部 plain array (无 pickle), MagicSim venv (numpy1) 可读.
"""
import argparse
import json

import numpy as np


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def to_input(rows, Ric):
    """(N,29) 规范系 qpos -> 输入系 [pos3, quat4, 指22]."""
    out = []
    for r in np.atleast_2d(rows):
        p, q, f = r[:3], r[3:7], r[7:29]
        out.append(np.concatenate([Ric @ p, R_to_quat(Ric @ quat_to_R(q)), f]))
    return np.stack(out)


p = argparse.ArgumentParser()
p.add_argument("--grasp_npy", required=True)
p.add_argument("--info_json", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

g = np.load(a.grasp_npy, allow_pickle=True).item()
info = json.load(open(a.info_json))
Rci = quat_to_R(np.asarray(info["canonical_from_input_rot_wxyz"]))
Ric = Rci.T

hoc = g["ho_c"]
cpos = np.asarray(hoc["pos"], np.float64) @ Rci        # (N,3) 规范系 -> 输入系 (R^T 右乘)
cnrm = np.asarray(hoc["normal"], np.float64) @ Rci
out = dict(
    grasp=to_input(np.asarray(g["grasp_qpos"], np.float64), Ric)[0],       # (29,)
    squeeze=to_input(np.asarray(g["squeeze_qpos"], np.float64), Ric)[0],
    pregrasp=to_input(np.asarray(g["pregrasp_qpos"], np.float64), Ric),    # (6,29)
    contact_pos=cpos, contact_normal=cnrm,
    contact_centroid=cpos.mean(axis=0),
    # 生成抓姿时物体在**规范姿态**下平放于桌面; env 必须按同一姿态摆放,
    # 否则非对称物体会上下颠倒 (Grasp3 实测差 179.9°, 抓姿落到桌面以下 12cm).
    canon_rot=np.asarray(info["canonical_from_input_rot_wxyz"], np.float64),
    source=np.frombuffer(a.grasp_npy.encode(), dtype=np.uint8),  # 溯源 (plain array)
)
np.savez(a.out, **out)
print(f"[make_prior] -> {a.out}")
print(f"  grasp 腕(输入系) pos={np.round(out['grasp'][:3],4)} quat={np.round(out['grasp'][3:7],4)}")
print(f"  pregrasp {out['pregrasp'].shape[0]} 个 | 接触点 {len(cpos)} 个 "
      f"质心 {np.round(out['contact_centroid'],4)}")
