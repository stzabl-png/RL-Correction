#!/usr/bin/env python
"""pour17_perception.npz —— 方法中立的感知数据(给 H2S2R baseline)。

★ 为什么需要这个文件: reference v1/v2 里**没有任何人手原始量** ——
  两者的 right_q/left_q 都是 (T,7) 机器人臂关节角(ours 的 IK 结果),
  right_f/left_f 是 (T,22) Sharpa 手指。用它们做 "原始视频轨迹核对" 会把
  ours 的 IK 与参考重铸一起带进 baseline。真正方法中立的源头是重建产物。

数据来源(全部来自 Reconstruct_and_Retarget 的 pour/17 重建 take):
  world_fused.npz    相机 K/c2w、双手 MANO 参数、两物体 6DoF、逐帧有效
  ref_qpos_*.npz     腕位姿 + Sharpa 22 维手指 q(DexPilot 重定向, 物体无关)
  object_valid_measured.npz  逐帧"真话版"物体可信标志(判据 ovm_v1)

约定(与 baseline 侧对齐):
  - 长度单位 米
  - 四元数 wxyz
  - left  -> object_0 (杯)   right -> object_1 (瓶)
  - 世界系 gravity_z_up_world(重力对齐、米制、EgoDex 设备标定给定)

用法: python make_pour17_perception.py --take <重建take目录> --out pour17_perception.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def q_wxyz(mats: np.ndarray) -> np.ndarray:
    """(T,4,4) -> (T,4) wxyz。"""
    q = Rotation.from_matrix(mats[:, :3, :3]).as_quat()      # scalar-last xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    T = a.take

    w = np.load(T / "world_fused.npz", allow_pickle=True)
    n = int(w["num_frames"])
    K = np.asarray(w["K"], float)
    c2w = np.asarray(w["c2w"], float)
    obj = np.asarray(w["object_ob_in_world_all"], float)      # (2,T,4,4)
    oids = [str(x) for x in w["object_ids"]]
    hv = np.asarray(w["hand_valid"], float)                   # (2,Th)
    Th = hv.shape[1]
    src = np.clip(np.floor(np.arange(n) * (Th / n)).astype(int), 0, Th - 1)

    out = {
        "frame_ids": np.arange(n, dtype=np.int32),
        "timestamps": np.full(n, np.nan, np.float64),         # 见 note: EgoDex 无绝对时戳
        "K": K, "c2w": c2w.astype(np.float32),
    }

    # ---- 手: 腕位姿 + 22 维 Sharpa q(ref_qpos), 以及 MANO 原始参数(备份路线) ----
    names = None
    for side in ("left", "right"):
        p = T / f"ref_qpos_{side}.npz"
        z = np.load(p, allow_pickle=True)
        out[f"{side}_wrist_pose_wxyz"] = np.concatenate(
            [np.asarray(z["wrist_pos"], np.float32),
             np.asarray(z["wrist_quat_wxyz"], np.float32)], axis=1)     # (T,7)
        out[f"{side}_hand_q"] = np.asarray(z["finger_qpos"], np.float32)  # (T,22)
        out[f"{side}_valid"] = np.asarray(z["valid"]).astype(bool)
        names = np.asarray(z["joint_names"])
    out["hand_joint_names"] = names

    si = {"left": 0, "right": 1}
    for side, i in si.items():                                  # MANO 备份(自行 retarget 用)
        out[f"{side}_mano_trans"] = np.asarray(w["hand_trans"], np.float32)[i][src]
        out[f"{side}_mano_rot"] = np.asarray(w["hand_rot"], np.float32)[i][src]
        out[f"{side}_mano_pose45"] = np.asarray(w["hand_pose"], np.float32)[i][src]
        out[f"{side}_mano_betas"] = np.asarray(w["hand_betas"], np.float32)[i][src]
        out[f"{side}_mano_valid"] = (hv[i][src] > 0.5)

    # ---- 物体: 位姿 + 逐帧真话版可信 ----
    for k, oid in enumerate(oids):
        P = obj[k][:n]
        out[f"{oid}_pose_wxyz"] = np.concatenate(
            [P[:, :3, 3], q_wxyz(P)], axis=1).astype(np.float32)         # (T,7)
    ovm = T / "object_valid_measured.npz"
    if ovm.is_file():
        z = np.load(ovm, allow_pickle=True)
        tw = np.asarray(z["trustworthy"])
        for k, oid in enumerate(oids):
            out[f"{oid}_valid"] = tw[k][:n].astype(bool)
            out[f"{oid}_scored"] = np.asarray(z["scored"])[k][:n].astype(bool)
        out["object_valid_criteria"] = np.array(str(z["criteria_version"]))
    else:
        for k, oid in enumerate(oids):
            out[f"{oid}_valid"] = np.ones(n, bool)

    out["meta"] = np.array(json.dumps({
        "source_take": str(T),
        "units": "meters", "quat": "wxyz",
        "world_frame": str(w["coordinate_frame"]),
        "fps_reconstruction": 15.0,
        "fps_source_video": 30.0,
        "hand_object_pairing": {"left": oids[0] + " (cup)", "right": oids[1] + " (bottle)"},
        "timestamps": "unavailable — EgoDex 不提供绝对时戳; 用 frame_ids / fps 推导",
        "camera_intrinsics": "设备标定(非估计)",
        "camera_extrinsics": "设备 SLAM(非估计), c2w, OpenCV 约定 x右 y下 z前",
        "hand_q_note": "22 维 Sharpa DexPilot 重定向结果, 物体无关; 想自行 retarget 用 *_mano_*",
        "object_valid_note": "trustworthy 判据 ovm_v1: 打过分∧遮挡<0.15∧未证伪∧无misplaced/teleport/lag∧conf_pos>=40",
    }, ensure_ascii=False))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    h = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(f"[perception] -> {a.out}  ({a.out.stat().st_size/1e6:.2f} MB)")
    print(f"[perception] sha256 {h}")
    for k in sorted(out):
        v = np.asarray(out[k])
        print(f"   {k:26s} {str(v.shape):14s} {v.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
