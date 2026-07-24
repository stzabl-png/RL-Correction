"""离线导出 Sharpa 22 关节参考 qpos — replay_world.npz -> ref_qpos.npz.

复用 ego_pipeline/Retargeting 的在线 retarget 调用链
(magicdexmate: build_sharpa_retargeting + compute_ref_value + to_mano + JointMapper),
把 retarget_isaacsim.py 每次回放现算的手指 qpos 一次性落盘, 供 RL 参考轨迹用.

不启动 Isaac. 用法:
  .venv-isaac/bin/python -m rl_rebuild.correction.export_qpos           # clip-11 默认
  ... --npz <replay_world.npz> --mesh <obj> --out <ref_qpos.npz> --mode dexpilot

输出 ref_qpos.npz:
  finger_qpos (T,22) SDK 序(弧度, 已限位裁剪) | joint_names (22,) | wrist_pos (T,3)
  wrist_quat_wxyz (T,4) 桌面局部系 | valid (T,) | fps | retarget_mode
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from rl_rebuild.correction import frames as F
from rl_rebuild.correction.load_replay import CLIP11, CLIP11_MESH


def retarget_joints(joints_t: "np.ndarray", hand: str = "right", mode: str = "dexpilot"):
    """MANO 21 关节点序列 (T,21,3, 任意刚体系) -> (finger_qpos (T,22) SDK 序, joint_names).
    与在线 replay 同一调用链; 手指 retarget 对刚体变换不变."""
    import sys
    sys.path.insert(0, paths.RR_RETARGETING)
    from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
    from magicdexmate.retarget.frames import to_mano
    from magicdexmate.retarget.mapping import JointMapper

    retargeting = build_sharpa_retargeting(hand, mode)
    mapper = JointMapper(retargeting, hand)
    fq = np.zeros((len(joints_t), 22))
    for i in range(len(joints_t)):
        fq[i] = mapper.to_sdk(retargeting.retarget(
            compute_ref_value(retargeting, to_mano(joints_t[i], hand))))
    return fq, list(mapper.sdk_names)


def export(npz_path, mesh_path, out_path, hand="right", mode="dexpilot",
           table_height=0.85, obj_gap=0.01, scene_rot="identity"):
    # magicdexmate 未装进 site-packages, 用 retarget_isaacsim 同一份源码 (pinocchio 系, 与 Kit 无关)
    import sys
    sys.path.insert(0, paths.RR_RETARGETING)
    from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
    from magicdexmate.retarget.frames import to_mano
    from magicdexmate.retarget.mapping import JointMapper

    d = np.load(npz_path, allow_pickle=True)
    T = len(d["frames"])
    joints = d[f"joints_{hand}"].astype(np.float64)
    valid = d[f"valid_{hand}"].astype(bool)
    obj_p = d["obj_pose"][:, :3].astype(np.float64)
    obj_q = d["obj_pose"][:, 3:7].astype(np.float64)   # 新流水线已是 wxyz

    # 与 load_replay 完全相同的填充+对齐 (手指 retarget 对刚体变换不变,
    # 但腕位姿要落在同一个桌面局部系里)
    iv = np.flatnonzero(valid)
    joints = joints[iv[np.abs(np.arange(T)[:, None] - iv[None, :]).argmin(1)]]
    joints_t, _, _, _ = F.align_replay(joints, obj_p, obj_q, mesh_path,
                                       table_height, obj_gap, scene_rot)

    retargeting = build_sharpa_retargeting(hand, mode)
    mapper = JointMapper(retargeting, hand)
    t0 = time.time()
    fq = np.zeros((T, 22))
    for i in range(T):
        fq[i] = mapper.to_sdk(retargeting.retarget(
            compute_ref_value(retargeting, to_mano(joints_t[i], hand))))
    print(f"[export] retarget {T} 帧 ({mode}) 耗时 {time.time() - t0:.1f}s")

    import os
    wrist_q = F.sharpa_base_quat_from_joints(joints_t)
    np.savez(
        out_path,
        source_mtime=os.path.getmtime(npz_path),   # 过期检测: 源文件重生成后必须重导
        finger_qpos=fq.astype(np.float32),
        joint_names=np.array(mapper.sdk_names),
        wrist_pos=joints_t[:, 0].astype(np.float32),
        wrist_quat_wxyz=wrist_q.astype(np.float32),
        valid=valid,
        fps=d["fps"],
        retarget_mode=mode,
    )

    # 摘要: 值域 / NaN / 帧间连续性 (跳变大 = retarget 不稳, 参考轨迹要慎用)
    dq = np.abs(np.diff(fq, axis=0)).max(axis=1)
    print(f"[export] 写入 {out_path}")
    print(f"  qpos 值域: [{fq.min():.3f}, {fq.max():.3f}] rad   NaN: {np.isnan(fq).sum()}")
    print(f"  帧间最大跳变: {dq.max():.3f} rad @ frame {dq.argmax()}->{dq.argmax() + 1} "
          f"(valid段内最大: {dq[valid[:-1] & valid[1:]].max():.3f})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=f"{CLIP11}/replay_world.npz")
    ap.add_argument("--mesh", default=CLIP11_MESH)
    ap.add_argument("--out", default=f"{CLIP11}/ref_qpos.npz")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--mode", default="dexpilot", choices=["vector", "dexpilot"])
    a = ap.parse_args()
    export(a.npz, a.mesh, a.out, a.hand, a.mode)
