"""OCIR (ocir-grasp-synthesis) 四类产物 -> DataUnit 适配器.

数据布局 (data/testing/):
  sequences/<clip>/human_demo.npz      ① 重建人手 (相机系: 物体4x4 + 手关节点在物体系)
  sequences/<clip>/object.obj          ② 物体 mesh
  grasp_synthesis/<clip>/grasp_pose_N.json  ③ 优化抓姿 (接触角色/点)
  grasp_traj/<clip>/trajectory.npz     ④ cuRobo 轨迹 (rl_correction_z_up 世界系, 30Hz)

坐标: ④ 已是我们的 z-up/桌高0.85 世界系; ① 经 T_wc = T_w_obj(④t0) ∘ inv(T_c_obj(①t0))
提升到同一世界系 (物体刚体桥接, 已数值验证: 腕高 0.94-1.19m 与 ④ 吻合).

variant:
  "human"  — 残差骨干 = 重建人手 (方案1, 基线组)
  "anchor" — 残差骨干 = cuRobo 已验证轨迹 (方案3, 处理组)
两种 variant 都装填 anchor_*/grasp 字段 (供观测/奖励扩展), 区别只在 track_*/human_finger
指向哪条骨干.
"""
from __future__ import annotations

import json
import os

import numpy as np

from rl_rebuild.correction import frames as F
from rl_rebuild.correction.export_qpos import retarget_joints
from rl_rebuild.correction.schema import (DataUnit, FrameConvention, GraspTarget,
                                          ObjectSemantics, RefTrajectory)

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _pose_to_mat(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = F.quat_to_rotmat(np.asarray(quat_wxyz, dtype=np.float64))
    T[:3, 3] = pos
    return T


def _resample_traj(pos, quat, L):
    return F.resample(pos, L), F.resample(quat, L, kind="quat")


def load_ocir(seq_dir, grasp_json, traj_dir, usd_path="", variant="anchor",
              target_hz=20.0, table_height=0.85, demo_fps=30.0,
              smooth_sigma=2.0, outlier_mult=3.0, cache_dir=None,
              semantics: ObjectSemantics | None = None, verbose=False) -> DataUnit:
    mesh_path = os.path.join(seq_dir, "object.obj")
    hd = np.load(os.path.join(seq_dir, "human_demo.npz"), allow_pickle=True)
    tz = np.load(os.path.join(traj_dir, "trajectory.npz"), allow_pickle=True)
    tj = json.load(open(os.path.join(traj_dir, "trajectory.json")))
    gp = json.load(open(grasp_json))
    assert tj["extra_metadata"]["world_frame"] == "rl_correction_z_up", "轨迹坐标系不符"
    assert str(hd["mano_side"]) == "right", "非右手 demo"

    # ---------- ④ cuRobo 轨迹 -> 世界系锚通道 (30Hz -> target_hz) ----------
    src_hz = 1.0 / tj["dt"]
    T4 = len(tz["segment"])
    L4 = int(round((T4 - 1) * target_hz / src_hz)) + 1
    a_wp, a_wq = _resample_traj(tz["hand_pos_camera"].astype(np.float64),
                                tz["hand_quat_camera"].astype(np.float64), L4)
    a_fq = F.resample(tz["finger_targets"].astype(np.float64), L4)
    # 物体锚通道: trajectory.npz 的规划物体是静止的 (不编码 carry 抬升),
    # 用 isaac_sim 验证回放的"实际"物体轨迹 (前 60 帧为其静置段, 裁掉后与 255 帧对齐)
    ot = np.load(os.path.join(traj_dir, "isaac_sim", "object_track.npz"), allow_pickle=True)
    a_op, a_oq = _resample_traj(ot["position_world"][60:].astype(np.float64),
                                ot["orientation_world_wxyz"][60:].astype(np.float64), L4)
    seg4 = F.resample(tz["segment"], L4, kind="nearest")
    finger_names = list(tj["joint_order"])
    names = list(tj["segment_names"])
    s_idx = {n: np.flatnonzero(seg4 == i) for i, n in enumerate(names)}

    # ---------- ① 人手 demo -> 同一世界系 ----------
    T_c_obj0 = hd["object_pose_camera"][0].astype(np.float64)
    T_w_obj0 = _pose_to_mat(tz["object_pos_camera"][0], tz["object_quat_camera"][0])
    T_wc = T_w_obj0 @ np.linalg.inv(T_c_obj0)
    opc = hd["object_pose_camera"].astype(np.float64)            # (Th,4,4)
    j_cam = (np.einsum("tij,tkj->tki", opc[:, :3, :3], hd["hand_joints_object"].astype(np.float64))
             + opc[:, None, :3, 3])
    j_w = np.einsum("ij,tkj->tki", T_wc[:3, :3], j_cam) + T_wc[:3, 3]
    T_w_obj_t = np.einsum("ij,tjk->tik", T_wc, opc)              # 物体世界轨迹 (demo)
    h_op = T_w_obj_t[:, :3, 3].copy()
    h_oq = np.stack([F.rotmat_to_quat(T_w_obj_t[t, :3, :3]) for t in range(len(opc))])
    valid_h = hd["valid_mask"].astype(bool)

    grasp = _grasp_target(gp, tz, tj, s_idx, seg4, finger_names)
    # cuRobo 预抓取路点: 接近段末(close 起点)的腕位姿 (桌面局部系, 退火引导用)
    close_i = int(s_idx["close"][0]) if len(s_idx.get("close", [])) else 0
    curobo_pregrasp = np.concatenate([a_wp[close_i], a_wq[close_i]]).astype(np.float32)

    if variant == "anchor":
        track_obj = np.concatenate([a_op, a_oq], axis=1)
        track_wrist = np.concatenate([a_wp, a_wq], axis=1)
        human_finger = a_fq
        mano = np.zeros((L4, 21, 3), dtype=np.float64)           # 锚骨干无 mano 点
        valid = np.ones(L4, dtype=bool)
        # 交互段 = close..carry (状态语义: 手指开始闭合到搬运结束)
        inter = (int(s_idx["close"][0]), int(s_idx["carry"][-1])) \
            if len(s_idx.get("close", [])) else (0, L4 - 1)
        L, fps = L4, float(target_hz)
    elif variant == "human":
        Th = len(valid_h)
        # 平滑+离群 (逐帧位姿估计噪声), 再重采样
        h_op_f, drift = F.fix_outliers(h_op, outlier_mult)
        h_op_s = F.smooth_channels(h_op_f, smooth_sigma)
        h_oq_s = F.smooth_quats(h_oq, smooth_sigma)
        j_s = F.smooth_channels(j_w.reshape(Th, -1), 1.5).reshape(Th, 21, 3)
        Lh = int(round((Th - 1) * target_hz / demo_fps)) + 1
        j_r = F.resample(j_s, Lh)
        h_op_r, h_oq_r = F.resample(h_op_s, Lh), F.resample(h_oq_s, Lh, kind="quat")
        valid = F.resample(valid_h, Lh, kind="nearest")
        # retarget (pinocchio) 与 Kit 冲突, 只能离线跑 -> 结果缓存, Kit 内读缓存.
        # cache_dir 按物体隔离 (TrainingData 布局下 seq_dir basename 会串号)
        cdir = cache_dir or os.path.join(
            "/home/lyh/Project/RL_Correction/data/ocir_cache", os.path.basename(seq_dir))
        cache = os.path.join(cdir, f"human_fq_{int(target_hz)}hz.npz")
        src_mtime = os.path.getmtime(os.path.join(seq_dir, "human_demo.npz"))
        if os.path.exists(cache) and abs(float(np.load(cache)["src_mtime"]) - src_mtime) < 1.0:
            c = np.load(cache, allow_pickle=True)
            fq, fq_names = c["fq"].astype(np.float64), [str(n) for n in c["names"]]
        else:
            fq, fq_names = retarget_joints(j_r)   # Kit 内到这行会报错: 先离线跑一次 loader
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache, fq=fq, names=np.array(fq_names), src_mtime=src_mtime)
        wrist_q = F.sharpa_base_quat_from_joints(j_r)
        track_obj = np.concatenate([h_op_r, h_oq_r], axis=1)
        track_wrist = np.concatenate([j_r[:, 0], wrist_q], axis=1)
        human_finger, finger_names = fq, fq_names
        mano = j_r
        lo, hi = F.object_aabb_obj(mesh_path)
        half_diag = np.linalg.norm(hi - lo) / 2
        # demo 末段是"放回桌面"; 任务=抓起举住 -> 裁剪到抬升峰值帧 (hold 段接管其后)
        pk = int(np.argmax(h_op_r[:, 2])) + 2
        pk = min(max(pk, 8), Lh)
        j_r, h_op_r, h_oq_r, fq, wrist_q, valid = (
            j_r[:pk], h_op_r[:pk], h_oq_r[:pk], fq[:pk], wrist_q[:pk], valid[:pk])
        track_obj = np.concatenate([h_op_r, h_oq_r], axis=1)
        track_wrist = np.concatenate([j_r[:, 0], wrist_q], axis=1)
        human_finger, mano, Lh = fq, j_r, pk
        tip_d = np.linalg.norm(j_r[:, F.MANO_TIPS] - h_op_r[:, None], axis=2).min(axis=1)
        im = valid & (tip_d < half_diag + 0.03)
        idx = np.flatnonzero(im)
        inter = (int(idx[0]), int(idx[-1])) if len(idx) else (0, Lh - 1)
        L, fps = Lh, float(target_hz)
        if verbose:
            print(f"[ocir/human] 剔除漂移帧 {drift.tolist()}  L={Lh}")
    else:
        raise ValueError(variant)

    ref = RefTrajectory(
        track_object=track_obj.astype(np.float32),
        track_wrist=track_wrist.astype(np.float32),
        mano_joints=mano.astype(np.float32),
        valid=valid,
        valid_seg=(int(np.flatnonzero(valid)[0]), int(np.flatnonzero(valid)[-1])),
        interaction_seg=inter,
        anchor_wrist=np.concatenate([a_wp, a_wq], axis=1).astype(np.float32),
        anchor_finger=a_fq.astype(np.float32),
        curobo_pregrasp=curobo_pregrasp,
        human_finger=human_finger.astype(np.float32),
        finger_names=finger_names,
        grasp=grasp,
    )
    # 初始物体位姿: OCIR 的摆位已在其物理验证中确认稳定 —— 信任原姿态,
    # 只做贴底 z 微调 (最低顶点 = 桌面 + 2mm). 不做 stable-pose 旋转投影
    # (实测投影会把已稳定的姿态翻 88°, 与锚轨迹脱节).
    t0 = ref.valid_seg[0]
    verts = F.load_obj_verts(mesh_path)
    init_q = track_obj[t0, 3:7].astype(np.float64)
    v_rot = F.rot_apply(np.broadcast_to(init_q, (len(verts), 4)), verts)
    z_snap = table_height + 0.002 - v_rot[:, 2].min()
    init_pose = np.concatenate([[track_obj[t0, 0], track_obj[t0, 1], z_snap], init_q])
    if verbose:
        print(f"[ocir/{variant}] L={L} 交互段={inter} 贴底调整 "
              f"{(z_snap - track_obj[t0, 2]) * 100:+.1f}cm")

    return DataUnit(
        clip_id=f"ocir/{os.path.basename(seq_dir)}/{variant}",
        mesh_path=mesh_path, usd_path=usd_path,
        object_init_pose=init_pose.astype(np.float32),
        goal_object_pose=track_obj[-1].astype(np.float32),
        ref=ref, fps=fps,
        frame=FrameConvention(table_height=table_height),
        semantics=semantics or ObjectSemantics(label="pp0 object", mass_kg=0.2, friction=0.5),
    )


def _grasp_target(gp, tz, tj, s_idx, seg4, finger_names) -> GraspTarget:
    """GraspTarget: 手指目标取 grasp json 的纯优化抓姿 (与 cuRobo 解耦, 消融可分),
    退化时用 cuRobo squeeze 末帧; 接触来自 json roles."""
    sq = int(s_idx["squeeze"][-1]) if len(s_idx.get("squeeze", [])) else len(seg4) - 1
    # squeeze 末帧在原始 30Hz 轨迹里的对应下标
    ratio = (len(tz["segment"]) - 1) / max(len(seg4) - 1, 1)
    sq_src = int(round(sq * ratio))
    roles = set(gp.get("active_contact_roles", []))
    contact5 = np.array([any(r.startswith(f) for r in roles) for f in FINGERS])
    opt = np.asarray(gp.get("optimized_action", []), dtype=np.float32)
    opt_names = gp.get("optimized_joint_names")
    if len(opt) >= 29 and opt_names:            # [pos3, quat4, 22指] in optimized_joint_names 序
        g_finger, g_names = opt[7:29].copy(), list(opt_names)
    else:                                        # 退化: cuRobo squeeze 手型
        g_finger, g_names = tz["finger_targets"][sq_src].astype(np.float32), list(finger_names)
    return GraspTarget(
        finger_q=g_finger,
        wrist_pose=np.concatenate([tz["hand_pos_camera"][sq_src],
                                   tz["hand_quat_camera"][sq_src]]).astype(np.float32),
        contact_fingers=contact5,
        finger_names=g_names,
        grasp_phase_frame=sq,
    )
