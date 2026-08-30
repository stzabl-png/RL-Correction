"""replay_world.npz -> DataUnit (MVP 加载器).

预处理链: 读取 -> xyzw->wxyz -> 无效帧最近邻填充 -> cv2zup+落桌对齐
       -> 物体轨迹离群剔除+高斯平滑 -> (可选) 重采样到控制频率 -> 装配.

用法 (任意带 numpy 的 python, 不启动 Isaac):
  python -m rl_rebuild.correction.load_replay          # clip-11 默认路径, 打印摘要
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_rebuild.correction import frames as F
from rl_rebuild.correction import paths
from rl_rebuild.correction.schema import (DataUnit, FrameConvention,
                                          ObjectSemantics, RefTrajectory)

CLIP11 = f"{paths.RR_OUTPUT}/RetargetOutput/egodex/part2/basic_pick_place/11"
CLIP11_MESH = (f"{paths.RR_OUTPUT}/ReconstructOutput/egodex/part2/"
               "basic_pick_place/11/object_mesh_scaled_final.obj")
CLIP11_SEMANTICS = ObjectSemantics(label="plastic cylinder", mass_kg=0.2, friction=0.5,
                                   mass_range=(0.1, 0.4), friction_range=(0.4, 0.7),
                                   source="human")


def _longest_true_run(mask):
    """最长连续 True 段 -> (起, 止) 含端点."""
    best, cur = (0, -1), None
    for i, m in enumerate(mask):
        if m:
            cur = i if cur is None else cur
            if i - cur > best[1] - best[0]:
                best = (cur, i)
        else:
            cur = None
    return best


def load(npz_path, mesh_path, usd_path="", clip_id="", hand="right",
         table_height=0.85, obj_gap=0.01, scene_rot="identity", quat_order="wxyz",
         smooth_sigma=2.0, outlier_mult=3.0, target_hz=None,
         semantics: ObjectSemantics | None = None, verbose=False,
         initial_pose_mode="stable") -> DataUnit:
    d = np.load(npz_path, allow_pickle=True)
    T = len(d["frames"])
    src_fps = float(d["fps"])
    joints = d[f"joints_{hand}"].astype(np.float64)               # (T,21,3)
    valid = d[f"valid_{hand}"].astype(bool)
    obj_p = d["obj_pose"][:, :3].astype(np.float64)
    # 新流水线 (hawor_to_world_replay) 的 obj_pose 是 [x,y,z, qw,qx,qy,qz] —
    # 已是 wxyz, 直接读! (旧 world_fused.npz 才是 xyzw, 用 quat_order='xyzw')
    obj_q = d["obj_pose"][:, 3:7].astype(np.float64)
    if quat_order == "xyzw":
        obj_q = F.quat_xyzw_to_wxyz(obj_q)

    # 离线 retarget 产物 (export_qpos.py), 与源同帧率.
    # 双手 bundle (如 screw_unscrew_bottle_cap_1) 按手拆成 ref_qpos_{left,right}.npz,
    # 单手 bundle 只有 ref_qpos.npz —— 先按手找, 没有再回落 (旧数据行为不变).
    finger, finger_names = None, None
    qpos_path = Path(npz_path).parent / f"ref_qpos_{hand}.npz"
    if not qpos_path.exists():
        qpos_path = Path(npz_path).parent / "ref_qpos.npz"
    if qpos_path.exists():
        q = np.load(qpos_path, allow_pickle=True)
        assert len(q["finger_qpos"]) == T, "ref_qpos 与 replay 帧数不一致, 请重新导出"
        # 自带快照 (datasets/) 跳过 mtime 判定: 两份文件是一起提交的必然配套,
        # 而 git 不保留 mtime, 不跳过的话别人 clone 完这条断言必然误报. 见 paths.is_bundled.
        if "source_mtime" in q.files and not paths.is_bundled(npz_path):
            import os
            assert abs(float(q["source_mtime"]) - os.path.getmtime(npz_path)) < 1.0, \
                f"ref_qpos 过期 (源 {npz_path} 已重新生成), 先跑 export_qpos.py"
        finger = q["finger_qpos"].astype(np.float64)
        finger_names = [str(n) for n in q["joint_names"]]

    # 无效帧最近邻填充 (与 retarget_isaacsim 同策略)
    iv = np.flatnonzero(valid)
    fill = iv[np.abs(np.arange(T)[:, None] - iv[None, :]).argmin(1)]
    joints = joints[fill]

    joints_t, obj_p_t, obj_q_t, info = F.align_replay(
        joints, obj_p, obj_q, mesh_path, table_height, obj_gap, scene_rot)

    # ---- 物体轨迹: 离群剔除 + 高斯平滑 (逐帧位姿估计必有抖动/漂移) ----
    obj_p_fixed, drift = F.fix_outliers(obj_p_t, outlier_mult)
    obj_p_s = F.smooth_channels(obj_p_fixed, smooth_sigma)
    obj_q_s = F.smooth_quats(obj_q_t, smooth_sigma)
    joints_s = F.smooth_channels(joints_t.reshape(T, -1), sigma=1.5).reshape(T, 21, 3)
    max_corr = np.linalg.norm(obj_p_s - obj_p_t, axis=1).max()
    if verbose:
        print(f"[smooth] 剔除漂移帧 {drift.tolist()}   物体位置最大修正 {max_corr*100:.2f} cm")
    if max_corr > 0.05:
        print(f"[smooth][WARN] 平滑修正量 {max_corr*100:.1f} cm > 5cm, 数据质量可疑, 建议肉眼检查")

    # ---- (可选) 重采样到控制频率: 位置线性 / 四元数 slerp / valid 最近邻 ----
    fps = src_fps
    if target_hz is not None and target_hz != src_fps:
        L = int(round((T - 1) * target_hz / src_fps)) + 1
        joints_s = F.resample(joints_s, L)
        obj_p_s = F.resample(obj_p_s, L)
        obj_q_s = F.resample(obj_q_s, L, kind="quat")
        valid = F.resample(valid, L, kind="nearest")
        if finger is not None:
            finger = F.resample(finger, L)
        fps = float(target_hz)

    # 漂移帧号从**源时钟**映射到最终时钟 (重采样后帧号会变, 不映射就指向错的帧).
    # 这些帧的物体位置是插值编出来的, 不是观测到的 —— 逐帧置信度的原料之一.
    # ⚠ 必须走和 valid 同一套 nearest 重采样, 不能对帧号做四舍五入:
    #   源 [39,40,41] 连续段四舍五入后是 [52,53,55], 中间漏掉 54 —— 连续坏段被打出洞,
    #   而"连续段"恰恰是判断参考可不可信最重要的结构.
    _dm = np.zeros(T, dtype=bool)
    _dm[drift] = True
    if target_hz is not None and target_hz != src_fps:
        _dm = F.resample(_dm, len(obj_p_s), kind="nearest").astype(bool)
    drift_out = np.flatnonzero(_dm)

    # ---- 派生量在最终时间轴上重算 ----
    wrist_q = F.sharpa_base_quat_from_joints(joints_s)
    tip_obj = np.linalg.norm(
        joints_s[:, F.MANO_TIPS] - obj_p_s[:, None], axis=2).min(axis=1)
    interaction = valid & (tip_obj < info["half_diag"] + 0.03)
    track_object = np.concatenate([obj_p_s, obj_q_s], axis=1)
    track_wrist = np.concatenate([joints_s[:, 0], wrist_q], axis=1)

    ref = RefTrajectory(
        track_object=track_object.astype(np.float32),
        track_wrist=track_wrist.astype(np.float32),
        mano_joints=joints_s.astype(np.float32),
        valid=valid,
        valid_seg=_longest_true_run(valid),
        interaction_seg=_longest_true_run(interaction),
        human_finger=None if finger is None else finger.astype(np.float32),
        finger_names=finger_names,
        obj_drift=drift_out,
    )
    # 初始物体位姿. 旧 clip 默认做 stable-pose 投影; static reconstruction
    # 显式使用 preserve, 因为其 FoundationPose 朝向是场景布置的权威输入.
    t_init = ref.valid_seg[0]
    mesh_verts = F.load_obj_verts(mesh_path)
    if initial_pose_mode == "stable":
        init_q, fix_deg = F.stable_pose_projection(
            mesh_verts, track_object[t_init, 3:7].astype(np.float64))
    elif initial_pose_mode == "preserve":
        init_q = track_object[t_init, 3:7].astype(np.float64)
        init_q = init_q / np.linalg.norm(init_q)
        fix_deg = 0.0
    else:
        raise ValueError(
            f"initial_pose_mode must be stable/preserve, got {initial_pose_mode!r}"
        )
    v_rot = F.rot_apply(np.broadcast_to(init_q, (len(mesh_verts), 4)), mesh_verts)
    init_pose = np.concatenate([
        [track_object[t_init, 0], track_object[t_init, 1],
         table_height + obj_gap - v_rot[:, 2].min()], init_q])
    if verbose and initial_pose_mode == "stable":
        print(f"[init] stable-pose 投影: 修正倾斜 {fix_deg:.1f} deg (t_init={t_init})")

    data_unit = DataUnit(
        clip_id=clip_id, mesh_path=str(mesh_path), usd_path=str(usd_path),
        object_init_pose=init_pose.astype(np.float32),
        goal_object_pose=track_object[-1].astype(np.float32),
        ref=ref, fps=fps,
        frame=FrameConvention(table_height=table_height),
        semantics=semantics or ObjectSemantics(),
    )
    d.close()
    return data_unit


if __name__ == "__main__":
    for hz, tag in [(None, "源 15fps"), (20.0, "重采样 20Hz")]:
        du = load(f"{CLIP11}/replay_world.npz", CLIP11_MESH,
                  usd_path=f"{CLIP11}/object.usd",
                  clip_id="egodex/part2/basic_pick_place/11",
                  target_hz=hz, semantics=CLIP11_SEMANTICS, verbose=True)
        r = du.ref
        disp = np.linalg.norm(du.goal_object_pose[:3] - du.object_init_pose[:3])
        lift = r.track_object[:, 2].max() - r.track_object[0, 2]
        hf = "None" if r.human_finger is None else r.human_finger.shape
        print(f"[{tag}] L={r.L} @ {du.fps:g}fps  valid={r.valid_seg}  "
              f"交互={r.interaction_seg}  位移={disp:.3f}m  抬升={lift*100:.1f}cm  "
              f"finger={hf}  物体={du.semantics.label} {du.semantics.mass_kg}kg\n")
