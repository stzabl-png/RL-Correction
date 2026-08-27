"""从重建轨迹**推断**物体的静置位姿 (不人工指定方向).

背景 (2026-08-05, screw_unscrew_bottle_cap/27):
重建的物体轨迹在**前 42 帧跟踪失败** —— 报瓶子横躺 (长轴离竖直 74~77°), 而源视频里
瓶子全程立在桌上。`object_confidence` 全是 1.0 的占位符, 这个错误不会被自动发现。

所以不能直接取某一帧的重建位姿, 但也不该人工钉一个"竖直"。这里用**物理约束**推断:

    人手接触前物体静置于桌面  =>  它的姿态必然是该网格的某个**稳定支撑姿态**

于是:
  1. 用 trimesh 算**新资产**的稳定支撑姿态 (瓶身只有"正立"一个)
  2. 逐帧检查重建姿态是否与某个稳定姿态相容 -> 不相容的帧被**自动否决**
     (前 42 帧的横躺不是稳定姿态, 就这样被剔掉, 不需要我判断哪几帧是坏的)
  3. 在通过的最长连续段里取中位位置 + 中位 yaw
  4. z 由"底面贴桌"重新定; xy 走相机锚定 (重建相机 -> 机器人 ZED)

⚠ 稳定姿态用**新资产**算, 不能用重建网格 —— 重建网格是 SAM3D 出的壳, 它的 46 个
  "稳定姿态"全是横躺 (质心/凸包都不对), 拿它当约束会得出反的结论。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RestingPose:
    """推断出的静置位姿 + 全部中间量 (可审计)."""
    pos_env: np.ndarray                 # env 世界系 xyz (z = 底面贴桌后的原点高度)
    quat_wxyz: np.ndarray               # 正立 + 推断出的 yaw
    yaw_deg: float
    frames_used: tuple[int, int]        # 采纳的连续段 [首, 尾]
    n_used: int
    n_rejected: int
    pos_mad_mm: np.ndarray              # 段内位置离散 (中位绝对偏差)
    yaw_mad_deg: float
    stable_tilt_deg: float              # 采纳姿态离"稳定姿态"的角差
    notes: list[str] = field(default_factory=list)


def _stable_axis_dirs(mesh_path: str, axis_local=(0, 0, 1), max_poses: int = 40):
    """网格的稳定支撑姿态 -> 每个姿态下 `axis_local` 的世界方向 (z-up 桌面)."""
    import trimesh
    m = trimesh.load(mesh_path, force="mesh")
    src = m if m.is_watertight else m.convex_hull      # 不水密时用凸包 (稳定性只看外形)
    tf, prob = trimesh.poses.compute_stable_poses(src, n_samples=1, threshold=0.0)
    out = []
    for T, p in zip(tf[:max_poses], prob[:max_poses]):
        out.append((T[:3, :3] @ np.asarray(axis_local, float), float(p)))
    return out


def infer_resting_pose(
    recon_dir: str,
    asset_mesh: str,
    *,
    recon_axis_local=(0, 1, 0),      # 重建网格的长轴 (object_mesh_scaled_final.obj 是 +Y)
    asset_axis_local=(0, 0, 1),      # 新资产的长轴 (bottle_body.obj 是 +Z)
    table_top_z: float = 0.85,
    obj_gap: float = 0.002,
    stable_tol_deg: float = 25.0,
    robot_cam_xy=None,
) -> RestingPose:
    """按"静置 => 必是稳定支撑姿态"从重建轨迹推断物体在 env 里的摆放."""
    from rl_rebuild.correction import frames as F
    from rl_rebuild.correction import place_camera as PC

    wf = np.load(os.path.join(recon_dir, "world_fused.npz"), allow_pickle=True)
    T = np.asarray(wf["object_ob_in_world"], float)          # (F,4,4) gravity_z_up_world
    notes = []

    # ---- ① 新资产的稳定支撑姿态 (只用外形, 与重建无关) ----
    stable = _stable_axis_dirs(asset_mesh, asset_axis_local)
    if not stable:
        raise RuntimeError(f"{asset_mesh}: 算不出稳定支撑姿态")
    tilts = [np.degrees(np.arccos(np.clip(abs(d[2]), 0, 1))) for d, _ in stable]
    notes.append(f"新资产稳定姿态 {len(stable)} 个, 长轴离竖直 "
                 f"{min(tilts):.0f}~{max(tilts):.0f}°")

    # ---- ② 按稳定姿态给帧分组, 取**占主导**的那个假设 ----
    # ⚠ 不能靠概率阈值把横躺姿态砍掉 —— 瓶子躺着也是合法静置姿态, 砍它就是变相人工
    #   指定方向。正确的判别是"这条轨迹里哪个姿态占主导": 物体在整段里静置于同一姿态,
    #   帧数最多的那个就是它; 跟踪失败的片段自然是少数。
    ax = np.array([T[k][:3, :3] @ np.asarray(recon_axis_local, float)
                   for k in range(len(T))])
    frame_tilt = np.degrees(np.arccos(np.clip(np.abs(ax[:, 2]), 0, 1)))

    uniq_tilts = sorted({round(t, 1) for t in tilts})
    groups = {}
    for t in uniq_tilts:
        groups[t] = np.abs(frame_tilt - t) <= stable_tol_deg
    matched = np.any(list(groups.values()), axis=0)

    hyp = sorted(((t, int(m.sum())) for t, m in groups.items()),
                 key=lambda kv: -kv[1])
    notes.append("竞争假设 (稳定姿态 -> 相容帧数): "
                 + ", ".join(f"离竖直{t:.0f}°={n}帧" for t, n in hyp[:4]))
    if not matched.any():
        raise RuntimeError("没有任何一帧与稳定支撑姿态相容 —— 重建姿态整体不可用")
    if len(hyp) > 1 and hyp[1][1] and hyp[0][1] < 2 * hyp[1][1]:
        notes.append(f"⚠ 主导假设({hyp[0][1]}帧)未显著压过次优({hyp[1][1]}帧), 结果存疑")

    ok = groups[hyp[0][0]]
    idx = np.flatnonzero(ok)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    seg = max(runs, key=len)
    notes.append(f"采纳主导姿态 (离竖直 {hyp[0][0]:.0f}°, {hyp[0][1]}/{len(T)} 帧), "
                 f"最长连续段 {seg[0]}-{seg[-1]}")
    bad = np.flatnonzero(~ok)
    if len(bad):
        notes.append(f"其余 {len(bad)} 帧 (如 {bad[0]}-{bad[-1]}) 离竖直 "
                     f"{frame_tilt[bad].min():.0f}~{frame_tilt[bad].max():.0f}°, "
                     f"与主导姿态不符 => 跟踪失败段")

    # ---- ③ 段内中位位置 / 中位 yaw ----
    pos = T[seg][:, :3, 3]
    med = np.median(pos, axis=0)
    mad = np.median(np.abs(pos - med), axis=0)
    yaws = []
    for k in seg:
        v = T[k][:3, :3] @ np.array([1.0, 0, 0])      # 长轴之外的一个局部轴, 投到水平面
        v[2] = 0
        if np.linalg.norm(v) > 1e-6:
            yaws.append(np.degrees(np.arctan2(v[1], v[0])))
    yaws = np.asarray(yaws)
    yaw = float(np.median(yaws))
    yaw_mad = float(np.median(np.abs(yaws - yaw)))

    # ---- ④ xy 相机锚定 + z 贴桌 ----
    cam = PC.load_camera_xy(os.path.join(recon_dir, "object_mesh_scaled_final.obj"))
    if cam is None:
        raise RuntimeError(f"{recon_dir}: world_fused.npz 缺 c2w, 无法相机锚定")
    rc = np.asarray(PC.ZED_NOMINAL[:2] if robot_cam_xy is None else robot_cam_xy, float)
    xy = med[:2] + (rc - cam)
    notes.append(f"相机锚定: 重建相机 xy {np.round(cam,3).tolist()} -> "
                 f"机器人 ZED {np.round(rc,3).tolist()}")

    # 正立 + yaw 的四元数; z 让底面贴桌
    a = np.radians(yaw)
    quat = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
    verts = F.load_obj_verts(asset_mesh)
    from rl_rebuild.correction.kinematics import quat_to_R
    vz = (verts @ quat_to_R(quat).T)[:, 2]
    z = table_top_z + obj_gap - float(vz.min())

    return RestingPose(
        pos_env=np.array([xy[0], xy[1], z]), quat_wxyz=quat, yaw_deg=yaw,
        frames_used=(int(seg[0]), int(seg[-1])), n_used=len(seg),
        n_rejected=int((~ok).sum()), pos_mad_mm=mad * 1000, yaw_mad_deg=yaw_mad,
        stable_tilt_deg=float(np.median(frame_tilt[seg])), notes=notes)


def stack_secondary(body: RestingPose, offset_m: float) -> np.ndarray:
    """附属件 (瓶盖) 摆在主件上: 沿主件长轴偏移 `offset_m`.

    偏移量取自 meta.json 的 ``screw.closed_cap_origin_offset_m`` (0.18) ——
    瓶身高 197mm, 盖高 17mm, 180+17=197 正好与瓶口齐平 (已核对)。
    """
    from rl_rebuild.correction.kinematics import quat_to_R
    axis = quat_to_R(body.quat_wxyz) @ np.array([0.0, 0.0, 1.0])
    return body.pos_env + axis * offset_m
