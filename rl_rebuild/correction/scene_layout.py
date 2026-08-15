"""通用场景摆放 —— **每个物体各自听自己的那只手**。

============================ 约定从哪来 ============================

`ref_builders/replay_grasp.py` 第 2 步已经定死了单物体的摆法:

    物体听手: XY = 交互开始帧手的抓取锚点; Z 稳定贴桌; 物体自身 track 弃用
    (噪声大, 且交互期被手遮挡)

本模块把它推广到 **N 个物体**, 规则不变, 只是"哪只手 / 哪一帧"改成**逐物体**判定:

    对每个物体 o:
        (hand_o, gs_o) = o 的最早接触区间(来自 contact_auto_<o>.json)
        XY = hand_o 在 gs_o 帧的抓取锚点(五指尖质心)
        Z  = 该物体最大支撑面贴桌

============================ 为什么必须逐物体 ============================

实测 clip 0 (screw_unscrew_bottle_cap):

| 物体 | 接触手 | 起始帧 | conf_rot | 被证伪帧 |
|---|---|---|---|---|
| 瓶身 object_0 | 左 | f3 | 34(可用) | 1/133 |
| 瓶盖 object_1 | 右 | f10 | **0(弃用)** | **104/133** |

两个物体由**不同的手**在**不同的时刻**接触; 而瓶盖自身的轨迹 104/133 帧被证伪 ——
正是"物体 track 不可信、手可信"这条约定要处理的情形。用单物体逻辑会把两者混为一谈。

⚠ 只做**摆放**(初始位姿)。任务目标(什么算成功)不在这里 —— 对"把盖拧回瓶子"这类
装配任务, 目标是**盖相对瓶身**的位姿而不是两个绝对位姿, 那是另一件事。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TABLE_HEIGHT = 0.85          # 与 replay_grasp / retarget_isaacsim 同一常数
OBJ_GAP = 0.01               # 物体底面与桌面的间隙; 0 会穿透薄桌板
FINGERTIPS = [4, 8, 12, 16, 20]      # OpenPose-21 五指尖


def _support_rest_quat(verts: np.ndarray) -> np.ndarray:
    """把最大支撑面压到桌面 → rest 四元数(wxyz)。凸包面积加权找最大面。"""
    try:
        import trimesh
        h = trimesh.convex.convex_hull(trimesh.PointCloud(verts))
        areas, normals = h.area_faces, h.face_normals
        # 把法向相近的面归并, 取总面积最大的那组
        keys = np.round(normals, 2)
        best_n, best_a = None, -1.0
        for k in np.unique(keys, axis=0):
            m = (keys == k).all(1)
            a = float(areas[m].sum())
            if a > best_a:
                best_a, best_n = a, normals[m][0]
    except Exception:
        best_n = None
    if best_n is None:
        return np.array([1.0, 0.0, 0.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    axis = np.cross(best_n, down)
    s, c = float(np.linalg.norm(axis)), float(best_n @ down)
    if s < 1e-8:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    ang = float(np.arctan2(s, c))
    axis = axis / s
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 -> wxyz。"""
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def _main_axis(V: np.ndarray) -> tuple[np.ndarray, float]:
    """网格主轴(局部系单位向量)+ 细长比。细长比 ≈1 表示没有有意义的主轴。"""
    ext = V.ptp(0)
    i = int(np.argmax(ext))
    ax = np.zeros(3)
    ax[i] = 1.0
    other = float(max(e for j, e in enumerate(ext) if j != i))
    return ax, float(ext[i] / max(other, 1e-9))


def _rest_quat(mesh, up_tol_deg: float = 30.0,
               recon_R: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """初始姿态裁定: **物理给候选, 重建挑一个**。

        稳定静置姿态候选 -> 筛掉侧躺(细长物体) -> 若重建旋转可信, 取与之最接近的
        候选; 否则取概率最高。

    2026-08-15 改动(用户裁定): 上游重建质量提升后不再发生倒置(pour17 新版
    conf_rot 68.5/52.5 且 rotation_usable=True), **射线开口检测已删除** ——
    它只覆盖"有内腔"的物体, 通用性不足。现在由重建旋转在物理候选里挑,
    重建不可信时退回概率最高。判不出来时退回旧规则(最大支撑面贴桌)。

    recon_R: 静置帧的重建旋转矩阵 (3,3); None 表示不可用。
    返回 (quat_wxyz, 理由)。
    """
    try:
        import trimesh
        V = np.asarray(mesh.vertices)
        axis, elong = _main_axis(V)
        T, P = trimesh.poses.compute_stable_poses(mesh, n_samples=6, threshold=0.005)
        if len(T) == 0:
            raise RuntimeError("无稳定姿态")
        cands = [(T[k][:3, :3], float(P[k])) for k in range(len(T))]
        note = [f"稳定候选{len(cands)}"]

        if elong > 1.25:                      # 有明确主轴才谈"立/躺"
            cos_tol = float(np.cos(np.radians(up_tol_deg)))
            up = [c for c in cands if abs(float((c[0] @ axis)[2])) > cos_tol]
            if up:
                cands = up
                note.append(f"筛直立(细长比{elong:.2f})->{len(cands)}")

        if recon_R is not None and len(cands) > 1:
            # ★ 只比**主轴指向**, 不比整体旋转 —— 回转体绕自身轴的自转是**自由**的
            #   (瓶子立着但自转 90°, 测地距 90° 而几何完全相同; 实测就踩到这个坑)。
            #   这与可信度契约的 rotation_free_axes 是同一条原则(P3 不可观测自由度)。
            a_ref = recon_R @ axis
            a_ref = a_ref / (np.linalg.norm(a_ref) + 1e-9)

            def _axis_ang(Rc):
                a = Rc @ axis
                a = a / (np.linalg.norm(a) + 1e-9)
                return float(np.degrees(np.arccos(abs(float(np.clip(a @ a_ref, -1, 1))))))
            R, p = min(cands, key=lambda c: _axis_ang(c[0]))
            note.append(f"按重建**主轴指向**挑最近候选(轴夹角 {_axis_ang(R):.1f}°, 概率 {p:.3f})")
        else:
            R, p = max(cands, key=lambda c: c[1])
            note.append(f"取概率最高 {p:.3f}" + ("" if recon_R is None else "(候选唯一)"))
        return _mat_to_quat(R), " | ".join(note)
    except Exception as e:
        return _support_rest_quat(np.asarray(mesh.vertices)), f"退回最大支撑面({type(e).__name__})"


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return v @ R.T


def contact_onsets(recon_dir: Path) -> dict[str, tuple[str, int]]:
    """→ {object_id: (hand, onset_frame)}，取每个物体**最早**的接触区间。

    数据源是逐物体的 `contact_auto_<oid>.json` —— **不能用并集** `contact_auto.json`:
    并集会把"右手在摸瓶盖"也算成"右手在摸瓶身"(clip 0 实测右手对并集是 f10 起,
    对瓶身其实是 f47 起, 差 37 帧)。
    """
    out: dict[str, tuple[str, int]] = {}
    for f in sorted(recon_dir.glob("contact_auto_object_*.json")):
        d = json.loads(f.read_text())
        oid = d.get("object_id") or f.stem.replace("contact_auto_", "")
        best = None
        for hand in ("left", "right"):
            segs = (d.get("annotations") or {}).get(hand) or []
            if segs and (best is None or segs[0][0] < best[1]):
                best = (hand, int(segs[0][0]))
        if best:
            out[oid] = best
    return out


def _motion_onset(P: np.ndarray, c0: int, win: int = 3) -> tuple[int, str]:
    """接触区间起点 -> **抓稳帧**精化: 物体开始运动的那一帧。

    立论: "碰到" ≠ "抓稳"。2D 邻接在真接触前就触发(screw18 实测早 10 帧;
    pour17 的杯早 28 帧 = 摆放锚点偏 4.2cm)。而**物体开始动**是抓稳的运动学铁证。
    实测精度: screw18 给 14 vs 人工真值 13(差 1 帧)。

    P: 该物体逐帧世界位置 (T,3);c0: 接触区间起点。→ (帧号, 理由)
    """
    T = len(P)
    if T < win + 4:
        return c0, "帧数不足, 用接触起点"
    d = np.linalg.norm(P[win:] - P[:-win], axis=1)
    # 静止基线取**前段的低分位**而不是中位数: 中位数会被早期抖动抬高(pour17 重建更新后
    # 前 8 帧混进 0.8~1.1cm 的噪声, 基线从 0.13 抬到 0.37cm, 阈值随之虚高)。
    base = float(np.percentile(d[:max(12, win * 3)], 25))
    thr = max(0.008, base * 3.0)
    # ★ 必须**持续 K 帧**超阈才算"开始动" —— 单帧越阈会被噪声骗(实测早 13 帧)。
    K = 3
    ok = d > thr
    t = None
    for s in range(max(0, c0 - 2), len(d) - K):
        if ok[s:s + K].all():
            t = int(s)
            break
    if t is None:
        return c0, f"未检出持续运动(阈{thr*100:.1f}cm×{K}帧), 用接触起点"
    return t, (f"运动起始 f{t}(接触起点 f{c0}, 阈 {thr*100:.1f}cm/3帧, 持续{K}帧)"
               if t != c0 else f"运动起始=接触起点 f{c0}")


def layout(recon_dir: Path, replay_npz: Path, *, table_height: float = TABLE_HEIGHT,
           obj_gap: float = OBJ_GAP) -> dict:
    """逐物体算初始摆放。→ {object_id: {pos, quat, hand, onset, ...}} + 诊断。"""
    import trimesh
    r = np.load(replay_npz, allow_pickle=True)
    joints = {"left": r["joints_left"], "right": r["joints_right"]}
    onsets = contact_onsets(recon_dir)
    if not onsets:
        raise RuntimeError(f"{recon_dir}: 没有 contact_auto_object_*.json —— "
                           f"先跑 contact 步骤, 摆放要靠它决定每个物体听哪只手")

    oids = [str(x) for x in r["object_ids"]] if "object_ids" in r.files else ["object_0"]

    # ★ 桌面高度必须**从数据量**, 不能直接用 RL 的 0.85 常数。
    #   手的关节在重建世界系里(clip 0 的桌面实测在 1.30m); 把物体摆到 0.85 而手不动,
    #   接触起始帧手离物体会差 45cm(实测 319mm/464mm)。
    #   量法: 取最可信物体在它接触起始帧的最低顶点 —— 接触刚开始时它还稳稳放在桌上。
    w = np.load(recon_dir / "world_fused.npz", allow_pickle=True)
    Tw = w["object_ob_in_world_all"] if "object_ob_in_world_all" in w.files else None
    conf = json.loads((recon_dir / "confidence_complete.json").read_text()) \
        if (recon_dir / "confidence_complete.json").is_file() else {}
    co = conf.get("objects") or {}
    ranked = sorted(oids, key=lambda o: -(co.get(o, {}).get("conf_pos_median") or 0))
    scene_table_z, table_src = None, "无法测量"
    for oid in ranked:
        if oid not in onsets or Tw is None:
            continue
        i = oids.index(oid)
        mp = recon_dir / "objects" / oid / "object_mesh_scaled_final.obj"
        if not mp.is_file():
            mp = recon_dir / "object_mesh_scaled_final.obj"
        Vo = np.asarray(trimesh.load(mp, process=False, force="mesh").vertices)
        gs = int(np.clip(onsets[oid][1], 0, Tw.shape[1] - 1))
        # ⚠ 用**1% 分位**而不是最低顶点: SAM3D 网格常有个别离群刺, 一个刺就能把桌面拉低
        #   (pour17 实测: 减面前后差 1.1cm, 就是一根刺被减掉了)。再对静置段取中位数抗噪。
        zs = []
        for t in range(max(0, gs - 5), gs + 1):
            M = Tw[i, int(np.clip(t, 0, Tw.shape[1] - 1))]
            zs.append(float(np.percentile(((M[:3, :3] @ Vo.T).T + M[:3, 3])[:, 2], 1.0)))
        scene_table_z = float(np.median(zs))
        table_src = (f"{oid}(conf_pos={co.get(oid, {}).get('conf_pos_median')}) "
                     f"在 f{max(0, gs-5)}~f{gs} 的顶点 1% 分位中位数")
        break
    if scene_table_z is None:
        scene_table_z = table_height
        table_src = "量不到, 退回 RL 常数"

    out, diag = {}, []
    for i, oid in enumerate(oids):
        if oid not in onsets:
            diag.append(f"{oid}: 无接触区间, 跳过(它没被碰过, 不该由手来锚)")
            continue
        hand, c0 = onsets[oid]
        J = joints[hand]
        # ★ 锚帧用"物体开始运动"精化, 不用接触区间起点(碰到 ≠ 抓稳)
        if Tw is not None and i < Tw.shape[0]:
            gs, gs_why = _motion_onset(np.asarray(Tw[i, :, :3, 3], dtype=float), int(c0))
        else:
            gs, gs_why = int(c0), "无物体轨迹, 用接触起点"
        gs_c = int(np.clip(gs, 0, len(J) - 1))
        anchor = J[gs_c][FINGERTIPS].mean(0)            # 五指尖质心 = 抓取锚点

        mesh_p = recon_dir / "objects" / oid / "object_mesh_scaled_final.obj"
        if not mesh_p.is_file():
            mesh_p = recon_dir / "object_mesh_scaled_final.obj"
        _mesh = trimesh.load(mesh_p, process=False, force="mesh")
        V = np.asarray(_mesh.vertices)
        # 重建旋转(静置段均值)—— 仅当该物体 rotation_usable 才交给 _rest_quat 挑候选
        _recon_R = None
        if Tw is not None and i < Tw.shape[0] and co.get(oid, {}).get("rotation_usable"):
            _recon_R = np.asarray(Tw[i, max(0, gs - 5):max(1, gs), :3, :3], float).mean(0)
            _u, _, _vt = np.linalg.svd(_recon_R)         # 正交化(均值不再是旋转阵)
            _recon_R = _u @ _vt
        q, q_why = _rest_quat(_mesh, recon_R=_recon_R)
        Vr = _rot(q, V)
        z = scene_table_z + obj_gap - float(Vr[:, 2].min())
        # 抓取目标点(无 affordance 时退化到质心)对准手锚点
        off = Vr.mean(0)
        pos = np.array([anchor[0] - off[0], anchor[1] - off[1], z])
        out[oid] = {"pos": pos.tolist(), "quat_wxyz": q.tolist(),
                    "anchor_hand": hand, "onset_frame": gs,
                    "anchor_xy": anchor[:2].tolist(), "quat_reason": q_why,
                    "contact_start": int(c0), "onset_reason": gs_why,
                    "mesh": str(mesh_p), "extent_cm": (Vr.ptp(0) * 100).round(1).tolist()}
        diag.append(f"{oid}: 听{'左' if hand == 'left' else '右'}手 @f{gs}  "
                    f"XY=({pos[0]:.3f},{pos[1]:.3f})  Z={z:.3f}  "
                    f"尺寸 {np.round(Vr.ptp(0) * 100, 1).tolist()}cm\n"
                    f"          锚帧: {gs_why}\n"
                    f"          姿态: {q_why}")
    diag.insert(0, f"场景桌面高度 {scene_table_z:.3f}m (量自 {table_src}); "
                   f"RL 桌面 {table_height:.2f}m -> 整场景需下移 {scene_table_z - table_height:.3f}m")
    return {"schema_version": "scene_layout_v1", "rl_table_height": table_height,
            "scene_table_z": scene_table_z, "scene_table_source": table_src,
            "scene_to_rl_shift_z": scene_table_z - table_height,
            "obj_gap": obj_gap, "objects": out, "diagnostics": diag,
            "rule": "每个物体听自己的那只手: XY=该手在该物体**运动起始帧**(抓稳)的五指尖质心; Z=贴桌; 姿态=稳定候选->筛直立->筛开口朝上->取概率最高(不用重建旋转)",
            "source_contact": "contact_auto_object_*.json (逐物体, 非并集)"}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon-dir", type=Path, required=True)
    ap.add_argument("--replay", type=Path, default=None,
                    help="replay_world.npz (默认 recon-dir 同级的 RetargetOutput)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--table-height", type=float, default=TABLE_HEIGHT)
    a = ap.parse_args(argv)
    replay = a.replay or Path(str(a.recon_dir).replace("ReconstructOutput",
                                                       "RetargetOutput")) / "replay_world.npz"
    d = layout(a.recon_dir, replay, table_height=a.table_height)
    for line in d["diagnostics"]:
        print("  " + line)
    out = a.out or (a.recon_dir / "scene_layout.json")
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
