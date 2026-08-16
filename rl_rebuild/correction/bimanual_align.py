"""重建轨迹 -> 机器人训练环境的**通用**摆放求解 + 质量门.

设计原则:一切锚在**接触通道**上,不看手的身份、不用无接触约束的手。

  一只手的位置可不可信, 取决于**它有没有接触约束**, 而不是它动不动。
    - 接触中的手:   物体给了物理锚点 -> 绝对位置可信
    - 自由悬空的手: 只有重建自身精度撑着, 会漂 -> "它没动"可信, "它在哪"不可信

  实测 (egodex part2/basic_pick_place/2): 左手全程无接触, 原视频里它本来就不动
  (这一点重建还原对了), 但**绝对位置被整体推近右手**, 导致"左->右手轴"与"物体
  搬运方向"夹角只有 4.8°(几何上应 ~90°). 所以**双手轴不能用来定朝向** ——
  除非两只手都在接触(真双手任务, 那时两只手都被物理锚定, 反而更好办).

摆放五步:
  1. yaw     **通常 = 0** —— 重建管线已用相机前向把世界 XY 对齐过
             (world_fused.npz: mode=middle_camera_forward_projected_xy),
             所以重建的 +X 就是人的正前方, 机器人也朝 +X, 天然对上.
             只有元数据缺失/不符时才退回"物体搬运方向"估计.
  2. XY 平移 交互手掌心(抓取帧) -> 机器人侧目标点 (由 phase_* 自动分派单手/双手)
  3. Z 平移  让抓取帧掌心距物体 = grasp_gap (默认 3.5cm, 落在实测可抓区间 2.5~4.5)
  4. 物体姿态 与重建姿态测地距最小的**稳定静置姿态** (trimesh 枚举, 带磁盘缓存)
  5. 物体位置 = 交互手掌心 XY + 贴桌 Z  (re-anchor, 消掉重建的手-物误差)

非交互手: **不做参考跟踪**, 机器人对应臂保持默认姿态。重建说"它没动"是可信的,
但"它在哪"不可信, 所以只保留语义、不继承位置误差。
"""
from __future__ import annotations

import json
import os

import numpy as np

from rl_rebuild.correction import frames as F
from rl_rebuild.correction import paths

# 单手任务时目标点相对该手默认位向前(+X)推的距离. 对所有 clip 同一个值, 非 per-clip 调参.
REACH_FORWARD = 0.15
GRASP_GAP = 0.035          # 抓取帧掌心距物体 (实测可抓区间 2.5~4.5cm 的中值)
OBJ_ANCHOR = "fingertips"   # 物体 re-anchor 到手的哪个点 (见 Step 5)
PALM_OFFSET = 0.12         # 掌心 = 腕 + 此值 · 手系+z (与 replay_grasp 同约定)
MIN_DIR_LEN = 0.05         # 方向向量最短长度; 短于此认为方向不可靠, 走 fallback


def detect_interaction(npz_path):
    """从 phase_* 读出交互结构 —— 全部自动分派的依据 (单手/双手/哪只手)."""
    d = np.load(npz_path, allow_pickle=True)
    out = {"hands": [], "grasp_frame": None, "release_frame": None, "per_hand": {}}
    for h in ("left", "right"):
        key = f"phase_{h}"
        if key not in d.files:
            continue
        idx = np.flatnonzero(d[key].astype(int) == 1)
        if len(idx) == 0:
            out["per_hand"][h] = None
            continue
        out["hands"].append(h)
        out["per_hand"][h] = (int(idx.min()), int(idx.max()))
    if out["hands"]:
        segs = [out["per_hand"][h] for h in out["hands"]]
        out["grasp_frame"] = min(s[0] for s in segs)
        out["release_frame"] = max(s[1] for s in segs)
    return out


def _quat_geodesic(qa, qb):
    """两个四元数之间的最小旋转角 (弧度), 对 q/-q 双覆盖不敏感."""
    dot = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def _pick_stable(quats, probs, ref_quat):
    """选与重建姿态**测地距最小**的稳定姿态; 没给参考就选概率最大的.

    尽量保留重建里的物体朝向 (对不对称物体重要: 杯把/盒口朝哪).
    最小测地距很大时说明重建朝向不可信 —— 报告里会带出这个角度.
    """
    if ref_quat is None:
        return int(np.argmax(probs))
    return int(np.argmin([_quat_geodesic(q, ref_quat) for q in quats]))


def stable_quat(mesh_path, ref_quat=None, cache_name="stable_poses.json"):
    """物体的稳定静置姿态 (wxyz) —— **带磁盘缓存**.

    ⚠ trimesh.compute_stable_poses 要跑 1~2 分钟 (凸包面枚举 + 倾倒图推演).
    直接在 Isaac 进程里调会阻塞主循环, 触发 Isaac 中止回调 -> 段错误 (实测 exit 139).
    所以结果缓存到 mesh 同目录, Isaac 里只读缓存.
    """
    cache = os.path.join(os.path.dirname(mesh_path), cache_name)
    # 同目录多网格 (screw 27 的瓶身+盖共用 reconstruction/): 通用名会互相打翻,
    # 网格专属缓存 <mesh名>.stable_poses.json 优先; 老布局 (一目录一网格) 不受影响.
    specific = os.path.join(os.path.dirname(mesh_path),
                            os.path.basename(mesh_path) + ".stable_poses.json")
    if os.path.exists(specific):
        cache = specific
    key = f"{os.path.basename(mesh_path)}:{os.path.getmtime(mesh_path):.0f}"
    # 自带快照 (datasets/): key 里的 mtime 在别人 clone 后必然对不上, 只比文件名。
    # 快照的 mesh 和缓存是一起提交的, 不可能不配套。不跳过的话就要在 Isaac 进程里
    # 现算 1~2 分钟 —— 那正是这个缓存要避免的段错误 (exit 139). 见 paths.is_bundled.
    lax = paths.is_bundled(mesh_path)
    if os.path.exists(cache):
        try:
            c = json.load(open(cache))
            if c.get("key") == key or (lax and str(c.get("key", "")).split(":")[0]
                                       == os.path.basename(mesh_path)):
                Q = [np.array(q, float) for q in c["quats"]]
                P = np.array(c["probs"], float)
                i = _pick_stable(Q, P, ref_quat)
                return Q[i], i, P
        except Exception:
            pass
    print(f"[stable] 现算 {os.path.basename(mesh_path)} 稳定姿态 (1~2 分钟, 之后走缓存)...")
    import trimesh
    m = trimesh.load(mesh_path, force="mesh")
    T, P = trimesh.poses.compute_stable_poses(m, n_samples=20)
    quats = [F.rotmat_to_quat(t[:3, :3]).tolist() for t in T]
    try:
        json.dump({"key": key, "quats": quats, "probs": P.tolist()}, open(cache, "w"), indent=1)
        print(f"[stable] 已缓存 -> {cache}")
    except Exception as e:
        print(f"[stable] ⚠ 缓存写入失败 ({e})")
    Q = [np.array(q, float) for q in quats]
    i = _pick_stable(Q, P, ref_quat)
    return Q[i], i, P


EXPECTED_XY_ALIGN = "middle_camera_forward_projected_xy"
EXPECTED_FRAME = "gravity_z_up_world"


def check_world_alignment(mesh_path):
    """读重建的世界系对齐元数据 (world_fused.npz, 在 mesh 同目录).

    **这是本模块最重要的一条**: 重建管线已经用**相机前向**把世界 XY 对齐过了
    (mode=middle_camera_forward_projected_xy, 取中间帧相机前向投到水平面当 +X).
    所以重建坐标里的 **+X 就是人的正前方**, 而机器人也朝 +X —— 天然对上, **不需要
    任何 yaw 旋转**.

    我早先自己发明朝向准则(双手轴 / 物体搬运方向)去转它, 反而把正确的相对关系
    转歪了 —— 盒子本来就在右前方(搬运 -43.8°), 硬转到 0° 等于把真实几何抹掉.
    """
    wf = os.path.join(os.path.dirname(mesh_path), "world_fused.npz")
    info = {"path": wf, "mode": None, "frame": None, "cam_fwd_deg": None}
    if not os.path.exists(wf):
        info["reason"] = "world_fused.npz 不存在"
        return False, info
    try:
        d = np.load(wf, allow_pickle=True)
        info["mode"] = str(d["world_xy_alignment_mode"]) if "world_xy_alignment_mode" in d.files else ""
        info["frame"] = str(d["coordinate_frame"]) if "coordinate_frame" in d.files else ""
        if "c2w" in d.files:                       # 实测相机水平朝向 (应贴近 0°)
            fw = d["c2w"][:, :3, 2]
            ang = np.degrees(np.arctan2(fw[:, 1], fw[:, 0]))
            info["cam_fwd_deg"] = float(np.abs(ang).mean())
        ok = (info["mode"] == EXPECTED_XY_ALIGN and info["frame"] == EXPECTED_FRAME)
        if not ok:
            info["reason"] = f"对齐模式/坐标系不符 (mode={info['mode']}, frame={info['frame']})"
        return ok, info
    except Exception as e:
        info["reason"] = f"读取失败: {e}"
        return False, info


def _yaw_from_task(J_prim, obj_p, inter):
    """任务朝向 -> 要转多少 yaw 才能对到 +X. 返回 (yaw, fallback级别, 原方向角°)."""
    gf, rf = inter["grasp_frame"], inter["release_frame"]
    for tier, v in ((1, obj_p[rf] - obj_p[gf]),            # ① 物体搬运方向
                    (2, J_prim[gf, 0] - J_prim[0, 0])):     # ② 交互手接近方向
        if np.linalg.norm(v[:2]) >= MIN_DIR_LEN:
            a = float(np.arctan2(v[1], v[0]))
            return -a, tier, float(np.degrees(a))
    return 0.0, 3, float("nan")                             # ③ 放弃旋转


def load_camera(mesh_path):
    """读相机(人头)位姿 c2w -> (位置(T,3), 视线方向(T,3)); 没有则 (None, None)."""
    wf = os.path.join(os.path.dirname(mesh_path), "world_fused.npz")
    if not os.path.exists(wf):
        return None, None
    try:
        c2w = np.load(wf, allow_pickle=True)["c2w"]
        return c2w[:, :3, 3].copy(), c2w[:, :3, 2].copy()
    except Exception:
        return None, None


def compute_placement(npz_path, mesh_path, robot_hands, robot_head=None,
                      obj_frame=None, table_top_z=0.85, obj_gap=0.002,
                      grasp_gap=GRASP_GAP, reach_forward=REACH_FORWARD,
                      palm_offset=PALM_OFFSET, scene_rot="identity",
                      obj_anchor=OBJ_ANCHOR):
    """求解摆放.

    robot_hands: {"left": (3,), "right": (3,)} 机器人左右手基座的**世界坐标**
    obj_frame:   物体摆放基准帧; None = 用 phase_* 检测出的抓取起始帧
    obj_anchor:  物体 re-anchor 到手的哪个点, "fingertips"(默认) 或 "palm"(旧)
    """
    d = np.load(npz_path, allow_pickle=True)
    J = {"left": d["joints_left"].astype(np.float64),
         "right": d["joints_right"].astype(np.float64)}
    obj_p = d["obj_pose"][:, :3].astype(np.float64)
    obj_q = d["obj_pose"][:, 3:7].astype(np.float64)

    Rs = F.scene_rotation(scene_rot)
    for h in J:
        J[h] = J[h] @ Rs.T
    obj_p = obj_p @ Rs.T

    inter = detect_interaction(npz_path)
    if not inter["hands"]:
        raise ValueError(f"{npz_path}: phase_* 没有任何接触帧, 无法定位交互结构")
    prim = inter["hands"][0] if len(inter["hands"]) == 1 else "right"    # 主交互手
    gf = inter["grasp_frame"] if obj_frame is None else int(obj_frame)

    # ---- 1. yaw: 优先用重建自带的相机前向对齐 (通常 = 不转) ----
    aligned, wa = check_world_alignment(mesh_path)
    if aligned:
        yaw, tier, task_deg = 0.0, 0, float("nan")     # tier 0 = 重建已对齐, 不动
    else:
        print(f"[align] ⚠ 重建未做相机前向对齐 ({wa.get('reason')}), 退回任务方向估计")
        yaw, tier, task_deg = _yaw_from_task(J[prim], obj_p, inter)
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pivot = obj_p[gf].copy()
    for h in J:
        J[h] = (J[h] - pivot) @ Rz.T + pivot
    obj_p = (obj_p - pivot) @ Rz.T + pivot

    palm = {}
    for h in J:
        wq = F.sharpa_base_quat_from_joints(J[h], h)
        palm[h] = J[h][:, 0] + F.rot_apply(
            wq, np.tile([[0.0, 0, palm_offset]], (len(J[h]), 1)))

    # ---- 2/3. XY 平移: **人头(相机) -> 机器人头** ----
    # 为什么锚在头而不是手: 人和机器人的"手相对身体的位置"差别很大 —— 实测这条 clip
    # 里人的双手几乎都在身体中线上(相对头 y≈±0.03), 而机器人双手左右分开 ±0.23.
    # 锚在手上会把人整个推偏(实测偏 31.8cm), 锚在头上等于"让机器人站在人当时的位置",
    # 演示被放进机器人的第一人称坐标系. 而且完全不依赖哪只手交互, 单手/双手/换手都通用.
    cam_pos_raw, cam_fwd_raw = load_camera(mesh_path)
    rh = {k: np.asarray(v, float) for k, v in robot_hands.items()}
    if cam_pos_raw is not None and robot_head is not None:
        cam_r = (cam_pos_raw @ Rs.T - pivot) @ Rz.T + pivot
        dxy = np.asarray(robot_head, float)[:2] - cam_r[gf, :2]
        anchor_mode = "head"
    else:                                   # 相机位姿缺失 -> 退回锚在交互手
        base_xy = ((rh["left"][:2] + rh["right"][:2]) / 2.0 if len(inter["hands"]) >= 2
                   else rh[prim][:2].copy())
        dxy = base_xy + np.array([reach_forward, 0.0]) - palm[prim][gf, :2]
        anchor_mode = "hand(fallback)"

    # ---- 4. 物体姿态: 最接近重建姿态的稳定姿态 ----
    q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    q_ref = obj_q[gf]
    q_stable, picked, probs = stable_quat(mesh_path, ref_quat=q_ref)
    obj_quat = F.quat_mul(q_yaw[None], q_stable[None])[0]
    obj_quat = obj_quat / np.linalg.norm(obj_quat)
    verts = F.load_obj_verts(mesh_path)
    v_rot = F.rot_apply(np.broadcast_to(obj_quat, (len(verts), 4)), verts)
    obj_rest_z = table_top_z + obj_gap - v_rot[:, 2].min()

    # ---- 5. re-anchor: 物体摆到手的哪个点 ----
    # 重建自带的手-物关系不可用 (实测接触段内指尖离物体表面**全程最近 7.06cm**,
    # 人手从没真正碰到过物体), 所以必须重新决定"物体在哪". 问题只在锚哪个点.
    #
    # 旧方案 "palm": 锚在掌心 (腕 + 掌法向 palm_offset). 隐含假设"手-物偏移 = 掌法向 12cm".
    #   实测这个假设错得离谱: 人捏东西时物体夹在**拇指和四指之间**, 不在掌心正下方.
    #   20 条 clip 里 **12 条**物体被放到了四指之外, 拇指够不着 -> 根本没有对指.
    # 新方案 "fingertips": 锚在**五指尖质心**, 即手合拢时真正围住东西的位置.
    #   物体中心相对"拇指尖->四指质心"轴的归一化投影按构造恒为 0.8, 结构上落在 (0,1) 内.
    #   20 条 clip 全部变成"物体夹在拇指和四指之间"; 指尖到表面中位 1.55->0.94cm.
    z_mid = obj_rest_z + 0.5 * (v_rot[:, 2].min() + v_rot[:, 2].max())   # 物体几何中高
    if obj_anchor == "fingertips":
        tip = J[prim][gf, F.MANO_TIPS].mean(0)          # 五指尖质心
        dz = z_mid - tip[2]                             # 指尖质心落到物体中高
        anchor_pt = tip
    elif obj_anchor == "palm":
        dz = obj_rest_z + grasp_gap - palm[prim][gf, 2]
        anchor_pt = palm[prim][gf]
    else:
        raise ValueError(f"未知 obj_anchor={obj_anchor!r}, 只支持 fingertips / palm")

    shift = np.array([dxy[0], dxy[1], dz])
    for h in J:
        J[h] = J[h] + shift
        palm[h] = palm[h] + shift
    anchor_pt = anchor_pt + shift
    obj_pos = np.array([anchor_pt[0], anchor_pt[1], obj_rest_z])

    # ---- 相机(人头)位姿: 走同一套变换, 供可视化核对 ----
    cam_pos = cam_fwd = None
    if cam_pos_raw is not None:
        cam_pos = (cam_pos_raw @ Rs.T - pivot) @ Rz.T + pivot + shift
        cam_fwd = (cam_fwd_raw @ Rs.T) @ Rz.T         # 方向只转不平移

    return {
        "shift": shift, "yaw": yaw, "joints": J, "palm": palm,
        "obj_pos": obj_pos, "obj_quat": obj_quat, "obj_rest_z": obj_rest_z,
        "cam_pos": cam_pos, "cam_fwd": cam_fwd,
        "inter": inter, "primary": prim, "grasp_frame": gf,
        "yaw_tier": tier, "task_dir_deg": task_deg, "world_align": wa,
        "anchor_mode": anchor_mode, "obj_anchor": obj_anchor,
        "mesh_path": mesh_path,
        "stable": {"n": len(probs), "picked": picked, "prob": float(probs[picked]),
                   "vs_recon_deg": float(np.degrees(_quat_geodesic(q_stable, q_ref)))},
    }


def placement_report(res, table_top_z=0.85, table_half=0.6,
                     shoulder=None, arm_reach=0.755, grasp_gap=GRASP_GAP):
    """质量门 —— 自动判定这条 clip 摆放后能不能直接拿去训练.

    不人工设计每条 clip, 但要自动标出哪条不能用.
    返回 (是否全过, 指标 dict); ok=False 的项就是拦下的理由.
    """
    inter, prim, gf = res["inter"], res["primary"], res["grasp_frame"]
    palm, J = res["palm"], res["joints"]
    a, b = inter["per_hand"][prim]
    m = {}

    # ---- 抓取几何: 手指到底够不够得着、有没有对指 ----
    # 旧指标 grasp_gap_m 量的是"掌心到物体中心 = 4.5cm 吗", 在 fingertips 锚点下没意义,
    # 而且它**从来没能发现真正的问题** —— 掌心距离对了, 拇指仍可能在 4cm 外.
    # 真正决定抓不抓得住的是: ① 五个指尖离物体表面多远 ② 物体在不在拇指和四指之间.
    tips = J[prim][gf, F.MANO_TIPS]
    if res.get("mesh_path") and np.isfinite(tips).all():
        verts = F.load_obj_verts(res["mesh_path"])
        vr = F.rot_apply(np.broadcast_to(res["obj_quat"], (len(verts), 4)), verts) \
            + res["obj_pos"]
        ds = np.array([np.linalg.norm(vr - t, axis=1).min() for t in tips])
        m["tip_gap_med_m"] = {"v": float(np.median(ds)), "ok": float(np.median(ds)) < 0.02,
                              "note": "抓取帧五指尖到物体表面的中位距离"}
        m["tip_gap_max_m"] = {"v": float(ds.max()), "ok": float(ds.max()) < 0.04,
                              "note": "最远的那根手指; 太大说明该指没参与"}
        # 对指性: 物体中心投影到"拇指尖->四指质心"轴, 归一化后应落在 (0,1)
        oth = tips[1:].mean(0)
        ax = oth - tips[0]
        L = float(np.linalg.norm(ax))
        pr = float((res["obj_pos"] - tips[0]) @ (ax / (L + 1e-9)) / (L + 1e-9))
        m["thumb_opposition"] = {"v": pr, "ok": 0.0 < pr < 1.0,
                                 "note": "物体在拇指和四指之间才算真抓握; <0或>1 = 无对指"}
    if res.get("obj_anchor") == "palm":              # 旧锚点才有意义
        gap = float(np.linalg.norm(palm[prim][gf] - res["obj_pos"]))
        m["grasp_gap_m"] = {"v": gap, "ok": abs(gap - grasp_gap) < 0.005}

    zmin = float(palm[prim][a:b + 1, 2].min())
    m["palm_z_min"] = {"v": zmin, "ok": zmin >= table_top_z}

    d_ctr = float(np.abs(res["obj_pos"][:2]).max())
    m["obj_within_table"] = {"v": d_ctr, "ok": d_ctr <= table_half}

    if shoulder is not None:
        dmax = float(np.linalg.norm(palm[prim][a:b + 1] - np.asarray(shoulder, float),
                                    axis=1).max())
        m["reach_max_m"] = {"v": dmax, "ok": dmax <= arm_reach}

    m["stable_vs_recon_deg"] = {"v": res["stable"]["vs_recon_deg"], "ok": True,
                                "note": ">60° 说明重建朝向不可信(记录不拦)"}
    m["yaw_tier"] = {"v": res["yaw_tier"], "ok": res["yaw_tier"] == 0,
                     "note": "0=重建相机前向对齐(期望) 1=物体搬运 2=接近方向 3=未转"}
    wa = res.get("world_align") or {}
    if wa.get("cam_fwd_deg") is not None:
        m["cam_fwd_dev_deg"] = {"v": wa["cam_fwd_deg"], "ok": wa["cam_fwd_deg"] < 20.0,
                                "note": "相机前向偏离+X的平均角, 越小说明对齐越好"}
    if len(inter["hands"]) >= 2:                     # 仅双手任务适用
        v = J["right"][gf, 0] - J["left"][gf, 0]
        ang = abs((abs(float(np.degrees(np.arctan2(v[1], v[0])))) + 90) % 180 - 90)
        m["hand_axis_vs_task_deg"] = {"v": ang, "ok": ang > 45.0,
                                      "note": "双手任务应接近 90°"}
    return all(x["ok"] for x in m.values()), m
