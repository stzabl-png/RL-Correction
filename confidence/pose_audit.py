#!/usr/bin/env python
"""物体位姿轨迹体检 —— 只诊断不修正, 按**粗错类型**逐帧定性。

## 为什么不是"打一个分"

目标不是把每帧修准, 而是**保证没有粗错**(姿态整个错、跟丢卡住、瞬移、穿桌),
小抖动可以接受。所以用一组各自针对一种失效模式的廉价检测器, 而不是一个 IoU 阈值 ——
IoU 把"稍偏"和"偏得离谱"混在一起, 还会被遮挡污染(实测 screw27 交互段 IoU 中位 0.53
纯粹是手挡出来的, 位姿其实是好的)。

## 判据(都用大余量粗判, 便于跨数据集迁移)

  explained      |mask∩proj|/|mask|      整体错位。抗遮挡
  d_cent_norm    质心距 / sqrt(mask面积)  跟丢/错位。抗遮挡, **且与网格尺度无关**
  tilt_deg       长轴离竖直角度(世界系)    姿态整个错(仅在"贴面静置"时有意义)
  above_mm       最低点离支撑面           穿透 / 悬空
  step_mm/deg    逐帧位移/转角            瞬移
  lag            mask 质心在动、投影质心不动  跟丢卡住 ← 不需要任何物理先验

## 输出

逐帧 failure_mode + 逐 take 汇总。**不修改任何原始数据。**
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import trimesh

# 判据参数 —— 全部经 33 条 + 人眼校准表验证过, 每条都写明为什么是这个值
TH = dict(
    d_cent_soft=0.35,      # 质心归一化偏移: 主判据(唯一经人眼验证可靠)
    exp_lo=0.40, exp_hi=0.85,   # explained 只用来抓"完全没盖住", 不再用 0.80 当硬线
    step_mm=50.0, step_deg=25.0,     # 瞬移
    lag_mask_px=8.0, lag_proj_px=2.0,
    sink_mm=-15.0,         # 穿透: 降级为软信号(支撑面估计误差会直接变误报)
    scale_lo=0.50, scale_hi=1.60,
    rot_obs_lo=0.08, rot_obs_hi=0.28,   # 旋转可观测性(实测: 苹果0.036 瓶轴0.019 手机0.281)
    # CoTracker 一致性阈值(实测标定: 对照组 r_err 0.025/spread 0.033;
    # 27坏段 r_err 0.121; 苹果 spread 0.09~0.19)
    ct_r_pivot=0.04, ct_r_scale=0.10, ct_sp_pivot=0.04, ct_sp_scale=0.08,
    ct_flow_soft=5.0, ct_flow_hard=10.0,
    ct_refute=0.20,   # CT信任度低于此 = 被反驳(非"没证据")。标定: 27坏段 tr≈0.19,
                      # 苹果幽灵段 tr=0, 对照组 tr≈1.0 —— 0.20 正好切在坏段之上
    cap_scale_unreliable=35,
)

# ★ 判据演化史(每一条都是被数据/人眼推翻后才改的, 别再走回头路)
#
# 删除「静置时长轴该竖直」  33 条实测 950 次假阳性 —— pick_place 的手机玩具本来就平躺
#                          (倾角稳定 87~88°、标准差 2°)。
# 删除「离任一稳定姿态多远」 又漏报 —— 瓶子横躺本来就是合法稳定姿态, screw27 已知坏帧
#                          离最近稳定姿态只有 13~27°。**物理可行 ≠ 正确。**
# 删除 unsupportable        人眼校准表确认假阳性(bpp/6 的 f37/f48/f58 轮廓吻合良好)。
# 降级 penetrating          人眼确认多为假阳性(bpp/13 的 f31/32/33)——支撑面是估出来的,
#                          估偏一点就误报。保留为软信号, 不单独定性。
# 主判据 d_cent_norm        人眼校准中唯一可靠: 真错 f5=8.09 / f90=0.41, 好帧 0.002~0.27。
# explained 降为辅助        判别力随遮挡单调衰减: 遮挡越重 mask 越小, 错的位姿越容易
#                          "盖住"缩水的 mask 拿高分(bpp/6 手机入盒被订书机遮挡即此例)。
#
# ★★ 结构性盲区: 轮廓判据对**近似对称物体的旋转**完全失明。
#    bpp/13 的苹果被转了 116°, 而 explained 0.95 / iou_occ 0.92 / 质心偏移 0.46px 全是绿的
#    —— 苹果轴长比 1.07, 转多少度轮廓都一样。用户靠枝干朝向(纹理)才看出来。
#    对策不是"检测", 是**事先声明不可观测**: 从网格算旋转可观测性, 低的直接判可信度 0,
#    交给 RL 自己探索。

def load_obj_mask(scene: Path, i: int):
    d = scene / "masks" / "objects" / "frames" / f"frame_{i:06d}_masks"
    acc = None
    if d.is_dir():
        for p in sorted(d.glob("object_*.png")):
            a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if a is not None:
                b = a > 127
                acc = b if acc is None else (acc | b)
    return acc


def load_hand_mask(scene: Path, i: int):
    d = scene / "masks" / "hands" / "frames" / f"frame_{i:06d}_masks"
    acc = None
    if d.is_dir():
        for s in ("left", "right"):
            a = cv2.imread(str(d / f"{s}_hand_0.png"), cv2.IMREAD_GRAYSCALE)
            if a is not None:
                b = a > 127
                acc = b if acc is None else (acc | b)
    return acc


def silhouette(V, F, K, T, hw, ss=1):
    H, W = hw
    if ss > 1:
        K = K.copy(); K[:2] /= ss; H, W = H // ss, W // ss
    Vc = (T[:3, :3] @ V.T).T + T[:3, 3]
    if not np.isfinite(Vc).all():
        return np.zeros((H, W), bool)
    d = Vc[:, 2:3]
    uv = (K @ Vc.T).T
    uv = uv[:, :2] / np.where(np.abs(d) < 1e-9, 1e-9, d)
    tri = uv[F]
    front = tri[np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) < 0]
    img = np.zeros((H, W), np.uint8)
    if len(front):
        cv2.fillPoly(img, [t.astype(np.int32) for t in front], 255)
    return img > 0


def decimate(V, F, voxel=0.003):
    key = np.floor(V / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    m = int(inv.max()) + 1
    Vd = np.zeros((m, 3)); np.add.at(Vd, inv, V)
    Vd /= np.bincount(inv, minlength=m).reshape(-1, 1)
    Fd = inv[F]
    ok = (Fd[:, 0] != Fd[:, 1]) & (Fd[:, 1] != Fd[:, 2]) & (Fd[:, 0] != Fd[:, 2])
    return Vd, Fd[ok]


def rotation_observability(V, F, K, n_view=6, ang=30.0, extents=None):
    """绕各轴转 ang 度后投影轮廓的 IoU 掉多少 —— 掉得少 = 该转动**不可观测**。

    实测: 苹果(球) [0.063,0.036,0.065] / 瓶子绕瓶轴 0.019 / 手机 [0.499,0.281,0.336]。
    取三轴最小值: 只要有一个方向转了看不出来, 整体旋转就不可信(位姿可能沿该方向错任意角度)。
    """
    Vc = V - V.mean(0)
    d = float(np.linalg.norm(extents)) * 3.0 if extents is not None else 1.0
    rng = np.random.default_rng(0)
    out = []
    for axis in range(3):
        ious = []
        for _ in range(n_view):
            q = rng.normal(size=4); q /= np.linalg.norm(q)
            w, x, y, z_ = q
            Rv = np.array([[1-2*(y*y+z_*z_), 2*(x*y-z_*w), 2*(x*z_+y*w)],
                           [2*(x*y+z_*w), 1-2*(x*x+z_*z_), 2*(y*z_-x*w)],
                           [2*(x*z_-y*w), 2*(y*z_+x*w), 1-2*(x*x+y*y)]])
            T0 = np.eye(4); T0[:3, :3] = Rv; T0[:3, 3] = [0, 0, d]
            a = silhouette(Vc, F, K, T0, (1080, 1920), ss=2)
            e = np.zeros(3); e[axis] = 1
            th = np.radians(ang); c, sn = np.cos(th), np.sin(th)
            Kx = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
            T1 = T0.copy(); T1[:3, :3] = Rv @ (np.eye(3) + sn*Kx + (1-c)*Kx@Kx)
            b = silhouette(Vc, F, K, T1, (1080, 1920), ss=2)
            u = (a | b).sum()
            if u:
                ious.append((a & b).sum() / u)
        out.append(1.0 - float(np.mean(ious)) if ious else 0.0)
    return out


def score_frame(dcn, exp, occl, rot_factor, cap_scale, modes, soft, ct=None):
    """逐帧可信度 0~100。分位置/旋转两路 —— 混成一个数会把两者都毁掉。

    苹果那个例子: 位置可信(90) 旋转不可信(0)。若只报一个总分, 要么高估要么低估。
    """
    if dcn is None or exp is None:
        return dict(conf=0, conf_pos=0, conf_rot=0, reason="no_evidence")
    s_pos = 100.0 * float(np.exp(-(dcn / TH["d_cent_soft"]) ** 2))
    s_cov = 100.0 * float(np.clip((exp - TH["exp_lo"]) / (TH["exp_hi"] - TH["exp_lo"]), 0, 1))
    w_cov = max(0.0, 1.0 - occl)                     # 遮挡越重, explained 越没判别力
    s_img = (s_pos + w_cov * s_cov) / (1.0 + w_cov)
    cap_ev = 100.0 * max(0.0, 1.0 - occl)            # 证据量上限: 看不见就不能说"可信"
    conf_pos = min(s_img, cap_ev, cap_scale)
    if "teleport" in modes:
        conf_pos *= 0.30
    if "lag" in modes:
        conf_pos *= 0.40
    if "penetrating" in soft:
        conf_pos *= 0.85
    # ---- CoTracker 第二观察员(有 cc 数据的帧) ----
    # ⚠ ct_t_err 不用于罚 conf_pos: 坏种子的反投影污染 + 可见点不对称会把旋转误差
    #   漏进平移分量(苹果实测: 位置明明对, t_err 却 0.17~0.24)。位置罚只用 flow_err
    #   (帧间相对量, 参考帧无关, 已在苹果幽灵窗口/bpp11 漂移段验证定位准)。
    ct_present = bool(ct) and ct.get("n_vis", 0) >= 6
    if ct_present:
        fl = ct.get("flow_err_px")
        if fl is not None:
            if fl > TH["ct_flow_hard"]:
                conf_pos *= 0.3
            elif fl > TH["ct_flow_soft"]:
                conf_pos *= 0.6
    # 旋转可信:
    #   CT 在场 => 旋转经纹理**可观测**(即使轮廓盲, 如瓶子/苹果), 由 CT 裁决:
    #     r_err(定位准) 与 spread(内部不自洽) 各给一个信任度, 取小
    #   CT 不在场 => 退回轮廓可观测性
    if ct_present:
        tr1 = (float(np.clip(1 - (ct["ct_r_err"] - TH["ct_r_pivot"]) / TH["ct_r_scale"], 0, 1))
               if ct.get("ct_r_err") is not None else 1.0)
        tr2 = (float(np.clip(1 - (ct["ct_r_spread"] - TH["ct_sp_pivot"]) / TH["ct_sp_scale"], 0, 1))
               if ct.get("ct_r_spread") is not None else 1.0)
        trust = min(tr1, tr2)
        conf_rot = conf_pos * max(rot_factor, 1.0) * trust
        # ★ refuted ≠ 低分: 低分是"没证据", refuted 是"有反证"。
        #   下游(RTS)对 refuted 帧要**剔除**旋转测量, 而不是只放大 σ ——
        #   否则一段"一致地错"的测量会让滤波器自信地跟着错(苹果: 82帧幽灵旋转,
        #   σ_rot 收缩到 2~5° 但旋转错 116°)。
        rot_refuted = trust < TH["ct_refute"]
    else:
        conf_rot = conf_pos * rot_factor
        rot_refuted = False
    return dict(conf=round(0.6 * conf_pos + 0.4 * conf_rot),
                conf_pos=round(conf_pos), conf_rot=round(conf_rot),
                rot_refuted=bool(rot_refuted))


def cent(m):
    if m is None or not m.any():
        return None
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


def audit_take(scene: Path, ss=2, max_frames=400, ct_dir: Path | None = None) -> dict:
    z = np.load(scene / "world_fused.npz", allow_pickle=True)
    K = np.asarray(z["K"], float)
    Tc = np.asarray(z["object_ob_in_cam"], float)
    Tw = np.asarray(z["object_ob_in_world"], float)
    n = min(len(Tc), max_frames)
    meshes = (sorted(glob.glob(str(scene / "objects" / "*" / "*.obj")))
              or sorted(glob.glob(str(scene / "*.obj"))))
    if not meshes:
        return {"take": str(scene), "error": "no mesh"}
    mesh = trimesh.load(meshes[0], force="mesh")
    V0 = np.asarray(mesh.vertices, float); F0 = np.asarray(mesh.faces, int)
    V0, F0 = decimate(V0, F0)
    long_axis = int(np.argmax(mesh.extents))
    hw = (1080, 1920)
    ct_map = {}
    if ct_dir is not None:
        ccp = Path(ct_dir) / f"cc_{scene.parent.name}_{scene.name}.json"
        if ccp.is_file():
            for r in json.loads(ccp.read_text()).get("per_frame", []):
                ct_map[r["frame"]] = r
    om = {}
    for i in range(n):
        m = load_obj_mask(scene, i)
        if m is not None and m.any():
            om[i] = m
            hw = m.shape
    if len(om) < 8:
        return {"take": str(scene), "error": f"masks too few ({len(om)})"}

    def ds(m, s):
        return (m if s == 1 else
                cv2.resize(m.astype(np.uint8), (m.shape[1] // s, m.shape[0] // s),
                           interpolation=cv2.INTER_NEAREST) > 0)

    # --- 尺度: 在**遮挡最少**的帧上拟合(遮挡帧的 IoU 不可信) ---
    occ = {}
    for i in list(om)[::max(1, len(om) // 24)]:
        h = load_hand_mask(scene, i)
        pj = silhouette(V0, F0, K, Tc[i], hw, ss)
        occ[i] = (float((pj & ds(h, ss)).sum()) / max(1, pj.sum())) if h is not None else 0.0
    clean = [i for i, _ in sorted(occ.items(), key=lambda kv: kv[1])[:6]]
    best = (1.0, -1.0)
    for s in np.arange(TH['scale_lo'], TH['scale_hi'] + 1e-9, 0.025):
        C = V0.mean(0); V = (V0 - C) * s + C
        v = []
        for i in clean:
            mk = ds(om[i], ss); pj = silhouette(V, F0, K, Tc[i], hw, ss)
            v.append((mk & pj).sum() / max(1, (mk | pj).sum()))
        mi = float(np.median(v))
        if mi > best[1]:
            best = (float(s), mi)
    scale = best[0]
    scale_ok = TH['scale_lo'] + 1e-6 < scale < TH['scale_hi'] - 1e-6
    C = V0.mean(0); V = (V0 - C) * scale + C
    ro = rotation_observability(V, F0, K, extents=mesh.extents)
    rot_factor = float(np.clip((min(ro) - TH["rot_obs_lo"])
                               / (TH["rot_obs_hi"] - TH["rot_obs_lo"]), 0, 1))
    cap_scale = 100.0 if scale_ok else float(TH["cap_scale_unreliable"])

    # --- 支撑面: 迭代 + 只用真静置帧 (不能取中位数, 会混进"提起端着") ---
    low = {i: (((Tw[i][:3, :3] @ V.T).T + Tw[i][:3, 3])[:, 2]).min() for i in om}
    disp = {i: (np.linalg.norm(Tw[i][:3, 3] - Tw[i - 1][:3, 3]) * 1000 if i > 0 else 0.0)
            for i in om}
    lv = np.array(list(low.values()))
    plane = float(np.percentile(lv, 15))
    for _ in range(6):
        rest = [i for i in om if abs(low[i] - plane) < 0.008 and disp[i] < 3.0]
        if len(rest) < 6:
            rest = [i for i in om if abs(low[i] - plane) < 0.008]
        if len(rest) < 6:
            plane = None; break
        new = float(np.median([low[i] for i in rest]))
        if abs(new - plane) < 5e-4:
            plane = new; break
        plane = new

    # ---- CT 资格门 ----
    # 反驳权不是白给的: CT 必须先过"静止段体检"。仲裁用唯一有真值的事实 ——
    # 静止物体旋转必然不变: 若 FP 在静止段很稳(<0.5°/帧)而 CT 仍在喊"旋转不对"
    # (r_err>0.08), 说谎的是 CT, 取消其整条 take 的证人资格(按 CT 不在场处理)。
    # 若 FP 自己在静止段漂(苹果 ~1.5°/帧、bpp/11), CT 的指控有据, 资格保留。
    # held-out 教训: bpp/2 被 CT 误反驳 96 帧 → 证据真空 → explained 0.905→0.662。
    ct_qualified, ct_qual = True, {}
    if ct_map:
        static = []
        for i in sorted(om):
            if i not in ct_map or disp.get(i, 99.0) >= 3.0:
                continue
            h = load_hand_mask(scene, i)
            if h is not None and (om[i] & h).any():
                continue
            static.append(i)
        pairs = [(a, b) for a, b in zip(static, static[1:]) if b == a + 1]
        fp_step = [float(np.degrees(np.arccos(np.clip(
            (np.trace(Tw[b][:3, :3] @ Tw[a][:3, :3].T) - 1) / 2, -1, 1))))
            for a, b in pairs]
        rerrs = [ct_map[i].get("ct_r_err") for i in static]
        rerrs = [r for r in rerrs if r is not None]
        if len(fp_step) >= 5 and len(rerrs) >= 5:
            fp_med, ct_med = float(np.median(fp_step)), float(np.median(rerrs))
            ct_qual = {"n_static": len(static), "fp_static_step_deg": round(fp_med, 2),
                       "ct_static_r_err": round(ct_med, 3)}
            if fp_med < 0.5 and ct_med > 0.08:
                ct_qualified = False
                ct_map = {}          # 取消资格: 整条按 CT 不在场处理(退回轮廓可观测性)

    # --- 逐帧指标 ---
    per, prev_mc, prev_pc = [], None, None
    for i in sorted(om):
        mk = ds(om[i], ss)
        pj = silhouette(V, F0, K, Tc[i], hw, ss)
        h = load_hand_mask(scene, i)
        hm = ds(h, ss) if h is not None else None
        am = mk.sum()
        ex = float((mk & pj).sum() / am) if am else 0.0
        mc, pc = cent(mk), cent(pj)
        dcn = (float(np.linalg.norm(mc - pc) / max(np.sqrt(am), 1e-6))
               if (mc is not None and pc is not None) else None)
        tilt = float(np.degrees(np.arccos(np.clip(abs(Tw[i][:3, :3][:, long_axis][2]), -1, 1))))
        above = (low[i] - plane) * 1000 if plane is not None else None
        sd = float(np.degrees(np.arccos(np.clip(
            (np.trace(Tw[i][:3, :3] @ Tw[i - 1][:3, :3].T) - 1) / 2, -1, 1)))) if i > 0 else 0.0
        mstep = float(np.linalg.norm(mc - prev_mc)) if (mc is not None and prev_mc is not None) else 0.0
        pstep = float(np.linalg.norm(pc - prev_pc)) if (pc is not None and prev_pc is not None) else 0.0
        prev_mc, prev_pc = mc, pc
        occf = float((pj & hm).sum() / max(1, pj.sum())) if hm is not None else 0.0

        modes, soft = [], []
        if dcn is not None and dcn > TH["d_cent_soft"]:
            modes.append("misplaced")
        if ex < TH["exp_lo"]:
            modes.append("misplaced")
        if disp[i] > TH["step_mm"] or sd > TH["step_deg"]:
            modes.append("teleport")
        if mstep > TH["lag_mask_px"] * (1.0 / ss) and pstep < TH["lag_proj_px"] * (1.0 / ss):
            modes.append("lag")
        if above is not None and above < TH["sink_mm"]:
            soft.append("penetrating")          # 软信号: 只扣分, 不定性
        sc = score_frame(dcn, ex, occf, rot_factor, cap_scale, modes, soft,
                         ct=ct_map.get(i))
        per.append(dict(frame=i, explained=round(ex, 4),
                        d_cent_norm=None if dcn is None else round(dcn, 4),
                        tilt_deg=round(tilt, 1),
                        above_mm=None if above is None else round(above, 1),
                        step_mm=round(disp[i], 2), step_deg=round(sd, 2),
                        mask_step_px=round(mstep, 2), proj_step_px=round(pstep, 2),
                        occl=round(occf, 3), modes=sorted(set(modes)), soft=soft, **sc))

    cnt = {}
    for r in per:
        for m in r["modes"]:
            cnt[m] = cnt.get(m, 0) + 1
    bad = [r for r in per if r["modes"]]
    return {
        "take": str(scene), "n_frames": n, "n_scored": len(per),
        "mesh_scale_fitted": round(scale, 3), "scale_iou": round(best[1], 3),
        "scale_reliable": bool(scale_ok),
        # 原始 mesh 三轴尺寸(未乘拟合尺度)。take_manifest 用它判"同视频的两个 take
        # 重建的是不是同一个物体"(瓶身 vs 瓶盖不能互相顶替)
        "mesh_extents_m": [round(float(v), 4) for v in mesh.extents],
        "rot_observability": [round(v, 3) for v in ro], "rot_factor": round(rot_factor, 3),
        "conf_median": round(float(np.median([r["conf"] for r in per])), 1),
        "conf_pos_median": round(float(np.median([r["conf_pos"] for r in per])), 1),
        "conf_rot_median": round(float(np.median([r["conf_rot"] for r in per])), 1),
        "conf_ge70": int(sum(1 for r in per if r["conf"] >= 70)),
        "conf_lt30": int(sum(1 for r in per if r["conf"] < 30)),
        "ct_frames": int(sum(1 for i in sorted(om) if i in ct_map)),
        "ct_qualified": bool(ct_qualified), "ct_qualification": ct_qual,
        "support_plane_z": None if plane is None else round(plane, 5),
        "lifted_frames": (0 if plane is None else
                          sum(1 for r in per if r["above_mm"] is not None and r["above_mm"] > 10)),
        "median_explained": round(float(np.median([r["explained"] for r in per])), 4),
        "median_occl": round(float(np.median([r["occl"] for r in per])), 3),
        "bad_frames": len(bad), "bad_ratio": round(len(bad) / max(1, len(per)), 3),
        "mode_counts": cnt, "per_frame": per,
    }


@contextmanager
def poseqa_lock(out_dir: Path):
    """poseqa 共享 json 的读-改-写锁。批量队列多 worker 并发跑 confidence 步时,
    pose_audit.json / TAKE_MANIFEST.json 是共享写点, 不锁会互相覆盖。"""
    lf = out_dir / ".poseqa.lock"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(lf, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def audit_one_incremental(scene: Path, out: Path, ss: int, ct_dir: Path | None) -> int:
    """单 take 增量: 只审这一条, 锁内更新 pose_audit.json 里对应 entry。
    与全库重扫对该 take 的结果逐字段一致(冒烟验收标准)。"""
    scene = scene.resolve()
    t0 = time.time()
    try:
        r = audit_take(scene, ss=ss, ct_dir=ct_dir)
    except Exception as e:
        r = {"take": str(scene), "error": f"{type(e).__name__}: {e}"}
    with poseqa_lock(out):
        p = out / "pose_audit.json"
        doc = json.loads(p.read_text()) if p.is_file() else {"thresholds": TH, "takes": []}
        doc["thresholds"] = TH
        doc["takes"] = [x for x in doc["takes"] if x.get("take") != r["take"]]
        doc["takes"].append(r)
        doc["takes"].sort(key=lambda x: x.get("take", ""))
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    if "error" in r:
        print(f"[audit] X {scene}  {r['error']}")
        return 1
    print(f"[audit] {scene.name}: s={r['mesh_scale_fitted']:.3f} "
          f"可信度 中位{r['conf_median']:5.1f} (位置{r['conf_pos_median']:5.1f}/"
          f"旋转{r['conf_rot_median']:5.1f})  用时 {time.time()-t0:.0f}s  → {p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput"))
    ap.add_argument("--include", default="egodex", help="只跑路径含此串的 take")
    ap.add_argument("--exclude", default="test_generate", help="跳过路径含此串的 take")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--ct-dir", type=Path, default=None,
                    help="cotracker_consistency 输出目录, 有则并入打分")
    ap.add_argument("--scene", type=Path, default=None,
                    help="只审这一条 take, 增量更新 pose_audit.json (pipeline confidence 步用)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.scene is not None:
        return audit_one_incremental(a.scene, a.out, a.ss, a.ct_dir)

    takes = sorted(Path(p).parent for p in
                   glob.glob(str(a.root / "**" / "world_fused.npz"), recursive=True))
    takes = [t for t in takes if a.include in str(t) and a.exclude not in str(t)]
    print(f"[audit] {len(takes)} 条 take")
    res, t0 = [], time.time()
    for k, t in enumerate(takes, 1):
        try:
            r = audit_take(t, ss=a.ss, ct_dir=a.ct_dir)
        except Exception as e:
            r = {"take": str(t), "error": f"{type(e).__name__}: {e}"}
        res.append(r)
        rel = str(t).replace(str(a.root) + "/", "")
        if "error" in r:
            print(f"  [{k}/{len(takes)}] X {rel}  {r['error']}")
        else:
            print(f"  [{k}/{len(takes)}] {rel:46s} s={r['mesh_scale_fitted']:.3f}"
                  f"{'' if r['scale_reliable'] else '!'} "
                  f"遮挡={r['median_occl']:.2f} 旋转可观测={min(r['rot_observability']):.3f} "
                  f"| 可信度 中位{r['conf_median']:5.1f} (位置{r['conf_pos_median']:5.1f}/"
                  f"旋转{r['conf_rot_median']:5.1f})  ≥70:{r['conf_ge70']:3d} <30:{r['conf_lt30']:3d}"
                  f"  {r['mode_counts']}")
    (a.out / "pose_audit.json").write_text(
        json.dumps({"thresholds": TH, "takes": res}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    ok = [r for r in res if "error" not in r]
    agg = {}
    for r in ok:
        for m, c in r["mode_counts"].items():
            agg[m] = agg.get(m, 0) + c
    tot = sum(r["n_scored"] for r in ok)
    print(f"\n[audit] {len(ok)}/{len(takes)} 条成功, 共 {tot} 帧, 用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"[audit] 失效模式合计(帧次): {agg}")
    print(f"[audit] 报告: {a.out / 'pose_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
