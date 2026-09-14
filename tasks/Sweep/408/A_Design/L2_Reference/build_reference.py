"""Sweep408 母带 (扫地: 右手扫把 + 左手簸箕)。

设计三条 (2026-09-11 用户裁定):
  ① **滤波 + 平滑**: 重建物轨有跟踪跳变 (扫把实测单帧转 42.9°, 逐帧转角中位仅 0.66°)。
     先按逐帧转角/位移剔野点并在两侧好帧之间插值, 再做移动平均低通。
  ② **z 重新设计, 不回放** (用户"第二问走 B"): 重建物轨在竖直方向不可信 ——
     扫把头全程悬在桌面上方 3.2~8.6cm (中位 5.5cm), **一帧都没接触**; 而簸箕被扶在
     桌上基本不动, 它的斗底 z 却抖了 3.6cm (std 0.80cm) —— 这就是噪声底。扫把抖
     8.3cm (std 1.71cm) 只有噪声的 2 倍, 真假分不出来, 绝对高度又是错的。
     ⟹ z 设计成**常量接触高度**: 扫把头底面压到桌面下 press_depth; 簸箕斗底离桌 pan_lip_gap。
     XY 与朝向仍走人的真实路径 (那部分可信: 逐帧 4mm, conf 72.8, σ 4.8mm)。
     这与 Clean/3 的做法同源 (海绵对盘面 z 钳 + 常量压深, XY 不钳)。
  ③ 手 = 物体 ∘ T_oh (GraspPose 先验, 输入系), 连续臂 IK, 时间展宽只放慢不改路径 (Sweep2 同款)。

几何 (入库网格局部系, 长轴 = 局部 z, 网格按 COM 居中):
  扫把 24.2cm: [z_min, z_min+10.1cm] 细柄 (2×2cm) | 其余 = 头 (宽到 3.8×5.7cm)
  簸箕 20.2cm: [z_min, z_min+10.1cm] 把手       | 其余 = 斗 (宽到 18.2cm)

用法:
  PY=/home/lyh/luhr/MagicSim/.venv/bin/python
  $PY tasks/Sweep/408/A_Design/L2_Reference/build_reference.py --scan        # 摆位扫描
  $PY tasks/Sweep/408/A_Design/L2_Reference/build_reference.py               # 造带
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
TAKE = os.environ.get("SWEEP_TAKE", "408")           # 2026-09-13: 175 复用同一构带器 (同一批实物, 只换动作)
DATA = os.path.join(ROOT, "datasets", f"sweep{TAKE}")
DEXO = "/home/lyh/Project/Dexonomy/assets/object/custom/processed_data"
TABLE_Z = 0.87
# oid -> (Dexonomy 导入名, 导入时的 --scale, 手, 先验, 柄段长度 m)
SPEC = {0: dict(dx="p4t408_broom_r2", s=1.30, hand="right", prior="Sweep408_broom", handle=0.101),
        1: dict(dx="p4t408_dustpan_r2", s=1.25, hand="left", prior="Sweep408_dustpan", handle=0.101)}

p = argparse.ArgumentParser()
p.add_argument("--output", default=f"tasks/Sweep/{TAKE}/A_Design/L2_Reference/sweep{TAKE}_reference_v3.npz")
p.add_argument("--anchor", default="tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json")
p.add_argument("--control_hz", type=float, default=20.0)
# ① 滤波
p.add_argument("--rot_jump_deg", type=float, default=8.0, help="逐帧转角超过它 = 跟踪跳变, 剔除后插值")
p.add_argument("--pos_jump_m", type=float, default=0.030, help="逐帧位移超过它 = 跟踪跳变")
p.add_argument("--lp_win", type=int, default=5, help="移动平均窗口 (源帧, 奇数)")
# ② z 设计
p.add_argument("--press_depth_m", type=float, default=0.001, help="扫把头底面压到桌面下多少")
p.add_argument("--pan_lip_gap_m", type=float, default=0.003, help="簸箕斗底离桌面多少 (Sweep2 clear_q 中心)")
p.add_argument("--extra_tilt_deg", type=float, default=20.0,
               help="扫把**额外抬柄**角 (只作用于 object_0)。重建只给 7° 倾角, 人扫地不可能那样: "
                    "0° 时右小臂扎进桌面 4.2cm/掌根 0.3cm, 按 die_table 一开跑就死。"
                    "实测 (2026-09-11): 0/10/20/30° -> 小臂离桌 -4.2/+0.7/+6.4/+13.6cm, "
                    "掌根 -0.3/+3.6/+7.7/+13.4cm, 指垫 +0.3/+2.3/+4.5/+8.3cm; 臂 IK 质量全程不变 "
                    "(中位 0.14~0.16cm)。用户 2026-09-11 裁定取 20° (总倾角 7+20=27°, 落在人握手刷的 20~40°)。")
p.add_argument("--pan_tilt_deg", type=float, default=10.0,
               help="簸箕**抬柄**角 (只作用于 object_1)。同一性质的修法: 真实簸箕就是柄翘起、唇贴地。"
                    "0° 时左手指垫贴着桌面 (小指 +0.0cm/无名 +0.7cm, 垫面其实已在桌下)。"
                    "实测 0/5/10/15/20° -> 指垫离桌 -0.0/+0.9/+2.0/+3.3/+2.3cm, 斗倾角 1.2/5.3/10.3/15.3/20.3°, "
                    "臂 IK 中位 0.21/0.19/0.17/0.16/0.17cm 但 **20° 时 max 崩到 6.98cm**。取 10°: "
                    "指垫 +2.0cm 安全, 斗倾角 10.3° 仍在 Sweep2 ready 闸 (<25°) 内, μ0.5 下 tan10°=0.18 立方块滑不出。")
# ③ 摆位 + 展宽
p.add_argument("--scene_yaw_deg", type=float, default=0.0)
p.add_argument("--center_x", type=float, default=-0.15)
p.add_argument("--center_y", type=float, default=0.0)
p.add_argument("--max_pos_step_m", type=float, default=0.008)
p.add_argument("--max_rot_step_deg", type=float, default=3.5)
p.add_argument("--scan", action="store_true", help="摆位扫描: 扫 yaw × 中心, 报双臂可达, 不写输出")
p.add_argument("--scan_stride", type=int, default=10)
# ---- 第五道闸: 两物体之间不许穿插 (2026-09-12 加) ----
p.add_argument("--pan_shift_mm", type=float, default=0.0,
               help="簸箕沿'远离扫把'的水平方向整体平移多少 mm。簸箕在视频里是'扶着基本不动'的角色, "
                    "位置有自由度; 扫把轨迹是真实扫地路径, 不动它。")
p.add_argument("--min_obj_gap_mm", type=float, default=2.0,
               help="两物体实际网格的最小间距闸 (mm)。<0 = 关掉此闸。")
p.add_argument("--gap_stride", type=int, default=16, help="闸的逐行采样步长")
# 2026-09-13: ArmIK.solve 的默认容差 (5mm/2.9°) 大于逐行手目标位移 (右 3.5mm / 左 0.54mm),
# 连续 IK 的"先判退出再走步"于是原封不动返回种子 ⇒ 母带出厂就是阶梯 (右臂 42.5% 行与上一行逐位相同,
# 非零步 1.737°, 二阶差分中位 1.176° = Pour 的 27 倍)。台账 §8。收紧后阶梯消失且 IK 精度好 43 倍。
p.add_argument("--prior_broom", default=None, help="覆写扫把先验名 (priors/<名>.npz), 用于换握姿做可达性验证")
p.add_argument("--prior_pan", default=None, help="覆写簸箕先验名")
p.add_argument("--ik_pos_tol_mm", type=float, default=0.2, help="臂 IK 位置收敛容差 (mm). 必须 << 逐行位移, 否则母带出阶梯")
p.add_argument("--ik_rot_tol_deg", type=float, default=0.29, help="臂 IK 姿态收敛容差 (度)")
p.add_argument("--pan_clear_mm", type=float, default=5.0,
               help="扫把经过簸箕上方时, 头底要高出簸箕顶面多少 mm。用户 2026-09-12 定 5mm。")
p.add_argument("--pan_level_roll", action="store_true",
               help="消掉簸箕的横滚 (绕其长轴水平投影转, 使局部 x 轴水平): 真簸箕放桌上是整条口沿贴桌; 重建给的 ~7° 横滚让一侧口角离桌 2.5cm, 2.5cm 方块会从口沿底下钻进去 (175 §8)")
p.add_argument("--lift_mode", default="global", choices=("global", "local", "none"),
               help="扫把过簸箕抬升的参照面: global=簸箕整体最高点; local=刷毛正下方的簸箕面; none=不抬, 全程压桌 (408 v3 实测口内刷毛也是 -0.1cm, 扫入靠物理接触; 175 用)")
p.add_argument("--lift_rim_max_mm", type=float, default=10.0,
               help="local 模式: 簸箕局部 y 高于此值的顶点 (沿/壁/柄) 不算参照面 —— 刷毛是软的, 允许压过沿; 只跟随斗底/斜坡")
p.add_argument("--pan_clear_band_cm", type=float, default=5.0,
               help="抬升的水平过渡带: 俯视投影间距 >此值 = 完全压桌, 0 = 完全抬起, 中间 smoothstep")
args = p.parse_args()


def q_to_R(q):
    q = np.asarray(q, np.float64) / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def R_to_q(R):
    q = Rot.from_matrix(np.asarray(R, np.float64)).as_quat()      # xyzw
    return np.r_[q[3], q[:3]]


# ---------------- ① 读物轨 + 换到入库网格系 + 滤波平滑 ----------------
def load_track(oi):
    """-> (R (N,3,3), p (N,3), conf (N,))  物体位姿, 已换算到**入库网格系**.

    重建物轨在 raw 网格系; 入库网格 v_in = s·v_raw − com (实测残差 0.000mm)。
    入库网格原点 = 缩放后网格的 COM, 对应 raw 系的点 com/s ⟹
      R_obj = R_rec,  p_obj = t_rec + R_rec·(com/s)
    即"让该物理点走重建轨迹, 物体绕它按 r2 尺度放大"。不缩回 raw 尺度: 放大是 r2
    交付有意为之 (sam3d_scale 偏小), 且 GraspPose 就是按放大后合成的。
    """
    sp = SPEC[oi]
    z = np.load(os.path.join(DATA, "poseqa", f"rts_sweep_dustpan_{TAKE}_object_{oi}.npz"))
    T = np.asarray(z["object_ob_in_world_smooth"], np.float64)
    com = np.asarray(json.load(open(f"{DEXO}/{sp['dx']}/info/simplified.json"))["com_offset"], np.float64) / sp["s"]
    R = T[:, :3, :3].copy()
    p = T[:, :3, 3] + np.einsum("nij,j->ni", R, com)
    conf = np.minimum(np.asarray(z["conf_pos"], np.float64), np.asarray(z["conf_rot"], np.float64)) / 100.0
    return R, p, conf


def despike(R, p, tag, verbose=True):
    """剔除跟踪跳变帧, 用两侧好帧 slerp / 线性插值补上."""
    n = len(R)
    rot = Rot.from_matrix(R)
    d_ang = np.zeros(n); d_pos = np.zeros(n)
    d_ang[1:] = (rot[:-1].inv() * rot[1:]).magnitude()
    d_pos[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1)
    bad = (d_ang > np.radians(args.rot_jump_deg)) | (d_pos > args.pos_jump_m)
    bad[0] = False
    # 一次跳变会让"跳过去"和"跳回来"两帧都超阈值; 只把**孤立**的尖峰当野点
    idx = np.flatnonzero(bad)
    good = np.ones(n, bool); good[idx] = False
    if good.sum() < n * 0.5:
        raise SystemExit(f"{tag}: 野点过半 ({(~good).sum()}/{n}), 阈值不合理")
    gi = np.flatnonzero(good)
    q = rot.as_quat()
    q = q * np.sign(np.einsum("ij,ij->i", q, np.roll(q, 1, 0)) + 1e-12)[:, None]   # 同半球
    from scipy.spatial.transform import Slerp
    sl = Slerp(gi, Rot.from_quat(q[gi]))
    t = np.clip(np.arange(n), gi[0], gi[-1])
    R2 = sl(t).as_matrix()
    p2 = np.stack([np.interp(np.arange(n), gi, p[gi, k]) for k in range(3)], 1)
    if verbose:
        print(f"[filter] {tag}: 逐帧转角 中位{np.degrees(np.median(d_ang[1:])):.2f}° max{np.degrees(d_ang.max()):.1f}° | "
              f"位移 中位{np.median(d_pos[1:])*1000:.1f}mm max{d_pos.max()*1000:.1f}mm | 剔除 {(~good).sum()}/{n} 帧")
    return R2, p2


def lowpass(R, p, win):
    """移动平均: 位置直接平均, 姿态用窗口内旋转平均 (Rot.mean)."""
    if win <= 1:
        return R, p
    n, h = len(R), win // 2
    P = np.stack([np.convolve(np.r_[[p[0, k]]*h, p[:, k], [p[-1, k]]*h],
                                  np.ones(win)/win, "valid") for k in range(3)], 1)
    rot = Rot.from_matrix(R)
    Rm = np.stack([rot[max(0, i-h):min(n, i+h+1)].mean().as_matrix() for i in range(n)])
    return Rm, P


# ---------------- ② 摆位 + z 重新设计 ----------------
MESH = {oi: np.asarray(trimesh.load(
    os.path.join(DATA, "objects", f"object_{oi}", "object_mesh_scaled_final.obj"),
    process=False).vertices, np.float64) for oi in SPEC}
HEAD = {}
for oi, sp in SPEC.items():
    V = MESH[oi]
    _seg = V[V[:, 2] > V[:, 2].min() + sp["handle"]]
    # 刚体在任意姿态下的最低点必是**凸包顶点**, 所以取凸包既精确又把点数压到几百
    HEAD[oi] = np.asarray(trimesh.convex.convex_hull(_seg).vertices, np.float64)
    print(f"[geom] object_{oi} {sp['dx']}: 网格 {len(V)} 顶点, 头/斗段 {len(_seg)} -> 凸包 {len(HEAD[oi])} "
          f"| 长轴 {(V[:,2].max()-V[:,2].min())*100:.1f}cm")


def register(R, p, yaw_deg, cx, cy, origin_xy):
    """场景配准: 绕竖轴整体转 yaw, 再把 origin_xy 平移到 (cx, cy). 纯刚体, 不含 z."""
    a = np.radians(yaw_deg)
    Rw = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
    p2 = np.einsum("ij,nj->ni", Rw, p - np.r_[origin_xy, 0.0]) + np.r_[cx, cy, 0.0]
    return np.einsum("ij,njk->nik", Rw, R), p2


def redesign_z_rowwise(R, p, oi, target_bottom_z_vec):
    """逐行整体平移 z, 使头/斗最低点落在**逐行给定**的目标高度上。"""
    H = HEAD[oi]
    out = p.copy()
    low = np.einsum("nij,kj->nki", R, H)[:, :, 2].min(axis=1) + p[:, 2]
    out[:, 2] += np.asarray(target_bottom_z_vec, float) - low
    return out, low


def redesign_z(R, p, oi, target_bottom_z):
    """逐帧整体平移 z, 使头/斗的最低点恰好落在 target_bottom_z."""
    H = HEAD[oi]
    out = p.copy()
    low = np.einsum("nij,kj->nki", R, H)[:, :, 2].min(axis=1) + p[:, 2]
    out[:, 2] += target_bottom_z - low
    return out, low


# ---------------- 共用: 造出物体轨 ----------------
def build_object_tracks(yaw_deg, cx, cy, verbose=True):
    tr = {}
    for oi in SPEC:
        R, p, conf = load_track(oi)
        R, p = despike(R, p, f"object_{oi}", verbose)
        R, p = lowpass(R, p, args.lp_win)
        tr[oi] = [R, p, conf]
    origin_xy = tr[0][1][0, :2].copy()                      # 以扫把首帧为场景原点
    for oi in SPEC:
        tr[oi][0], tr[oi][1] = register(tr[oi][0], tr[oi][1], yaw_deg, cx, cy, origin_xy)
    # 抬柄: 绕"竖直 × 长轴水平投影"转, 把柄端抬起 (头端仍落在桌面, 由随后的 z 重设保证)。
    # 只作用于扫把; 簸箕必须保持平放 —— 它的唇要贴桌, 立方块才进得去。
    for _oi, _deg in ((0, args.extra_tilt_deg), (1, args.pan_tilt_deg)):
        if abs(_deg) < 1e-9:
            continue
        R = tr[_oi][0]
        Rn = np.empty_like(R)
        for t in range(len(R)):
            ax = R[t] @ np.array([0.0, 0.0, 1.0])                 # 长轴, 局部 +z 指向头/斗端
            h = np.array([ax[0], ax[1], 0.0]); h /= max(np.linalg.norm(h), 1e-9)
            u = np.cross(np.array([0.0, 0.0, 1.0]), h); u /= max(np.linalg.norm(u), 1e-9)
            Rn[t] = Rot.from_rotvec(np.radians(_deg) * u).as_matrix() @ R[t]
        tr[_oi][0] = Rn
    # 消横滚 (2026-09-14, 175): 绕簸箕长轴 (局部 +z 的水平投影) 转, 让局部 x 轴水平 → 两口角等高; 抬柄 (pitch) 保留
    if args.pan_level_roll:
        R = tr[1][0]; Rn = np.empty_like(R); phis = []
        for t in range(len(R)):
            ax = R[t] @ np.array([0.0, 0.0, 1.0]); h = np.array([ax[0], ax[1], 0.0]); h /= max(np.linalg.norm(h), 1e-9)
            xw = R[t] @ np.array([1.0, 0.0, 0.0])
            phi = np.arctan2(xw[2], np.linalg.norm(xw[:2]))            # x 轴的仰角 = 横滚
            # 绕 h 转 -phi: x 轴在 (h, up) 平面内的分量被放平 (x ⟂ z, 所以 x 主要在 h 的法平面里)
            Rn[t] = Rot.from_rotvec(-phi * h).as_matrix() @ R[t]; phis.append(np.degrees(phi))
        tr[1][0] = Rn
        chk = [np.degrees(np.arctan2((Rn[t] @ [1., 0, 0])[2], np.linalg.norm((Rn[t] @ [1., 0, 0])[:2]))) for t in range(0, len(Rn), max(1, len(Rn)//6))]
        print(f"[pan] 消横滚: 原横滚 中位 {np.median(phis):.1f}° 范围 [{np.min(phis):.1f},{np.max(phis):.1f}] → 残余 {np.max(np.abs(chk)):.2f}°")
    # 簸箕整体平移 (第五道闸的修法): 方向 = 全程"簸箕质心 − 扫把质心"的水平分量
    if abs(args.pan_shift_mm) > 1e-9:
        _d = (tr[1][1] - tr[0][1])[:, :2].mean(0); _d /= max(np.linalg.norm(_d), 1e-9)
        tr[1][1] = tr[1][1] + np.r_[_d, 0.0] * (args.pan_shift_mm / 1000.0)
        print(f"[pan] 簸箕整体平移 {args.pan_shift_mm:.1f}mm 沿 [{_d[0]:+.3f} {_d[1]:+.3f} 0]")
    # ---- z 重设: **簸箕先做**, 扫把后做 (扫把要读簸箕的最终高度来决定抬多少) ----
    lows = {}
    tr[1][1], lows[1] = redesign_z(tr[1][0], tr[1][1], 1, TABLE_Z + args.pan_lip_gap_m)

    # ---- 扫把的 z: 经过簸箕上方时抬起来 (2026-09-12 加) ----
    # 为什么需要: 我原来给两件物体各自独立定贴桌高度 (扫把压桌下 1mm / 簸箕唇离桌 3mm),
    # 从没检查过"扫把扫到簸箕上方时会不会低于簸箕顶面"。实测 40.4% 的行扫把插进簸箕,
    # 最深 6.86mm, 而逐行分离方向的加权平均是 [-0.09,-0.42,-0.90] —— **竖直分量 0.90**,
    # 即扫把是**从上方压进**簸箕的, 不是侧撞。所以平移簸箕没用 (实测挪 3cm 只从 -6.56 到 -5.95mm,
    # 因为各行要推的水平方向互相抵消), 正确的修法是让扫把的 z 随"到簸箕的水平距离"变化。
    # 物理上也对: 真人扫地经过簸箕也要抬过去, 否则会把簸箕铲翻。
    from shapely.geometry import MultiPoint
    H0, H1 = HEAD[0], HEAD[1]
    R0, P0 = tr[0][0], tr[0][1]
    R1, P1 = tr[1][0], tr[1][1]
    n = len(R0)
    pan_top = np.einsum("nij,kj->nki", R1, H1)[:, :, 2].max(axis=1) + P1[:, 2]
    base_t = TABLE_Z - args.press_depth_m
    lift_t = pan_top + args.pan_clear_mm / 1000.0
    band = args.pan_clear_band_cm / 100.0
    tgt0 = np.empty(n); wlist = np.empty(n)
    if args.lift_mode == "local":
        # 2026-09-13 (175, 台账 175 §5): 全局最高点会把"扫进簸箕"的笔画整段抬到斗壁顶上 (175 实测刷毛离桌中位 5.4cm),
        # 而人真正做的是刷毛贴着斗底往里扫。参照面 = **每个刷毛面采样点正下方**的簸箕表面 (xy 半径 6mm 内簸箕顶点的最高 z;
        # 下方没有簸箕就是桌面), 取所有刷毛点里最高的那个 + 余量。斗底上 → 贴斗底; 有刷毛压在斗壁/柄上 → 抬到壁顶;
        # 都不在簸箕上方 → 桌面 (再按足迹距离 smoothstep 过渡, 免得台阶)。第一版用"头部凸包外扩 1cm 内的顶点"太粗
        # (头长 12cm 横跨 14cm 宽的斗, 总能碰到壁顶, 实测最大抬升 82mm), 作废。
        from scipy.spatial import cKDTree as _KD
        _Vp = np.asarray(trimesh.load(os.path.join(DATA, "objects", "object_1", "object_mesh_scaled_final.obj"), process=False).vertices)
        _Vp = _Vp[_Vp[:, 1] <= args.lift_rim_max_mm / 1000.0]                  # 只留斗底/斜坡/低沿 (簸箕局部 y ≤ rim_max)
        _Vp = _Vp[np.random.RandomState(0).choice(len(_Vp), min(30000, len(_Vp)), replace=False)]
        _Vb = np.asarray(trimesh.load(os.path.join(DATA, "objects", "object_0", "object_mesh_scaled_final.obj"), process=False).vertices)
        _Vb = _Vb[(_Vb[:, 1] < -0.046) & (_Vb[:, 2] > 0.005) & (_Vb[:, 2] < 0.110)]          # 刷毛面 (task_spec.SWEEP408 掩膜同式)
        _Vb = _Vb[np.random.RandomState(0).choice(len(_Vb), min(800, len(_Vb)), replace=False)]
        _r_xy = 0.006
    for t in range(n):
        if args.lift_mode == "none":
            wlist[t] = 0.0; tgt0[t] = base_t; continue
        a = MultiPoint((H0 @ R0[t].T + P0[t])[:, :2]).convex_hull
        b = MultiPoint((H1 @ R1[t].T + P1[t])[:, :2]).convex_hull
        d = a.distance(b)                                     # 俯视投影间距 (重叠时为 0)
        u = np.clip(1.0 - d / max(band, 1e-9), 0.0, 1.0)
        w = u * u * (3.0 - 2.0 * u)                            # smoothstep, 免得 z 有台阶
        wlist[t] = w
        if args.lift_mode == "local" and w > 1e-6:
            Wp = _Vp @ R1[t].T + P1[t]; Wb = _Vb @ R0[t].T + P0[t]
            kd = _KD(Wp[:, :2]); groups = kd.query_ball_point(Wb[:, :2], r=_r_xy)
            under = [Wp[g, 2].max() for g in groups if g]                       # 每个刷毛点正下方的簸箕表面高
            if under:
                loc_top = max(under); tgt0[t] = max(loc_top + args.pan_clear_mm / 1000.0, base_t)   # 直接压在簸箕上: 不再按 w 混合
            else:
                tgt0[t] = base_t                                                   # 刷毛下方没有簸箕面: 贴桌, 不预抬 (台阶由细分吸收)
        else:
            tgt0[t] = (1 - w) * base_t + w * max(lift_t[t], base_t)
    tr[0][1], lows[0] = redesign_z_rowwise(tr[0][0], tr[0][1], 0, tgt0)
    if verbose:
        m = wlist > 0.01
        print(f"[lift] 扫把过簸箕抬升 ({args.lift_mode}): 受影响 {int(m.sum())}/{n} 行 ({100*m.mean():.1f}%) | "
              f"最大抬升 {np.max(tgt0 - base_t)*1000:.1f}mm | 簸箕顶面 {np.median(pan_top - TABLE_Z)*1000:.1f}mm 离桌 | "
              f"余量 {args.pan_clear_mm:.0f}mm 过渡带 {args.pan_clear_band_cm:.0f}cm")
    return tr, lows


# ---------------- ③ 手 = 物体 ∘ T_oh, 连续臂 IK ----------------
from rl_rebuild.correction.kinematics import ArmIK                                  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER     # noqa: E402

anchor_T = np.asarray(json.load(open(os.path.join(ROOT, args.anchor)))["anchor_T"], np.float64)
if args.prior_broom:
    SPEC[0]["prior"] = args.prior_broom
if args.prior_pan:
    SPEC[1]["prior"] = args.prior_pan
PRIOR = {oi: np.load(os.path.join(ROOT, "tasks/pregrasp/priors", SPEC[oi]["prior"] + ".npz"))
         for oi in SPEC}
T_OH = {oi: (np.asarray(PRIOR[oi]["grasp"], np.float64)[:3],
             q_to_R(np.asarray(PRIOR[oi]["grasp"], np.float64)[3:7])) for oi in SPEC}
IK = {oi: ArmIK(SPEC[oi]["hand"], anchor_link="arm_center", anchor_T=anchor_T) for oi in SPEC}


def wrist_targets(R, p, oi):
    """手 = 物体 ∘ T_oh —— 与 probe_hold 的 traj=obj 同一条公式."""
    p_oh, R_oh = T_OH[oi]
    return p + np.einsum("nij,j->ni", R, p_oh), np.einsum("nij,jk->nik", R, R_oh)


def solve_continuous(oi, P, Q, seed=None, iters=150):
    """逐帧连续 IK, 上一帧解做下一帧种子 (关节不跳)."""
    ik = IK[oi]
    q = seed if seed is not None else ik.q_default
    out, perr, rerr = [], [], []
    for t in range(len(P)):
        r = ik.solve(P[t], Q[t], q0=q, iters=iters,
                     pos_tol=args.ik_pos_tol_mm / 1000.0, rot_tol=np.radians(args.ik_rot_tol_deg))
        q = r["q"]; out.append(q.copy()); perr.append(r["pos_err"]); rerr.append(r["rot_err"])
    return np.asarray(out), np.asarray(perr), np.degrees(np.asarray(rerr))


# ---------------- 摆位扫描 ----------------
if args.scan:
    print("\n=== 摆位扫描 (双臂可达; 手 = 物体 ∘ T_oh, z 已按接触重设) ===")
    print("  判据: 两臂 IK 位置误差 (中位/最大/超5cm帧数) + 双腕最小间距 (碰撞代理)")
    rows = []
    for yaw in range(-180, 180, 30):
        for cx in (-0.25, -0.20, -0.15, -0.10):
            for cy in (-0.10, 0.0, 0.10):
                tr, _ = build_object_tracks(yaw, cx, cy, verbose=False)
                st = args.scan_stride
                res, wp = {}, {}
                for oi in SPEC:
                    P, Q = wrist_targets(tr[oi][0][::st], tr[oi][1][::st], oi)
                    _, pe, re_ = solve_continuous(oi, P, Q, iters=80)
                    res[oi] = (np.median(pe), pe.max(), int((pe > 0.05).sum()), np.median(re_))
                    wp[oi] = P
                gap = np.linalg.norm(wp[0] - wp[1], axis=1)
                score = max(res[0][0], res[1][0]) + 0.5 * max(res[0][1], res[1][1])
                rows.append((score, yaw, cx, cy, res, gap.min(), len(wp[0])))
    rows.sort(key=lambda r: r[0])
    print(f"\n  {'yaw':>5} {'cx':>6} {'cy':>6} | {'右臂 中位/max/>5cm':>22} | {'左臂 中位/max/>5cm':>22} | 腕距min")
    for score, yaw, cx, cy, res, gmin, n in rows[:15]:
        f = lambda r: f"{r[0]*100:5.2f}/{r[1]*100:5.2f}cm/{r[2]:2d}"
        print(f"  {yaw:5d} {cx:6.2f} {cy:6.2f} | {f(res[0]):>22} | {f(res[1]):>22} | {gmin*100:5.1f}cm")
    print(f"\n  (共扫 {len(rows)} 组, 每组 {rows[0][6]} 帧抽样; 按 max(两臂中位)+0.5·max(两臂最大) 排序)")
    raise SystemExit(0)

# ---------------- 造带 ----------------
tr, lows = build_object_tracks(args.scene_yaw_deg, args.center_x, args.center_y)
NSRC = len(tr[0][0])
for oi in SPEC:
    R, p = tr[oi][0], tr[oi][1]
    H = HEAD[oi]
    low = np.array([(H @ R[t].T + p[t])[:, 2].min() for t in range(NSRC)])
    print(f"[z] object_{oi}: 重设前 头/斗最低点 {lows[oi].min():.4f}~{lows[oi].max():.4f}m "
          f"(离桌 {np.median(lows[oi]-TABLE_Z)*100:+.1f}cm 中位) -> 重设后 {low.min():.4f}~{low.max():.4f}m "
          f"(离桌 {(np.median(low)-TABLE_Z)*100:+.2f}cm, 残差 {np.abs(low-np.median(low)).max()*1000:.3f}mm)")

# 时间展宽 (Sweep2 同款: 只放慢不改路径)
src_fps = 15.0
dur = (NSRC - 1) / src_fps
base = np.arange(0.0, dur + 1e-9, 1.0 / args.control_hz) * src_fps
if base[-1] < NSRC - 1:
    base = np.r_[base, NSRC - 1.0]
base = np.unique(np.r_[base, 0.0, NSRC - 1.0])


def interp(oi, times):
    R, p = tr[oi][0], tr[oi][1]
    lo = np.clip(np.floor(times).astype(int), 0, NSRC - 2)
    u = (times - lo)[:, None]
    P = p[lo] * (1 - u) + p[lo + 1] * u
    from scipy.spatial.transform import Slerp
    sl = Slerp(np.arange(NSRC), Rot.from_matrix(R))
    return sl(np.clip(times, 0, NSRC - 1)).as_matrix(), P


def rot_ang(A, B):
    return float(np.arccos(np.clip((np.trace(A.T @ B) - 1.0) / 2.0, -1.0, 1.0)))


bt = {oi: interp(oi, base) for oi in SPEC}
times = [float(base[0])]
for r in range(1, len(base)):
    ratio = 1.0
    for oi in SPEC:
        Rr, Pp = bt[oi]
        ratio = max(ratio, np.linalg.norm(Pp[r] - Pp[r-1]) / args.max_pos_step_m,
                    rot_ang(Rr[r-1], Rr[r]) / np.radians(args.max_rot_step_deg))
    times.extend(np.linspace(base[r-1], base[r], int(np.ceil(ratio)) + 1)[1:].tolist())
times = np.asarray(times, np.float64)


def steps_of(times):
    """逐行实际步长 (位移 m, 转角 rad), 取两物体的逐行最大."""
    f = {oi: interp(oi, times) for oi in SPEC}
    n = len(times)
    ps = np.zeros(n - 1); rs = np.zeros(n - 1)
    for oi in SPEC:
        R, P = f[oi]
        ps = np.maximum(ps, np.linalg.norm(np.diff(P, axis=0), axis=1))
        rs = np.maximum(rs, np.asarray([rot_ang(R[t], R[t+1]) for t in range(n-1)]))
    return f, ps, rs


# **迭代细分**: 初次细分是在 base 结点之间线性插的, 而 Slerp 的结点在整数源帧上 ——
# 跨结点时角速率会变, 于是实测步长可能仍超界 (实测 3.69° > 3.5°)。这里直接按实测
# 步长把越界的区间对半插, 反复到全部达标, 比"对齐结点"更稳且与路径无关。
for _round in range(12):
    fin, ps, rs = steps_of(times)
    bad = (ps > args.max_pos_step_m) | (rs > np.radians(args.max_rot_step_deg))
    if not bad.any():
        break
    mids = 0.5 * (times[:-1][bad] + times[1:][bad])
    times = np.unique(np.r_[times, mids])
    print(f"[tape] 细分第 {_round+1} 轮: {int(bad.sum())} 个区间越界 -> 插入中点, 行数 {len(times)}")
else:
    raise SystemExit("时间展宽未收敛, 检查 max_pos_step_m / max_rot_step_deg")
fin, ps, rs = steps_of(times)
NF = len(times)
print(f"[tape] 源 {src_fps:.0f}Hz {NSRC} 帧 -> 规则 {len(base)} 行 -> 展宽 {NF} 行 ({NF/args.control_hz:.1f}s @{args.control_hz:.0f}Hz)")
print(f"[tape] 展宽后逐行 (两物取大): 位移 max {ps.max()*1000:.2f}mm | 转角 max {np.degrees(rs.max()):.2f}°")
assert ps.max() <= args.max_pos_step_m + 1e-9 and rs.max() <= np.radians(args.max_rot_step_deg) + 1e-9

# ---------- 第五道闸: 两物体不许穿插 ----------
# 2026-09-12 加。**此前四道闸 (滤波/z重设/时间展宽/臂IK) 没有一道看两个物体之间**, 于是
# 母带带着穿插出厂: 实测 40.4% 的行扫把插进簸箕, 最深 6.86mm, 第 0 行就有 82 个顶点插 3.3mm。
# 后果: 物理一开 PhysX 就在互推两个 0.15/0.20kg 的物体 —— GUI 里表现为"提扫把带起簸箕",
# 训练里则是一股说不清的额外扰动 (term/die_rel 87% 里有多少是它, 当时分不出来)。
# 根因: 扫把抬柄 20° / 簸箕抬柄 10° / 各自 z 重设, 三步都绕**各自**的轴独立做, 相对关系被改却没复核。
if args.min_obj_gap_mm >= 0:
    import trimesh as _tm
    _PQ = _tm.proximity.ProximityQuery(_tm.load(
        os.path.join(DATA, "objects", "object_1", "object_mesh_scaled_final.obj"), process=False))
    _VB = MESH[0][::80]
    _rows = np.arange(0, NF, int(args.gap_stride))
    _worst, _wrow = 1e9, -1
    for _t in _rows:
        _Rb, _Pb = fin[0][0][_t], fin[0][1][_t]
        _Rp, _Pp = fin[1][0][_t], fin[1][1][_t]
        _loc = ((_VB @ _Rb.T + _Pb) - _Pp) @ _Rp
        _g = -_PQ.signed_distance(_loc).max()
        if _g < _worst:
            _worst, _wrow = _g, int(_t)
    print(f"[gate5] 两物体最小间距 {_worst*1000:+.2f}mm (最差在行 {_wrow}; 闸 ≥{args.min_obj_gap_mm:.1f}mm, "
          f"采样每 {args.gap_stride} 行)")
    assert _worst * 1000 >= args.min_obj_gap_mm, (
        f"两物体穿插/贴太近: 最小间距 {_worst*1000:+.2f}mm < {args.min_obj_gap_mm}mm (行 {_wrow})。"
        f" 用 --pan_shift_mm 把簸箕挪开, 或 --min_obj_gap_mm -1 显式关闸 (不建议)。")

# 臂 IK
ARMQ, REPORT = {}, {}
for oi in SPEC:
    R, P = fin[oi]
    wp, wq = wrist_targets(R, P, oi)
    q, pe, re_ = solve_continuous(oi, wp, wq)
    js = np.degrees(np.abs(np.diff(q, axis=0))).max() if NF > 1 else 0.0
    ARMQ[oi] = q.astype(np.float32)
    REPORT[SPEC[oi]["hand"]] = dict(ok_ratio=float((pe < 0.02).mean()), pos_max_cm=float(pe.max()*100),
                                    pos_med_cm=float(np.median(pe)*100), rot_max_deg=float(re_.max()),
                                    joint_step_max_deg=float(js))
    print(f"[ik] {SPEC[oi]['hand']:5s} 位置 中位 {np.median(pe)*100:.2f}cm max {pe.max()*100:.2f}cm | "
          f"姿态 max {re_.max():.1f}° | 关节单步 max {js:.2f}° | <2cm 占比 {(pe<0.02).mean()*100:.1f}%")

# 指行: squeeze 常量 + grasp 备查 (口径同 Clean/3, env 自己做 grasp->squeeze 斜坡)
FIN = {oi: (np.asarray(PRIOR[oi]["grasp"], np.float64)[7:29],
            np.asarray(PRIOR[oi]["squeeze"], np.float64)[7:29]) for oi in SPEC}
conf_src = {oi: tr[oi][2] for oi in SPEC}
conf = {oi: np.interp(times, np.arange(NSRC), conf_src[oi]).astype(np.float32) for oi in SPEC}
RH, LH = 0, 1                                   # object_0=扫把(右) object_1=簸箕(左)
out = {
    "right_q": ARMQ[RH], "left_q": ARMQ[LH],
    "right_f": np.repeat(FIN[RH][1][None], NF, 0).astype(np.float32),
    "left_f": np.repeat(FIN[LH][1][None], NF, 0).astype(np.float32),
    "right_f_grasp": FIN[RH][0].astype(np.float32), "left_f_grasp": FIN[LH][0].astype(np.float32),
    "fin_names": np.asarray(GENERIC_JOINT_ORDER, dtype=object),
    "source_frame": times.astype(np.float32),
    "control_hz": np.float32(args.control_hz), "source_fps": np.float32(src_fps),
    "table_z": np.float32(TABLE_Z),
    "scene_yaw_deg": np.float32(args.scene_yaw_deg),
    "scene_center_xy": np.asarray([args.center_x, args.center_y], np.float32),
    "press_depth_m": np.float32(args.press_depth_m),
    "extra_tilt_deg": np.float32(args.extra_tilt_deg),
    "pan_tilt_deg": np.float32(args.pan_tilt_deg),
    "pan_lip_gap_m": np.float32(args.pan_lip_gap_m),
    "lp_win": np.int32(args.lp_win), "rot_jump_deg": np.float32(args.rot_jump_deg),
    "ik_pos_tol_mm": np.float32(args.ik_pos_tol_mm), "ik_rot_tol_deg": np.float32(args.ik_rot_tol_deg),
    "anchor_T": anchor_T.astype(np.float32),
    "obj_pos_0": fin[0][1].astype(np.float32), "obj_quat_0": np.stack([R_to_q(r) for r in fin[0][0]]).astype(np.float32),
    "obj_pos_1": fin[1][1].astype(np.float32), "obj_quat_1": np.stack([R_to_q(r) for r in fin[1][0]]).astype(np.float32),
    "confidence_0": conf[0], "confidence_1": conf[1],
    "scene_pose_0": np.r_[fin[0][1][0], R_to_q(fin[0][0][0])].astype(np.float32),
    "scene_pose_1": np.r_[fin[1][1][0], R_to_q(fin[1][0][0])].astype(np.float32),
    "ik_report": json.dumps(REPORT, ensure_ascii=False),
    "meta": (f"Sweep{TAKE} v1: 右手扫把/左手簸箕; 物轨去尖峰+低通; **z 按接触重设不回放** "
             f"(扫把头底压桌下{args.press_depth_m*1000:.0f}mm, 簸箕斗底离桌{args.pan_lip_gap_m*1000:.0f}mm, "
             "依据: 簸箕静止时 z 噪声 std 0.8cm, 重建竖直方向不可信); 手=物体∘GraspPose(输入系); "
             f"扫把抬柄{args.extra_tilt_deg:.0f}° (0° 时右小臂扎桌 4.2cm) / 簸箕抬柄{args.pan_tilt_deg:.0f}° (0° 时左指垫贴桌); "
             "连续臂 IK; 指=squeeze 常量; 15Hz->20Hz 保结点时间展宽只放慢不改路径"),
}
op = os.path.join(ROOT, args.output)
os.makedirs(os.path.dirname(op), exist_ok=True)
np.savez(op, **out)
print(f"\n[out] {op}  ({NF} 行, {len(out)} 键)")
