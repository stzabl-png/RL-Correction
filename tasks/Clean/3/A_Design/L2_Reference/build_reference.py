"""Clean/3 母带构建器 (擦盘子, 双手持物起步, 真摩擦握).

数据事实 (2026-09-07 实测, 台账 §5): 双腕冻结; 盘 rts 姿态不可用 (圆盘绕法向自转不可观测,
conf_rot 16) 但法向倾角稳定 ~39°; 海绵与盘面在重建里**不贴合** (盘系法向距离中位 12.6cm)。
因此母带不是"回放重建", 而是**用重建的相对擦拭图案 + 功能几何重新贴合**:

  盘   : 位置 = rts 位置轨低通; 姿态 = 常量 (水平 + 可选小倾角), 用户裁定。
  海绵 : 盘面内 XY = 重建的海绵在盘系内的面内偏移 (--sponge_xy_mode plate, 默认)
         或世界水平偏移 (world); 法向高度 = 贴合盘面径向剖面 top(r) + 擦盘面偏移 - 压深;
         姿态 = 擦盘面朝盘 (canonical z-up) + 面内 yaw ψ(t) (重建长轴投影) + 常量偏置。
  腕   : T_world_hand = T_world_obj × T_obj_hand (GraspPose prior, 输入系), 连续臂 IK
         (Sweep2 的 稀疏多解 DP + 稠密局部精修)。指 = squeeze 常量。
  时钟 : 15Hz 重建 → 20Hz, 保留关键结点, 只对超过 0.8cm/3.5° 的步做时间展宽 (Sweep2 同款)。

用法:
  PY=/home/lyh/luhr/MagicSim/.venv/bin/python
  $PY tasks/Clean/3/A_Design/L2_Reference/build_reference.py --scan_yaw      # 双臂可达扫描
  $PY tasks/Clean/3/A_Design/L2_Reference/build_reference.py --plate_yaw_deg <a> --sponge_yaw_deg <b>
"""
from __future__ import annotations

import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--output", default="tasks/Clean/3/A_Design/L2_Reference/clean3_reference_v1.npz")
p.add_argument("--control_hz", type=float, default=20.0)
p.add_argument("--plate_prior", default="tasks/pregrasp/priors/Clean3_plate_left.npz")
p.add_argument("--sponge_prior", default="tasks/pregrasp/priors/Clean3_sponge_right.npz")
p.add_argument("--anchor", default="tasks/Sweep/2/A_Design/L2_Reference/anchor_T_sweep2.json",
               help="arm_center 实测锚 (同一 GraspTaskCfg 机器人; A0 探针核对 runtime_anchor_T)")
p.add_argument("--table_z", type=float, default=0.87)
p.add_argument("--plate_center_x", type=float, default=-0.12)
p.add_argument("--plate_center_y", type=float, default=0.06)
p.add_argument("--plate_height", type=float, default=0.12, help="盘心离桌面高度 (m)")
p.add_argument("--plate_yaw_deg", type=float, default=0.0, help="盘规范系绕世界 z 的 yaw")
p.add_argument("--plate_tilt_deg", type=float, default=0.0, help="盘法向绕世界 x 的小倾角")
p.add_argument("--plate_flip", action="store_true",
               help="盘倒扣 (擦外底): 只把**盘网格**绕规范 x 转 180°, 规范系仍 z-up —— 海绵照样在盘上方、"
                    "右手不受影响, 只有左手(端盘)跟着翻。擦拭面剖面自动改用网格另一面。\n"
                    "⚠⚠ 2026-09-10 实测: **本旗单独用会让盘脱手** —— 现有 GraspPose(27_Quadpod, 拇指压顶面/四指托底) "
                    "在倒扣下 3s 静持左手漂 33cm/176°, 盘直接掉。倒扣必须配一套**为倒扣盘合成的新 GraspPose**, "
                    "见台账 §5.13。本旗保留只为那条路线准备, 单用无效。\n"
                    "⚠ 别拿 --plate_tilt_deg 180 代替: 那个转的是规范系, 会把海绵和右手一起翻到盘底下。")
p.add_argument("--sponge_yaw_deg", type=float, default=0.0, help="海绵面内 yaw 常量偏置")
p.add_argument("--sponge_xy_mode", choices=("plate", "world"), default="plate")
p.add_argument("--sponge_xy_recenter", action="store_true",
               help="把海绵面内图案的中位平移到盘心 (重建里海绵长期偏在盘外时用; take1 需要). "
                    "与 plate_motion_scale 的 p_ref 归心同构")
p.add_argument("--sponge_xy_scale", type=float, default=1.0,
               help="海绵面内图案幅度缩放 (归心之后再乘). 1.0=原样")
p.add_argument("--no_hand_ref", action="store_true",
               help="不写人手指姿层 human_*_f (默认写)。人手层只影响**指列**, 且只有 env 开 CLEAN_S2_HAND_REF 才生效, "
                    "所以默认写进去不改变任何现役行为 —— 这是 Ablation Base/A1 共用同一条母带的前提 (§5.15)")
p.add_argument("--min_wrist_gap_cm", type=float, default=10.0,
               help="双腕间距下限 (cm). 稠密 IK 之后核, 不过就拒绝写盘 —— 2026-09-10 教训: "
                    "只按可达+覆盖选 yaw 会选出'两手叠在胸口'的解 (take18 首版 min 3.2cm), 而 IK 照样全通。"
                    "0 = 关掉这道闸 (只打印不拒绝)")
p.add_argument("--psi_mode", choices=("track", "const"), default="track",
               help="海绵面内 yaw: track=重建长轴投影 ψ(t) (可限幅), const=中位数常量")
p.add_argument("--psi_clip_deg", type=float, default=180.0, help="track 模式下 |ψ-中位| 限幅")
p.add_argument("--psi_lp_s", type=float, default=0.5, help="track 模式下 ψ 低通窗 (s)")
p.add_argument("--press_depth_m", type=float, default=0.002)
p.add_argument("--sponge_pitch_deg", type=float, default=0.0,
               help="海绵绕自身短轴(规范 x)俯仰, 让一端探进碟心 (刚体海绵 D4 缺口的折中); 0=平放")
p.add_argument("--geometry_only", action="store_true",
               help="只算参考几何 (判据模块口径的覆盖率/行程 + 稀疏可达), 不做稠密 IK 不写文件")
p.add_argument("--lp_window_s", type=float, default=0.67, help="盘位置低通窗 (s)")
p.add_argument("--plate_motion_scale", type=float, default=1.0)
p.add_argument("--scene_yaw_deg", type=float, default=0.0)
p.add_argument("--max_pos_step_m", type=float, default=0.008)
p.add_argument("--max_rot_step_deg", type=float, default=3.5)
p.add_argument("--scan_yaw", action="store_true", help="扫 (盘 yaw × 海绵 yaw) 的双臂稀疏 IK 可达率后退出")
p.add_argument("--scan_step_deg", type=float, default=30.0)
p.add_argument("--scan_json", default=None,
               help='组合扫描: JSON 列表, 每项可覆写 plate_yaw_deg/sponge_yaw_deg/plate_motion_scale/'
                    'plate_center_x/plate_center_y/plate_height; 逐项报双臂稀疏可达率后退出')
p.add_argument("--audit_side", choices=("right", "left"), default=None)
p.add_argument("--take", default="3",
               help="运动来源 take (rts 物轨 + replay_world 手轨). 资产(网格/先验)恒用 --assets_take, "
                    "因为多母带训练要求三条 take 共用同一套物体资产 (台账 §5.12)")
p.add_argument("--assets_take", default="3",
               help="物体网格来源 take (盘顶面剖面 / 海绵擦盘面偏移). 默认 3 = 已验证的 18cm 盘 + 13.2cm 海绵")
p.add_argument("--src_frames", default=None,
               help="源帧截取 'a:b' (半开区间, 0 基, 作用在 rts/replay 的 15Hz 源帧上)。默认全长。take 8 的抹布尾段 [293,299] 手把布整个攥住, 是主判据 d_cent 抓到的真错位, 用 --src_frames 0:293 截掉 (丢 0.47s)。")
p.add_argument("--sponge_assets_take", default=None,
               help="海绵资产单独来源 take (默认跟 --assets_take)。take18 倒扣盘线用 --assets_take 18 --sponge_assets_take 3: 盘用 take18 自己的(倒扣, 与新先验同尺), 布仍用 take3 的"
               " (take18 的布是薄壳/长轴短 40%%, 功能池只有 38 条, 用户 2026-09-10 裁定)")
p.add_argument("--no_layout", action="store_true", help="不写 datasets/.../scene_layout.json")
p.add_argument("--prelude", action="store_true",
               help="母带前面加抓稳前奏: 接近 K_app 行 (prior pregrasp→grasp) + 合拢 K_close 行 (指 grasp→squeeze) + 静持 K_settle 行; "
                    "前奏期物体钉在母带 0 行位姿 (Stage-1 抓稳段 env 用, 台账 §5.6)")
p.add_argument("--k_app", type=int, default=20); p.add_argument("--k_close", type=int, default=10)
p.add_argument("--k_settle", type=int, default=20)
p.add_argument("--oh_override", default=None,
               help="probe_hold 产出的 settled_oh_<tag>.json: 用物理沉降后的手-物相对位姿代替 GraspPose 名义位姿做腕目标 (v2 重锚)")
args = p.parse_args()

import numpy as np  # noqa: E402

from rl_rebuild.correction.kinematics import ArmIK  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
_DS = os.path.join(ROOT, "datasets", "clean_tableware")
DATA = os.path.join(_DS, str(args.take))              # 运动来源 (rts/replay/输出 layout)
ASSETS = os.path.join(_DS, str(args.assets_take))     # 物体资产 (网格)
SPONGE_ASSETS = os.path.join(_DS, str(args.sponge_assets_take or args.assets_take))   # 海绵资产 (可与盘分家)
PLATE_RIM_R = 0.085          # 径向剖面最外档 (m)


# ---------------------------------------------------------------- 旋转工具 ----
def q_to_R(q):
    q = np.asarray(q, np.float64); q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def R_to_q(R):
    m = np.asarray(R, np.float64)
    vals = np.array([1+m.trace(), 1+m[0, 0]-m[1, 1]-m[2, 2],
                     1-m[0, 0]+m[1, 1]-m[2, 2], 1-m[0, 0]-m[1, 1]+m[2, 2]])
    i = int(np.argmax(vals)); q = np.zeros(4)
    if i == 0:
        q[0] = 0.5*np.sqrt(max(vals[0], 0)); d = max(4*q[0], 1e-12)
        q[1:] = [(m[2, 1]-m[1, 2])/d, (m[0, 2]-m[2, 0])/d, (m[1, 0]-m[0, 1])/d]
    else:
        j, k = (i % 3) + 1, ((i + 1) % 3) + 1
        q[i] = 0.5*np.sqrt(max(vals[i], 0)); d = max(4*q[i], 1e-12)
        q[0] = (m[k-1, j-1]-m[j-1, k-1])/d
        q[j] = (m[i-1, j-1]+m[j-1, i-1])/d
        q[k] = (m[i-1, k-1]+m[k-1, i-1])/d
    return q / np.linalg.norm(q)


def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose_T(pos, R_or_q):
    T = np.eye(4); R = np.asarray(R_or_q)
    T[:3, :3] = R if R.shape == (3, 3) else q_to_R(R); T[:3, 3] = pos; return T


def rotation_angle(R0, R1):
    return float(np.arccos(np.clip((np.trace(R0.T @ R1) - 1.0) / 2.0, -1.0, 1.0)))


def interp_T(track, times):
    n = len(track); out = np.zeros((len(times), 4, 4))
    for k, t in enumerate(times):
        i = min(int(np.floor(t)), n - 2); a = float(t - i)
        pos = (1-a)*track[i, :3, 3] + a*track[i+1, :3, 3]
        q0, q1 = R_to_q(track[i, :3, :3]), R_to_q(track[i+1, :3, :3])
        if np.dot(q0, q1) < 0: q1 = -q1
        dot = float(np.clip(np.dot(q0, q1), -1, 1))
        if dot > 0.9995:
            q = q0 + a*(q1-q0); q /= np.linalg.norm(q)
        else:
            th = np.arccos(dot); q = (np.sin((1-a)*th)*q0 + np.sin(a*th)*q1)/np.sin(th)
        out[k] = pose_T(pos, q)
    return out


def moving_average(x, win):
    win = max(int(win), 1)
    if win == 1: return x.copy()
    pad = win // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], win - 1 - pad, 0)], 0)
    k = np.ones(win) / win
    return np.stack([np.convolve(xp[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


# ------------------------------------------------------------------ 数据 ----
def load_rts(oi):
    return np.load(os.path.join(DATA, "poseqa", f"rts_clean_tableware_{args.take}_object_{oi}.npz"))


zr = {i: load_rts(i) for i in (0, 1)}
raw = {i: np.asarray(zr[i]["object_ob_in_world_smooth"], np.float64) for i in (0, 1)}
replay = np.load(os.path.join(DATA, "replay_world.npz"), allow_pickle=True)
source_fps = float(np.asarray(replay["fps"]))
assert source_fps == 15.0, source_fps
SRC_A, SRC_B = 0, len(raw[0])
if args.src_frames:                      # 源帧截取 (默认不截, take3/18 逐位复现)
    _a, _b = args.src_frames.split(":")
    SRC_A = int(_a) if _a else 0
    SRC_B = int(_b) if _b else len(raw[0])
    assert 0 <= SRC_A < SRC_B <= len(raw[0]), f"--src_frames {args.src_frames} 越界 (共 {len(raw[0])} 帧)"
    raw = {i: raw[i][SRC_A:SRC_B] for i in (0, 1)}
    print(f"[reference] 源帧截取 [{SRC_A}, {SRC_B}) -> {SRC_B - SRC_A} 帧 "
          f"(原 {len(zr[0]['object_ob_in_world_smooth'])}, 丢 {(len(zr[0]['object_ob_in_world_smooth']) - (SRC_B - SRC_A)) / source_fps:.2f}s)")
NF = len(raw[0])
priors = {"left": np.load(os.path.join(ROOT, args.plate_prior)),
          "right": np.load(os.path.join(ROOT, args.sponge_prior))}
R_ci = q_to_R(priors["left"]["canon_rot"])             # 输入系 -> 规范系 (两物体同为 Rx(+90°))
assert np.allclose(R_ci, q_to_R(priors["right"]["canon_rot"]), atol=1e-6)
assert np.allclose(R_ci @ np.array([0, 1, 0]), [0, 0, 1], atol=1e-6), "输入 +y 应映射到规范 +z"
with open(os.path.join(ROOT, args.anchor)) as f:
    anchor_T = np.asarray(json.load(f)["anchor_T"], np.float64)
T_OH = {s_: np.asarray(priors[s_]["grasp"][:7], np.float64) for s_ in ("left", "right")}
if args.oh_override:
    _so = json.load(open(os.path.join(ROOT, args.oh_override)))["settled"]
    for s_ in ("left", "right"):
        T_OH[s_] = np.asarray(_so[s_]["pos"] + _so[s_]["quat_wxyz"], np.float64)
        d = T_OH[s_] - priors[s_]["grasp"][:7]
        print(f"[reference] {s_} T_obj_hand 覆写为沉降位姿: Δpos={np.round(d[:3]*100,2).tolist()}cm "
              f"(spread {_so[s_]['pos_spread_cm']:.2f}cm)")


def load_obj_vertices(path):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")])


def _extent_cm(v):
    """网格 bbox 边长 (cm, 输入系 xyz) —— 换资产 take 后不能再写死。"""
    return [round(float(x) * 100.0, 2) for x in (np.asarray(v).max(0) - np.asarray(v).min(0))]


def plate_top_profile():
    """盘顶面径向剖面 (输入系: 法向 +y): r 档 -> 顶面 y 最大值。"""
    v = load_obj_vertices(os.path.join(ASSETS, "objects", "object_0", "object_mesh_scaled_final.obj"))
    r = np.linalg.norm(v[:, [0, 2]], axis=1)
    # 规范系 z = 输入 +y; 倒扣后规范系 z = 输入 -y ⟹ 朝上的那一面换成另一面
    y = -v[:, 1] if args.plate_flip else v[:, 1]
    edges = np.linspace(0.0, PLATE_RIM_R + 0.005, 19)
    rs, tops = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.any():
            rs.append(0.5 * (a + b)); tops.append(float(y[m].max()))
    return np.asarray(rs), np.asarray(tops)


prof_r, prof_top = plate_top_profile()
sp_v = load_obj_vertices(os.path.join(SPONGE_ASSETS, "objects", "object_1", "object_mesh_scaled_final.obj"))
SPONGE_FACE_OFFSET = float(-sp_v[:, 1].min())    # 擦盘面 (输入 -y) 到海绵原点的距离


def top_at(r):
    return np.interp(np.clip(r, prof_r[0], prof_r[-1]), prof_r, prof_top)


# ----------------------------------------------------- 源率 (15Hz) 母带轨迹 ----
def build_source_tracks(plate_yaw_deg, sponge_yaw_deg):
    """返回两物体在 RL 世界系的 4x4 轨迹 (15Hz, NF 帧) 与诊断量。"""
    T0, T1 = raw[0], raw[1]
    # ---- 盘: 位置低通 + 配准; 姿态常量 ----
    p0 = moving_average(T0[:, :3, 3], round(args.lp_window_s * source_fps))
    p_ref = np.median(p0, axis=0)
    R_scene = Rz(np.radians(args.scene_yaw_deg))
    origin = np.array([args.plate_center_x, args.plate_center_y, args.table_z + args.plate_height])
    plate_pos = origin + args.plate_motion_scale * (R_scene @ (p0 - p_ref).T).T
    R_plate_can = Rz(np.radians(plate_yaw_deg)) @ Rx(np.radians(args.plate_tilt_deg))   # 规范系(z-up)在世界
    R_flip = Rx(np.pi) if args.plate_flip else np.eye(3)      # 只翻网格, 不动规范系
    R_plate_in = R_plate_can @ R_flip @ R_ci                                             # 输入系在世界
    # ---- 海绵: 面内偏移 + 贴合高度 + 面内 yaw ----
    d_world = T1[:, :3, 3] - T0[:, :3, 3]
    a_world = T1[:, :3, :3] @ np.array([0.0, 0.0, 1.0])            # 海绵长轴 (输入 z) 在重建世界
    if args.sponge_xy_mode == "plate":
        Rp = T0[:, :3, :3]                                          # 重建盘输入系
        d_can = np.einsum("nji,nj->ni", Rp, d_world) @ R_ci.T       # 盘输入系 -> 盘规范系
        a_can = np.einsum("nji,nj->ni", Rp, a_world) @ R_ci.T
        xy = d_can[:, :2]
        ax, ay = a_can[:, 0], a_can[:, 1]
    else:
        xy = (R_scene @ d_world.T).T[:, :2]
        aw = (R_scene @ a_world.T).T
        ax, ay = aw[:, 0], aw[:, 1]
    # 面内图案归心 / 缩放 (与盘的 plate_motion_scale 同构; 默认恒等, take3/18 不受影响)
    if args.sponge_xy_recenter:
        _c = np.median(xy, axis=0)
        xy = xy - _c
        print(f"[reference] 海绵面内图案归心: 中位偏移 {np.round(_c*100,2).tolist()}cm 已减去")
    if abs(args.sponge_xy_scale - 1.0) > 1e-9:
        xy = xy * args.sponge_xy_scale
        print(f"[reference] 海绵面内图案幅度 ×{args.sponge_xy_scale}")
    # 海绵规范系长轴 = -y (obb 13.3cm 在 y); Rz(ψ)(0,-1,0) = (sinψ, -cosψ) ∥ (ax, ay)
    psi_raw = np.unwrap(np.arctan2(ax, -ay))
    psi_med = float(np.median(psi_raw))
    if args.psi_mode == "const":
        psi = np.full(NF, psi_med)
    else:
        psi = moving_average(psi_raw[:, None], round(args.psi_lp_s * source_fps))[:, 0]
        psi = psi_med + np.clip(psi - psi_med, -np.radians(args.psi_clip_deg), np.radians(args.psi_clip_deg))
    psi = psi + np.radians(sponge_yaw_deg)
    r = np.linalg.norm(xy, axis=1)
    # 刚体海绵落座: 13.3cm 长跨在 9cm 碟心平底与斜坡缘之间, 高度由足迹最高支撑点定 (与判据 seat_height 同定义)
    fx = np.arange(-0.03, 0.031, 0.01); fy = np.arange(-0.06, 0.061, 0.01)
    fxx, fyy = np.meshgrid(fx, fy, indexing="ij"); fxx, fyy = fxx.ravel(), fyy.ravel()
    foot = np.stack([fxx, fyy, -SPONGE_FACE_OFFSET * np.ones_like(fxx)], 1)     # 擦盘面采样点 (海绵规范系)
    R_pitch = Rx(np.radians(args.sponge_pitch_deg))
    h = np.zeros(NF)
    for t in range(NF):
        pts = (Rz(psi[t]) @ R_pitch @ foot.T).T                                   # 海绵规范系 -> 盘规范系 (未平移)
        px, py = xy[t, 0] + pts[:, 0], xy[t, 1] + pts[:, 1]
        # 落座: 使最高支撑点恰好贴面 (再压 press_depth): h = max_i (top(r_i) - z_i)
        h[t] = (top_at(np.sqrt(px**2 + py**2)) - pts[:, 2]).max() - args.press_depth_m
    sponge_T = np.zeros((NF, 4, 4)); plate_T = np.zeros((NF, 4, 4))
    for t in range(NF):
        plate_T[t] = pose_T(plate_pos[t], R_plate_in)
        R_s_can = R_plate_can @ Rz(psi[t]) @ R_pitch
        off = R_plate_can @ np.array([xy[t, 0], xy[t, 1], h[t]])
        sponge_T[t] = pose_T(plate_pos[t] + off, R_s_can @ R_ci)
    diag = dict(sponge_r=r, sponge_h=h, psi=psi, plate_pos=plate_pos,
                on_plate=(r <= PLATE_RIM_R))
    return {0: plate_T, 1: sponge_T}, diag


# ----------------------------------------------------- 20Hz 重采样 + 时间展宽 ----
def resample_expand(src):
    duration = (NF - 1) / source_fps
    base_times = np.arange(0.0, duration + 1e-9, 1.0 / args.control_hz) * source_fps
    if base_times[-1] < NF - 1:
        base_times = np.r_[base_times, NF - 1]
    critical = {0, NF - 1}
    for row in range(NF - 1):
        for i in (0, 1):
            a, b = src[i][row], src[i][row + 1]
            if (np.linalg.norm(b[:3, 3] - a[:3, 3]) > args.max_pos_step_m or
                    rotation_angle(a[:3, :3], b[:3, :3]) > np.radians(args.max_rot_step_deg)):
                critical.update((row, row + 1))
    base_times = np.unique(np.r_[base_times, sorted(critical)])
    base_tracks = {i: interp_T(src[i], base_times) for i in (0, 1)}
    source_times = [float(base_times[0])]
    for row in range(1, len(base_times)):
        ratio = 1.0
        for i in (0, 1):
            a, b = base_tracks[i][row - 1], base_tracks[i][row]
            ratio = max(ratio, np.linalg.norm(b[:3, 3] - a[:3, 3]) / args.max_pos_step_m,
                        rotation_angle(a[:3, :3], b[:3, :3]) / np.radians(args.max_rot_step_deg))
        count = int(np.ceil(ratio))
        source_times.extend(np.linspace(base_times[row - 1], base_times[row], count + 1)[1:].tolist())
    source_times = np.asarray(source_times, np.float64)
    out = {i: interp_T(src[i], source_times) for i in (0, 1)}
    for i in (0, 1):
        ps = np.linalg.norm(np.diff(out[i][:, :3, 3], axis=0), axis=1)
        rs = np.asarray([rotation_angle(a[:3, :3], b[:3, :3]) for a, b in zip(out[i][:-1], out[i][1:])])
        assert ps.max() <= args.max_pos_step_m + 1e-9 and rs.max() <= np.radians(args.max_rot_step_deg) + 1e-9
    return out, source_times


# ------------------------------------------------------------------- IK ----
def hand_targets(tool_T, side):
    T_oh = pose_T(T_OH[side][:3], T_OH[side][3:7])
    hand_T = [Twt @ T_oh for Twt in tool_T]
    P = np.asarray([T[:3, 3] for T in hand_T]); Q = np.asarray([R_to_q(T[:3, :3]) for T in hand_T])
    return P, Q, hand_T[0]


def sparse_reach(ik, P, Q, rows, seed_bank):
    ok = 0; worst = 0.0
    for row in rows:
        trials = [ik.solve(P[row], q_to_R(Q[row]), q0=q0, iters=200, pos_tol=0.005, rot_tol=0.05)
                  for q0 in seed_bank]
        if any(t["ok"] for t in trials): ok += 1
        else: worst = max(worst, min(t["pos_err"] for t in trials))
    return ok / len(rows), worst


def solve_continuous(ik, P, Q, seed):
    from scipy.interpolate import PchipInterpolator
    rng = np.random.default_rng(seed)
    key_rows = np.unique(np.r_[np.arange(0, len(P), 10), len(P)-1]).astype(int)
    pools = []
    seed_bank = [ik.q_default] + [rng.uniform(ik.lower, ik.upper) for _ in range(24)]
    for row in key_rows:
        trials = [ik.solve(P[row], q_to_R(Q[row]), q0=q0, iters=300, pos_tol=0.005, rot_tol=0.05)
                  for q0 in seed_bank]
        pool = []
        for ans in trials:
            if ans["ok"] and all(np.linalg.norm(ans["q"] - q) > 0.08 for q in pool):
                pool.append(np.asarray(ans["q"], np.float64))
        if not pool:
            best = min(trials, key=lambda r: r["pos_err"]**2 + (0.35*r["rot_err"])**2)
            raise RuntimeError(f"IK key row {row} has no solution: "
                               f"{100*best['pos_err']:.2f}cm/{np.degrees(best['rot_err']):.2f}deg")
        pools.append(pool)
    costs = [np.array([np.linalg.norm(q - ik.q_default)**2 for q in pools[0]])]
    parents = []
    for prev, cur, prev_cost in zip(pools[:-1], pools[1:], costs):
        edge = np.array([[np.linalg.norm(q1-q0)**2 for q1 in cur] for q0 in prev])
        total = prev_cost[:, None] + edge
        parents.append(np.argmin(total, axis=0)); costs.append(np.min(total, axis=0))
    j = int(np.argmin(costs[-1])); path = [pools[-1][j]]
    for k in range(len(parents)-1, -1, -1):
        j = int(parents[k][j]); path.append(pools[k][j])
    path = np.asarray(path[::-1])
    q_seed = PchipInterpolator(key_rows, path, axis=0)(np.arange(len(P)))
    reports = []
    for row, (p_tgt, quat_tgt, q0) in enumerate(zip(P, Q, q_seed)):
        R_tgt = q_to_R(quat_tgt)
        local_seeds = [q0] + ([np.asarray(reports[-1]["q"], np.float64)] if reports else [])
        trials = [ik.solve(p_tgt, R_tgt, q0=s, iters=300, pos_tol=0.002, rot_tol=0.02) for s in local_seeds]
        good = [r for r in trials if r["ok"]]
        target = local_seeds[-1]
        if not good:
            trials = [ik.solve(p_tgt, R_tgt, q0=s, iters=350, pos_tol=0.002, rot_tol=0.02) for s in seed_bank]
            good = [r for r in trials if r["ok"]]
        ans = (min(good, key=lambda r: np.linalg.norm(r["q"] - target)) if good
               else min(trials, key=lambda r: r["pos_err"]**2 + (0.35*r["rot_err"])**2))
        reports.append(ans)
    return reports


iks = {s: ArmIK(s, anchor_link="arm_center", anchor_T=anchor_T) for s in ("right", "left")}
oid = {"right": 1, "left": 0}

if args.scan_json:
    combos = json.loads(args.scan_json)
    rng = np.random.default_rng(0)
    banks = {s: [iks[s].q_default] + [rng.uniform(iks[s].lower, iks[s].upper) for _ in range(7)] for s in iks}
    rows = np.unique(np.r_[np.arange(0, NF, 20), NF - 1]).astype(int)
    print(f"[scan_json] {len(combos)} combos × rows={len(rows)} × seeds=8", flush=True)
    for c in combos:
        for k, v in c.items():
            setattr(args, k, v)
        src, dg = build_source_tracks(args.plate_yaw_deg, args.sponge_yaw_deg)
        rep = {}
        for side in ("left", "right"):
            P, Q, _ = hand_targets(src[oid[side]], side)
            fails = []
            for row in rows:
                trials = [iks[side].solve(P[row], q_to_R(Q[row]), q0=q0, iters=200, pos_tol=0.005, rot_tol=0.05)
                          for q0 in banks[side]]
                if not any(t["ok"] for t in trials):
                    fails.append((int(row), round(100 * min(t["pos_err"] for t in trials), 1)))
            rep[side] = (1 - len(fails) / len(rows), fails)
        print(f"[scan_json] {c} -> left ok={rep['left'][0]:.2f} right ok={rep['right'][0]:.2f} "
              f"right_fails(frame,cm)={rep['right'][1][:8]} left_fails={rep['left'][1][:4]}", flush=True)
    raise SystemExit(0)

if args.scan_yaw:
    rng = np.random.default_rng(0)
    banks = {s: [iks[s].q_default] + [rng.uniform(iks[s].lower, iks[s].upper) for _ in range(11)]
             for s in iks}
    rows = np.unique(np.r_[np.arange(0, NF, 15), NF - 1]).astype(int)
    grid = np.arange(-180.0, 180.0, args.scan_step_deg)
    print(f"[scan] rows={len(rows)} plate_yaw×sponge_yaw grid={len(grid)}×{len(grid)} "
          f"(plate_center=({args.plate_center_x},{args.plate_center_y}) h={args.plate_height})")
    results = []
    # 盘 yaw 只影响左臂; 海绵 yaw 偏置只影响右臂 (给定盘 yaw). 先扫左臂, 再对每个盘 yaw 扫右臂.
    left_ok = {}
    for py in grid:
        src, _ = build_source_tracks(py, 0.0)
        P, Q, _ = hand_targets(src[0], "left")
        left_ok[py] = sparse_reach(iks["left"], P, Q, rows, banks["left"])
        print(f"[scan] plate_yaw={py:6.1f}  left ok={left_ok[py][0]:.2f} worst={100*left_ok[py][1]:.1f}cm")
    good_py = [py for py in grid if left_ok[py][0] >= 0.99] or \
        sorted(grid, key=lambda py: -left_ok[py][0])[:3]
    for py in good_py:
        for sy in grid:
            src, diag = build_source_tracks(py, sy)
            P, Q, _ = hand_targets(src[1], "right")
            ok, worst = sparse_reach(iks["right"], P, Q, rows, banks["right"])
            results.append((py, sy, left_ok[py][0], ok, worst))
            print(f"[scan] plate_yaw={py:6.1f} sponge_yaw={sy:6.1f}  left ok={left_ok[py][0]:.2f} "
                  f"right ok={ok:.2f} worst={100*worst:.1f}cm")
    results.sort(key=lambda r: (-(r[2] + r[3]), r[4]))
    print("[scan] best (plate_yaw, sponge_yaw, left_ok, right_ok, worst_cm):")
    for r in results[:8]:
        print(f"   {r[0]:6.1f} {r[1]:6.1f}  {r[2]:.2f} {r[3]:.2f} {100*r[4]:.1f}")
    raise SystemExit(0)

if args.geometry_only:
    import sys as _sys, torch as _torch
    _sys.path.insert(0, os.path.join(ROOT, "tasks/Clean/3/A_Design/L3_Learning"))
    from progress_batch import CleanGeometry, CleanProgressBatch, CleanSignals
    src, dg = build_source_tracks(args.plate_yaw_deg, args.sponge_yaw_deg)
    q_ci = priors["left"]["canon_rot"]
    # 判据几何按**本次实测**的盘剖面/海绵面偏移走 —— 默认值是 take3 盘的, 换盘后必须跟着换
    # (否则 --geometry_only 的覆盖/接触是拿 take3 碟形剖面量另一块盘, 2026-09-10 take18 实测差 13mm)
    _geo = CleanGeometry(prof_r=tuple(float(v) for v in prof_r), prof_y=tuple(float(v) for v in prof_top),
                         sponge_face_offset=float(SPONGE_FACE_OFFSET))
    S = CleanSignals(1, "cpu", q_ci, _geo); P = CleanProgressBatch(1, "cpu", S, _geo)
    pp = _torch.tensor(src[0][:, :3, 3], dtype=_torch.float32); pq = _torch.tensor([R_to_q(T[:3, :3]) for T in src[0]], dtype=_torch.float32)
    sp = _torch.tensor(src[1][:, :3, 3], dtype=_torch.float32); sq = _torch.tensor([R_to_q(T[:3, :3]) for T in src[1]], dtype=_torch.float32)
    P.reset(_torch.tensor([0]), pp[:1]); contact = []; ntouch = []
    for r_ in range(NF):
        sig = S(pp[r_:r_+1], pq[r_:r_+1], sp[r_:r_+1], sq[r_:r_+1]); P.step(sig, pp[r_:r_+1], _torch.tensor([False]))
        contact.append(float(sig["contact"][0])); ntouch.append(int(sig["touch"][0].sum()))
    cov = P.cover[0].numpy() & S.disk.numpy()
    gx = np.arange(-0.09, 0.09, 0.01) + 0.005; cx, cy = np.meshgrid(gx, gx, indexing="ij"); rr = np.sqrt(cx**2 + cy**2)
    inner = (rr < 0.045) & S.disk.numpy(); outer = (rr >= 0.045) & S.disk.numpy()
    rng = np.random.default_rng(0)
    banks = {s_: [iks[s_].q_default] + [rng.uniform(iks[s_].lower, iks[s_].upper) for _ in range(7)] for s_ in iks}
    rows = np.unique(np.r_[np.arange(0, NF, 20), NF - 1]).astype(int)
    reach = {}; _wt = {}
    for side in ("right", "left"):
        Pq, Qq, _ = hand_targets(src[oid[side]], side)
        _wt[side] = Pq
        reach[side] = sparse_reach(iks[side], Pq, Qq, rows, banks[side])[0]
    # ★双腕间距 (2026-09-10): 只看可达与覆盖会选出"两手叠在胸口"的解 (take18 首版 min 2.7cm)。
    #   腕目标 = T_world_obj × inv(T_obj_hand), 与 sponge yaw 强耦合 —— 必须与可达同时判。
    _ws = np.linalg.norm(_wt["left"] - _wt["right"], axis=1) * 100
    print(f"[geometry] pitch={args.sponge_pitch_deg:.0f}° press={args.press_depth_m*1000:.0f}mm: contact_rows={np.mean(contact):.2f} "
          f"touch_pts p50={np.median(ntouch):.0f}/91 coverage={float(P.coverage()[0]):.3f} "
          f"(碟心 {cov[inner].sum()}/{inner.sum()}, 缘带 {cov[outer].sum()}/{outer.sum()}) travel={float(P.travel[0])*100:.0f}cm "
          f"h p50={np.median(dg['sponge_h'])*100:.2f}cm | sparse reach right={reach['right']:.2f} left={reach['left']:.2f} "
          f"| 腕距(cm) min={_ws.min():.1f} p10={np.percentile(_ws,10):.1f} p50={np.median(_ws):.1f}", flush=True)
    raise SystemExit(0)

src_tracks, diag = build_source_tracks(args.plate_yaw_deg, args.sponge_yaw_deg)
print(f"[reference] plate pos range(cm)={np.round((diag['plate_pos'].max(0)-diag['plate_pos'].min(0))*100,1).tolist()} "
      f"path={100*np.linalg.norm(np.diff(diag['plate_pos'],axis=0),axis=1).sum():.0f}cm | "
      f"sponge r(cm) pct10/50/90={np.round(np.percentile(diag['sponge_r']*100,[10,50,90]),1).tolist()} "
      f"on_plate={diag['on_plate'].mean():.2f} | h pct={np.round(np.percentile(diag['sponge_h']*100,[10,50,90]),2).tolist()}cm "
      f"| psi range={np.round(np.degrees([diag['psi'].min(), diag['psi'].max()]),0).tolist()}deg")
tool_T, source_times = resample_expand(src_tracks)
times_s = np.arange(len(source_times)) / args.control_hz
print(f"[reference] 15Hz {NF} 帧 -> {len(source_times)} 行 @ {args.control_hz:.0f}Hz ({times_s[-1]:.1f}s)")

def slerp_q(q0, q1, a):
    q0, q1 = np.asarray(q0, np.float64), np.asarray(q1, np.float64)
    if np.dot(q0, q1) < 0: q1 = -q1
    d = float(np.clip(np.dot(q0, q1), -1, 1))
    if d > 0.9995:
        q = q0 + a * (q1 - q0); return q / np.linalg.norm(q)
    th = np.arccos(d); return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def prelude_hand_targets(side, T_obj0):
    """接近行的手目标: prior pregrasp 6 行 (物体系) 插值到 K_app 行, 末行 = 当前 T_OH (grasp)。返回 (P,Q) 世界系。"""
    pg = np.asarray(priors[side]["pregrasp"], np.float64)[:, :7]
    knots = np.concatenate([pg, T_OH[side][None, :7]], 0)            # 7 个结点: pre0..pre5, grasp
    P, Q = [], []
    for k in range(args.k_app):
        u = k / max(args.k_app - 1, 1) * (len(knots) - 1); i = min(int(np.floor(u)), len(knots) - 2); a = u - i
        pos = (1 - a) * knots[i, :3] + a * knots[i + 1, :3]; quat = slerp_q(knots[i, 3:7], knots[i + 1, 3:7], a)
        T = T_obj0 @ pose_T(pos, quat); P.append(T[:3, 3]); Q.append(R_to_q(T[:3, :3]))
    return np.asarray(P), np.asarray(Q)


def prelude_finger_rows(side):
    pg, g, sq = (np.asarray(priors[side][k], np.float64) for k in ("pregrasp", "grasp", "squeeze"))
    f_app = np.stack([(1 - a) * pg[0, 7:29] + a * g[7:29] for a in np.linspace(0, 1, args.k_app)])
    f_close = np.stack([(1 - a) * g[7:29] + a * sq[7:29] for a in np.linspace(0, 1, args.k_close + 1)[1:]])
    f_settle = np.repeat(sq[None, 7:29], args.k_settle, 0)
    return np.concatenate([f_app, f_close, f_settle], 0)


PRE_ROWS = (args.k_app + args.k_close + args.k_settle) if args.prelude else 0
if args.prelude:
    for i in (0, 1):                                                   # 前奏期物体钉在 0 行位姿
        tool_T[i] = np.concatenate([np.repeat(tool_T[i][:1], PRE_ROWS, 0), tool_T[i]], 0)
    source_times = np.concatenate([np.full(PRE_ROWS, source_times[0]), source_times])
    times_s = np.arange(len(source_times)) / args.control_hz
    print(f"[reference] 前奏 {PRE_ROWS} 行 (接近 {args.k_app} / 合拢 {args.k_close} / 静持 {args.k_settle}); "
          f"release_row_min={args.k_app + args.k_close}; 总行数 {len(times_s)}")

arm_q, ik_report, hand0 = {}, {}, {}
hand_P = {}          # 逐侧腕位置目标 (双腕间距闸用)
for side in ((args.audit_side,) if args.audit_side else ("right", "left")):
    P, Q, hand0[side] = hand_targets(tool_T[oid[side]][PRE_ROWS:], side)
    hand_P[side] = P
    if args.prelude:
        Pp, Qp = prelude_hand_targets(side, tool_T[oid[side]][0])
        # 合拢+静持行手目标 = grasp 位姿 (与擦拭 0 行相同)
        P = np.concatenate([Pp, np.repeat(P[:1], args.k_close + args.k_settle, 0), P], 0)
        Q = np.concatenate([Qp, np.repeat(Q[:1], args.k_close + args.k_settle, 0), Q], 0)
    reports = solve_continuous(iks[side], P, Q, 20260907 + int(side == "left"))
    qs = [np.asarray(a["q"], np.float64) for a in reports]
    pe = [float(a["pos_err"]) for a in reports]; re = [float(a["rot_err"]) for a in reports]
    ok = [bool(a["ok"]) for a in reports]
    arm_q[side] = np.asarray(qs, np.float32)
    ik_report[side] = dict(ok_ratio=float(np.mean(ok)), pos_max_cm=100*max(pe),
                           rot_max_deg=float(np.degrees(max(re))),
                           joint_step_max_deg=float(np.degrees(np.abs(np.diff(qs, axis=0)).max())))
    bad = np.flatnonzero(~np.asarray(ok))
    print(f"[reference] {side}: {ik_report[side]} bad_rows={bad.tolist()[:20]}")
if args.audit_side:
    raise SystemExit(0)

# 母带附带的判据原料: 逐行海绵在盘规范系的面内位置/半径/贴合标记
src_r = np.interp(source_times, np.arange(NF), diag["sponge_r"])
src_on = np.interp(source_times, np.arange(NF), diag["on_plate"].astype(float)) > 0.5
if args.prelude:
    src_on[:PRE_ROWS] = False                          # 前奏不是擦拭
out = {
    "right_q": arm_q["right"], "left_q": arm_q["left"],
    "right_f": (np.concatenate([prelude_finger_rows("right"), np.repeat(priors["right"]["squeeze"][None, 7:29], len(times_s) - PRE_ROWS, 0)], 0)
                if args.prelude else np.repeat(priors["right"]["squeeze"][None, 7:29], len(times_s), 0)).astype(np.float32),
    "left_f": (np.concatenate([prelude_finger_rows("left"), np.repeat(priors["left"]["squeeze"][None, 7:29], len(times_s) - PRE_ROWS, 0)], 0)
               if args.prelude else np.repeat(priors["left"]["squeeze"][None, 7:29], len(times_s), 0)).astype(np.float32),
    "prelude_rows": np.int32(PRE_ROWS), "release_row_min": np.int32(args.k_app + args.k_close if args.prelude else 0),
    "k_app": np.int32(args.k_app if args.prelude else 0), "k_close": np.int32(args.k_close if args.prelude else 0),
    "right_f_grasp": priors["right"]["grasp"][7:29].astype(np.float32),
    "left_f_grasp": priors["left"]["grasp"][7:29].astype(np.float32),
    "fin_names": np.array(GENERIC_JOINT_ORDER, dtype=object),
    "source_frame": source_times.astype(np.float32),
    "control_hz": np.float32(args.control_hz), "source_fps": np.float32(source_fps),
    "table_z": np.float32(args.table_z),
    "plate_yaw_deg": np.float32(args.plate_yaw_deg), "plate_tilt_deg": np.float32(args.plate_tilt_deg),
    "sponge_yaw_deg": np.float32(args.sponge_yaw_deg), "sponge_xy_mode": np.array(args.sponge_xy_mode),
    "sponge_xy_recenter": np.array(bool(args.sponge_xy_recenter)),
    "sponge_xy_scale": np.float32(args.sponge_xy_scale),
    "source_take": np.array(str(args.take)), "assets_take": np.array(str(args.assets_take)),
    "sponge_assets_take": np.array(str(args.sponge_assets_take or args.assets_take)),
    "src_frames": np.array(str(args.src_frames or "")), "src_a": np.array(SRC_A), "src_b": np.array(SRC_B),
    "plate_flip": np.array(int(args.plate_flip)),
    "psi_mode": np.array(args.psi_mode), "psi_clip_deg": np.float32(args.psi_clip_deg),
    "plate_motion_scale": np.float32(args.plate_motion_scale),
    "plate_center": np.array([args.plate_center_x, args.plate_center_y, args.plate_height], np.float32),
    "press_depth_m": np.float32(args.press_depth_m), "sponge_pitch_deg": np.float32(args.sponge_pitch_deg),
    "plate_top_profile_r": prof_r.astype(np.float32), "plate_top_profile_y": prof_top.astype(np.float32),
    "sponge_face_offset": np.float32(SPONGE_FACE_OFFSET),
    "sponge_r": src_r.astype(np.float32), "sponge_on_plate": src_on,
    "anchor_T": anchor_T.astype(np.float32),
    "T_oh_left": T_OH["left"].astype(np.float32), "T_oh_right": T_OH["right"].astype(np.float32),
    "oh_override": np.array(args.oh_override or ""),
    "meta": ("Clean3 v1: plate=lowpassed rts position + constant orientation; sponge=recon in-plane "
             f"pattern ({args.sponge_xy_mode}) re-seated on plate top profile, scrub face down; "
             "wrists = object x GraspPose (input frame); continuous arm IK; fingers = squeeze constant; "
             "15Hz->20Hz knot-preserving time expansion"),
}
for i in (0, 1):
    out[f"obj_pos_{i}"] = tool_T[i][:, :3, 3].astype(np.float32)
    out[f"obj_quat_{i}"] = np.asarray([R_to_q(T[:3, :3]) for T in tool_T[i]], np.float32)
    # ⚠ zr 是**未截断**的源数组, 必须与 raw 同样切 [SRC_A:SRC_B], 否则与 np.arange(NF) 长度不匹配
    cf = np.minimum(np.asarray(zr[i]["conf_pos"]), np.asarray(zr[i]["conf_rot"]))[SRC_A:SRC_B]
    out[f"confidence_{i}"] = np.interp(source_times, np.arange(NF), cf).astype(np.float32)
# 盘姿态被常量替换: 盘的置信度只按位置算 (旋转不再来自重建)
out["confidence_0"] = np.interp(source_times, np.arange(NF),
                                np.asarray(zr[0]["conf_pos"])[SRC_A:SRC_B]).astype(np.float32)
out["ik_report"] = np.array(str(ik_report))

# ---- 人手指姿层 (2026-09-10, §5.15): DexPilot 重定向产物 ref_qpos_*.npz 的 finger_qpos ----
#   只取**增量**并焊在 GraspPose 的 squeeze 姿上 (pour17 母带同款做法: "手=首帧焊 GraspPose 增量"),
#   于是 k=0 处与现役指列逐位相同, 抓握不被破坏; 人手贡献的是逐行的**指关节articulation**。
#   ⚠ 只写不用: env 侧 CLEAN_S2_HAND_REF=0 时这两列完全不参与计算。
#   腕位/腕朝向**不取** —— EgoDex 腕根实测几乎不动 (take3 左腕 20s 走 0.8cm), 擦拭靠腕旋转+指屈伸,
#   而我们的腕参考是物体轨迹反推的, 两者策略不同, 混用会自相矛盾 (用户 2026-09-10 裁定: 只进指列)。
if not args.no_hand_ref:
    _hf = {}
    for _side, _key in (("right", "right_f"), ("left", "left_f")):
        _p = os.path.join(DATA, f"ref_qpos_{_side}.npz")
        if not os.path.exists(_p):
            print(f"[reference] ⚠ 缺 {_p}, 跳过人手指姿层")
            _hf = None; break
        _z = np.load(_p, allow_pickle=True)
        _jn = [str(x) for x in _z["joint_names"]]
        _want = [n.replace("right_", f"{_side}_", 1) for n in GENERIC_JOINT_ORDER]
        assert _jn == _want, f"{_side} ref_qpos 指关节顺序与母带 fin_names 不一致:\n  {_jn[:3]}\n  {_want[:3]}"
        _fq = np.asarray(_z["finger_qpos"], np.float64)              # (NF,22) @15Hz
        _vd = np.asarray(_z["valid"]).astype(bool)
        assert _vd.all(), f"{_side} ref_qpos 有无效帧 {int((~_vd).sum())}/{len(_vd)}"
        # 按 source_times 插到母带行 (与其它列同一条时间轴)
        # source_times 是**截断后**的行号, 而 _fq 是全长源帧 ⇒ 查表位置要加回 SRC_A
        _fi = np.stack([np.interp(source_times + SRC_A, np.arange(len(_fq)), _fq[:, j]) for j in range(22)], 1)
        _delta = _fi - _fi[0]                                        # 首帧归零 ⇒ 焊在 squeeze 上
        _hf[_side] = (out[_key] + _delta).astype(np.float32)
    if _hf is not None:
        out["human_right_f"] = _hf["right"]; out["human_left_f"] = _hf["left"]
        out["hand_ref_src"] = np.array("ref_qpos_{left,right}.npz finger_qpos (dexpilot), 首帧归零后焊在 squeeze 姿")
        _dd = np.degrees(np.abs(np.concatenate([_hf["right"] - out["right_f"],
                                                _hf["left"] - out["left_f"]], 1)))
        print(f"[reference] 人手指姿层已写: 逐关节增量 |Δ| 中位={np.median(_dd):.2f}° 最大={_dd.max():.2f}° "
              f"(k=0 处 Δ={_dd[0].max():.3f}°, 应为 0)")

os.makedirs(os.path.dirname(os.path.join(ROOT, args.output)), exist_ok=True)
# ★双腕间距闸 (2026-09-10): 腕目标 = T_world_obj × inv(T_obj_hand), 与 sponge yaw 强耦合;
#   转 yaw 能把右腕转到左腕头上而 IK 全通 —— **可达 ≠ 不碰**。take18 首版 min 3.2cm 就是这么出来的。
_n_gap = min(len(hand_P["left"]), len(hand_P["right"])) if len(hand_P) == 2 else 0
_gap_cm = (np.linalg.norm(hand_P["left"][:_n_gap] - hand_P["right"][:_n_gap], axis=1) * 100
           if _n_gap else np.array([np.inf]))
if _n_gap:
    print(f"[reference] 双腕间距(cm): min={_gap_cm.min():.1f} p10={np.percentile(_gap_cm,10):.1f} p50={np.median(_gap_cm):.1f}")
if args.min_wrist_gap_cm > 0 and _gap_cm.min() < args.min_wrist_gap_cm:
    raise SystemExit(
        f"[reference] ✗ 拒绝写盘: 双腕间距最小 {_gap_cm.min():.1f}cm < 下限 {args.min_wrist_gap_cm:.1f}cm "
        f"(take 3 定版是 10.7cm)。双臂会在机器人胸前重叠, IK 全通也没用。\n"
        f"    改 --sponge_yaw_deg 重扫 (--geometry_only 会同时打可达/覆盖/腕距), 或 --min_wrist_gap_cm 0 强行放行。")

np.savez(os.path.join(ROOT, args.output), **out)
print(f"[reference] -> {args.output}  rows={len(times_s)}  hand0 left={np.round(hand0['left'][:3,3],3).tolist()} "
      f"right={np.round(hand0['right'][:3,3],3).tolist()}")

if not args.no_layout:
    lay = {
        "schema_version": "clean_held_scene_v1",
        "rl_table_height": args.table_z,
        # ⚠ objects[*].pos 已是 RL 世界系; screw_assembly._free_aux_from_layout 用
        #   z - scene_table_z + table_top_z 换桌高, 故这里 scene_table_z 必须 = rl 桌高 (恒等)。
        "scene_table_z": args.table_z,
        "scene_table_z_source_estimate": float(np.median(raw[0][:, 2, 3]) - args.plate_height),
        "scene_table_source": "无物体落桌; 源世界桌高估计 = 盘 rts 位置中位 z - plate_height; objects.pos 直接给 RL 系",
        "source_frame": 0,
        "registration": dict(scene_yaw_deg=args.scene_yaw_deg, plate_center_xy=[args.plate_center_x, args.plate_center_y],
                             plate_height=args.plate_height, plate_yaw_deg=args.plate_yaw_deg,
                             plate_tilt_deg=args.plate_tilt_deg, sponge_yaw_deg=args.sponge_yaw_deg,
                             sponge_xy_mode=args.sponge_xy_mode, lp_window_s=args.lp_window_s),
        "objects": {
            "object_0": dict(identity="plate", anchor_hand="left",
                             pos_source=raw[0][0, :3, 3].tolist(), quat_source=R_to_q(raw[0][0, :3, :3]).tolist(),
                             pos=tool_T[0][0, :3, 3].tolist(), quat_wxyz=R_to_q(tool_T[0][0, :3, :3]).tolist(),
                             quat_reason="常量: 水平(规范 z-up)+tilt, 重建姿态不用 (圆盘绕法向自转不可观测)",
                             mesh=f"datasets/clean_tableware/{args.assets_take}/objects/object_0/object_mesh_scaled_final.obj",
                             extent_cm=_extent_cm(load_obj_vertices(os.path.join(
                                 ASSETS, "objects", "object_0", "object_mesh_scaled_final.obj")))),
            "object_1": dict(identity="sponge", anchor_hand="right",
                             pos_source=raw[1][0, :3, 3].tolist(), quat_source=R_to_q(raw[1][0, :3, :3]).tolist(),
                             pos=tool_T[1][0, :3, 3].tolist(), quat_wxyz=R_to_q(tool_T[1][0, :3, :3]).tolist(),
                             quat_reason="擦盘面朝盘 + 面内 yaw ψ(t) 来自重建长轴投影",
                             mesh=f"datasets/clean_tableware/{args.sponge_assets_take or args.assets_take}/objects/object_1/object_mesh_scaled_final.obj",
                             extent_cm=_extent_cm(sp_v)),
        },
        "reference_npz": args.output,
    }
    with open(os.path.join(DATA, "scene_layout.json"), "w") as f:
        json.dump(lay, f, indent=1, ensure_ascii=False)
    print(f"[reference] scene_layout.json 写入 {DATA}")
