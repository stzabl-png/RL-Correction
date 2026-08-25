"""Pour 数据体检①: 高置信 Pose + RTS 平滑物体轨迹, 在仿真场景里过目。

两遍循环:
  第一遍 [路标] —— 高置信帧逐个**定格** (dwell 帧), 终端打印帧号/conf/σ;
  第二遍 [轨迹] —— 整条平滑轨迹连续回放。
坐标: 以**仿真里瓶子的摆位**为锚 (A = T_sim_瓶0 · T_rec_瓶0⁻¹), 两物体同用 A
  ⟹ 看到的是重建**真实的相对几何** (杯子落在重建说的地方, 不是训练摆位)。

用法 (本地 GUI):
    SHARPA_WANDB=0 RL_HAND_JOINTS=1 PYTHONPATH=. $PY -m tasks.pour.viz_hc_traj \
        --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz --prior_yaw 19.5 \
        --prior_b tasks/pregrasp/priors/Pour17_cup.npz --prior_b_yaw 90
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--prior_b", required=True)
p.add_argument("--prior_b_yaw", type=float, default=-1.0)
p.add_argument("--rts_dir", default="/home/lyh/Project/Reconstruct_and_Retarget/"
                                    "results/pour_17_better/poseqa")
p.add_argument("--conf", type=float, default=60.0, help="高置信阈 (conf_pos 与 conf_rot)")
p.add_argument("--dwell", type=int, default=20, help="每个路标定格帧数 (连续帧路标自动快进)")
p.add_argument("--speed", type=int, default=3, help="轨迹回放: 每数据帧渲染几帧")
p.add_argument("--objs", default="0",
               help="播放哪些物体的轨迹: 0=杯(默认, 一步一验), 1=瓶, 0,1=双双")
p.add_argument("--hc", action="store_true",
               help="高置信轨迹模式: 只在高置信帧更新位姿, 低置信段定格在上一可信帧 "
                    "(零阶保持) —— 即 P1 监督真正可用的运动")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("viz_hc_traj")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp import bimanual as BM  # noqa: E402
from tasks.pregrasp.bimanual_native_env import BimanualNativeEnv  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402

# ---------------- 场景 (与 pose_pregrasp 同款组装) ----------------
cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.direct_grasp_prob = 0.0
cfg.approach_t0_max = 0.0
cfg.retract_start = True          # 只为过双臂防雷断言 (纯可视化)
cfg.prior_b_npz = args.prior_b
cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
_a1, _o1 = cfg.action_space, cfg.observation_space
cfg.action_space, cfg.observation_space = 2 * _a1, 2 * _o1
cfg._obs_single = _o1
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
env = BimanualNativeEnv(cfg)
env.reset()

# ---------------- 数据 ----------------
def _mat2quat(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-6:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
        q = np.zeros(4); q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4.0
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
        return q
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                     (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])

Z = {}
HI = {}
for i in (0, 1):
    z = np.load(f"{args.rts_dir}/rts_pour_17_object_{i}.npz", allow_pickle=True)
    T = np.asarray(z["object_ob_in_world_smooth"], np.float64)
    hi = ((np.asarray(z["conf_pos"]) >= args.conf)
          & (np.asarray(z["conf_rot"]) >= args.conf)
          & np.asarray(z["measurement_used"]).astype(bool)
          & (np.asarray(z["innov_pos_nis"]) < 5.0)
          & (np.asarray(z["innov_rot_nis"]) < 5.0))   # ★只留上界 (下界反杀好帧)
    Z[i] = dict(T=T, sp=np.asarray(z["sigma_pos_m"]), sr=np.asarray(z["sigma_rot_deg"]),
                cp=np.asarray(z["conf_pos"]), cr=np.asarray(z["conf_rot"]))
    HI[i] = hi
N = len(Z[0]["T"])
union = np.flatnonzero(HI[0] | HI[1])
# ★身份对应 (2026-08-19 用户确认+手物距离实测): obj0=杯(左手接,9.4cm贴身),
#   obj1=瓶(右手倒,10.7cm贴身)。此前本工具绑反 (obj0→右手) 并把名字标反,
#   训练配置 clips.py 一直是对的 (object_0=left / object_1=right)。
NM = {0: "杯", 1: "瓶"}
print(f"[viz] 高置信路标: 杯(obj0) {int(HI[0].sum())}/{N} | 瓶(obj1) {int(HI[1].sum())}/{N} | "
      f"并集 {len(union)} 帧 (过滤: conf≥{args.conf:.0f}+真测量+NIS<5)")

# ---------------- 对齐: 逐物体各用**自己的训练摆位**当锚 (2026-08-19 用户裁定:
# 场景要和 GraspPose 摆放一致——各回各位, 各自在原位演自己的轨迹) ----------------
def _quat2mat(q):
    w, x, y, z_ = q
    return np.array([[1-2*(y*y+z_*z_), 2*(x*y-w*z_), 2*(x*z_+w*y)],
                     [2*(x*y+w*z_), 1-2*(x*x+z_*z_), 2*(y*z_-w*x)],
                     [2*(x*z_-w*y), 2*(y*z_+w*x), 1-2*(x*x+y*y)]])

W = env.scene.env_origins[0].cpu().numpy().astype(np.float64)
# 绑定: obj0(杯)→B侧(左手), obj1(瓶)→A侧(右手) —— 与 clips.py/手物距离一致
_sides = {0: env._B, 1: env._A}
with BM.use_side(env, env._A):
    p_sim_bottle = env.object.data.root_pos_w[0].cpu().numpy().astype(np.float64)
with BM.use_side(env, env._B):
    p_sim_cup = env.object.data.root_pos_w[0].cpu().numpy().astype(np.float64)
# 全局水平转角: 重建"杯→瓶"方向 对齐 场景"杯位→瓶位"方向 (相对几何零损)
_v_rec = (Z[1]["T"][0][:2, 3] - Z[0]["T"][0][:2, 3])   # obj0(杯)→obj1(瓶)
_v_sim = (p_sim_bottle - p_sim_cup)[:2]                # 杯位(B)→瓶位(A)
_yaw = float(np.arctan2(_v_sim[1], _v_sim[0]) - np.arctan2(_v_rec[1], _v_rec[0]))
_c, _s = np.cos(_yaw), np.sin(_yaw)
_Rz = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])
# ★相对位移回放 (2026-08-19 用户裁定): 物体从**训练静置位姿**出发, 叠加重建的
#   "相对首帧位移算子" (经全局转角共轭, 方向对齐场景):
#       T_sim_i(t) = Gr · [T_rec_i(t) · T_rec_i(0)⁻¹] · Gr⁻¹ ∘ T_rest_i
#   t=0 ⟹ 算子恒等 ⟹ 就是桌上的物理起点, 按 Enter 零跳变; 倾角幅度精确保留,
#   方向经同一个全局转角映射 (瓶朝杯的方向在场景里指向杯)。
T_rest = {}
for i in (0, 1):
    with BM.use_side(env, _sides[i]):
        _p = env.object.data.root_pos_w[0].cpu().numpy().astype(np.float64)
        _q = env.object.data.root_quat_w[0].cpu().numpy().astype(np.float64)
    Tr = np.eye(4); Tr[:3, :3] = _quat2mat(_q); Tr[:3, 3] = _p
    T_rest[i] = Tr
_T0inv = {i: np.linalg.inv(Z[i]["T"][0]) for i in (0, 1)}
# ★世界系相对轨迹, 支点=物体自己 (2026-08-19 定稿):
#   R_sim(t) = Rz·[R_rec(t)·R_rec(0)⁻¹]·Rzᵀ · R_rest ; p_sim(t) = p_rest + Rz·Δp_rec
#   为什么不能用"自身系位移": 训练资产是旧 pour17 网格+自动摆放(明确不用重建旋转),
#   它的身体系与新重建 SAM3D 网格系毫不相干 —— 身体系位移会绕错误的轴转 (实测
#   杯朝向不对)。世界系相对量与网格系无关: 幅度=重建原值, 方向经全局转角 Rz 映射,
#   t=0 恒等零跳变, 支点在物体自己不会飞。
_hc_last = {}
def _hc_frame(i, k):
    """高置信零阶保持: 返回 ≤k 的最近高置信帧 (没有则 0)。"""
    hs = np.flatnonzero(HI[i][:k + 1])
    return int(hs[-1]) if len(hs) else 0

def _pose_sim(i, k):
    if args.hc:
        k = _hc_frame(i, k)
    dR = Z[i]["T"][k][:3, :3] @ _T0inv[i][:3, :3]        # 世界系相对转动
    dp = Z[i]["T"][k][:3, 3] - Z[i]["T"][0][:3, 3]       # 世界系相对平移
    out = np.eye(4)
    out[:3, :3] = _Rz @ dR @ _Rz.T @ T_rest[i][:3, :3]
    out[:3, 3] = T_rest[i][:3, 3] + _Rz @ dp
    return out
PLAY = [int(x) for x in args.objs.split(",") if x.strip() != ""]
print(f"[viz] 自身系相对轨迹回放 | 起点=训练静置 | 播放对象: "
      f"{[NM[i] for i in PLAY]} (其余钉在桌上)")

# 机器人是布景: 双臂每帧钉在站姿 (pose_pregrasp 同款; 不钉的话 reset 起步姿态
# 不受控 —— 实测左臂在天上)
hand = env.hand
_q_pin = hand.data.default_joint_pos.clone()
_v_pin = torch.zeros_like(_q_pin)

def show_frame(k):
    hand.write_joint_state_to_sim(_q_pin, _v_pin)
    hand.set_joint_position_target(_q_pin)
    for i in (0, 1):
        Tw = _pose_sim(i, k) if i in PLAY else T_rest[i]
        pose = np.concatenate([Tw[:3, 3], _mat2quat(Tw[:3, :3])])
        pt = torch.tensor(pose, dtype=torch.float32, device=env.device).unsqueeze(0)
        idx = torch.zeros(1, dtype=torch.long, device=env.device)
        with BM.use_side(env, _sides[i]):
            env.object.write_root_pose_to_sim(pt, idx)
            env.object.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), idx)

# 位移剖面 (相对首帧), 终端直接量物体动没动; 数据前段是静止的 (人还没抓)
_disp = {i: np.linalg.norm(Z[i]["T"][:, :3, 3] - Z[i]["T"][0, :3, 3], axis=1)
         for i in (0, 1)}
_onset = int(np.argmax(_disp[1] > 0.02)) if (_disp[1] > 0.02).any() else N
print(f"[viz] 起动帧 ≈ {_onset} (位移>2cm); 之前的数据段物体静止是**正常的** "
      f"(演示里人还没抓)")
# ---------------- 轨迹画线 (2026-08-23): 整条回放轨迹一次性画进 GUI ----------------
# 绿=高置信段 红=低置信段 (与路标同口径: conf≥阈+真测量+NIS<5, 段两端都高置信才绿)。
# 质心线(粗)看位置连续性; 瓶加画"口向探针"线(细, 质心+R·上轴×12cm) ——
# 旋转瞬移在质心线上不可见, 在探针线上是一段长红直弦 (帧51/52/85/90)。
try:
    from isaacsim.util.debug_draw import _debug_draw as _ddm  # noqa: E402
except ImportError:
    try:
        from omni.isaac.debug_draw import _debug_draw as _ddm  # noqa: E402
    except ImportError:
        _ddm = None
if _ddm is not None:
    _dd = _ddm.acquire_debug_draw_interface()

    def _polyline(pts, hi, width):
        s = [tuple(map(float, p)) for p in pts[:-1]]
        e = [tuple(map(float, p)) for p in pts[1:]]
        col = [((0.1, 0.9, 0.2, 1.0) if (hi[k] and hi[k + 1])
                else (1.0, 0.15, 0.1, 1.0)) for k in range(len(pts) - 1)]
        _dd.draw_lines(s, e, col, [float(width)] * len(s))

    for i in PLAY:
        _polyline(np.stack([_pose_sim(i, k)[:3, 3] for k in range(N)]),
                  HI[i], 6.0)
    if 1 in PLAY:       # 瓶: 口向探针 (把旋转放大成位移)
        _upb = T_rest[1][:3, :3].T @ np.array([0.0, 0.0, 1.0])
        _tip = []
        for k in range(N):
            Tw = _pose_sim(1, k)
            _tip.append(Tw[:3, 3] + Tw[:3, :3] @ (_upb * 0.12))
        _polyline(np.stack(_tip), HI[1], 2.0)
    print("[viz] 轨迹画线 ON: 绿=高置信 红=低置信 | 粗线=质心 | "
          "细线=瓶口向探针 (旋转瞬移=长红直弦)")
else:
    print("[viz] ⚠ debug_draw 扩展不可用, 跳过画线")
import select  # noqa: E402
import sys  # noqa: E402

def _enter_pressed():
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        sys.stdin.readline()
        return True
    return False

print("[viz] GUI: 待机=物理静置桌上; 按 Enter 开播 (第一遍=轨迹连播, 第二遍=路标定格; "
      "一轮播完回待机, Ctrl+C 退出)")
dt = env.sim.get_physics_dt()
while app.is_running():
    # ---- 待机: 首帧位姿写一次, 之后纯物理静置 (真实搁在桌上), 等 Enter ----
    show_frame(0)
    print("[viz] ⏸ 待机中 —— 按 Enter 开始播放", flush=True)
    while app.is_running() and not _enter_pressed():
        hand.write_joint_state_to_sim(_q_pin, _v_pin)
        hand.set_joint_position_target(_q_pin)
        env.sim.step(render=True)
        env.scene.update(dt)
    if not app.is_running():
        break
    # 第一遍: 连续轨迹 (先看会动的)
    print(f"[轨迹] {'高置信零阶保持' if args.hc else '平滑轨迹'}整条回放 "
          f"({N} 帧 × {args.speed})", flush=True)
    for k in range(N):
        if k % 20 == 0:
            _tl = []
            for i in (0, 1):
                u0 = np.linalg.inv(Z[i]["T"][0][:3, :3]) @ np.array([0., 0., 1.])
                wu = Z[i]["T"][k][:3, :3] @ u0
                _tl.append(np.degrees(np.arccos(np.clip(wu[2], -1, 1))))
            print(f"[轨迹] 帧 {k:3d} | 杯位移 {_disp[0][k]*100:5.1f}cm 倾 {_tl[0]:5.1f}° "
                  f"(conf {Z[0]['cr'][k]:3.0f}) | 瓶位移 {_disp[1][k]*100:5.1f}cm "
                  f"倾 {_tl[1]:5.1f}° (conf {Z[1]['cr'][k]:3.0f})", flush=True)
        for _ in range(args.speed):
            if not app.is_running():
                break
            show_frame(k)
            env.sim.step(render=True)
            env.scene.update(dt)
    # 第二遍: 路标定格 (连续帧路标只停 5 帧, 跳变处才足额定格)
    prev = -99
    for k in union:
        tags = [f"{NM[i]} conf {Z[i]['cp'][k]:.0f}/{Z[i]['cr'][k]:.0f}"
                f" σ {Z[i]['sp'][k]*1000:.0f}mm/{Z[i]['sr'][k]:.1f}°"
                for i in (0, 1) if HI[i][k]]
        _dw = args.dwell if k - prev > 1 else 5
        prev = int(k)
        print(f"[路标] 帧 {k:3d}/{N} | 瓶位移 {_disp[1][k]*100:5.1f}cm | "
              + " | ".join(tags), flush=True)
        for _ in range(_dw):
            if not app.is_running():
                break
            show_frame(int(k))
            env.sim.step(render=True)
            env.scene.update(dt)
try:
    _slot.release()
except Exception:
    pass
app.close()
