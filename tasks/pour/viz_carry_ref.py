"""Carry 参考纯前馈可视化: 双臂逐行播 right_q/left_q, 双物体同步播 obj 目标行.

零策略零物理跟随 —— 看的是"参考本身长什么样" (臂弧 + 物体契约是否同步).
carry3/carry4 同键位, --ref 切换对比.

用法 (本地 GUI):
    SHARPA_WANDB=0 RL_HAND_JOINTS=1 PYTHONPATH=. \
    /home/lyh/luhr/MagicSim/.venv/bin/python -m tasks.pour.viz_carry_ref \
        --grasp_prior tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz --prior_yaw 19.5 \
        --prior_b tasks/pregrasp/priors/Pour17_cup_thumbfix.npz --prior_b_yaw 90 \
        --ref tasks/pour/carry4_pour17.npz
按 Enter 开播; 播完回待机; Ctrl+C 退出.
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--prior_b", required=True)
p.add_argument("--prior_b_yaw", type=float, default=-1.0)
p.add_argument("--ref", default="tasks/pour/carry4_pour17.npz")
p.add_argument("--speed", type=int, default=2, help="每行渲染几帧 (越大越慢)")
p.add_argument("--fingers", default="grasp", choices=["grasp", "open"],
               help="手指摆位: grasp=抓姿指型(演示更真) / open=默认")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("viz_carry_ref")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp import bimanual as BM  # noqa: E402
from tasks.pregrasp.bimanual_native_env import BimanualNativeEnv  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.direct_grasp_prob = 0.0
cfg.approach_t0_max = 0.0
cfg.retract_start = True
cfg.prior_b_npz = args.prior_b
cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
_a1, _o1 = cfg.action_space, cfg.observation_space
cfg.action_space, cfg.observation_space = 2 * _a1, 2 * _o1
cfg._obs_single = _o1
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
env = BimanualNativeEnv(cfg)
env.reset()

z = np.load(args.ref, allow_pickle=True)
RQ = np.asarray(z["right_q"], np.float64)
LQ = np.asarray(z["left_q"], np.float64)
N = len(RQ)
G = int(z["grip_frames"]) if "grip_frames" in z.files else 0
print(f"[viz] {args.ref}: {N} 行 (热身 {G}) | meta: {z['meta'] if 'meta' in z.files else '-'}")
if "free_lo" in z.files:
    print(f"[viz] 自由段行号 [{int(z['free_lo'])},{int(z['free_hi'])}) (手锚弧段)")

OBJ = {}
for s, side in ((0, "A"), (1, "B")):
    OBJ[side] = (np.asarray(z[f"obj_pos_{s}"], np.float64),
                 np.asarray(z[f"obj_quat_{s}"], np.float64))

hand = env.hand
q0 = hand.data.default_joint_pos.clone()
_v0 = torch.zeros_like(q0)
if args.fingers == "grasp":
    # 指型 = squeeze 抓姿模板, 直接从 prior npz 读 (裸配置没有 _p2_fin,
    # 之前静默回退成张开手 —— 2026-08-24 用户发现"全程绷直"即此 bug)
    for ns, pr in ((env._A, args.grasp_prior), (env._B, args.prior_b)):
        # 模板 29 维 = [7臂 + 22指(GENERIC序)], 指部必须过 _generic_perm 换 USD 序
        # (台账 2026-07-30: 漏换序 → 手型全串位)
        _sq = np.asarray(np.load(pr, allow_pickle=True)["squeeze"],
                         np.float64).ravel()[7:29]
        with BM.use_side(env, ns):
            _perm = getattr(env, "_generic_perm", None)
            if _perm is not None:
                _sq = _sq[_perm]
            q0[:, env.hand_jids] = torch.tensor(
                _sq, dtype=torch.float32, device=env.device).unsqueeze(0)

# 瓶倾角剖面 (从 obj 行现算, 与 env._c_tiltref 同口径) 供终端打印
_bp, _bq = OBJ["A"]
def _q2m(q):
    w, x, y, z_ = q
    return np.array([[1-2*(y*y+z_*z_), 2*(x*y-w*z_), 2*(x*z_+w*y)],
                     [2*(x*y+w*z_), 1-2*(x*x+z_*z_), 2*(y*z_-w*x)],
                     [2*(x*z_-w*y), 2*(y*z_+w*x), 1-2*(x*x+y*y)]])
_R0 = _q2m(_bq[0]); _upb = _R0.T @ np.array([0.0, 0.0, 1.0])
TILT = np.degrees(np.arccos(np.clip(
    [(_q2m(q) @ _upb)[2] for q in _bq], -1.0, 1.0)))

def show_row(k):
    q = q0.clone()
    for ns, rows in ((env._A, RQ), (env._B, LQ)):
        with BM.use_side(env, ns):
            q[:, env.arm_jids] = torch.tensor(rows[k], dtype=torch.float32,
                                              device=env.device).unsqueeze(0)
    hand.write_joint_state_to_sim(q, _v0)
    hand.set_joint_position_target(q)
    for side in ("A", "B"):
        ns = env._A if side == "A" else env._B
        pos, quat = OBJ[side]
        pose = torch.tensor(np.concatenate([pos[k], quat[k]]), dtype=torch.float32,
                            device=env.device).unsqueeze(0)
        pose[:, :3] += env.scene.env_origins[:1]
        idx = torch.zeros(1, dtype=torch.long, device=env.device)
        with BM.use_side(env, ns):
            env.object.write_root_pose_to_sim(pose, idx)
            env.object.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), idx)

import select  # noqa: E402
import sys  # noqa: E402

def _enter():
    r, _, _ = select.select([sys.stdin], [], [], 0)
    if r:
        sys.stdin.readline(); return True
    return False

print("[viz] 待机 —— 按 Enter 播放 (双臂逐行 + 物体契约同步)")
dt = env.sim.get_physics_dt()
while app.is_running():
    show_row(0)
    while app.is_running() and not _enter():
        show_row(0)
        env.sim.step(render=True)
        env.scene.update(dt)
    if not app.is_running():
        break
    for k in range(N):
        if k % 20 == 0:
            tag = ""
            if "free_lo" in z.files and int(z["free_lo"]) <= k < int(z["free_hi"]):
                tag = "← 手锚弧段(倒水)"
            print(f"[viz] 行 {k:3d}/{N} | 瓶倾角 {TILT[k]:5.1f}° {tag}", flush=True)
        for _ in range(args.speed):
            if not app.is_running():
                break
            show_row(k)
            env.sim.step(render=True)
            env.scene.update(dt)
    print("[viz] 播完, 回待机 (Enter 重播)")
try:
    _slot.release()
except Exception:
    pass
app.close()
