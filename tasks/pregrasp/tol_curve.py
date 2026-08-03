"""P0.0 到位误差容差曲线 —— 抓取相位对"起点不在 prior 位姿上"的鲁棒性.

问题: 现有 100% 策略是从**精确的** GraspPose 位姿起步训出来的 (`arm_dev_max=0.05`,
起点就是答案). 接近段交过来的必然带位置/姿态误差. 这个误差多大时抓取还成立?

这条曲线定两个数 (见 docs/PLAN_PICK_LIFT.md §4):
  - 相位切换阈值 eps_pos / eps_rot —— 接近段必须收敛到多准才允许切
  - 对齐势函数 w_align 的尺度 —— 势场要把误差压到什么量级才算"到位"

做法: 在 q_pregrasp 附近采样腕位姿扰动 -> IK -> 起始关节位形池 -> 复位时随机抽,
其余一切与 `eval.py` 的正式口径 A (c0=0, gentle=1.0, 确定性 mu) 完全一致.
一个 Isaac session 内跑完所有档位 (省 12 次启动).

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.tol_curve --headless \
        --clip Grasp5 --checkpoint logs/GraspPose_5/<ts>/stage1_nn/last.pth \
        --grasp_prior tasks/pregrasp/priors/Grasp5.npz
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="Grasp5")
parser.add_argument("--grasp_prior", type=str, default="")
parser.add_argument("--prior_yaw", type=float, default=-1.0,
                    help="钉死的物体 yaw (度), 取 screen_prior Gate1 的 best_yaw")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=300, help="每档跑多少控制步")
parser.add_argument("--pool", type=int, default=192, help="每档采样多少个 IK 起始位形")
parser.add_argument("--out", type=str, default="data/tol_curve.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("tol_curve")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

env_cfg = GraspTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
if args.grasp_prior:
    apply_grasp_prior(env_cfg, args.grasp_prior, args.prior_yaw)
env_cfg.scene.num_envs = args.num_envs
env_cfg.closure_init_max = 0.0             # 正式口径 A: 手张开从头合拢
raw = GraspTaskEnv(env_cfg)
raw.gentle = 1.0
env = GymStyleEnvWrapper(raw, clip_actions=env_cfg.clip_actions)

agent = PPO(env, output_dir="/tmp/_tol_tmp",
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
print(f"[tol] loading {args.checkpoint}")
agent.restore_test(args.checkpoint)
agent.set_eval()

N, dev = raw.num_envs, raw.device
FAIL_KEYS = ("fell", "thrown", "pushed", "stuck", "table_crash", "timeout")


# ---------------------------------------------------------------- 起始位形池
def axang_R(ax, ang):
    ax = ax / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def geo_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


def build_pool(p0, R0, ik, seed_q, dp, dr, m, rng):
    """采样 m 个"离标称位姿 dp 米 / dr 弧度"的位姿, IK 成关节位形池 (K,7).

    ⚠ 标称位姿取 **ik.fk(q_pregrasp)** 而不是仿真读到的腕位姿 —— 扰动必须在求解器
    自己的坐标/模型里定义, 否则 δ=0 的基线(直接用 q_pregrasp)与 δ>0 的档位之间会混进
    一个模型↔仿真的系统偏移, 曲线的横轴就不干净了.
    """
    if dp == 0.0 and dr == 0.0:
        return None, 1.0, (0.0, 0.0)
    qs, ok, achieved = [], 0, []
    for _ in range(m):
        u = rng.normal(size=3)
        tgt_p = p0 + u / np.linalg.norm(u) * dp
        tgt_R = axang_R(rng.normal(size=3), dr) @ R0 if dr > 0 else R0
        r = ik.solve(tgt_p, tgt_R, q0=seed_q, iters=80)
        if r["ok"] and r["pos_err"] < 0.002:
            qs.append(r["q"])
            ok += 1
            p_a, R_a = ik.fk(r["q"])           # 实际到达的位姿 (校验横轴)
            achieved.append((np.linalg.norm(p_a - p0), geo_deg(R_a, R0)))
    if not qs:
        return None, 0.0, (0.0, 0.0)
    a = np.array(achieved)
    return (torch.tensor(np.stack(qs), dtype=torch.float32, device=dev),
            ok / m, (a[:, 0].mean(), a[:, 1].mean()))


# ---------------------------------------------------------------- 评测块
def run_block(tag, steps):
    obs_dict = env.reset()
    succ = episodes = verify_fails = 0
    fails = {k: 0 for k in FAIL_KEYS}
    with torch.no_grad():
        for _ in range(steps):
            _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"]}
            if "pointcloud" in obs_dict:
                _inp["pointcloud"] = obs_dict["pointcloud"]
            act = agent.model.act_inference(_inp)
            obs_dict, r, done, info = env.step(torch.clamp(act, -1.0, 1.0))
            s = raw._sig
            verify_fails += int(s["verify_fail"].sum())
            d = done.bool() if torch.is_tensor(done) else torch.tensor(done, device=dev).bool()
            if d.any():
                idx = torch.nonzero(d, as_tuple=False).squeeze(-1)
                episodes += len(idx)
                ns = s["newly_success"][idx]
                succ += int(ns.sum())
                for k in FAIL_KEYS:
                    fails[k] += int((s[k][idx] & ~ns).sum())
    sr = succ / max(episodes, 1)
    top = sorted(((v, k) for k, v in fails.items() if v > 0), reverse=True)[:3]
    print(f"  [{tag}]  成功率 {sr*100:6.2f}%  ({succ}/{episodes})  "
          f"失败 top: {'  '.join(f'{k}={v}' for v, k in top) or '-'}  "
          f"验证重试 {verify_fails/max(episodes,1):.1f}/回合", flush=True)
    return dict(sr=sr, succ=succ, episodes=episodes, fails=fails,
                verify_fail_per_ep=verify_fails / max(episodes, 1))


# ---------------------------------------------------------------- 主流程
ik = ArmIK(env_cfg.hand_side, anchor_link="arm_center", anchor_T=raw._anchor_T)
seed_q = raw.q_pregrasp.cpu().numpy().astype(np.float64)
p0, R0 = ik.fk(seed_q)                       # 标称位姿 = 求解器模型里的 q_pregrasp 末端
# 仿真读数只用于报告"离线模型 vs 仿真"的偏差, 不参与扰动定义
env.reset()
_z = torch.zeros((N, env_cfg.action_space), device=dev)
for _ in range(3):
    env.step(_z)
p_sim = (raw.wrist_pos_w[0] - raw.scene.env_origins[0]).cpu().numpy().astype(np.float64)
print(f"\n[tol] 标称末端(模型) p={np.round(p0,4)} | 仿真腕读数 p={np.round(p_sim,4)} | "
      f"模型↔仿真 {np.linalg.norm(p0-p_sim)*100:.2f}cm (含 ee_link↔wrist_link 的固定偏移, "
      f"不影响横轴 —— 扰动全在模型系里定义)")

LEVELS = [(0.000, 0.0), (0.005, 0.0), (0.010, 0.0), (0.020, 0.0), (0.030, 0.0),
          (0.000, 5.0), (0.000, 10.0), (0.000, 15.0),
          (0.010, 5.0), (0.020, 10.0)]
rng = np.random.default_rng(0)
results = []
print(f"[tol] clip={args.clip} | {N} env × {args.steps} 步/档 | 口径 A (c0=0, gentle=1.0)\n")
for dp, dr_deg in LEVELS:
    dr = np.radians(dr_deg)
    pool, ok_frac, (ach_p, ach_r) = build_pool(p0, R0, ik, seed_q, dp, dr, args.pool, rng)
    raw.arm_start_pool = pool
    tag = f"pos {dp*100:4.1f}cm  rot {dr_deg:4.1f}°"
    if pool is None and (dp > 0 or dr > 0):
        print(f"  [{tag}]  ⚠ IK 全部失败, 跳过")
        continue
    n_pool = 0 if pool is None else len(pool)
    print(f"  起始位形池 {n_pool} 个 (IK 成功 {ok_frac*100:.0f}%, "
          f"实际到达 {ach_p*100:.2f}cm / {ach_r:.1f}°)", flush=True)
    res = run_block(tag, args.steps)
    res.update(d_pos_m=dp, d_rot_deg=dr_deg, pool=n_pool, ik_ok=ok_frac,
               achieved_pos_cm=ach_p * 100, achieved_rot_deg=ach_r)
    results.append(res)

raw.arm_start_pool = None
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(dict(clip=args.clip, checkpoint=args.checkpoint,
                   num_envs=N, steps=args.steps, levels=results), f, indent=1)
print(f"\n[tol] 写出 {args.out}")
print(f"{'='*72}\n{'位置误差':>10s} {'姿态误差':>10s} {'成功率':>10s}")
for r in results:
    print(f"{r['d_pos_m']*100:9.1f}cm {r['d_rot_deg']:9.1f}° {r['sr']*100:9.2f}%")
print("=" * 72)
env.close()
app.close()
