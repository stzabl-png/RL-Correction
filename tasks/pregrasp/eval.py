"""PreGrasp 确定性评测 — 关探索噪声 (只用 mu), 大批量数真实成功率 + 失败普查.

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.eval \
      --checkpoint logs/PreGrasp_0/<run>/stage1_nn/last.pth --headless

与训练期 TB 指标的区别: 无 sigma 采样噪声, 成功按"验证通过事件/完成回合数"精确计,
gentle 钉 1.0 (只影响奖励读数, 不影响成功判据). 两个口径:
  A. c0=0     — 任务的标准起点 (手张开, 从头合拢), 这是**正式口径**
  B. c0~U(0,0.9) — 训练时的复位分布 (对照, 看起点随机化的影响)
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="Grasp2")
parser.add_argument("--grasp_prior", type=str, default="", help="prior npz 路径 (B 组)")
parser.add_argument("--prior_yaw", type=float, default=-1.0,
                    help="钉死的物体 yaw (度), 取 screen_prior Gate1 的 best_yaw")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=450, help="每口径跑多少控制步 (~3-7回合/env)")
parser.add_argument("--approach", action="store_true",
                    help="端到端口径: 从人手轨迹起点 q_ref[0] 出发, 接近+抓取全程; 阈值/预算用最终值. "
                         "配 --stance_prefix K 时起点是默认站姿 (PLAN D2)")
parser.add_argument("--stance_prefix", type=int, default=0,
                    help="必须与训练时相同 (0=无前缀)")
parser.add_argument("--orient_blend", action="store_true",
                    help="必须与训练时相同 (改 obs 的参考通道, 两种口径评测都要带)")
parser.add_argument("--place", action="store_true",
                    help="必须与训练时相同 (成功判据=放置达标)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("eval")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
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
    apply_grasp_prior(env_cfg, args.grasp_prior, args.prior_yaw, approach=args.approach)
elif args.approach:
    raise SystemExit("--approach 必须配 --grasp_prior (对齐势的终点来自 GraspPose)")
env_cfg.orient_blend = args.orient_blend   # 与 approach 无关: 参考通道常开, obs 必须对齐
if args.place:
    env_cfg.place_task = True
    env_cfg.episode_length_s = 20.0
    env_cfg.freeze_wrist = False   # 与训练一致: gs 后腕参考解冻 (搬运段)
if args.approach:
    env_cfg.direct_grasp_prob = 0.0   # 全部回合从接近段起步 (不走直接抓取分支)
    env_cfg.approach_t0_max = 0.0     # 从人手轨迹 t=0 (q_ref[0]) 出发, 无 t0 随机化脚手架
    env_cfg.stance_prefix_frames = args.stance_prefix
env_cfg.scene.num_envs = args.num_envs
raw = GraspTaskEnv(env_cfg)
raw.gentle = 1.0                          # 全价口径 (只影响奖励读数)
env = GymStyleEnvWrapper(raw, clip_actions=env_cfg.clip_actions)

agent = PPO(env, output_dir="/tmp/_eval_tmp",
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
print(f"[eval] loading {args.checkpoint}")
agent.restore_test(args.checkpoint)
agent.set_eval()

N, dev = raw.num_envs, raw.device
FAIL_KEYS = ("fell", "thrown", "pushed", "stuck", "table_crash", "timeout")


def run_block(tag, c0_max, steps):
    raw.cfg.closure_init_max = c0_max
    obs_dict = env.reset()
    succ = 0
    episodes = 0
    fails = {k: 0 for k in FAIL_KEYS}
    verify_fails = 0
    ep_len_succ = []
    # 失败几何普查: 记录每回合物体抖动偏移, 按结局分桶
    start_xy = raw.obj_start_pos[:, :2].clone()
    obj_nom = raw.obj_init_pos[:2].clone()
    off_succ, off_fail = [], []
    ep_start_t = torch.zeros(N, device=dev)
    pads_hist = torch.zeros(5, device=dev)
    pad_steps = 0
    with torch.no_grad():
        for t in range(steps):
            _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"]}
            if "pointcloud" in obs_dict:
                _inp["pointcloud"] = obs_dict["pointcloud"]
            act = agent.model.act_inference(_inp)
            obs_dict, r, done, info = env.step(torch.clamp(act, -1.0, 1.0))
            s = raw._sig
            pads_hist += s["pads_on"].float().sum(dim=0)
            pad_steps += N
            verify_fails += int(s["verify_fail"].sum())
            d = done.bool() if torch.is_tensor(done) else torch.tensor(done, device=dev).bool()
            if d.any():
                idx = torch.nonzero(d, as_tuple=False).squeeze(-1)
                episodes += len(idx)
                ns = s["newly_success"][idx]
                succ += int(ns.sum())
                off = (start_xy[idx] - obj_nom).norm(dim=1) * 1000.0   # mm
                off_succ += off[ns].tolist()
                off_fail += off[~ns].tolist()
                ep_len_succ += (t - ep_start_t[idx[ns]]).tolist()
                for k in FAIL_KEYS:
                    fails[k] += int((s[k][idx] & ~ns).sum())
                ep_start_t[idx] = t
                start_xy[idx] = raw.obj_start_pos[idx, :2]   # reset 后的新回合起点
    sr = succ / max(episodes, 1)
    print(f"\n{'='*66}\n[{tag}]  成功率 = {sr*100:.2f}%   ({succ}/{episodes} 回合)")
    print(f"  成功回合中位长度: {np.median(ep_len_succ):.0f} 步"
          if ep_len_succ else "  (无成功回合)")
    tot_f = max(episodes - succ, 1)
    print("  失败构成: " + "  ".join(
        f"{k}={v} ({v/tot_f*100:.0f}%)" for k, v in fails.items() if v > 0))
    print(f"  验证失败(重试)事件总数: {verify_fails}  (平均 {verify_fails/max(episodes,1):.1f} 次/回合)")
    print(f"  逐垫有效接触率: " + " ".join(
        f"{n}={v/max(pad_steps,1):.2f}" for n, v in
        zip(("拇", "食", "中", "环", "小"), pads_hist.tolist())))
    if off_succ and off_fail:
        print(f"  物体抖动偏移 |xy| : 成功回合均值 {np.mean(off_succ):.2f}mm vs "
              f"失败回合均值 {np.mean(off_fail):.2f}mm"
              f"   (差异大 → 失败集中在抖动边角)")
    return sr


print(f"\n[eval] 确定性 mu | clip={args.clip} | {N} env × {args.steps} 步 × 2 口径")
sr_a = run_block("口径A: c0=0 (标准起点, 正式口径)", 0.0, args.steps)
sr_b = run_block("口径B: c0~U(0,0.9) (训练分布对照)", 0.9, args.steps)
print(f"\n{'='*66}")
print(f"  正式成功率 (c0=0, 确定性): {sr_a*100:.2f}%   训练分布对照: {sr_b*100:.2f}%")
print(f"  判据: ≥4 垫有效力接触 + 向心达标 + 物体不动 0.2s → 腕升 1cm → "
      f"物体升 ≥5mm 且 ≥3 垫保持 0.25s")
print(f"{'='*66}")
env.close()
app.close()
