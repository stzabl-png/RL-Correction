"""接近段逐步几何探针: 回放 ckpt, 逐步记录 腕/参考 相对 GraspPose 的位置/姿态误差.

用途: 判断录像里"靠近→远离→再靠近"是参考轨迹固有形状还是策略自己的行为
(两条 d_pos 曲线重合 = 参考固有; 分叉 = 策略偏离).

    python -m tasks.pregrasp.diag_trace --checkpoint <ckpt> --clip Grasp3 \\
        --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz \\
        --prior_yaw 215 --stance_prefix 60 --headless --out trace.npz
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="Grasp3")
parser.add_argument("--grasp_prior", type=str, required=True)
parser.add_argument("--prior_yaw", type=float, default=-1.0)
parser.add_argument("--stance_prefix", type=int, default=0)
parser.add_argument("--orient_blend", action="store_true", help="必须与训练一致")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=320)
parser.add_argument("--out", type=str, default="trace.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("diag_trace")
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
apply_grasp_prior(env_cfg, args.grasp_prior, args.prior_yaw, approach=True)
env_cfg.direct_grasp_prob = 0.0
env_cfg.approach_t0_max = 0.0
env_cfg.stance_prefix_frames = args.stance_prefix
env_cfg.orient_blend = args.orient_blend
env_cfg.scene.num_envs = args.num_envs
raw = GraspTaskEnv(env_cfg)
raw.gentle = 1.0
env = GymStyleEnvWrapper(raw, clip_actions=env_cfg.clip_actions)

agent = PPO(env, output_dir="/tmp/_trace_tmp",
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

N = raw.num_envs
L = raw.q_ref.shape[0]
rec = {k: [] for k in ("d_pos", "d_rot", "phase", "ref_t",
                       "wrist_pos", "ref_pos", "done")}
obs_dict = env.reset()
with torch.no_grad():
    for t in range(args.steps):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        act = agent.model.act_inference(_inp)
        obs_dict, r, done, info = env.step(torch.clamp(act, -1.0, 1.0))
        d_pos, d_rot, _ = raw._align_err()
        tt = raw.ref_t.clamp(max=L - 1)
        rec["d_pos"].append(d_pos.cpu().numpy())
        rec["d_rot"].append(d_rot.cpu().numpy())
        rec["phase"].append(raw.task_phase.cpu().numpy())
        rec["ref_t"].append(raw.ref_t.cpu().numpy())
        rec["wrist_pos"].append((raw.wrist_pos_w - raw.scene.env_origins).cpu().numpy())
        rec["ref_pos"].append(raw.ref_wrist_pos[tt].cpu().numpy())
        d = done.bool() if torch.is_tensor(done) else torch.tensor(done).bool()
        rec["done"].append(d.cpu().numpy())

out = {k: np.stack(v) for k, v in rec.items()}
out["grasp_pos"] = raw._grasp_pos_w.cpu().numpy()
out["grasp_quat"] = raw._grasp_quat_w.cpu().numpy()
out["ref_wrist_pos_full"] = raw.ref_wrist_pos.cpu().numpy()
out["gs"] = raw.gs
out["prefix"] = args.stance_prefix
np.savez(args.out, **out)
print(f"[trace] 写出 {args.out}: {args.steps} 步 × {N} env, gs={raw.gs}, L={L}")
app.close()
