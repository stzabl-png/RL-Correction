"""零训练确定性评测 (支持 minimal / bimanual 口径) —— 物理栈对齐验收专用。

与 record.py 的场景构建逐项一致, 但不开相机不录像, 纯跑 N 步统计成功率:

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.eval_min --headless \
      --checkpoint approach100.pth --clip Grasp3 \
      --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215 \
      --approach --approach_only --minimal --num_envs 512 --steps 300

用途: 拿本地冠军 ckpt 在远端跑 —— ≥99% 说明物理栈对齐, 0% 说明还有层没对上。
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="Grasp2")
parser.add_argument("--grasp_prior", type=str, default="")
parser.add_argument("--prior_yaw", type=float, default=-1.0)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--approach", action="store_true")
parser.add_argument("--approach_only", action="store_true")
parser.add_argument("--minimal", action="store_true")
parser.add_argument("--orient_blend", action="store_true")
parser.add_argument("--stance_prefix", type=int, default=0)
parser.add_argument("--place", action="store_true")
parser.add_argument("--bimanual", action="store_true")
parser.add_argument("--prior_b", type=str, default="")
parser.add_argument("--prior_b_yaw", type=float, default=-1.0)
parser.add_argument("--curobo_ref", type=str, default="")
parser.add_argument("--ff_freeze_cm", type=float, default=5.0)
parser.add_argument("--ff_pull", type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("eval_min")
app = AppLauncher(args).app

import os  # noqa: E402

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
    if args.approach_only:
        env_cfg.approach_only = True
        env_cfg.action_space = 7
    apply_grasp_prior(env_cfg, args.grasp_prior, args.prior_yaw, approach=args.approach)
elif args.approach:
    raise SystemExit("--approach 必须配 --grasp_prior")
env_cfg.orient_blend = args.orient_blend
if args.approach:
    env_cfg.direct_grasp_prob = 0.0
    env_cfg.approach_t0_max = 0.0
    env_cfg.stance_prefix_frames = args.stance_prefix
if args.minimal:
    env_cfg.minimal_no_ff = True
    env_cfg.minimal_fixed_res = False
    import numpy as _np
    env_cfg.eps_pos, env_cfg.eps_rot = 0.01, _np.radians(15.0)
    env_cfg.eps_pos0, env_cfg.eps_rot0 = env_cfg.eps_pos, env_cfg.eps_rot
    env_cfg.w_imit0_approach = 0.0
    env_cfg.w_imit_ramp = 0.0
    env_cfg.stance_prob, env_cfg.retract_ratio = 1.0, 1.0
    env_cfg.direct_grasp_prob = 0.0
_EnvCls = GraspTaskEnv
if args.curobo_ref:
    assert args.bimanual
    env_cfg.curobo_ref_npz = os.path.abspath(args.curobo_ref)
    env_cfg.minimal_no_ff = False
    env_cfg.curobo_ff_freeze_cm = args.ff_freeze_cm
    env_cfg.curobo_ff_pull = args.ff_pull
if args.bimanual:
    assert args.prior_b and os.path.exists(args.prior_b)
    from tasks.pregrasp.bimanual_env import BimanualApproachEnv as _EnvCls  # noqa: F811
    env_cfg.prior_b_npz = os.path.abspath(args.prior_b)
    env_cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
    _a1, _o1 = env_cfg.action_space, env_cfg.observation_space
    env_cfg.action_space, env_cfg.observation_space = 2 * _a1, 2 * _o1
    env_cfg._obs_single = _o1
    for _sec, _key in (("algorithm", "priv_info_dim"), ("network", "actor_priv_dim")):
        _d = agent_cfg.get(_sec, {})
        if _key in _d:
            _d[_key] = 2 * int(_d[_key])
env_cfg.scene.num_envs = args.num_envs
env_cfg.closure_init_max = 0.0

base = _EnvCls(env_cfg)
base.gentle = 1.0
env = GymStyleEnvWrapper(base, clip_actions=env_cfg.clip_actions)
agent = PPO(env, output_dir="/tmp/evalmin", create_output_dir=False,
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True))
print(f"[eval_min] loading {args.checkpoint}")
agent.restore_test(os.path.abspath(args.checkpoint))
agent.set_eval()

obs_dict = env.reset()
n = args.num_envs
ever = torch.zeros(n, dtype=torch.bool, device=base.device)
with torch.no_grad():
    for t in range(args.steps):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        mu = agent.model.act_inference(_inp)
        obs_dict, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        ever |= base._sig["newly_success"].to(ever.device)
        if t % 25 == 0 and "d_pos" in base._sig:
            _d = base._sig["d_pos"]
            print(f"[trace] 步{t:3d} d_pos 均值 {_d.mean()*100:6.2f}cm | "
                  f"最小 {_d.min()*100:5.2f} | 最大 {_d.max()*100:5.2f} | "
                  f"到位 {base.arrived.float().mean()*100:5.1f}%", flush=True)

rate = ever.float().mean().item() * 100
print(f"\n[eval_min] 确定性成功率 {rate:6.2f}% (n={n}, {args.steps} 步内至少成功一次)")
env.close()
app.close()
