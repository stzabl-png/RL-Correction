"""GUI 实时观看 checkpoint 策略抓取 (确定性 mu, 循环回合, 关窗口退出).

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.play \
      --checkpoint logs/PreGrasp_0/<run>/stage1_nn/last.pth

不加 --headless (要开窗口). 可选:
  --c0_max 0.9   训练分布起点 (看失败模式)
  --slow 2       慢放倍率 (2 = 半速)
"""
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="Grasp2")
parser.add_argument("--grasp_prior", type=str, default="", help="prior npz 路径 (B 组)")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--c0_max", type=float, default=0.0)
parser.add_argument("--slow", type=float, default=1.0, help="慢放倍率 (2=半速)")
parser.add_argument("--eye", type=str, default="0.55,-0.85,1.45")
parser.add_argument("--lookat", type=str, default="-0.10,-0.08,0.92")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("play")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402
import os  # noqa: E402

from isaaclab.envs import ViewerCfg  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

env_cfg = GraspTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
if args.grasp_prior:
    env_cfg.grasp_prior_npz = args.grasp_prior
env_cfg.scene.num_envs = args.num_envs
env_cfg.closure_init_max = args.c0_max
env_cfg.viewer = ViewerCfg(
    eye=tuple(float(x) for x in args.eye.split(",")),
    lookat=tuple(float(x) for x in args.lookat.split(",")),
    origin_type="env", env_index=0, resolution=(1600, 900))

raw = GraspTaskEnv(env_cfg)
raw.gentle = 1.0
env = GymStyleEnvWrapper(raw, clip_actions=env_cfg.clip_actions)

agent = PPO(env, output_dir="/tmp/_play_tmp",
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
print(f"[play] loading {args.checkpoint}")
agent.restore_test(os.path.abspath(args.checkpoint))
agent.set_eval()

print("[play] 确定性 mu 策略, 循环回合中 —— 关窗口或 Ctrl-C 退出")
obs_dict = env.reset()
ep = succ = 0
dt = env_cfg.decimation / 240.0 * args.slow
with torch.no_grad():
    while app.is_running():
        t0 = time.time()
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        mu = agent.model.act_inference(_inp)
        obs_dict, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        s = raw._sig
        if bool(done[0]):
            ep += 1
            ok = bool(s["newly_success"][0])
            succ += ok
            why = "✅ 成功" if ok else next(
                (k for k in ("fell", "thrown", "pushed", "stuck",
                             "table_crash", "timeout") if bool(s[k][0])), "?")
            print(f"[play] env0 回合 {ep}: {why}   (累计 {succ}/{ep})", flush=True)
        el = time.time() - t0
        if el < dt:
            time.sleep(dt - el)
env.close()
app.close()
