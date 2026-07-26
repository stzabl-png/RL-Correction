"""checkpoint -> 回放录像 (mp4) + 该回合逐项统计. 也是将来收割成功轨迹的底座.

  .venv-isaac/bin/python -m rl_rebuild.correction.record \
      --checkpoint logs/correction_clip11/<run>/stage1_nn/best.pth --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--zero_action", action="store_true",
                    help="不载策略, 全零残差回放参考轨迹 (基线对照录像)")
parser.add_argument("--rsi_prob", type=float, default=None)
parser.add_argument("--out_dir", type=str, default=None,
                    help="默认: <run_dir>/videos")
parser.add_argument("--robot", type=str, default="flying",
                    choices=("flying", "dexmate"), help="须与训练时一致")
parser.add_argument("--clip", type=str, default=None,
                    help="缺省从 checkpoint 路径 correction_<clip>_<tag> 自动推断 (与训练同物体)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--eye", type=str, default="1.1,1.1,1.5")
parser.add_argument("--lookat", type=str, default="0.0,0.0,0.95")
parser.add_argument("--fps", type=int, default=0,
                    help="回放帧率覆盖: 0=实时(20fps); 10=半速慢放; 5=1/4速")
parser.add_argument("--no_stop_on_done", action="store_true",
                    help="默认 env0 首次终止即停录(避免重置跳变); 加此旗子录完整时长")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True                     # 离屏渲染必需

# GPU 独占槽位: 录像开的是**带渲染**的 Isaac, 和训练同时满载最容易触发电源 OCP.
# 这里阻塞等待 —— autorecord.sh 已先落 PAUSE 标记, 训练会在 epoch 边界让出槽位.
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("record")

app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.envs import ViewerCfg  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from rl_rebuild.correction.env.registry import make_env  # noqa: E402

EnvCls, EnvCfgCls = make_env(args.robot)

if args.zero_action:
    assert args.clip, "--zero_action 必须显式给 --clip"
    ckpt, run_dir = None, None
    out_dir = args.out_dir or os.path.join("logs", "_baseline_videos")
    tag = f"zero_action_{args.clip}"
else:
    ckpt = os.path.abspath(args.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    out_dir = args.out_dir or os.path.join(run_dir, "videos")
    tag = os.path.splitext(os.path.basename(ckpt))[0]
os.makedirs(out_dir, exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "agents/ppo_correction.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

from rl_rebuild.correction import clips  # noqa: E402

# clip 缺省 -> 从实验名目录 correction_<clip>_<tag> 推断 (录像必须与训练同物体)
if args.clip is None and not args.zero_action:
    _exp = os.path.basename(os.path.dirname(run_dir))          # correction_<clip>_<tag>
    _rest = _exp[len("correction_"):] if _exp.startswith("correction_") else _exp
    args.clip = next((k for k in sorted(clips.CLIPS, key=len, reverse=True)
                      if _rest == k or _rest.startswith(k + "_")), None)
    if args.clip is None:
        raise SystemExit(f"[record] 无法从路径推断 clip (run='{_exp}'), 请显式传 --clip")
    print(f"[record] 自动推断 clip={args.clip}  (来自 {_exp})")

env_cfg = EnvCfgCls()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.scene.num_envs = args.num_envs
if args.rsi_prob is not None:
    env_cfg.rsi_prob = args.rsi_prob
_eye = tuple(float(x) for x in args.eye.split(","))
_lookat = tuple(float(x) for x in args.lookat.split(","))
env_cfg.viewer = ViewerCfg(eye=_eye, lookat=_lookat,
                           origin_type="env", env_index=0, resolution=(720, 540))

base = EnvCls(env_cfg, render_mode="rgb_array")
ep_total = base.ep_total
if args.fps > 0:
    base.metadata["render_fps"] = args.fps      # 编码帧率<采集率(20Hz) => 慢放
base = gym.wrappers.RecordVideo(
    base, video_folder=out_dir, name_prefix=tag,
    step_trigger=lambda s: s == 0, video_length=ep_total, disable_logger=True)
env = GymStyleEnvWrapper(base, clip_actions=env_cfg.clip_actions)

agent = None
if not args.zero_action:
    agent = PPO(env, output_dir=os.path.join(out_dir, ".tmp"),
                full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
                create_output_dir=False)   # 推理路径: 不建日志目录/不初始化 wandb
    print(f"[record] loading {ckpt}")
    agent.restore_test(ckpt)
    agent.set_eval()
else:
    print("[record] 零残差基线回放 (不载策略)")

obs_dict = env.reset()
with torch.no_grad():
    for t in range(ep_total + 2):
        if agent is None:
            mu = torch.zeros(args.num_envs, env_cfg.action_space, device=obs_dict["obs"].device)
        else:
            _inp = {
                "obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"],
            }
            if "pointcloud" in obs_dict:              # 点云策略需要 PointNet 分支
                _inp["pointcloud"] = obs_dict["pointcloud"]
            mu = agent.model.act_inference(_inp)
        obs_dict, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        # 默认: 镜头所在的 env0 一终止就停 (重置瞬移会让视频看着像快进)
        if not args.no_stop_on_done and bool(done[0]):
            print(f"[record] env0 在第 {t + 1} 步终止, 停止录制")
            break

log = env.unwrapped.extras.get("log", {})
print(f"\n[record] {tag}  (确定性 μ 策略, {args.num_envs} env)")
for k in sorted(log):
    print(f"  {k:<24} {log[k]:+.4f}")
env.close()
app.close()
print(f"[record] 视频输出: {out_dir}/{tag}-episode-0.mp4")
