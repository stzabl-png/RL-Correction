"""checkpoint -> PreGrasp 抓取回放录像 (mp4). 验收用: 亲眼看抓姿与验证抬升.

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.record \
      --checkpoint logs/PreGrasp_0/<run>/stage1_nn/last.pth --headless

默认录 ep_total(153) 步 ≈ 连续 3 个回合 (成功即重置, 会看到跳变, 属正常);
--fps 10 半速慢放看细节. c0=0 正式口径, gentle=1.0.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="Grasp2")
parser.add_argument("--grasp_prior", type=str, default="", help="prior npz 路径 (B 组)")
parser.add_argument("--prior_yaw", type=float, default=-1.0,
                    help="钉死的物体 yaw (度) = screen_prior Gate1 的 best_yaw")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--c0_max", type=float, default=0.0,
                    help="复位合拢随机上限: 0=正式口径 / 0.9=训练分布 (看失败模式用)")
parser.add_argument("--out_dir", type=str, default=None, help="默认 <run_dir>/videos")
parser.add_argument("--eye", type=str, default="0.55,-0.85,1.45")
parser.add_argument("--lookat", type=str, default="-0.10,-0.08,0.92")
parser.add_argument("--fps", type=int, default=0, help="0=实时(20fps); 10=半速慢放")
parser.add_argument("--steps", type=int, default=0, help="0=ep_total (≈3 回合)")
parser.add_argument("--approach", action="store_true",
                    help="端到端口径: 从人手轨迹起点 q_ref[0] 出发, 接近+抓取全程. "
                         "配 --stance_prefix K 时起点是默认站姿 (PLAN D2)")
parser.add_argument("--stance_prefix", type=int, default=0,
                    help="必须与训练时相同 (0=无前缀)")
parser.add_argument("--orient_blend", action="store_true",
                    help="必须与训练时相同 (改 obs 的参考通道)")
parser.add_argument("--place", action="store_true",
                    help="必须与训练时相同 (成功判据=放置达标)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True                     # 离屏渲染必需

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("record")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.envs import ViewerCfg  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

ckpt = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(os.path.dirname(ckpt))
out_dir = args.out_dir or os.path.join(run_dir, "videos")
tag = (os.path.splitext(os.path.basename(ckpt))[0]
       + (f"_c0max{args.c0_max:.1f}" if args.c0_max > 0 else ""))
os.makedirs(out_dir, exist_ok=True)

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
    env_cfg.direct_grasp_prob = 0.0   # 全部回合从接近段起步
    env_cfg.approach_t0_max = 0.0     # 从人手轨迹 t=0 (q_ref[0]) 出发
    env_cfg.stance_prefix_frames = args.stance_prefix
env_cfg.scene.num_envs = args.num_envs
env_cfg.closure_init_max = args.c0_max
env_cfg.viewer = ViewerCfg(
    eye=tuple(float(x) for x in args.eye.split(",")),
    lookat=tuple(float(x) for x in args.lookat.split(",")),
    origin_type="env", env_index=0, resolution=(720, 540))

base = GraspTaskEnv(env_cfg, render_mode="rgb_array")
base.gentle = 1.0
n_steps = args.steps or base.ep_total
if args.fps > 0:
    base.metadata["render_fps"] = args.fps
base = gym.wrappers.RecordVideo(
    base, video_folder=out_dir, name_prefix=tag,
    step_trigger=lambda s: s == 0, video_length=n_steps, disable_logger=True)
env = GymStyleEnvWrapper(base, clip_actions=env_cfg.clip_actions)

agent = PPO(env, output_dir=os.path.join(out_dir, ".tmp"),
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
print(f"[record] loading {ckpt}")
agent.restore_test(ckpt)
agent.set_eval()

obs_dict = env.reset()
succ = ep = 0
with torch.no_grad():
    for t in range(n_steps + 2):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        mu = agent.model.act_inference(_inp)
        obs_dict, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        raw = env.unwrapped
        s = raw._sig
        if bool(done[0]):
            ep += 1
            ok = bool(s["newly_success"][0])
            succ += ok
            why = "✅ 成功" if ok else next(
                (k for k in ("fell", "thrown", "pushed", "stuck",
                             "table_crash", "timeout") if bool(s[k][0])), "?")
            print(f"[record] env0 回合 {ep} @步 {t}: {why}")

print(f"\n[record] env0: {succ}/{ep} 成功  |  视频: {out_dir}/{tag}-episode-0.mp4")
env.close()
app.close()
