"""checkpoint -> 扭盖策略回放录像 (mp4). 验收用: 亲眼看抓法与拧动.

  OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. $PY -u \
      -m tasks.recon_kailang.bottle_reconstruction.unscrew_record \
      --checkpoint logs/unscrew/Unscrew2/<run>/stage1_nn/best.pth --headless

成功回合以"终止步奖励 > 15"识别 (成功奖 +20; reset 会把 released_latch 洗掉,
不能在 done 之后读 env 状态 —— 冒烟 v1 的同款坑, 见 LEDGER).
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="screw_unscrew_cap1_task")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out_dir", type=str, default=None, help="默认 <run_dir>/videos")
parser.add_argument("--eye", type=str, default="0.45,-0.45,1.40")
parser.add_argument("--lookat", type=str, default="-0.11,0.09,1.08")
parser.add_argument("--fps", type=int, default=0, help="0=实时(20fps); 10=半速慢放")
parser.add_argument("--ref", action="store_true", help="v2 轨迹跟随任务")
parser.add_argument("--dyn", action="store_true", help="Stage B 全物理瓶 (隐含 --ref)")
parser.add_argument("--steps", type=int, default=0, help="0=ep_total (约 1~2 回合)")
parser.add_argument("--stop_first", action="store_true",
                    help="录到目标 env 首个回合结束即停 (避免 reset 瞬移入镜)")
parser.add_argument("--env_index", type=int, default=0,
                    help="镜头跟随/统计的 env (配合 --seed 两遍法录成功回合)")
parser.add_argument("--seed", type=int, default=-1,
                    help=">=0 时在建环境后播种, 使两遍 rollout 可复现")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True                     # 离屏渲染必需

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_record")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.envs import ViewerCfg  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.unscrew_env import (  # noqa: E402
    UnscrewTaskCfg,
    UnscrewTaskEnv,
)
from tasks.recon_kailang.bottle_reconstruction.unscrew_ref_env import (  # noqa: E402
    UnscrewDynTaskCfg,
    UnscrewRefTaskCfg,
    UnscrewRefTaskEnv,
)

ckpt = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(os.path.dirname(ckpt))
out_dir = args.out_dir or os.path.join(run_dir, "videos")
tag = os.path.splitext(os.path.basename(ckpt))[0]
os.makedirs(out_dir, exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "unscrew_ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

if args.dyn:
    args.ref = True
env_cfg = (UnscrewDynTaskCfg() if args.dyn
           else UnscrewRefTaskCfg() if args.ref else UnscrewTaskCfg())
clips.configure_cfg(env_cfg, args.clip)
env_cfg.scene.num_envs = args.num_envs
env_cfg.viewer = ViewerCfg(
    eye=tuple(float(x) for x in args.eye.split(",")),
    lookat=tuple(float(x) for x in args.lookat.split(",")),
    origin_type="env", env_index=args.env_index, resolution=(720, 540))

base = (UnscrewRefTaskEnv if args.ref else UnscrewTaskEnv)(env_cfg, render_mode="rgb_array")
if args.seed >= 0:
    base.seed(args.seed)
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
raw = env.unwrapped
succ = ep = 0
ang_prev = 0.0
with torch.no_grad():
    for t in range(n_steps + 2):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        mu = agent.model.act_inference(_inp)
        obs_dict, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        E = args.env_index
        if t % 20 == 0:
            n_c = int(raw._cap_contacts()[E].sum())
            print(f"[record] 步{t:3d} 拧角 {float(torch.rad2deg(raw.screw_angle[E])):6.1f}° "
                  f"接触指数 {n_c} 释放 {int(raw.released_latch[E])} "
                  f"放置 {int(raw.placed_latch[E])}")
        if bool(done[E]):
            ep += 1
            ok = float(r[E]) > 15.0            # 成功奖 +20 只在释放步出现
            succ += ok
            print(f"[record] env{E} 回合 {ep} @步 {t}: "
                  f"{'✅ 拧开释放' if ok else '✗ 未成功'} (终止前拧角≈{ang_prev:.0f}°)")
            if args.stop_first:
                break
        ang_prev = float(torch.rad2deg(raw.screw_angle[E]))

print(f"\n[record] env{args.env_index}: {succ}/{ep} 成功  |  视频目录: {out_dir}")
env.close()
app.close()
