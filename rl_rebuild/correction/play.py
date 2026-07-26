"""交互式在 IsaacSim GUI 里跑训练好的策略: 按 Enter 跑一回合, 亲眼看抓取效果.

  # 开 GUI (不加 --headless), 从头抓 (rsi_prob=0), 单 env 最清楚:
  .venv-isaac/bin/python -m rl_rebuild.correction.play \
      --checkpoint logs/correction_pp0_anchor_base/<run>/stage1_nn/ep_4600_step_0150M_reward_162.92.pth

  --rsi_prob 0    从头完整抓 (真实测试; 默认)
  --rsi_prob 0.5  半路把物体塞手里, 看它能否保持握持
  --num_envs 1    单 env (默认, 最清楚); 调大可一屏看多个
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default=None,
                    help="给了 clip 就自动加载该 clip 最新 checkpoint (短命令, 推荐)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="显式 checkpoint 路径 (不给则按 --clip 自动找最新)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--rsi_prob", type=float, default=0.0,
                    help="0=从头完整抓(真实测试) / 0.5=半路塞手里看保持")
parser.add_argument("--eye", type=str, default="0.7,0.7,1.25")
parser.add_argument("--lookat", type=str, default="0.0,0.0,0.95")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# GPU 独占槽位: 同一时刻只允许一个 Isaac 进程占 GPU (见 utils/gpu_guard.py).
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("play")

app = AppLauncher(args).app          # 不传 --headless => 开 GUI 窗口

import yaml  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ViewerCfg  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402

if args.checkpoint:                  # 显式路径: 用它, 顺带从路径推断 clip
    ckpt = os.path.abspath(args.checkpoint)
    if args.clip is None:
        _exp = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ckpt))))
        _rest = _exp[len("correction_"):] if _exp.startswith("correction_") else _exp
        args.clip = next((k for k in sorted(clips.CLIPS, key=len, reverse=True)
                          if _rest == k or _rest.startswith(k + "_")), None)
        if args.clip is None:
            raise SystemExit(f"[play] 无法推断 clip (run='{_exp}'), 请显式 --clip")
else:                                # 只给 --clip: 自动找该 clip 最新 checkpoint
    if args.clip is None:
        raise SystemExit("[play] 请给 --clip (自动找最新 ckpt) 或 --checkpoint <路径>")
    import glob
    import re
    _pths = glob.glob(f"logs/correction_{args.clip}_*/*/stage1_nn/ep_*.pth")
    if not _pths:
        raise SystemExit(f"[play] 找不到 {args.clip} 的 checkpoint (logs/correction_{args.clip}_*/)")
    ckpt = os.path.abspath(max(_pths, key=lambda p: int(re.search(r"step_(\d+)M", p).group(1))))
    print(f"[play] 自动选最新: {os.path.basename(ckpt)}")

_HERE = os.path.dirname(os.path.abspath(__file__))
agent_cfg = yaml.safe_load(open(os.path.join(_HERE, "agents/ppo_correction.yaml")))
agent_cfg["algorithm"]["num_actors"] = args.num_envs

env_cfg = SharpaCorrectionEnvCfg()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.rsi_prob = args.rsi_prob
env_cfg.scene.num_envs = args.num_envs
_eye = tuple(float(x) for x in args.eye.split(","))
_lookat = tuple(float(x) for x in args.lookat.split(","))
env_cfg.viewer = ViewerCfg(eye=_eye, lookat=_lookat, origin_type="env", env_index=0)

base = SharpaCorrectionEnv(env_cfg)
env = GymStyleEnvWrapper(base, clip_actions=env_cfg.clip_actions)
# create_output_dir=False: 推理路径不建日志目录、不初始化 TBWriter/wandb
# (否则忘了传 SHARPA_WANDB=0 就会弹交互式登录把进程卡死 — CLAUDE.md 第 1 条)
agent = PPO(env, output_dir="/tmp/play_tmp", full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
print(f"\n[play] 加载 {os.path.basename(ckpt)}")
agent.restore_test(ckpt)
agent.set_eval()

_thr = env_cfg.success_hold_frac * env_cfg.hold_steps
print(f"[play] clip={args.clip}  rsi_prob={args.rsi_prob}  envs={args.num_envs}")
print("[play] 按 Enter 跑一回合看效果, Ctrl+C 退出\n")
ep = 0
try:
    while True:
        input(f">>> 按 Enter 跑第 {ep + 1} 回合 ...")
        obs = env.reset()
        lift, ncon, hold = 0.0, 0, 0        # 回合中采样峰值 (done 后 env 已复位, 读数会归零)
        with torch.no_grad():
            for _ in range(base.ep_total):
                _inp = {
                    "obs": agent.running_mean_std(obs["obs"]),
                    "priv_info": obs["priv_info"]}
                if "pointcloud" in obs:              # 点云策略需要 PointNet 分支
                    _inp["pointcloud"] = obs["pointcloud"]
                mu = agent.model.act_inference(_inp)
                obs, _r, done, _i = env.step(torch.clamp(mu, -1.0, 1.0))
                # 在 done(复位)之前采样, 取整回合峰值
                if not bool(done[0]):
                    lift = max(lift, (base.object.data.root_pos_w[0, 2]
                                      - base.scene.env_origins[0, 2] - base.obj_rest_z).item())
                    ncon = max(ncon, int(base._tip_contacts()[0].sum().item()))
                    hold = max(hold, int(base.hold_ok[0]))
                if bool(done[0]):
                    break
        # 结果: 用回合内峰值 hold 计数判成功
        succ = hold >= _thr
        print(f"    第{ep + 1}回合: {'✅ 抓起并保持住' if succ else '❌ 没抓稳'}  "
              f"物体抬升 {lift * 100:+.1f}cm  当前接触指数 {ncon}/5\n")
        ep += 1
except (KeyboardInterrupt, EOFError):
    print(f"\n[play] 退出 (共跑 {ep} 回合)")
env.close()
app.close()
