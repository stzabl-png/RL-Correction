"""残差修正训练入口 (--robot 选飞手 / DexMate) — 复用冠军 PPO 栈 (algo/ppo + v3net + wrapper).

  .venv-isaac/bin/python -m rl_rebuild.correction.train --num_envs 1024 --headless
  快速冒烟:  ... --num_envs 128 --max_agent_steps 100000 --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="flying", choices=("flying", "dexmate"),
                    help="flying=飞手(腕浮动根+wrench-PD, 动作28) / "
                         "dexmate=DexMate臂+Sharpa(关节驱动, 动作29). 见 env/registry.py")
parser.add_argument("--clip", type=str, default="clip11",
                    help="数据源 (clips.py 注册表): clip11 | pp0_human | pp0_anchor")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_agent_steps", type=int, default=None)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--video_every", type=int, default=2000,
                    help="每 N 次迭代自动录一段回放视频到 <run>/videos (0=关闭; "
                         "须为 save_frequency=100 的倍数). 录像期间训练会在 epoch 边界"
                         "让出 GPU 槽位挂起, 等录像进程完全退出后再继续 (见 utils/gpu_guard.py)")
parser.add_argument("--grasp_prior", action="store_true",
                    help="抓握段手指模仿目标用 GraspPose 纯抓姿 (消融 Exp1)")
parser.add_argument("--curobo_guide", action="store_true",
                    help="接近段加 cuRobo 预抓取路点引导 (退火; 消融 Exp1)")
parser.add_argument("--tag", type=str, default=None,
                    help="实验名后缀 (区分消融臂); 缺省按开关自动 gc/base")
parser.add_argument("--rsi_prob", type=float, default=None,
                    help="DeepMimic RSI 起步概率初值 (覆盖 cfg 默认0.8); 0=总从t0, 用于评测")
parser.add_argument("--no_pointcloud", action="store_true",
                    help="对照组: 关掉几何点云 (同时置 n_obj_points=0)")
parser.add_argument("--n_obj_points", type=int, default=None, help="覆盖点云点数 (调试显存)")
parser.add_argument("--rsi_mode", type=str, default=None, choices=["linear", "success"],
                    help="RSI 退火方式: linear=按步数线性 / success=成功率门控棘轮(能抓就降)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---- 功耗防护 (2026-07-24 整机断电事故后加, 详见 rl_rebuild/utils/gpu_guard.py) ----
# num_envs 软上限: 断电那几个 run 实际跑的是 2048 (从 ckpt 名 ep_100_step_0006M 反推),
# 是本文件默认值的 2 倍 —— GPU 持续功耗和尖峰幅度都随之翻倍, 而这台机器供电余量极薄.
# 需要更大规模时显式 RL_MAX_ENVS=2048 覆盖, 别再靠命令行悄悄传上去.
_MAX_ENVS = int(os.environ.get("RL_MAX_ENVS", "1024"))
if args.num_envs > _MAX_ENVS:
    print(f"[train] ⚠ num_envs {args.num_envs} 超过软上限 {_MAX_ENVS}, 已压回 {_MAX_ENVS}。"
          f"\n[train]   原因: 5090(600W)+14700K(253W) 在本机余量极薄, 2048 envs 的功耗尖峰"
          f"曾两次触发电源 OCP 整机瞬断 (2026-07-22 14:46 / 2026-07-24 16:15)。"
          f"\n[train]   确实需要更大规模: RL_MAX_ENVS={args.num_envs} 显式覆盖。")
    args.num_envs = _MAX_ENVS

# GPU 独占槽位: 在 Isaac 起来之前排队, 保证同一时刻只有一个 Isaac 进程在 GPU 上算.
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("train")

app = AppLauncher(args).app

import json  # noqa: E402
import subprocess  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from datetime import datetime  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402


class CorrectionPPO(PPO):
    """PPO + 里程碑记录: 首次达到各成功率档位时, 把步数/迭代/墙钟写进
    <run>/milestones.json (实验矩阵的 sample-efficiency 指标)."""

    MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.80, 0.90)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ms_path = os.path.join(self.output_dir, "milestones.json")
        self._ms_hit: dict = {}
        self._sr_ema = 0.0          # 成功率 EMA, 防单批噪声误触发

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        sr = self.extra_info.get("success_rate")
        if sr is None:
            return
        self._sr_ema = 0.98 * self._sr_ema + 0.02 * float(sr)
        self.writer.add_scalar("success_rate_ema", self._sr_ema, self.agent_steps)
        raw = getattr(self, "_raw_env", None)
        if raw is not None:
            # ConTrack 式自适应 imitation: 初期弱(0.2×放手探索任务), success 上升再加强到满.
            if not hasattr(self, "_base_imit"):
                self._base_imit, self._base_traj = raw.rw.lam_imit, raw.rw.lam_traj
            f = 0.2 + 0.8 * min(self._sr_ema / 0.3, 1.0)
            raw.rw.lam_imit, raw.rw.lam_traj = self._base_imit * f, self._base_traj * f
            self.writer.add_scalar("lam_imit", raw.rw.lam_imit, self.agent_steps)
            # RSI 课程退火: rsi_prob 初值 -> 0, 逼策略学"从头抓". 两种方式:
            if not hasattr(self, "_rsi_init"):
                self._rsi_init, self._rsi_cur = raw.cfg.rsi_prob, raw.cfg.rsi_prob
            if self._rsi_init > 0:
                if raw.cfg.rsi_mode == "success":
                    # 成功率门控棘轮: 当前RSI下基本能抓(sr_ema≥gate)就降 step, 只降不升, 到0
                    if self._sr_ema >= raw.cfg.rsi_gate_sr:
                        self._rsi_cur = max(0.0, self._rsi_cur - raw.cfg.rsi_step)
                    raw.cfg.rsi_prob = self._rsi_cur
                else:                       # linear: 按步数线性退火
                    prog = max(0.0, self.agent_steps - raw.cfg.rsi_warmup_steps) / max(raw.cfg.rsi_decay_steps, 1)
                    raw.cfg.rsi_prob = self._rsi_init * max(0.0, 1.0 - prog)
                self.writer.add_scalar("rsi_prob", raw.cfg.rsi_prob, self.agent_steps)
            # cuRobo 路点引导按能力退火: lam_curobo: init -> 0, sr_ema 到 wean_sr 时归零.
            if getattr(raw, "curobo_wp", None) is not None:
                frac = max(0.0, 1.0 - self._sr_ema / max(raw.cfg.curobo_wean_sr, 1e-6))
                raw.rw.lam_curobo = raw.cfg.lam_curobo_init * frac
                self.writer.add_scalar("lam_curobo", raw.rw.lam_curobo, self.agent_steps)
        for m in self.MILESTONES:
            key = f"{int(m * 100)}%"
            if key not in self._ms_hit and self._sr_ema >= m:
                self._ms_hit[key] = {
                    "agent_steps": int(self.agent_steps),
                    "epoch": int(self.epoch_num),
                    "wall_min": round((self.data_collect_time + self.rl_train_time) / 60, 1),
                }
                with open(self._ms_path, "w") as f:
                    json.dump(self._ms_hit, f, indent=1)
                print(f"[milestone] success_rate_ema >= {key} @ {self.agent_steps} steps")
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from rl_rebuild.correction.env.registry import make_env  # noqa: E402

EnvCls, EnvCfgCls = make_env(args.robot)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "agents/ppo_correction.yaml")) as f:
    agent_cfg = yaml.safe_load(f)

from rl_rebuild.correction import clips  # noqa: E402

env_cfg = EnvCfgCls()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.use_grasp_prior = args.grasp_prior      # 消融开关 (env __init__ 读)
env_cfg.use_curobo_guide = args.curobo_guide
if args.n_obj_points is not None:
    env_cfg.n_obj_points = args.n_obj_points
if args.no_pointcloud:                          # 点云对照开关
    env_cfg.enable_pointcloud = False
    env_cfg.n_obj_points = 0
if args.rsi_prob is not None:                   # RSI 起步概率覆盖 (评测用 0)
    env_cfg.rsi_prob = args.rsi_prob
if args.rsi_mode is not None:
    env_cfg.rsi_mode = args.rsi_mode
_tag = args.tag or ("gc" if (args.grasp_prior or args.curobo_guide) else "base")
_rb = "" if args.robot == "flying" else f"_{args.robot}"
agent_cfg["algorithm"]["experiment_name"] = f"correction_{args.clip}{_rb}_{_tag}"
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
agent_cfg["seed"] = args.seed
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent_cfg["algorithm"]["minibatch_size"] = min(args.num_envs * 8, 32768)
if args.max_agent_steps is not None:
    agent_cfg["algorithm"]["max_agent_steps"] = args.max_agent_steps
if args.load_path is not None:
    agent_cfg["load_path"] = args.load_path

log_dir = os.path.join(
    "logs", agent_cfg["algorithm"]["experiment_name"],
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print(f"[train] log_dir={log_dir}  envs={args.num_envs}  "
      f"minibatch={agent_cfg['algorithm']['minibatch_size']}")

env_raw = EnvCls(env_cfg)
env = GymStyleEnvWrapper(env_raw, clip_actions=env_cfg.clip_actions)
agent = CorrectionPPO(env, output_dir=log_dir, full_config=ConfigWrapper(agent_cfg, env_cfg))
agent._raw_env = env_raw        # 退火 hook 用: 按 sr_ema 调 rw.lam_curobo
# 录像让出点: 每个 epoch 边界 (刚存完 ckpt) 检查一次, 有录像请求就释放 GPU 槽位挂起,
# 等录像进程完全退出再继续 —— 避免两个 Isaac 同时满载把电源打到 OCP.
agent.epoch_hook = _slot.yield_if_paused

# 自动录像监视器 (子进程): 每 video_every 次迭代取周期快照录一段回放,
# 视频落在 <run>/videos/. 训练结束后它会补录完剩余快照再自行退出.
if args.video_every > 0:
    _watcher_log = open(os.path.join(log_dir, "autorecord.log"), "w")
    subprocess.Popen(
        [os.path.join(_HERE, "autorecord.sh"), os.path.abspath(log_dir),
         str(args.video_every), args.clip,     # ← 传 clip, 否则 record.py 默认 clip11 录错物体
         args.robot,                            # ← 传 robot, 否则录像端动作维度对不上 (28 vs 29)
         sys.executable],                       # ← 传本进程 python (跨机通用, 免写死本地路径)
        stdout=_watcher_log, stderr=subprocess.STDOUT)
    print(f"[train] 自动录像已启动: 每 {args.video_every} 迭代 -> {log_dir}/videos/ "
          f"(监视器日志 autorecord.log)")

if args.resume and agent_cfg["load_path"] not in (None, "None"):
    print(f"[train] resume from {agent_cfg['load_path']}")
    agent.restore_train(agent_cfg["load_path"])

agent.train()
print("[train] 训练结束; 若仍有待录快照, 后台监视器会补录完并自行退出 (autorecord.log)")
env.close()
app.close()
