"""PreGrasp 任务训练入口.

  PY=/home/lyh/luhr/MagicSim/.venv/bin/python
  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.train --num_envs 1024 --headless
  快速冒烟:  ... --num_envs 128 --max_agent_steps 100000 --headless

与 rl_rebuild/correction/train.py 的关系: 只保留通用机制 (GPU 独占槽位 / num_envs
功耗软上限 / 里程碑记录), **不带** RSI 退火 / imitation 退火 / cuRobo 引导 —— 那些
钩子直接摸 rw.lam_* , 对本任务全是空集, 塞进旧入口只会污染已验证的文件.
录像: record.py 还不认识本任务 (动作 30 维 + 无参考回放), 暂不接 autorecord.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default="Grasp2", help="数据源 (clips.py 注册表)")
parser.add_argument("--name", type=str, default="PreGrasp_0",
                    help="run 名 (= logs/<name>/ 与 TensorBoard 实验名)")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_agent_steps", type=int, default=None)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--grasp_prior", action="store_true",
                    help="启用 Dexonomy GraspPose prior (tasks/pregrasp/priors/<clip>.npz)")
parser.add_argument("--prior_npz", type=str, default=None,
                    help="显式指定 prior npz (覆盖 --grasp_prior 的默认路径); 多候选筛选后用它")
parser.add_argument("--prior_yaw", type=float, default=-1.0,
                    help="**钉死**的物体 yaw (度) = screen_prior Gate1 的 best_yaw. "
                         "不给 = 退回旧的自搜 yaw (会破坏与视频一致性, 接近段不可用)")
parser.add_argument("--direct_grasp_prob", type=float, default=None,
                    help="接近段课程: 直接从抓取相位起步的回合比例初值 (默认取 cfg 的 0.5). "
                         "调高 = 更多数据保抓取, 少给接近段探索")
parser.add_argument("--ppo3", action="store_true",
                    help="组3 PPO: entropy .005 / e_clip .1 / mini_epochs 8 / horizon 64 "
                         "(多试但步子小)")
parser.add_argument("--no_eps_curr", action="store_true",
                    help="关掉切换阈值课程 (直接用最终的 1.04cm/2.18°) —— E 组消融")
parser.add_argument("--no_imit", action="store_true",
                    help="模仿罚恒为 0 (完全不要求像人) —— F 组消融")
parser.add_argument("--ref_look", type=int, default=None,
                    help="参考前瞻帧数 (1=原行为 7 维; 5=借鉴 ConTrack, 35 维, 几何间隔 1/2/4/8/16)")
parser.add_argument("--approach", action="store_true",
                    help="开接近段 (pick_lift 完整任务). 不给 = 只训抓取 (旧任务)")
parser.add_argument("--obj_jitter", type=float, default=None,
                    help="覆盖物体 xy 抖动 (m); 课程 3mm->6->10->15, 每档一条 run")
parser.add_argument("--ar_ema", type=float, default=0.995,
                    help="arrive_rate 慢 EMA 系数 (课程回退速度; N3 试 0.98, 半衰期 2M→0.5M 步)")
parser.add_argument("--stance_prefix", type=int, default=0,
                    help=">0: 参考轨迹前拼 K 帧 默认站姿->人手轨迹起点 (PLAN D2 端到端起点); 评测须同值")
parser.add_argument("--orient_blend", action="store_true",
                    help="方案一 (§2.15): 参考姿态混合到 GraspPose, 边前进边转向; 评测须同 flag")
parser.add_argument("--cone", action="store_true",
                    help="方案二 (§2.16): 锥形信任管接替增量模仿罚; 训练时配 --no_imit")
# 大batch(3072)调参覆盖 (不填=用 ppo.yaml 默认)
parser.add_argument("--kl_threshold", type=float, default=None, help="覆盖 kl_threshold (放开策略步长)")
parser.add_argument("--mini_epochs", type=int, default=None, help="覆盖 mini_epochs (每batch多更新)")
parser.add_argument("--minibatch", type=int, default=None, help="覆盖 minibatch_size (不填=min(num_envs*8,32768))")
parser.add_argument("--lr", type=float, default=None, help="覆盖 learning_rate (√k 缩放)")
parser.add_argument("--entropy_coef", type=float, default=None, help="覆盖 entropy_coef (大batch保探索)")
parser.add_argument("--adam_betas", type=str, default=None, help="覆盖 Adam betas, 逗号分隔 如 0.5,0.9")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---- 功耗防护 (同 rl_rebuild/correction/train.py; 事故记录见 CLAUDE.md ⚡条) ----
_MAX_ENVS = int(os.environ.get("RL_MAX_ENVS", "1024"))
if args.num_envs > _MAX_ENVS:
    print(f"[train] ⚠ num_envs {args.num_envs} 超过软上限 {_MAX_ENVS}, 已压回 "
          f"(需要更大规模用 RL_MAX_ENVS={args.num_envs} 显式覆盖)")
    args.num_envs = _MAX_ENVS

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("train")

app = AppLauncher(args).app

import json  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from datetime import datetime  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402


class MilestonePPO(PPO):
    """PPO + 首达各成功率档位的步数记录 (sample-efficiency 指标).
    与旧 CorrectionPPO 的区别: 没有任何 reward/RSI 退火钩子."""

    MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.80, 0.90)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ms_path = os.path.join(self.output_dir, "milestones.json")
        self._ms_hit: dict = {}
        self._sr_ema = 0.0
        self._sr_slow = 0.0     # 慢速 EMA (4 倍慢): 只有**巩固**的成功率才推高价格

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        sr = self.extra_info.get("success_rate")
        if sr is None:
            return
        self._sr_ema = 0.98 * self._sr_ema + 0.02 * float(sr)
        self.writer.add_scalar("success_rate_ema", self._sr_ema, self.agent_steps)
        # 笨拙课程: 轻柔类惩罚按成功率涨价 (0.2 -> 1.0), 见 cfg.gentle_init 注释
        # ⚠ 三重保护 (每条都是一次失败 run 换来的):
        #   棘轮只涨不跌 —— 无棘轮出极限环 (sr 19%→7%, 靠退步换降价);
        #   慢速 EMA 定价 —— 用瞬时 EMA 定价会在蜜月尖峰上把价顶到 0.78, 超出已巩固
        #     技能, 尝试期望值转负, 技能被慢性剪除 (续跑实测 14.5%→5.8%);
        #   每 epoch 限速 +0.005 —— 全程涨价至少铺开 5M 步, 价格永远不冲到技能前头.
        self._sr_slow = 0.995 * self._sr_slow + 0.005 * float(sr)
        raw = getattr(self, "_raw_env", None)
        if raw is not None:
            tgt = 0.2 + 0.8 * min(self._sr_slow / 0.2, 1.0)
            cur = getattr(raw, "gentle", 0.2)
            raw.gentle = max(cur, min(tgt, cur + 0.005))
            self.writer.add_scalar("gentle", raw.gentle, self.agent_steps)
        # ---- 组1 可行性课程: 由 **arrive_rate** 驱动 ----------------------
        # 相关能力是"到得了", 不是"抓得住", 所以驱动量用 arrive_rate 而非 success_rate.
        # 三条一起从"宽松"收到"目标", 纪律同 gentle: 慢速 EMA + 每 epoch 限速 + 棘轮.
        ar = self.extra_info.get("approach/arrive_rate")
        if raw is not None and getattr(raw.cfg, "approach", False) and ar is not None:
            a = getattr(self, "_ar_ema", 0.995)
            self._ar_slow = a * getattr(self, "_ar_slow", 0.0) + (1.0 - a) * float(ar)
            g = min(self._ar_slow / max(raw.cfg.curr_arrive_target, 1e-6), 1.0)
            c = raw.cfg.curr_rate
            ep_f, er_f = raw._eps_final
            # 目标值 = 起点 --g--> 终点; 实际值只能朝目标**单调收紧**, 每 epoch 限速
            raw.cfg.eps_pos = max(raw.cfg.eps_pos0 + (ep_f - raw.cfg.eps_pos0) * g,
                                  raw.cfg.eps_pos - c * raw.cfg.eps_pos0)
            raw.cfg.eps_rot = max(raw.cfg.eps_rot0 + (er_f - raw.cfg.eps_rot0) * g,
                                  raw.cfg.eps_rot - c * raw.cfg.eps_rot0)
            bud = raw.cfg.approach_extra0 + \
                (raw.cfg.approach_extra_steps - raw.cfg.approach_extra0) * g
            raw.phase_timeout_t[0] = raw.gs + int(max(bud, raw.cfg.approach_extra_steps))
            if not getattr(self, "_no_imit", False):
                raw.cfg.w_imit_ramp = min(g, raw.cfg.w_imit_ramp + c)
            self.writer.add_scalar("curr/ar_slow", self._ar_slow, self.agent_steps)
        # 接近段课程: 直接抓取起步比例 0.5 -> 0.1, t0 上限 0.8 -> 0 (逼它最终从头做).
        # 与 gentle 同一套纪律: 慢速 EMA 定价 + 每 epoch 限速 + 棘轮只降不升.
        if raw is not None and getattr(raw.cfg, "approach", False):
            g = min(self._sr_slow / 0.3, 1.0)          # 巩固成功率 30% 时退火到位
            # ⚠ 退火终点固定 0.1, **起点取 run 的初值** —— 原来把 0.5 写死在公式里,
            #   初值调成 0.8 时第一次更新就会被 min() 直接压回 0.5, 实验等于没做.
            if not hasattr(self, "_dgp0"):
                self._dgp0 = float(raw.cfg.direct_grasp_prob)
            raw.cfg.direct_grasp_prob = min(raw.cfg.direct_grasp_prob,
                                            max(0.1, self._dgp0 - (self._dgp0 - 0.1) * g))
            raw.cfg.approach_t0_max = min(getattr(raw.cfg, "approach_t0_max", 0.8),
                                          max(0.0, 0.8 - 0.8 * g))
            self.writer.add_scalar("curr/direct_grasp_prob", raw.cfg.direct_grasp_prob,
                                   self.agent_steps)
            self.writer.add_scalar("curr/approach_t0_max", raw.cfg.approach_t0_max,
                                   self.agent_steps)
        # 接触分数图: 每 epoch 折扣一次 + 定期落盘 (常驻, 见 cfg.score_map)
        if raw is not None and getattr(raw, "score_on", False):
            raw.decay_score()
            if self.epoch_num % 100 == 0:
                raw.save_score_map(os.path.join(self.output_dir, "score_map.npz"),
                                   extra_meta=dict(epoch=int(self.epoch_num),
                                                   agent_steps=int(self.agent_steps)))
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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)

env_cfg = GraspTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
if args.obj_jitter is not None:
    env_cfg.obj_jitter_xy = args.obj_jitter
if args.direct_grasp_prob is not None:
    env_cfg.direct_grasp_prob = args.direct_grasp_prob
if args.ref_look is not None:
    env_cfg.ref_look_frames = args.ref_look
if args.no_eps_curr:                 # E 组: 阈值不放松, 从一开始就是最终值
    env_cfg.eps_pos0, env_cfg.eps_rot0 = env_cfg.eps_pos, env_cfg.eps_rot
env_cfg.stance_prefix_frames = args.stance_prefix
env_cfg.orient_blend = args.orient_blend
env_cfg.cone_trust = args.cone
_prior = args.prior_npz or (os.path.join(_HERE, "priors", f"{args.clip}.npz")
                            if args.grasp_prior else None)
if _prior:
    assert os.path.exists(_prior), f"缺 prior 文件 {_prior} (先跑 make_prior.py / screen_prior.py)"
    # 一次做齐三件事: 挂 prior / 关 pregrasp_align(D7) / 钉 yaw(D1). 见 cfg.apply_grasp_prior
    apply_grasp_prior(env_cfg, _prior, args.prior_yaw, approach=args.approach)
elif args.approach:
    raise SystemExit("--approach 必须配 prior (对齐势的终点来自 GraspPose)")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
agent_cfg["seed"] = args.seed
agent_cfg["algorithm"]["experiment_name"] = args.name
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent_cfg["algorithm"]["minibatch_size"] = min(args.num_envs * 8, 32768)
if args.minibatch is not None:
    agent_cfg["algorithm"]["minibatch_size"] = args.minibatch
if args.kl_threshold is not None:
    agent_cfg["algorithm"]["kl_threshold"] = args.kl_threshold
if args.mini_epochs is not None:
    agent_cfg["algorithm"]["mini_epochs"] = args.mini_epochs
if args.lr is not None:
    agent_cfg["algorithm"]["learning_rate"] = args.lr
if args.entropy_coef is not None:
    agent_cfg["algorithm"]["entropy_coef"] = args.entropy_coef
if args.adam_betas is not None:
    agent_cfg["algorithm"]["adam_betas"] = [float(x) for x in args.adam_betas.split(",")]
if args.ppo3:
    agent_cfg["algorithm"]["entropy_coef"] = 0.005
    agent_cfg["algorithm"]["e_clip"] = 0.1
    agent_cfg["algorithm"]["mini_epochs"] = 8
    agent_cfg["algorithm"]["horizon_length"] = 64
if args.max_agent_steps is not None:
    agent_cfg["algorithm"]["max_agent_steps"] = args.max_agent_steps
if args.load_path is not None:
    agent_cfg["load_path"] = args.load_path

log_dir = os.path.join("logs", args.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print(f"[train] log_dir={log_dir}  clip={args.clip}  envs={args.num_envs}  "
      f"obj_jitter={env_cfg.obj_jitter_xy*1000:.0f}mm  "
      f"minibatch={agent_cfg['algorithm']['minibatch_size']}")

env_raw = GraspTaskEnv(env_cfg)
env = GymStyleEnvWrapper(env_raw, clip_actions=env_cfg.clip_actions)
agent = MilestonePPO(env, output_dir=log_dir, full_config=ConfigWrapper(agent_cfg, env_cfg))
agent._raw_env = env_raw
agent._no_imit = args.no_imit    # F 组: 模仿罚恒 0        # 笨拙课程钩子用: 按 sr_ema 更新 env.gentle
agent._ar_ema = args.ar_ema      # N3 组: 可行性课程 EMA 速度
# 录像让出点: epoch 边界检查暂停请求, 避免两个 Isaac 同时满载触发电源 OCP
agent.epoch_hook = _slot.yield_if_paused

if args.resume and agent_cfg["load_path"] not in (None, "None"):
    print(f"[train] resume from {agent_cfg['load_path']}")
    agent.restore_train(agent_cfg["load_path"])

agent.train()
print("[train] 训练结束")
if getattr(env_raw, "score_on", False):
    env_raw.save_score_map(os.path.join(log_dir, "score_map.npz"),
                           extra_meta=dict(final=True))
env.close()
app.close()
