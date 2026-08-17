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
parser.add_argument("--retract", action="store_true",
                    help="**完整任务(接近+抓握)**用退避式起点族取代沿人手轨迹采 t0。"
                         "与 --approach_only 的区别: 这个仍然要抓要抬, 判据是抓稳+微抬升。")
parser.add_argument("--approach_only", action="store_true",
                    help="**只学接近**: 站姿->GraspPose, 不抓不抬不合拢 "
                         "(docs/APPROACH_DESIGN.md). 自动开退避起点族 + 外壳口径臂罚。")
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
parser.add_argument("--grasp_first", action="store_true",
                    help="先抓后飞门控 (§2.17): 阶段A 全回合从抖动 GraspPose 起步纯练抓取, "
                         "sr/from_grasp 慢 EMA ≥ gf_target 后放行接近分支")
parser.add_argument("--gf_target", type=float, default=0.8,
                    help="门控毕业线 (抖动起点分布上的抓取成功率 EMA; 0.9 会与技能平台期打平, 永不触发)")
parser.add_argument("--place", action="store_true",
                    help="PickAndPlace (§2.19): 抬升验证后接 搬运+放置, 成功=放到人手示范位 ±3cm; 评测须同 flag")
parser.add_argument("--start_jitter", action="store_true",
                    help="直接抓取回合起点用抖动池 (±eps0 包络), 不依赖 --grasp_first. "
                         "用途: 给已会飞的 ckpt 补'从到达偏差状态起抓'这门课 (O1 交接断裂的解药)")
# 大batch(3072)调参覆盖 (不填=用 ppo.yaml 默认)
parser.add_argument("--kl_threshold", type=float, default=None, help="覆盖 kl_threshold (放开策略步长)")
parser.add_argument("--mini_epochs", type=int, default=None, help="覆盖 mini_epochs (每batch多更新)")
parser.add_argument("--minibatch", type=int, default=None, help="覆盖 minibatch_size (不填=min(num_envs*8,32768))")
parser.add_argument("--lr", type=float, default=None, help="覆盖 learning_rate (√k 缩放)")
parser.add_argument("--entropy_coef", type=float, default=None, help="覆盖 entropy_coef (大batch保探索)")
parser.add_argument("--adam_betas", type=str, default=None, help="覆盖 Adam betas, 逗号分隔 如 0.5,0.9")
# ---- 自动停止 (2026-08-09): 周期性进程内确定性评测 + 收敛/平台期判停 ----
parser.add_argument("--auto_stop", choices=("off", "dry", "on"), default="off",
                    help="off=完全不跑(默认, 行为不变) | dry=评测+记录但**不停** (先用它验证判据) "
                         "| on=达标或平台期就结束训练")
parser.add_argument("--eval_every", type=int, default=100,
                    help="每多少 epoch 插一次确定性评测 (100 epoch≈3.3M 步≈27min, "
                         "单次评测 ~82s = 5%% overhead)")
parser.add_argument("--eval_steps", type=int, default=160,
                    help="评测控制步数; 必须 > 回合上限 (153) 才能保证每个 env 走完一回合")
parser.add_argument("--stop_target_sr", type=float, default=0.99,
                    help="早退线: 确定性成功率达到它 (且连续 --stop_target_hits 次) 就停")
parser.add_argument("--stop_target_hits", type=int, default=2,
                    help="达标要连续几次评测 (2 次 = 稳住 ~55min, 滤掉蜜月尖峰)")
parser.add_argument("--stop_patience", type=int, default=5,
                    help="主判据: 最佳成功率连续这么多次评测没涨够 --stop_delta 就停")
parser.add_argument("--stop_delta", type=float, default=0.01,
                    help="算'涨了'的最小增量 (1pt ≈ 3×SE@n=1024, 小于它当噪声)")
parser.add_argument("--stop_min_steps", type=float, default=20e6,
                    help="这么多步之前一律不停 (只记账). 难物体的爬坡期可以很长, "
                         "别让平台期判据在前期把它掐死")
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

import contextlib  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from datetime import datetime  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402

from tasks.pregrasp.auto_stop import StopDecider  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402


def _scalar_snapshot(cfg):
    """cfg 上所有标量字段的浅快照 (给 eval_distribution 的还原自检用)."""
    out = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg, k)
        except Exception:
            continue
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
    return out


@contextlib.contextmanager
def eval_distribution(raw):
    """把 env 临时切到 `tasks/pregrasp/eval.py` 的**口径A** (正式对外口径), 退出原样还原.

    ⚠ 用 contextmanager 而不是手写成对赋值: 漏还原任何一项都是**静默失效** ——
    不报错, 但之后的训练跑在被偷偷改过的分布上, 几小时后看曲线才发现 (§6.5 同款)。

    与训练分布的差别 (这就是 CLAUDE.md 说"训练期 TB 会低估"的全部来源):
      closure_init_max 0.9→0   手从完全张开起步, 不给合拢脚手架
      direct_grasp_prob →0     全部回合从接近段起步 (课程只退火到 0.1, 永远差这一截)
      approach_t0_max   →0     从 q_ref[0] 出发, 无起点随机化
      eps_pos/eps_rot   →final 切换阈值用最终的紧公差
      arm_start_pool    →None  eval.py 不建抖动池, 起点是精确 pregrasp
      gentle            →1.0   全价惩罚 (只影响奖励读数, 不影响成功判据)
    """
    cfg = raw.cfg
    _KEYS = ("closure_init_max", "direct_grasp_prob", "approach_t0_max",
             "eps_pos", "eps_rot", "retract_ratio", "stance_prob")
    saved = {k: getattr(cfg, k) for k in _KEYS if hasattr(cfg, k)}
    saved_gentle, saved_pool = float(raw.gentle), raw.arm_start_pool
    saved_to0 = int(raw.phase_timeout_t[0])
    # 全量标量快照: 上面那份 _KEYS 只还原"我知道自己改了的", 这份负责抓"改了但忘了列"
    # —— 后者是静默失效, 不 assert 的话要等几小时后看曲线才发现
    audit = _scalar_snapshot(cfg)
    try:
        cfg.closure_init_max = 0.0
        raw.gentle = 1.0
        raw.arm_start_pool = None
        if getattr(cfg, "approach", False):
            cfg.direct_grasp_prob = 0.0
            cfg.approach_t0_max = 0.0
            if hasattr(raw, "_eps_final"):
                cfg.eps_pos, cfg.eps_rot = raw._eps_final
            raw.phase_timeout_t[0] = raw.gs + int(cfg.approach_extra_steps)
        if getattr(cfg, "retract_start", False):
            # ★ 退避式接近的正式口径 = **全部从对称站姿起步**。
            #   注意方向: 这里设成 **1.0**(最难), 而不是像 approach_t0_max 那样设成 0。
            #   两套课程方向相反, 混了就是把评测变成"空降到终点"(DESIGN_LOOP §2.24)。
            cfg.stance_prob = 1.0
            cfg.retract_ratio = 1.0
        if getattr(cfg, "approach_only", False):
            raw.phase_timeout_t[0] = int(cfg.approach_only_steps)
        yield
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
        raw.gentle, raw.arm_start_pool = saved_gentle, saved_pool
        raw.phase_timeout_t[0] = saved_to0
        drift = {k: (v, audit[k]) for k, v in _scalar_snapshot(cfg).items()
                 if k in audit and v != audit[k]}
        assert not drift, (
            f"评测后 cfg 没还原干净 (评测值, 原值): {drift} —— "
            f"eval_distribution 改了这些字段但没列进 _KEYS, 训练分布已被污染")


class MilestonePPO(PPO):
    """PPO + 首达各成功率档位的步数记录 (sample-efficiency 指标)
    + 周期性确定性评测/自动停止 (--auto_stop).
    与旧 CorrectionPPO 的区别: 没有任何 reward/RSI 退火钩子."""

    MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.80, 0.90)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ms_path = os.path.join(self.output_dir, "milestones.json")
        self._ms_hit: dict = {}
        self._sr_ema = 0.0
        self._sr_slow = 0.0     # 慢速 EMA (4 倍慢): 只有**巩固**的成功率才推高价格
        # ---- 自动停止 (train.py 尾部按 CLI 覆盖; 判据在 auto_stop.StopDecider) ----
        self._auto_stop = "off"
        self._eval_every, self._eval_steps = 100, 160
        self._stopper = StopDecider()
        self._last_curr_sig = None
        self._eval_hist: list = []
        self._stop_path = os.path.join(self.output_dir, "auto_stop.json")

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
        # ---- 先抓后飞门控 (§2.17): 阶段A 纯抓取(dgp 钉 1.0), 毕业才放行接近课程 ----
        gated = getattr(self, "_grasp_first", False) and \
            not getattr(self, "_gate_open", False)
        if gated and raw is not None:
            fg = self.extra_info.get("sr/from_grasp")
            if fg is not None:
                # 0.95 而非课程惯用的 0.995: 毕业是一次性事件, 0.995 的半衰期 (~2M 步)
                # 会让进度条比技能本身晚数百万步; 0.95 仍要求 ~1M 步的持续高成功率.
                self._fg_slow = 0.95 * getattr(self, "_fg_slow", 0.0) + 0.05 * float(fg)
                self.writer.add_scalar("curr/gate_fg_slow", self._fg_slow, self.agent_steps)
                if self._fg_slow >= getattr(self, "_gf_target", 0.9):
                    self._gate_open = True
                    gated = False
                    raw.cfg.direct_grasp_prob = 0.5   # 放行: 回基线初值
                    self._dgp0 = 0.5                  # 退火锚点重置
                    self._sr_slow = 0.0               # 接近课程从头自然启动
                    print(f"[grasp_first] 毕业 @ {self.agent_steps/1e6:.2f}M 步 "
                          f"(抖动起点分布上的抓取慢EMA≥{self._gf_target}), 接近分支已放行")
        # ---- 组1 可行性课程: 由 **arrive_rate** 驱动 ----------------------
        # 相关能力是"到得了", 不是"抓得住", 所以驱动量用 arrive_rate 而非 success_rate.
        # 三条一起从"宽松"收到"目标", 纪律同 gentle: 慢速 EMA + 每 epoch 限速 + 棘轮.
        ar = self.extra_info.get("approach/arrive_rate")
        if raw is not None and getattr(raw.cfg, "approach", False) and ar is not None \
                and not gated:
            a = getattr(self, "_ar_ema", 0.995)
            self._ar_slow = a * getattr(self, "_ar_slow", 0.0) + (1.0 - a) * float(ar)
            g = min(self._ar_slow / max(raw.cfg.curr_arrive_target, 1e-6), 1.0)
            c = raw.cfg.curr_rate
            ep_f, er_f = raw._eps_final
            # ---- 回退闸 (2026-08-16 用户裁定) ----
            # 原实现: 收紧受每 epoch 限速, **放松却是瞬时的** —— `max(目标, 当前−限速)`
            #   里 g 一掉目标就变松, max 直接取松的那个。而 g 由 arrive_rate 驱动,
            #   arrive_rate 本身在剧烈抖动(实测 0.07~0.58), 等于让噪声在开车:
            #   成功率掉 -> 公差放松 -> 策略学会"贴着桌子勉强够角度"的糙解 -> 精度上不去
            #   -> 成功率再掉。实测 eps_rot 在 5.6~10.5° 之间来回甩, 从不收敛。
            # 用户裁定: **允许退, 但要真学坏了才退** —— 不是一有波动就退。
            #   判据: 慢 EMA 连续 curr_retreat_epochs 次低于历史最好的 curr_retreat_frac。
            _best = max(getattr(self, "_ar_best", 0.0), self._ar_slow)
            self._ar_best = _best
            _bad = self._ar_slow < _best * raw.cfg.curr_retreat_frac
            self._ar_bad = (getattr(self, "_ar_bad", 0) + 1) if _bad else 0
            _allow_loosen = self._ar_bad >= raw.cfg.curr_retreat_epochs
            if _allow_loosen:
                self._ar_bad = 0          # 退一档就重新计数, 免得连着退
            _tp = max(raw.cfg.eps_pos0 + (ep_f - raw.cfg.eps_pos0) * g,
                      raw.cfg.eps_pos - c * raw.cfg.eps_pos0)
            _tr = max(raw.cfg.eps_rot0 + (er_f - raw.cfg.eps_rot0) * g,
                      raw.cfg.eps_rot - c * raw.cfg.eps_rot0)
            if not _allow_loosen:         # 平时是棘轮: 只许收紧, 不许比现在更松
                _tp = min(_tp, raw.cfg.eps_pos)
                _tr = min(_tr, raw.cfg.eps_rot)
            elif _tp > raw.cfg.eps_pos or _tr > raw.cfg.eps_rot:
                print(f"[curr] 公差回退 @ {self.agent_steps/1e6:.2f}M: "
                      f"arrive 慢EMA {self._ar_slow:.3f} < 历史最好 {_best:.3f} × "
                      f"{raw.cfg.curr_retreat_frac} 连续 {raw.cfg.curr_retreat_epochs} 次 | "
                      f"eps {raw.cfg.eps_pos*100:.2f}->{_tp*100:.2f}cm "
                      f"{np.degrees(raw.cfg.eps_rot):.1f}->{np.degrees(_tr):.1f}°")
            raw.cfg.eps_pos, raw.cfg.eps_rot = _tp, _tr
            bud = raw.cfg.approach_extra0 + \
                (raw.cfg.approach_extra_steps - raw.cfg.approach_extra0) * g
            raw.phase_timeout_t[0] = raw.gs + int(max(bud, raw.cfg.approach_extra_steps))
            if not getattr(self, "_no_imit", False):
                raw.cfg.w_imit_ramp = min(g, raw.cfg.w_imit_ramp + c)
            self.writer.add_scalar("curr/ar_slow", self._ar_slow, self.agent_steps)
        # 接近段课程: 直接抓取起步比例 0.5 -> 0.1, t0 上限 0.8 -> 0 (逼它最终从头做).
        # 与 gentle 同一套纪律: 慢速 EMA 定价 + 每 epoch 限速 + 棘轮只降不升.
        if raw is not None and getattr(raw.cfg, "approach", False) and not gated:
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
        # ---- 退避式接近的课程 (2026-08-16, docs/APPROACH_DESIGN.md §2) ----
        # ★ 方向与上面那两条**相反**, 别混:
        #     approach_t0_max: 1 -> 0  (0 = 从头做 = 最难)
        #     retract_ratio  : 0 -> 1  (1 = d 铺满 [0,D] = 最难)
        #     stance_prob    : 0 -> 1  (1 = 全从站姿 = 正式任务口径)
        #   2026-08-16 就是把这两套方向搞混, 让评测变成"空降到抓握帧"(DESIGN_LOOP §2.24)。
        # 两段式: 先把 retract_ratio 拉满(学会各种距离的接近), 再把 stance_prob 拉满
        # (学会从站姿这个更远、姿态也不同的起点出发)。
        if raw is not None and getattr(raw.cfg, "retract_start", False):
            g = min(self._sr_slow / 0.3, 1.0)
            # 两段式 (2026-08-16 用户裁定, 恢复我误删的第②段):
            #   ① g∈[0,0.5]: retract_ratio 0→1  放宽下界, 从"只在终点"扩到"铺满整条路"
            #   ② g∈[0.5,1]: stance_prob  0→1  压低上界, 从"铺满"收到"全部最远端"
            #   bias=1 时训练分布 **完全等于** 评测分布(全部从站姿出发)。
            # ⚠ 只有第②段能让"最远那一档"从 1/84 变成 100% —— 少了它, 拉满后仍是均匀,
            #   而评测 100% 考最远档, 于是 TB 高、评测 0(2026-08-16 实测 0.37~0.55 vs 0%)。
            raw.cfg.retract_ratio = max(raw.cfg.retract_ratio, min(1.0, 2.0 * g))
            raw.cfg.stance_prob = max(raw.cfg.stance_prob,
                                      min(1.0, max(0.0, 2.0 * g - 1.0)))
            self.writer.add_scalar("curr/far_bias", raw.cfg.stance_prob, self.agent_steps)
            self.writer.add_scalar("curr/retract_ratio", raw.cfg.retract_ratio,
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

    # ================= 自动停止 =================
    def _curriculum_done(self, raw) -> bool:
        """课程是否已全部退火到位.

        判停前**必须**过这一关: 三条课程都在把任务变难 (gentle 涨价 / 阈值收紧 /
        起点脚手架撤除), 退火期成功率平甚至跌是设计内现象, 此时判平台期必误杀。
        """
        c = raw.cfg
        if getattr(self, "_grasp_first", False) and not getattr(self, "_gate_open", False):
            return False                      # 阶段A 还没毕业
        if raw.gentle < 0.995:                # 笨拙课程 (两种任务都有)
            return False
        if not getattr(c, "approach", False):
            return True                       # grasp-only: 只有笨拙课程
        if c.direct_grasp_prob > 0.1 + 1e-6 or getattr(c, "approach_t0_max", 0.0) > 1e-6:
            return False
        if hasattr(raw, "_eps_final"):        # 单调收紧, 落点是精确的 final 值
            if c.eps_pos > raw._eps_final[0] * 1.01 or c.eps_rot > raw._eps_final[1] * 1.01:
                return False
        if not getattr(self, "_no_imit", False) and getattr(c, "w_imit_ramp", 1.0) < 0.999:
            return False
        return True

    def _curriculum_signature(self, raw):
        """所有课程变量的快照. 两次评测之间**完全没变** = 课程停摆.

        为什么需要它: 每条课程的驱动量都是成功率 (gentle←sr_slow, eps←ar_slow,
        dgp←sr_slow), 成功率卡在低位时课程**也一起冻住**, `_curriculum_done`
        永远为假 —— 于是最该早停的绝望 run (sr 卡 2%) 反而永远停不下来。
        课程没在推进时"任务变难"这个理由不成立, 平坦就是真的没学到, 允许判平台期。
        (solved 分支不放行: 课程没退火完的高成功率是在更容易的任务上刷的.)
        """
        c = raw.cfg
        return (round(float(raw.gentle), 4),
                round(float(getattr(c, "direct_grasp_prob", 0.0)), 4),
                round(float(getattr(c, "approach_t0_max", 0.0)), 4),
                round(float(getattr(c, "eps_pos", 0.0)), 6),
                round(float(getattr(c, "eps_rot", 0.0)), 6),
                round(float(getattr(c, "w_imit_ramp", 0.0)), 4),
                bool(getattr(self, "_gate_open", False)))

    @torch.no_grad()
    def deterministic_eval(self, steps: int):
        """进程内确定性评测 (只跑 mu, 无 sigma 采样), 口径 = eval.py 口径A.

        **每个 env 只记第一个完成回合** -> n = num_envs 个独立样本。不这么做的话
        (像 eval.py 现在那样在固定步窗里数所有完成回合) 会系统性**高估**: 成功回合
        短 (提前 terminated), 同样时间里能多跑几个, 窗口边界偏向成功。
        """
        raw = self._raw_env
        n, dev = raw.num_envs, raw.device
        self.set_eval()          # 同时把 running_mean_std 切 eval -> 评测数据不污染归一化统计量
        with eval_distribution(raw):
            obs = self.env.reset()
            done_once = torch.zeros(n, dtype=torch.bool, device=dev)
            succ = torch.zeros(n, dtype=torch.bool, device=dev)
            used = steps
            for t in range(steps):
                inp = {"obs": self.running_mean_std(obs["obs"]),
                       "priv_info": obs["priv_info"]}
                if self.use_pc:
                    inp["pointcloud"] = obs["pointcloud"]
                act = torch.clamp(self.model.act_inference(inp), -1.0, 1.0)
                obs, _r, done, _info = self.env.step(act)
                d = done.bool()
                succ |= d & ~done_once & raw._sig["newly_success"]
                done_once |= d
                if bool(done_once.all()):
                    used = t + 1
                    break
        # ⚠ 训练侧 obs 必须重取: 上面 reset 打断了 1024 个在跑的回合, self.obs 已失效
        self.obs = self.env.reset()
        self.set_train()
        return float(succ.float().mean()), int(done_once.sum()), used

    def maybe_eval_and_stop(self):
        """epoch 边界钩子 (ckpt 刚落盘的干净点). 由 train.py 与 gpu_guard 让出点串联."""
        if self._auto_stop == "off" or self.epoch_num % self._eval_every:
            return
        raw = self._raw_env
        curr_ok = self._curriculum_done(raw)
        sig = self._curriculum_signature(raw)
        stalled = (sig == self._last_curr_sig)      # 首次评测 _last=None -> False
        self._last_curr_sig = sig
        sr, n_done, used = self.deterministic_eval(self._eval_steps)
        self.writer.add_scalar("eval/success_rate", sr, self.agent_steps)
        self.writer.add_scalar("eval/curriculum_done", float(curr_ok), self.agent_steps)
        self.writer.add_scalar("eval/curriculum_stalled", float(stalled), self.agent_steps)
        self._eval_hist.append(dict(agent_steps=int(self.agent_steps), epoch=int(self.epoch_num),
                                    sr=round(sr, 4), curriculum_done=bool(curr_ok),
                                    curriculum_stalled=bool(stalled)))
        if n_done < raw.num_envs:
            print(f"[eval] ⚠ 只有 {n_done}/{raw.num_envs} 个 env 在 {used} 步内跑完一回合 —— "
                  f"--eval_steps 需要 > 回合上限", flush=True)

        st = self._stopper
        was_best = sr > st.best_sr
        reason = st.update(sr, curr_ok, stalled, self.agent_steps)
        if was_best:
            # ⚠ 与基类的 best.pth 不同: 那个按 mean_rewards 存, 多项奖励里回报高 ≠ 成功率高
            self.save(os.path.join(self.nn_dir, "eval_best"))

        # flush: Isaac 的 C 层和 Python 共用 fd 但缓冲策略不同, 不 flush 的话这几行会被
        # 吞掉 (冒烟实测 49 条进度只落盘 5 条). 判停结论也落 auto_stop.json + TB, 双保险.
        print(f"[eval] ep{self.epoch_num} {self.agent_steps/1e6:.1f}M | 确定性成功率 "
              f"{sr*100:6.2f}% (n={n_done}) | 最佳 {st.best_sr*100:.2f}% | "
              f"课程{'已到位' if curr_ok else ('停摆' if stalled else '推进中')} | "
              f"达标{st.hits} 平台{st.stale}", flush=True)
        with open(self._stop_path, "w") as f:
            json.dump(dict(mode=self._auto_stop, best_sr=st.best_sr,
                           fired=st.fired, history=self._eval_hist), f, indent=1)
        if reason:
            print(f"[auto_stop] 触发 @ {self.agent_steps/1e6:.2f}M 步 ({self.epoch_num} epoch): {reason}",
                  flush=True)
            if self._auto_stop == "on":
                # 不动基类循环结构: 把上限拉到当前步数, while 条件下一轮自然为假
                self.max_agent_steps = self.agent_steps
                print("[auto_stop] 结束训练 (最佳权重在 stage1_nn/eval_best.pth)", flush=True)
            else:
                print("[auto_stop] dry-run: **只报告不停止**, 继续训练以便对照判据是否过早/过晚",
                      flush=True)


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
if args.grasp_first:      # 阶段 A: 100% 直接抓取回合 (毕业后训练钩子放行回 0.5)
    assert args.approach, "--grasp_first 是接近任务的课程, 必须配 --approach"
    env_cfg.direct_grasp_prob = 1.0
if args.place:
    assert args.approach, "--place 基于端到端任务, 必须配 --approach"
    env_cfg.place_task = True
    env_cfg.episode_length_s = 20.0   # 回合变长 (搬运+放置 ~100 步), 抬高兜底上限
    # grasp_only 时代把 gs 后的腕参考冻结 (物体钉桌上, 腕飘走=抓空气); place 的物体
    # 是真实物理且就该被搬走 —— 解冻, 参考播完整的 靠近→抓住→搬运→放置 (ref builder 注释)
    env_cfg.freeze_wrist = False
_prior = args.prior_npz or (os.path.join(_HERE, "priors", f"{args.clip}.npz")
                            if args.grasp_prior else None)
if _prior:
    assert os.path.exists(_prior), f"缺 prior 文件 {_prior} (先跑 make_prior.py / screen_prior.py)"
    # 一次做齐三件事: 挂 prior / 关 pregrasp_align(D7) / 钉 yaw(D1). 见 cfg.apply_grasp_prior
    if args.approach_only:
        # ⚠ 必须在 apply_grasp_prior **之前** —— 动作空间(7)与观测维度都依赖它,
        #   apply_grasp_prior 里就会按 _obs_base(cfg) 把 observation_space 算死。
        env_cfg.approach_only = True
        env_cfg.action_space = 7
    apply_grasp_prior(env_cfg, _prior, args.prior_yaw, approach=args.approach)
elif args.approach:
    raise SystemExit("--approach 必须配 prior (对齐势的终点来自 GraspPose)")
if args.retract and not args.approach_only:
    # 完整任务 + 退避起点族: 仍然要抓要抬, 只是起点换成"从 GraspPose 沿 radial 退 d"。
    # 动作空间保持 13 (抓握需要手指通道), 判据保持抓稳+微抬升。
    if not args.approach:
        raise SystemExit("--retract 必须同时给 --approach")
    env_cfg.retract_start = True
    env_cfg.retract_ratio = 0.0        # 训练从最简单端起 (与 approach_only 同一套语义)
    env_cfg.stance_prob = 0.0
    print("[retract] 完整任务(接近+抓握)改用退避式起点族; 动作空间保持 13 维")
if args.approach_only:
    if not args.approach:
        raise SystemExit("--approach_only 必须同时给 --approach (要接近段的相位机)")
    env_cfg.retract_start = True
    # ★ **训练**的课程初值 = 最简单端。cfg 里的默认 1.0/1.0 是**评测口径**
    #   (全部从站姿起步), 训练必须从 0 起, 由 sr_slow 逐步拉满。
    #   两个旋钮都是"越大越难" —— 与 approach_t0_max 方向相反, 别混
    #   (2026-08-16 就是混了这个, 评测被改成"空降到终点", DESIGN_LOOP §2.24)。
    env_cfg.retract_ratio = 0.0
    env_cfg.stance_prob = 0.0
    # (--eval_steps 的自动跟随统一放到 env 构造之后, 按真正的 ep_total 算)
    print(f"[approach_only] 只学接近: 站姿->GraspPose | 预算 "
          f"{env_cfg.approach_only_steps} 步 | 到位判据 "
          f"{env_cfg.eps_pos0*100:.0f}cm/{env_cfg.eps_rot0*57.3:.0f}° 保持 "
          f"{env_cfg.switch_hold} 步 | 硬终止: 臂外壳穿桌 / 手碰物体")
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
agent._grasp_first = args.grasp_first
agent._gf_target = args.gf_target
if args.grasp_first or args.start_jitter:
    # ---- §2.17 阶段A 起点抖动池 (用户设计, 2026-08-03) ----
    # 覆盖课程全程合法到达误差的包络 U(0, eps_pos0)×U(0, eps_rot0): 接近段到达
    # 永远 ≤ 当时的 eps ≤ 这个包络, 毕业 = 在这个分布上抓取 ≥ gf_target ——
    # 交接自带容错. 复用 tol_curve.py 验证过的 IK 抖动法 + arm_start_pool 钩子.
    import numpy as _np
    from rl_rebuild.correction.kinematics import ArmIK as _ArmIK
    _ik = _ArmIK(env_cfg.hand_side, anchor_link="arm_center", anchor_T=env_raw._anchor_T)
    _sq = env_raw.q_pregrasp.cpu().numpy().astype(_np.float64)
    _p0, _R0 = _ik.fk(_sq)
    _rng = _np.random.default_rng(args.seed)

    def _axang(v, th):
        v = v / _np.linalg.norm(v)
        K = _np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return _np.eye(3) + _np.sin(th) * K + (1 - _np.cos(th)) * K @ K

    _qs = [_sq] * 64                       # ~14% 精确起点
    _tries = 0
    while len(_qs) < 448 and _tries < 1600:
        _tries += 1
        _u = _rng.normal(size=3)
        _r = _ik.solve(_p0 + _u / _np.linalg.norm(_u) * _rng.uniform(0, env_cfg.eps_pos0),
                       _axang(_rng.normal(size=3), _rng.uniform(0, env_cfg.eps_rot0)) @ _R0,
                       q0=_sq, iters=80)
        if _r["ok"] and _r["pos_err"] < 0.002:
            _qs.append(_r["q"])
    env_raw.arm_start_pool = torch.tensor(_np.stack(_qs), dtype=torch.float32,
                                          device=env_raw.device)
    print(f"[grasp_first] 起点抖动池 {len(_qs)} 位形 (≤{env_cfg.eps_pos0*100:.0f}cm/"
          f"≤{_np.degrees(env_cfg.eps_rot0):.0f}°, 含 64 精确起点; IK 命中 "
          f"{len(_qs)-64}/{_tries}) | 毕业线 {args.gf_target}")
agent._auto_stop = args.auto_stop
# ★ eval_steps 必须 > 实际回合上限, 否则会有 env 跑不完一回合 ⟹ 成功率分母不是
#   num_envs(下面有断言拦)。**按 env 真正算出来的 ep_total 跟随**, 不要按 cfg 常量猜 ——
#   2026-08-16 我先按 approach_only_steps(250) 猜, 提到 300; 完整任务的 ep_total 是
#   403, 于是同一道断言又崩一次。ep_total 只有 env 构造完才知道, 所以修正放在这里。
if getattr(env_cfg, "retract_start", False) and args.eval_steps <= env_raw.ep_total:
    args.eval_steps = int(env_raw.ep_total * 1.2)
    print(f"[retract] --eval_steps 自动提到 {args.eval_steps} "
          f"(env 实际回合上限 ep_total={env_raw.ep_total})")
agent._eval_every, agent._eval_steps = args.eval_every, args.eval_steps
agent._stopper = StopDecider(target_sr=args.stop_target_sr, target_hits=args.stop_target_hits,
                             patience=args.stop_patience, delta=args.stop_delta,
                             min_steps=int(args.stop_min_steps))
if args.auto_stop != "off":
    assert args.eval_steps > env_raw.ep_total, \
        (f"--eval_steps {args.eval_steps} ≤ 回合上限 {env_raw.ep_total} —— "
         f"会有 env 跑不完一回合, 成功率分母不是 num_envs")
    print(f"[auto_stop] {args.auto_stop} | 每 {args.eval_every} epoch 评测 "
          f"({args.num_envs} env × ≤{args.eval_steps} 步) | 达标线 {args.stop_target_sr:.0%}"
          f"×{args.stop_target_hits} | 平台期 {args.stop_patience} 次未涨 {args.stop_delta:.0%}"
          f" | {args.stop_min_steps/1e6:.0f}M 步前不停")

# 录像让出点: epoch 边界检查暂停请求, 避免两个 Isaac 同时满载触发电源 OCP
# ⚠ 串联而非覆盖 —— 直接赋值会把 gpu_guard 的让出点顶掉 (录像再也拿不到卡)
_yield_slot = _slot.yield_if_paused
agent.epoch_hook = lambda: (_yield_slot(), agent.maybe_eval_and_stop())

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
