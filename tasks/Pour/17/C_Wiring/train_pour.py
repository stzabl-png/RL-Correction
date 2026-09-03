"""Pour17 训练入口 (接线工程).

  PY=/home/lyh/luhr/MagicSim/.venv/bin/python
  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Pour/17/C_Wiring/train_pour.py \
      --num_envs 512 --headless
  冒烟: ... --num_envs 64 --max_agent_steps 60000 --headless

钩子 (全部单旋钮 EMA 体制, 与 RSI/#13 拍板一致):
  p_t0    = 0.2 + 0.6*EMA(sr/gate4)      —— RSI 配比 (#11)
  相B开闸  = EMA(sr/gate1) >= 0.7 单向棘轮 —— #13 拍板3 (v1 只立旗+记录, 接触奖金
            与转运臂门放开的执行体挂 phase_b 旗, 见 pour_env TODO)
TB: 引擎自带 + sr/gate1-4 + prog/clock_frac + term/* 每 epoch 倾倒。
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="Pour17_0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_agent_steps", type=int, default=None)
parser.add_argument("--load_path", default=None)
parser.add_argument("--no_autorec", action="store_true",
                    help="关掉自动录像循环(默认训练进程自拉, 代码钉子防漏挂)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
assert args.headless or os.environ.get("POUR_GUI"), "训练必须 --headless (CLAUDE.md 铁则)"

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot(f"pour17_train_{args.name}")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pour_env as PE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


class PourPPO(PPO):
    """PPO + Pour 逐关率倾倒 + RSI 配比/相位单旋钮."""

    def __init__(self, *a, raw_env=None, **kw):
        super().__init__(*a, **kw)
        self._raw = raw_env
        self._ema_g1 = 0.0
        self._ema_g2 = 0.0
        self._ema_g3 = 0.0
        self._ema_g4 = 0.0
        # L5-8 药②: 无偏解锁 (t0出生口径 + 认证通过率, 且要求持续)
        self._ema_t0 = {1: 0.0, 2: 0.0, 3: 0.0}
        self._ema_cert = 0.0
        self._sustain = {1: 0, 2: 0, 3: 0}
        self._phase_b = bool(getattr(raw_env, "phase_b", False))  # C线开局即真

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        raw = self._raw
        if raw is None:
            return
        # 进度文件 (录像循环的读数源, 代码钉子)
        try:
            with open(os.path.join(self.output_dir, "progress_steps.txt"), "w") as _f:
                _f.write(str(int(self.agent_steps)))
        except Exception:
            pass
        rates = raw.PB.pop_rates()
        for k, v in rates.items():
            self.writer.add_scalar(k, v, self.agent_steps)
        for k, v in raw.pop_racc().items():          # 逐项奖惩台账
            self.writer.add_scalar(k, v, self.agent_steps)
        for k, v in raw.pop_term().items():          # L5-23 死因分项台账
            self.writer.add_scalar(k, v, self.agent_steps)
        if getattr(raw, "KCAP", 0) > 0:              # LIFT 单科考主针
            for k, v in raw.pop_lift().items():
                self.writer.add_scalar(k, v, self.agent_steps)
        # ★L5-21 NaN 防护: 空分母窗发 NaN(L5-17), 而 0.98*x+0.02*NaN = NaN ——
        # 一次污染后 EMA 永久为 NaN, 所有 `>=阈值` 恒 False ⟹ 课程永不解锁、
        # 相B永不开闸。实测验证过。旧行为(代入0.0)是把"没测到"当成"测到了0",
        # 方向保守但同样是假值。正解: **无数据 = 无证据 = 不动估计**。
        def _ema(prev, r, a=0.02):
            return prev if r != r else (1.0 - a) * prev + a * r
        self._ema_g1 = _ema(self._ema_g1, rates["sr/gate1"])
        self._ema_g2 = _ema(self._ema_g2, rates["sr/gate2"])
        self._ema_g3 = _ema(self._ema_g3, rates["sr/gate3"])
        self._ema_g4 = _ema(self._ema_g4, rates["sr/gate4"])
        # 渐进RSI 解锁 (L5-8 药②③): 无偏指标(t0出生口径) + 持续20窗 + 认证闸
        # 旧版用 sr/gate(有偏: 短回合先结算致虚高) 在假阳性上放开了课程, 实测
        # 解锁后 G2 断崖到 0 且再未恢复 —— 换指标+持续判定+G2额外认证闸。
        self._ema_cert = _ema(self._ema_cert, rates.get("sr/cert_pass", float("nan")))
        self.writer.add_scalar("curr/ema_cert", self._ema_cert, self.agent_steps)
        for _gi in (1, 2, 3):
            _r = rates.get(f"sr_t0/gate{_gi}", float("nan"))
            self._ema_t0[_gi] = _ema(self._ema_t0[_gi], _r)
            self.writer.add_scalar(f"curr/ema_t0_g{_gi}", self._ema_t0[_gi],
                                   self.agent_steps)
            if _r != _r:
                # ★空分母窗: 既不推进也不清零 sustain。清零会惩罚长回合线
                # (EASY 线 72% 的窗没有回合结算), 推进则是拿陈旧证据解锁。
                continue
            _ok = self._ema_t0[_gi] >= 0.30
            if _gi == 2:
                _ok = _ok and self._ema_cert >= 0.15      # 药③: 认证站稳才放 g2
            self._sustain[_gi] = self._sustain[_gi] + 1 if _ok else 0
            if self._sustain[_gi] >= 20 and _gi not in raw.unlocked:
                raw.unlocked.add(_gi)
                raw._rebuild_entries()
                print(f"[渐进RSI] Gate{_gi} 出生点解锁 @ "
                      f"{self.agent_steps/1e6:.2f}M (t0口径EMA={self._ema_t0[_gi]:.2f}"
                      f", 持续{self._sustain[_gi]}窗)")
        self.writer.add_scalar("curr/n_entries", len(raw.entries),
                               self.agent_steps)
        self.writer.add_scalar("curr/ema_gate1", self._ema_g1, self.agent_steps)
        # RSI 配比单旋钮 (#11) + L5-8 药①: t0 底线 0.5
        # (旧 0.2 致 80% 回合白送 G1/G2, 前段技能失去梯度 —— 实测 G1 塌到 0.2)
        raw.p_t0 = max(0.5, 0.2 + 0.6 * self._ema_g4)
        self.writer.add_scalar("curr/p_t0", raw.p_t0, self.agent_steps)
        # 相B单向棘轮 (#13): 立旗记录; 执行体挂 phase_b
        if not self._phase_b and self._ema_g1 >= 0.7:
            self._phase_b = True
            raw.phase_b = True
            print(f"[相位] 相B开闸 @ {self.agent_steps/1e6:.2f}M 步 "
                  f"(EMA gate1={self._ema_g1:.2f})")
        self.writer.add_scalar("curr/phase_b", float(self._phase_b), self.agent_steps)


cfg = PE.build_cfg(num_envs=args.num_envs)
cfg.seed = args.seed
raw = PE.PourEnv(cfg)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)

with open(os.path.join(_HERE, "ppo_pour.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["seed"] = args.seed
agent_cfg["algorithm"]["experiment_name"] = args.name
agent_cfg["algorithm"]["num_actors"] = args.num_envs
if args.max_agent_steps:
    agent_cfg["algorithm"]["max_agent_steps"] = args.max_agent_steps
# minibatch 不得超过 batch (horizon*num_envs)
bsz = 32 * args.num_envs
agent_cfg["algorithm"]["minibatch_size"] = min(
    agent_cfg["algorithm"]["minibatch_size"], bsz)

log_dir = os.path.join("logs", args.name)
os.makedirs(log_dir, exist_ok=True)
# ---- 世界指纹 (完整版, L5-12): ckpt 只能在出生世界回放, 这里落"身份证" ----
import hashlib  # noqa: E402
import world_fingerprint as WF  # noqa: E402
try:
    # ★L5-37 多母带: 逐条 md5, 记母带集合 (单条时退化为原来的单 md5)。
    _masters = getattr(raw, "_masters", [PE.MASTER])
    _md5s = [hashlib.md5(open(m, "rb").read()).hexdigest() for m in _masters]
    _rmd5 = _md5s[0] if len(_md5s) == 1 else ",".join(_md5s)
except Exception:
    _rmd5 = None
WF.write(raw, os.path.join(log_dir, "world.json"), extra={
    "reference": {"path": PE.MASTER, "md5": _rmd5},
    "policy_io": {"obs_dim": PE.OBS_DIM, "act_dim": PE.ACT_DIM},
    "sensors": {"pad_force_threshold_N": PE.PAD_FTH,
                "pads_min_per_hand": PE.PADS_MIN},
    "method": {"variant": os.environ.get("POUR_VARIANT", "HYB").upper(),
               "beta_r": os.environ.get("POUR_BETA_R", "2.0"),
               "beta_l": os.environ.get("POUR_BETA_L", "1.0"),
               "name": args.name, "seed": args.seed},
})

agent = PourPPO(env, output_dir=log_dir,
                full_config=ConfigWrapper(agent_cfg, {}), raw_env=raw)
if args.load_path:
    agent.restore_train(args.load_path)
print(f"[train_pour] N={args.num_envs} obs={PE.OBS_DIM} act={PE.ACT_DIM} "
      f"logs={log_dir}", flush=True)
# ---- 代码钉子: 训练自拉录像循环 (POUR_*/CUDA 环境自动继承; 父进程死循环自退) ----
if not args.no_autorec:
    import subprocess
    _vdir = os.path.join(log_dir, "videos")
    os.makedirs(_vdir, exist_ok=True)
    subprocess.Popen(
        ["bash", os.path.join(_HERE, "autorecord_pour.sh"),
         os.path.join(log_dir, "progress_steps.txt"),
         os.path.join(log_dir, "stage1_nn"), _vdir, args.name,
         sys.executable, str(os.getpid())],
        stdout=open(os.path.join(log_dir, "autorec.out"), "a"),
        stderr=subprocess.STDOUT)
    print(f"[train_pour] 录像循环已自拉 (每1M步一支 -> {_vdir})", flush=True)
agent.train()
try:
    _slot.release()
except Exception:
    pass
app.close()
