"""Sweep408 Stage-1 抓稳段 训练入口 (PPO; 课程 = release_row 按成功率 EMA 单向退火)。

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Sweep/408/C_Wiring/train_grip.py --headless --num_envs 512 --name Sweep408_grip_s42

物理规矩由本脚本按 task_config.PHYS 显式写环境变量 (可被外部覆写); 发车前 grep 日志 "难度覆写"。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--name", default="Sweep408_grip_s42")
p.add_argument("--arm", default=None, choices=("L0","L1","L2","L3"), help="四条阶梯选臂 (= SWEEP408_ARM)")
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--release_row", type=int, default=None, help="课程起点覆写")
p.add_argument("--max_agent_steps", type=int, default=0)
p.add_argument("--load_path", default="")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.headless, "训练必须 --headless (CLAUDE.md 铁则)"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if args.arm:
    os.environ["SWEEP408_ARM"] = args.arm      # 必须在 import task_config 之前
import task_config as TC  # noqa: E402
for k, v in TC.PHYS.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot(f"sweep408_train_{args.name}")
app = AppLauncher(args).app
import yaml  # noqa: E402
import grip_env as GE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = TC.REPO


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class GripPPO(PPO):
    def __init__(self, *a, raw_env=None, **kw):
        super().__init__(*a, **kw)
        self._raw = raw_env; self._ema = 0.0
        self._up = self._down = 0; self._floor = TC.RELEASE_MAX

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        rates = self._raw.pop_rates()
        for k, v in rates.items():
            if v == v:
                self.writer.add_scalar(k, v, self.agent_steps)
        with open(os.path.join(self.output_dir, "progress_steps.txt"), "w") as f:
            f.write(str(int(self.agent_steps)))
        # 课程按**认证率**退火, 不用 success —— success 要走完 80% 母带, 早期恒 0, 课程会永远卡住。
        sr = rates.get("sr/cert", float("nan"))
        if sr == sr:
            self._ema = 0.95 * self._ema + 0.05 * sr
        self.writer.add_scalar("curr/ema_cert", self._ema, self.agent_steps)
        # ★v2 **双向**课程 (v1 死在单向: 退到 release=20 认证塌了回不去, 白烧 22.5M 步 = 75% 预算)
        cur = self._raw.release_row_cur
        if self._ema >= TC.ANNEAL_DOWN_EMA:
            self._down += 1; self._up = 0
        elif self._ema <= TC.ANNEAL_UP_EMA:
            self._up += 1; self._down = 0
        else:
            self._up = self._down = 0
        if self._down >= TC.ANNEAL_SUSTAIN and cur > TC.RELEASE_MIN:
            self._raw.release_row_cur = max(TC.RELEASE_MIN, cur - TC.RELEASE_STEP)
            self._down = 0; self._ema = 0.0
            print(f"[课程] ↓ release_row {cur} -> {self._raw.release_row_cur} @ {self.agent_steps/1e6:.2f}M", flush=True)
        elif self._up >= TC.ANNEAL_SUSTAIN and cur < TC.RELEASE_MAX:
            self._raw.release_row_cur = min(TC.RELEASE_MAX, cur + TC.RELEASE_STEP)
            self._up = 0; self._ema = 0.0
            self._floor = min(self._floor, cur)      # 记下"触底回退过"的最低档
            print(f"[课程] ↑ 回退 release_row {cur} -> {self._raw.release_row_cur} "
                  f"(cert EMA {self._ema:.2f} 持续偏低) @ {self.agent_steps/1e6:.2f}M", flush=True)
        self.writer.add_scalar("curr/release_floor", float(self._floor), self.agent_steps)
        self.writer.add_scalar("curr/release_row_cur", float(self._raw.release_row_cur), self.agent_steps)
        _slot.yield_if_paused()


cfg = GE.build_cfg(args.num_envs); cfg.seed = args.seed
raw = GE.Sweep408GripEnv(cfg)
if args.release_row is not None:
    raw.release_row_cur = int(args.release_row)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_grip.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["seed"] = args.seed
acfg["algorithm"]["experiment_name"] = args.name
acfg["algorithm"]["num_actors"] = args.num_envs
acfg["algorithm"]["minibatch_size"] = min(acfg["algorithm"]["minibatch_size"], 32 * args.num_envs)
if args.max_agent_steps:
    acfg["algorithm"]["max_agent_steps"] = args.max_agent_steps
log_dir = os.path.join(ROOT, "logs", args.name); os.makedirs(log_dir, exist_ok=True)
world = {
    "schema": 2, "task": "Sweep408_grip_stage1_v2", "arm": TC.ARM, "seed": args.seed, "clip": TC.CLIP,
    "arm_flags": {"soft": TC.USE_SOFT, "table": TC.USE_TABLE, "contact": TC.USE_CONTACT},
    "reference": {"path": TC.REFERENCE, "sha256": _sha256(TC.REFERENCE)},
    "priors": {k: {"path": v, "sha256": _sha256(v)} for k, v in
               (("broom", TC.PRIOR_BROOM), ("pan", TC.PRIOR_PAN))},
    "physics": {k: os.environ.get(k) for k in TC.PHYS}, "pan_mass_kg": TC.PAN_MASS_KG,
    "no_table_contact": bool(TC.NO_TABLE_CONTACT),
    "policy_io": {"obs_dim": GE.OBS_DIM, "priv_dim": GE.PRIV_DIM, "act_dim": GE.ACT_DIM},
    "stage1": {k: getattr(TC, k) for k in
               ("K_CLOSE", "RELEASE_MAX", "RELEASE_MIN", "RELEASE_STEP", "RELEASE_JITTER", "CERT_BUDGET",
                "CLOCK_SLACK", "CERT_POS", "CERT_ROT_DEG", "CERT_STEPS", "DIE_POS", "DIE_ROT_DEG",
                "TABLE_MARGIN", "OBJ_DROP_Z", "PAD_FTH", "BROOM_SUPPORT_MIN", "PAN_PADS_MIN",
                "ARM_STEP", "ARM_DEV", "FIN_STEP", "FIN_DEV", "R_ADV", "B_CERT", "B_SUCCESS", "B_DIE",
                "W_ACT", "SOFT_W", "SOFT_SPAN_POS", "SOFT_SPAN_ROT_DEG", "W_TABLE", "TABLE_SIGMA",
                "W_CONTACT", "SUCCESS_CLOCK_FRAC", "ANNEAL_DOWN_EMA", "ANNEAL_UP_EMA", "ANNEAL_SUSTAIN")},
    "tape_rows": int(raw.T_REF), "T_EP": int(raw.T_EP),
    "arm_sag_rad": raw.arm_sag.cpu().numpy().round(5).tolist(),
    "release_row_start": int(raw.release_row_cur),
}
with open(os.path.join(log_dir, "world.json"), "w") as f:
    json.dump(world, f, indent=2)
agent = GripPPO(env, output_dir=log_dir, full_config=ConfigWrapper(acfg, {}), raw_env=raw)
if args.load_path:
    agent.restore_train(args.load_path)
print(f"[train_grip] N={args.num_envs} obs={GE.OBS_DIM} act={GE.ACT_DIM} logs={log_dir} "
      f"release={raw.release_row_cur} 母带 {raw.T_REF} 行", flush=True)
agent.train()
try: _slot.release()
except Exception: pass
sys.stdout.flush()
os._exit(0)
