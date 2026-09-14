"""Clean/3 Stage-1 抓稳段 训练入口 (PPO; 课程 = release_row 按成功率 EMA 单向退火).

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Clean/3/C_Wiring/train_clean.py --headless --num_envs 512 --name Clean3_hold_s42
物理规矩由本脚本按 task_config.PHYS 显式写环境变量 (可被外部环境变量覆写); 发车前 grep 日志 "难度覆写".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--name", default="Clean3_hold_s42")
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--max_agent_steps", type=int, default=None)
p.add_argument("--load_path", default=None)
p.add_argument("--release_row", type=int, default=None, help="课程起点覆写")
p.add_argument("--anneal_ema", type=float, default=0.7, help="成功率 EMA 达此值且持续 anneal_sustain 窗 → release_row -RELEASE_STEP")
p.add_argument("--anneal_sustain", type=int, default=10)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.headless, "训练必须 --headless (CLAUDE.md 铁则)"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
for k, v in TC.PHYS.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot(f"clean3_train_{args.name}")
app = AppLauncher(args).app
import torch  # noqa: E402
import yaml  # noqa: E402
import clean_env as CE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = TC.REPO


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class CleanPPO(PPO):
    def __init__(self, *a, raw_env=None, **kw):
        super().__init__(*a, **kw)
        self._raw = raw_env; self._ema = 0.0; self._sustain = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        rates = self._raw.pop_rates()
        for k, v in rates.items():
            if v == v:
                self.writer.add_scalar(k, v, self.agent_steps)
        with open(os.path.join(self.output_dir, "progress_steps.txt"), "w") as f:
            f.write(str(int(self.agent_steps)))
        sr = rates.get("sr/success", float("nan"))
        if sr == sr:
            self._ema = 0.95 * self._ema + 0.05 * sr
        self.writer.add_scalar("curr/ema_success", self._ema, self.agent_steps)
        self._sustain = self._sustain + 1 if self._ema >= args.anneal_ema else 0
        if self._sustain >= args.anneal_sustain and self._raw.release_row_cur > TC.RELEASE_MIN:
            self._raw.release_row_cur = max(TC.RELEASE_MIN, self._raw.release_row_cur - TC.RELEASE_STEP)
            self._sustain = 0; self._ema = 0.0        # 退火后重新累积
            print(f"[课程] release_row -> {self._raw.release_row_cur} @ {self.agent_steps/1e6:.2f}M", flush=True)
        self.writer.add_scalar("curr/release_row_cur", float(self._raw.release_row_cur), self.agent_steps)
        _slot.yield_if_paused()


cfg = CE.build_cfg(args.num_envs); cfg.seed = args.seed
raw = CE.CleanHoldEnv(cfg)
if args.release_row is not None:
    raw.release_row_cur = int(args.release_row)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_clean.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["seed"] = args.seed
acfg["algorithm"]["experiment_name"] = args.name
acfg["algorithm"]["num_actors"] = args.num_envs
acfg["algorithm"]["minibatch_size"] = min(acfg["algorithm"]["minibatch_size"], 32 * args.num_envs)
if args.max_agent_steps:
    acfg["algorithm"]["max_agent_steps"] = args.max_agent_steps
log_dir = os.path.join(ROOT, "logs", args.name); os.makedirs(log_dir, exist_ok=True)
world = {
    "schema": 1, "task": "Clean3_hold_stage1", "seed": args.seed, "clip": TC.CLIP,
    "reference": {"path": TC.REFERENCE, "sha256": _sha256(TC.REFERENCE)},
    "priors": {k: {"path": v, "sha256": _sha256(v)} for k, v in (("plate", TC.PRIOR_PLATE), ("sponge", TC.PRIOR_SPONGE))},
    "physics": {k: os.environ.get(k) for k in TC.PHYS}, "aux_mass_kg": 0.05,
    "policy_io": {"obs_dim": CE.OBS_DIM, "priv_dim": CE.PRIV_DIM, "act_dim": CE.ACT_DIM},
    "stage1": {k: getattr(TC, k) for k in ("K_CLOSE", "RELEASE_MAX", "RELEASE_MIN", "RELEASE_STEP", "RELEASE_JITTER", "HOLD_ROWS", "T_EP",
                                            "CERT_POS", "CERT_ROT_DEG", "CERT_STEPS", "DROP_POS", "DROP_ROT_DEG", "PAD_FTH",
                                            "PLATE_SUPPORT_MIN", "SPONGE_PADS_MIN", "ARM_STEP", "ARM_DEV", "FIN_STEP", "FIN_DEV",
                                            "W_CONTACT", "W_HOLD_POS", "W_HOLD_ROT", "W_WITHIN", "B_CERT", "B_SUCCESS", "B_DROP", "W_ACT")},
    "arm_sag_rad": raw.arm_sag.cpu().numpy().round(5).tolist(),
    "release_row_start": int(raw.release_row_cur),
}
with open(os.path.join(log_dir, "world.json"), "w") as f:
    json.dump(world, f, indent=2)
agent = CleanPPO(env, output_dir=log_dir, full_config=ConfigWrapper(acfg, {}), raw_env=raw)
if args.load_path:
    agent.restore_train(args.load_path)
print(f"[train_clean] N={args.num_envs} obs={CE.OBS_DIM} act={CE.ACT_DIM} logs={log_dir} release={raw.release_row_cur}", flush=True)
agent.train()
try: _slot.release()
except Exception: pass
sys.stdout.flush()
os._exit(0)
