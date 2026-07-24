"""M1 验收: obs 健全性 / 零动作回归 / reward 方向性 / 吞吐.

  .venv-isaac/bin/python -m rl_rebuild.correction.m1_check --num_envs 64 --headless
"""
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg  # noqa: E402

cfg = SharpaCorrectionEnvCfg()
cfg.scene.num_envs = args.num_envs
env = SharpaCorrectionEnv(cfg)
N, dev = env.num_envs, env.device


def rollout(policy_fn, steps):
    """跑 steps 步, 返回 (逐项 reward 累计快照, 是否有终止, obs 样本)."""
    any_term = False
    obs = None
    with torch.inference_mode():
        env.reset()
        for k, v in env._ep_sums.items():
            v.zero_()
        for i in range(steps):
            a = policy_fn(i)
            obs, _, term, _, _ = env.step(a)
            any_term |= bool(term.any())
        sums = {k: v.mean().item() for k, v in env._ep_sums.items()}
    return sums, any_term, obs


def toward_policy(sign):
    """settle 段零动作; active 后腕残差满幅指向(+1)/背离(-1)物体, 并闭/张指.
    偏移必须发生在计分窗口内 — 势函数差分只结算窗口内的距离变化,
    恒定偏移会平移首末两端而在求和中抵消 (这正是反年金设计的性质)."""
    def fn(i):
        a = torch.zeros(N, 28, device=dev)
        if i < cfg.settle_steps:
            return a
        origins = env.scene.env_origins
        palm = env._palm_pos() - origins
        obj = env.object.data.root_pos_w - origins
        d = (obj - palm)
        d = d / (d.norm(dim=1, keepdim=True) + 1e-9)
        a[:, 0:3] = sign * 1.0 * d       # 满幅 15cm 残差
        a[:, 6:28] = 0.5 * sign
        return a
    return fn


STEPS = cfg.settle_steps + 60            # 静置 + 60 个 active 步
zero_fn = lambda i: torch.zeros(N, 28, device=dev)

s_zero, term_zero, obs = rollout(zero_fn, STEPS)
s_toward, term_toward, _ = rollout(toward_policy(+1), STEPS)
s_away, term_away, _ = rollout(toward_policy(-1), STEPS)

# ---- 吞吐 (完整 episode, 零动作) ----
with torch.inference_mode():
    env.reset()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(env.ep_total):
        env.step(zero_fn(0))
    torch.cuda.synchronize()
fps = env.ep_total * N / (time.time() - t0)

pol, cri = obs["policy"], obs["priv_info"]
checks = [
    ("obs 维度 policy=144", pol.shape == (N, 144)),
    ("priv_info=7 / proprio_hist=(8,44)", cri.shape == (N, 7)
     and obs["proprio_hist"].shape == (N, cfg.prop_hist_len, 44)),
    ("obs 无 NaN/Inf", bool(torch.isfinite(pol).all() and torch.isfinite(cri).all())),
    ("obs 在截断界内", bool((pol.abs() <= cfg.clip_obs).all())),
    ("零动作: task=0", abs(s_zero["task"]) < 1e-6),
    ("零动作: contact=0", abs(s_zero["contact"]) < 1e-6),
    ("零动作: 无终止", not term_zero),
    ("方向性: 靠近 approach > 零动作", s_toward["approach"] > s_zero["approach"] + 0.3),
    ("方向性: 背离 approach < 零动作", s_away["approach"] < s_zero["approach"] - 0.3),
    ("方向性: 靠近产生接触", s_toward["contact"] > 0.0),
    # 吞吐断言只在 >=256 env 生效 (小 N 远离 GPU 饱和区, 数字无参考价值)
    ("吞吐 >= 5k env-steps/s (N>=256 时)", fps >= 5_000 or N < 256),
]
print(f"\n== M1 验收 ({N} env) ==")
ok = True
for name, passed in checks:
    ok &= bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
print(f"\napproach 累计: 靠近={s_toward['approach']:+.3f}  零动作={s_zero['approach']:+.3f}  "
      f"背离={s_away['approach']:+.3f}")
print(f"contact 累计:  靠近={s_toward['contact']:+.3f}   task 靠近={s_toward['task']:+.3f}")
print(f"吞吐: {fps:,.0f} env-steps/s")
print("==>", "M1 验收通过" if ok else "有 FAIL")
env.close()
app.close()
raise SystemExit(0 if ok else 1)
