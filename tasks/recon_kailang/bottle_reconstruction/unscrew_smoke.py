"""扭盖任务冒烟: 脚本动作验证管线与物理可行性 (对照 tasks/pregrasp/smoke.py 的分段式).

  OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=. $PY -u \
      -m tasks.recon_kailang.bottle_reconstruction.unscrew_smoke --headless

段1  零动作 30 步     -> 悬停在预抓姿 (腕距目标 <3cm), 瓶身不动, 帽不动
段2  合拢 (指残差 +1) 40 步 -> 指尖向帽收拢; 报告 tip-cap 距离与接触数 (台账 U2 的直接测量)
段3  合拢 + j7 滚转 60 步   -> 若已接触, screw_angle 应有非零变化 (台账 U3)
段4  随机小动作 30 步  -> 数值稳定 (obs 无 NaN)

⚠ 段2/段3 是**测量**不是判据 —— 开环脚本凑不出协调抓握是预期内的 (DESIGN_LOOP §2.1);
  它们量出"几何上差多远", 供台账 U1/U2 判读. PASS 只要求: 管线通、无 NaN、段1 稳.
"""
from __future__ import annotations

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="screw_unscrew_cap1_task")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_smoke")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.unscrew_env import (  # noqa: E402
    UnscrewTaskCfg,
    UnscrewTaskEnv,
)


def main() -> int:
    cfg = UnscrewTaskCfg()
    clips.configure_cfg(cfg, args.clip)
    cfg.scene.num_envs = args.num_envs
    env = UnscrewTaskEnv(cfg)
    failures: list[str] = []
    stats: dict = {"clip": args.clip, "num_envs": args.num_envs}
    try:
        obs, _ = env.reset()
        print(f"[smoke] obs policy {tuple(obs['policy'].shape)}")
        N = env.num_envs
        dev = env.device
        nan_steps = 0
        origins = env.scene.env_origins

        def snap(tag):
            cap = env.cap.data.root_pos_w - origins
            tips = env.tip_pos_w - origins[:, None]
            d = (tips - cap[:, None]).norm(dim=-1)
            capc = env._cap_contacts().sum(dim=1)
            body = (env.object.data.root_pos_w - origins - env.obj_init_pos
                    )[:, :2].norm(dim=1)
            wrist_err = (env.wrist_pos_w - origins - env.ref_wrist_pos[0]
                         ).norm(dim=1)
            ang = torch.rad2deg(env.screw_angle)
            per_tip = [round(float(v) * 100, 1) for v in d.mean(dim=0)]
            print(f"[{tag}] tip-cap {d.min(-1).values.mean()*100:.1f}cm (逐指 {per_tip}) "
                  f"接触数 {capc.float().mean():.1f} | 腕距参考 {wrist_err.mean()*100:.1f}cm "
                  f"| 瓶位移 {body.mean()*100:.2f}cm | 拧角 {ang.mean():.1f}°/max {ang.max():.1f}°")
            return dict(tip_cap_cm=float(d.min(-1).values.mean() * 100),
                        n_contact=float(capc.float().mean()),
                        wrist_err_cm=float(wrist_err.mean() * 100),
                        body_cm=float(body.mean() * 100),
                        screw_deg_mean=float(ang.mean()),
                        screw_deg_max=float(ang.max()))

        def run(tag, action_fn, steps):
            nonlocal nan_steps
            terms = 0
            for i in range(steps):
                a = action_fn(i)
                obs, _, term, trunc, _ = env.step(a)
                nan_steps += int(any(torch.isnan(v).any().item() for v in obs.values()))
                terms += int(term.any().item())
            s = snap(tag)
            s["terminated_steps"] = terms
            return s

        zero = torch.zeros(N, cfg.action_space, device=dev)
        stats["seg1_hold"] = run("段1 零动作", lambda i: zero, 30)
        if stats["seg1_hold"]["wrist_err_cm"] > 3.0:
            failures.append(f"段1 腕未稳住在预抓姿 ({stats['seg1_hold']['wrist_err_cm']:.1f}cm)")
        if stats["seg1_hold"]["body_cm"] > 1.0:
            failures.append("段1 零动作下瓶身被扰动")
        # ⚠ snap 读的是 reset **后**的状态, 终止循环会被掩盖 —— 必须直接查终止计数
        # (第一版就是漏了这条: 指尖插进瓶颈引发 knock-reset 循环, 其余指标全绿).
        if stats["seg1_hold"]["terminated_steps"] > 0:
            failures.append(f"段1 出现 {stats['seg1_hold']['terminated_steps']} 步终止 (零动作不应终止)")

        close = zero.clone()
        close[:, 7:29] = 1.0                      # 全指合拢
        stats["seg2_close"] = run("段2 合拢", lambda i: close, 40)

        twist = close.clone()
        twist[:, 6] = 1.0                         # j7 滚转 + 保持合拢
        stats["seg3_twist"] = run("段3 合拢+滚转", lambda i: twist, 60)

        gen = torch.Generator(device="cpu").manual_seed(0)
        stats["seg4_random"] = run(
            "段4 随机小动作",
            lambda i: torch.randn(N, cfg.action_space, generator=gen).to(dev) * 0.3, 30)

        stats["nan_steps"] = nan_steps
        if nan_steps:
            failures.append(f"{nan_steps} 步 obs 出现 NaN")
        stats["failures"] = failures
        stats["passed"] = not failures
        payload = json.dumps(stats, indent=2, ensure_ascii=False)
        print("\n[unscrew-smoke-report]\n" + payload, flush=True)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                f.write(payload + "\n")
        if failures:
            print("[unscrew-smoke] FAIL: " + "; ".join(failures), file=sys.stderr)
            env.close()
            import os
            os._exit(1)
        print("[unscrew-smoke] PASS", flush=True)
    finally:
        env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
