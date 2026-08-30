"""v2 (轨迹跟随) 冒烟: 零动作回放, 验证 U8/U9 与门控时钟.

  OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=. $PY -u \
      -m tasks.recon_kailang.bottle_reconstruction.unscrew_ref_smoke --headless

零动作 = 双臂跟人手参考 (右臂含手指合拢参考). 检查:
  - IK 质量 (构建期打印, U8)
  - 瓶身回放: 倾角走到 ~106° 再回正, 无爆炸
  - 右腕跟踪误差 (跟不上 => term_arm_err 会杀回合, 终止计数暴露)
  - 时钟在 hold 帧冻结 (零动作拧不满, 不应释放)
  - 无 NaN
"""
from __future__ import annotations

import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="screw_unscrew_cap1_task")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--report", default="")
parser.add_argument("--dyn", action="store_true", help="Stage B: 动态瓶")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_ref_smoke")
app = AppLauncher(args).app

import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.unscrew_ref_env import (  # noqa: E402
    UnscrewDynTaskCfg,
    UnscrewRefTaskCfg,
    UnscrewRefTaskEnv,
)


def main() -> int:
    cfg = UnscrewDynTaskCfg() if args.dyn else UnscrewRefTaskCfg()
    clips.configure_cfg(cfg, args.clip)
    cfg.scene.num_envs = args.num_envs
    env = UnscrewRefTaskEnv(cfg)
    failures: list[str] = []
    stats: dict = {"clip": args.clip, "steps": args.steps}
    try:
        obs, _ = env.reset()
        print(f"[smoke] obs policy {tuple(obs['policy'].shape)}")
        N = env.num_envs
        origins = env.scene.env_origins
        zero = torch.zeros(N, cfg.action_space, device=env.device)
        nan_steps = term_steps = 0
        max_tilt = body_v_max = cap_w_max = 0.0
        wrist_errs = []
        for i in range(args.steps):
            obs, _, term, trunc, _ = env.step(zero)
            nan_steps += int(any(torch.isnan(v).any().item() for v in obs.values()))
            term_steps += int(term.any().item())
            oq = env.object.data.root_quat_w
            tilt = torch.rad2deg(torch.arccos(
                (1 - 2 * (oq[:, 1] ** 2 + oq[:, 2] ** 2)).clamp(-1, 1))).max()
            max_tilt = max(max_tilt, float(tilt))
            body_v_max = max(body_v_max,
                             float(env.object.data.root_lin_vel_w.norm(dim=1).max()))
            cap_w_max = max(cap_w_max,
                            float(env.cap.data.root_ang_vel_w.norm(dim=1).max()))
            t = env.ref_clock
            werr = (env.wrist_pos_w - origins - env.ref_wrist_pos[t]).norm(dim=1)
            wrist_errs.append(float(werr.mean()))
            if i % 25 == 0:
                capc = int(env._cap_contacts()[0].sum())
                lgrip = int(env._left_bottle_contacts()[0].sum())
                print(f"[step{i:3d}] clock={int(t[0])} 瓶倾角 {float(tilt):5.1f}° 左握指数 {lgrip} "
                      f"右腕误差 {float(werr.mean())*100:5.1f}cm 帽接触 {capc} "
                      f"拧角 {float(torch.rad2deg(env.screw_angle[0])):5.1f}° "
                      f"released={int(env.released_latch[0])}")
        import numpy as np
        we = np.array(wrist_errs)
        stats.update(dict(
            nan_steps=nan_steps, term_steps=term_steps,
            max_body_tilt_deg=max_tilt,
            body_v_max=body_v_max, cap_w_max=cap_w_max,
            wrist_err_cm_median=float(np.median(we) * 100),
            wrist_err_cm_p95=float(np.percentile(we, 95) * 100),
            final_clock=int(env.ref_clock[0]),
            hold_frame=env.hold_frame,
            released=bool(env.released_latch.any()),
            screw_deg_max=float(torch.rad2deg(env.screw_angle).max()),
        ))
        if nan_steps:
            failures.append(f"{nan_steps} 步 NaN")
        if body_v_max > 1.0 and not args.dyn:
            failures.append(f"瓶身速度峰值 {body_v_max:.2f}m/s > 1.0 (抖动/冲量)")
        if term_steps > 0 and not args.dyn:
            failures.append(f"{term_steps} 步出现终止 (零动作跟参考不应死)")
        if stats["wrist_err_cm_p95"] > 8.0:
            failures.append(f"右腕跟踪 95 分位 {stats['wrist_err_cm_p95']:.1f}cm > 8cm (U8)")
        if max_tilt < 60.0:
            failures.append(f"瓶身未转平 (最大倾角 {max_tilt:.0f}°), 回放轨迹可疑")
        if stats["final_clock"] != env.hold_frame and not stats["released"]:
            failures.append(f"时钟未停在 hold 帧 ({stats['final_clock']} vs {env.hold_frame})")
        stats["failures"] = failures
        stats["passed"] = not failures
        payload = json.dumps(stats, indent=2, ensure_ascii=False)
        print("\n[unscrew-ref-smoke-report]\n" + payload, flush=True)
        if args.report:
            with open(args.report, "w", encoding="utf-8") as f:
                f.write(payload + "\n")
        if failures:
            print("[unscrew-ref-smoke] FAIL: " + "; ".join(failures), file=sys.stderr)
            env.close()
            import os
            os._exit(1)
        print("[unscrew-ref-smoke] PASS", flush=True)
    finally:
        env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
