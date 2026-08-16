"""GraspTaskEnv v2 冒烟: 三段脚本动作, 验证管线 + 合拢参考/阶段机/微抬升验证.

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.smoke --headless

段1  a_c=-1 30 步   -> 合拢参考被恰好抵消, c 不涨, 无接触 (验证"停住"语义)
段2  零动作 80 步   -> 参考斜坡自动合拢; **期望走完整链**: 接触 -> 候选 -> 微抬升验证
                      (能不能真过验证要看抓的质量, 走到 verify 段管线就算通)
段3  随机小动作 40 步 -> 数值稳定性 (obs 无 NaN)

⚠ 验证的是**管线与信号**, 不是策略质量.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--grasp_prior", default="", help="prior npz 路径 (B 组)")
p.add_argument("--prior_yaw", type=float, default=-1.0, help="钉死的物体 yaw (度)")
p.add_argument("--approach", action="store_true", help="开接近段")
p.add_argument("--ref_look", type=int, default=None, help="参考前瞻帧数")
p.add_argument("--no_eps_curr", action="store_true")
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--hand_side", choices=["right", "left"], default="right",
               help="交互手 (左手调试用; 抓姿/轨迹按右手生成的 clip 会有几何镜像误差, 扫雷为主)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("smoke")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, Phase, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
cfg.hand_side = args.hand_side
clips.configure_cfg(cfg, args.clip)
if args.ref_look is not None:
    cfg.ref_look_frames = args.ref_look
if args.no_eps_curr:
    cfg.eps_pos0, cfg.eps_rot0 = cfg.eps_pos, cfg.eps_rot
if args.grasp_prior:
    apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw,
                      approach=args.approach)
cfg.scene.num_envs = args.num_envs
cfg.closure_init_max = 0.0          # 冒烟从 c=0 起, 看参考斜坡自己走
env = GraspTaskEnv(cfg)
env.gentle = 1.0                    # 冒烟按全额惩罚看信号量级 (训练里由课程调)
obs, _ = env.reset()
print(f"\n[smoke] obs policy {tuple(obs['policy'].shape)}", flush=True)

reached = set()


def run(tag, steps, act_fn):
    nan = 0
    for i in range(steps):
        obs, rew, term, trunc, _ = env.step(act_fn(i))
        nan += int(torch.isnan(obs["policy"]).any())
        s = env._sig
        reached.update(int(v) for v in env.task_phase.tolist())
        if bool(s["newly_cand"].any()):
            print(f"    ✊ 候选抓取形成 -> 进入微抬升验证 (step {i})", flush=True)
        if bool(s["newly_success"].any()):
            print(f"    🏁 通过微抬升验证 (step {i})", flush=True)
        if bool(s["verify_fail"].any()):
            print(f"    ✗ 验证失败: 物体没跟上 (step {i})", flush=True)
        if i % 10 == 0 or i == steps - 1:
            print(f"[{tag} {i:3d}] phase={env.task_phase.tolist()} "
                  f"c={env.closure[0]:.2f} pads={int(s['n_pads'][0])} "
                  f"F={[round(float(v),2) for v in s['mag'][0]]}N "
                  f"cent={s['cent'][0]:+.2f} Q={s['quality'][0]:+.2f} "
                  f"rise={s['rise'][0]*1000:+.1f}mm "
                  + (f"screw={np.degrees(float(env.screw_angle[0])):+.1f}° "
                     f"eng={int(env.screw_engaged[0])} "
                     if getattr(env, 'screw_spec', None) is not None else "")
                  + f"table={s['table_pen'][0]*1e4:.1f} cross={s['cross_pen'][0]*100:.2f} "
                  f"rew={rew[0]:+.3f} term={int(term.sum())} trunc={int(trunc.sum())}"
                  + (f" | φ={env._phi()[0]:.2f} d_pos={s['d_pos'][0]*100:5.2f}cm "
                     f"d_rot={float(np.degrees(s['d_rot'][0].item())):5.1f}° "
                     f"res={env.res_step_cm[0]:.3f}cm" if env.cfg.approach else ""),
                  flush=True)
    assert nan == 0, f"{tag}: 观测出现 NaN"


N, A = args.num_envs, cfg.action_space
zero = torch.zeros(N, A, device=env.device)
pause = zero.clone(); pause[:, 7] = -1.0                   # a_c=-1: 恰好抵消参考斜坡
run("停住", 30, lambda i: pause)
run("零动作(参考合拢)", 80, lambda i: zero)
g = torch.Generator(device="cpu").manual_seed(0)
run("随机小动作", 40,
    lambda i: (torch.rand(N, A, generator=g) * 2 - 1).to(env.device) * 0.3)

names = [Phase.NAMES[v] for v in sorted(reached)]
print(f"\n[smoke] ✅ 管线通过: 150 步无 NaN | 到过的阶段: {names}", flush=True)
env.close()
app.close()
