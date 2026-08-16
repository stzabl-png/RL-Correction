"""退避式起点族的冒烟测试: 建族 / 准入门 / 起点分布 / 起点合法性。

    SHARPA_WANDB=0 PYTHONPATH=. RL_RETRACT_START=1 $PY -m tasks.pregrasp.smoke_retract \\
        --headless --clip Grasp3 --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz \\
        --prior_yaw 215
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp3")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("smoke_retract")
app = AppLauncher(args).app

import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
# ⚠ approach_only 必须在 apply_grasp_prior **之前** 设 —— 动作空间(7)和观测维度都靠它
import os as _os
if _os.environ.get("SMOKE_FULL") == "1":      # 完整任务(接近+抓握): 13 维, 不砍
    pass
else:
    cfg.approach_only = True
    cfg.action_space = 7
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.retract_start = True
cfg.scene.num_envs = args.num_envs
E = GraspTaskEnv(cfg)

fails = []


def check(tag, ratio, sp, expect):
    """按给定课程设置复位, 报起点分布 + 起点合法性 (外壳离桌 / 手->物体)。"""
    cfg.retract_ratio, cfg.stance_prob = ratio, sp
    E.reset()
    d0 = E.retract_d0.clone()          # ⚠ 先存: 下面 step 里若有回合结束会被重写
    n_st = int(torch.isnan(d0).sum())
    # ⚠ 必须**走一步 env.step**: `_sig`(含 arm_gap) 是在奖励计算里填的, reset 之后是空的。
    #   第一版我写成 `if "arm_gap" in E._sig` —— 它恒为假, 于是"外壳为正"这个判据
    #   **一次都没执行**, 却照样打印了"✅通过"。和 _fgate_dg 那次同一个病:
    #   **恒不触发的检查 = 假通过**。这里改成真跑一步再读, 拿的是物理稳态值,
    #   比 IK 指令位形的离线估计权威(实测两者能差 4cm)。
    zero = torch.zeros(E.num_envs, int(E.single_action_space.shape[0]), device=E.device)
    E.step(zero)
    assert "arm_gap" in E._sig, "step 之后 _sig 仍没有 arm_gap —— 判据接错了"
    gap = E._sig["arm_gap"] * 100.0
    pd = E._pad_dist_normal()[0].min(dim=1).values * 100.0
    dd = d0[~torch.isnan(d0)] * 100.0
    print(f"\n[{tag}] retract_ratio={ratio} stance_prob={sp}   ({expect})")
    print(f"  起点: 站姿 {n_st}/{E.num_envs} | 退避 d = "
          + (f"{dd.min():.1f}~{dd.max():.1f}cm (均值 {dd.mean():.1f})"
             if len(dd) else "(无)"))
    print(f"  起点合法性(物理稳态, 外壳口径): 臂外壳离桌 "
          f"{gap.min():.2f}~{gap.max():.2f}cm "
          f"{'⛔ 有负值!' if float(gap.min()) < 0 else '✅ 全正'}")
    if float(gap.min()) < 0:
        fails.append(f"{tag}: 臂外壳离桌最小 {gap.min():.2f}cm < 0")
    elif float(gap.min()) < cfg.arm_shell_margin * 100:
        fails.append(f"{tag}: 臂外壳离桌最小 {gap.min():.2f}cm < 余量 "
                     f"{cfg.arm_shell_margin*100:.1f}cm (未穿但已贴线)")
    print(f"  起点手->物体最近指垫 {pd.min():.2f}~{pd.max():.2f}cm")


# ---- 判据活性检查: 每条判据都必须**真的能触发**, 否则就是假通过 ----
# (今天两次栽在"恒不触发的检查照样打印通过": _fgate_dg 被覆盖成 None、
#  以及本文件第一版的 `if "arm_gap" in E._sig`。所以判据要单独验活性。)
print("\n[判据活性] 逐条确认 approach_only 的判据接上了:")
print(f"  approach_only={cfg.approach_only} retract_start={cfg.retract_start} "
      f"arm_table_shell={cfg.arm_table_shell}")
if cfg.approach_only:
    print(f"  **动作空间 {cfg.action_space} 维(只有臂)** | 观测 {cfg.observation_space} 维 "
          f"(去掉了 closure/每指残差/动作缓冲那 12 个空转通道)")
else:
    print(f"  **动作空间 {cfg.action_space} 维(臂+手)** | 观测 {cfg.observation_space} 维 "
          f"| 完整任务: 接近 + 抓握 + 微抬升验证")
print(f"  回合预算 {cfg.approach_only_steps} 步 -> episode_length_s="
      f"{cfg.episode_length_s:.2f}s (env 内 ep_total={E.ep_total})")
print(f"  到位判据 eps_pos={cfg.eps_pos*100:.1f}cm eps_rot="
      f"{__import__('numpy').degrees(cfg.eps_rot):.1f}° 保持 {cfg.switch_hold} 步")
print(f"  硬终止 手->物体 < {cfg.approach_hit_obj_m*100:.1f}cm | 臂外壳离桌 < 0")

check("A 最简单", 0.0, 0.0, "全部就在 GraspPose 上")
check("B 半程", 0.5, 0.0, "退避 d 落在前半段")
check("C 全退避", 1.0, 0.0, "退避 d 铺满 [0,D]")
check("D 正式口径", 1.0, 1.0, "全部从对称站姿起步")

print("\n" + "=" * 70)
if fails:
    print("⛔ 冒烟未通过:")
    for f in fails:
        print("   " + f)
else:
    print("✅ 冒烟通过: 四档课程的起点全部合法 (物理稳态外壳离桌 ≥ 余量)")
print("=" * 70 + "\n")
app.close()
