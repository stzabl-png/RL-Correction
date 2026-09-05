# Sweep2 移植记录 (task_sweep@a8994fe2 → unscrew17_pull, 2026-09-05)

- 拉入: tasks/Sweep/2 全套 + datasets/sweep_2_better + priors Sweep2_{broom,dustpan} + logs/{expert,checkpoints}(-f) + PLAYBOOK。
- clips.py: sweep 段换其自包含版 (旧版指私有重建路径)。
- **基类移植门 ×3** (`tasks/pregrasp/env.py`, 全部 `getattr(cfg,"fixed_attached_tools",False)` 守护, 旗不设=行为逐位不变):
  ① 焊接任务不设 retract_start; ② canon_rot 通用静置摆放跳过; ③ 保留任务给定重建首帧位姿 (否则 物体→GraspPose 变换被破坏, 实测钉死 yaw IK 27.8cm 假报"够不着")。
  ⚠ 他们分支同段 diff 还想删我们的 `bypass_lift_scaffold` 分支 —— **未采纳** (unscrew 线特性)。
- 冒烟: `smoke_sweep.py --headless` PASS (8env×64步, gates=[1,1,0,0], 附着复位审计亚毫米, GraspPose IK 0.07cm)。
  排障记: 首两跑死于 "Failed to create simulation view backend" = **显存挤兑** (当时本机 4 个用户任务占 14/16GB), 非代码问题。
- 训练入口: `SWEEP_ABLATION=full python tasks/Sweep/2/C_Wiring/train_sweep.py --headless` (SWEEP_REF_NPZ / SWEEP_CUBE_VARIANTS_NPZ 可覆写)。
