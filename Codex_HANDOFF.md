# Sweep 项目交接入口

更新：2026-09-06。本文只负责新窗口阅读顺序、资料位置及执行边界；算法在Codex_tasks.md，Task3当前状态在Codex_new_data.md。

## 1. 接入位置与首次检查

- SSH host：msc-a6000；用户msc-auto；hostname mscauto-Lambda-Vector。
- 唯一当前项目根目录：/home/msc-auto/RL_sweep；分支task_sweep。
- 本地Codex项目入口不是远端源码镜像，不要误在本地创建或修改另一份源码。
- 先只读核对whoami、hostname、pwd、git branch、git status、任务进程和GPU归属；保留已有dirty改动。
- 当前整理只涉及文档与明确授权的旧录像清理，没有训练/resume。历史tmux名称不是自动恢复指令。
- 其他任务的机器路径和环境（例如CLAUDE.md中的历史PreGrasp机器）不能覆盖本项目路径。

## 2. 新窗口按此顺序阅读

1. /home/msc-auto/RL_sweep/Codex_HANDOFF.md（本文）：入口和边界。
2. /home/msc-auto/RL_sweep/Codex_tasks.md：成功单轨迹算法、从输入到reference/物理/PPO/录像的完整方法、运行步骤和文档职责表。原通用手册已合并，不再另读旧文件。
3. /home/msc-auto/RL_sweep/Codex_mistakes.md：特别是坐标COM、可见资产修复、保留重建运动、刷毛方向全程检查及真实terminal录像。
4. /home/msc-auto/RL_sweep/Codex_commit.md：工程提交、范围与回退定位。
5. /home/msc-auto/RL_sweep/Codex_new_data.md：Task3全部当前状态、32/80成功步骤、128/180失败及下一步。
6. 按任务选读：Codex_ablation.md（消融）、Codex_random.md（方块位置实验）、docs/TRAINING_DESIGN_GUIDE.md（历史设计来源）。不要用来源文档覆盖当前Sweep算法。

不要把历史REVIEW或旧交接中的“待确认/缺canonical”当成当前事实。Task3目前32 v3、80 v6均已获用户验收；最新索引在Codex_new_data.md第1节。

## 3. 必须实际观看的视觉证据

所有路径以 /home/msc-auto/RL_sweep 为根：
- 成功Sweep2 ego：datasets/sweep_2_better/44ce97212ec9c5c97ad567934f473707.mp4
- 原始GraspPose render：datasets/sweep_2_better/sweep2_grasppose/demo_replay/
- 成功Sweep2 zero-residual：outputs_video/sweep2_v1_zero_replay.mp4
- 完整方法24M：outputs_video/Sweep2AblationNoOfflineFull__20260903_policy_0024M/policy.mp4
- 已验收take32：outputs_video/Task3_take32_v3/full_zero_retry1/trajectory.mp4；同目录closeup.mp4
- 已验收take80：outputs_video/Task3_take80_v6/full_zero/trajectory.mp4
- 128/180最新进度目录见Codex_new_data.md，不得把ego或静态片段当成完整物理通过。

阅读和看视频都不可省略。抓姿需ego → 原始候选render → 转换后prior → 物理运动逐层对拍；IK数值通过不能替代视觉。录像沿用成功Sweep2相机方案。

## 4. 当前执行边界

Task3具体状态只维护在Codex_new_data.md。当前阶段仍不启动训练、不resume；新轨迹cube仍隐藏/无碰撞/无终止影响。后续训练接入、cube标定和资源预算需遵守用户届时的明确要求，不因轨迹视频验收就自动启动。

不改变成功单轨迹算法；资产修复在Task3独立副本，原始输入和成功Sweep2资产保留。只能共享注册完整双工具运动，不能独立旋转/冻结工具以掩盖失败。困难轨迹允许保存证据后暂跳过。

禁止影响其他用户进程、GPU reset、广域终止或破坏性Git命令。tasks/pregrasp/arm_shell_points.npz的既有修改不得覆盖、stage或提交。文档编辑/提交本次已获授权；之后按用户具体范围处理。没有push授权。

## 5. Git与证据定位

GitHub身份固定HARROLDX，origin为git@github-harroldx:stzabl-png/RL-Correction.git，禁止使用裸github.com默认身份。不得读取、复制或输出私钥。

Codex_commit.md解释每个里程碑的变更与验证；git log显示实际commit。回退先查看目标差异并按文件恢复或在独立工作树复现，不对dirty共享工作树做reset --hard。

本次文档整理前原稿/working-tree.patch：logs/task3_docs_cleanup_20260906/pre_edit/。录像清理清单：logs/task3_docs_cleanup_20260906/cleanup_manifest.json；旧视频旁数值与图片证据：同目录old_video_evidence/。被明确删除的旧mp4不在Git中，不能声称提交可恢复它们。

完整文档分工见Codex_tasks.md末节。
