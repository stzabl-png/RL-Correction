# Codex 任务台账：Sweep2 Entry-Success 残差训练

更新时间：2026-08-31。本文件只描述当前有效方案；Deep20 试验已停止，原因与历史保留在 Codex_commit.md 和 Codex_mistakes.md。

## 当前目标

把 sweep_2_better 的重建工具轨迹转换为 DexMate 双臂 reference，并训练 14 DoF joint residual policy，使固定 cube 第一次有效进入 dustpan。成功必须来自 Isaac 物理状态，不能由 reference row、奖励值或视觉观感代替。

成功契约：

- Gate1 ready。
- Gate2 broom near 且 cube moved >= 5 mm。
- Gate3 entered：cube 中心跨过 mouth，并满足横向 footprint、承载高度和盆内深度下界。
- Gate4 与 Gate3 同步，成功当步立即终止。
- fully_inside/deep_inside 只做诊断。
- 最终 512 deterministic episodes，success rate >= 0.50。

## 训练数据流

datasets/sweep_2_better
-> build_reference.py 与 GraspPose/IK
-> sweep2_reference_v1.npz
-> 固定 cube、平滑开放 dustpan collider、双 FixedJoint
-> 25/40 mm right-arm entry experts + canonical failure
-> logs/expert/transitions/
-> 40 mm Actor BC
-> 25/40/failure Critic return regression
-> 1024-env pure on-policy PPO
-> 每 3M 完整诊断包
-> 512 deterministic evaluation

关键约束：

- cube 固定在 [-0.0259767957, -0.1788897067, 0.8830000162] m。
- 前 80 control steps 严格 reference-only；这些样本训练 Critic但不训练 Actor。
- confidence 控制 residual step/deviation bound。
- human pose 只形成中低 confidence 区域的运动方向 shape prior。
- 手指固定为 GraspPose。
- 当前训练只读取 logs/expert/transitions/，禁止读取 transitions_deep20。
- 不恢复旧 15M/48M policy；当前 run 从头初始化。

## 当前运行

tmux：sweep2_entryrestore_v2_1024_20260831

run：
/home/msc-auto/RL_sweep/logs/Sweep2_entryrestore_v2_fixed1024_seed42_20260831

artifact prefix：
Sweep2EntryRestoreV2__20260831_policy

配置：

- num_envs=1024
- seed=42
- max_agent_steps=100000000
- GPU0，共享使用已获用户授权
- 3M agent steps 一个 checkpoint/metrics/rollout/video 节点
- 自动录像前训练在 epoch 边界 yield，录像结束后恢复

启动脚本：
logs/Sweep2_entryrestore_v2_fixed1024_seed42_20260831/launch_pipeline.sh

日志：

- pipeline.log
- smoke_1env.log
- train.log
- autorecord.log
- progress_steps.txt（PPO 开始后出现）
- world.json（环境与 warmup 初始化后出现）
- bc_summary.txt（warmup 后出现）

截至最近检查：

- 1-env random smoke PASS。
- 1024-env 场景创建完成。
- 仿真正在启动。
- 尚未产生 PPO agent steps。
- 不得重复启动同名或第二个 1024-env run。

## 诊断产物

每 3M 节点：

logs/checkpoints/Sweep2EntryRestoreV2__20260831_policy_<XXXXM>/
- checkpoint.pth
- metrics.json
- rollout.npz
- record.log

outputs_video/Sweep2EntryRestoreV2__20260831_policy_<XXXXM>/
- policy.mp4
- topdown_frames/frame_*.png

录像规则：

- 成功物理 rollout 当步终止。
- DirectRLEnv terminal 后会自动 reset，因此 recorder 不得渲染 done 返回后的状态。
- 冻结源是最后一个有效 pre-reset RGB 图像。
- 以 20 FPS 重复 40 帧，形成 2 秒展示冻结。
- 15M 回归的正确冻结源为 frame_0231.png，不是 reset 后的 frame_0232.png。

## 已验证基线

旧 15M checkpoint 回归：

- terminal transition step=232。
- trace success=True，Gate=[1,1,1,1]。
- cube_pan.z≈63.073 mm。
- topdown 最后一帧 frame_0231.png。
- MP4 总计 272 帧：232 个有效画面 + 40 个冻结帧。
- 路径：logs/entry_restore_regression_20260831/ 和 outputs_video/Sweep2_entry_restore_regression_20260831/。

旧保留节点：

- logs/checkpoints/Sweep2__20260830_policy_{0003M,0012M,0015M,0048M}/
- outputs_video/Sweep2__20260830_policy_{0003M,0012M,0015M,0048M}/

这些是历史基线，不能冒充当前 v2 run 的结果。

## 安全与运维

- 用户允许与 feiyang 共享 GPU，但禁止操作对方任何 PID、tmux 或资源优先级。
- 只能精确停止本任务 tmux/PID，禁止 pkill、killall、GPU reset。
- 训练和本任务 recorder 不重叠。
- 不删除、覆盖或混写历史 checkpoint/run。
- 训练达到较高窗口成功率也不会自动代表 512 回合验收通过。

监控命令：

ssh msc-a6000 'tmux capture-pane -pt sweep2_entryrestore_v2_1024_20260831:0 -S -80'
ssh msc-a6000 'tail -n 100 /home/msc-auto/RL_sweep/logs/Sweep2_entryrestore_v2_fixed1024_seed42_20260831/train.log'
ssh msc-a6000 'cat /home/msc-auto/RL_sweep/logs/Sweep2_entryrestore_v2_fixed1024_seed42_20260831/progress_steps.txt 2>/dev/null || true'

停止时必须先核对精确 session 与 PID，然后只向 sweep2_entryrestore_v2_1024_20260831 发送 Ctrl-C。

## 下一步

1. 等待 BC/Critic warmup 和 PPO 开始。
2. 3M 检查 Gate3/Gate4 必须同步。
3. 检查失败 episode 是否仍集中在 step104 mouth collision。
4. 检查 deterministic 视频冻结源及 trace terminal state。
5. 选择候选 checkpoint 做 512 回合 deterministic 验收。
