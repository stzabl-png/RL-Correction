# Codex 任务台账：Sweep2 Full-Inside 残差训练

更新时间：2026-09-01。本文件只描述当前有效方案；Deep20 试验已停止，原因与历史保留在 Codex_commit.md 和 Codex_mistakes.md。

## 当前目标

把 sweep_2_better 的重建工具轨迹转换为 DexMate 双臂 reference，并训练 14 DoF joint residual policy，使固定 cube 的完整 footprint 进入 dustpan。成功必须来自 Isaac 物理状态，不能由 reference row、奖励值或视觉观感代替。

成功契约：

- Gate1 ready。
- Gate2 broom near 且 cube moved >= 5 mm。
- Gate3 entered：cube 中心跨过 mouth，并满足横向 footprint、承载高度和盆内深度下界。
- Gate4 在 fully_inside 后成立，成功当步立即终止。
- fully_inside 是 Gate4/success 的硬判据；deep_inside/deep_margin 只做诊断。
- 最终 512 deterministic episodes，success rate >= 0.50。

## 训练数据流

datasets/sweep_2_better
-> build_reference.py 与 GraspPose/IK
-> sweep2_reference_v1.npz
-> 固定 cube、平滑开放 dustpan collider、双 FixedJoint
-> 两条 fully-inside expert + 25 mm near-success + canonical failure
-> logs/expert/transitions_fullinside/
-> 两条 expert Actor BC
-> 两条 full-success/near-success/failure Critic return regression
-> 1024-env pure on-policy PPO
-> 每 3M 完整诊断包
-> 512 deterministic evaluation

关键约束：

- cube 固定在 [-0.0259767957, -0.1788897067, 0.8830000162] m。
- 前 80 control steps 严格 reference-only；这些样本训练 Critic但不训练 Actor。
- confidence 控制 residual step/deviation bound。
- human pose 只形成中低 confidence 区域的运动方向 shape prior。
- 手指固定为 GraspPose。
- 当前训练只读取 logs/expert/transitions_fullinside/，禁止读取旧 entry 或 Deep20 transition。
- 不恢复任何历史 policy；当前 run 从随机网络初始化。

## 当前运行

tmux：sweep2_fullinside_v3_1024_20260901

run：
/home/msc-auto/RL_sweep/logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901

artifact prefix：
Sweep2FullInsideV3__20260901_policy

配置：

- num_envs=1024
- seed=42
- max_agent_steps=100000000
- GPU0，共享使用已获用户授权
- 3M agent steps 一个 checkpoint/metrics/rollout/video 节点
- 自动录像前训练在 epoch 边界 yield，录像结束后恢复

启动脚本：
logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/launch_pipeline.sh

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
- 1024-env 初始化、Actor BC 与 Critic 数据预热完成。
- 已进入 PPO；最近检查为 196,608 agent steps。
- 不得重复启动同名或第二个 1024-env run。

## 诊断产物

每 3M 节点：

logs/checkpoints/Sweep2FullInsideV3__20260901_policy_<XXXXM>/
- checkpoint.pth
- metrics.json
- rollout.npz
- record.log

outputs_video/Sweep2FullInsideV3__20260901_policy_<XXXXM>/
- policy.mp4
- topdown_frames/frame_*.png

录像规则：

- 成功物理 rollout 当步终止。
- recorder 在专用录制模式下抑制 terminal 自动 reset，并渲染真实 terminal physics state。
- 冻结源必须是 fully_inside 成立当步的真实终态 RGB 图像。
- 以 20 FPS 重复 40 帧，形成 2 秒展示冻结。
- 录像必须冻结真实 terminal physics frame，不得使用 terminal 前帧或 reset 后帧。

## 已验证专家与录像

- 两条 expert 均按当前 fully_inside 判据通过统一环境回放。
- trace 必须同时满足 success=True、Gate=[1,1,1,1] 与完整 footprint 入盆。
- 录像采用机器人左前方略高的中景；成功终态追加 40 个相同帧。
- 历史实验已按最小可复现集清理；当前保留两条 expert 所需输入、near-success、failure、标准回放和当前 v3 run。

## 安全与运维

- 用户允许与 feiyang 共享 GPU，但禁止操作对方任何 PID、tmux 或资源优先级。
- 只能精确停止本任务 tmux/PID，禁止 pkill、killall、GPU reset。
- 训练和本任务 recorder 不重叠。
- 不删除、覆盖或混写历史 checkpoint/run。
- 训练达到较高窗口成功率也不会自动代表 512 回合验收通过。

监控命令：

ssh msc-a6000 'tmux capture-pane -pt sweep2_fullinside_v3_1024_20260901:0 -S -80'
ssh msc-a6000 'tail -n 100 /home/msc-auto/RL_sweep/logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/train.log'
ssh msc-a6000 'cat /home/msc-auto/RL_sweep/logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/progress_steps.txt 2>/dev/null || true'

停止时必须先核对精确 session 与 PID，然后只向 sweep2_fullinside_v3_1024_20260901 发送 Ctrl-C。

## 下一步

1. 等待 BC/Critic warmup 和 PPO 开始。
2. 3M 检查 Gate3 浅进入与 Gate4 fully-inside 的漏斗差异。
3. 检查失败 episode 是否仍集中在 step104 mouth collision。
4. 检查 deterministic 视频冻结源及 trace terminal state。
5. 选择候选 checkpoint 做 512 回合 deterministic 验收。
