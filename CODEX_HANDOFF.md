# Sweep2 Full-Inside 交接说明

更新时间：2026-08-31。当前权威目标已经从 Deep20 回退为旧 15M 所用的首次有效进入（full-inside）。Deep20 是已停止的历史实验，不再是训练或验收标准。

## 1. 阅读顺序

1. CODEX_HANDOFF.md：当前目标、运行状态和安全边界。
2. Codex_tasks.md：当前 run、路径、命令和已完成验证。
3. SWEEP_TRAJECTORY_PLAYBOOK.md：从重建轨迹到 residual PPO 的通用流程。
4. Codex_mistakes.md：历史错误与预防检查。
5. Codex_commit.md：工程演进记录。
6. docs/TRAINING_DESIGN_GUIDE.md：Pour/Step4 的设计来源与消融背景。

## 2. 项目与安全边界

- SSH：msc-a6000
- 根目录：/home/msc-auto/RL_sweep
- 分支：sweep-task
- 当前不做 cube 随机化。
- 用户允许本任务与 feiyang 共享 GPU0，但禁止停止、发送信号、renice 或修改任何外部进程。
- 本任务训练与本任务自动录像不能重叠；autorecord 通过 pause/yield 协议协调。
- 禁止 reset --hard、checkout 丢弃改动、GPU reset 和广域进程终止。
- 未经用户明确要求不 push。

## 3. 当前任务

输入为 datasets/sweep_2_better 的 ego 视频重建结果。右手固定抓扫把、左手固定抓簸箕；工具轨迹经 IK 形成双臂 reference，策略只学习 14 DoF 累计关节 residual。confidence 调节 residual envelope，人手轨迹在中低 confidence 区域提供运动方向 shape prior。

当前成功定义：

- Gate1：场景与入口姿态 ready。
- Gate2：扫把接近且 cube 产生真实位移。
- Gate3：cube 中心第一次满足有横向、深度和承载高度约束的 entered。
- Gate4：cube 完整 footprint 进入后锁存，表示 operational success。
- 成功当步立即终止物理 rollout。
- fully_inside、deep_inside、deep_margin 仅为诊断字段，不影响成功或 task reward。
- deterministic 视频不继续仿真；只把最后一个有效 pre-reset 图像重复 40 帧，在 20 FPS 下冻结 2 秒。

最终验收仍需至少 512 个 deterministic episodes，full-inside rate >= 0.50。训练窗口成功率不能代替最终验收。

## 4. 数据与算法

数据流：

ego reconstruction -> build_reference.py -> sweep2_reference_v1.npz
-> sweep_env.py 冻结物理世界 -> 25/40 mm right-arm expert
-> build_expert_dataset.py -> Actor/Critic warmup -> pure on-policy PPO
-> 每 3M checkpoint/metrics/trace/video -> 512 回合验收

核心路径：

- reference：tasks/Sweep/2/A_Design/L2_Reference/
- tracker：tasks/Sweep/2/A_Design/L3_Learning/progress_batch.py
- environment：tasks/Sweep/2/C_Wiring/sweep_env.py
- transitions：logs/expert/transitions/
- warmup：tasks/Sweep/2/C_Wiring/bc_warmup.py
- training：tasks/Sweep/2/C_Wiring/train_sweep.py
- recording：tasks/Sweep/2/C_Wiring/record_sweep.py
- evaluation：tasks/Sweep/2/C_Wiring/eval_sweep.py

Actor observation 为 191 维，Critic privileged state 为 22 维，action 为双臂 14 维。前 80 control steps 强制零 residual，且从 Actor objective、entropy、bounds、KL 和 advantage normalization 中排除。Actor BC 同时使用 40 mm fully-inside expert 与旧 15M fully-inside rollout；Critic 使用 40 mm/15M full-success、25 mm near-success 与 canonical failure return。

冻结输入：

- reference：tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz
- full-inside transitions：logs/expert/transitions_fullinside/
- expert 视频：outputs_video/sweep2_expert_entry25_v1.mp4、sweep2_expert_entry40_v1.mp4
- canonical replay：outputs_video/sweep2_v1_smooth_asset_physical_entry_replay.mp4
- cube world start：[-0.0259767957, -0.1788897067, 0.8830000162] m

禁止使用 logs/expert/transitions_deep20/ 启动当前训练。

## 5. 当前运行

当前从头训练：

- tmux：sweep2_fullinside_v3_1024_20260901
- run：logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/
- artifact prefix：Sweep2FullInsideV3__20260901_policy
- seed：42
- environments：1024
- max agent steps：100M
- launch：logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/launch_pipeline.sh
- train log：logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/train.log
- checkpoint root：logs/checkpoints/Sweep2FullInsideV3__20260901_policy_*

1-env random smoke 已通过。最近一次交接检查时 1024-env 场景已创建并正在启动仿真，尚未写 progress_steps.txt；不得重复启动第二个 run。

监控：

ssh msc-a6000 'tmux capture-pane -pt sweep2_fullinside_v3_1024_20260901:0 -S -80'
ssh msc-a6000 'tail -n 100 /home/msc-auto/RL_sweep/logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/train.log'
ssh msc-a6000 'cat /home/msc-auto/RL_sweep/logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/progress_steps.txt 2>/dev/null || true'

## 6. 已完成回归

- CPU tracker self-test PASS。
- 旧 15M checkpoint 在恢复后的环境中于 step232 重现 entry success。
- terminal trace Gate 为 [1,1,1,1]，cube_pan.z 约 63.07 mm。
- 正确的最后有效图像是 topdown frame_0231.png。
- recorder 已跳过 terminal 后自动 reset 的 frame_0232。
- 回归 MP4 共 272 帧：232 个有效图像加 40 个冻结帧。
- 回归产物：logs/entry_restore_regression_20260831/ 与 outputs_video/Sweep2_entry_restore_regression_20260831/。

## 7. 最近操作顺序

1. 只读监控当前 tmux，等待初始化、BC、Critic warmup 和 PPO。
2. 确认 world.json 的 success.definition 为 fully_inside，transition 路径指向 logs/expert/transitions_fullinside/。
3. 到 3M 检查 Gate funnel、episode length、mouth clearance、push、视频和 terminal trace。
4. 自动录像必须以最后有效 pre-reset frame 冻结 2 秒。
5. 候选 checkpoint 运行至少 512 回合 deterministic evaluation。
