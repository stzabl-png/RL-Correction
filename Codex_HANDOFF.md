# Sweep2 Full-Inside 交接说明

更新时间：2026-09-01。当前权威目标为 fully-inside：方块完整 footprint 进入簸箕后才算成功。Deep20 是已停止的历史实验，不再是训练或验收标准。

## 1. 阅读顺序

1. Codex_HANDOFF.md：当前目标、运行状态、安全边界和接手顺序。
2. Codex_tasks.md：当前 Sweep2 实例的完整算法、结构化数据流、路径、命令和实测结果。
3. SWEEP_TRAJECTORY_PLAYBOOK.md：未来处理其他 Sweep 轨迹时可复用的已验证通用方法。
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
- fully_inside 是 Gate4/success 的硬判据；deep_inside、deep_margin 仅为诊断字段，不影响成功或 task reward。
- deterministic 视频在真实 terminal physics state 上停止；将该终态图像重复 40 帧，在 20 FPS 下冻结 2 秒。

最终验收仍需至少 512 个 deterministic episodes，full-inside rate >= 0.50。训练窗口成功率不能代替最终验收。

## 4. 数据与算法

数据流：

ego reconstruction -> build_reference.py -> sweep2_reference_v1.npz
-> sweep_env.py 冻结物理世界 -> 两条 full-success expert、near-success 与 failure
-> build_expert_dataset.py -> Actor/Critic warmup -> pure on-policy PPO
-> 每 3M checkpoint/metrics/trace/video -> 512 回合验收

核心路径：

- reference：tasks/Sweep/2/A_Design/L2_Reference/
- tracker：tasks/Sweep/2/A_Design/L3_Learning/progress_batch.py
- environment：tasks/Sweep/2/C_Wiring/sweep_env.py
- transitions：logs/expert/transitions_fullinside/
- warmup：tasks/Sweep/2/C_Wiring/bc_warmup.py
- training：tasks/Sweep/2/C_Wiring/train_sweep.py
- recording：tasks/Sweep/2/C_Wiring/record_sweep.py
- evaluation：tasks/Sweep/2/C_Wiring/eval_sweep.py

Actor observation 为 191 维，Critic privileged state 为 22 维，action 为双臂 14 维。前 80 control steps 强制零 residual，且从 Actor objective、entropy、bounds、KL 和 advantage normalization 中排除。Actor BC 使用两条 fully-inside expert；Critic 使用两条 full-success expert、25 mm near-success 与 canonical failure return。

当前 v4 在原 Gate、expert action、BC 和 PPO 结构上新增 mouth-floor 约束：mouth clearance 低于 `+0.5 mm` 后施加 `-4*((0.5mm-clearance)/3.5mm)^2`，低于 `-3.0 mm` 时硬失败。该设计阻止策略通过下压簸箕、提前终止来获取正回报；不使用左臂动作冻结。

冻结输入：

- reference：tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz
- full-inside transitions：logs/expert/transitions_fullinside/
- expert 数据：两条 fully-inside expert；另保留 25 mm near-success 与 canonical failure 供 Critic 预热
- canonical replay：outputs_video/sweep2_v1_smooth_asset_physical_entry_replay.mp4
- cube world start：[-0.0259767957, -0.1788897067, 0.8830000162] m

禁止使用 logs/expert/transitions_deep20/ 启动当前训练。

## 5. 当前运行

当前从头训练：

- tmux：sweep2_floorpenalty_v4_1024_20260901
- run：logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/
- artifact prefix：Sweep2FloorPenaltyV4__20260901_policy
- seed：42
- environments：1024
- max agent steps：100M
- launch：logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/launch_pipeline.sh
- train log：logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/train.log
- checkpoint root：logs/checkpoints/Sweep2FloorPenaltyV4__20260901_policy_*

按用户决定，本 run 复用现有四条 transition，不重放 expert、不重采离线 return，并跳过 1-env smoke，直接完成 1024-env 初始化、Actor BC、Critic 回归和前 10 个 critic-only epochs；当前处于完整 on-policy PPO。最近核查已超过 7.2M agent steps。不得重复启动第二个 run。

监控：

ssh msc-a6000 'tmux capture-pane -pt sweep2_floorpenalty_v4_1024_20260901:0 -S -80'
ssh msc-a6000 'tail -n 100 /home/msc-auto/RL_sweep/logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/train.log'
ssh msc-a6000 'cat /home/msc-auto/RL_sweep/logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/progress_steps.txt 2>/dev/null || true'

## 6. 已完成回归

- CPU tracker self-test PASS。
- 第二条 fully-inside expert 在统一环境回放中通过成功判据。
- terminal trace Gate 为 [1,1,1,1]，cube_pan.z 约 63.07 mm。
- 两条 expert 均在统一 fully-inside contract 下回放成功。
- recorder 可渲染真实 terminal physics state，并抑制该次自动 reset；terminal 前帧和 reset 后帧均不得作为成功冻结源。
- 主视频采用机器人左前方略高的中景，覆盖上半身、双臂与桌面操作区；终态追加 40 帧冻结。
- v4 3M deterministic rollout 因 mouth clearance `-3.25 mm` 失败，总 reward 已降为 `-4.96`，证明旧穿桌捷径不再赚钱。
- v4 6M 训练窗口 Gate4 `69.37%`、`fully_inside=70.27%`；约 6M–7M Gate4 均值约 `72.2%`。
- v4 6M deterministic rollout 在 step 299 达成 Gate4，最小/终态 mouth clearance `+1.34/+1.54 mm`，总 reward `+24.60`；录像为 300 帧真实物理过程加 40 帧终态冻结。
- 已清理被 v4 取代的 v3 原始 run 日志；当前 v4、expert、来源证据、checkpoint 和 3M/6M 诊断均保留。
- 代码修正 Git commit：`07e0f94`；v4 文档与清理台账 commit：`4c24374`。

## 7. 最近操作顺序

1. 先核对 tmux、PID、`world.json`、progress 和 Git 状态，不重复 launch，不触碰 feiyang 的进程。
2. 到 9M 检查下一份完整诊断包，并与 6M 比较 Gate3/Gate4、mouth clearance、效率和 deterministic 行为。
3. 若 9M 未显著优于 6M，保留 6M 作为候选；不能只按训练步数选择 checkpoint。
4. 自动录像必须渲染真实 Gate4 terminal physics state，并冻结该终态 2 秒。
5. 候选 checkpoint 运行至少 512 回合 deterministic evaluation；`fully_inside rate >= 0.50` 才通过最终验收。
