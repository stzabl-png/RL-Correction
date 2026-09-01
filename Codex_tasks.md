# Codex 任务台账：Sweep2 Full-Inside 残差训练

更新时间：2026-09-01。本文件是 Sweep2 当前实例的权威技术说明和运行台账，记录已采用的算法、数据、路径、训练状态与验收方式。未来处理其他 Sweep 轨迹时，应使用 `SWEEP_TRAJECTORY_PLAYBOOK.md` 的通用流程，不得直接复制本实例的动作、坐标或 checkpoint。

## 1. 当前目标与成功契约

输入是 `datasets/sweep_2_better/` 的 ego 视频重建结果。右手固定抓扫把，左手固定抓簸箕；工具轨迹经 IK 转成双臂 reference，策略学习 14 DoF 累计关节 residual，使固定 cube 的完整 footprint 进入 dustpan。

成功只能由 Isaac 物理状态判定：

- Gate1：场景、工具和入口姿态 ready。
- Gate2：扫把接近且 cube 真实位移至少 5 mm。
- Gate3 `entered`：cube 中心越过 mouth，并满足横向、深度和承载高度约束；只代表浅进入。
- Gate4 `fully_inside`：cube 完整 footprint 清过 mouth 后锁存，是唯一 operational success。
- success 当步立即终止物理 rollout；`deep_inside`、`deep_margin` 仅用于诊断。
- 最终验收为至少 512 个 deterministic episodes，full-inside rate >= 0.50。训练窗口成功率不能替代最终验收。

Deep20 已停止，不参与当前成功、reward 或验收。

## 2. 已验证数据流

```text
datasets/sweep_2_better
  -> build_reference.py：工具 6DoF + GraspPose + 双臂 IK
  -> sweep2_reference_v1.npz
  -> sweep_env.py：固定物理世界、双 FixedJoint、开放 dustpan collider
  -> 两条 fully-inside expert + 25 mm near-success + canonical failure
  -> build_expert_dataset.py：按当前 reward 重采 transition/return
  -> Actor BC + Critic return regression
  -> 1024-env pure on-policy PPO
  -> 每 3M checkpoint/metrics/trace/video
  -> 512 deterministic evaluation
```

冻结输入：

- reference：`tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz`
- transition 根目录：`logs/expert/transitions_fullinside/`
- transition 角色与 hash：`logs/expert/transitions_fullinside/manifest.json`
- dustpan asset：`tasks/Sweep/2/assets/dustpan_smooth_entry/object_mesh_scaled_final.obj`
- cube world start：`[-0.0259767957, -0.1788897067, 0.8830000162] m`

当前 run 禁止读取 `logs/expert/transitions/` 或 `logs/expert/transitions_deep20/`，也不恢复任何历史 policy。

## 3. Reference、Observation 与 Action

- Actor observation：191 维。
- Critic privileged state：22 维。
- Action：左右臂 14 个 joint residual；手指固定为 GraspPose。
- control period：0.05 s。
- 前 80 control steps 是 4 秒 scripted reference-only 前缀，实际 residual 强制为零。
- 前缀 transition 可训练 Critic，但 `actor_mask=0`，不进入 Actor objective、entropy、bounds、KL 或 Actor advantage normalization。
- confidence 决定 residual step bound 与 cumulative deviation bound。
- human pose 不做绝对腕位跟踪，只在中低 confidence 区域形成相邻运动方向 shape prior。
- nominal contact 前 reference row 可 open-loop 前进；随后需 broom near 或 cube moved 才继续，避免任务时钟脱离物体。

## 4. Expert 与预热

当前 transition 集包含四种角色：

- 两条 `full_success`：供 Actor BC，并共同给 Critic 提供成功 return。
- 25 mm `near_success`：已达到 Gate3，但未满足当前 fully-inside；只供 Critic 学习接近成功的 return 结构。
- `failure`：canonical zero-residual failure，供 Critic 建立失败端回报。

Actor BC 只学习两条 fully-inside expert。非零修正帧权重为 1，零前缀权重为 0.05。Critic 使用两条 full-success、near-success 与 failure 的 discounted return；若启用 critic-only warmup，Actor 在该阶段保持冻结。之后进入 pure on-policy PPO。

数据角色不得从文件名推断，必须读取 `manifest.json` 的 `demo_role`、`success_frame`、shape 和 SHA-256。

## 5. Reward 与失败

当前 reward 服务 fully-inside 目标：

- earn-only mouth progress：奖励首次取得的入口进展。
- earn-only full_progress：从中心 entered 继续推进到完整 footprint 入盆；只奖励历史最大值增量。
- Gate 首次达成奖励。
- broom 接近、推动质量。
- pan level、clear、still 质量。
- confidence tracking 与 human shape prior。
- action、smoothness 和 left-arm anchor penalty。

两个 progress 都必须是绝对几何量、不可往返刷分、reset/播种不付奖励。失败包括 cube 掉下桌面和 dustpan mouth 明显穿桌。reward 上升不能替代 Gate 或真实几何检查。

## 6. 关键代码路径

- reference：`tasks/Sweep/2/A_Design/L2_Reference/`
- tracker：`tasks/Sweep/2/A_Design/L3_Learning/progress_batch.py`
- environment：`tasks/Sweep/2/C_Wiring/sweep_env.py`
- expert collector：`tasks/Sweep/2/C_Wiring/make_expert.py`
- transition builder：`tasks/Sweep/2/C_Wiring/build_expert_dataset.py`
- warmup：`tasks/Sweep/2/C_Wiring/bc_warmup.py`
- training：`tasks/Sweep/2/C_Wiring/train_sweep.py`
- recording：`tasks/Sweep/2/C_Wiring/record_sweep.py`
- evaluation：`tasks/Sweep/2/C_Wiring/eval_sweep.py`

## 7. 当前训练

- tmux：`sweep2_fullinside_v3_1024_20260901`
- run：`logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/`
- artifact prefix：`Sweep2FullInsideV3__20260901_policy`
- envs：1024
- seed：42
- max agent steps：100M
- GPU：GPU0，与 feiyang 共享已获用户授权；禁止操作对方任何进程或资源优先级。
- launch：`logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/launch_pipeline.sh`
- train log：`logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/train.log`
- checkpoint root：`logs/checkpoints/Sweep2FullInsideV3__20260901_policy_*`

1-env random smoke、Actor BC 和 Critic 数据预热均已完成，当前处于 PPO。不得重复启动第二个 1024-env run。

监控：

```bash
ssh msc-a6000 'tmux capture-pane -pt sweep2_fullinside_v3_1024_20260901:0 -S -80'
ssh msc-a6000 'tail -n 100 /home/msc-auto/RL_sweep/logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/train.log'
ssh msc-a6000 'cat /home/msc-auto/RL_sweep/logs/Sweep2_fullinside_v3_fixed1024_seed42_20260901/progress_steps.txt 2>/dev/null || true'
```

停止时必须先核对精确 tmux 与 PID，只能向本任务 session 发送 Ctrl-C；禁止 `pkill`、`killall` 或 GPU reset。

## 8. 诊断、录像与验收

每 3M 节点必须包含：

```text
logs/checkpoints/Sweep2FullInsideV3__20260901_policy_<XXXXM>/
  checkpoint.pth
  metrics.json
  rollout.npz
  record.log

outputs_video/Sweep2FullInsideV3__20260901_policy_<XXXXM>/
  policy.mp4
  topdown_frames/frame_*.png
```

优先检查 Gate1→Gate2→Gate3→Gate4 funnel、terminal step、mouth clearance、cube-pan 几何、push/pan quality、Actor residual 与 left residual。尤其要区分 Gate3 浅进入和 Gate4 完整进入。

录像使用机器人左前方略高的中景，覆盖上半身、双臂和桌面操作区。专用录制模式抑制 terminal 自动 reset，渲染 `fully_inside` 成立当步的真实 terminal state；随后在 20 FPS 下追加 40 个相同帧，冻结 2 秒。terminal 前帧和 reset 后帧都不能冒充成功终态。

候选 checkpoint 最终运行至少 512 个 deterministic episodes；仅当 full-inside rate >= 0.50 才通过。

## 9. 当前状态与下一步

截至本次文档重构前，训练 tmux 存活且 PPO 正在推进。实时步数以 `progress_steps.txt` 为准，不在本段固化易过期的数字。

下一步：

1. 到 3M 读取完整诊断包，检查 Gate3/Gate4 漏斗和 step104 mouth collision 是否仍集中出现。
2. 核对 deterministic 视频的真实 terminal state、新视角和 2 秒冻结。
3. 根据 3M 证据决定继续训练或修正；不得只凭 reward 或单条视频下结论。
4. 对候选 checkpoint 执行 512 回合 deterministic 最终验收。
