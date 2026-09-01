# Codex 任务：Sweep2 Full-Inside 残差训练

更新时间：2026-09-01。本文件是 Sweep2 当前实例的权威算法说明和运行台账。阅读者应能从这里理解：原始重建数据如何变成 reference，策略每一步看到什么、输出什么，Gate/reward 如何由物理状态产生，expert transition 如何预热 Actor/Critic，以及 PPO、诊断和最终验收如何衔接。

未来处理其他 Sweep 轨迹时，应遵循 `SWEEP_TRAJECTORY_PLAYBOOK.md` 的通用流程；不得直接复制本实例的动作、坐标或 checkpoint。

## 1. 问题定义

输入是 `datasets/sweep_2_better/` 的 ego 视频重建结果。右手固定抓扫把，左手固定抓簸箕。重建工具轨迹经双臂 IK 转为 nominal joint reference；策略不从零生成动作，而是在 reference 上学习双臂 14 DoF 累计关节 residual。

最终控制目标是：在 Isaac 物理仿真中，利用扫把和簸箕把固定起点的 cube 完整扫入 dustpan。reference、reward 数值或视频观感都不能直接声明成功。

当前成功契约：

- Gate1：pan 姿态、入口 corridor 和场景状态 ready。
- Gate2：Gate1 已成立，扫把接近 cube，且 cube 相对初始位置真实移动至少 5 mm。
- Gate3 `entered`：Gate2 已成立，cube 中心越过 mouth，并满足横向 footprint、承载高度和盆内后边界约束；它只记录浅进入。
- Gate4 `fully_inside`：Gate3 已成立，cube 的完整 footprint 清过 mouth；这是唯一 operational success。
- Gate 全部是单调锁存；Gate4 成立当步立即终止物理 rollout。
- `deep_inside`、`deep_margin`、稳定步数等只用于诊断，不参与当前 success。
- 最终验收为至少 512 个 deterministic episodes，full-inside rate >= 0.50。

Deep20 已停止，不参与当前 success、reward、训练目标或验收。

## 2. 端到端算法总览

```text
ego 重建数据
  │ 工具/物体 6DoF、confidence、human hand trajectory、GraspPose
  ▼
build_reference.py
  │ wrist-tool 刚性关系 + 每帧双臂 7DoF IK
  ▼
sweep2_reference_v1.npz
  │ joint reference、工具 reference、human joint direction、confidence、contact_row
  ├─────────────────────────────────────────────────────────┐
  ▼                                                         │
SweepEnv 在线物理闭环                                        │
  obs(191) + priv(22)                                       │
  -> Actor 输出 14D normalized residual increment           │
  -> confidence 缩放 step/deviation envelope                │
  -> 累计 residual + 当前 reference row = joint target      │
  -> Isaac physics                                           │
  -> 真实 cube/tool state                                    │
  -> Gate、reward、done、next observation                    │
  │                                                         │
  ├─ expert replay -> build_expert_dataset.py               │
  │      -> full-success / near-success / failure transition│
  │      -> Actor BC + Critic return regression             │
  │                                                         │
  └─ 1024-env on-policy rollout -> PPO update ──────────────┘
          -> 每 3M 诊断节点 -> 512 deterministic 验收
```

算法分成三个时间尺度：

1. 离线一次性构建 reference、物理资产、expert 和 transition。
2. 每个 0.05 s control step 执行一次 residual policy、物理仿真、Gate/reward 更新。
3. 每个 PPO epoch 先收集 32-step on-policy rollout，再进行 5 轮 minibatch 更新。

## 3. 从重建轨迹到 joint reference

`build_reference.py` 读取工具/物体 6DoF、左右手轨迹、confidence 和 GraspPose。对每只手建立固定 wrist-tool transform，再对每个源时刻做 7DoF arm IK，产生：

- `right_q[T,7]`、`left_q[T,7]`：双臂 nominal joint reference。
- `obj_pos_i[T,3]`、`obj_quat_i[T,4]`：pan/broom 工具 reference。
- `human_right_q[T,7]`、`human_left_q[T,7]`：human hand trajectory 的关节空间表达。
- `confidence_0[T]`、`confidence_1[T]`：插值到 control timebase 的工具 confidence。
- `contact_row`：reference 中测得的最接近 nominal contact 行。
- `cube_start_w`：候选任务起点；当前实例在环境内进一步固定为已验收坐标。

环境加载后把右、左 7 DoF 拼成 `ref_arm[T,14]`。human joint 序列只转成相邻差分 `human_dh[t]`，用作方向 shape prior；它不是机器人必须追踪的绝对关节目标。

冻结输入：

- reference：`tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz`
- transition：`logs/expert/transitions_fullinside/`
- transition 角色与 hash：`logs/expert/transitions_fullinside/manifest.json`
- dustpan asset：`tasks/Sweep/2/assets/dustpan_smooth_entry/object_mesh_scaled_final.obj`
- cube world start：`[-0.0259767957, -0.1788897067, 0.8830000162] m`

当前 run 禁止读取旧 entry/Deep20 transition，也不恢复任何历史 policy。

## 4. 物理世界与坐标

- DexMate 双臂，右手 broom、左手 dustpan；手指始终固定为 GraspPose。
- 两个工具通过物理 FixedJoint 附着；reset 后位置误差必须小于 3 mm、姿态误差小于 2°。
- dustpan 使用开放 compound collider：盆底、入口 ramp、左右侧壁和后壁，不能用封住入口的 convex hull。
- 所有成功几何都在 pan local frame 中计算：`x` 是横向，`+y` 是 pan 法向，`+z` 从 handle 指向 mouth；cube 向 `-z` 深入。
- cube half extent 为 12.5 mm；pan half-width 为 60 mm，mouth z 为 95 mm，承载中心高度范围为 18–30 mm。

环境每步从 Isaac 读取 pan、broom、cube 的位置、四元数、线速度和角速度；将 cube、扫把最近工作点和相对速度变换到 pan frame，再统一用于 observation、Gate、reward、诊断和 termination。不存在另一个只供 reward 使用的“隐藏成功代理”。

## 5. Actor、Critic 输入

### 5.1 Actor observation：191 维

| 分组 | 维度 | 内容 |
|---|---:|---|
| 当前关节状态 | 56 | `q(14)`、`0.1*qd(14)`、`ref-q(14)`、归一化累计 residual(14) |
| reference look-ahead | 56 | 相对当前行的 `+1/+4/+8/+16` 四个 14D joint delta |
| pan 状态 | 13 | position、quaternion、linear velocity、缩放 angular velocity |
| broom 状态 | 13 | position、quaternion、linear velocity、缩放 angular velocity |
| cube 世界状态 | 6 | position、linear velocity |
| 任务相对状态 | 12 | `cube_pan(3)`、`broom-cube(3)`、左右 confidence(2)、progress、corridor、relative speed、moved |
| 任务记忆与时钟 | 20 | Gate1–4、stable ratio、reference row ratio、上一 14D policy action |
| 接触与入口几何 | 14 | broom 最近点相对 cube/pan、cube 相对速度、pan up、mouth clearance、4 个 containment margin |
| policy-active 标志 | 1 | 是否已经结束 4 秒 scripted prelude |

合计 191 维。输入最终执行 `float -> clamp[-10,10] -> nan_to_num(0)`，训练器再使用 running mean/std 归一化。

### 5.2 Critic privileged state：22 维

Critic 除 191 维 policy observation 外，还接收 22 维 privileged state：

- cube 在 pan frame 中的位置 3 维。
- cube 世界线速度 3 维。
- broom distance 与 cube-pan relative speed，各 1 维。
- broom 最近点相对 pan/cube 3 维。
- cube 在 pan frame 中的相对速度 3 维。
- mouth clearance、pan tilt、pan linear speed、pan angular speed，各 1 维。
- 完整 footprint 的 4 个 containment margins。

网络采用 separate critic：Actor MLP 为 `[256,128]`；Critic 有 privileged embedding `[64,32,8]` 和主干 `[256,256,128]`。Actor 不直接读取 privileged-only 22 维分支。

## 6. 每个 control step 的 residual 闭环

设当前 reference 行为 `r_t`，Actor 输出归一化动作 `a_t∈[-1,1]^14`。实际执行流程如下：

1. **4 秒 reference 前缀**：episode 前 80 步令 `policy_active=0`，所以执行动作强制为零；之后才允许 residual。
2. **confidence 形成动作包络**：每只手当前 confidence 扩展到对应 7 个关节。高 confidence 使用更窄包络，低 confidence 允许更大修正。
3. **逐步累计 residual**：

   ```text
   step(c) = step_hi + c * (step_lo - step_hi)
   dev(c)  = dev_hi  + c * (dev_lo  - dev_hi)
   cum_res[t+1] = clip(cum_res[t] + a_t * step(c), -dev(c), +dev(c))
   q_target[t]  = ref_arm[r_t] + cum_res[t+1]
   ```

4. **当前数值（弧度）**：右臂 `step_hi/lo=0.020/0.008`、`dev_hi/lo=0.25/0.10`；左臂为 `0.010/0.005`、`0.12/0.05`。右臂负责扫动，因此探索包络约为左臂两倍。
5. **执行物理**：将 `q_target` 写入 14 个 arm joint position target，finger 继续写固定 GraspPose；Isaac 执行 physics substeps。
6. **读取真实结果**：重新计算 cube/tool state、Gate、reward、failure 和下一 observation。
7. **推进 reference row**：在 nominal `contact_row` 前可以 open-loop 前进；之后只有 broom near 或 cube moved 达标时才推进，防止 reference 时钟继续走而物体留在原地。

因此 residual action 是“对 reference 的有界累计修正”，不是绝对关节命令，也不是每步互相独立的 offset。

## 7. Gate、progress 与终止状态机

`progress_batch.py` 是不依赖 Isaac 的统一状态机；离线 expert 重采和在线环境使用同一套几何定义。

每步顺序：

```text
Isaac state
  -> cube/tool 转到 pan frame
  -> ready、broom_near、moved、entered、fully_inside
  -> Gate1..Gate4 单调锁存
  -> progress/full_progress 的历史最大值增量
  -> task reward
  -> success / failure / timeout
```

`entered` 要求横向 footprint 合法、中心越过 mouth、承载高度合法；`fully_inside` 进一步要求 cube 的 mouth-side face 也清过 mouth。Gate3 只锁存 `entered`，Gate4 才锁存 `fully_inside`。

终止条件：

- success：Gate4 新成立或已成立。
- failure：cube 低于桌面 30 mm，或 dustpan mouth clearance 低于 `-3.0 mm`。
- timeout：达到最大 episode length 且未 success/failure。

录像模式是唯一例外：它抑制 DirectRLEnv 的 terminal auto-reset，以便渲染真实 terminal physics state；这不会改变训练的物理轨迹或成功定义。

## 8. Reward 的组成与因果

总 reward 是十项之和：

```text
reward = task + acquire + push + pan_quality + success_quality
       + mouth_floor + track + shape + action + left_anchor
```

### 8.1 Task reward

```text
task = 4 * mouth_delta * corridor
     + 4 * full_delta
     + 0.5 * new_Gate1
     + 1.0 * new_Gate2
     + 4.0 * new_Gate3
     + 12.0 * new_Gate4
```

- `mouth_delta`：从固定起点向 mouth 推进的历史最大值增量。
- `full_delta`：从中心 entered 到完整 footprint 清口的历史最大值增量。
- 两者都采用 earn-only ratchet：只对超过历史最好值的正增量付钱，停住或来回摆动不能刷分，reset/播种不付钱。

### 8.2 物理质量与正则

- `acquire`：扫把接近 cube 的 contact potential 首次改善。
- `push = 4 * assisted_delta`：只有 inward progress 同时满足扫把接触质量、正确后方/横向/高度和入口 corridor 才奖励。
- `pan_quality`：pan level、mouth clearance、低线速度和低角速度联合 potential 的首次改善。
- `success_quality = 2 * pan_quality * new_Gate4`：完整进入当步奖励稳定、平整、离桌合理的 pan。
- `mouth_floor`：当 mouth clearance 低于 `+0.5 mm` 后施加直接二次惩罚：`-4*((0.5mm-clearance)/3.5mm)^2`。在 `-3.0 mm` 达到 `-4/step` 并触发失败，使 3M/6M 曾出现的下压短回合不再获得正收益。
- `track`：confidence-weighted 工具 reference 超差惩罚。confidence 越低，容差越大；它永远不是 success proxy。
- `shape = 0.2 * w_hand * max(cos(actual_joint_delta, human_joint_delta),0)`：仅中低 confidence 生效；`confidence>=0.70` 时权重 0，`0.40–0.70` 为 0.5，更低为 0.8。
- `action = -0.002||a_t||² - 0.001||a_t-a_{t-1}||²`。
- `left_anchor = -0.001||left_cum_res/left_dev_hi||²`，抑制簸箕臂无必要漂移。

reward 提升只能说明优化信号变化；是否成功必须看 Gate4 和真实 footprint 几何。

## 9. Expert transition 数据流

当前 transition manifest 包含四条 rollout：

- 两条 `full_success`：同时供 Actor BC 和 Critic。
- 25 mm `near_success`：达到 Gate3 但未达到当前 Gate4，只供 Critic。
- canonical `failure`：未完成任务，只供 Critic。

`build_expert_dataset.py` 在当前环境中逐步重放每条源 rollout，并保存：

```text
obs, priv_info, actions, rewards,
next_obs, next_priv_info,
done, success, rows, actor_mask,
return_target
```

`return_target` 使用 `gamma=0.99` 的 discounted return，并乘 `0.01` reward scale。每条输出记录样本数、entered/success frame、`demo_role`、源文件和输出文件 SHA-256。训练时必须读取 manifest 的角色和 hash，不能从文件名猜用途。

重要约束：成功契约或 reward 一旦修改，必须在新环境下重放并重新生成 transition/return；不能只改标签。

## 10. 初始化：Actor BC 与 Critic 回归

随机网络不直接进入 PPO。`bc_warmup.py` 依次执行两个阶段。

### 10.1 Actor BC

- 数据：只拼接两条 fully-inside expert。
- 目标：Actor mean `mu(obs)` 拟合 expert 14D residual action。
- loss：逐样本 action MSE；非零 interaction action 权重 1，零动作权重 0.05。
- `actor_mask=0` 的 4 秒 scripted prefix 权重归零，绝不模仿当时“策略采样了但环境没有执行”的动作。
- observation running mean/std 使用全部四条数据拟合，使后续 near/failure 输入也在同一归一化尺度。
- 优化：300 epochs，batch size 256，Adam `lr=3e-4`，gradient norm clip 1.0。

### 10.2 Critic return regression

- 数据：两条 full-success + 25 mm near-success + canonical failure。
- 输入：归一化 observation + 22D privileged state。
- 目标：归一化后的 `return_target`。
- loss：Smooth L1/Huber。
- 优化期间只启用 critic/value 参数，Actor 参数显式冻结。
- 优化：200 epochs，batch size 256，Adam `lr=3e-4`，gradient norm clip 1.0。

预热完成后保存独立 BC checkpoint；它是当前随机初始化 run 的起点，不是历史 policy 恢复。

## 11. On-policy PPO

预热之后，所有 rollout 和更新都是 pure on-policy PPO；expert action 不会混入在线 rollout。

主要配置：

- 1024 environments，horizon 32，因此一个完整 rollout epoch 产生 32,768 agent steps。
- `gamma=0.99`，GAE `lambda=0.95`。
- minibatch 8192，5 mini-epochs。
- PPO clip `epsilon=0.2`；value 也使用 clipped loss。
- Adam 初始/最高 `lr=3e-4`，最低 `3e-5`，KL threshold 0.02。
- entropy coefficient 0.001，bounds coefficient 0.0001，gradient norm clip 1.0。
- policy sigma init 0.05，floor 0.02。

PPO 还有两个保护层：

1. **前 10 个 PPO epochs critic-only**：仍收集当前 policy 的 on-policy rollout，但更新 loss 只保留 Critic；Actor、entropy、bounds 和 LR 调度冻结，避免随机 Critic advantage 破坏已经可用的 BC Actor。
2. **scripted-prefix actor mask**：前 80 步仍进入 Critic return/GAE，但从 Actor surrogate loss、entropy、bounds、KL 和 advantage normalization 统计中排除。Critic 学到前缀状态价值，Actor 只为真正执行 residual 的状态负责。

PPO 更新链：

```text
32-step on-policy rollout
  -> bootstrap value + GAE
  -> advantage 只用 actor_mask=1 样本统计归一化均值/方差
  -> 5 × minibatch：clipped actor loss + clipped critic loss
                   - entropy bonus + bounds penalty
  -> gradient clip -> optimizer step
  -> epoch metrics、Gate rates、reward terms、residual usage
```

## 12. 关键代码与权威产物

- reference：`tasks/Sweep/2/A_Design/L2_Reference/build_reference.py`
- tracker：`tasks/Sweep/2/A_Design/L3_Learning/progress_batch.py`
- environment：`tasks/Sweep/2/C_Wiring/sweep_env.py`
- expert collector：`tasks/Sweep/2/C_Wiring/make_expert.py`
- transition builder：`tasks/Sweep/2/C_Wiring/build_expert_dataset.py`
- warmup：`tasks/Sweep/2/C_Wiring/bc_warmup.py`
- PPO entrypoint：`tasks/Sweep/2/C_Wiring/train_sweep.py`
- PPO config：`tasks/Sweep/2/C_Wiring/ppo_sweep.yaml`
- shared PPO masking：`rl_rebuild/algo/ppo/ppo.py`、`rl_rebuild/algo/ppo/experience.py`
- recording：`tasks/Sweep/2/C_Wiring/record_sweep.py`
- evaluation：`tasks/Sweep/2/C_Wiring/eval_sweep.py`

run 内 `world.json` 冻结 success、policy I/O、time、reference/asset/transition 路径及 SHA-256；它是判断某个 checkpoint 实际训练世界的权威配置，不应仅凭目录名推断。

## 13. 当前训练与监控

- tmux：`sweep2_floorpenalty_v4_1024_20260901`
- run：`logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/`
- artifact prefix：`Sweep2FloorPenaltyV4__20260901_policy`
- seed：42
- max agent steps：100M
- GPU：GPU0，与 feiyang 共享已获用户授权；禁止操作对方进程、tmux 或资源优先级。
- launch：`logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/launch_pipeline.sh`
- train log：`logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/train.log`
- checkpoint root：`logs/checkpoints/Sweep2FloorPenaltyV4__20260901_policy_*`

v3 在 8,192,000 steps 停止：其约 2M Gate4 峰值为 18.2%，但 3M/6M deterministic rollout 均在 Gate2 后因 mouth 穿桌提前失败，之后窗口 Gate4 塌缩到约 1%。v4 保持 Gate、expert action 和 PPO 不变，只加入 mouth-floor 强惩罚并把硬失败下限放宽到 `-3 mm`。

按用户决定，v4 直接复用现有四条 transition，不重放 expert、不重采 `return_target`，并跳过 1-env smoke，直接从随机网络进行 1024-env 初始化、BC/Critic 预热和 PPO。由于 reward 已改变，复用的 Critic 离线 return 与新在线 reward 不完全一致；这是本次明确接受的实验条件。1024-env 初始化、Actor BC、离线 Critic 回归和前 10 个 critic-only epochs 均已完成，当前处于完整 on-policy PPO。

已验证里程碑：

- 3M：训练窗口 Gate4 `8.74%`；deterministic rollout 在 step 322 因 mouth clearance `-3.25 mm` 失败，Gate `[1,1,0,0]`，总 reward `-4.96`。这说明新惩罚已经把原先可获正回报的穿桌捷径改成负回报，但此时策略尚未学会完整进入。
- 6M：checkpoint 实际为 `6,029,312` steps。训练窗口 Gate1/Gate2/Gate3/Gate4 分别为 `100% / 98.20% / 72.97% / 69.37%`，`fully_inside=70.27%`（111 episodes）。deterministic rollout 在 step 299 达成 Gate4，最小 mouth clearance `+1.34 mm`、终态 `+1.54 mm`，总 reward `+24.60`；录像含 300 帧真实物理过程和 40 帧终态冻结。
- 约 6M–7M：训练窗口 Gate4 均值约 `72.2%`，单窗口峰值 `83.3%`；reward 与 Gate4 同向改善，不再出现 v3 的“失败但高回报”分离。

监控：

```bash
ssh msc-a6000 'tmux capture-pane -pt sweep2_floorpenalty_v4_1024_20260901:0 -S -80'
ssh msc-a6000 'tail -n 100 /home/msc-auto/RL_sweep/logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/train.log'
ssh msc-a6000 'cat /home/msc-auto/RL_sweep/logs/Sweep2_floorpenalty_v4_fixed1024_seed42_20260901/progress_steps.txt 2>/dev/null || true'
```

停止时必须先核对精确 tmux 与 PID，只能向本任务 session 发送 Ctrl-C；禁止 `pkill`、`killall` 或 GPU reset。

## 14. 3M 诊断、录像与最终验收

每 3M 节点必须包含：

```text
logs/checkpoints/Sweep2FloorPenaltyV4__20260901_policy_<XXXXM>/
  checkpoint.pth
  metrics.json
  rollout.npz
  record.log

outputs_video/Sweep2FloorPenaltyV4__20260901_policy_<XXXXM>/
  policy.mp4
  topdown_frames/frame_*.png
```

诊断优先顺序：

1. Gate1→Gate2→Gate3→Gate4 funnel，特别区分浅进入与完整进入。
2. terminal step 和 step104 mouth collision 是否集中。
3. cube-pan z、4 个 containment margins、mouth clearance。
4. broom-assisted progress、pan quality、右/左 residual usage。
5. Actor/Critic loss、entropy、KL 只用于解释策略变化，不能替代任务指标。

录像采用机器人左前方略高的中景。专用模式抑制 terminal auto-reset，渲染 Gate4 成立当步的真实 physics state；随后在 20 FPS 下重复该终态 40 帧，冻结 2 秒。terminal 前帧和 reset 后帧均不合格。

候选 checkpoint 最终运行至少 512 个 deterministic episodes；仅当 full-inside rate >= 0.50 才通过。表格 A 的 RL 成功率和效率以该评测为准；明显失败的 ablation 不进入后续 DP distillation 表格 B。

## 15. 当前状态与下一步

v3 已停止；其根因、3M/6M 诊断结论和关键数值已写入本文、`Codex_commit.md` 与 `Codex_mistakes.md`。确认不再被训练、expert manifest 或回退路径引用后，旧 v3 原始 run 日志目录已清理。当前 v4 checkpoint、3M/6M 诊断、视频、expert 数据及其来源证据全部保留。

v4 在 `sweep2_floorpenalty_v4_1024_20260901` 中持续运行，最近核查已超过 7M agent steps；实时状态以 tmux、`train.log` 和 `progress_steps.txt` 为准。

下一步：

1. 到 9M 读取下一份完整诊断包，与 6M 比较 Gate3/Gate4、mouth clearance、效率和 deterministic 行为。
2. 若 9M 未显著优于 6M，保留 6M 作为早期候选；不能只按训练步数选择 checkpoint。
3. 对候选 checkpoint 执行至少 512 回合 deterministic 最终验收，报告成功率和效率分布。
4. 只有 `fully_inside rate >= 0.50` 才进入表格 A；明显失败的 ablation 不进入 DP distillation 表格 B。
