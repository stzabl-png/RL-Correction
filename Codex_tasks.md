# Codex 任务：Sweep2 Full-Inside Residual RL

更新时间：2026-09-03。本文件是 `datasets/sweep_2_better/` 这一条 Sweep2 实例的权威算法规范和运行台账。它必须回答“当前完整方法究竟如何运行”：固定输入、张量定义、逐步控制、Gate/reward、初始化、PPO、产物和验收都以本文及对应实现为准。

`SWEEP_TRAJECTORY_PLAYBOOK.md` 的职责不同：它只回答“换一条 Sweep 重建数据时，哪些东西必须重新生成、检查和放行”。Playbook 不保存本实例的运行目录、checkpoint 曲线或固定毫米数；本文也不把 Sweep2 的具体坐标误写成跨任务通用常量。

## 1. 当前任务契约

- 输入：`datasets/sweep_2_better/` 的 ego reconstruction。
- 机器人：DexMate 双臂；右手固定抓扫把，左手固定抓簸箕，finger 始终保持各自 GraspPose。
- nominal motion：重建工具轨迹经过逐帧双臂 IK 得到 joint reference。
- policy：不生成整条机器人动作，只学习左右臂共14维的有界累计关节 residual。
- control period：`0.05 s`；episode 前80个 control steps即4秒只播放 reference并强制零 residual。
- 物体：25 mm cube，当前固定世界起点为 `[-0.0259767957, -0.1788897067, 0.8830000162] m`。
- operational success：且仅当 Gate4 `fully_inside` 成立。
- 当前训练初始化：完全不使用 expert trajectory 预热。

Deep20、Gate3浅进入、训练 reward、训练窗口成功率和单条视频都不能替代 Gate4 批量验收。

## 2. 当前端到端数据流

```text
ego reconstruction
  ├─ 工具/物体 6DoF trajectory
  ├─ 左右 human hand trajectory
  ├─ 逐帧 confidence
  └─ broom/dustpan GraspPose
          │
          ▼
build_reference.py
  ├─ 固定 wrist-tool transform
  ├─ 每帧右臂7DoF IK + 左臂7DoF IK
  └─ 时间轴、四元数、joint-limit 与 jump 检查
          │
          ▼
sweep2_reference_v1.npz
  ├─ ref_arm[T,14]
  ├─ tool pose reference
  ├─ human_arm[T,14] -> human_dh[T,14]
  ├─ confidence[T,2]
  └─ contact_row / cube_start
          │
          ▼
SweepEnv × 1024
  ├─ reset固定机器人、工具、桌面、cube与状态机
  ├─ Actor observation obs[1024,191]
  ├─ Critic privileged state priv[1024,22]
  ├─ Actor采样 normalized action[1024,14]
  ├─ confidence缩放 residual step/deviation envelope
  ├─ q_target = ref_arm[row] + cumulative_residual
  ├─ Isaac physics substeps
  ├─ 读取真实 cube/pan/broom state
  └─ 计算 Gate、reward、done、next obs/priv
          │
          ▼
随机 Actor/Critic + 在线 PPO
  ├─ epoch 0–9：在线 rollout，critic-only update
  ├─ epoch 10起：完整 Actor-Critic PPO
  ├─ 每32步 rollout；5个 mini-epochs
  ├─ 周期保存 checkpoint/metrics/rollout
  └─ 周期录制 deterministic video
          │
          ▼
候选 checkpoint deterministic 批量评测
  └─ 至少512 episodes；fully_inside rate >= 0.50 才通过
```

## 3. Reconstruction 到 reference

权威脚本：`tasks/Sweep/2/A_Design/L2_Reference/build_reference.py`。

脚本读取工具/物体6DoF、左右人手轨迹、confidence和机器人 GraspPose。它先根据初始抓取关系建立固定 wrist-tool transform，再把每一帧工具目标转换为机器人 wrist target，对左右臂分别求7DoF IK。输出至少包含：

- `right_q[T,7]`、`left_q[T,7]`：右/左 nominal arm reference；环境拼接为 `ref_arm[T,14]`。
- `obj_pos_i[T,3]`、`obj_quat_i[T,4]`：簸箕与扫把工具 reference。
- `human_right_q[T,7]`、`human_left_q[T,7]`：人手运动转换到关节空间后的序列。
- `confidence_0[T]`、`confidence_1[T]`：插值到 control timebase 的工具重建可信度。
- `contact_row`：nominal 扫把最接近 cube 的 reference 行。
- `cube_start_w`：候选物体起点；本实例由环境固定为已验收坐标。

环境只使用 human joint 的相邻差分：

```text
human_dh[t] = human_arm[t+1] - human_arm[t]
```

它是运动方向 shaping，不是要机器人追踪的人手绝对姿态。每条新轨迹必须重新生成 reference，不能复制本实例的 joint reference、contact row、cube坐标或 checkpoint。

当前 reference：`tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz`。

## 4. 固定物理世界与坐标

- 机器人、桌面、cube、broom与dustpan均在同一个 Isaac 物理世界中运行。
- broom和dustpan通过真实 FixedJoint 附着到对应手；reset后工具位置误差必须小于3 mm、姿态误差小于2°。
- dustpan使用开放 compound collider：盆底、连续入口ramp、左右侧壁和后壁；禁止用封住入口的单一convex hull。
- 成功几何统一在 pan local frame 计算：`x`为横向，`+y`为pan法向，`+z`从handle指向mouth，cube向`-z`深入。
- cube half extent为12.5 mm；pan half-width为60 mm；mouth z为95 mm；承载中心高度合法范围为18–30 mm。

每个 physics step 后，从 simulator 重新读取 pan、broom、cube 的位置、四元数、线速度和角速度。cube位置、扫把工作点和相对速度统一转换到 pan frame，供 observation、Gate、reward、termination和诊断共同使用。不存在只供reward使用的隐藏成功代理。

## 5. Reset 与 episode 时序

reset 必须同时恢复：

1. 双臂 reference 起点与固定 finger GraspPose；
2. broom/dustpan FixedJoint 目标关系；
3. cube固定起点与速度；
4. reference row、累计 residual、previous action；
5. Gate1–4锁存、历史最佳 progress/full-progress；
6. timeout、failure和录像抑制reset标志。

前80步 `policy_active=0`：Actor仍可被调用，但环境执行动作强制为零，只播放 nominal reference。该前缀进入 Critic return/GAE，因为它属于真实episode；但必须从 Actor surrogate loss、entropy、bounds、KL和Actor advantage normalization中排除。

reference row在 nominal contact前可按时钟推进；contact后只有 broom near 或 cube moved 达标才继续推进，防止策略在没有物理交互时仅靠时间轴走完整段动作。

## 6. Actor observation：191维

| 分组 | 维度 | 精确定义 |
|---|---:|---|
| 当前关节状态 | 56 | `q(14)`、`0.1*qd(14)`、`ref-q(14)`、按当前上限归一化的累计residual(14) |
| reference look-ahead | 56 | reference行 `+1/+4/+8/+16` 相对当前行的四组14D delta |
| pan状态 | 13 | position(3)、quaternion(4)、linear velocity(3)、缩放angular velocity(3) |
| broom状态 | 13 | position(3)、quaternion(4)、linear velocity(3)、缩放angular velocity(3) |
| cube世界状态 | 6 | position(3)、linear velocity(3) |
| 任务相对状态 | 12 | `cube_pan(3)`、`broom-cube(3)`、左右confidence(2)、progress、corridor、relative speed、moved |
| 记忆与时钟 | 20 | Gate1–4、stable ratio、reference row ratio、上一14D action |
| 接触与入口几何 | 14 | broom工作点相对cube/pan、cube相对速度、pan up、mouth clearance、4个containment margins |
| policy active | 1 | 是否已结束4秒scripted prefix |

合计191维。环境输出先转换为float，截断到`[-10,10]`并对NaN/Inf做安全替换。当前无预热方法不从transition拟合normalization；running mean/std只随在线训练分布更新。

## 7. Critic privileged state：22维

Critic接收完整191维policy observation，并额外读取22维真实物理量：

- cube在pan frame的位置3维；
- cube世界线速度3维；
- broom distance与cube-pan relative speed各1维；
- broom工作点相对pan/cube共3维；
- cube在pan frame的相对速度3维；
- mouth clearance、pan tilt、pan linear speed、pan angular speed各1维；
- cube完整footprint的4个containment margins。

Actor MLP为`[256,128]`。Critic先用`[64,32,8]`处理privileged branch，再进入`[256,256,128]`主干。Actor在训练和推理时都不能读取22维privileged-only输入。

## 8. 14D累计 residual 控制

Actor产生归一化动作 `a_t∈[-1,1]^14`。对当前reference行 `r_t`：

```text
step(c) = step_hi + c * (step_lo - step_hi)
dev(c)  = dev_hi  + c * (dev_lo  - dev_hi)
cum_res[t+1] = clip(cum_res[t] + a_t * step(c), -dev(c), +dev(c))
q_target[t]  = ref_arm[r_t] + cum_res[t+1]
```

confidence越高，step和最大deviation越小；confidence越低，允许策略做更大修正。当前弧度参数：

| arm | low-confidence step | high-confidence step | low-confidence dev | high-confidence dev |
|---|---:|---:|---:|---:|
| right/broom | 0.020 | 0.008 | 0.25 | 0.10 |
| left/pan | 0.010 | 0.005 | 0.12 | 0.05 |

右臂负责扫动，所以包络约为左臂两倍。`q_target`写入14个arm joint position target；finger继续固定在GraspPose。action不是绝对关节命令，也不是互相独立的单步offset。

## 9. Confidence 与 human shaping

reference confidence先缩放到`[0,1]`，每只手的值扩展到对应7个关节。它有两个作用：

1. 控制上一节的residual step/deviation envelope；
2. 控制工具reference tracking代价的容差/权重。

human shaping根据两只工具confidence的较低值分档：

```text
confidence_floor >= 0.70  -> w_hand = 0.0
0.40 <= floor < 0.70     -> w_hand = 0.5
floor < 0.40             -> w_hand = 0.8
shape = 0.2 * w_hand * relu(cos(dq_actual, human_dh[row]))
```

它只奖励实际joint delta与human motion direction同向，不写入绝对human pose，也不参与success判定。

完整方法始终使用 reconstruction confidence，并启用上述中低置信 human direction shaping。

## 10. Gate、几何与终止

权威状态机：`tasks/Sweep/2/A_Design/L3_Learning/progress_batch.py`。

每步严格按下列顺序计算：

```text
post-physics state
 -> transform to pan frame
 -> ready / broom_near / moved / entered / fully_inside
 -> Gate1..Gate4 monotonic latch
 -> earn-only progress deltas
 -> reward
 -> success / failure / timeout
```

- Gate1：pan姿态、mouth corridor和场景满足ready。
- Gate2：Gate1已锁存，broom接近cube，且cube相对初始位置真实移动至少5 mm。
- Gate3 `entered`：Gate2已锁存；cube中心跨过mouth，同时满足横向footprint、承载高度和盆内后边界约束。它只是浅进入诊断。
- Gate4 `fully_inside`：Gate3已锁存；cube mouth-side face也清过mouth，完整footprint位于入口内。这是唯一success。

Gate只从0变1。Gate4成立当步立即终止真实物理rollout。失败条件包括cube低于桌面30 mm或dustpan mouth clearance低于`-3.0 mm`；其余在最大episode length终止为timeout。`deep_inside`、`deep_margin`不参与success或task reward。

## 11. Reward

```text
reward = task + acquire + push + pan_quality + success_quality
       + mouth_floor + track + shape + action + left_anchor

task = 4 * mouth_delta * corridor
     + 4 * full_delta
     + 0.5 * new_Gate1
     + 1.0 * new_Gate2
     + 4.0 * new_Gate3
     + 12.0 * new_Gate4
```

- `mouth_delta`：从固定起点向mouth推进的历史最佳值正增量。
- `full_delta`：从entered到完整footprint清口的历史最佳值正增量。
- 两者都是earn-only ratchet：倒退、停住或来回摆动不重复付钱，reset/播种不付钱。
- `acquire`：broom接近cube的contact potential首次改善。
- `push=4*assisted_delta`：只有同时满足接触质量、正确后方/横向/高度和入口corridor的inward progress才付钱。
- `pan_quality`：pan level、mouth clearance、低线速度和低角速度的联合改善。
- `success_quality=2*pan_quality*new_Gate4`：只在真正fully-inside当步奖励稳定pan。
- `track`：confidence-weighted工具reference偏差惩罚，绝不是success proxy。
- `shape`：第9节的human direction奖励。
- `action=-0.002||a_t||²-0.001||a_t-a_{t-1}||²`。
- `left_anchor=-0.001||left_cum_res/left_dev_hi||²`。

mouth-floor防止策略把簸箕压入桌面后廉价终止：

```text
mouth_floor = -4 * relu((0.0005 - clearance) / 0.0035)^2
```

clearance低于`+0.5 mm`开始连续惩罚，低于`-3.0 mm`硬失败。该项允许极轻微接触，但严重下压不能获得正收益。

## 12. 无离线专家预热初始化

当前完整方法必须满足：

```text
warmup_role = none
actor_epochs = 0
critic_epochs = 0
actor_samples = 0
critic_samples = 0
warmup_transitions = []
BC checkpoint = absent
```

因此不执行Actor BC、不执行离线Critic return regression，也不用transition拟合observation normalization。CLI为兼容旧命令仍接受expert文件路径，但`warmup_role=none`分支不得调用`warm_actor_critic`，这些路径不能影响初始化。

历史 `Sweep2FloorPenaltyV4__20260901` 曾使用expert BC和离线Critic预热；该事实只解释旧24M checkpoint来源，不是当前算法；旧轨迹专家预热指的是先在isaac通过零回放得到一条成功的轨迹，并进行actor和crtic预热训练，例如该轨迹的生成：/home/msc-auto/RL_sweep/outputs_video/sweep2_expert_entry25_v1.mp4。

## 13. 在线 PPO

- 1024 environments；horizon 32；每个完整rollout epoch产生32,768 agent steps。
- `gamma=0.99`；GAE `lambda=0.95`。
- minibatch 8192；每批rollout执行5个mini-epochs。
- PPO clip `epsilon=0.2`；value使用clipped loss。
- Adam learning rate最高`3e-4`、最低`3e-5`；KL threshold 0.02。
- entropy coefficient 0.001；bounds coefficient 0.0001；gradient norm clip 1.0。
- Gaussian policy sigma初始0.05，floor 0.02。

epoch 0–9仍采集当前随机policy的在线rollout，但只更新Critic/value参数；Actor、entropy、bounds和LR调度冻结。这10个epoch的目的是让value先适配当前在线分布，不是expert预热。epoch 10起执行完整PPO：

```text
32-step on-policy rollout
 -> bootstrap value + GAE
 -> 只用 actor_mask=1 样本计算 advantage normalization
 -> 5 × minibatch clipped actor/value update
 -> entropy + bounds regularization
 -> gradient clip + optimizer step
 -> epoch metrics / Gate funnel / reward terms / residual usage
```

## 14. Checkpoint、录像与评测

训练必须按启动前锁定的周期保存 `checkpoint.pth`、`metrics.json`、`rollout.npz` 和 deterministic video；具体训练预算与保存节奏写入对应run的启动记录，不属于算法定义。
- recorder视角为机器人左前方略高的中景，覆盖上半身、双臂和桌面操作区。
- terminal时抑制DirectRLEnv auto-reset，渲染Gate4成立当步的真实physics state；随后重复该终态40帧，在20 FPS下冻结2秒。
- terminal前一帧或reset后画面都不能作为成功录像。

训练节点的`sr/gate4`只描述该窗口完成episodes，不是最终论文数字。候选checkpoint必须运行至少512个deterministic episodes，并报告：Gate1→Gate4 funnel、success/failure数、成功步数/时间与效率分布、mouth clearance分布和最小值、terminal cube-pan几何、失败Gate、穿桌/提前终止/浅进入误判、seed、world/config、checkpoint SHA-256和输出路径。只有`fully_inside rate >= 0.50`通过最终验收。

## 15. 权威代码、运行指纹与安全边界

- reference builder：`tasks/Sweep/2/A_Design/L2_Reference/build_reference.py`
- tracker：`tasks/Sweep/2/A_Design/L3_Learning/progress_batch.py`
- environment：`tasks/Sweep/2/C_Wiring/sweep_env.py`
- train：`tasks/Sweep/2/C_Wiring/train_sweep.py`
- PPO config：`tasks/Sweep/2/C_Wiring/ppo_sweep.yaml`
- shared PPO masking：`rl_rebuild/algo/ppo/ppo.py`、`rl_rebuild/algo/ppo/experience.py`
- recorder：`tasks/Sweep/2/C_Wiring/record_sweep.py`
- evaluator：`tasks/Sweep/2/C_Wiring/eval_sweep.py`

每个run的`world.json`必须冻结reference/asset hash、policy I/O、success contract、timebase、confidence/human配置和warmup摘要。判断checkpoint世界只能看其`world.json`和hash，不能凭目录名推断。

共享服务器上只能操作本任务明确拥有的tmux、PID和目录；禁止影响feiyang、kailang或其他用户任务，禁止`pkill`、`killall`、GPU reset和广域终止。不得自动resume或启动额外1024-env run。GitHub只能使用`github-harroldx`且身份必须为`HARROLDX`；未经授权不commit、不push。不得触碰现有`tasks/pregrasp/arm_shell_points.npz`修改。

## 16. 后续完整方法多轨迹计划

用户提供新数据的机器人GraspPose后，从`datasets/sweep_new_data/`筛选4条左手簸箕基本静止、右手扫把扫动且物体trajectory confidence较高的数据，与当前Sweep2组成5条。五条均使用本文完整方法，不为不同轨迹修改算法；但每条仍须独立生成reference并通过Playbook规定的GraspPose、IK、物理world、zero-residual、random smoke与录像检查。
