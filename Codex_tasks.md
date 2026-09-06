# Sweep2 单轨迹成功算法：原理、数据处理与运行手册

更新：2026-09-06。根目录：/home/msc-auto/RL_sweep，服务器：msc-a6000，分支：task_sweep。

本文合并原通用处理手册，集中解释成功 Sweep2 单轨迹“数据怎么变成机器人参考、策略究竟学什么、如何训练、如何得到可信录像”。具体多轨迹接入进度全部见 Codex_new_data.md。文中的训练命令是复现说明，本次整理没有启动训练或恢复作业。

## 先用一个例子理解整个方法

人类视频告诉我们扫帚和簸箕大致如何运动，但它不是机器人能直接执行的关节命令。第一阶段用重建工具位姿、工具网格及机器人抓姿，求出机器人每一时刻应该如何摆放双臂。这得到一条固定的 nominal reference，即“原计划”。

第二阶段在真实物理仿真中执行原计划。重建误差、物体接触和执行滞后可能使原计划无法把方块完整扫入簸箕。因此 PPO 学习在原计划附近逐步修正双臂关节。策略每次只输出14个小修正量，不重新生成整条动作，也不直接控制工具世界坐标。机器人通过关节驱动运动，手与工具通过 FixedJoint 相连；方块仍需被真实接触推动。

例如右臂某关节当前 reference 是0.50rad，已有累计修正0.02rad，本步动作0.5、允许增量0.008rad，则新累计修正是0.024rad，关节目标0.524rad。下一步在这个累计值上继续更新，而不是重新从零开始。confidence 高时更相信重建、限制修正；confidence 低时允许较大修正，同时人手运动方向给出辅助引导。

这里有三种不同的“正确”：IK误差小表示腕目标可达；零残差录像合理表示资产和物理执行链合理；训练后的 Gate4 表示方块真的完整进入簸箕。前一种不能自动证明后一种。下文分别给出检查方法。

## 阅读路线

第一次阅读：先读第1–5节建立数据与物理概念，再读第8–13节理解动作、奖励和PPO，最后按第16–19节查看准备、运行与结果。第6–7节是张量接口字典，修改网络或观测时再逐项核对。

所有未写绝对路径的代码、输入、日志路径均相对 /home/msc-auto/RL_sweep。方法以对应实现及保存的 world.json 为证据；旧 checkpoint 的来源不能靠名字猜测。

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


### 3.1 先把输入放到同一个时钟和坐标系

原视频用于检查人真正抓在哪里、刷毛朝哪边；poseqa中的工具6DoF才是reference的运动输入；GraspPose是机器人手相对于工具的固定姿态；人手轨迹提供另一路方向先验。这四类数据不能互相替代。

builder读取 poseqa/rts_sweep_dustpan_2_object_0.npz 和 object_1.npz 的 object_ob_in_world_smooth，读取 replay_world 的fps及左右手轨迹。当前实现显式检查重建时钟15Hz，机器人控制时钟20Hz。视频帧号不能直接当作control row。大位移/大旋转所在的源关键帧会保留；必要时在两个源时刻之间插值并延长执行时间，不删动作、不压缩旋转幅度。位置插值与旋转插值共同形成工具目标序列。

把源场景映射到机器人桌面时使用一套共享的刚体注册：两个工具用相同变换，保留它们原来的相对运动。坐标转换是解释同一条轨迹；为了使IK更容易而单独旋转其中一个工具，则已经改了任务。

### 3.2 GraspPose 为什么能把工具目标变成腕目标

设 T_world_object(t) 是工具每帧世界位姿，T_object_hand 是prior中固定的手相对工具位姿，则：

    T_world_hand(t) = T_world_object(t) × T_object_hand

这里的乘法同时变换位置和姿态。prior的前3维是平移，后4维是wxyz四元数，后续是固定手指关节角。它不是世界坐标中的腕位置。候选若处于canonical坐标，必须先正确还原到input mesh坐标，不能直接把候选数组当作prior。

完整核对链是 candidate canonical → input OBJ → USD mesh/rigid root → world。原点偏移只加到位置，法向和旋转不加平移。具体新的数据转换问题由 Codex_new_data.md 记录；单轨迹的已验证prior不能被新数据覆盖。

### 3.3 双臂IK不是只解第一帧

当前 solve_continuous 先每隔10行建立一组关键帧，多初值搜集冗余机械臂的可行解；再用动态规划选相邻关节变化较小的分支。随后PCHIP产生密集初值，逐行用数值IK细化，同时考虑前一帧解，减少跨分支跳变。

必须分别核对可达率、FK位置/姿态误差、关节限位和最大相邻步长。“每一帧都有解”仍可能在相邻帧从限位一端跳到另一端。有限位关节不能简单做角度unwrap伪装连续。失败应回到抓姿、注册、源数据检查，而不是修改PPO遮掩。

human_arm也通过机器人运动学得到，但进入奖励的是相邻关节差分方向，不是把重建人手绝对腕位强加给机器人。reference保存了工具运动与human辅助信息两条数据流。

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


### 13.1 一轮训练实际在做什么

1. 每个环境读当前观测，Actor给出高斯动作分布；训练时采样，确定性录像时使用策略均值。环境执行有界累计residual。
2. 物理推进后得到reward、done和下一观测，同时保存动作旧log probability、value及Actor mask。1024个环境各收32步，共32768个agent steps；agent steps不是视频帧数。
3. Critic估计未来折扣回报。GAE把实际reward与value误差沿时间回传，得到advantage：这个动作比当时预期更好还是更差。
4. PPO比较同一批动作在新旧策略下的概率比，clip限制一次更新幅度。它提高正advantage动作的概率、降低负advantage动作的概率；Critic则拟合回报目标。
5. 同一批数据做5轮minibatch更新后丢弃，重新用当前策略采集。它不是反复训练一个固定专家数据集。
6. 前80步执行动作被强制为零，不应让Actor因自己没执行的动作获奖或受罚。因此Actor各项统计要mask；Critic仍需学习整个episode的回报。
7. 前10个epoch只更新Critic，让价值估计先适应当前在线状态分布。这不同于前80个control steps：前者是训练优化器的阶段，后者是每次episode的物理动作前缀。

低confidence只改变“可修正多少”和“参考偏差容忍度”，不能直接让success为真；human shaping也只是一项较小的方向奖励。真正的任务信号始终来自方块相对簸箕的真实几何。

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


## 16. 从输入到训练的操作顺序（合并后的通用流程）

| 阶段 | 做什么、为什么 | 应保留的产物/放行依据 |
|---|---|---|
| 输入审计 | 对拍ego、重建render、候选抓姿；记录shape、fps、单位、hash与工具语义 | 输入清单、抓姿图、坐标定义；不凭候选排名验收 |
| prior/资产 | 将候选转换到input坐标；修可见mesh本体；生成独立USD与开放簸箕碰撞体 | prior、OBJ/USD、坐标闭环、抓区/拓扑检查 |
| reference | 共享场景注册、时间插值、双臂IK、human方向与confidence对齐 | NPZ、源帧映射、FK/限位/步长报告 |
| zero-residual | 不加载策略，真实关节驱动完整回放 | 首中末帧、完整视频、实际物理trace；允许任务失败但原因可解释 |
| 任务几何 | 在pan frame测量入口、承载高度、cube尺寸与起点 | world指纹；不能把一个实例的毫米数直接用于另一个实例 |
| 小规模验证 | tracker CPU检查、1环境zero/random smoke | reset、bounds、NaN、Gate及终态录像正常 |
| 训练 | full、随机Actor/Critic、无离线预热、在线PPO | 唯一run/tmux、日志、world.json、checkpoint、metrics |
| 诊断与评测 | 看Gate funnel、物理几何及确定性录像 | 成功终态证据；最终批量评测与训练窗口分开报告 |

reference、资产和checkpoint相互关联，不能拿另一轨迹的joint reference、cube起点或policy直接宣称迁移成功。可选expert只用于诊断或历史复现，不是当前训练前置步骤。

资产修复应优先保持抓取区域和工具坐标系；碰撞体必须与可见几何相符，不能用隐形几何掩盖错误。若改变实际任务几何，应重测绝对阈值，但保留算法机制。任何新实验仍需相应执行授权。

## 17. 单轨迹复现入口与命令

以下在服务器执行，命令相对根目录。先检查已有进程与GPU归属，再启动自己的唯一作业；本次文档整理不执行这些训练命令。

### 17.1 已验证输入、环境和准备入口

    cd /home/msc-auto/RL_sweep
    export PYTHONPATH=. SHARPA_WANDB=0
    export VEGA_URDF=/home/msc-auto/data/vega_urdf/vega_1p_sharpa.urdf
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

历史成功full run解释器是 /home/msc-auto/miniconda3/envs/isaac/bin/python，准确命令保存在 logs/Sweep2_ablation_nooffline_full_seed42_20260903/launch_train.sh。近期资产/轨迹诊断使用 /home/msc-auto/rlcorr-venv/bin/python（IsaacSim5.1）。不要仅因名字相似就交换运行环境；checkpoint旁world.json是世界指纹，环境与代码版本须一并核对。

成功输入为 datasets/sweep_2_better/；机器人抓姿在 tasks/pregrasp/priors/Sweep2_broom.npz 与 Sweep2_dustpan.npz；平滑簸箕生成脚本 tasks/Sweep/2/A_Design/build_smooth_dustpan_asset.py，成品 tasks/Sweep/2/assets/dustpan_smooth_entry/。原候选render见 datasets/sweep_2_better/sweep2_grasppose/demo_replay/。

已验证reference应优先直接复用并核对hash。重新构建用于检查时必须写新路径，避免覆盖成功基准，例如：

    /home/msc-auto/rlcorr-venv/bin/python tasks/Sweep/2/A_Design/L2_Reference/build_reference.py --output logs/sweep2_reference_rebuild_check.npz

该命令使用builder中Sweep2默认输入。当前工作树含后续接口改动，因此“命令能运行”不代表新输出与成功checkpoint完全一致；复现24M时使用冻结reference及其hash，不擅自重建替换。

CPU状态机检查：

    /home/msc-auto/rlcorr-venv/bin/python tasks/Sweep/2/A_Design/L3_Learning/selftest_progress.py

一环境物理/随机检查入口（第二条加--random才采样随机residual；128步覆盖80步前缀后的动作区间）：

    /home/msc-auto/rlcorr-venv/bin/python tasks/Sweep/2/C_Wiring/smoke_sweep.py --num_envs 1 --steps 128 --headless
    /home/msc-auto/rlcorr-venv/bin/python tasks/Sweep/2/C_Wiring/smoke_sweep.py --num_envs 1 --steps 128 --random --headless

zero-residual recorder不带checkpoint，例如写入新的独立诊断路径：

    /home/msc-auto/rlcorr-venv/bin/python tasks/Sweep/2/C_Wiring/record_sweep.py --out outputs_video/sweep2_review_new.mp4 --trace logs/sweep2_review_new.npz --headless --enable_cameras

### 17.2 已成功训练的完整命令如何解读

真实历史启动脚本：logs/Sweep2_ablation_nooffline_full_seed42_20260903/launch_train.sh。它指定GPU1、独立GPU锁、method full、1024环境、seed42、24M预算、离线epochs均0、6M录像；默认3M诊断。脚本中的旧run名是已有成果目录，不能直接重跑覆盖。

获准复现实验时，复制其参数到新的唯一run/name、artifact_prefix、日志和tmux，并按当时批准GPU设置CUDA_VISIBLE_DEVICES及RL_ISAAC_LOCK_DIR。CLI要求expert25/expert40/expert_full/failure路径，但full设置的warmup_role=none使其不进入训练预热；world.json中transitions存在hash不等于使用了它们，实际warmup_transitions必须为空。

checkpoint保存于 logs/checkpoints/Sweep2AblationNoOfflineFull__20260903_policy_0024M/checkpoint.pth，配套world.json记录 obs191/priv22/action14、control_dt .05、scripted80、full/confidence/human与无预热信息。重放或评测必须读取配套world，不能混用历史warmup v4的模型。

### 17.3 成功policy录像与批量评测

录像入口示例（新的输出路径避免覆盖已验收录像）：

    /home/msc-auto/miniconda3/envs/isaac/bin/python tasks/Sweep/2/C_Wiring/record_sweep.py --checkpoint logs/checkpoints/Sweep2AblationNoOfflineFull__20260903_policy_0024M/checkpoint.pth --method full --out outputs_video/sweep2_full24m_review_new.mp4 --trace logs/sweep2_full24m_review_new.npz --headless --enable_cameras

正式512回合协议入口：

    /home/msc-auto/miniconda3/envs/isaac/bin/python tasks/Sweep/2/C_Wiring/eval_sweep.py --checkpoint logs/checkpoints/Sweep2AblationNoOfflineFull__20260903_policy_0024M/checkpoint.pth --method full --protocol final512 --num_envs 512 --episodes 512 --out logs/sweep2_full24m_final512_new.json --headless

这些命令仍消耗共享计算资源，只有相应运行授权后执行。没有运行评测就不能补写“512回合通过”。

## 18. 当前成功证据与历史分界

当前完整方法是 Sweep2AblationNoOfflineFull__20260903：随机Actor/Critic、confidence与human开启，无expert离线预热。用户指定的成功24M录像是 outputs_video/Sweep2AblationNoOfflineFull__20260903_policy_0024M/policy.mp4。6M/12M/18M/24M录像也保留在相应目录。

18M训练窗口Gate4为85.71%（140回合），24M为80.42%（143回合）。这与单条成功录像一起证明该配置确实学到了扫入行为，但不能写成独立512回合成功率。用户此前取消固定50回合曲线评测；当前已报告的是训练窗口，未以这次文档整理补做最终测试。三组详细曲线及复合消融边界见 Codex_ablation.md。

历史 Sweep2FloorPenaltyV4__20260901 使用过expert BC和离线Critic预热，与当前full分开。其24M训练窗口86.09%、训练停止于26,116,096 steps只是历史记录；不得混入当前full结果。

历史穿桌reward漏洞：失败轨迹也能靠接近、形状等正奖励获利。加入连续mouth-floor惩罚和-3mm硬失败后，旧3M/6M失败路径离线重算总回报从+2.539/+3.090降为-.084/-.753。该经验是先证明失败捷径不再赚钱，再训练，不是冻结左臂或修改成功定义。原手册相关教训已合入第10–11节及本节。

## 19. 文档分工：查什么去哪里

| 文档/目录（相对根目录） | 唯一主要职责 |
|---|---|
| Codex_tasks.md（本文） | 单轨迹成功算法解释、通用处理步骤、训练/录像/评测运行记录；原通用手册已合并，仅维护本文 |
| Codex_new_data.md | Task3全部当前状态：32/80成功步骤、128/180失败原因、版本路径、验收、后续训练接入 |
| Codex_HANDOFF.md | 新窗口入口：应按什么顺序读、必须看什么、准确路径与当前边界；不复制长算法和过期探索过程 |
| Codex_mistakes.md | 修改时发生的重大错误、已证实根因、修正与防复发检查 |
| Codex_commit.md | Git/工程里程碑：提交主题、变更范围、验证、前一版本与回退定位；不代替git本身 |
| Codex_ablation.md | Sweep2主消融协议、曲线和解释边界 |
| Codex_random.md | 方块跨位置/随机化专项实验、配置与结果；不自动授权其他任务随机化 |
| README.md | 仓库总览及入口；通用项目介绍 |
| CLAUDE.md | 早期项目操作约定，包含其他机器/PreGrasp历史；不能覆盖当前Sweep路径和用户要求 |
| DEXMATE_JOINTS.md | 机器人关节命名与机构说明 |
| docs/MANUAL.md、docs/PREGRASP_MANUAL.md | 继承的基础环境/PreGrasp操作手册 |
| docs/TRAINING_DESIGN_GUIDE.md、docs/POUR*.md | Pour/Step4及训练方法来源；不是当前Sweep运行状态 |
| docs/FRAME_ALIGNMENT.md、docs/RECON_TO_ENV_ALIGNMENT.md、docs/WORLD_FINGERPRINT.md | 坐标对齐、输入到环境映射、世界指纹方法 |
| docs/DEPLOY_A6000.md、docs/DEPLOY_NEW_MACHINE.md | 服务器与环境部署方法 |
| docs/HANDOFF_*.md及其他历史设计文档 | 相应日期/任务的历史背景，按题名查阅；不作为当前停点 |
| tasks/Sweep/new_data/*REVIEW*.md | Task3底层历史审计来源；当前结论统一汇总在Codex_new_data.md |
| logs/ | 命令、数值审计、失败证据、训练日志；不是另一份当前状态文档 |
| outputs_video/ | 成功视频与最新任务进度；阅读目录内标注区分ego、静态检查、完整物理录像 |
| datasets/、tasks/**/assets、references、priors | 输入及运行依赖，不是可随意清理的临时文件 |

更新规则：算法/单轨迹运行写本文；新数据进度写Codex_new_data；新窗口导航写HANDOFF；重大错误写mistakes；有意义的Git更新写commit台账。一次变更无需复制到所有文档，互相引用即可。
