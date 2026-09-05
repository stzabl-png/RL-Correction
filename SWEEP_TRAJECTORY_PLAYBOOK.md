# Sweep 重建轨迹通用处理与训练手册

本手册记录已经在 Sweep 任务中验证过、可迁移到未来其他 Sweep 轨迹的方法。它回答“拿到一条新轨迹后应按什么顺序做、每一阶段凭什么放行”。具体任务的坐标、文件路径、run 名、checkpoint 和实时状态应写入该任务的 `Codex_tasks.md`，不得写成本手册的固定前提。

## 一、适用边界与核心原则

每条新 Sweep 轨迹必须独立生成 reference、policy 和评测结果，并独立验证 GraspPose、IK 与物理世界。当前默认方法不要求生成 expert 或 transition；可以复用方法、代码结构和检查标准，但不能直接复制另一条轨迹的 joint action、cube 坐标、IK 结果或 checkpoint。

通用顺序：

```text
重建输入审计
  -> reference 与 IK 出厂检查
  -> 物理世界和零残差回放
  -> zero-residual 与 random-policy 小环境验证
  -> 随机初始化 Actor/Critic
  -> 前 10 个在线 critic-only PPO epochs
  -> 多环境 PPO
  -> 周期诊断
  -> deterministic 最终验收
```

每一阶段必须用可观测证据放行。reward、命令位移、reference row 或视觉上的“差不多”都不能替代真实物理几何。

## 二、输入审计

新轨迹至少应提供：ego 视频、工具/物体 6DoF trajectory、逐帧 confidence、左右手 trajectory、broom/dustpan GraspPose 和工具 mesh。

首先记录：

- 每个输入的路径、shape、dtype、帧率、时间戳和 hash。
- 世界坐标、相机坐标、工具坐标与机器人 base 坐标的定义。
- reconstruction frame、simulation control step 和 video frame 的映射。
- confidence 的来源、范围、缺失值与平滑方式。

输入契约不明确时，不进入 IK 或物理调参。

## 三、Reference 与 IK

使用工具 6DoF 与 GraspPose 建立 wrist-tool 刚性关系，对左右工具目标逐帧做 7DoF arm IK，生成双臂 joint reference。手指保持固定抓取姿态。

reference 出厂检查：

- 长度和 timebase 与源轨迹一致。
- 四元数符号连续，无姿态翻转。
- IK reachable rate、joint limits 和相邻 joint jump 合格。
- 首帧 wrist-tool 与 GraspPose 一致。
- 双臂、身体和工具无明显穿模。
- confidence 能正确映射到 residual envelope。

human trajectory 不宜直接变成绝对腕位目标；已经验证的做法是在中低 confidence 区域只提供相邻运动方向 shape prior，避免重建误差强拉机器人。

## 四、物理世界

建立固定桌面、机器人、固定 finger GraspPose、右手 broom、左手 dustpan 和任务物体。工具通过真实物理约束固定，reset 后位置误差应小于 3 mm。

dustpan collider 应使用开放 compound 结构：盆底、连续入口 ramp、左右侧壁和后壁。不要使用封住入口的单一 convex hull。必须验证入口不穿桌、reset 不击飞物体、多环境 clone 只访问自身 subtree。

更换轨迹时必须重新测量工具姿态、pan mouth frame、承载高度和物体初始相对位置，不能沿用另一任务的固定坐标。

## 五、零残差回放

先完整运行 zero-residual physical replay，记录：

- reset 冲击与 FixedJoint 稳定性。
- broom 工作面与 cube 的接近、接触和推动。
- dustpan mouth 的高度、朝向和桌面 clearance。
- cube 在 pan frame 中的 x/y/z 轨迹。
- 首个失败阶段、最小几何缺口和 terminal 原因。

零 residual 可以失败，但失败必须可解释。禁止通过瞬移物体、扩大物体、探针播种或放宽成功判据伪装成 expert。

## 六、Expert 生成方法（可选诊断/历史复现）

当前完整方法不使用 expert trajectory 做 Actor BC、离线 Critic 回归或 observation normalization。Expert 工具仅保留用于历史复现或可选物理诊断，不是新 Sweep 的训练前置条件。

## 七、已验证案例：Sweep2 的 25/40 mm expert

历史 Sweep2 v4 曾构造 25/40 mm 末段扩展，并用两条 fully-inside success、25 mm near-success 和 canonical failure 完成 Actor/Critic 离线预热。该案例只保留为旧 checkpoint provenance；当前默认方法不再生成或消费这些 expert transition。

## 八、成功契约

建议在 pan frame 中定义 mouth plane、横向宽度、承载高度和后端边界。至少区分：

- Gate1：场景 ready。
- Gate2：工具接近且物体真实移动。
- Gate3 `entered`：中心跨过 mouth，表示浅进入。
- Gate4 `fully_inside`：完整 footprint 清过 mouth，才是 operational success。

Gate 必须 earn-only 锁存。success 当步终止物理 rollout。更换物体尺寸、dustpan 几何或任务目标时，要重新测量所有绝对阈值；不能跨任务照搬毫米数。

## 九、Transition、Actor 与 Critic

当前默认流程从随机网络开始：不执行 Actor BC、离线 Critic return regression 或 transition observation normalization。前 10 个 PPO epochs 使用当前 policy 的在线 rollout 且只更新 Critic，之后启用完整 PPO；这段在线保护期不属于 expert warmup。

旧 v4 的 transition/BC/Critic 分工已被无预热流程整体取代，只在复现旧 checkpoint 时查阅历史台账。

## 十、Reward 设计

推荐组合：

- earn-only mouth progress。
- 从 entered 到 fully_inside 的 earn-only full-progress。
- Gate 首次奖励。
- broom 接近/推动质量。
- pan level/clear/still 质量。
- confidence tracking 和 human shape prior。
- action、smoothness 与非任务臂 anchor penalty。

两个 progress 都遵守：使用绝对几何量、只奖励历史最大值增量、reset/播种不付钱。任何 shaped reward 都不能替代 success Gate。

## 十一、已验证案例：限制簸箕 mouth 穿桌捷径

一个已经出现过且很容易误判的失败模式是：训练 reward 持续升高，但 Gate4 成功率下降；deterministic 策略在 Gate2 后把簸箕 mouth 压入桌面，短回合仍从 acquired、task、shape 等正奖励中获利。只放宽穿桌 terminal 阈值不能解决问题，因为它给策略留下了“先吃正奖励、再廉价终止”的路径。

本次 Sweep2 采用连续路径惩罚与硬边界组合：mouth clearance 低于 `+0.5 mm` 时施加

```text
mouth_floor = -4 * relu((0.0005 - clearance) / 0.0035)^2
```

并在低于 `-3.0 mm` 时硬失败。离线把该公式重算到既有 3M/6M 失败 rollout 后，总 reward 分别由 `+2.539 / +3.090` 变为 `-0.084 / -0.753`，先证明捷径不再赚钱，再发射正式训练。新 run 的 6M deterministic rollout 在 step 299 达成 fully-inside，最小 clearance `+1.34 mm`；训练窗口 Gate4 从 3M 的 `8.74%` 提升到 6M 的 `69.37%`，6M–7M 均值约 `72.2%`。

迁移到其他 Sweep 任务时应复用方法而不是照抄毫米数：重新测量桌面、簸箕 mouth 和接触 offset，确定软惩罚起点、增长区间与不可接受硬下限；用至少一条已知失败 rollout 离线重算 reward，并要求失败捷径的总回报显著低于真正成功。reward 改变后原则上应重采 Critic return；本案例复用旧 transition 是用户明确接受的实验例外，不应自动推广。

## 十二、小规模验证与训练放大

当前默认流程按以下顺序放行：

1. Python/static check 与 CPU tracker self-test。
2. 一环境 zero-residual replay，检查 reference、接触、Gate 与终止状态。
3. 一环境 random-policy smoke，检查 reset、action bound、NaN 和日志路径。
4. 核对无预热指纹：`warmup_role=none`、Actor/Critic samples 均为 0、无 BC checkpoint。
5. 再启动多环境 PPO，并使用独立 tmux、日志、checkpoint 和 artifact prefix；前 10 个 epoch 只在线更新 Critic。

在共享 GPU 上，只能操作本任务明确拥有的 session、PID 和目录。网络断开后先检查已有 tmux，不得因看不到原 shell 就重复启动。

## 十三、诊断、录像与最终验收

每个周期节点至少保存 checkpoint、metrics、rollout、record log、policy video 和连续 context frames。优先检查 Gate funnel、terminal step、mouth collision、cube-pan 几何、push/pan quality 和 residual 使用，而不是只看 reward。

录像必须显示真实 terminal physics state。若环境会在 `env.step` 返回前自动 reset，应在专用 recorder 中抑制该次 reset 后再渲染；terminal 前帧和 reset 后帧都不能作为成功终态。展示冻结只能重复终态图像，不能继续施加动作或仿真。

最终验收使用足量 deterministic episodes，并预先写明成功率门槛。训练窗口成功率、单条成功视频或 expert 回放只用于诊断，不能替代最终统计验收。

旧 run 的清理必须发生在诊断结论写入台账之后。先逐项确认目录不被当前训练、expert manifest、checkpoint 回退或论文证据引用；保留当前 run、关键节点诊断、expert 数据及来源证据，只删除已被取代且不再引用的日志树。

## 十四、新 Sweep 轨迹迁移清单

1. 建立新的任务目录和独立 `Codex_tasks.md` 实例段，记录输入与坐标系。
2. 重新生成 reference，完成 IK 与 GraspPose 出厂检查。
3. 重建并验证开放 dustpan 物理世界，完成 zero-residual replay。
4. 完成 zero-residual 与 random-policy smoke，并核对无预热指纹。
5. 使用新的 tmux/run/checkpoint/video 前缀，从随机 Actor/Critic 启动训练。
6. 周期检查 Gate funnel 和真实 terminal 视频，最后执行 deterministic 批量验收。

当前后续计划是在用户补充机器人 GraspPose 后，从 `datasets/sweep_new_data/` 选择 4 条与 Sweep2 操作模式高度相似、物体 confidence 较高的轨迹，与当前轨迹组成 5 条数据。五条使用同一套无专家预热完整方法。

任何阶段若发现坐标、资产、接触语义或成功判据错误，应退回对应阶段修正，不要在 reward 或 PPO 超参数上掩盖上游问题。
