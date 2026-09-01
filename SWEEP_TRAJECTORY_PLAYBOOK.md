# Sweep 重建轨迹通用处理与训练手册

本手册记录已经在 Sweep 任务中验证过、可迁移到未来其他 Sweep 轨迹的方法。它回答“拿到一条新轨迹后应按什么顺序做、每一阶段凭什么放行”。具体任务的坐标、文件路径、run 名、checkpoint 和实时状态应写入该任务的 `Codex_tasks.md`，不得写成本手册的固定前提。

## 一、适用边界与核心原则

每条新 Sweep 轨迹必须独立生成 reference、expert、transition、policy 和评测结果。可以复用方法、代码结构和检查标准，但不能直接复制另一条轨迹的 joint action、cube 坐标、IK 结果或 checkpoint。

通用顺序：

```text
重建输入审计
  -> reference 与 IK 出厂检查
  -> 物理世界和零残差回放
  -> 单变量 expert 扩展
  -> success/near-success/failure 数据定级
  -> transition 重采样与 Actor/Critic 预热
  -> 小环境验证
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

## 六、Expert 生成方法

从最接近成功且物理可信的回放开始，一次只改一个变量。已经验证有效的优先做法是：保持世界、左臂、reference 前段和接触语义不变，只在右臂末段沿 pan-local inward axis 增加平滑 IK depth extension。

推荐流程：

1. 选定 source-relative 接触区间，明确行号和时间映射。
2. 对右臂末段加入从 0 平滑增长到目标深度的位移；不要突然跳变。
3. 重新做逐帧 IK，检查 joint limits、最大相邻角变化和末端实际位移。
4. 在同一个冻结世界中做一环境 deterministic replay。
5. 同时保存 action、物理 state、Gate、cube-pan 几何、视频和日志。
6. 按当前成功契约定级为 `full_success`、`near_success` 或 `failure`，不能按脚本名称定级。

命令的 25/40 mm 是末端目标扩展量，不等于 cube 的实际入盆深度。IK、碰撞、刷毛接触和 pan 运动都会改变最终物理结果，因此每个候选都必须重新回放测量。

## 七、已验证案例：Sweep2 的 25/40 mm expert

本节只展示通用方法如何落地，不把 Sweep2 的固定路径或运行状态当作未来任务默认值。

形成过程：

1. 先完成固定 cube、开放 dustpan collider、双 FixedJoint 和 zero-residual replay，确认主要剩余缺口来自末段向 pan 内推进不足，而不是资产封口、reset 冲击或坐标系错误。
2. 冻结左臂、世界和 source-relative reference，只修改右臂末段；沿 pan-local inward axis 分别构造 25 mm 与 40 mm 平滑 depth extension。
3. 每条候选重新进行 IK 和一环境物理回放。25 mm 候选保留了有效接触并显著缩小入口缺口；40 mm 候选在不越过 URDF joint limits、最大相邻关节变化约 3° 的情况下提供更深推进。
4. 在当时的 entry 判据下，正式回放分别在 frame 384 和 frame 382 达到目标，产出 `sweep2_expert_entry25_v1` 与 `sweep2_expert_entry40_v1` 的 trace、metrics、NPZ 和视频。
5. 成功契约后来收紧为完整 footprint `fully_inside` 后，不能沿用旧标签：按新 reward 和几何重新采样 transition，25 mm 被重新定级为 `near_success`（Gate3 entered，但没有 Gate4），40 mm 被验证为 `full_success`。
6. 当前训练案例中，40 mm 与另一条 independently verified fully-inside expert 共同用于 Actor BC；两条 full-success、25 mm near-success 和 canonical failure 一起用于 Critic return regression。

这个案例验证了三条可迁移结论：

- 用同一 source-relative 轨迹做 25/40 mm 单变量对照，可以判断是否主要缺少 inward depth。
- commanded depth 只是干预量，expert 标签必须由统一环境中的真实 terminal 几何决定。
- 成功契约一旦变化，所有旧 expert 必须重新定级并重采 transition；不能只改 manifest 标签。

未来轨迹不要求仍使用 25/40 mm。应先测量其实际几何缺口，再选择小/大两个安全深度形成鉴别实验。

## 八、成功契约

建议在 pan frame 中定义 mouth plane、横向宽度、承载高度和后端边界。至少区分：

- Gate1：场景 ready。
- Gate2：工具接近且物体真实移动。
- Gate3 `entered`：中心跨过 mouth，表示浅进入。
- Gate4 `fully_inside`：完整 footprint 清过 mouth，才是 operational success。

Gate 必须 earn-only 锁存。success 当步终止物理 rollout。更换物体尺寸、dustpan 几何或任务目标时，要重新测量所有绝对阈值；不能跨任务照搬毫米数。

## 九、Transition、Actor 与 Critic

把每条已定级 rollout 在当前环境和 reward 下重采为 transition，保存 observation、privileged state、action、reward、done、Gate、return、actor mask、来源 hash 和数据角色。

已经验证的分工：

- Actor BC 只学习 `full_success` expert。
- `near_success` 不进入 Actor BC，避免策略模仿停在 Gate3 的动作。
- Critic 同时学习 full-success、near-success 和 failure 的 discounted return，获得更完整的价值排序。
- scripted reference-only 前缀可以训练 Critic，但必须从 Actor loss、entropy、bounds、KL 和 Actor advantage normalization 中排除。
- 预热结束后进入 pure on-policy PPO，不在 PPO rollout 中偷偷混入 scripted expert action。

数据 manifest 应记录 schema、维度、样本数、角色、success frame、源/输出 SHA-256。训练代码必须按角色读取，不能依赖文件名猜测。

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

按以下顺序放行：

1. Python/static check 与 CPU tracker self-test。
2. 一环境 deterministic expert replay。
3. transition schema、shape、mask、return 和 hash 检查。
4. 一环境 random-policy smoke，检查 reset、action bound、NaN 和日志路径。
5. 小批量 Actor/Critic warmup，检查 loss 与参数更新对象。
6. 再启动多环境 PPO，并使用独立 tmux、日志、checkpoint 和 artifact prefix。

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
4. 根据实测缺口设计小/大 inward-depth 单变量 expert 对照。
5. 按该任务成功契约重新定级、重采 transition，并验证 manifest。
6. 依次通过一环境 expert、transition、random smoke 和 warmup 门。
7. 使用新的 tmux/run/checkpoint/video 前缀启动训练。
8. 周期检查 Gate funnel 和真实 terminal 视频，最后执行 deterministic 批量验收。

任何阶段若发现坐标、资产、接触语义或成功判据错误，应退回对应阶段修正，不要在 reward 或 PPO 超参数上掩盖上游问题。
