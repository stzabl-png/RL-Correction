# Sweep 重建轨迹通用处理与 Entry-Success 训练手册

本手册描述当前有效的 Sweep residual-RL 流程。每条新重建轨迹单独生成 reference、专家、transition、policy 和评测；不得直接复制 Sweep2 动作或 checkpoint。Deep20 是已停止的历史扩展，不属于当前成功标准。

## 一、输入契约

每条轨迹至少包含 ego MP4、工具/物体 6DoF trajectory、逐帧 confidence、左右手 trajectory、broom/dustpan GraspPose 和工具 mesh。必须记录路径、shape、帧率、坐标系和 hash，严格区分 reconstruction row、simulation control step 和 video frame。

## 二、Reference

build_reference.py 用工具 6DoF 与 GraspPose 建立 wrist-tool 刚性关系，对左右工具目标逐帧做 7DoF arm IK，生成双臂 joint reference。confidence 决定 residual envelope；human trajectory 不做绝对腕位追踪，只在中低 confidence 段形成相邻运动方向 shape prior。

出厂检查包括：长度/timebase、四元数连续性、IK reachable rate、joint limits、相邻 jump、首帧 GraspPose 一致性、双臂碰撞与工具姿态。

## 三、物理世界

固定桌面、DexMate、固定 finger GraspPose、右手 broom、左手 dustpan、固定起点 cube。工具通过两个物理 FixedJoint 固定；reset 后位置误差要求小于 3 mm。

dustpan 使用视觉 mesh 加开放 compound collider：盆底、连续入口 ramp、左右侧壁、后壁。不可用封住入口的 convex hull。入口不能穿桌或在 reset 时击飞 cube。多环境只遍历当前 env subtree，并复用 clone xform。

## 四、零残差回放

先运行完整 reference 的 zero-residual physical replay，检查 reset 冲击、工具姿态、入口高度、broom 工作面、cube-pan 相对轨迹和失败阶段。零 residual 可以失败，但失败必须可解释，且不能通过瞬移 cube、放大 cube 或修改判据伪装成功。

## 五、专家

以最接近成功的回放为基线，一次只改一个变量。资产正确后，保持左臂/reference/世界不变，只对右臂末段沿 pan-local inward axis 做平滑 IK depth extension。

当前 Sweep2 保存：

- 25 mm entry expert。
- 40 mm entry expert。
- canonical failure。
- Actor 只 BC 40 mm。
- Critic 使用 25/40 success 与 failure。
- 所有当前 transition 位于 logs/expert/transitions/。

专家第一次 entered 即可终止。broom 可先推动，随后由 pan 相对运动完成 entry；success 由 cube-pan 几何决定，不强制持续扫把接触。

## 六、成功契约

pan frame 中：

- x：入口横向。
- +y：簸箕承载面法向。
- +z：从 handle 指向 mouth。
- cube 向 -z 进入。

entered 同时要求：

- cube 横向 footprint 在 pan width 内。
- cube centre z 已跨过 mouth plane，但未穿过盆后端下界。
- cube centre y 位于实测承载高度范围。
- Gate1 ready 与 Gate2 moved/broom-near 已先成立。

Gate3 在首次 entered 锁存，Gate4 同步锁存为 success，当步立即终止。fully_inside、deep_inside 和 deep_margin 可继续输出用于分析，但不决定奖励或验收。

## 七、Observation 与 Action

Actor observation：191 维。Critic privileged state：22 维。Action：左右臂 14 个 joint residual；手指固定。

每步：

1. Actor/Critic 读取状态。
2. 前 80 步 action 强制为零。
3. 当前 row confidence 生成 step bound 和 cumulative bound。
4. cumulative residual 加到 reference arm。
5. Isaac 执行 physics substeps。
6. 由真实 cube/tool 状态计算 reward、Gate、failure 和 next observation。

reference row 在 nominal contact 前 open-loop 前进；之后只有 broom near 或 cube moved 达标才继续，避免任务时钟脱离物体。

## 八、训练

1. 用 40 mm transition 训练 Actor；非零修正帧权重 1，零前缀权重 0.05。
2. 用 25/40/failure 的 discounted return 预热 Critic。
3. PPO 前若配置 critic-only warmup，Actor 保持冻结。
4. 随后 pure on-policy PPO。
5. scripted prelude transition 可训练 Critic，但 actor_mask=0，因此不进入 Actor loss、entropy、bounds、KL 或 Actor advantage normalization。
6. 训练每 3M agent steps 生成不可变诊断节点。

当前不使用 logs/expert/transitions_deep20/，不恢复旧 checkpoint，从随机网络开始 warmup。

## 九、Reward 与失败

Reward 只服务 entry 目标：

- earn-only mouth progress。
- Gate 首次奖励。
- broom 接近/推动质量。
- pan level/clear/still 质量。
- confidence tracking。
- human shape prior。
- action/smoothness 与 left-arm anchor penalty。

Deep20 progress 不进入 task reward。失败包括 cube 掉下桌面和 dustpan mouth 明显穿桌。任何 reward 上升都不能替代 Gate 和真实几何诊断。

## 十、录像

训练成功当步终止，不为了视频继续仿真。DirectRLEnv 可能在 terminal env.step 返回前自动 reset，所以 recorder 必须：

1. 保存 terminal transition 到 trace。
2. 不渲染 done 返回后的 reset 状态。
3. 使用上一张有效 pre-reset RGB 图像作为展示终帧。
4. 在 MP4 末尾追加 40 个相同源帧，20 FPS 下为 2 秒。
5. topdown context 的最后文件必须是有效物理帧。

15M 回归的正确最后画面是 frame_0231.png；frame_0232 是旧实现误录的 reset 状态，已禁止。

## 十一、诊断与验收

每 3M 节点至少包含 checkpoint.pth、metrics.json、rollout.npz、record.log、policy.mp4 和 topdown_frames。

优先检查：

- Gate1->Gate2->Gate3/4 funnel。
- terminal step 分布。
- mouth clearance failure。
- cube_pan、push、pan quality。
- Actor/left residual usage。
- deterministic terminal trace 与冻结画面。

候选 checkpoint 必须运行至少 512 deterministic episodes，entry-success rate >= 0.50。训练窗口 rate 只用于筛选。

## 十二、Sweep2 当前实例

当前 v2 run：

- tmux：sweep2_entryrestore_v2_1024_20260831
- run：logs/Sweep2_entryrestore_v2_fixed1024_seed42_20260831/
- artifact：Sweep2EntryRestoreV2__20260831_policy
- envs=1024，seed=42，max=100M
- old-entry transitions：logs/expert/transitions/
- 每 3M 自动诊断
- 最终 512 deterministic 验收

旧 15M 回归于 terminal transition step232 成功，最后有效图像为 frame_0231.png，视频追加 40 帧冻结。该回归证明恢复的 entry contract 与旧成功 checkpoint 一致，不代表当前 v2 已完成训练。
