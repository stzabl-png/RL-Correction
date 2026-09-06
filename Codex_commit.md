# Sweep Residual RL 关键工程演进

更新时间：2026-09-03。本文件只保留影响当前算法、复现和安全边界的关键里程碑。逐次 probe、被推翻的中间方案和防复发细节见 `Codex_mistakes.md`；当前实例权威说明见 `Codex_tasks.md`。

## 2026-08-30 — Sweep2 reference 与物理任务建立

- 从 `datasets/sweep_2_better/` 的 ego reconstruction 读取双手、扫把、簸箕、物体轨迹和 confidence，并结合机器人 GraspPose 构建逐帧双臂 IK reference。
- 固定右手扫把、左手簸箕，策略只输出双臂 14 DoF 累计关节 residual；前 80 control steps 使用 scripted reference 且强制零 residual。
- 建立开放式 dustpan compound collider、双工具 FixedJoint、固定 cube 起点和统一 pan-local 几何。
- 通过 zero-residual、接触和入口 probe 排除了封口 collider、reset 冲击、错误坐标系和后重置状态误判。

## 2026-08-30 — Gate、reward 与训练入口

- 建立 Gate1 ready、Gate2 接近并真实移动、Gate3 entered、Gate4 success 的单调锁存状态机。
- reward 使用绝对几何的 earn-only progress，配合接触质量、pan quality、reference tracking、human direction shaping、动作和平稳性正则。
- Actor observation 为 191 维，Critic 额外读取 22 维 privileged state，action 为双臂 14 维。
- 加入 checkpoint 世界指纹、deterministic evaluation、真实 terminal recorder 和共享 GPU/tmux 安全边界。

## 2026-08-31 — 训练与录像可靠性修正

- 前 80 步 scripted prefix 从 Actor loss、entropy、bounds、KL 和 Actor advantage normalization 中排除，但仍进入 Critic return/GAE。
- recorder 在终止时抑制 auto-reset，保存 Gate4 成立当步的真实物理画面，并在 20 FPS 下重复40帧冻结2秒；terminal 前一帧和 reset 后画面都不允许冒充终态。
- checkpoint、metrics、rollout、video 和日志使用一致 artifact prefix；恢复训练时保持 agent-step 计数和调度状态。

## 2026-09-01 — Success 收紧为 fully-inside

- Gate3 保留浅进入 `entered`；Gate4/success 改为 cube 完整 footprint 清过簸箕 mouth 的 `fully_inside`。
- 增加从中心进入到完整 footprint 清口的 earn-only full-progress；`deep_inside` 和 `deep_margin` 仅作诊断。
- Gate4 成立当步立即终止真实物理 rollout。训练窗口成功率和单条视频都不能代替 deterministic 批量验收。

## 2026-09-01 — Mouth-floor reward 修正

- 旧策略曾在 Gate2 后把簸箕压入桌面并提前失败，但整条失败回报仍为正。
- mouth clearance 低于 `+0.5 mm` 后施加 `-4 * relu((0.0005-clearance)/0.0035)^2`；低于 `-3.0 mm` 时硬失败。
- 该修正允许极轻微接触，同时让严重下压路径不再获利；没有冻结左臂动作。

## 2026-09-01 — 历史 v4 基线

- `Sweep2FloorPenaltyV4__20260901` 使用两条 fully-inside expert 做 Actor BC，并使用两条 success、25 mm near-success 和 canonical failure 做离线 Critic return regression及 observation normalization。
- 训练于 `26,116,096` agent steps 停止；3M至24M Gate4依次为 `8.74, 69.37, 75.45, 75.00, 74.42, 86.17, 80.41, 86.09%`。
- 24M deterministic 在 step 299 成功，训练窗口成功 episode 平均 mouth clearance 为 `5.84 mm`。当前只保留24M checkpoint/video作为历史 warmup 基线；它不是新任务的初始化方式。

## 2026-09-02 至 2026-09-03 — Sweep 主消融收敛为公平协议

- 主比较为完整方法、`w/o human` 和 `w/o confidence`；消融只在当前 Sweep2 轨迹上做。
- 早期几轮因三组 expert 预热数据不一致形成复合变量，结果不可作为公平单变量结论；现象只在 `Codex_ablation.md` 留档。
- 最终协议统一网络、reference、物理世界、Gate/reward主体、seed、训练预算和评测口径，仅改变 human/confidence 开关。
- 三组每隔6M录像，训练到24M；最终成功率必须由固定 deterministic rollout 统计，不能直接采用训练窗口。

## 2026-09-03 — 当前默认：完全取消专家轨迹预热

- 当前完整方法和同轮两条消融都从随机 Actor/Critic 开始，不执行 Actor BC、离线 Critic return regression、transition-based observation normalization，也不生成 BC checkpoint。
- 有效运行指纹为 `warmup_role=none`、`actor_samples=0`、`critic_samples=0`、`warmup_transitions=[]`。
- 保留前 10 个 PPO epochs 的在线 critic-only 保护期：rollout 来自当前随机 policy，只更新 Critic；它不读取 expert transition，因此不属于专家预热。
- 训练入口为兼容旧命令仍可接收 transition 路径，但 `warmup_role=none` 时这些路径不得参与初始化。

## 新数据进度索引

Task3的当前状态、成功步骤与未通过原因统一见Codex_new_data.md；本文件只保存工程/Git里程碑。

## 长期安全与发布边界

- 服务器根目录为 `/home/msc-auto/RL_sweep`，分支为 `task_sweep`。
- GitHub 只能使用 `git@github-harroldx:stzabl-png/RL-Correction.git`；身份检查必须显示 `HARROLDX`，不得使用裸 `github.com` 或读取私钥。
- 不得触碰 feiyang、kailang 或其他用户的进程、tmux 和 GPU 资源；禁止 `pkill`、`killall`、GPU reset 和广域终止。
- 未经用户明确授权不得创建 commit、push、自动 resume、修改 Gate/reward/expert 或启动新的大规模训练。
- 不得丢弃、覆盖、stage 或提交现有的 `tasks/pregrasp/arm_shell_points.npz` 修改。


## 2026-09-06 — Task3验收整理、录像清理与算法文档合并

- 本次父版本：a8994fe（data: publish Sweep2 reconstruction dataset）。实际本次提交用 git log --oneline --grep="docs: consolidate Sweep workflow and Task3 acceptance" 定位，避免文档自引用commit hash。
- 用户明确授权更新原dirty Codex_tasks.md、合并旧手册、清理Task3旧视频并创建commit。旧文档原稿及已有工作树patch保存在logs/task3_docs_cleanup_20260906/pre_edit/。
- 本次内容：补全单轨迹数据/IK/物理/residual/PPO/运行步骤；原通用手册内容并入Codex_tasks.md并移除旧文件；Task3当前状态集中Codex_new_data.md；更新HANDOFF、重大错误、关联文档入口。
- 验收记录：用户已确认32 v3和80 v6无方块完整轨迹；128/180因完整IK/物理检查未过而暂存。没有声称新policy已训练成功。
- 清理：12个旧Task3视频共7,003,314字节；最终32/80与最新128/180进度保留，旧trace/json/图片归档。逐文件清单随本次文档提交保存；未跟踪的已删除视频不能由Git恢复。
- 提交范围：文档、Task3历史审计说明、清理清单；不混入共享代码dirty改动、arm_shell_points.npz、大型资产、datasets、reference、checkpoint。此commit是文档/证据索引里程碑，不是完整Task3运行依赖快照。
- 验证：文档引用/核心路径、保留mp4可读性、保护文件hash、git diff --check与显式暂存范围核验。未启动训练/resume，未push。
- 回退定位：git show <目标提交> --stat查看范围；只对需要的文档比较父版本。若目标是回退本次之前的未提交文档内容，查pre_edit原稿，不能把父commit误认为本次开始时的dirty文件。
