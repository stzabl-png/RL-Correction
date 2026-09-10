
## 最新权威快照：human修复后代码交付（2026-09-10）

以下覆盖旧的排队/暂停/训练未启动快照。原成功Sweep2冻结；新轨迹只处理9、32、36、80，fixed优先，non-fixed后置。所有视频位于outputs_video。

- take9：旧Cube15 run因human左侧输入错误已在约4.92M停止；已删除该run模型与TensorBoard副产物，保留日志/指标/视频。HumanV2从真实原始双手腕运动独立重建333帧，source_frame105..284，最大位置误差右19.99mm、左13.78mm，非human轨迹数组核验不变。新run Task3Take9FixedCube15HumanV2_20260910，tmux task9_cube15_humanv2_20260910，GPU1、24M、1024环境、seed42、6M录像、3M诊断、不resume。写入时进度622592步，不能与旧3M指标混用，尚无成功结论。
- take32：已认可调平无物块视频outputs_video/Task3_take32_fixed_level_v1/full_zero/trajectory.mp4（566帧/28.3秒）。尚未训练。必须恢复独立左右human、处理实际跟踪偏差和接触/物块选点后再训练。旧human=q的构建输出仅诊断，不能直接用作full输入。
- take36/80：本轮通用资产/抓姿版本尚未完成并验证；不恢复历史队列。

迁移结论：已有一个成功实例，不代表通用资产+抓姿在其他参考上已形成可执行且可学习任务。需区分工具目标、IK误差、实际关节跟踪、接触/入盆几何与奖励反馈。take32离线IK最大19.98mm、关节步长5.4deg，但实际pan轴倾角中位17.5deg，左j5实际-目标角中位约21.8deg；FixedJoint与FK坐标链吻合，具体增益/力矩/接触原因尚未确定。目标pan轴0deg是“尽量底部平行”的实施代理，不等于不规则底面实测水平。

旧take9 3M窗口Gate1/2=100%、Gate3/4=0、push=0；盆底过滤同时清除了盆外进度，是待验证的奖励适配问题，尚未修复，也不能据此证明PPO失效。15mm零回放实际row47已移动，名义contact_row151；选点应依据实际工作面与接触阶段。take32 1125候选中948初始安全、186后续近接触，均被保守凸包前置条件淘汰，不等于不存在可行位置。物块起点固定，所谓随机主要是当前选择方法不稳定。

后续顺序：①独立human来源与时钟一致性（9已修复，32待修复）；②查实际执行/关节跟踪、桌面与负载约束；③依据实际刷毛—物块—入口关系确定接触阶段和起点，确认residual范围内存在改善机会；④分开盆外推进奖励与盆内承载成功检查，保持防盆下假成功；⑤单条新轨迹受控验证后再扩展36/80。零残差不必直接扫成功，IK约2cm及少量轻微超限可接受，不能以复制human或提高IK精度掩盖实际问题。

具体证据与代码入口见tasks/Sweep/new_data/workflows/TRANSFER_DIAGNOSIS.md、README.md、take9_human_restore_report.json。该分析不是新增训练授权或原Sweep2改动。

# Task3：新 Sweep 轨迹处理、验收与训练记录

## 当前补充：按用户确认切换15 mm物块（2026-09-09晚间）

用户明确同意直接使用15 mm物块，任务1仍为9、32、36、80。新轨迹优先沿用15 mm；原成功Sweep2维持25 mm。本次在Task3的 `training_env.py` 加入配置尺寸适配：仿真启动前修改实体USD Cube的size/extent（视觉与碰撞共用），reset中心高度取桌高+cube_half+0.5 mm，Gate/接触沿用同一training_geometry.cube_half。保留原质量5 g及摩擦参数。未改原Sweep2源码。

25 mm的v3诊断已完成180步：首秒水平位移约2.1e-8 m、最高速度8.6e-5 m/s，开局重叠已消除；后续最大速度0.735 m/s，约row60–90间出现物块横移，因此不能称作整段物理通过或训练就绪。按用户新要求，后续采用15 mm独立配置，不启动25 mm训练。

15 mm后台任务：tmux `task1_cube15_prepare_probe_20260909`，先执行CPU选点 `tasks/Sweep/new_data/prepare_take9_cube15.py`，成功后等待GPU1稳定空闲并接续单环境180步物理录像；失败则停止，不启动训练。启动脚本 `logs/task1_fixed_20260909/launch_cube15_probe.sh`，总日志 `cube15_prepare_probe.log`；静态输出 `cube15/take9_cube_v2_report.json`，配置/参考目标 `take9_powerdisk_fixed_cube15_v1.json/.npz`（分别在new_data/configs、references）；物理产物 `take9_cube15_probe/{summary.json,trace.npz,zero_cube.mp4}`，完成退出码 `take9_cube15_probe_exit_code`。

下次唤醒先查该tmux、日志及summary，确认实际物块15 mm、初始高度和开局稳定，再继续正式训练准备。当前没有新的正式训练，32/36/80作业尚未启动；用户允许长等待时结束本轮对话保存额度。下方25 mm准备记录为历史。

## 当前执行状态：任务1已授权推进（2026-09-09 晚间）

用户已明确要求执行任务1，范围为9、32、36、80；take32已完成调平版据此次确认继续接入，无需再次索要此前视频验收。长时间IK、录像或训练可启动唯一tmux并保存日志后结束对话，由用户下次唤醒接续。当前先完成take9新物块的物理检查，后续继续32训练接入及36/80通用资产构建；任务2非fixed保持后置。用户另允许新轨迹缩小物块，例如20 mm，若采用须同步实体碰撞体、reset高度和几何判定尺寸；本轮找到25 mm安全候选，暂保留25 mm。

实时核对推翻了旧的“take9仍暂停”快照：原 `Task3Take9FixedFull_20260909` 最终跑到24,018,944 steps后退出，tmux和训练进程均已不存在。其起点错误，结果保留作诊断。此前文档所称“自动录像删除手动标记”为未经验证的归因；`gpu_guard.py` 和train.log明确显示暂停900秒后强制清标记并恢复，手动标记也受此超时影响。不要再将普通pause.request当作持久人工停止机制；原成功路径代码本轮未改。

已新增 `tasks/Sweep/new_data/prepare_take9_cube_v2.py`（输出最终候选v3）和 `probe_fixed_cube.py`。最终独立配置/参考为 `tasks/Sweep/new_data/configs/take9_powerdisk_fixed_training_v3.json`、`tasks/Sweep/new_data/references/take9_powerdisk_fixed_training_v3.npz`。工具、双臂、human、confidence轨迹保持原训练v1不变，只更新cube及接近时刻/刷毛点元数据。

候选cube中心世界坐标 `[-0.04766187,-0.08354810,0.883] m`，pan横向中线x=0，25 mm边长。完整扫把凸包分离轴检查：初始净空10.45 mm，作用前最小净空9.72 mm；第149行刷毛至cube盒表面间距7.25 mm，局部运动向盆内，后续183行。该行是nominal接近时刻，并非已发生物理接触；零残差不要求推动或扫入。最初仅按最近点排序的v2候选运动方向朝外，未录制、未训练，不采用。

物理检查tmux：`task1_cube9_probe_20260909`；GPU1先连续60秒无compute进程且低利用率，沿用GPU1独占锁。启动脚本 `logs/task1_fixed_20260909/launch_cube9_probe.sh`；日志 `cube_probe.log`；输出子目录 `take9_cube_v3_probe/`（`zero_cube.mp4`、`trace.npz`、`summary.json`）；退出码 `take9_probe_exit_code`。180控制步、单环境、真实cube碰撞、零residual、force_replay，仅检查reset及nominal轨迹，尚非训练成功证明。完成后先检查首秒位移/速度及视频，若仍击飞则修正；不自动跳过结果审查启动训练。

CPU搜索与报告：`logs/task1_fixed_20260909/take9_cube_search.npz`、`take9_cube_candidates.json`、`take9_cube_v2_report.json`（报告明确指向最终v3）。尚未开始32/36/80新处理作业，尚未启动新的24M训练。

更新：2026-09-06（服务器本地日期仍为09-05）。服务器msc-a6000；根目录/home/msc-auto/RL_sweep；分支task_sweep。Task3后续细节只在本文维护；Codex_HANDOFF.md负责入口，Codex_tasks.md负责原单轨迹成功算法与运行步骤。

## 本次启动交付状态

2026-09-06T06:06:04.145532+00:00：用户明确要求任务启动后即可结束，不等待仿真环境创建完成。32已按最终参数启动进程PID1130607，处于环境初始化阶段；本次不声称已进入训练采样。80由同一GPU1队列在32结束后接续；36仍在GPU0等待队列。三条均为独立cubev2位置、40M总步数、12M录像周期、3M诊断检查点，从头开始，不resume。持久tmux和日志负责后续运行，当前窗口结束不终止任务。

后续同类“启动训练”请求：检查实际进程与启动参数、记录日志和队列即可交付，不为等待环境创建或出现第一条训练指标持续占用对话。需要训练结果时另按用户要求检查。

## 当前运行参数（最新）：40M，每12M录像

用户将录像周期从10M改为12M。三条cubev2命令统一为--max_agent_steps 40000000 --record_every_steps 12000000 --diag_every_steps 3000000，录像节点12M/24M/36M。原10M启动先遇到3M诊断周期不能整除的参数断言；改5M后正在初始化时用户改为12M，已终止该初始化，未resume旧策略，以12M参数重新启动。

32现在GPU1启动，80随后接续，tmux task3_cubev2_running32_then80_gpu1_20260906；脚本logs/task3_cube_reposition_20260906/start32_then80_gpu1.sh。GPU1约40GB空余，本任务与已有进程共用GPU，不停止或修改其他用户作业。36仍按用户原要求等待GPU0，tmux task3_cubev2_36_gpu0_20260906。各条launch_train.sh为最终可执行参数。下文10M/5M为历史设置，不再使用。

## 最新训练预算：40M，每10M录像

用户最新要求重新启动三条新配置训练，延长到每条40M，录像周期按上下文“每10步”解释为每10M步，即10M/20M/30M/40M。三条cubev2启动脚本已更新为--max_agent_steps 40000000、--record_every_steps 10000000；保持full、seed42、1024环境、各自物块位置，从头训练，不resume。GPU1的32→80和GPU0的36持久队列仍有效，等待其他用户计算进程退出后自动使用更新命令，不重复建队列。

旧32训练6M/12M视频及其输出目录内顶视帧已删除，清单logs/task3_cube_reposition_20260906/old_training_video_cleanup.json。仅删除旧训练视频目录，保留已验收32/80/36零残差视频、原Sweep2成功视频、旧训练检查点和数值日志。下文24M/6M均为历史预算，当前以40M/10M为准。

## 最新状态：方块起点v2，旧训练已停止

2026-09-06：用户要求停止旧训练，按每条轨迹初始双工具位置分别选择红色方块起点，随后补充“推不到也没事，不强求”。已停止32旧训练和全部旧队列；80/36旧run没有启动。旧检查点保留，不resume。

方块保持原成功Sweep2边长25mm、颜色和物理参数不变。新起点位于初始pan入口外，横向完整方块处于入口宽度内，依据早期刷头表面投影选点。参考世界坐标（m）：

| take | cube XYZ | 接触参考行 | 新配置 |
|---|---|---|---|
| 32 | [-.052168015, -.065287008, .883] | 80 | take_32_training_v2.json |
| 80 | [-.070850147, -.205925091, .883] | 95 | take_80_training_v2.json |
| 36 | [-.165175255, -.128757342, .883] | 80 | take_36_training_v2.json |

这不是零残差扫入成功声明。32/80入口网格回放未得到Gate2；用户明确放宽推动要求后停止继续搜索，36该轮搜索中途停止、没有完整结论。选点通过初始入口几何检查，不声称已验证无碰撞或容易成功。80早期刷头较高，仍需策略学习补偿。

每条新建references/take_<ID>_training_v2.npz，只相对v1改变cube_start_w和contact_row；其他数组逐项相等。已验收零残差reference、prior、工具资产和手臂运动不变，PPO/reward/观测/动作不变。各自full、seed42、1024环境、24M、每6M录像，从头开始。

新run为logs/Task3_take<ID>_full_seed42_cubev2_20260906/，命令在各目录launch_train.sh。GPU1目前已有其他用户计算，新32→80队列等待空闲；GPU0的36也等待已有任务退出。tmux分别为task3_cubev2_32_then80_gpu1_20260906、task3_cubev2_36_gpu0_20260906。队列脚本/日志为logs/task3_cube_reposition_20260906/queue_gpu{1,0}.{sh,log}。selected_positions.json保留选点坐标、初始pan坐标及验证边界。不要重复启动或恢复旧run。

**下文训练v1运行中/排队状态是停止前历史记录；当前以本节为准。**

## 1. 当前状态与重要参考

用户已验收take32 v3、take80 v6、take36 v4的无方块零残差录像，并明确授权只训练这三条独立policy。零残差验收代表抓姿、资产和重建运动可接受，不代表新策略已学会扫块。128、180永久弃用；304只是历史候选，不在本轮范围。

| take | 已验收配置（tasks/Sweep/new_data/configs/） | 有效reference（references/） | 最终录像（outputs_video/） |
|---|---|---|---|
| 32 | take_32_v3.json | take_32_reference_v3.npz，566行/28.30秒 | Task3_take32_v3/full_zero_retry1/trajectory.mp4，同目录closeup.mp4 |
| 80 | take_80_v6.json | take_80_reference_v3.npz，449行/22.45秒 | Task3_take80_v6/full_zero/trajectory.mp4 |
| 36 | take_36_v4.json | take_36_reference_v2.npz，607行/30.35秒 | Task3_take36_v4/full_zero/trajectory.mp4 |

| 训练 | GPU | 本次整理时状态 | 持久日志目录 |
|---|---|---|---|
| take32 | 1 | 正在运行，已生成3M检查点 | logs/Task3_take32_full_seed42_20260906/ |
| take80 | 1 | 等take32结束后顺序启动 | logs/Task3_take80_full_seed42_20260906/ |
| take36 | 0 | 已进入队列，等待现有计算进程退出 | logs/Task3_take36_full_seed42_20260906/ |

重要参考仍是原Sweep2单轨迹成功方法，不用新数据问题反向修改该算法。阅读顺序：Codex_HANDOFF.md → Codex_tasks.md → Codex_mistakes.md → Codex_commit.md → 本文。必须实际观看的基准：
- ego：datasets/sweep_2_better/44ce97212ec9c5c97ad567934f473707.mp4
- 原始GraspPose render：datasets/sweep_2_better/sweep2_grasppose/demo_replay/
- 零残差：outputs_video/sweep2_v1_zero_replay.mp4
- full 24M成功录像：outputs_video/Sweep2AblationNoOfflineFull__20260903_policy_0024M/policy.mp4

## 2. 输入、授权与边界

原始输入为datasets/sweep_new_data/sweep_dustpan/，GraspPose交付为datasets/sweep_p4_grasppose_16takes_20260905/。32/80的object_0为簸箕、object_1为扫帚；36恰好相反，须先做输入语义适配。原始datasets不改动。

三条分别沿用成功full方案：seed42、1024环境、每条24M、每6M录像；14维累计residual、confidence/human开启、前80控制步零残差、无离线expert预热、前10个在线critic-only epochs、随后PPO。不是共享多轨迹policy，也不修改reward/PPO来掩盖坏输入。只启动已授权的新run，不resume、不再加入候选。

验收回放的cube隐藏、禁用碰撞及reward/termination影响；训练使用独立training配置启用方块并按各自工具几何标定。不能混用这两个入口。资产修复只在Task3独立副本中；保护实际抓区、原手指角及正确运动。用户允许丢弃明显异常重建片段，不能借此删除正常但困难的运动。不得影响其他用户进程，不得触碰tasks/pregrasp/arm_shell_points.npz或覆盖已有共享修改。

## 3. 成功轨迹共同遵循的处理链

1. 看ego和重建render，明确人抓柄还是盆边，观察首帧、转腕过程和工作面方向。
2. 看原始GraspPose render；数值排名和functional标签只是筛选线索，必须匹配抓取位置、手掌朝向。
3. 转换canonical候选为input坐标prior，并渲染静态机器人手。令Rci为canonical_from_input，c为COM：

       p_input = Rci.T @ p_canonical + c
       R_input_hand = Rci.T @ R_canonical_hand

   contact位置同样加COM，法向只旋转；手指角不做空间平移。Task3实现为prior_frame.py，不能让旧的漏COM路径覆盖正确prior。
4. 核对candidate canonical → input OBJ → USD mesh → rigid root → world。USD若与OBJ已一致就不能再补180度；每一层都要用实际变换和点坐标检查。
5. 仿照成功Sweep2修可见簸箕mesh，保护抓区，重测入口；碰撞体开放，不使用封住mouth的单一凸包。
6. 共享注册双工具到桌面，32/80保留完整源6DoF运动；新候选先检查语义编号与资产/位姿准确性；通过固定工具—手关系得到腕目标，再用既有IK求双臂reference，同时生成human方向和confidence。
7. 做FK、限位、连续性检查。小的离散步长过大可同步加密两手和两工具时间轴，但保留原样本和运动；不能把有限位分支跳变用插值或unwrap伪装成可达。
8. 用trajectory_env.py初始化各自finger prior，检查reset与FixedJoint，录制完整zero-residual。实际root-to-world从physics tensor trace核验，不能把USD初始缓存误当实际动态状态。
9. 亲自连续播放到末尾，结合数值检查后交用户确认。32、80已完成连续播放核验；36完成全程采样画面与607帧数值方向核验，并获用户验收。本次三条训练另获用户明确授权。

## 4. take32：从被拒绝到v3验收的每一步

### 4.1 原来哪里错了

旧v1虽然543行IK数值通过，但抓法/簸箕方向被用户拒绝，还含未裁定位置的红方块。根因不是简单“整个工具应该再旋转180度”：新canonical有非零COM，旧转换遗漏平移；旧扫帚候选抓在头部，与ego柄部抓法不符；簸箕可见本体需要修复。OBJ到USD root本身是单位变换到浮点精度，未发现需要额外USD旋转。

### 4.2 正确转换和重选抓法

Task3 prior_frame.py补回COM，pan contact到源mesh最近顶点的中位距离从16.8082mm降到1.4285mm。pan使用fingertip_mid__0_1_grasp.npy，broom改为柄部8_Prismatic_2_Finger__17_2_grasp.npy，分别生成priors/take_32_dustpan_v2.npz与take_32_broom_v2.npz。选择依据是ego/render/转换后静态抓姿及完整IK共同证据，不因某个标签就接受。

### 4.3 修簸箕本体、保留手实际抓住的柄

源pan的-Y在注册后朝上，但原mesh这一面是背面。assets/take_32/dustpan_v3/只修改盆体、颈部及入口，z<=-30mm柄部顶点和全部face索引保留。中间v2围绕偏移轴过渡造成颈部扭曲，被舍弃；v3通过归一化椭圆截面旋转并插值中心避免该问题。验证watertight、winding、正体积，生成独立USD与开放入口碰撞体。repair_take32_asset.py保留构建方法。

### 4.4 重建完整reference，不使用被拒绝旧v1

初次新构建最大关节步长10.3055°；共享时间扩展后561行仍8.7404°。retime_take32_candidate.py在新candidate的大步边间同步插入5个样本，原561行全部保留，最终566行。双工具注册相对原RTS最大偏差<=9.24e-8m/0.000182°，证明是同一运动经共享刚体变换及时间加密，没有单工具轨迹重写。

最终reference：tasks/Sweep/new_data/references/take_32_reference_v3.npz。右FK最大1.9939mm/1.0466°、关节步长5.7483°；左1.9970mm/1.0466°、步长3.6134°。

### 4.5 物理修正与录像验收

Task3子类修正旧初始化错误读取Sweep2右手finger prior的问题，保持共享训练算法。运行时断言禁止继承的pan自动单独平移改掉reference。第一次近景因near plane=1m裁切被中止；retry1设.01m，录完566行28.30秒、source0..299、全部零残差，无cube影响。ego、候选、完整总览与近景均实际看过，随后用户确认验收。

物理跟踪中位/最大：broom4.99/44.90mm，pan5.70/24.49mm；姿态峰值7.62°/3.56°。手—工具闭环<.00018mm；mesh最低相对桌面broom+.536mm、pan-2.313mm。这是已披露的物理跟踪/接触局限，不能与约2mm的IK误差混为一谈。

证据：logs/task3_take32_v2_20260905/{source_path_audit.json,retime_audit.json,coordinate_audit.json,sweep2_default_regression.json}。原命令launch_full_v3_retry1.sh；最终视频目录保存trace、numeric_audit、record_summary、runtime_coordinate_chain。

## 5. take80：保留原运动与柄，只换刷头后验收

### 5.1 先获得正确的双手抓法和完整运动

ego双手抓柄。pan选fingertip_small__25_35_grasp.npy，prior为take_80_dustpan_v3.npz；broom选fingertip_middle__19_18_grasp.npy，prior为take_80_broom_v2.npz。canonical转换计入COM。pan独立dustpan_v3翻正盆体、保留柄部接触区、按该mesh尺寸降低入口。共享注册后source0..299完整20Hz参考449行。

用户认可v3运动，但拒绝粗糙扫帚。v3参考仍是最终有效运动依赖：references/take_80_reference_v3.npz，不能因为视频版本号旧而删除reference/柄部资产。

### 5.2 为什么整把换成take32后反而错了

v4把take32整把扫帚按首帧世界向下对齐，使用Rz+90°，将donor刷毛-Y映成take80+X。ego/旧v3实际是先朝内、转腕后朝下；这个旋转只照顾首帧，导致后续刷毛反向。用户明确拒绝v4。v5尝试整把donor不旋转，虽然方向正确，但新抓姿首帧IK失败，最佳约5.11cm/9.67°，没有生成有效reference。

### 5.3 最终v6只改刷头，既有柄与抓姿不动

用户明确允许保留旧柄只改毛刷。graft_take80_broom_head.py执行：
1. 从take80旧broom_v2保留z<=.012m的柄部完整三角面，保留原顶点坐标及UV；切口封闭。
2. 从已验收take32 broom_v2取z>=-.025m刷头，仅平移[0,-.008,.028]m，不旋转，保留刷毛input -Y。
3. 柄头两部分分别封闭、在连接处重叠；不是单一布尔焊接mesh。USD只保留一个RigidBody根节点。
4. 原contact最大z=-.003038m，整个接触区在保留柄范围内；柄顶点位移严格为0。
5. config使用assets/take_80/broom_head_v6/，完全复用v3 reference、左右手prior和全部手指角。不重算IK，不复制take32运动。

### 5.4 完整物理核验和用户确认

449行22.45秒，成功Sweep2同视角、1280×720/20fps。旧v3与新v6近景并排连续播放至ended=true，确认首帧朝内、转腕后向下、柄头无可见脱节。物理trace中从row50/source18起刷毛world-Z最大-.975725，偏离向下最多约12.65°，整段都核对而非只看首帧。

broom tracking中位4.717mm、最大34.727mm；pan最大9.812mm；手—工具位置闭环<.000121mm。刷头全程最低离桌25.322mm，pan最低-2.227mm，继承的原轨迹高度/滞后仍保留。本轮没有强行压低扫帚做扫块成功。用户随后明确“这个没问题了”，v6当前已验收。

证据：logs/task3_take80_brush_direction_20260905/{graft.log,assembly_audit.json,audit_direction.py,launch_full80_v6.sh}；资产目录graft_audit.json；最终录像目录numeric_audit.json、brush_direction_audit.json、trajectory_trace.npz。

## 6. take36：语义适配、局部资产修复与刷毛方向修正

### 6.1 输入与抓姿

36原数据object_0是右手扫帚、object_1是左手簸箕。prepare_take36_input.py在prepared/take_36_trimmed中交换obj_pose_all、obj_verts_local、obj_valid_all的工具编号，左右人手不交换，原始数据保持不变。camera/world注册统一处理；qpos的来源时间戳在确认关节内容不变后重新绑定，保留原记录。

扫帚原rank1靠近头部，不符合持柄抓法。最终选fingertip_small__9_21_grasp.npy（rank11），生成take_36_broom_v2.npz；虽然原functional标签为False，但实际接触在柄z=-62.88..-42.05mm，结合静态手、接触和可达性采用。簸箕原候选抓得过高，限制颈部修复；改用fingertip_small__5_38_grasp.npy，接触最大z=-52.8mm，生成take_36_dustpan_v2.npz。转换均计入canonical COM。

### 6.2 簸箕和完整reference

原簸箕盆体开口方向错误。assets/take_36/dustpan_v2配合repair_spec_v2.json翻正盆体，只在z=-.05..-.03m窄颈区过渡，z<=-.05m的柄完全保留。早期宽过渡方案会扭曲盆体，故弃用。使用repair_pan_asset.py生成独立mesh/USD并测量入口。scene_table_z=.7963428804444793；修复后最低点改变，通过共享场景高度注册调整双工具，不单独移动某一工具。

源293→294附近出现31.235°异常旋转，完整300帧构建最大关节步长39.194°。按用户允许，仅丢弃末尾294..299这6个明显异常源帧，保留0..293；不删除原数据。裁尾后570行参考全部IK可达、FK<2mm，但右臂最大离散步长18.302°。沿用retime_grasp_candidate.py同步加密双手、双工具与human时间轴，插入37行、保留所有原样本，得到607行。右/左最大参考关节步长5.953°/1.235°，FK均<2mm。

有效reference是take_36_reference_v2.npz；可复现构建入口logs/task3_take36_20260906/build_reference_v2.sh，证据source_trim.json、retime_report.json及构建日志。candidate先保留诊断结果，再经同步加密写最终reference。

### 6.3 v3为何被拒绝，v4如何成功

v3抓姿与数值回放通过，但用户指出刷毛方向错误。原刷毛input -Y经运动映射后近乎水平，首帧world-Z约-.148；正确朝下工作面应为input +X。不能旋转整把扫帚去破坏已正确的柄和手腕。

rotate_take36_broom_head.py仅围绕input +Z旋转刷头+90°，轴心XY=[.00004824,-.006067955]；z>=-.01m为完整旋转，-.03..-.01m平滑过渡，z<=-.03m的柄顶点严格不动。生成assets/take_36/broom_head_v2；mesh拓扑、watertight、winding通过，USD rigid root保持单位变换。左右prior、手指角、607行reference和簸箕不变。

v4完整607行物理回放，刷毛world-Z全程处于[-.999977,-.749933]，均朝下；实际最大关节步长6.053°。broom跟踪中位/最大6.133/37.232mm、姿态最大6.319°；pan为2.787/9.658mm、1.880°；手—工具闭环<.000130mm。刷头/簸箕最低离桌分别+3.128/+2.111mm。已检查覆盖全程的采样画面及全部607帧方向数据，用户随后明确认可并要求训练，当前v4已验收。

复现：logs/task3_take36_20260906/record_v4.sh；保留head_correction.log、full_record_v4.log、full_audit_v4.log，最终录像目录的numeric_audit.json、bristle_direction_audit.json、runtime_coordinate_chain.json、trajectory_trace.npz。不要再采用被拒绝v3刷头。

## 7. 失败轨迹与候选边界

- take128：簸箕和扫帚重建均错误，用户明确永久弃用。
- take180：资产重建粗糙，源163→164附近旋转异常、右臂跳变；替换资产和清理异常后虽得到数值可回放版本，抬臂姿态及视觉仍不合要求，用户明确弃用。数值通过不等于验收。
- 两条的生成资产、prior、reference、配置、录像与专用临时文件已删除；原始datasets和GraspPose交付保留，不恢复训练。
- take304只作为历史筛选候选。当前仅训练32/80/36，不继续处理304。筛选数值依据保留在logs/task3_reselect_20260906/source_metrics.json。

## 8. 训练接入与运行入口

解释完整算法优先读Codex_tasks.md。环境为/home/msc-auto/rlcorr-venv/bin/python，PYTHONPATH=.，VEGA_URDF=/home/msc-auto/data/vega_urdf/vega_1p_sharpa.urdf，OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1、SHARPA_WANDB=0。

每条使用独立configs/take_<ID>_training_v1.json和references/take_<ID>_training_v1.npz；从已验收参考复制，保留arm、human、source、工具运动和confidence，仅更新训练需要的cube/contact/语义几何元数据。训练启用真实方块；零残差验收配置仍保持无方块，不回写已验收运动。

training_env.py继承原SweepEnv，沿用原reward、done、obs和action计算，只适配各自finger prior、开放簸箕碰撞体、实际工具几何和lip高度，并禁止运行时偷偷单独平移工具。train_sweep.py通过可选--task_config选择该入口，world.json记录任务依赖与hash；录制器继承同一配置。默认Sweep2入口不变。原默认路径160步回归报告为logs/task3_take32_v2_20260905/sweep2_default_regression.json：状态完全一致、reward/obs/priv误差约1e-6；这是有限回归，不是重跑24M。

36训练方块中心[-.163590163,-.184372187,.883000016]，接触参考row205，刷头工作面为input +X。按实际pan量测half_width=.065、inside_z_min=.011、mouth_z=.091、center_y范围[.001,.011]、start_outside=.065、deep_inside_margin=.02、lip_y=-.015。这些是资产几何标定，不是修改算法；不能直接套旧Sweep2的绝对尺寸。training_lip_y可配置，32/80仍沿用默认-.0145。36按用户要求直接排队，本轮没有另外启动36训练smoke；启动后训练效果尚待日志与后续策略录像确认。

持久启动入口：
- 32/80：tmux task3_train32_then80_20260906，logs/task3_training_prepare_20260906/run_training_queue.sh；先32、结束后80，GPU1。
- 36：tmux task3_train36_gpu0_20260906，logs/Task3_take36_full_seed42_20260906/queue_gpu0.sh；每60秒只读检查GPU0计算进程，全部退出后调用同目录launch_train.sh。queue.log已记录排队。当前等待其他用户，未占用GPU0启动训练。
- 三条实际训练命令均保存在各自logs/Task3_take<ID>_full_seed42_20260906/launch_train.sh，24M、seed42、1024环境、offline actor/critic预热为0、每6M录像、3M诊断检查点。
- 检查状态用tmux、queue.log、train.log和progress_steps.txt；不要重复运行launch脚本，也不要把排队状态误称训练已开始。当天工作到此结束，后台队列继续。

32回放复现入口logs/task3_take32_v2_20260905/launch_full_v3_retry1.sh；80为logs/task3_take80_brush_direction_20260905/launch_full80_v6.sh；36见上节。重录使用新out_dir，保留已验收录像。

## 9. 清理与证据保留

此前按用户要求清理被取代录像，并删除128/180生成产物468个文件、243,832,338字节；清单为logs/task3_reselect_20260906/cleanup_manifest.json、cleanup_extra.json。原始数据不在删除范围。

本次收尾删除60个文件，共16,501,137字节（约16.5MB）。只删除logs中已被最终版本替代的临时图像、旧比较录像、废弃候选副本、零字节渲染日志、已取消smoke及旧一次性脚本。逐文件路径、大小、SHA256和原因见logs/task3_day_close_20260906/cleanup_manifest.json。保留原Sweep2成功训练与消融、所有检查点、expert数据、正在运行/排队的三条训练、最终复现脚本和必要失败证据。旧图片被清理后，不把历史记录中的旧路径当成当前交付。

旧文档与整理前working-tree.patch仍在logs/task3_docs_cleanup_20260906/pre_edit/；旧录像旁JSON/trace继续保留，已删除的未跟踪二进制不能靠Git恢复。Git提交历史看Codex_commit.md和git log，不把工作树改动误称已提交；保护既有Codex_tasks.md及共享代码修改。

## 2026-09-06：32完成40M后的失败分析

32 cubev2完成40M，Gate3/4全程0。12M/24M/36M回放显示：方块在前80步零残差阶段已移动约97mm，策略开放时错过作用位置；入口抬高到约20mm，方块仍在桌面、位于盆下；旧brush_contact_local局部采样未随新cube起点同步，造成接触代理覆盖偏差。实际推动但Gate2不触发，push奖励基本为0；任意累计位移又使参考继续前进。原成功轨迹也有前80步接触，因此不建议直接改prelude/PPO/Gate条件。需先成套核验cube起点、brush_contact_local、contact_row与第80步后可操作状态。完整证据与限制：logs/task3_failure32_20260906/analysis.md，录像采样overview.jpg。本次只分析，80训练与36队列保持不变，未启动评测或新训练。

## 2026-09-09：take9 FixedJoint 与薄入口簸箕

take9沿用原Sweep2工具体系和新PowerDisk抓姿，最终仅推进FixedJoint版本；真实刚体抓持方案已搁置，原成功Sweep2不改。左右对应为：右手扫帚 `98_45`，左手簸箕 `45_20`。已知前约7秒右臂穿过左臂，用户允许后续训练前裁掉；本轮没有处理该异常。

用户对簸箕几何的最终要求是：前缘为薄入口并尽量贴桌，入口到盆底连续、没有阻挡物块的翘坡；两侧挡边允许弯曲并向入口逐渐降低，后部和手柄保留。此前 `entryfix_v2–v11` 因混淆局部坐标、侧面轮廓和真正工作面而作废。最终 `dustpan_thin_entry_20260909` 从原始take9 mesh直接生成，上下表面一起连续映射，名义前缘厚度0.5 mm，保留正厚度而非把网格压成零面积平面；z<=45 mm后部和手柄严格不变。几何检查：142696顶点、285384面、watertight、winding一致、正体积、无零面积面。USD与OBJ坐标一致。

新碰撞体由修复mesh切成49个局部凸片，覆盖盆底和两侧，入口开放；不再使用横跨盆口的坡面盒或单一凸包。最终资产、`collision.json`、`report.json`、剖面图、前后对比和复现说明都在 `tasks/Sweep/new_data/assets/take9_powerdisk/dustpan_thin_entry_20260909/`。

检查reference以take9 v7为起点，保留扫帚和右臂，簸箕横向校平、俯倾3°、向机器人侧移20 mm并按桌面高度重新落位，再重算左臂IK。527行IK全部通过，最大位置误差0.100 mm、最大姿态误差0.057°、相邻关节步长2.19°。该reference与 `preview_config.json` 是无物块资产/轨迹检查专用，不是训练输入；`human_left_q`尚未按新姿态重建。

完整FixedJoint物理回放：`logs/task3_take9_powerdisk_20260908/thin_entry_preview_20260909/`，三路视频为 `overview.mp4`、`closeup.mp4`、`pan_entry.mp4`，40步静置+527步运动。工具闭环误差极小；实际入口受左臂跟踪影响会局部抬起，入口中央测量带近侧间隙约2.53–10.25 mm，另一侧/上表面更高。用户明确“不需要完全死贴桌面，尽量就好”，随后确认资产很好并要求当日收尾。未放物块、未验证扫入、未启动take9训练。

明日继续顺序：先基于最终薄入口资产建立正式FixedJoint配置；按用户先前规则裁掉开头约7秒异常段；重新生成训练reference和human引导；分别重标25 mm红色物块起点、刷头工作面、`brush_contact_local`、`contact_row`、簸箕入口/内部几何；检查接触判定、盆下误入和参考推进；再生成无物块零残差视频供复核。原成功Sweep2继续保留80步规则，新take9从第1步允许residual；在用户通过回放前不启动训练。

当日清理只覆盖可确认由本轮生成且已被最终版本取代的文件：远端47个文件、239,250,483字节，清单与SHA256见 `logs/task3_take9_day_close_20260909/`；本地同步删除 `dustpan_entryfix_v2–v11` 图片和旧 `take9_entryfix_v1` 回放副本。最终资产、最终回放、可复现脚本、v7基准和现有配置依赖均保留。

## 2026-09-09：通用资产方案、take9训练暂停及cube起点根因

用户最新决定：原成功 Sweep2 方案继续冻结；Task3新轨迹统一使用 take9 最终薄入口簸箕、Sweep2来源扫把以及 take9 fixed 抓姿（右 `98_45`、左 `45_20`）。资产对齐到接收轨迹的初始工具姿态，允许小幅平移、簸箕调平、约2 cm以内IK误差及剔除极少异常帧。当前先完成fixed方案；36、80后续按同样规则处理，非fixed右手真实抓持后置。

### take32通用资产调平版

已生成 `tasks/Sweep/new_data/configs/take_32_universal_fixed_semantic_v1.json` 和 `tasks/Sweep/new_data/references/take_32_universal_fixed_semantic_v1.npz`。保留32原轨迹的整体工具方向与扫入关系，仅根据旧/新簸箕工作面法向差减少约6.4°侧倾；566帧全部保留，左右IK可达率100%，位置误差上限约10 mm。无物块、零residual、FixedJoint回放位于 `outputs_video/Task3_take32_universal_fixed_semantic_v1/full_zero/trajectory.mp4`。配置仍标记 `pending_zero_residual_video_review`，录像summary仍标记 `pending_visual_and_numeric_review`；未获得用户当前明确验收，不得称作训练就绪，也未启动该版训练。

### take9错误训练输入与暂停状态

正式run使用 `take9_powerdisk_fixed_training_v1.json`，tmux为 `task3_take9_fixed_full_20260909`，GPU1、1024环境、24M、seed42、从第1步允许residual；已生成3M/6M/9M checkpoint。用户查看开局击飞后要求暂停。6M录像和rollout的几何复核确认：初始方块中心为 `[-0.04856736, 0.00735200, 0.883] m`，方块与首帧刷头直接相交；781个扫把mesh顶点落在25 mm方块体积内，最近表面点距中心8.9 mm。方块前0.5秒横向移动约20.7 mm，而前五步累计policy residual不足约0.3°，所以根因是初始碰撞，不是策略首步异常。

`logs/task3_take9_fixed_training_20260909/prepare_training.py` 的候选搜索只要求方块完整footprint位于簸箕入口外、横向落在走廊内、底面高于工作面，并最小化后续工作点距离；它没有检查第0行至接触行之前的扫把mesh与方块定向体积安全间距。因此原 `contact_row=97` 的静态审计不能证明开局无碰撞。

15:33 PDT实时核验发现自动录像脚本清理了此前手动pause标记，训练曾意外从约9.63M继续推进；已重新写入 `/home/msc-auto/.cache/rl_correction_gpu1/pause.request`，值为 `manual_take9_review_20260909`。15:33:23 PDT日志确认训练再次释放GPU槽位，记录步数为10,682,368。接手必须先确认进度仍停止，不得删除标记或resume。现有checkpoint受错误起点影响，只保留诊断，修正cube后应从随机初始化重新训练。

下一步只处理take9 cube起点：在保留现有fixed工具、薄入口簸箕、裁剪reference和盆下过滤的前提下，新增第0行至名义接触前的完整扫把mesh/方块OBB净空约束，同时保证方块在桌面、入口走廊内且后续存在有效扫入运动。选定后先做短零residual物理回放，确认初始静止、无穿模和无瞬时冲量，再准备新的独立训练run。36、80和非fixed均暂不推进。

## 当前任务范围与阶段顺序（2026-09-09）

后续新轨迹仅处理 **9、32、36、80**。原成功 Sweep2 fixed 轨迹是不可修改的冻结基准；128、180、304及其他候选不属于当前范围。

任务1是四条新轨迹的 fixed 路线：统一采用已认可的 take9 薄入口簸箕、Sweep2来源扫把和通用抓姿，在每条轨迹初始帧完成资产替换/对齐；按用户标准允许小幅位置与调平、约2 cm以内IK误差和极少异常剔除，同时检查穿模、扫把方向、cube起点、接触时刻、入口工作面及盆下误判。take9先修复初始cube与刷头重叠，再继续32、36、80并分别训练独立policy。

任务2是任务1完成后的 non-fixed 尝试：左手簸箕仍 fixed，右手扫把使用手指全握柄的抓姿；之后再研究摩擦增大或轻微收紧。任务2当前暂停，不得影响任务1或原成功Sweep2。


### 15 mm 实体回放结果（2026-09-09 晚间）

轨迹9 `take9_powerdisk_fixed_cube15_v1` 已完成180帧实体零残差回放；实体边长0.015 m、判定半边长0.0075 m，前1秒横向位移约2.4e-8 m，初始稳定检查通过。全程峰值速度0.792 m/s，后段碰撞仍须结合实际接触时序复核；不能据此宣称全程通过或训练成功。新训练尚未启动。结果：`logs/task1_fixed_20260909/take9_cube15_probe/summary.json`，视频同目录 `zero_cube.mp4`。下一步复核接触后启动take9新训练，再继续32、36、80。原Sweep2基准未改。

## 2026-09-10 最新执行：视频目录纠正与take9启动

所有新视频一律保存 `/home/msc-auto/RL_sweep/outputs_video`；logs只存日志和数值证据。take9 15mm回放已移动到 `outputs_video/Task3_take9_cube15_v1/zero_replay/zero_cube.mp4`。用户明确要求直接开启训练，不等待初始化、不做smoke。已执行tmux启动命令：`task3_take9_cube15_20260910`，脚本 `logs/Task3Take9FixedCube15_20260910/launch_train.sh`，日志同目录train.log，GPU1、1024环境、seed42、24M、每6M录像、每3M诊断、无离线预热、不resume。仅确认启动命令已执行，不宣称初始化完成。

随后处理32：看过旧调平视频抽帧，当前CPU任务 `task1_take32_level_20260910` 逐帧保持入口水平朝向、让pan局部上轴对齐世界竖直，最低网格点距桌0.5mm，再解左臂IK；复用同一donor资产与碰撞体，坐标变换合入工具世界姿态，避免旧semantic资产与未变换碰撞体不一致。脚本/日志/报告在 `logs/task1_take32_level_20260910/`。后续设计15mm物块并录制回放，然后启动32训练；尚未完成32的新回放或训练。任务范围仍为9、32、36、80，原成功Sweep2保持冻结。

### take32后台接续（2026-09-10）
左臂完全调平首次求解在row52出现20.41mm误差（略超2cm），已保留build.log并增加已有解精修/多初值重试，不放宽2cm位置界限。当前IK tmux为task1_take32_level_20260910，日志build_retry.log。接续tmux为task1_take32_pipeline_retry_20260910；脚本continue_pipeline.py，日志pipeline_retry.log，实时阶段pipeline_status.json。依次等待IK产出、CPU物块选点、录制566帧完整回放，然后执行take32训练启动命令；任一步失败即停并记录。视频目标outputs_video/Task3_take32_fixed_level_cube15_v1/zero_replay/zero_cube.mp4。录像沿用GPU1互斥锁和有属主的短暂让出标记，结束后只清理自己的标记；take32训练使用同一GPU1锁，按资源可用状态排队，不与take9同时计算。启动后不等待初始化、不加smoke。当前未宣称32的IK、回放或训练已完成。

### 实时状态更新：take32实用IK容差（2026-09-10）
轨迹9已完成在线critic-only阶段并进入PPO，当前不足1M。take32此前在row378因20.00345mm位置误差被严格20mm断言拦住；并非大误差失败。按用户约2cm的实用要求，仅添加0.1mm数值余量（20.1mm），保持旋转/全程关节连续性检查，重新启动CPU IK与接续流程。新日志build_retry2.log、pipeline_retry2.log；未声称32回放已生成或训练已启动。

## 用户最新IK标准（2026-09-10，覆盖旧严格阈值）
新轨迹IK统一按实用标准：位置误差2cm以内可接受，略超2cm也允许，不因微小越界反复重算或阻塞整条轨迹。当前工程默认求解目标20mm，轻微超限允许至25mm并报告实际误差；25mm是本轮实施取值，不是用户指定永久上限。旋转误差结合工具底面、握持姿态和实际回放判断，不以毫厘级拟合为目标；保留连续性、穿模与异常比例检查，极少异常帧可同步剔除。该标准仅用于新轨迹，不修改原成功Sweep2。take32构建脚本已更新；正在运行的Python仍使用其启动时加载的旧版本，不为微小改动中断已有计算，若该轮失败则用新版接续。

### 实时接续：已启用放宽版本
此前存活的旧Python在row379以20.399mm误差退出。现已实际重新启动放宽后的build_level.py（目标20mm，允许轻微超限至25mm），日志build_relaxed.log；接续日志pipeline_relaxed.log，tmux task1_take32_pipeline_relaxed_20260910。此前改文件不等于运行中进程已采用新阈值；本次启动才实际采用。take9最新1,310,720步，32新视频和训练尚未产生。

### take32视频阻塞原因已纠正
566帧调平IK已完成，最大19.9766mm，相邻关节最大5.4009度。流程卡在prepare_cube.py无候选，尚未进入录像；不应让物块选点阻塞已完成调平轨迹的展示。现单独启动task32_level_video_20260910，脚本record_level_only.py，日志record_level_only.log，状态level_video_status.json（均在logs/task1_take32_level_20260910）。视频目标outputs_video/Task3_take32_fixed_level_v1/full_zero/trajectory.mp4，无物块，566帧。物块选点与32正式训练仍未完成。take9最新1,966,080步，录像通过已有GPU1让出协议临时使用资源，完成后继续训练。

### take32录像误报与修复
旧录像日志实际在初始化失败：pan reset偏移42.17mm，触发25mm断言，未生成视频；Isaac异常退出仍返回0，父脚本因此误报完成。已增加视频存在、大小及566帧summary检查。新增task-owned record_level_fixed_reset.py，使用PowerDiskEnv同一fixed初始化，不改变参考轨迹或放宽42mm偏差。已重新启动task32_level_video_fixedreset_20260910，日志record_level_fixed_reset.log。视频是否完成必须核对产物，不能只看退出码。

### take32调平视频已实际交付
fixed reset独立录制已完成。ffprobe核对566帧、28.3秒，record_summary最后row565；视频outputs_video/Task3_take32_fixed_level_v1/full_zero/trajectory.mp4。初始化工具偏移0、2个FixedJoint。此为无物块零残差回放，尚非物块选点完成或训练成功。旧42mm偏移来自预览初始化流程差异，改用PowerDisk固定初始化后消除初始参考偏移；实际运动贴合仍需视觉审查。物块选点仍无候选、32训练未启动。

## take9停止并恢复独立human数据（2026-09-10）
用户明确要求停止本次训练、清理部分副产物、修复human再重启。已终止且核对退出的进程：train PID1932086、autorecord PID1939988，旧run约4.92M。删除仅限该run初始checkpoint、3M checkpoint和TensorBoard事件共4,426,341字节；清单logs/task9_human_restore_20260910/cleanup_manifest.json。保留日志、world、3M metrics、全部已认可轨迹视频、原Sweep2及其他run。
修复脚本logs/task9_human_restore_20260910/restore.py从原始take9 replay_world的joints_left/right读取腕位置与Sharpa腕方向，逐项核对prepared ref_qpos，沿用Sweep2插值和首帧焊接方法，按当前source_frame 105..284的333行重新生成独立human IK。原Sweep2 builder不改，Task3求解使用用户允许的20mm目标容差。所有非human轨迹字段逐项不变。复制human_left_q的旧prepare_training.py已修正并备份，避免重跑再次污染。
重建与重启链：take9_human_restore_20260910、task9_human_restart_20260910；日志restore.log/restart.log，状态status.json。报告完成且数据一致性通过后自动执行独立HumanV2启动命令，不等待初始化、不做smoke。新旧run不混合，不resume。其他奖励和物理问题本轮不修改。

### take9 human修复后重新启动
独立左右人手来源恢复并完成数组一致性核验，报告logs/task9_human_restore_20260910/restore_report.json。新配置take9_fixed_cube15_human_v2.json，新reference同名npz（new_data/configs及references）。新tmux task9_cube15_humanv2_20260910，日志/启动脚本logs/Task3Take9FixedCube15HumanV2_20260910/。从随机初始化训练24M、1024环境、seed42、6M录像、3M诊断，GPU1；未resume，未等待初始化，无smoke。此次只修复human输入，不将其他已发现的物理或奖励问题称为已修复。
