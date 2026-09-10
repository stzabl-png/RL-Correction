
## 最新权威快照：human修复后代码交付（2026-09-10）

以下覆盖旧的排队/暂停/训练未启动快照。原成功Sweep2冻结；新轨迹只处理9、32、36、80，fixed优先，non-fixed后置。所有视频位于outputs_video。

- take9：旧Cube15 run因human左侧输入错误已在约4.92M停止；已删除该run模型与TensorBoard副产物，保留日志/指标/视频。HumanV2从真实原始双手腕运动独立重建333帧，source_frame105..284，最大位置误差右19.99mm、左13.78mm，非human轨迹数组核验不变。新run Task3Take9FixedCube15HumanV2_20260910，tmux task9_cube15_humanv2_20260910，GPU1、24M、1024环境、seed42、6M录像、3M诊断、不resume。写入时进度622592步，不能与旧3M指标混用，尚无成功结论。
- take32：已认可调平无物块视频outputs_video/Task3_take32_fixed_level_v1/full_zero/trajectory.mp4（566帧/28.3秒）。尚未训练。必须恢复独立左右human、处理实际跟踪偏差和接触/物块选点后再训练。旧human=q的构建输出仅诊断，不能直接用作full输入。
- take36/80：本轮通用资产/抓姿版本尚未完成并验证；不恢复历史队列。

迁移结论：已有一个成功实例，不代表通用资产+抓姿在其他参考上已形成可执行且可学习任务。需区分工具目标、IK误差、实际关节跟踪、接触/入盆几何与奖励反馈。take32离线IK最大19.98mm、关节步长5.4deg，但实际pan轴倾角中位17.5deg，左j5实际-目标角中位约21.8deg；FixedJoint与FK坐标链吻合，具体增益/力矩/接触原因尚未确定。目标pan轴0deg是“尽量底部平行”的实施代理，不等于不规则底面实测水平。

旧take9 3M窗口Gate1/2=100%、Gate3/4=0、push=0；盆底过滤同时清除了盆外进度，是待验证的奖励适配问题，尚未修复，也不能据此证明PPO失效。15mm零回放实际row47已移动，名义contact_row151；选点应依据实际工作面与接触阶段。take32 1125候选中948初始安全、186后续近接触，均被保守凸包前置条件淘汰，不等于不存在可行位置。物块起点固定，所谓随机主要是当前选择方法不稳定。

后续顺序：①独立human来源与时钟一致性（9已修复，32待修复）；②查实际执行/关节跟踪、桌面与负载约束；③依据实际刷毛—物块—入口关系确定接触阶段和起点，确认residual范围内存在改善机会；④分开盆外推进奖励与盆内承载成功检查，保持防盆下假成功；⑤单条新轨迹受控验证后再扩展36/80。零残差不必直接扫成功，IK约2cm及少量轻微超限可接受，不能以复制human或提高IK精度掩盖实际问题。

具体证据与代码入口见tasks/Sweep/new_data/workflows/TRANSFER_DIAGNOSIS.md、README.md、take9_human_restore_report.json。该分析不是新增训练授权或原Sweep2改动。


## 2026-09-10 最新执行：视频目录纠正与take9启动

所有新视频一律保存 `/home/msc-auto/RL_sweep/outputs_video`；logs只存日志和数值证据。take9 15mm回放已移动到 `outputs_video/Task3_take9_cube15_v1/zero_replay/zero_cube.mp4`。用户明确要求直接开启训练，不等待初始化、不做smoke。已执行tmux启动命令：`task3_take9_cube15_20260910`，脚本 `logs/Task3Take9FixedCube15_20260910/launch_train.sh`，日志同目录train.log，GPU1、1024环境、seed42、24M、每6M录像、每3M诊断、无离线预热、不resume。仅确认启动命令已执行，不宣称初始化完成。

随后处理32：看过旧调平视频抽帧，当前CPU任务 `task1_take32_level_20260910` 逐帧保持入口水平朝向、让pan局部上轴对齐世界竖直，最低网格点距桌0.5mm，再解左臂IK；复用同一donor资产与碰撞体，坐标变换合入工具世界姿态，避免旧semantic资产与未变换碰撞体不一致。脚本/日志/报告在 `logs/task1_take32_level_20260910/`。后续设计15mm物块并录制回放，然后启动32训练；尚未完成32的新回放或训练。任务范围仍为9、32、36、80，原成功Sweep2保持冻结。

# Sweep 项目交接入口

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

更新：2026-09-09。本文负责阅读顺序、路径和当前运行快照；Task3过程与后续补充统一写Codex_new_data.md，原单轨迹成功算法写Codex_tasks.md。

## 0. 2026-09-09 15:33 PDT 最新停点（优先于下方旧快照）

当前优先完成 Task3 FixedJoint 路线。原成功 Sweep2 是唯一已成功训练的固定基准，其资产、reference、80步零 residual 前缀、PPO/reward/Gate 和 checkpoint 均不得因新轨迹问题而修改。新轨迹统一使用已认可的 take9 薄入口簸箕、Sweep2 来源扫把及 take9 fixed 抓姿（右 `98_45`、左 `45_20`），对齐到各轨迹初始物体姿态；允许小幅位置/调平、约2 cm以内实用IK误差和剔除极少异常帧，但须检查连续性、穿模和扫把朝向入口。

take9 正式 run 为 `Task3Take9FixedFull_20260909`，tmux `task3_take9_fixed_full_20260909`，GPU1，24M、1024环境、seed42、从第1步开放 residual。已保存3M/6M/9M checkpoint。6M录像和 rollout 已证明训练输入错误：25 mm物块在首帧与扫把刷头重叠，781个扫把mesh顶点位于方块体积内，最近扫把表面点距方块中心8.9 mm，小于12.5 mm半边长；前0.5秒物块横向移动约20.7 mm并被击飞，前五步累计策略残差不足约0.3°，因此主要原因是初始几何重叠，不是PPO首步突变。旧选点器只优化第97行附近距离并检查簸箕走廊/地板，没有约束第0行至接触前的扫把安全间距。

用户已要求暂停 take9。15:33 PDT 发现自动录像清理曾删除手动 pause 标记，使进度意外从约9.63M继续推进；已重新写入 `/home/msc-auto/.cache/rl_correction_gpu1/pause.request`，内容 `manual_take9_review_20260909`。15:33:23 PDT日志确认训练再次释放GPU槽位，最终记录步数为10,682,368。接手先核对进度不再增长及GPU占用；不要删除该标记、resume或重复启动。下一步是重新选择cube起点，加入首帧至名义接触前的完整定向footprint/扫把mesh安全间距检查，再做短零残差物理回放确认开局静止；修复后须从头训练，不能把已受错误起点污染的checkpoint当作可恢复基线。

take32 已完成通用资产调平版：`configs/take_32_universal_fixed_semantic_v1.json`、`references/take_32_universal_fixed_semantic_v1.npz`，566帧、双臂IK可达率100%、未裁帧；仅按工作面法向减少簸箕约6.4°侧倾，回放在 `outputs_video/Task3_take32_universal_fixed_semantic_v1/full_zero/trajectory.mp4`。用户尚未在当前上下文明确验收该版，因此保持 `pending_visual_and_numeric_review`，不得启动训练。36、80的同类通用资产版本尚未完成。非fixed方案后置：左手继续fixed，右手再尝试整手握柄；当前不恢复该探索。

下方“当前运行状态”是较早文档快照，涉及“take9未启动”“32/80/36旧队列”等内容均只作历史背景，不能覆盖本节和实时核验。

## 0.1 当前任务总表

后续新轨迹范围明确只有：**take9、take32、take36、take80**。原成功 Sweep2 是冻结的独立基准，不计入这四条新轨迹；take128、take180、take304及其他候选不在当前任务范围。

任务1（当前阶段）：保持 fixed 方案。四条新轨迹统一使用已认可的 take9 薄入口簸箕、Sweep2 来源扫把及对应通用抓姿/资产，在各自初始帧对齐并做必要的小幅位置或调平；允许实用的约2 cm以内IK误差和极少异常帧剔除，但要避免穿模，检查扫把工作面、cube起点、接触时刻、簸箕入口/盆下误判。先修正 take9 的初始cube重叠，再按同一流程完成32、36、80并训练各自独立policy。

任务2（后续阶段）：尝试 non-fixed。左手簸箕继续 fixed；右手扫把改为手指全握住扫把柄的抓法，随后再评估摩擦和轻微收紧。当前不执行任务2，也不让其改动任务1或原成功 Sweep2。

## 1. 位置与首次检查

SSH msc-a6000，用户msc-auto，hostname mscauto-Lambda-Vector。唯一当前项目根目录/home/msc-auto/RL_sweep，分支task_sweep。本地目录只是访问入口，不创建源码镜像。先只读核验身份、路径、分支、git status、tmux及GPU进程归属，保留已有dirty修改。其他项目历史路径不能覆盖此处。

## 2. 阅读顺序和文档职责

以下均以/home/msc-auto/RL_sweep为根：
1. Codex_HANDOFF.md：交接入口与边界。
2. Codex_tasks.md：最重要的原Sweep2单轨迹成功算法、数据处理到训练录像的解释和运行步骤；已合并旧SWEEP_TRAJECTORY_PLAYBOOK.md，不再维护第二份手册。
3. Codex_mistakes.md：重大错误与避免复发，尤其COM、资产本体、刷毛方向和真实terminal录像。
4. Codex_commit.md：已有提交内容与回退定位，实际提交以git log为准。
5. Codex_new_data.md：Task3唯一详细状态，32/80/36成功步骤、128/180失败原因、三条训练与清理证据。
6. 按任务选读Codex_ablation.md、Codex_random.md、docs/TRAINING_DESIGN_GUIDE.md；不能让历史设计覆盖现行算法。

完整文档职责见Codex_tasks.md末节。历史REVIEW的待确认、缺canonical及旧路径不是当前结论。

## 3. 必须观看的证据

- Sweep2 ego：datasets/sweep_2_better/44ce97212ec9c5c97ad567934f473707.mp4
- 原始GraspPose render：datasets/sweep_2_better/sweep2_grasppose/demo_replay/
- Sweep2零残差：outputs_video/sweep2_v1_zero_replay.mp4
- full 24M成功录像：outputs_video/Sweep2AblationNoOfflineFull__20260903_policy_0024M/policy.mp4
- take32已验收：outputs_video/Task3_take32_v3/full_zero_retry1/trajectory.mp4，同目录closeup.mp4
- take80已验收：outputs_video/Task3_take80_v6/full_zero/trajectory.mp4
- take36已验收：outputs_video/Task3_take36_v4/full_zero/trajectory.mp4

阅读和视觉检查都不能省略。需对拍ego、原始GraspPose、make_prior静态姿态、完整物理回放，并核对candidate canonical → input OBJ → USD rigid root → world。IK可达不等于资产正确；刷毛要看全程。录像保持成功Sweep2视角方案。

## 4. 当前运行状态

2026-09-09最新停点：take9 FixedJoint 方案已完成簸箕薄入口资产并获用户认可。最终资产目录为 `tasks/Sweep/new_data/assets/take9_powerdisk/dustpan_thin_entry_20260909/`，详细说明见其中 `REVIEW.md`；完整无物块三机位回放在 `logs/task3_take9_powerdisk_20260908/thin_entry_preview_20260909/`。入口薄、两侧向前降低，轨迹以尽量贴桌为准，不要求全程零间隙。尚未用物块验证、尚未接入训练；明日继续时先读 `Codex_new_data.md` 的take9小节。不要恢复真实抓持方案，不修改原成功Sweep2。`configs/take9_powerdisk_fixed.json` 仍引用旧 `entryfix_v1`，在完成训练几何标定前不要删除该依赖，也不要把preview配置称作训练配置。

本轮没有启动新训练或留下take9 tmux。下面2026-09-06的32/80/36队列记录属于历史快照，接手时必须查实时状态，不能据此重复启动。

用户明确只训练新增32、80、36，每条独立full policy、seed42、1024环境、24M、每6M录像；不改原单轨迹算法，不resume。零残差均已验收，训练成功尚未确认。

- GPU1：take32正在训练，随后take80。tmux task3_train32_then80_20260906；队列logs/task3_training_prepare_20260906/run_training_queue.sh。
- GPU0：take36已排队，等待现有计算进程全部退出后启动。tmux task3_train36_gpu0_20260906；队列和日志logs/Task3_take36_full_seed42_20260906/queue_gpu0.sh、queue.log。
- 各自训练日志logs/Task3_take<ID>_full_seed42_20260906/；启动命令launch_train.sh，训练配置tasks/Sweep/new_data/configs/take_<ID>_training_v1.json。
- 128、180永久弃用，生成产物已清理、原始数据保留；304不在当前工作范围。今日任务已收尾，不自行开启新候选或重复启动已有训练。

状态是写入时快照，接手先只读查实时日志。训练配置启用方块；已验收trajectory配置仍隐藏方块、禁用碰撞及终止影响。两种入口不要混用。具体资产、reference版本和成功方法见Codex_new_data.md。

## 5. 保护边界与恢复资料

资产修复只在Task3副本；保留正确柄、抓姿、原始输入和成功Sweep2资产。用户允许丢弃明显异常重建，不允许任意改写正常运动掩盖失败。

不影响其他用户进程、不GPU reset、不广域终止、不破坏性Git操作。tasks/pregrasp/arm_shell_points.npz绝不能修改、stage或提交。保留既有Codex_tasks.md及共享代码dirty修改。训练已获授权，无需重新询问现有队列；不扩大到额外run或push。

GitHub身份HARROLDX，origin为git@github-harroldx:stzabl-png/RL-Correction.git；不用裸github.com默认身份，不读取或输出私钥。回退先查差异，不能对共享dirty工作树reset --hard。

旧文档和历史工作树patch：logs/task3_docs_cleanup_20260906/pre_edit/。本次日志删除明细：logs/task3_day_close_20260906/cleanup_manifest.json。128/180删除清单：logs/task3_reselect_20260906/cleanup_manifest.json、cleanup_extra.json。保留最终trace/JSON、成功训练、消融、expert数据和所有检查点；已删除的未跟踪二进制不保证可从Git恢复。


### 15 mm 实体回放结果（2026-09-09 晚间）

轨迹9 `take9_powerdisk_fixed_cube15_v1` 已完成180帧实体零残差回放；实体边长0.015 m、判定半边长0.0075 m，前1秒横向位移约2.4e-8 m，初始稳定检查通过。全程峰值速度0.792 m/s，后段碰撞仍须结合实际接触时序复核；不能据此宣称全程通过或训练成功。新训练尚未启动。结果：`logs/task1_fixed_20260909/take9_cube15_probe/summary.json`，视频同目录 `zero_cube.mp4`。下一步复核接触后启动take9新训练，再继续32、36、80。原Sweep2基准未改。

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
