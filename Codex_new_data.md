# Task3：新 Sweep 轨迹处理与验收记录

更新：2026-09-06。根目录 /home/msc-auto/RL_sweep；服务器 msc-a6000；分支 task_sweep。本文是 Task3 当前状态的唯一汇总。成功算法与通用处理方法见 Codex_tasks.md；新窗口导航见 Codex_HANDOFF.md。

## 1. 当前结论与阅读顺序

用户已明确验收 take32 v3 和 take80 v6 的双手抓姿、工具资产及完整无方块轨迹录像。take128、take180尚未通过完整重建轨迹检查，保留最新资产与失败证据后暂时跳过。本轮没有训练、resume或新的物理运行。

新窗口按顺序读：Codex_HANDOFF.md → Codex_tasks.md → Codex_mistakes.md → Codex_commit.md → 本文；需要方法来源时再读 docs/TRAINING_DESIGN_GUIDE.md。原通用手册已合并进 Codex_tasks.md，不再有第二份算法手册。

必须实际观看成功基准：
- ego：datasets/sweep_2_better/44ce97212ec9c5c97ad567934f473707.mp4
- 原始GraspPose render：datasets/sweep_2_better/sweep2_grasppose/demo_replay/
- 零残差：outputs_video/sweep2_v1_zero_replay.mp4
- 无离线预热完整方法24M：outputs_video/Sweep2AblationNoOfflineFull__20260903_policy_0024M/policy.mp4

| take | 当前配置 | 保留的结果 | 结论 |
|---|---|---|---|
| 32 | tasks/Sweep/new_data/configs/take_32_v3.json | outputs_video/Task3_take32_v3/full_zero_retry1/trajectory.mp4（同目录closeup.mp4） | 用户已验收，566行/28.30秒 |
| 80 | tasks/Sweep/new_data/configs/take_80_v6.json | outputs_video/Task3_take80_v6/full_zero/trajectory.mp4 | 用户已验收，449行/22.45秒 |
| 128 | tasks/Sweep/new_data/configs/take_128_v6.json | outputs_video/Task3_take128_v6/progress/ | 最新IK/资产诊断；内含最后录成的v4静态片段，非v6通过证据 |
| 180 | tasks/Sweep/new_data/configs/take_180_v3.json | outputs_video/Task3_take180_v3/progress/ | 新扫帚完成，完整IK失败；只有ego来源录像，无有效物理全程录像 |

“32/80成功”在本文指当前无方块轨迹阶段验收，不是新策略已训练成功。cube起点、接触/Gate几何、random smoke、训练和最终评测都尚未完成。

## 2. 目标、输入与不可改变的边界

输入：datasets/sweep_new_data/sweep_dustpan/{32,80,128,180}/。四条均为object_0=dustpan/left，object_1=broom/right；仍必须以ego实际抓法对拍，不能凭left/right字样或candidate排名选抓姿。

GraspPose交付：datasets/sweep_p4_grasppose_16takes_20260905/。交付时核验16884文件，整树汇总SHA256为793a0d9c25ec7f49692a8eb4e3ad8ae51262831b3ecf989012c2853ebeb675d0。region_rank.json已有canonical_frame，不缺simplified.json。候选grasp_qpos与squeeze_qpos为(1,29)，pregrasp_qpos为(6,29)。

四条最终各训练独立policy，统一沿用成功Sweep2 full算法：14维累计residual、confidence/human开启、前80control steps零残差、无离线expert预热、前10个在线critic-only epochs、随后PPO。既定后续预算为每条24M、6M录像，但目前禁止启动或resume。不是改成共享多轨迹policy，也不是改reward/PPO去掩盖坏输入。

当前回放cube隐藏、无碰撞、无reward/termination影响；schema内临时cube/contact字段仍是provisional。工具只允许共享初始场景注册，不单独旋转/冻结某工具，不删困难段。资产有明显错误可用其他轨迹正确资产替代；如果旧柄抓姿已正确，可以只换刷头。新资产在Task3独立副本中，源datasets与成功Sweep2资产不动。

## 3. 两条成功轨迹共同遵循的处理链

1. 看ego和重建render，明确人抓柄还是盆边，观察首帧、转腕过程和工作面方向。
2. 看原始GraspPose render；数值排名和functional标签只是筛选线索，必须匹配抓取位置、手掌朝向。
3. 转换canonical候选为input坐标prior，并渲染静态机器人手。令Rci为canonical_from_input，c为COM：

       p_input = Rci.T @ p_canonical + c
       R_input_hand = Rci.T @ R_canonical_hand

   contact位置同样加COM，法向只旋转；手指角不做空间平移。Task3实现为prior_frame.py，不能让旧的漏COM路径覆盖正确prior。
4. 核对candidate canonical → input OBJ → USD mesh → rigid root → world。USD若与OBJ已一致就不能再补180度；每一层都要用实际变换和点坐标检查。
5. 仿照成功Sweep2修可见簸箕mesh，保护抓区，重测入口；碰撞体开放，不使用封住mouth的单一凸包。
6. 共享注册双工具到桌面，保留完整源6DoF运动；通过固定工具—手关系得到腕目标，再用既有IK求双臂reference，同时生成human方向和confidence。
7. 做FK、限位、连续性检查。小的离散步长过大可同步加密两手和两工具时间轴，但保留原样本和运动；不能把有限位分支跳变用插值或unwrap伪装成可达。
8. 用trajectory_env.py初始化各自finger prior，检查reset与FixedJoint，录制完整zero-residual。实际root-to-world从physics tensor trace核验，不能把USD初始缓存误当实际动态状态。
9. 亲自连续播放到末尾，结合数值检查后交用户确认。32、80均完成此步骤；确认不扩大为训练授权。

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

## 6. take128：资产修好后，完整IK仍不连续

保留ego左手抓pan侧边的抓法；原pan+Y向上，不翻面，仅修入口并保护abs(x)>=65mm抓区。最新take_128_v6.json替换take32正确扫帚，Rz180°把刷毛映为input+Y，canonical/OBJ/USD检查通过。

失败证据按尝试顺序：
- 柄部候选43、62完整IK分别出现153.213°、173.603°跳变，即使各帧位置误差<2mm也不能使用。
- 最新v6出现350.664°跳变，右第3关节从-175.955°到+174.709°，在有限位两端；source217.5→218.0。不能unwrap越过机械限位。
- 7组共享XY与4组共享yaw/XY检查后，共享总yaw31°抽查双手18/18可达，但完整重算仍58.779°最大步长。关键帧抽查不能替代全程。
- v6首帧物理右手附着3.759mm>3mm（左.373mm），也未过。
- 源工具在source79→80/80→81旋转20.850°/19.099°，human wrist同期仅.121°/.094°，说明上游工具—人手重建不一致；尚不能把它当作全部IK失败的唯一根因。

当前暂时跳过，不宣称数学上不存在可行抓姿，不放宽8°连续性门槛，不删轨迹。后续应先复核源工具姿态及抓姿对腕可达域的影响，再做有限共享注册验证。

证据在logs/task3_assets_20260905/{discontinuity128.json,source128_orientation.json,right128_yaw31.log}及同目录注册检查报告；之前候选在logs/task3_remaining_20260905。最新输出progress内保留last_completed_static_v4/，明确只是最后录成的旧静态检查，v6没有有效完整录像。

## 7. take180：正确扫帚已替换，但完整目标与桌面/可达域冲突

原扫帚重建近似勺状。take_180_v3.json使用take32正确扫帚与对应抓姿，独立broom_donor32_v3；R=diag(-1,-1,1)，t=[-.01171733235,.01726142322,-.01937352774]m，scale=1，长轴不变，contact中心对齐。同步映射prior，工具root运动仍用180自身原轨迹。canonical/COM/OBJ/USD闭环、网格拓扑及静态抓姿图通过，首帧CPU IK可达。

但完整运动不通过：
- row310/source163.9167目标wrist z=.81953m，比桌面.87m低50.47mm；全程最低.81517m。同一ego附近人手在桌面上方。
- 新资产目标mesh最低-21.306mm（source138），是几何目标分析，不是成功物理回放。
- source159→160工具旋转28.936°/human1.655°；source163→164工具41.429°/human4.306°，上游运动存在明显不一致。
- 新donor同一困难行32初值0/32成功，最佳15.7772cm/29.0068°；7组共享XY注册，每组双手16pose仅11/16通过。
- 首帧Isaac自动单独平移pan约4.45mm，被Task3保护断言拦截，不能删除断言后算通过。

因此只保留最新资产、prepared/take180_registered_geometry_v3.npz和诊断证据，没有完整reference/机器人录像。后续从源tool pose与human wrist一致性继续；平面平移不能解决上述高度问题，不通过整体抬起簸箕离桌或抑制工具旋转掩盖。

证据：logs/task3_assets_20260905/{source180_geometry.json,registration180.json,ego180_frame164.jpg}；logs/task3_remaining_20260905/{audit180_donor_row310.log,static180_v3.log,donor180_grasp.png}。

## 8. 复现入口、共享算法保护与后续工作

环境：/home/msc-auto/rlcorr-venv/bin/python；PYTHONPATH=.；VEGA_URDF=/home/msc-auto/data/vega_urdf/vega_1p_sharpa.urdf；OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1；SHARPA_WANDB=0。诊断入口均在tasks/Sweep/new_data/。

take32原录制命令在logs/task3_take32_v2_20260905/launch_full_v3_retry1.sh；80在logs/task3_take80_brush_direction_20260905/launch_full80_v6.sh。需重新录制时使用新的out_dir/log，保留当前已验收版本，先检查共享GPU和自己的进程；不能直接覆盖最终录像。

CPU/几何：prior_frame.py、audit_grasp_frames.py、audit_take32_frames.py、audit_shared_registration.py、audit_trajectory.py；物理记录：record_trajectory.py、record_trajectory_overview.py、trajectory_env.py；完整物理审计：audit_take_replay.py。donor配置不能被source-only prepare脚本重写，否则丢失donor映射。

成功Sweep2默认160步HEAD/当前工作树对照：q、tools、row、gates、ref_arm、finger_q完全相同，reward/obs/priv最大差约1.066e-6/6.268e-7/6.557e-7，报告logs/task3_take32_v2_20260905/sweep2_default_regression.json。这是有限默认路径回归，不是完整24M重新训练。既有共享代码dirty修改仍被保留；本轮文档整理未改这些代码或arm_shell_points.npz。

后续：32/80轨迹验收已完成；128/180保持暂存。若进入训练接入阶段，需先单独裁定各自cube start与接触/入口几何、Gate绝对参数和小规模验证，再由用户授权启动统一full方法。本次不启动训练、不resume、不改算法、不影响其他用户作业。

## 9. 录像清理、归档及Git

2026-09-06按用户指令删除12个被取代的Task3视频，共7,003,314字节。32只留v3/full_zero_retry1总览和近景；80只留v6/full_zero总览；128最新progress保留ego及最后录成的v4静态；180最新progress仅ego并明确没有物理全程录像。Sweep2成功视频、消融视频、原始datasets、有效reference/prior/资产均保留。

旧视频旁trace/json/关键图片迁至logs/task3_docs_cleanup_20260906/old_video_evidence/，保持原相对层级。逐文件删除hash、大小与迁移路径见cleanup_manifest.json，Git跟踪副本为tasks/Sweep/new_data/CLEANUP_20260906.json。旧文档原稿和整理前working-tree.patch保存在该目录pre_edit/；旧视频已删除，不承诺Git能恢复这些未跟踪二进制。

旧REVIEW文件作为历史证据保留，开头指向本文；其中旧路径或“待确认”仅代表当时状态。当前以本文版本表和用户确认记录为准。Git本次记录文档合并、Task3成功/失败总结及清理清单；共享代码和大型运行依赖不混入文档提交。
