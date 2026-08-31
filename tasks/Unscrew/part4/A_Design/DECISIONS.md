# DECISIONS.md —— Unscrew(拧瓶盖) 实例全史权威

> 规则: 每个拍板、标定结论、事故验尸、判据改动都在此落账, 带日期;
> 结论被推翻时不删旧账, 加"★已推翻"引到新账。判据数值以 L3_Learning/progress.py
> 为准, 本文件记"为什么是这个数"。
> 前史: 旧扭盖任务 (recon_kailang 单 clip, Dyn1~23/U1~U39) 的台账在
> `tasks/recon_kailang/bottle_reconstruction/LEDGER_unscrew.md` (unscrew_bottle_newEnv
> 分支)。本实例是 V5 框架重新实例化, 移植其中已被数据验证的结论 (逐条注明 U 号)。

### T0-1 任务定义 —— ✅ 拍板 (2026-08-29)
- 成功 = G4: 拧开释放(G3) 后盖放到母带终点、瓶回位(placed), 双臂撤回站姿且
  物体不被碰倒碰歪。终态口径全在 progress.py。
- 数据引擎口径: 一个实例吃 datasets/unscrew_bottle 全部 17 条可用 clip
  (`UNSCREW_CLIP` 选条, 默认 32=README 榜首); **每次用一条数据训练, 用 RL 从
  粗糙重建轨迹恢复出一条物理可行的好轨迹, 批量导出演示数据**。
- 分工拍板 (2026-08-29 用户确认): cuRobo 只管机器段 (站姿→站位 Approach /
  收尾→站姿 Retreat, 残差冻结照谱); 从缝1到放盖收尾的**整个交互段**
  (抓稳/认证/带瓶/拧盖/放置) = RL, 参考 = 置信度物体轨迹 + 置信度人手轨迹。

### T0-2b 死通道零依赖 —— ✅ 拍板 (2026-08-30, 用户裁定"不用死的那个")
参考构建**只消费实测为活的通道**, 死的腕平移一个字节不碰:
| 通道 | 状态 | 用途 |
|---|---|---|
| 物体轨迹 + conf_pos/rot | 活 (盖 66cm/瓶 9cm) | 任务定义: 皮筋/时钟门/w_obj/成功判据 |
| 右腕四元数流 | 活 (59°) | 对握姿的 roll 种子 (U35 掌轴对准的底料) |
| 右手指流 + conf_fingers | 活 (44°) | 拧盖手法形状指引 (P-HYB, ×W_HCONF) |
| contact/affordance | 活 (2D 证据) | 场景摆放锚 (scene_layout) / 贴实奖金几何 |
| GraspPose prior (Screw27_body) | 非本数据 (Dexonomy) | **左手抓取全套**: 镜像腕位姿+指模板+squeeze |
| ~~腕平移 (双手)~~ | **死** (静态填充 ≤0.6cm) | **零依赖** |
| ~~左腕四元数流/左指流~~ | 活但弃用 | 描述的是"人的握法"且锚在死点上, 与 prior 握位几何不配 |
落地: 左腕 = Screw27_body 镜像 (q'=(w,-x,y,-z)) 锚瓶行, 绕瓶轴方位 ArmIK 可达率
扫描 (clip32: yaw*=240°, 与旧 clip grasp_prompt 接触区 150-240° 方位带吻合——
数据侧证据与 prior 侧选择互证)。手物相对位姿 by construction 恒定 (框架 v2 哲学)。
死通道若上游修复 (恢复 HaWoR 腕平移), make_reference 换回真腕轨迹是局部改动。

### T0-2 三 Prior 验收 —— ✅ (2026-08-29, 逐项核对)
- **物体轨迹+conf**: `object_valid_measured.npz` 的 conf_pos/rot_per_frame
  (0-100, 判据 ovm_v1)。clip32: 瓶 p50 88/63, 盖 p50 85/66, 拧盖窗多帧红档
  —— 三档体制正好吃这种数据。
- **人手轨迹**: ⚠ **腕平移仍是静态填充死数据** (clip32 实测全程位移 <0.6cm
  而盖走 66cm; 与旧 clip 同病, 上游未修)。腕姿态流(82°/59°)与手指流
  (44°/49°)是活的, 且带逐帧置信度 (`hand_confidence.npz` conf_pos/conf_fingers)。
  ∴ 交互段臂参考由**物体轨迹推导** (框架正统: v2 物体轨迹反解 IK);
  人手形状指引走**手指通道** (V5 文档预留的"拧瓶盖三指扭动 prior 通道"),
  权重 = W_HAND(物conf档) × W_HCONF(人手conf档)。
- **GraspPose**: 左手瓶 = `Screw27_body.npz` (含 squeeze 层) —— 本批 17/18 条
  CAD 与 water_bottle/screw27 **字节相同** (md5: 瓶 c9d18519 / 盖 c755e07c),
  prior 直接复用合法。右手盖 = 设定 B (盖 3.5cm 太小, GraspPose 处理不了):
  无 grasp prior, 指尖抓取由人手指流+affordance (`contact/expected_area_*`) 引导。

### T0-3 Gate 阶段机 —— ✅ 拍板 (2026-08-29)
- G1(+5) **只判左手** ≥3/5 垫稳 10 步: 演示时序右手在瓶拿起转平后才进场
  (clip32: 瓶 onset f22 / 盖 onset f32), 双手同判会让 G1 永不点火。
- G2(+8) 提升认证 (框架同款, 滑移只判左腕-瓶; 盖被螺旋钉在瓶上, 判双物 z 升)。
- G3(+10) = **拧开释放**: screw_assembly 的 detach 锁存 (拧满 turns 即脱开)。
  仿真螺旋角是唯一真值 —— 重建盖转角不可观 (螺轴对称, 数据集 README)。
- placed (不付奖) = 双物到**母带末交互行**位姿 (瓶3cm/15° 盖5cm/30° hold15)。
  ⚠ 不是静置位: 盖起点在瓶上、终点在桌上, rest≠end (与 Pour 的关键差异)。
- G4(+15 终局) 撤退归位, 框架同款。
- 每步件: adv/皮筋/工资/反射/斜坡/贴实(C线) 框架原样 + **r_screw = 2.0×Δθ**
  (双向计, 回拧扣分, U6 防刷分结构; 270° 总额 ≈9.4, 与 G3 的 10 同量级)。

### T0-4 参考体制选型 —— ✅ 拍板 (2026-08-29)
- 默认 **P-HYB** (V5 文档明示: 手内操作主导的拧瓶盖 → HYB), POUR_VARIANT=OBJ
  可跑消融对照。主档 tmix = **双物体短板** min(瓶,盖)×min(pos,rot)
  (README 排名同口径; Pour 只看主物体, 本任务盖是短板必须计入)。

### T0-5 螺旋抽象移植 —— ✅ (2026-08-29, 全部承旧台账实证)
走 `tasks/pregrasp/screw_assembly` 既有通路 (secondary 自动激活), 新加三个
默认关闭的钩子 (旧任务零影响):
- `screw_drive_gain` (U34+U39): 拇/食/中分级驱动 1指1/3速→3指全速, 无接触 0
  —— 无摩擦解析螺旋被亚阈值轻擦免费空转 (旧实测近随机策略 41% 假 release);
  slow-screw `max_angular_velocity=2.0 rad/s` (U34 用户裁定"不要太快的拧",
  20 rad/s 下单指戳 5 步拧完 = 拇指戳局部最优的制度性根源, pk22)。
- `screw_omega_damping=0.9` (v3 螺纹粘滞): 轻弹惯性立刻衰减。
- `screw_detach_at_full` (UnscrewRef1): 确定性策略拧满停手不该永卡在 ω>0 条件。
- `turns=0.75` (U30b): 演示实测 ~266° 分离; 2.0 圈是标准件假设 (难 2.7×)。
  逐 clip 可用 `UNSCREW_TURNS` 覆写 (本批各 clip 实际拧角未逐条测, 欠账见下)。

### T0-6 场景角色拍板 —— ✅ (2026-08-29)
`screw_primary="body"`: 瓶=env.object (listen-to-left-hand 摆放 + upright 投影),
盖=env.aux (reset_screw 合拢瓶顶) —— preengaged 的物理正确开局。若 primary=cap,
主体摆放机制会把盖摆到它自己的桌面锚点 (f32 时盖还在倾斜瓶顶的空中), 开局即错。
upright 投影 (U24a): 重建静置帧 ~20.5° FoundationPose 噪声 > 平底圆柱 18.3°
倾倒极限, 不投则瓶必自倒 (旧台账总根因, 参考侧与摆放侧同投影同口径)。
垫序与 Pour 相反: 前5=左vs瓶, 后5=右vs盖 (configure_cfg 按交互手=left 重建)。

### T1-1 β 剂量 —— ✅ 已诊断，后由 T2-4 裁定取代
初值 βL=2.0 来自 Pour17；clip32 已实测 β=1/1.5/2/3 四档。
结果与最终裁定见 T2-4。

### T1-2 母带 —— ✅ v1/v2 链路通（v2 实测见 T2-5）
- v1 (make_reference.py, 离线): clip32 全链 273 行 (app60/缝25/交互103/缝25/退60),
  自检家族五件全绿 (放音 G 链全通收入 38 / 批量逐位一致 / RSI 四点位 / 变体
  反向 / 体制)。盖脱离帧从数据测 (瓶体系相对漂移 >4cm): f35, 交互行 13。
  ⚠ 脱离检测**不能**用"投影轴×装配位"口径 —— 瓶姿态 20° 噪声投影后与原始
  盖位差 6cm, 会把脱离帧误报到首行 (自检实锤后改体系内相对量)。
  ⚠ 盖脱离后的行按**世界平移**换基 (与瓶同一平移), 不能按脱离点连续拼
  —— U24a 投影的座位高度差会把送放段整体压低 ~5cm, 盖末行破 D1 (自检实锤)。
- **已知欠账①**: 离线右臂 IK 达标率仅 2% (URDF 锚系统差 + 可能的 U35c 肘几何
  约束) —— v1 臂行是降级品。修复路径: probe_rest.py 拿实测 anchor_T 重跑
  make_reference; 仍塌则逐 clip 判 UNSCREW_HOLD_YAW (U35c 候补, 未实施);
  终解 = build_reference.py v2 在 Isaac 内站位捕获+增量空间 IK (框架正统)。
- **已知欠账②**: 机器段默认 smoothstep 占位 (无碰撞背书) —— plan_machine_segs
  产出 Approach/Retreat 后重跑 make_reference 即消。
- **已知欠账③**: v1 交互 1 行=1 重建帧, 20Hz 播放 1.33× 实时 (Pour17 v1 同
  口径); 高速窗时间扩张留给 v2。

### T1-3b cuRobo 本机自给自足 —— ✅ 收官 (2026-08-30)
"MagicSim 定制版"之谜破案: `curobo.motion_planner` 这套 API 就是 **NVlabs/curobo
新版主线** (v0.8+, warp 内核免编译) —— MagicSim 只是 vendor 了上游+加了机器人 yml。
本机配置 (全程无 MagicSim):
- 包: git clone NVlabs/curobo → ~/WorkSpace/curobo, pip -e (--no-deps) +
  cuda-core[cu12] 内核后端进 isaac env。⚠ isaacsim 钉 packaging==23.0/
  websockets==12.0, 装依赖时被拱过一次已还原; viser (websockets≥13) 只有
  RobotBuilder/Debugger 用 —— **重跑 yml 生成器时临时升 websockets, 完事还原**。
- 机器人配置: tools/make_vega1p_sharpa_curobo_yml.py 从仓库 URDF 自动生成
  `datasets/vega_urdf/vega_1p_sharpa_curobo.yml` (254 球/76 links; MorphIt)。
  臂/头网格自 github.com/luaiabuelsamen/vega_curobo (Dexmate 官方资产) 按名
  补齐 —— link 系与网格原点与我们 URDF **逐位相同** (L_arm_l2/l4 实测);
  躯干 = FK 推导手工球; 顺手修掉 meshes/vega_1p 指向他人机器的入库断符号链接。
- 排障两笔 (都进了生成器): ①猜名关节 F_wheel 不存在 → lock 按 cspace 过滤;
  ②站姿 5 对**亚毫米**伪碰撞 (肘弯 l5-l7 0.19mm + 相邻指节, 拟合球微突出) →
  IK 收敛 1e-7 但 success=False 的病根, 忽略表并入 (规划时手指锁死, 无害)。
- 冒烟 (tools/smoke_curobo_vega.py): 双臂 14 DoF / FK 合理 / 双工具位姿规划
  1.7s / 终点误差 0.0mm ★全通。worker 默认配置链: CUROBO_ROBOT_YML → MagicSim
  树 → 仓库自带生成品。
- 与 MagicSim 人工调的球存在差异: 首次真实规划若"过保守/起点碰撞", 用
  RobotBuilder.refit_link_spheres 排查 (登记为观察项, 不预修)。

### T1-3 cuRobo 机器段 —— ✅ 接线完毕 (2026-08-29; 缺包问题已由 T1-3b 解决)
- 链路: plan_machine_segs.py (Isaac 侧组 targets: 桌+双物障碍+新站姿躯干锁角
  40.52/73.66/0.39°) → curobo_plan_worker (干净子进程) → Approach.npz (双臂
  联合位姿规划到站位腕靶, obj_inflate 1cm) / Retreat.npz (cspace 直达站姿,
  物体钉在母带终位当障碍) → make_reference 自动消费, 缝1/缝2 焊接斜坡桥接
  规划解与离线 IK 解的分支差。
- ★上面“依赖 MagicSim 定制 fork / 本机未装”的初始判断已由 T1-3b **推翻**；
  当前 worker 首选本机 NVlabs/curobo 主线及仓库自带机器人 yml。
- ⚠ 旧 pregrasp_suite 的躯干锁角 45/90/0 是**旧世界值**, 本实例驱动已用新站姿;
  若复用 suite 本体, 先改这三个数。

### T2-0 Claude 交接审计 —— ✅ 代码收口；实跑见 T2-2/T2-5 (2026-08-30)
- 本地会话 `新配置搭建` (`ae387a7f-1d46-415b-94fa-b4bdc2dc634e`) 共落四个提交:
  `1de87065` 数据导入、`332e7c3e` V5 任务实例、`51b38f60` 死通道零依赖与
  cuRobo 自包含、`a0a299f0` cuRobo 机器人配置与冒烟。17 条 clip 可注册；新增
  17 份左手瓶 affordance 已逐点核对来源、归一化和有限性，属于有效未跟踪产物。
- 已交付的设计主线保持不变: cuRobo 仅负责 Approach/Retreat；缝1后的全交互由
  RL 残差处理；物体/置信度/右手指与姿态是活通道，腕平移死通道零依赖；四级
  Gate、58 动作/507 观测、三指接触分级螺旋驱动均已接线。
- 接管时发现并修正的假完成/串线风险:
  1. `probe_rest` 曾在未投影解析螺旋时直接 step，写出瓶盖同根位的假静置 JSON，
     且“预期 18cm”只打印不断言。无效文件已移到
     `/tmp/unscrew32_env_rest_invalid_20260830.json`；现在每步先投影螺旋、校验
     轴向/径向闭合 <5mm，再原子写入；并直接用基础环境启动，不依赖尚未生成的母带，
     消除 `env_rest → v1 → env_rest` 循环依赖。
  2. D6 使用一个正则源匹配多 body、过滤端却展开 5 项，触发 PhysX
     `expected 9, found 5`；改为四个显式远端 body 传感器，各过滤四个对侧远端体。
  3. 左抓 prior 曾预先镜像后又被 `make_reference` 再镜像；现只保留原始右手
     `Screw27_body.npz`，构带时恰好镜像一次。基类 approach/prior 脚手架与数据
     直立摆放冲突，已用受限 `bypass_lift_scaffold` 明确断开。
  4. 手工回放探针原先先 step 后 `apply_screw`，且从不刷新接触增益，导致螺旋
     永久零增益/释放不可能；v2 构建、β、IK、验收现与 DirectRLEnv 顺序一致。
  5. `probe_ikcheck`、`probe_acceptance` 当时改成了物理完美硬闸；
     **此项后由 T2-4 用户裁定取代**：IK 坏行只诊断，少于 3/4 数值稳定才返回非零。
     训练启动前仍硬验 v2、真实静置、双 cuRobo 段、clip/turns/β 与有限数组；
     训练稳定性通过才原子写绑定 v2 全文件 MD5 与现场完整世界指纹的
     `acceptance_v2.json`。训练入口在导入 IsaacLab 前验
     母带/凭据，建环境后再对现场世界，离线占位母带不能误发射。
  6. 世界指纹从旧 9 字段补齐为机器人/母带/IO/判据摘要/螺纹/方法参数/物性；
     关键字段缺失或不可读属于“未验”并拒绝续训、评测与录像，不再静默放行。
  7. cuRobo worker/机器段/v1/v2/静置/验收产物均改为原子落盘；机器段另写
     clip、段名、交互几何摘要和静置哈希，构带时逐项验 provenance，防止旧规划
     与新静置或新交互轨迹串线。
  8. 真正影响成败/螺旋接触的阈值集中到 `progress.py` 单一事实源，判据摘要
     schema=2；体制自检验证“阈值改动必改摘要、纯奖励改动不改摘要”。

- 离线复核: 17 clip 的注册/静态摆放/60k affordance 全部可读；静态重建单测
  11/11；五件判据自检全绿（体制件含 9 项）；clip32 离线审计母带 273 行，
  严格位置/姿态双门槛下右臂 IK 仅 2%（全行中位 14.61cm/19.7°）、左臂 56%；
  机器段为 smoothstep，文件只留 `/tmp`，未覆盖正式母带。
- 当前两张 GPU 持续满载既有训练/录像；未杀任何既有进程。真实 `probe_rest`
  及后续 T2-1 仍待卡空后按顺序执行，故此处不宣称 Isaac 验收完成。

### T2-1 冒烟/验收 —— ✅ 全序列完成 (顺序不可乱)
probe_rest → make_reference(重跑, 实测锚) → [plan_machine_segs ×2 →
make_reference(重跑)] → smoke_zero (A 静置对账 <5mm 铁则 / 机器段死线 0 误触)
→ probe_beta (βL 基线) → build_reference (v2) → probe_ikcheck (诊断) →
probe_acceptance (稳定≥3/4 后写凭据) → 训练入口预检/现场世界核对 →
训练冒烟 (双变体各短发)。

### T2-2 本机 Isaac 实跑: probe_rest→v1→cuRobo 机器段 —— ✅ (2026-08-30)
- probe_rest 实跑连闯三关: ① settle 循环每子步 `apply_screw` (缺投影时盖穿瓶
  掉桌, 假 rest 文件已在 T2-0 记案); ② 探针走基类, build_cfg 的 10 垫传感器
  撞上基类 finger_active(5) —— 探针把 `cfg.contact_sensors` 切回前 5 左垫
  (UnscrewEnv._setup_scene 同款处理); ③ 观测宽度 167 vs 占位 8 —— 实例上
  豁免 `_check_obs_dim` (探针不消费观测)。实测: 瓶倾角 0.0°, 盖-瓶轴向差
  18.0cm, anchor_T/站姿 44 指列入档。进程卡 `app.close()` 2 小时属已知病,
  kill -9 收尾。
- v1 实测锚重建: 右臂 IK 2%→13% (交互行贴人手流, v2 重铸域), 左臂 72%。
- **cuRobo Approach 首跑全灭 → 八轮二分定案** (全部离线 worker 复现, 免 Isaac):
  ① 去物体障碍仍败; ② 拆单臂: 右=位置✅姿态✅合体❌, 左=位置/姿态各自❌;
  ③ 关自碰撞 + 500 IK 种子仍败 ⟹ 纯运动学; ④ FK 交叉对账: cuRobo
  FK(ArmIK 左解) 中位 0.56cm/1.2° —— 两运动学链一致、env→base=+[0.5,0,0]
  正确、限位 0 越限; ⑤ cspace 直达已验证构型 + 空世界原地微动也被拒;
  ⑥ 自碰撞点名: 该构型无任何自碰; ⑦ 限位排查 = **真凶: ArmIK 钳限位出解**
  —— 左 j7 33 行钉 -79° 下限、右 j7 91/103 行钉 -64° 下限, "0.42cm 达标"
  是贴边换来的; cuRobo 带限位余量把贴边构型整族拒收。且左手镜像锚姿态本身
  超左腕 j7 行程 (现场 ArmIK 20 重启也只到 5.6cm/47.9°, j7 at_limit)。
- 三条修法 (均已落盘):
  1. make_reference 左抓方位扫描加 **限位内点判据** (全关节余量 >3° 才算
     达标行): yaw* 240°→270°, 内点可达率 50%。
  2. 机器段弃位姿 IK, 改 **cspace 直达内点 pregrasp 构型**: 收缩限位 3° 的
     ArmIK 解 station+净空 —— 左 = +5cm 径向 (距锚 7.82cm/53.6°, 左腕物理
     极限), 右 = +4cm 抬升 (0.37cm/2.8°; 8cm 抬升超可达域, 离线实测 ≤5cm
     才通)。构型存 `machine_pre_q_r/l` 入 v1, 纳入规划 digest。锚不可达
     部分由缝1+RL 消化 —— 数据引擎"从粗糙轨迹恢复"的正业, 不是缺陷。
  3. Retreat 起点 = machine_pre (**非**交互末行): 交互末行手贴瓶/盖/桌,
     碰撞检查器必判 start in collision (仅桌/仅瓶/仅杯微动全❌, 空世界✅);
     "松手撤离" 归缝2 数据斜坡, 与缝1 "贴近合拢" 对称 —— 缝 by design 是
     不做碰撞背书的桥, cuRobo 只认净空↔站姿。
- 结果: Approach 81 路点/终帧 0.00°/9.1s, Retreat 81 路点/0.00°/5.8s
  (均满障碍: 桌 + 充气 1cm 双物体, 物体钉各自段的母带位)。v1 全链
  273→316 行 (app 81/ret 82), 自检五件全绿。smoke_zero: A 静置对账
  瓶 0.20cm / 盖 0.26cm (<5mm 铁则过), obs (4,507), 机器段回放垫接触
  全零; ⚠ 800 步全程判定在 t=400 被人工中断 (共享卡让出), 待空卡补跑
  收尾行。
- ⚠ 遗留观察: 右臂交互行 j7 91/103 贴限 = 人手流腕姿对右臂运动学不友好的
  又一证据 (13% 达标同源), v2 Isaac 重铸 + RL 残差是既定救治路径; 若 v2 后
  仍贴限, 考虑对握锚绕螺轴 yaw 重定向 (离线实测 station 各 yaw 全可达,
  pregrasp 在 yaw+60°~180° 可达)。

### T2-3 placed 收紧: 护送判据 escort (2026-08-30 用户裁定)
- **问题** (用户看录像发现): 旧 placed 只判双物体终位姿 + hold15, "盖自己
  脱落/被甩到目标邻域" 也算成功 —— 大量假成功视频里右手根本没拿着盖放。
- **裁定**: 成功 = **右手拿着瓶盖放在桌子上**, 不是瓶盖自己落在桌子上。
- **机制** (只加一条, 一次一参数纪律; 首版"带上连击计数"被母带证伪 —— 人是
  转平低位拧开, 释放高度仅 0.92m, 带上方只剩 2cm 根本凑不齐连击):
  **过带单事件** —— 任何落到桌面的盖必恰好穿过放下带顶 (桌面+ESCORT_BAND=3cm)
  一次; 穿带那步若 [单步降>ESCORT_FALL=8mm(=16cm/s@20Hz) 且 无右垫-盖接触]
  ⇒ escort_fail 粘滞, placed 永不立。受控放下 = 带着接触穿带或慢放; 自由
  落体哪怕从带顶上 2cm 起掉, 穿带步降幅 >3cm 必抓, 与释放高度无关。
  新 step 输入 pads_r_cap (右垫-盖 ≥1 垫, env 从 _pads_f() 后 5 列取);
  None=停用 (离线探针兼容, env 必显式传)。
- **入账**: CRITERIA_SCHEMA 2→3 (ESCORT_BAND/ESCORT_FALL 进判据摘要);
  双版同步 (progress.py + progress_batch.py); diag/escort_fail 观察针;
  变体自检加 ④ 护送正路照常 G4 / ⑤ 自由落体 → escort_fail & placed 不立
  证伪对; 体制自检 schema==3 + ESCORT 阈值敏感性。
- **证伪信号**: 训练里若 sr/gate3 高而 sr/gate4≈0 且 diag/escort_fail 高
  ⇒ 策略只会拧不会放 (判据在拦真行为, 不是 bug); 若 escort_fail≈0 且
  录像仍见自由落体成功 ⇒ 判据漏(查 PAD_FTH 与盖的接触感度)。

### 观察针预登记 (防事后挑数)
- diag/screw_deg, diag/released, diag/n_triad, diag/gain, diag/cap_any (螺旋链)
- ep_rew/screw vs ep_rew/fshape vs 年金项: 引导不许压过任务核 (U4 承旧)
- sr/gate1..4 + sr_t0 口径 + n/ep_done (分母针, L5-17 纪律)
- 挤奶哨兵 (U29 系三次尸检的教训): 若 Mean Rewards ↑ 而 release/placed ↓,
  先查每步年金总量×回合长 ≈ Mean 否 —— 完成的一次性收益必须 > 放弃的剩余年金。

### T2-4 reference 是 correction 先验，不是零动作答案 (2026-08-30 用户裁定)

- **裁定**：重建轨迹存在穿模/穿孔、IK 偏差和零动作接触失败是可接受的；本任务
  本来就要用 RL correction 恢复物理可行好轨迹，不应先把 reference 修成答案。
- **实测基线**：`smoke_zero --steps 800` 完整通过（静置瓶 0.20cm/盖 0.26cm，
  obs=(4,507)，机器段接触死线 0 误触）；`probe_beta` 的 β=1/1.5/2/3
  在关键段末均为左垫 0、未持住。22 关节顺序已逐名核对且为 identity，排除映射错。
- **剂量**：βL 取 1.0，等于 Screw27_body 原始 squeeze，避免 β>1 对关节构型
  外推；βR 保持 0。探针失败入账，不阻塞 v2 或训练。
- **硬闸重定义**：`probe_ikcheck` 的 1cm/10° 坏行只作诊断；正式入口仍要求
  probe 静置、cuRobo 机器段、Isaac v2、有限数组和剂量/哈希一致。
  `probe_acceptance` 改写 `unscrew_trainability_v1` 凭据：零动作全链记录滑移、
  释放和终点误差，但只以 NaN/Inf、状态/速度/力发散为失败，要求稳定环境≥3/4。
- **最终物理正确性**：仍由 G1→G4、真实接触门控螺旋、escort 判据和独立 eval
  约束；放宽的是 reference 前置门槛，不是成功判据。

### T2-5 v2/训练/部署闭环 —— ✅ (2026-08-30)

- **v2 产物**：316 行，段长 81/25/103/25/82，MD5
  `0d78d12af9c426a629f96c570ae88983`。`meta_v2` 记录位置坏行 172、姿态坏行
  130、关键窗坏行 15；这些是 correction 基线，不是训练门禁。
- **IK 诊断**：右臂 94/103 坏行（关键窗 5），左臂 78/103（关键窗 10）；
  全数组有限，探针按裁定返回通过。
- **可训练性凭据**：4/4 环境有限且不发散；最大垫力 3.846–6.911N，最大物体
  半径 1.164m，最大关节绝对值 3.0711rad。零动作成功 0/4、滑移/终点偏差
  均完整写入 baseline，只作诊断。正式入口和现场世界指纹均通过。
- **PPO 实跑**：HYB、OBJ 都用 4 env × 32 horizon 完成 rollout 和一次优化更新，
  均产出可加载 `best.pth`；最终 checkpoint 的 342,871 个张量值全有限，
  TensorBoard 在 Isaac 关闭前 flush，world.json 与事件文件均非空。RSI 解锁
  交互冒烟另实测 HYB 的 leash/pen、OBJ 的 adv/leash/bonus/slope 非零，
  两者都有螺旋转角/释放诊断。
- **空分母修复**：首轮没有认证样本时 `sr/cert_pass` 原始值继续用 NaN 表示
  “未观测”，日志器点名跳过；课程 EMA 把未观测率保守映射为 0，避免 NaN
  永久污染解锁状态。离线回归与真实 PPO 均已证实。
- **部署**：启动器/验证器使用仓库相对路径、LFS/母带/凭据硬检查、用户独占
  Isaac 临时目录；`docs/DEPLOY_UNSCREW.md` 给出目标服务器安装、验收和
  HYB/OBJ 发射命令。

### T2-6 交接审计: 绿灯下的三层病 + 真实螺纹副落地 (2026-08-31)

**起因**: 用户要求复查上一轮交接 (`0c545894`) 的搭建情况。审计结论 = 代码
本身大多正确 (finite_rate/writer.flush/yml 去绝对路径/rest_anchor_T 逐位同),
但**判据被整体调软之后, 全链的"绿"不再含信息**: `probe_acceptance` 只查
NaN/发散, `probe_ikcheck` 只查有限 —— 这两条对"瓶已经躺在地上"的母带同样亮绿。
凭据自己记着的基线是: 左手滑移峰 43~47cm、瓶终态倾 90° (倒了)、拧角 0°、
盖终点偏 56~86cm、站位垫 L1/R0 (两只手都没抓上)。T2-4 "reference 不必零动作
成功"是用户裁定, 保留; 但**下面三层是"策略根本学不会"而不是"参考不完美"**。

#### 病一: 参考的臂行不可达 (不是噪声, 是运动学不可能)
逐行 FK 实测 (离线, ArmIK): 右臂对**自己的**腕目标中位差 44.7cm, 103 行里
100 行贴在关节限位上, 搬运段整段冻在同一构型 (ia50 与 ia102 的 FK 逐位相同)
—— 而盖参考自己飞到桌上。这与用户要的护送判据直接冲突: 照这条母带,
`placed` 永远立不了。三条根因, 全部离线可证:
1. **腕自转被过约束**。右腕绕螺轴的自转照抄人腕四元数流; 人前臂随手 270°,
   机器腕 j7 行程只有 143°。实测: 只解位置不管姿态, 搬运段 103 行**全部
   0.0cm 可达** —— 不可达的从来只有姿态。
2. **演示的自转在脱离后没有任务含义**。用户提供的数据事实: 盖的**旋转**
   置信度低 (位置高), 且脱离前盖与瓶同体。判据侧 `placed` 只看轴倾角
   (`_axis_tilt`), 绕自身轴的自转不可观也不可判 —— 它是**规范自由度**。
3. **IK 流水线三处自伤**: 失败解当下一行热启种子 (一次解崩拖垮整段) /
   "最近达标行顶替" 会跨分支跳 / 解完再 σ=1.5 高斯平滑把不同分支平均掉
   (实测左臂 0.14cm 的解平滑后差 9.3cm/62.7°)。

**修** (全部在 `make_reference.py`, 离线可复核):
- 右腕自转 = 解出来的量: 拧盖窗一个**常量抓握自转** (扫描, 刚性抓握),
  脱离后**逐行限速 6°/行 漂移**, 手与盖一起转 (不滑手);
- 盖脱离后姿态改为 **rigid ride** (跟着手走, 位置仍用重建);
- 瓶身自转在脱离后**冻结** (只保留轴的摆动, 平行输运实现 —— 逐步剔局部 z
  增量再积分会让轴跑偏, 单元测试实测轴差 0.74, 已在函数里断言拦住)。
  演示里人拧完还在继续转瓶 (末行相对脱离行 101°), 左手抓点跟着绕到瓶另一侧,
  左臂要"绕过瓶子"才跟得上 —— 这是左臂搬运段 6~21cm 不可达的真凶;
- 左抓方位扫描把**站位行可达**升为硬条件 (原来只看全程平均, 选出来的方位角
  站位行差 5~19cm —— 手根本没到瓶上, G1 三垫永远不成形, 就是凭据里
  "左垫 1 个接触" 的来源), 网格 30°→15°;
- IK 解算: 逐行热启 + 坏行**冻结**(不跳分支) + 连冻 2 行放行重新锁定 +
  20°/行限速 + σ=0.5 轻平滑后**重投影**(只在更准且不跨分支时采纳);
- `UNSCREW_HOLD_YAW` (U35c 持瓶朝向重定向) 逐 clip 标定并落进 `task_config`。

**结果 (clip32, hold_yaw=-35°)**: 右臂达标 12.6%→**77%** (全行中位
14.61cm→**0.25cm**/0.6°, 90 分位 3.18cm, 最大逐行跳变 173.8°→32.4°);
左臂 66%→**100%** (中位 0.09cm/0.2°, 0 冻结行, 0 贴限行);
左机器段 pregrasp 锚 7.82cm/53.6°→**0.37cm/0.1°**。
盖末行倾角由重建的 107° 变为 rigid ride 的 2° (自转不可判, 判据只看轴倾角)。

#### 病二: 假摩擦 —— "碰一下瓶盖就自己脱落"
本仓库的螺旋抽象仍是旧口径: 接触门 `screw_drive_gain` (有接触就白给转动)
+ `screw_omega_damping=0.9` (ω 每子步衰减 ≠ 力矩阈值)。老方法线 (worktree
`RLC-dynline`, `LEDGER_unscrew` U40) 已按用户裁定"最接近真实情况建模"修过,
**本仓库没有落实** —— 现在移植:
- 咬合期盖换**球形重惯量** I_eff=5e-3 kg·m²: 指尖→盖的可传扭矩由 PhysX
  摩擦锥真实裁决 (≤μ·N·r, 捏得紧才传得多), 脱扣即还原实物惯量;
- 解析螺纹阻力: 静锁 breakaway 0.04 N·m (已破封的松盖量级) + 库仑 0.015 +
  粘滞 0.03 (τ=0.075 → 稳态 2 rad/s ≈ 人手拧速), |ω|<0.05 且 τ<阈值 回锁;
- 力矩估计三条防幻影 (老台账三次带毒尸检的教训, 逐条移植): 只看盖侧 /
  按矢量差分再投影 / **零接触即强制归零**的真值门;
- 反作用扭矩回瓶身 (左手必须抗扭); `max_angular_velocity` 退化成 4 rad/s
  安全夹; drive_gain 仍逐步计算 (它是观测里的接触特征, obs 507 维不变),
  只是不再乘进角速度。
- 参数进 `clips.py::_unscrew_take`, `ScrewSpec` 扩字段, 体制自检 ⑦ 断言
  整条参数流进 runtime spec (缺一个就退回假摩擦口径)。
- 新增 `B_SmokeTest/probe_thread.py` (H40.0 五段物理探针: 静置不转 / 阈下
  锁死且估计器校准 / 阈上解锁且稳态转速对得上解析值 / 撤力回锁不回退 /
  持续阈上拧满脱扣) —— 训练前必过。

#### 病三: 成功判据只拦住了"甩下去", 拦不住"慢慢松手"
T2-3 的过带单事件判据只能抓"无接触快速穿过放下带顶"。老方法线 U41 的同一条
用户裁定其实是三件: 还要**真的拿过** (释放后 ≥2 垫触盖累计 ≥10 步) 和
**放得住** (释放后单步降幅峰值 <4cm/步 = 0.8m/s)。移植后 `CRITERIA_SCHEMA`
3→4, 变体自检从 5 件加到 7 件, 并逐条断言**是哪一条**拦下的:
⑤ 脱手自由落体 → 过带; ⑥ 无接触慢沉 (降幅 <ESCORT_FALL, 过带抓不到)
→ 持盖步数; ⑦ 拿着但半途 6cm/步 坠落再接住 → 降幅峰值。

#### 其它收口
- `reference_planning_digest` 收窄到规划**真正消费**的量 (machine_pre/station/
  末行障碍位置); 原实现把整段交互行一起哈希, 与自己的 docstring 矛盾, 逼出
  无谓重规划。machine_pre 一旦变 (本轮就变了) 仍照样触发重规划。
- `probe_acceptance` 保持不阻塞 (T2-4), 但凭据新增 `trainability` 台账并把
  告警显式打印: 站位垫不足 3 / 右手零接触 / 回放拧角 ~0 / 零接触却有传入
  力矩 (幻影探测器) / 母带冻结行过多。绿灯从此有信息量。
- `launch_remote.sh` 的 `RL_ISAAC_NO_GUARD=1` 改成可覆写 (共享机要排队闸);
  `wandb_writer` 的非有限标量按 tag 只报一次; cuRobo yml 相对路径按仓库根
  解析 (不再依赖 cwd)。

#### 假设与证伪信号 (DESIGN_LOOP 台账, 训练开跑前登记)
- **H6.1 (参考可达性是 G1 的前提)**: 站位垫 L≥3 且 `sr/gate1` 在 +2M 内破 20%。
  证伪 → 左抓 prior/β 还有几何错, 回头查 pregrasp 内点构型与 squeeze 剂量。
- **H6.2 (真实螺纹副下拧转必须靠捏紧)**: `diag/screw_tau_mNm` 的分布随训练
  上移, 且 `diag/cap_any`≈0 的步上 `screw_tau_mNm`≈0 (幻影探测器)。
  证伪 (零接触仍有力矩) → 又漏了一条数值通道, 停训先修。
- **H6.3 (breakaway 0.04 N·m 不是天花板)**: +8M 时 `screw_tau_mNm` 95 分位
  ≥30。证伪 → 降到 0.025 或加指压课程 (老台账 H40.1 同款处方)。
- **H6.4 (反作用扭矩不砸左手)**: `term/D4_slip`、`term/D1_drop` 不高于本轮
  基线 2×。证伪 → `react_on_bottle` 降权或左手 squeeze 加量。
- **H6.5 (U41 三件不误伤)**: `diag/carry_steps` 在 release 后能涨到 ≥10,
  `diag/fall_peak` 中位 <4cm/步。证伪 (carry 恒 0) → 说明策略拧开就撒手,
  是 r_chold 类的持盖年金缺失, 而不是判据错。
