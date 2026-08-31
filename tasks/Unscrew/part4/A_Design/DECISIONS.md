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

### T1-1 β 剂量 —— ⬜ 初值未标定
βL=2.0 (瓶 0.53kg 与 Pour17 重物侧同级, 抄其实证值), βR=0 (盖侧无 prior)。
**必须 probe_beta.py 复标后回填** —— 剂量窗两端都要测。

### T1-2 母带 —— ✅ v1 链路通 (2026-08-29) / ⬜ v2 待 Isaac 标定
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

### T2-0 Claude 交接审计 —— ✅ 代码收口 / ⬜ Isaac 实跑待空卡 (2026-08-30)
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
  5. `probe_ikcheck`、`probe_acceptance` 原先即使失败也 exit 0；现关键拧盖窗
     >1cm / >10° IK 坏行或验收少于 3/4 会返回非零。训练启动前另硬验 v2、真实静置、
     双 cuRobo 段、clip/turns/β 与有限数组；验收通过才原子写绑定 v2 全文件 MD5
     与现场完整世界指纹的 `acceptance_v2.json`。训练入口在导入 IsaacLab 前验
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

### T2-1 冒烟/验收 —— ⬜ 待跑 (顺序不可乱)
probe_rest → make_reference(重跑, 实测锚) → [plan_machine_segs ×2 →
make_reference(重跑)] → smoke_zero (A 静置对账 <5mm 铁则 / 机器段死线 0 误触)
→ probe_beta (βL 标定) → build_reference (v2) → probe_ikcheck →
probe_acceptance (≥3/4 后写验收凭据) → 训练入口预检/现场世界核对 →
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
