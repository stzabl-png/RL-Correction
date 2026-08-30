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

### T1-3 cuRobo 机器段 —— ✅ 接线完毕 / ⬜ 本机缺包 (2026-08-29)
- 链路: plan_machine_segs.py (Isaac 侧组 targets: 桌+双物障碍+新站姿躯干锁角
  40.52/73.66/0.39°) → curobo_plan_worker (干净子进程) → Approach.npz (双臂
  联合位姿规划到站位腕靶, obj_inflate 1cm) / Retreat.npz (cspace 直达站姿,
  物体钉在母带终位当障碍) → make_reference 自动消费, 缝1/缝2 焊接斜坡桥接
  规划解与离线 IK 解的分支差。
- ⚠ worker 依赖 **MagicSim 定制版 cuRobo** (`curobo.motion_planner` API,
  非 PyPI nvidia-curobo), 本机未装、盘上无检出。跨机口径已参数化:
  `MAGICSIM_ROOT=<检出路径>` 或 `CUROBO_ROBOT_YML=<yml>`。装法: 从有 MagicSim
  的机器拷 `Third_Party/curobo`, pip install -e 进本 isaac 解释器。
- ⚠ 旧 pregrasp_suite 的躯干锁角 45/90/0 是**旧世界值**, 本实例驱动已用新站姿;
  若复用 suite 本体, 先改这三个数。

### T2-1 冒烟/验收 —— ⬜ 待跑 (顺序不可乱)
probe_rest → make_reference(重跑, 实测锚) → [plan_machine_segs ×2 →
make_reference(重跑)] → smoke_zero (A 静置对账 <5mm 铁则 / 机器段死线 0 误触)
→ probe_beta (βL 标定) → build_reference (v2) → probe_ikcheck →
probe_acceptance → 训练冒烟 (双变体各短发)。

### 观察针预登记 (防事后挑数)
- diag/screw_deg, diag/released, diag/n_triad, diag/gain, diag/cap_any (螺旋链)
- ep_rew/screw vs ep_rew/fshape vs 年金项: 引导不许压过任务核 (U4 承旧)
- sr/gate1..4 + sr_t0 口径 + n/ep_done (分母针, L5-17 纪律)
- 挤奶哨兵 (U29 系三次尸检的教训): 若 Mean Rewards ↑ 而 release/placed ↓,
  先查每步年金总量×回合长 ≈ Mean 否 —— 完成的一次性收益必须 > 放弃的剩余年金。
