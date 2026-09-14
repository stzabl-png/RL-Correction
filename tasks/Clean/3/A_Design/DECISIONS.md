# Clean/3 设计决策日志 (擦盘子: 抹布/海绵 × 盘)

任务源起 (2026-09-05 用户): clean_tableware 五条 manifest 全绿, "按 Pour 格式新建任务体, 可能有数据缺失"。
本任务体版式 = `tasks/Pour/17` (A_Design L1/L2/L3 + B_SmokeTest + C_Wiring), 通用设计 G-A/G-B 适用。

## 0. 数据核账 (2026-09-05, 实测于 egodex_auto/clean_tableware)

| take | obj0 conf(pos/rot) | obj1 | 手 L/R | 物体数 | C1 抓握核心 | 判 |
|---|---|---|---|---|---|---|
| **3 ★主条** | 86.5/0.0 good | 73/73 good | 85.7/90.5 | 2 | 右×obj1 [113,299] 5指全片 + 左×obj1 [0,186] 4指 | README 钦点最佳 |
| 1 备条 | 89/0.0 | 57/12 mixed | 81/95.2 | 2 | 左×obj0 + 右×obj1 | obj1 弱 |
| 0 / 8 | 83 / 69 | — | good | **1 (v17A 漏标)** | — | 单物体, 出局 |
| 18 | 62/3 | 52/44 | 90.5/95.2 | 2 | 四(手×物)对全片 | 最弱重建, 接触信号最丰 (留作接触蒸馏源) |

take 3 实测: 300 帧; rts npz×2 (物轨+conf ✓); 双手 DexPilot ref_qpos 300/300 ✓;
**静腕变体证实** (双腕全程位移 0.2~0.3cm, 与 unscrew17 同款 ARKit 病 —— 按 18 号裁定用掌心代理/只借增量);
obj0 = 11.8×1.6×12.1cm (**目检坐实=黄色塑料碗**, f170 抽帧; 网格厚 1.6cm 偏扁—重建可能只得碗底盘面, A1 入库时核网格完整性); obj1 = 3.8×1.9×6.6cm (**目检坐实=洗碗刷**)。
语义 (f170 目检): **左手扶碗, 右手持刷在碗内刷洗** —— 拍板⑤已有预答, 待用户复核。场景为塑料餐具桌 (散落餐具+瓶)。

## 1. 数据缺口清单 (如实, 按阻塞度排)

| # | 缺口 | 阻塞 | 谁补 |
|---|---|---|---|
| D1 | **Dexonomy GraspPose 池全无** (`DELIVER/` 无 clean 条目; take 已备好 `contact/grasp_prompt.{json,md}` + C1 接触路线可直接喂) | 阻 L1-3/母带 | 用户跑 Dexonomy (同 unscrew C1-A 流程; 选型时**装臂最低点筛** `lowest_point_screen.py` 直接当过滤器) |
| D2 | 纹理产物无 (`objects/object_*/textured/` 五条都没有) | 只阻观感 (U15/U16 路线) | 上游 SAM3D 纹理步 |
| D3 | 静腕 (双手 0.2~0.3cm) | 不阻 (已有成法) | 设计侧: 掌心代理 + 首帧焊接借增量 |
| D4 | 海绵是形变体, 网格是刚体扫描 | 设计拍板 | 刚体近似 (pour 液体代理先例); 质量/摩擦无实测 → G-A 规矩 0.1kg/μ5 起步, 待议 |
| D5 | scene_layout/keyframes 无 | 入库时我算 (unscrew17 同款, 非上游缺) | 我 |

## 2. 任务定义草案 (待拍板)

- **动作语义**: 右手持海绵在盘面擦拭; 左手角色目检后定 (扶盘 or 递海绵)。
- **成功判据代理** (无流体/污渍, 同 pour "落点代替流体" 哲学): 提案 =
  海绵-盘接触 (垫力>阈 ∧ 海绵底面贴盘面) 的**覆盖率×行程**: 盘面栅格化, 接触印记扫过格数
  ≥ X% ∧ 擦拭行程 ≥ Y cm ∧ 盘位姿保持 (|Δpos|<3cm, 倾角<15° —— 盘不能被擦跑/擦翻)。
  X/Y 从母带零动作回放实测标定 (L5-27 铁则: 参考自己必须做得到)。
- **保持物体 Pose**: 盘 = HOLD_POSE 同款渐进罚全程 (擦拭时盘是"被保持者"); 海绵 = 交互期豁免。
- 起步姿态: 对称站姿 → cuRobo 接近 (Pour L1-4 版式, LeftApproach 构建器可直接复用于单臂)。

## 3. 待拍板 (★=阻塞开工) —— **2026-09-07 已全部拍板, 见 §5.0**

① obj0/obj1 身份已抽帧预判 (碗/刷), 待你复核; ② ★Dexonomy 池 (D1, 用户侧);
③ 判据代理数字 X/Y; ④ 海绵刚体近似 + 质量摩擦; ⑤ 左手角色 (预判=扶碗); ⑥ 单臂先行还是双臂同期。

## 4. 施工顺序 (Pour 版式映射)

A1 入库 `datasets/clean_tableware/3/` (staging + scene_layout/keyframes 重算) → A2 先验转换
(Dexonomy 池到位后, make_prior + 装臂最低点筛) → A3 分段器过 take 出五段帧界 → A4 母带
(cuRobo 接近 + 六级梯 + 交互窗按 rts + 桌面净空钳制全顶点版) → A5 判据 (progress 双版 + 自检族)
→ A6 把关链 (smoke 零动作/放音铁则/站位物立硬门) → 发射。B 拍板点穿插同 Unscrew §10 五点制。

## 5. 拍板与规划 v2 (2026-09-07; 用户裁定: 持物起步 / 真摩擦握 / 尺度不动 / 盘 0.3kg 海绵 0.05kg)

### 5.0 拍板记录

| # | 问题 | 裁定 | 来源 |
|---|---|---|---|
| ① 物体身份 | obj0 = **盘子** ⌀18cm (Dexonomy `clean3_plate18_ped`, 用户 09-07 定 18cm), obj1 = **洗碗布/海绵** 7.6×3.8×13.2cm (`clean3_spongeB`, 剥内壳) | 用户口径; §0 "碗/刷"旧判作废 |
| ② GraspPose | 盘(左手): `Dexonomy/output/DELIVER/clean3_plate_left/grasp_data/27_Quadpod__1_42_grasp.npy` (rank 18/114, geo 0.371, 拇指压面+食中无名托底, Isaac 抬 4.2cm@50g/μ3); 海绵(右手): `DELIVER/clean3_sponge_right/grasp_data/11_Power_Sphere__3_2_grasp.npy` (rank 3/20, geo 0.436, 底面零接触, Isaac 抬 7.0cm@50g/μ3) | 用户点选 |
| ③ 尺度 | 盘 ×0.75 (18cm) 与世界/海绵 ×1.99 **不一致, 保持现状**; 擦拭路径出盘沿(约 3cm) **不处理** | 用户 |
| ④ 起步 | **双手持物起步, 无 approach/grasp 段** (视频全程持物) | 用户 |
| ⑤ 握持 | **真摩擦握**, 不用 FixedJoint 焊接 | 用户 |
| ⑥ 盘姿态参考 | 常量: 水平 + 小倾角 (rts conf_rot 均值 16, 姿态轨不可用) | 我荐, 用户准 |
| ⑦ 物理 | 盘 **0.3kg**, 海绵 **0.05kg**; 摩擦**未议, 假定 G-A μ1/μ1** (物/指垫) | 用户(质量) / 假定(摩擦) |
| ⑧ 左手角色 | 静态扶盘 (位置轨低通, conf_pos 75) | §3⑤ 预答 |

数据侧现状 (2026-09-07 复核): 双腕冻结 0.2cm; 缩放后 C1 接触路线 `contact_fingers.json` 四组全 `no_grasp_core` → 示范抓姿不可用, 只能用 Dexonomy 几何池 (demo_angle=null); 海绵 rts pos/rot 67/67 可用; 盘 rts pos 75 / rot 16。
`contact/grasp_prompt.*` 与 `expected_area_*`/`affordance_*` 生成于 09-04 22:33 最终缩放之前, 尺寸/路径过时, 不再引用。

### 5.1 版式: Pour17 v5 / Unscrew part4 框架身, **不是 Sweep2**

- Sweep2 = FixedJoint 焊接 + 14 维臂残差, 手指不受控 —— 与 ⑤ 冲突, 只借其 **物轨→腕参考** 的母带思路 (`build_reference.py`: T_world_hand = T_world_obj × inv(T_obj_hand), 连续臂 IK, 时间展宽 0.8cm/3.5° 每步)。
- 真摩擦握 ⇒ 58 维 [R臂7, L臂7, R指22, L指22] + HOLD_POSE (盘) + GRIP_SHAPE r_opp/r_follow (L5-36) + 双手垫传感器各对己方物体 + D6 跨侧互撞 + `bypass_lift_scaffold`。**接线母本 = `tasks/Unscrew/part4/C_Wiring/task_env.py`** (双手各持一物, 左 Object / 右 Aux, 同构)。
- **唯一新机制 = 手内复位 `_settle_inhand_reset`**: 臂 = 母带 0 行 IK; 物体按 T_world_obj = T_world_hand × inv(T_obj_hand) 写入; 指目标 grasp_qpos → squeeze_qpos 沉降 N 步 (复刻 Dexonomy Isaac lift 协议); 审计物-手相对位姿漂移。= Sweep2 `_settle_attachment_reset` 去掉 FixedJoint。
- 质量按物体分别覆写: 现 `apply_phys_rule` 是 `POUR_OBJ_MASS` 单值广播到全部 named_bodies, 需扩成按名给值 (向后兼容单值), 横幅仍 grep `质量=0.300` / `质量=0.050`。

### 5.2 施工序 (A0 新增, 先于一切)

| 段 | 内容 | 出口判据 |
|---|---|---|
| **A0 握持探针** | 在本仓 env 按任务物理 (0.3/0.05kg, μ1/1) 做手内复位 → 静持 3s → 母带擦拭零动作放音 | 两物体相对手漂移 <1cm/10°, 不掉, 盘倾角 <15°。**Dexonomy 的 50g/μ3 抬升不算数**。失败分支: 盘换 13_Precision_Sphere 族 (README: 抬起后保持最平) 或其余 30 个 Isaac 过关候选; 海绵换其余 7 个 |
| A1 入库 | `datasets/clean_tableware/3/` 按 `sweep_2_better` 布局拷 take; 网格换 Dexonomy 两份 (`clean3_plate18_ped/mesh/simplified.obj`, `clean3_spongeB_11_Power_Sphere/mesh/simplified.obj`); 底座不入 (仅合成脚手架); scene_layout 按 env 桌高重算 | manifest 存在性强制核 |
| A2 先验 | `tasks/pregrasp/make_prior.py --grasp_npy … --info_json (region_rank canonical_frame)` ×2 → `priors/Clean3_plate_left.npz` / `Clean3_sponge_right.npz`; `reach_filter` 双臂可达 | IK 残差 <0.5cm (Sweep2 教训: 钉死 yaw 假报够不着 27.8cm) |
| A3 母带 | 右臂 = 海绵 rts 轨 × inv(T_obj_hand) → 臂 IK; 左臂 = 盘位置低通 + 常量姿态 (⑥) → IK; 指行 = squeeze 常量; 海绵擦盘面对盘面 **z 钳** (不穿盘, 名义压深 0~2mm), **XY 不钳** (③); 时间展宽 0.8cm/3.5°; 第 0 行即持物就位 | 四道闸 + 零动作放音 |
| A4 判据 | 覆盖率 × 行程 (盘面栅格, 擦盘面贴合且法向对齐计"擦到") + 盘 HOLD_POSE 渐进罚 + 双物掉落 = D1 终止; 出盘部分不计分不罚 | X/Y 由零动作放音标定 (L5-27 铁则) |
| A5 接线 | `clean_env.py` (Unscrew part4 母本): Object=盘(左) Aux=海绵(右); 去螺旋组件; 进度换 wipe; 按名质量覆写; world_fingerprint 同款 | 自检族过 |
| A6 把关 | smoke: 手内复位审计 / 零动作放音持物全程 / 判据自检 / 横幅 grep 质量=0.300 与 0.050 + 摩擦=1.00/1.00 + 指垫 1 | 全绿再报批发射 |

### 5.3 可证伪信号 (DESIGN_LOOP)

| 假设 | 证伪信号 | 应对 |
|---|---|---|
| H1 Quadpod 盘握在 0.3kg/μ1 下静持得住 | A0 静持漂移 >1cm/10° 或掉 | 换候选 (5.2 A0 分支) |
| H2 擦拭反力不把海绵顶出 / 不把盘压翻 | A0 放音期间盘倾角 >15° 或海绵漂移 >1cm | 降压深 → 换候选 |
| H3 常量盘姿态 + 海绵原路径可物理执行 | 零动作放音 progress < 标定 X/Y 的 80% | 母带重做 (压深/展宽) |
| H4 指残差 + HOLD/GRIP 足以保持 | 训练 term/drop 率不随步数下降 | 收紧指残差界 / 加 squeeze 前馈 |

### 5.4 已知风险
- **盘杠杆**: 海绵压盘心, 支点在盘沿指托, 力臂 ≤6cm; `27_Quadpod__1_42` 接触最近 3.25cm 到盘心, 是杠杆最小的一族; 仍以 A0 实测为准。
- **尺度 ③**: 海绵路径可能出盘沿 ~3cm; 覆盖率按盘面计, 出盘不计不罚。
- **摩擦未议**: 本规划假定 μ1/μ1; 若用户另有要求, A0 前改。
- 真摩擦握下 Dexonomy Isaac 抬升结论 (50g/μ3, 手-桌碰撞关) 只作候选初筛, 不作训练物理保证。

### 5.5 开工实录 (2026-09-07, A0~A4 用户批准全段开工)

**数据新发现 (影响 A3 设计, 先记后做):**
- 重建里海绵**不贴盘面**: 盘系法向距离中位 12.6cm (世界 dz 中位 10cm), 盘系面内偏移 2.8~6.3cm (pct10~90)。
  单目深度沿视线不可观测, "贴合"这一维是重建最弱的维。⇒ 母带的海绵高度不能"钳", 必须**按盘面径向剖面重新贴合**
  (top(r) + 擦盘面偏移 1.91cm − 压深 2mm)。这是对 §5.2 A3 "z 钳"的修订: 由钳改贴。XY 按用户裁定不动。
- 盘 rts 法向倾角稳定 37~43° (pct10~90): 圆盘绕法向自转不可观测 ⇒ conf_rot 16 是"自转"的分, 法向倾角本身
  可能是真的 (人把盘朝自己倾着刷)。但用户已裁定盘姿态常量水平, 倾角只作记录 (`--plate_tilt_deg` 留口)。
- 海绵擦盘面法向离竖直 18~32°, 速度中位 3.4cm/s (最大 11.4), 15fps×300 帧 = 20s。
- 盘/海绵网格与 Dexonomy 同框核验: `plate_0.18.obj` = take 盘网格逐顶点 ×0.75 (max|dv| 5e-9);
  `sponge_outer.obj` 与 take 海绵同包围盒。Dexonomy `processed_data/*/mesh/simplified.obj` 是**规范系**(z-thin), 不能直接入库。
- 盘径向剖面 (输入系 y): r<4.5cm 碟心平底 y=−0.73cm, 4.5~8.5cm 斜坡升到 +1.22cm (缘)。

**A1 入库 ✅** `datasets/clean_tableware/3/`: take 全套 (去 mp4/framescan.log/_prescale_backup) + 网格换 Dexonomy 输入系两份
+ 纹理视觉 (盘 OBJ/USD 缩放 ×0.75; 海绵原样) + `retarget/textures/`。clips.py 注册 `Clean3_plate` (主=盘/左) `Clean3_sponge`。
物理语义 mass 0.30/0.05, friction 1.0。scene_layout.json 由母带构建器写 (RL 系位姿, scene_table_z=rl 桌高 ⇒ aux 换桌高公式恒等)。
**A2 先验 ✅** `tasks/pregrasp/priors/Clean3_plate_left.npz` / `Clean3_sponge_right.npz` (make_prior, dexonomy conda python; numpy1 可读)。
盘腕(输入系) pos=[−0.146, 0.061, −0.050]; 海绵腕 pos=[0.050, 0.040, −0.100]。
**A3 母带** `tasks/Clean/3/A_Design/L2_Reference/build_reference.py`: 盘=位置低通(0.67s)+常量姿态; 海绵=盘系面内图案
(`--sponge_xy_mode plate`) + 贴合高度 + 面内 yaw ψ(t); 腕=物×GraspPose; 连续臂 IK (Sweep2 算法); 20Hz 时间展宽;
`--scan_yaw` 双臂可达扫描定 (盘 yaw, 海绵 yaw 偏置)。配准: 盘心 (−0.12, +0.06, 桌+0.12), scene_yaw 0, 锚用 Sweep2 实测 arm_center。
**A0 探针** `tasks/Clean/3/C_Wiring/{hold_env.py, probe_hold.py}`: 手内复位 (物体按实测手位姿 × inv(T_obj_hand) 放入,
grasp→squeeze 24 步斜坡) → 静持 3s → 母带放音; 记漂移/盘倾角/擦盘面贴合/覆盖率/行程; 盘碰撞体显式 128 hull + shrinkWrap
(浅碟防桥平)。物理: POUR_OBJ_MASS=0.3 (主体盘) / 海绵 0.05 (aux USD 烘焙) / μ1×3 (右垫经 extra_supergrip_bodies 补绑)。
- **右臂可达扫描 (2026-09-07)**: 左臂(盘) 盘 yaw −90°~45° 全可达。右臂(海绵) 跟踪重建面内 yaw ψ(t) 时只有 69~75% 行可达,
  失败集中在源帧 20~100, 与盘位/运动幅度无关 —— ψ(t) 前 120 帧 ≈ −140°, 后半段 ≈ −25°, 中途转了 115°, 掌朝下腕姿跟不上。
  擦拭以平移为主 ⇒ 母带海绵面内 yaw 改**常量** (`--psi_mode const`, 中位 −50.8° + 偏置), 偏置与摆放由第二轮扫描定。
  海绵长轴与盘法向夹角中位 72° (基本平躺), 与"擦盘面朝盘"一致。
- **刚体海绵落座**: 13.3cm 长海绵跨在 9cm 碟心平底与斜坡缘之间, 贴合高度按足迹最高支撑点算 (不是盘心 top(r)),
  母带构建器与判据 `seat_height` 同定义; 否则母带会让海绵端头插进斜坡 1.5cm。
- **A3 母带 v1 定案 (2026-09-07)** `clean3_reference_v1.npz`: 盘心 (−0.12, +0.06, 桌+0.12), 盘 yaw 0 / tilt 0, 盘运动幅度 0.5
  (低通后位置范围 4.6×2.3×4.8cm, 路径 36cm), 海绵 XY = 盘系面内图案 (r p10/50/90 = 2.8/5.4/6.3cm, 全在盘内), ψ 常量
  (中位 −50.8° + 偏置 −130°), 落座高度 2.94cm (端头永远搭在缘上: 13.3cm 海绵在 ⌀18 碟里必然一端出缘), 压深 2mm。
  15Hz 300 帧 → 20Hz 424 行 (21.1s)。双臂 IK: 右 ok 1.00 pos 0.20cm rot 0.12° 关节步 1.6°; 左 ok 1.00 pos 0.20cm rot 1.06° 关节步 1.5°。
  首行腕位: 左 (−0.29, 0.105, 1.074) 右 (−0.20, −0.084, 1.083)。锚 = Sweep2 实测 arm_center (A0 核 runtime 锚差)。
- **A0 探针 v1 (2026-09-07, β=1 全量 squeeze) → H1 证伪**: 物理核对全对 (质量 0.300/0.050, 摩擦 1/1, 指垫 1, runtime 锚差 0,
  手-物复位误差 0.2~0.5cm); 但 grasp→squeeze 合拢阶段两物体相对手各滑 4~5cm, 静持 3s 海绵 3/4 掉, 盘倾 17°;
  放音行 101 盘掉 (倾 56°), 海绵行 0 掉。基类静态 FK 报 grasp 姿下左垫离盘面 1.7~2.9cm (Dexonomy 口径 1mm) —— 与 Unscrew
  线记录的"Dexonomy 手模型 vs Isaac 手 指长/掌形差异"同款。Dexonomy 左/右 MJCF 关节序与 GENERIC_JOINT_ORDER 逐项一致 (排除映射错)。
  参考几何 (纯运动学, 无物理) 自身: 接触行 100%, 覆盖 56.5% 全在 4.5~8.5cm 缘带, 碟心 0~4.5cm **0/60 格** —— 刚体 13.3cm 海绵
  跨在 9cm 碟心上永远碰不到碟底 (D4 形变体缺口的具体形态), 待用户拍板 (选项: 海绵俯仰探底 / 只判缘带 / 接受)。
- **A0 诊断 (2026-09-07)**: ① Dexonomy MJCF 左/右手 vs 本仓 Isaac URDF, 同一 grasp 关节角下五指 DP 体在掌系位置逐指一致
  (|Δ| ≤ 0.01cm, 双手) —— "指长/掌形差异"假设**证伪**, elastomer 与 DP 同原点 (垫是 DP 的子 link)。
  ② β=0 (指停 grasp 姿) 时双手指垫接触力全 0, 海绵/盘在 0.1s 合拢斜坡里先自由落体 ⇒ v1 "合拢后滑 4~5cm" 是探针协议假阴性:
  Dexonomy Isaac 抬升里物体先靠桌托着再合拢, 悬空 in-hand 复位必须在合拢期间把物体钉在手相对位姿, 指力建立后再松手 (已改)。
  ③ 先验接触点核对: 盘 10 点中 9 点落在真实盘面 ±4mm (拇指缘顶 y=1.04 vs 顶 1.08; 三指缘底/斜坡底), 1 点在盘底下方 1cm
  (DP 壳口径); Dexonomy 自身的 Isaac 抬升把盘当**单凸包** (collider_type convex), 碟心/缘底被填平 —— 其 4.2cm 抬升不代表真盘缘。
- **A0 探针 p2~p4 (2026-09-07, 钉住合拢协议后)**: 海绵 (右, 11_Power_Sphere) 静持 3s 漂移 0.33cm/2.5°, 拇/小指 8N —— **H1 海绵通过**。
  盘 (左, 27_Quadpod) 不掉但沉降 2.2cm/11° (倾 9°): 拇/食指捏缘 12~13N, 中/无名/小指托底只有 0~4N。
  敏感性: β 1.5/2.0 → 3.1~4.0cm/11~14° (更差); 质量 0.15kg → 2.2cm/8.5° (**与质量无关**); 左腕沿盘法向 +4/+8/+12mm → 1.4/2.5/1.9cm。
  hold 时序: 0.1~0.5s 内沉降到位后 3s 内变化 <0.01cm ⇒ 是 **GraspPose 名义相对位姿 ≠ 物理平衡位姿的零点偏差**, 非握不住。
  根因线索: 池内所有盘候选的"托底"接触点系统性落在盘底以下 2.5mm~1cm (r=4.5cm y=−1.4 / r=3.7cm y=−2.2, 盘底 −1.15) ——
  正是 Dexonomy 合成时**底座**所在, 即那些指是顶在底座上; 真实持盘时它们悬空, 盘转到中指托住为止。
  v1 母带全放音: 海绵行 12 被撞出手 (盘沉降后海绵参考高度压进盘面 1cm, gap −11mm), 盘行 211 掉 —— H2 证伪 (在 v1 锚下)。
  **对策 (v2)**: 探针记沉降后的 T_obj_hand, 母带以它做腕目标 (`--oh_override`), 探针摆物同用 → 盘从平衡位姿起步, 海绵按真实盘位贴合。

### 5.6 抓稳段优先 + 判据定案 (2026-09-07 用户裁定)

**裁定**: ① 碟心先不算, 判据只看缘带可达区; ② A4 阈值与 A5 接线开做; ③ **先把抓稳段训练好再做交互**。
抓稳段语义 (用户原话): 物体保持在 GraspPose 位置; 手在 GraspPose 指引下接近并抓稳物体; 物体保持原位置不动; 逐渐减少钉住时长让策略逐步接手。

**A4 阈值 (按 L5-27 取参考自身几何成绩的 80%)**: 参考自身 覆盖 0.565 / 接触中行程 87cm (缘带) ⇒
`cover_min=0.45`, `travel_min=0.70m`, 盘倾角变化 <15°, 盘位移 <3cm, 双物不掉。"擦到"后续加力条件 (海绵↔盘法向力 ≥2N,
A5 接 aux 接触传感器后由零动作放音标定; 几何口径先用)。Δ覆盖 earn-only 鼓励擦新区域; 参考跟踪只当皮带。

**Stage-1 抓稳段 env (先训)** —— 母本 Unscrew part4 (Pour17 v5 身), 58 维残差 [R臂7, L臂7, R指22, L指22]:
- 回合结构 (20Hz): 母带前奏 = 接近 K_app=20 行 (腕从 prior `pregrasp[0]` 沿 6 步接近轨迹插值到 grasp, 指 pregrasp→grasp)
  → 合拢 K_close=10 行 (指 grasp→squeeze) → 静持 K_settle 行 (squeeze 常量)。物体世界位姿 = 母带 0 行 (GraspPose 名义位姿), 前奏期不动。
- **钉住 (pin)**: 前 `release_row` 行两物体每物理子步重写到名义世界位姿 (零速度, 重力关), 之后放开只靠指力。
  课程: release_row 从 K_app+K_close+40 (放开前多 2s 缓冲) 按成功率 EMA 退火到 **K_app+K_close** (合拢一结束即放手)。
- **G0 抓稳认证**: 放手后连续 10 步 两物体**世界位姿**相对名义位姿漂移 <1.5cm/8° ∧ 盘拇指+≥2 托底指有力 ∧ 海绵 ≥3 垫有力。
  认证时锁存位姿 (HOLD_POSE latch); 之后偏离渐进罚, >3cm/20° 或落桌 = D1 终止。**成功 = 认证 ∧ 持到回合末 (放手后 3s)**。
  注意: 被动 (零动作) 时盘会沉降 2.2cm/11° —— 阈值 1.5cm/8° 要求策略比被动更好, 这正是要学的东西。
- 奖励: 钉住期 = 接触建立 (垫数×力, 对向势 r_opp/跟腕比 r_follow); 放手后 = −漂移渐进罚 + 认证一次性 + 成功一次性 + 动作罚。
- 观测: 沿 Unscrew 507 块布局, 物体块换 [两物相对手位姿差 / 世界漂移 / 钉住剩余步 / 双手 10 垫力]。
- 起步固定 (母带 0 行), 抖动后加。

**Stage-2 交互段 (后)**: 认证后接擦拭行 (母带 v1 的 424 行), A4 判据; 海绵参考改成相对**实际**盘位姿逐步重锚 (盘沉降后
预烘世界轨迹会压进盘面 1cm, v1/v2 放音里海绵正是这么被撞出手的)。

**文件**: `L2_Reference/build_reference.py --prelude` → `clean3_reference_v3.npz` (v1 锚 + 前奏, 带 prelude_rows/release_row_min);
`C_Wiring/task_config.py` / `clean_env.py` (Stage-1) / `train_clean.py` / `smoke_clean.py`; 探针 `--kinematic` 支持前奏行目检。

**可证伪**: H5 钉住课程能退火到 K_app+K_close 且成功率 >0.8 (否则被动沉降不可克服 → 换候选/改握法);
H6 认证后 3s 内漂移 <1.5cm/8° 的比例随训练上升 (TB term/drop 下降)。
- **§5.6 落地进度 (2026-09-07 晚)**: A4 阈值回填 (cover 0.45 / travel 0.70m, 自检 PASS)。母带 **v3** = v1 锚 + 前奏 50 行
  (接近 20: prior pregrasp 4cm→0.3cm 六步插值; 合拢 10: grasp→squeeze; 静持 20), 总 474 行, 双臂 IK 100% (pos ≤0.2cm rot ≤0.9°),
  `release_row_min=30`。运动学目检 (`--kinematic`): 前奏期物体钉在世界原位、手从 4.2cm 外接近; 擦拭期物体锁在手上; 物体碰撞关闭
  (钉住+碰撞会被物理子步按穿插踢开, 实测海绵 0.9cm/9°, 关掉后物体跟参考 ≤0.2cm/0.3°)。对训练 env 的提示: 钉住期物体与手指的接触
  冲量正是要的 (指力建立), 但**物体与物体**间 (海绵-盘) 的穿插冲量要避免 —— 前奏期海绵参考已在盘面上方压深 2mm, 需把前奏期海绵抬离盘面
  或钉住期豁免海绵-盘碰撞。
- **Stage-1 接线 ✅ (2026-09-07 深夜)** `C_Wiring/{task_config,clean_env,smoke_clean,train_clean}.py + ppo_clean.yaml`:
  用户三点修订落地 —— 碰撞全开 / 手直接从 grasp 位起步 (母带 v1 第 0 行, 无接近段) / 只求放手后物体不动。
  obs 367 (臂/指 q,qd + 残差 + 双手手/物位姿 + 手内相对位姿 + 世界漂移 + 物速 + 10 垫力/触发 + 相位 + 指前馈 + 上步动作), priv 22, act 58。
  钉住 = `_apply_action` 每物理子步写回名义位姿; 放手行 release_row ∈ [cur−4, cur], cur 由训练入口按成功率 EMA≥0.7 持续 10 窗
  单向退火 50→10 (步 5)。一次性 PD 下垂补偿 ≤1.1°。
  **冒烟 (8 env)**: 零动作 放手后 盘 0.96cm/8.2° 海绵 0.27cm/3.5°, 无掉落, 成功 1/8 (8.2° 卡阈值); 随机动作 掉落 44% 成功 0。
  ⇒ 任务定义成立; 但 8° 恰是被动平衡 (钉住 2.5s 让指力充分建立后盘沉降比探针小得多), 建议发车时收紧到 1.0cm/5°
  (`CLEAN_CERT_POS_CM/ROT_DEG` 环境变量), 让策略必须比被动更稳; 待用户定。
- **PPO 冒烟 (64 env, 60k 步, `logs/Clean3_hold_smoke`)**: 回路通; 成功率 0→0.76, 认证 →1.0, 掉落 1.0→0, 盘漂移 1.36→0.55cm / 10.0→6.2°,
  海绵 3.1→1.25cm / 10.3→6.7°, 接触分 盘 1.57→1.80 (满 2) 海绵 0.69→0.75。课程未触发 (EMA 0.28 <0.7, 步数太短)。
  FPS ~200 @64 env (小规模开销主导)。发车候选: 512 env, 50M 步, seed 42; 认证阈值 1.5cm/8° (默认) 或 1.0cm/5° (建议) 待用户定。
- **2026-09-08 用户三点**: ① 加手指交叉/重叠罚 (同手相邻指尖掌系侧向间距 <1cm 或翻转 = 交叉; 3D 距 <1.5cm = 重叠; W=1, 双手 6 对均值);
  ② 认证阈值默认 1.0cm/5°; ③ 发车放 Denso 2080Ti。
  首版罚一上, 零动作 cross_frac 0.957 —— 根因是 Dexonomy 右手 squeeze 姿本身让小指扫到无名指下 (CMC +22.4°/MCP_AA −10.7°,
  侧向差 −0.67cm, 3D 1.49cm)。前馈去交叉: 右小指 CMC/MCP_AA 合拢剂量置 0 (URDF FK: 侧向 1.84cm / 3D 2.48cm), 罚保留约束策略。
  去交叉后冒烟 (8 env): 零动作 cross_frac 0 / 罚 0; 但右小指不再扫入后海绵被动握持变弱 —— 海绵 2.1cm/11.7°, 2/8 掉, 接触分 0.83→0.34
  (盘 1.16cm/8.9° 不掉); 随机动作 掉 44%, cross_frac 0.10 (罚生效)。⇒ 海绵的稳握交给策略 (指残差 ±0.6 rad, 罚约束不许交叉)。
- **发车 (2026-09-08)**: Denso GPU0 `Clean3_hold_s42` (512 env, seed 42, 50M 步, 认证 1.0cm/5°, 罚 W_CROSS=1, 课程 release 50→10),
  脚本 `C_Wiring/denso_clean3.sh`, 日志 `~/Clean3_hold_s42.log`, run `~/RL_Correction/logs/Clean3_hold_s42`。部署清单: datasets/clean_tableware/3
  (去 cache/masks) + tasks/Clean + priors/Clean3_* + rl_rebuild/ + tasks/pregrasp/*.py; 母带/先验 md5 两端一致。
- **实录 (2026-09-08 04:50 UTC, 10M 步/655 epoch, FPS ~550)**: 曲线两段 —— 0~7.7M 步 (epoch ≤500) 策略"钉住一解除就扔海绵"
  (drop_sponge≈1, success 0, 回合奖 −5→+1.5; 1M 步时曾据此误判为奖励逃逸不可逆); epoch 504~512 (~7.8M 步) 8 个 epoch 内相变: drop→0,
  success→0.91, 海绵漂移 22°→3°, 回合奖 7→44; 课程按 EMA≥0.7×10 窗推进 release 50→45(ep541)→40(582)→35(625), 每档约 0.6M 步。
  **H6 成立 (相变后 term/drop=0), H5 待课程走到 10 再判**。要盯: `shape/cross_frac` 相变同刻 0.01→0.41、`ep_rew/cross` −1→−5/回合且在涨;
  盘漂移随课程 0.3→0.8cm 回升。教训: 逃逸相可持续 15% 预算后自发相变, 别在 2% 预算处下"必死"结论。
- **回放工具 `C_Wiring/record_clean.py` (2026-09-08)**: 载 ckpt 确定性 mu, 4 env 并行, env0 三机位 mp4 (front/side/top) + **逐步奖惩 CSV/npz**
  (五项分量/漂移/认证/掉落/10 垫力/相邻指尖间距) + 回合汇总 json; 与 world.json 核对物理/常量/IO。`clean_env._tick` 加只读快照字段 (不改行为)。
  last.pth (10M) 读数: r35 档 4/4 成功 (return +21.6, 放手后 盘 0.6cm/3.1° 海绵 0.24cm/3.2°); r50 档 0/4 —— 不掉、认证过, 但回合末海绵转角 5~7.5° 漂出 5° 窗。
  **交叉罚来源定位 = 左手(盘) 中指-无名指对翻转 (gap 最小 −0.3cm, ~50% 步)**, 右手三对 ≥1.7cm 干净; r_cross ≈ −8/回合 压不住 +25 奖。
  产物镜像 `logs/Clean3_hold_s42/` (README 列全)。Isaac 录像进程不能调 app.close() (挂死占显存), 直接 os._exit。

### 5.7 Stage-2 交互段 = 最简六项配方 (用户 2026-09-08 批准; 三项默认: 认证不带接触条件 / 转角用全转角 / 先单 seed 30M)

**边界 (用户原话)**: 不用 cuRobo 接近, 盘子和擦布直接出生在手里; 刚开始留时间让 RL 学抓稳; 然后跟着轨迹交互; 最后完成任务。
**Stage-1 收官**: `Clean3_hold_s42` 2026-09-08 09:02 UTC 停 (19M 步; 课程 14.09M 到终点 release=10; 之后 success 0.5~0.86 晃, drop 0, cross_frac 0.73 且在涨,
回合奖为负 = 罚压过奖)。ckpt/TB/回放镜像 `logs/Clean3_hold_s42/` (README 列全)。它证明固定腕稳握可学, 也证明交叉罚 1.0 压不住。
**回合结构 (20Hz)**: 行 0 手在母带 v1 第 0 行 grasp 位, 两物体钉在 GraspPose 名义位姿; 指 grasp→squeeze 斜坡 10 行后常量; 臂前馈第 0 行;
release 行由课程 (50→10, 步 5, 认证 EMA≥0.7×10 窗) 给 —— 唯一课程。放手瞬间锁存两物体在各自掌系的位姿; 连续 10 步 相对锁存 位移<1cm ∧ 全转角<5° (两物体)
⇒ 认证 +5, 时钟开始; 放手后 60 行未认证 ⇒ 超时结束不另罚。交互段: 臂前馈播放 v1 第 k 行 + 58 维残差 (臂界用 Pour 标定值, 不是 Stage-1 的 0.004);
时钟门与皮筋定义在**盘面系里的海绵位姿**上 (法向门 1cm, 面内 5cm; 皮筋按母带 confidence_0/1 分档, 不加黄窗); k 前进一行 adv +1。
第一版不做在线重锚 (让残差吸收盘沉降 ~1cm/8°, 关系皮筋给梯度)。结束 = 时钟走完或 A4 达标 (+20): 缘带覆盖 ≥0.45 ∧ 接触中行程 ≥0.70m ∧ 盘倾角 <15° ∧ 盘偏离该行参考 <3cm ∧ 双物在手。
**死线 (-10 终止)**: 任一物体相对锁存 位移>3cm 或 全转角>20°; 盘倾角 >30°; 盘偏离参考 >10cm; 撞桌。海绵-盘接触是任务, 不罚。
**握形**: 左手中指/无名指侧摆残差界置 0, 右小指前馈 β=0 沿用; cross_frac 只做诊断, 无罚项。
**奖励只 5 项**: adv / leash / 认证 +5 / 成功 +20 / 死亡 -10 (+ 极小动作罚)。Stage-1 的接触奖、窗内奖、渐进罚、交叉罚全部去掉。
**观测 (掌系, 无放手倒计时)**: 双物相对手掌位姿+速度、相对锁存偏差、掌系重力方向、盘面系海绵位姿及后几行参考、时钟进度与档位、指垫力、前馈指姿、上步动作。
**物理**: 盘 0.3kg / 海绵 0.05kg / μ1 / 指垫 1 (§5.0⑦)。
**预注册假设 (一项对一改)**: H-C1 点火 3M 认证率≥0.5 (超 8M 不到 → 只加回接触奖); H-C2 15M 内 release 到 10 (卡档 → 认证转角 5°→8°);
H-C3 训练末 相对位移最大值中位 ≤2cm ∧ 转角 ≤15°, cross_frac<0.05 (违 → 加相对位姿软罚); H-C4 时钟走完比例 ≥0.8 (停在法向门外 → 加在线重锚);
H-C5 20M 确定性 16 回合 A4 成功 ≥0.5 (覆盖不够而握持正常 → 面内门 5→3cm)。
**正式口径**: 新写 `eval_clean.py` (16 env t0, 抖动 0 / ±3cm 各一组): 认证率 / 时钟走完 / A4 成功 / 覆盖 / 行程 / 相对位姿最大值 / 交叉率; 录像与逐步账本沿用 `record_clean.py`。
**施工**: ① Stage-1 env 派生交互版 (掌系偏差+锁存+死线; 臂播母带; v1 抽海绵盘面系参考; 借 Pour 进度机分档逻辑做门/皮筋; A4 接里程碑)
② eval_clean / smoke (零动作+随机各一回合) ③ train_clean 课程改按认证 EMA + world.json 新开关 ④ denso_clean3.sh。约 1.5 天。
**发车**: Denso GPU0 (已腾), 512 env, 30M, seed 42 先一条, 点火后补 seed; 判读点 3M/8M/15M/20M, 先看分母再看率。
- **Stage-2 施工完成 + 冒烟 (2026-09-08 ~09:25 UTC)**: `C_Wiring/clean_task_env.py` (CleanTaskEnv ← CleanHoldEnv; obs 364/priv 22/act 58; T_EP 661 = 50+60+1.3×424),
  `task_config.py` S2_* 常量, `smoke_task.py`, `train_task.py` (课程键 sr/cert), `eval_clean.py` (确定性 16 env, 读 env.last_ep 终局摘要, 与 world.json 核 stage2 常量),
  `denso_clean3_task.sh NAME GPU SEED`。臂每步界取到 cfg.arm_residual_max×0.25 (R=[0.0038,0.0053,0.0045,0.0050,0.0219,0.0125,0.0194]); 硬钳 4 关节;
  母带档位 绿236/黄110/红78; psc_ref z 2.61~2.94cm。Denso 冒烟 (8 env) PASS。
  **零动作基线 (release=50)**: 放手后盘沉降 0.69cm/5.1° (卡在 5° 认证线外, 与 Stage-1 冒烟一致 ⇒ 策略必须比被动好), 海绵不动;
  **盘一沉海绵就离盘 (gap 19mm, contact=0), e_n=1.6cm > 门 1cm** ⇒ 时钟起不来 —— H-C4 (关系皮筋够用) 的压力点就在 k=0: 残差要把海绵压下 ~1.5cm, 梯度来自法向皮筋。
  零动作: cert 0, die_rel 0.25 (盘沉降超 20°? 2/8), cert_timeout 0.75; 随机动作: cert 0.06, die_rel 0.29。
  **待用户拍板发车**: `bash tasks/Clean/3/C_Wiring/denso_clean3_task.sh Clean3_task_s42 0 42` (GPU0, 512 env, 30M)。
- **发车 (2026-09-08 09:34 UTC, 用户批准)**: Denso GPU0 `Clean3_task_s42` pid 3665398, 512 env, 30M, seed 42; 横幅核过 (headless / 0.3kg μ1 指垫1 / obs 364 / T_EP 661 / 课程键 sr/cert); 起步 FPS ~560~1100。日志 `~/Clean3_task_s42.log`, run `logs/Clean3_task_s42/`。判读点 3M/8M/15M/20M (H-C1~5)。

### 5.8 今晚 2×2 因子实验 (用户 2026-09-08 晚: msc GPU0/GPU4/GPU5(停 vLLM) + 本机 4080S 各一条)

**候选旗 (§5.7 候选清单前两项, 冒烟实证)**:
- `CLEAN_S2_CLOCK_CONTACT=1`: 时钟推进 (=adv) 要求海绵接触盘面, 封"悬停 9mm 刷满 424 行 adv 而不擦"的漏洞。
- `CLEAN_S2_REANCHOR=1`: 海绵臂前馈重锚。三版迭代 (本机零动作+强制认证冒烟): ① 盘位姿偏差前馈 → 正反馈跑飞 (压盘→盘沉→偏差涨→再压, 压进 1.5cm, dq 卡钳位, 3/8 die_rel);
  ② 锁存 10 步后冻结 → 仍过冲 1.2cm (锁存窗内已含压盘响应); ③ **定案: 盘面系海绵位置误差闭环伺服** e=psc−psc_ref[k], Δx=−0.3·R·e (死区 2mm), 经 PhysX 雅可比 DLS 积分到右臂 7 关节 (钳 0.15 rad), 认证后生效, 门/皮筋目标不动。
  (③ 冒烟结果见下条追记)
**世界差异 (重要)**: msc 的臂下垂标定 sag 与 Denso/本机差一个量级 (msc j6 4.58° vs 0.05°), 钉住期盘倾 2° vs 0.8°, 零动作 die_rel 0.875 vs 0.25 —— msc (MagicSim_IsaacLab, Jun 10 版) 是**另一个世界**;
  本机与 Denso 逐位一致 (sag/IK err 相同)。⇒ 跨机成绩不可比, 因子设计按世界分组。URDF 路径靠 `VEGA_URDF` 环境变量 (msc/Denso 同款 ~/data/vega_urdf)。
  **★ 根因查明并修复 (2026-09-10 02:40~03:10 UTC)**: 不是 IsaacLab 版本。三机 pip (isaacsim 5.1.0.0/isaaclab 0.54.3)、IsaacLab 源码 430 文件哈希、URDF 全同;
  唯一差异 = msc `~/RL_Correction/assets/vega_1p_sharpa_fixedtorso.usd` 是 7/28 旧版 (md5 f77f235d = 本机 `_stance0803.usd`, 躯干未锁死), Denso/本机是 8/27 提交 e6b08cf5 的
  "躯干锁死 USD (跨机漂移事故修复件)" (md5 143385ad)。msc 那份 RL_Correction 是非 git 的 rsync 副本, 代码更新了 assets 没跟上。修法: 旧文件备份 `.usd.bak_jul28`, 用户手动
  `scp -o ProxyJump=UCBY` 覆盖; 零动作冒烟 (`smoke_task.py --num_envs 8 --modes zero`) 核验: msc sag 十四关节与 Denso 逐位相同 `[0.10,0.02,0.12,-0.20,0.43,0.05,-1.06,-0.10,0.02,-0.09,0.01,0.17,-0.32,-0.74]`,
  rates 同族: msc cert 0.167 / die_rel 0.333 / drop 0 / relp_max_plate 0.67cm vs Denso 同命令 0.125 / 0.188 / 0 / 0.66cm (旧世界 cert 0 / die_rel 0.875 / drop 8/8)。⇒ **msc 自此与 Denso/本机同世界**, 8 卡可用于 Clean3 对照;
  §5.8 msc 三条 (taskM/MC/MR) 结果作废。教训: 往非 git 机器同步代码必须连 assets/ 一起 (或核 md5); Clean3 的 world.json 目前无 robot.usd_md5 段 (Pour 有), 应补入指纹 CRITICAL, 否则这种漂移只能靠 sag 数字肉眼发现。
**分配**: Denso 世界: base s42 (Denso GPU0, 在训) + **+C+R s42 (本机)**; msc 世界: base s42 (GPU0) / +C s42 (GPU4) / +R s42 (GPU5)。
  msc 内比主效应, Denso 世界内比"两者都加"的合成效应; seed 复制留到明天。
**判读 (同 §5.7 H-C1~5)**: 3M 认证; 8M 时钟与接触 (`prog/clock_frac` vs `task/contact_frac` 分离 = 悬停剥削实证); 15M 课程; 20M 评测 (`eval_clean` 16 回合, 各机各自跑)。
- **发车 (2026-09-08 10:23 UTC)**: msc `Clean3_taskM_s42` GPU0 pid 3676597 / `Clean3_taskMC_s42` GPU4 pid 3678468 (CLOCK_CONTACT) / `Clean3_taskMR_s42` GPU5 pid 3679025 (REANCHOR 伺服版, 发车前 A4 冒烟: 10 步内法向误差 1.42→0.31cm 无过冲);
  本机 `Clean3_taskCR_s42` pid 543285 (两旗, 4080S, FPS ~1150~2300); 日志 msc `~/<run>.log`, 本机 `launch_logs/Clean3_taskCR_s42.log`。夜间巡检 cron (7,37 * * * *) 已建, 笔记 scratchpad/night_notes.md。
  发射脚本: `msc_clean3_task.sh NAME GPU SEED [FLAGS]` (含 VEGA_URDF), `local_clean3_task.sh NAME SEED [FLAGS]`。
- **10:36 UTC 追记**: msc `Clean3_taskMR_s42` 首发 GPU5 起步即崩 (CUBLAS_STATUS_ALLOC_FAILED): 我们的 vLLM 被 `~/bin/vlm_supervise.sh` 守护自动重启占回 42.5GB。未停守护 (影响他人重建管线, 待用户定); MR 原样改 **GPU7** 重发 (旧日志 `~/Clean3_taskMR_s42.gpu5_oom.log`)。
- **3M 判读 (2026-09-08 11:20 UTC, Denso 世界两条)**: base `Clean3_task_s42` cert 1.00 / 课程已到 25 / clock_frac 0.50 / contact 0.97 / coverage 0.53 / travel 63cm / success 0 / **die_rel 0.997** (relp 2.2/2.4cm, rot 15/18°) / cross 0。
  `+C+R Clean3_taskCR_s42` cert 1.00 / 课程 25 / clock_frac 0.42 / contact 0.97 / coverage 0.52 / travel 66cm / success 0 / die_rel 0.995 / cross 0.36↑ / reanchor dq 卡钳位 8.4°。
  H-C1 成立。**当前绑定约束 = 3cm/20° 相对位姿死线**: 两条都会擦 (覆盖已过线, 行程差 5cm), 但物体在手里缓慢蠕变, 回合在时钟一半被杀。3M 看不出两旗收益 (基线 contact 0.97, 悬停剥削未发生; 伺服饱和)。8M 复判 H-C3/H-C4。
- **8M/3M 判读 (2026-09-08 12:50 UTC)**: 本机 `+C+R` 8.9M: cert 0.99 / 课程终点 10 / clock_frac 0.47 / gate 0.97 / contact 0.97 / coverage 0.51 / travel 69.5cm / success 0 / die_rel 0.99 (relp 2.4/2.1cm, rot 16.5/15.6°) / reanchor dq 卡钳位 8.2°。
  H-C3、H-C4 未达; 门通过率 0.97 ⇒ 不是卡门, 是 ~200 行处被 3cm/20° 死线杀。覆盖/行程已到 A4 线, 只差活到时钟走完 ⇒ 按 H-C3 预注册: 下一步加相对位姿软罚 (待用户)。
  msc 世界三条 3M: 认证全 0, 放手即死 (die_rel 0.65~0.75 + die_tilt 0.14~0.35), 课程卡 50 —— 世界问题 (sag 标定差量级), 对两旗不提供信息; 建议明早停掉, 因子实验挪回 Denso 世界。
- **8M 判读 base (2026-09-08 14:20 UTC)**: `Clean3_task_s42` 8.4M: cert 1.00 / 课程 10 / clock_frac 0.47 / gate 0.99 / contact 0.97 / coverage 0.49 / travel 60cm / success 0 / **die_rel 1.00** (relp 1.7/1.9cm, rot 盘 16° / 海绵 22°)。
  本机 `+C+R` 14.3M 同构: clock 0.50 / contact 1.00 / coverage 0.55 / travel 72cm / success 0 / die_rel 1.00 (rot 海绵 23.6°)。
  **夜间结论**: 六项最简配方能学会抓稳 (认证 1.0, 3M 内课程走完) 并擦到 A4 的覆盖/行程线, 但 100% 回合在时钟一半被 3cm/20° 相对位姿死线杀, 主犯是海绵在手里的转角 (22~24°); 两面候选旗在 Denso 世界内没有可见收益 (基线接触已 0.97, 伺服卡钳位)。
  按 H-C3 预注册的下一步 = 加相对位姿软罚 (1cm/5° 起渐进到死线), 待用户拍板; msc 三条 (另一世界, 认证全 0) 建议停。
- **回放 (2026-09-08 12:25 UTC, 本机, `record_task.py`, last.pth, release=10, 4 env 确定性)**: base 17.1M: clock 0.77 / coverage 0.70 / travel 83cm / 4/4 死于相对位姿 (~338~383 行; relp 盘 2.3 / 海绵 2.8cm, rot 17°/17°);
  +C+R 30M: clock 0.56 / coverage 0.75 / travel 81cm / 4/4 死于相对位姿 (~250 行; relp 3.1/2.0cm, rot 19.7°/12.8°)。确定性回放里覆盖/行程都远超 A4 线 (0.45/70cm), 只差不被死线杀。
  产物 `logs/<run>/videos/last_r10_{front,side,top}.mp4` + `_steps.csv/npz` + `_summary.json`; 两 run 目录各有 README 与 `tb_scalars.csv`。

### 5.9 H-C3 对照: 相对位姿软罚 `CLEAN_S2_SOFT_REL=1` (用户 2026-09-08 批准, 本机)
**动机**: base 与 +C+R 两条 (Denso 世界) 均 100% 回合在时钟 0.5~0.77 处被 3cm/20° 死线杀 (海绵转角 17~24°), 覆盖/行程已超 A4 线; 奖励里死线之前没有任何"别让物体动"的梯度。
**改动 (只此一项)**: 认证后每步 r_soft = −0.5 × mean_物体 clamp(max((dp−1cm)/2cm, (dr−5°)/15°), 0, 1); 死线处满值 −0.5/步 < adv +1 (不诱导早死)。其余与 base 完全一致 (无接触门/无重锚)。
实现: task_config `S2_SOFT_REL/S2_SOFT_W`, clean_task_env r_soft (+`ep_rew/soft` 账本, `_tick.r_soft`), world.json stage2 记旗, record_task 按 world.json 还原。
冒烟 (本机, 零动作+随机, 强制认证): 罚量与手算一致 (随机动作 步 80: 盘 2.2cm/16.6° → −0.235/步), PASS。
**发车**: 本机 `Clean3_taskS_s42` (seed 42, 512 env, 30M, 2026-09-08 19:53 UTC), 对照 = `Clean3_task_s42` (Denso, 同 seed 同世界)。
**预注册**: H-C3': 8M 时 `hold/relrot_max_sponge_deg` 中位 ≤15° 且 `term/die_rel` ≤0.5, 15M 时 `sr/success` ≥0.3 (base 17M 为 0.01)。证伪 (漂移没压下去) ⇒ W 0.5→1.0 一项; 证伪 (漂移压下去但 clock_frac 反降 <0.4) ⇒ 罚过重, W→0.25。

### 5.10 自主迭代 (用户 2026-09-08 授权: Pour 最简三条跑完后用 Denso 4 卡继续做抓紧与持续稳定手物关系, 可加鲁棒性设计; 用户 ~03:00 UTC 回)
**共同事实**: base / +C+R / (软罚 W0.5 在本机跑) 都能认证 1.0、擦到 A4 线, 100% 回合在时钟 0.5~0.77 被 3cm/20° 死线杀 (海绵转角 17~24°)。
**三条对照 (Denso GPU1~3, base + 一项, seed 42, 512 env, 30M)**:
| run | 旗 | 机制 | 预注册 (8M) | 证伪后唯一动作 |
|---|---|---|---|---|
| `Clean3_taskS1_s42` | `CLEAN_S2_SOFT_REL=1 CLEAN_S2_SOFT_W=1.0` | 软罚加倍 (与本机 W0.5 成扫描) | 海绵转角中位 ≤15° ∧ die_rel ≤0.5 | 若 clock_frac <0.4 = 罚过重 → 回 W0.5 |
| `Clean3_taskG_s42` | `CLEAN_S2_GATE_HOLD=1` | 时钟推进要求两物体在 1cm/5° 窗内 (稠密奖与握持绑定, 无新奖励项) | 同上, 且 clock_frac ≥0.4 | 若时钟停滞 (gate_frac <0.5) → 窗放到 1.5cm/8° |
| `Clean3_taskP_s42` | `CLEAN_S2_PUSH=1` | 认证后随机推力 (p 0.02/步, 8 行, 0.5~1.5×mg, 掌系随机向) = CartPole 扰动课程 | 无推力评测 (eval 默认关推) die_rel 比 base 低 ≥0.2 | 若 die_rel 反升 → p 0.02→0.01 |
评测口径: `eval_clean.py` (16 env, 无推力; 鲁棒性另跑 `CLEAN_S2_PUSH_OVERRIDE=1`)。base 30M 跑完即评。
- **20:37 UTC Denso 四卡失联 (驱动 Unknown Error, 需人重启)**: 四条训练空转已 kill; `Clean3_task_s42` 抢救到 19.71M (本地 logs/), Pour 最简 P17 19.66M / P25 20.02M(完整) / P31 19.23M。
  UCBY 不可用 (无栈、磁盘满); msc 是另一世界。⇒ 今晚只剩本机 4080S: `Clean3_taskS_s42` (软罚 W0.5, 19:53 UTC) + `Clean3_taskG_s42` (推进门带握持, 20:52 UTC, RL_ISAAC_NO_GUARD=1 双跑, 各 ~450 FPS)。
  §5.10 的 S1 (W1.0) 与 P (推力) 等 Denso 重启后再发。Pour 三条 eval10 改在本机顺序跑 (P17/P31 用 19M ckpt)。
- **S (软罚 W0.5) 8M 判读 (2026-09-09 00:10 UTC, 与 base 同步数比)**: die_rel 0.993/0.999, 海绵转角 18.3°/19.7°, 盘转角 14.8°/16.2°, coverage 0.63/0.49, travel 72/64cm, success 0/0。
  H-C3' 未达 (转角只降 1~1.5°)。预注册动作 W→1.0 因本机显存/吞吐不发, 排队 Denso。**base 崩前 19.7M: success 0.16 / die_rel 0.75 / clock 0.76 / travel 82cm** (17M→19.7M 自行上爬), cross_frac 涨到 0.56。
- **G (推进门带握持) 8M 判读 (2026-09-09 02:10 UTC, 同步数比 G/S/base)**: clock 0.38/0.53/0.50, coverage 0.43/0.63/0.49, die_rel 0.96/0.99/1.00, 海绵转角 15.9°/18.3°/19.7°, 盘转角 18.5°/14.8°/16.2°, **ep_len 434/258/237**, cross 0.18/0.14/0.00。
  预注册未达 (die_rel 0.96, clock 0.38); gate_frac 0.95 不触发窗放宽。特征: 回合长翻倍、海绵转角最低、时钟最慢 —— 稠密奖绑握持后策略在停下来修握持。S@10M: coverage 0.68 vs base 0.46, 但 cross 涨到 0.76 (与 base 19M 的 0.56 同款: 交叉是两条线共同的晚期习惯, 硬钳只钳了 4 个关节)。
- **夜间收官 (2026-09-09 03:10 UTC)**: S@12.8M vs base 同步数: clock 0.60/0.53, coverage 0.62/0.46, travel 80/64cm, die_rel 0.997/1.0, 海绵转角 16.2°/21.5°, 盘 15.5°/13.7°。G@9.9M vs base: **success 0.034/0, die_rel 0.91/1.0, 盘转角 9.5°/13.6°**, 海绵 20.7°/19.3°, ep_len 390/240。
  三条都没过预注册线, 但各有一项变好: S 擦得更多更稳 (覆盖/行程/海绵转角), G 盘握得更稳且首个非零成功。两条本机继续跑到 30M。
  **明早建议**: ① Denso 重启后发 S+G 合并 (软罚+握持门) 与 W1.0、推力 P 三条 (§5.10); ② 交叉需要更彻底的处理 (硬钳扩到所有侧摆/或把交叉几何进死线), base/S 晚期都涨到 0.5~0.76; ③ 海绵转角是共同瓶颈 (16~21°), 考虑对海绵单独收紧软罚或加海绵侧接触条件。
- **15M 判读 (2026-09-09 06:51 UTC)**: S@15M vs base@15M: success 0/0, clock 0.59/0.58, coverage 0.64/0.50, travel 76/68cm, die_rel 0.999/0.993, 盘转角 14.9°/17.2°, 海绵 16.7°/16.7°, cross 0.01/0.77。
  最新: S@18.7M **success 0.062** die_rel 0.94 clock 0.74 travel 90cm; G@15.9M success 0.006 die_rel 0.97 盘转角 8.1° 海绵 20.2°; base@19.7M success 0.16 die_rel 0.76。
  H-C3' 15M 线 (success ≥0.3) 三条都未达; 成功率随训练量在 15M 后才起步 (base 17M 0.01→19.7M 0.16, S 15M 0→18.7M 0.06)。S 比 base 同步数擦得更多 (覆盖 0.64/0.50) 且交叉低一个量级 (0.01/0.77)。

### 5.11 ★ 擦盘子首次完整达成 (2026-09-09 14:51 UTC, 本机 eval_clean 16 回合 t0 确定性, 课程终点档 release=10, 无扰动)
| run | 步数 | 认证 | **成功** | 时钟走完 | 覆盖中位 | 行程中位 | 相对位姿中位 盘 / 海绵 | 死因 | 交叉 |
|---|---|---|---|---|---|---|---|---|---|
| **S 软罚 W0.5** (`Clean3_taskS_s42`) | 30M | 16/16 | **16/16** | 16/16 | 0.64 | 110cm | 1.77cm 13.7° / 1.28cm 7.7° | 无 | 0.28 |
| G 握持门 (`Clean3_taskG_s42`) | 30M | 16/16 | 9/16 | 0/16 (靠 A4 达标) | 0.61 | 81cm | 1.07cm 12.2° / 1.31cm 14.1° | rel 5 | 0.41 |
| base (`Clean3_task_s42`, Denso 事故截断) | 19.7M | 16/16 | 1/16 | 1/16 | 0.48 | 81cm | 1.42cm 21.0° / 1.81cm 16.1° | rel 15 | 0.62 |
训练期 30M 末段: S success 0.80 (峰 0.90 @29.8M) die_rel 0.18; G 0.55 (峰 0.74) die_rel 0.40。同步数 19.7M: S 0.215 / G 0.04 / base 0.16 ⇒ 20M→30M 是起飞段, base 没跑到; 软罚在起飞段明显更陡且终局 16/16。
**判读**: H-C3' 15M 线当时未达, 但 30M 终判成立: 相对位姿软罚 (1cm/5° 起渐进到死线, W 0.5) 是让"擦到"变成"握稳着擦完"的那一项; 海绵转角从 16~21° 压到 7.7°, 盘 13.7°。
握持门 (G) 盘更稳但海绵仍 14°、时钟只走一半, 9/16。**下一步**: ① base 跑满 30M 补齐同预算对照 (Denso 重启后) ② S 换 seed 复制 ③ 抖动/推力鲁棒性评测 (`CLEAN_S2_PUSH_OVERRIDE=1`) ④ 交叉率 0.28 仍需治理。


### 5.12 多母带扩数据: take 1 / take 18 入库并造带 (2026-09-10, 用户"1 18 也按照 3 的标准准备好训练")

**盘点**: EgoDex `clean_tableware` 源 28 条; 上游重建 5 条 (0/1/3/8/18); 双物体 3 条 (1/3/18);
take 0/8 只重建出一个物体 (v17A 漏标) 出局。此前只有 take 3 入库并训出成绩。

**关键裁定 (我定, 依据是结构约束不是偏好)**: 三条 take **共用 take 3 的物体资产与两份 GraspPose 先验**,
每条只贡献运动。理由: ① 多母带训练要求资产一致 —— 一个 env 装不下三套网格, 不共用就根本没法混训;
② 三条的盘都被同一条类别先验归一到 ⌀24cm, 是同一物体的不同重建, take 3 那份已过 Dexonomy 合成 +
Isaac 抬升 + 30M 训练 (16/16) 的完整检验; ③ 重跑 Dexonomy 只会得到同族握法, 却要外部工具 + 人工点选。
⟹ **不新增 GraspPose, 不新增物体 USD**。take 1/18 自己的 `objects/` 入库只作溯源, 训练链不读。

**A1 入库**: `datasets/clean_tableware/{1,18}/` 从上游 recon 输出原样拷 (recon 产物 + poseqa + contact + objects),
无缩放无换网格; 各自 `PROVENANCE.md` 末尾已写明"只作运动来源"。take1 138M / take18 192M。

**构建器改造** `L2_Reference/build_reference.py` (向后兼容, take 3 回归逐项复现 v1 的数):
- `--take` (运动来源: rts/replay/layout 落盘) 与 `--assets_take` (物体网格, 恒 3) 拆开; rts 文件名按 take。
- 新增 `--sponge_xy_recenter` / `--sponge_xy_scale`: 海绵面内图案归心 + 幅度缩放, 与盘的
  `plate_motion_scale` 同构, 默认恒等。**take 1 必需** —— 它重建的海绵在盘系半径中位 12.1cm/p90 17.5cm
  (盘半径 9cm), 66% 的行不在盘上 (视频那段在洗别的餐具); 归心减 (−6.53,−0.36)cm + ×0.45 后在盘率 0.95。
- 母带元数据补 `source_take` / `assets_take` / `sponge_xy_recenter` / `sponge_xy_scale`。

**A3 母带定案** (盘心/桌高/压深/低通/物理全同 take 3, 只差运动与海绵 yaw 偏置):

| 母带 | take | 行数 | T_EP | 覆盖率 | 行程 | 接触行 | 在盘率 | 稠密 IK 右/左 | sponge_yaw |
|---|---|---|---|---|---|---|---|---|---|
| clean3_reference_v1 | 3 | 424 | 661 | 0.569 | 87cm | 1.00 | 1.00 | 1.00 / 1.00 | −130 |
| clean1_reference_v1 | 1 | 458 | 705 | 0.648 | 103cm | 1.00 | 0.95 | 1.00 / 1.00 | −20 (归心×0.45) |
| clean18_reference_v1 | 18 | 465 | 714 | 0.634 | 167cm | 1.00 | 1.00 | 1.00 / 1.00 | +15 |

A4 线 (覆盖 ≥0.45 ∧ 行程 ≥70cm) 三条全过。碟心 0/60 三条都碰不到 (13.2cm 海绵进不了 9cm 平底), 资产几何所致。

**踩到的坑**: `--scan_yaw` / `--scan_json` 的稀疏可达每 20 源帧采一次, **会漏行** ——
take 1 在 sponge_yaw=+15 稀疏全通, 稠密 IK 却在第 400 行 2.20cm 失败; 第一版 (未归心) 更是第 450 行 4.91cm 失败。
⟹ **yaw 偏置取可达带内部而非边缘, 且只认稠密 IK 的 ok_ratio**。

**A6 冒烟** (零动作 8 env, `smoke_task.py --modes zero`, 物理 0.3/0.05kg μ1): 两条都 **PASS**。
take 1 认证 1.000 / die_rel 0.500 / 掉落 0 / T_EP 705; take 18 认证 0.500 / die_rel 0.500 / 掉落 0 / T_EP 714
(对照 take 3: 认证 0.125 / die_rel 0.188 / 掉落 0)。认证率差异来自盘运动幅度 (take 1 的盘几乎不动, path 11cm)。

**怎么用**: `CLEAN_REF_NPZ=<母带>` 传给 `local_clean3_task.sh` / `denso_clean3_task.sh` / `msc_clean3_task.sh`;
不传 = take 3。**三条已训练就绪, 未发车** (起训练按规矩要单独批准)。
建议的下一步是拿 §5.11 的冠军配方 (最简六项 + 相对位姿软罚 W0.5) 在 take 18 与 take 1 上各起一条同 seed 对照,
验"这套配方是不是只对 take 3 的运动成立"。

#### 5.12 更正 (2026-09-10 用户 GUI 目检推翻两条结论)

用户跑 `gui_clean_ref.sh` 目检后指出两处, 复核后**两处都成立, §5.12 上文的结论按此更正**。

**① take 1 不是擦盘, 是挤洗洁精 —— 出局。**
抽帧 f0/60/120/180/220/280 目检源视频: 左手扶蓝盘, **右手全程握黄色洗洁精瓶往盘上挤**, 无擦拭。
重建 obj1 的 8.1×20.8×5.8cm 里那个 20.8cm 就是瓶高。这一条根因同时解释了造带时的三个异常:
(a) obj1 在盘系半径中位 12.1cm、66% 行不在盘上 —— 瓶子本来就悬在盘边不贴盘;
(b) 我加的 `--sponge_xy_recenter --sponge_xy_scale 0.45` 是**在把挤瓶轨迹硬掰成擦拭图案**;
(c) GUI 里是海绵而视频里是瓶 —— 多母带口径套用了 take 3 的海绵资产。
⟹ 母带移到 `archive/clean_take1_soap/`; 数据留在 `datasets/clean_tableware/1/` (PROVENANCE 已更正),
是一条干净的"双手持物 + 挤压瓶"数据, 留给将来的挤压/倾倒类任务。
**教训**: 新 take 入库前必须**先目检源视频确认动作语义**, 再谈几何。归心/缩幅这类旋钮能把不合适的
轨迹掩饰成合适的, 我正是用它掩盖了"这根本不是擦盘"。

**② take 18 首版双腕重叠在胸口 —— 重定 yaw。**
用户看到双臂扭曲重叠。复核: 首版 (sponge_yaw=15) **双腕间距 min 3.2cm / p50 7.7cm** (take 3 是 10.7/14.7)。
根因 = 我定 yaw 时只看了"双臂可达 + 覆盖率", **没有判两手会不会撞**。腕目标 = T_world_obj × inv(T_obj_hand),
与 sponge yaw 强耦合, 转 yaw 就把右腕转到左腕头上, 而 IK 照样解得出来 —— 可达 ≠ 不碰。
已给 `build_reference.py --geometry_only` 加**双腕间距诊断** (min/p10/p50), 与可达、覆盖同行打印。
yaw 全周扫描 (步长 30°) 后定 **sponge_yaw = 60**:
| yaw | 可达(右) | 腕距 min | 零动作 die_rel | 海绵手内滑移峰值 |
|---|---|---|---|---|
| 15 (首版) | 1.00 | 3.2cm ✗ | 0.500 | 1.58cm |
| **60 (定版)** | 1.00 | **11.0cm** | **0.000** | **0.30cm** |
| 75 | 1.00 | 13cm | 0.727 | 2.3cm |
| 90 | 1.00 | 15.2cm | 0.933 ✗ | 2.54cm |
腕距不是越大越好: yaw 越往 90 转, 海绵越是被**横着拖**而非顺长轴拖, 抓握力矩变大, 零动作就滑脱。
定版 take18: 465 行 / T_EP 714 / 覆盖 0.634 / 行程 167cm / 双臂稠密 IK 1.00 / 腕距 min 10.9cm /
零动作 **死亡 0.000** 掉落 0 海绵滑移 0.30cm (对照 take 3: 腕距 10.7cm, 死亡 0.188)。冒烟 PASS。

**新判据入册**: 母带的海绵 yaw 偏置必须同时过三关 —— ① 双臂稠密 IK ok_ratio=1.00
② **双腕间距 min ≥ 10cm** ③ 零动作不掉不死。缺第 ② 关就会出"两手叠在胸口"。

**扩数据的真实盘点 (2026-09-10, 28 条源视频抽帧缩略图目检)**: 源里**大半是擦餐具不是擦盘**,
真正"一手端盘一手抹布"的约 12~15 条。但上游只重建了 5 条, 其中 take 0/8 **只追踪到 1 个物体**
(缺 object_1, v17A 漏标), take 1 是挤洗洁精 ⟹ **现成可用的擦盘 take 只有 3 和 18 两条**。
要再扩必须走上游重建管线: 优先 take 0/8 (动作对, 只差补第二个物体), 再挑源里其它擦盘条新重建。

**护栏落地 (2026-09-10)**: `build_reference.py --min_wrist_gap_cm` (默认 10.0) —— 稠密 IK 之后核双腕间距,
不过就**拒绝写盘**, 报最小值并提示改 yaw 重扫; `0` 可强行放行。已验: 首版 yaw=15 被拒 (min 2.6cm, 无文件产出),
定版 yaw=60 放行且与已发布 `clean18_reference_v1.npz` **逐位一致 (差 0.00e+00)**。
这条闸的意义: "IK 全通"这个绿灯本身不足以说明母带可用, 必须再加一条"两条臂不许打架"。

### 5.13 take 18 盘是倒扣的 (2026-09-10 用户 GUI 目检提出) —— 查清并给出路线

**事实确认 (三重证据一致)**:
1. 源视频抽帧放大: take 18 绿盘**倒扣**, 中间那圈是圈足, 擦的是外底; take 3 黄盘正放, 擦的是内面。
2. 网格自身形状: take 3 盘沿 y=+1.27cm / 盘心 −1.05cm ⟹ 凹面朝 **+y**; take 18 盘沿 +0.07 / 盘心 +0.53
   ⟹ 凹面朝 **−y**。两条方向相反。
3. 重建位姿: 两条的网格局部 +y 在世界都朝上 (cos 0.773 / 0.628)。
⟹ take 3 是"凹面(内面)朝上", take 18 是"凸面(外底)朝上"。当前 take18 母带按 take 3 的口径把盘当正放, **与视频不符**。

**已实现的开关** `build_reference.py --plate_flip`: 只把**盘网格**绕规范 x 转 180° (规范系仍 z-up ⟹ 海绵照样在盘上方、
右手不受影响, 只有左手跟着翻), 擦拭面剖面自动改用网格另一面。
⚠ 不能用 `--plate_tilt_deg 180` 代替 —— 那个转的是规范系, 会把海绵和右手一并翻到盘底下 (实测右臂可达直接 0.00)。

**实测三条, 前两条是好消息, 第三条是拦路的**:
| 项 | 结果 | 判 |
|---|---|---|
| 左臂(端盘)翻转后可达 | 稠密 IK ok 1.00, 双腕间距 min 13.1cm (未翻是 11.0) | ✅ 几何过 |
| 擦拭面覆盖 | 覆盖 **1.000**, 碟心 **60/60**, 缘带 156/156 (未翻是 0.634 / 碟心 0/60) | ⚠ 见下 |
| **倒扣握持** | **3s 静持左手漂 33.2cm / 175.7°, 盘直接脱手** (未翻 2.97cm / 13.9°) | ❌ **拦路** |

**❌ 拦路根因**: 现有盘 GraspPose `27_Quadpod` 是**拇指压顶面 + 四指托底**, 在底座上合成的。盘一倒扣,
重力就把盘从拇指上拽下来 —— 这个握法**结构上不支持倒扣**。⟹ "最小改动"做不成: 翻盘必然要换握法。

**⚠ 第二个坑 (即使换了握法也在)**: 训练用的是 **take 3 的盘资产**, 而 take 3 的视频里**盘底从未被看见** ——
SAM3D 把背面补成了一个**平盖**。实测: take3 翻转后朝上的面在 r<6cm 是平的 (1.04~1.16cm) 然后直落到 −0.53;
take 18 自己观测到的真盘底则在 r≈5cm 有**圈足凸脊** (1.21cm) 再回落到 0.6。两者相关系数仅 **0.471**, RMS 差 0.60cm。
⟹ 拿 take3 资产翻过来擦, 擦的是**重建虚构的平盖**, 不是真盘底; 覆盖率 1.000 也正是平盖带来的假象 (判据随之失去区分度,
且与 take 3 不再是同一把尺, 跨带成绩不可比)。

**三条路线与代价**

| 路线 | 做什么 | 代价 | 结论 |
|---|---|---|---|
| **A 翻 take3 资产** | `--plate_flip` + 为倒扣盘重合成 GraspPose | Dexonomy 一轮 + 人工点选; 擦的仍是虚构平盖; 判据与 take3 不同尺 | ✗ 不推荐 |
| **B 用 take18 自己的盘** | 入库其网格(×0.75 到 18cm) + 烘 USD + Dexonomy 合成倒扣握法 + make_prior + 重造带 | 约一天, 含人工点选; **资产与 take3 分家 ⟹ 多母带混训作废** | 要保真时走这条 |
| **C 不翻 (现状)** | 零改动, 台账如实记为已知偏差 | 与视频的"擦哪一面"不符 | ✅ 现在推荐 |

**推荐 C, 理由不是省事**: 母带本来就**不回放重建的盘姿态** —— 盘 conf_rot 只有 3~16 不可用, 姿态早已按用户裁定 ⑥
换成常量水平。"擦哪一面"在当前口径里**本来就是设计选择而非数据**。而 take 18 存在的目的是**运动多样性对照**,
要成立就必须与 take 3 同资产同判据同握法, 只差运动。翻面会同时改掉资产语义与判据尺度, 把对照实验毁掉。
⟹ 先按 C 跑完泛化对照; 等要"复现视频"这个目标时再单独走 B, 作为"倒扣盘"变体任务。

`--plate_flip` 旗**保留但默认关**, 帮助文本已写明单用会脱手, 给将来走 B 时用。

### 5.14 take 18 推倒重做 (2026-09-10 用户裁定, 走 §5.13 的 B 路线)

用户裁定: "既然重建里是反的, 那就把 take18 为准备做的错误内容都先删除, 只保留重建数据, 然后重新生成 GraspPose 再造母带。"

**已删 (移到 `archive/clean_take18_wrong_upright/`, 附 README 说明作废原因)**:
`clean18_reference_v1.npz` (465 行, 擦盘内面, 与视频相反) · `datasets/clean_tableware/18/scene_layout.json` ·
`launch_logs/smoke_take18.log` · `probe_hold_t18_{flip,noflip}.json` · `settled_oh_t18_{flip,noflip}.json`;
scratchpad 里的候选带 (y60/y75/flip) 已清。
**保留**: `datasets/clean_tableware/18/` 全部重建数据 (poseqa 物轨 / replay_world 手轨 / objects 网格 / contact / confidence)。
`tasks/Clean/3/A_Design/L2_Reference/` 现在只剩 take 3 一条可用母带, README 已改。

**GraspPose 重合成由用户另一路进行中** —— 目标 = **倒扣盘**的端盘握法 (盘底朝上)。
拿到新先验后我这边接着做的三步 (等用户消息):
1. take 18 自己的盘网格入库并缩到 ⌀18cm (上游 [23.97, 3.24, 24.0] × 0.75 → [17.98, 2.43, 18.0], 与 take 3 同尺寸口径),
   烘 textured USD + 物理 cache USD;
2. `build_reference.py --take 18 --assets_take 18 --plate_flip` + 新先验造带, 过三关: 稠密 IK ok=1.00 /
   双腕间距 min ≥10cm (闸已内置) / 零动作不掉不死;
3. 目检 (`gui_clean_ref.sh 18`) + 冒烟, 再报批发射。

**要记住的两条前提** (§5.13 实测):
- 倒扣后左臂**可达没问题** (稠密 IK 1.00, 腕距 13.1cm), 拦路的只有握法 —— 所以新 GraspPose 是唯一硬缺口;
- 换成 take 18 自己的网格后**资产与 take 3 分家**, 多母带混训作废; take 18 从"运动对照"变成"倒扣盘变体任务"。

### 5.15 消融 Base / A1 / A2 (2026-09-10 用户批准发车, 对位 Pour 的 Base/NH/NHNC)

**为什么 Pour 的 Ablation1 在 Clean 上要重新映射**: Pour 的 `POUR_VARIANT=OBJ` 撤的是**人手形状指引**
(不加载 `human_*_q/f` + 关掉奖励里的 `W_HAND` 绿0/黄0.5/红0.8)。而 Clean 母带**根本没有 human_* 层** ——
造带脚本读 `replay_world.npz` 只取了 fps 一个标量, 指列是 squeeze 常量 (std 恰好 0.0), 奖励里也没有形状项。
⟹ **Clean 的现役 Base 本来就等价于 Pour 的 NH**。要做这条对照必须**先把人手层补上**, 方向与 Pour 相反。

**核实纪要 (我先犯了两个错, 用户目检推翻)**
- ❌ 我只量了腕的**平移**就断言"手臂链冻死、人手轨迹是废的"。实测朝向后推翻: take3 右腕相对首帧转角
  中位 **21.7°** 最大 33.7° 逐帧累计 406°, 左腕 4.7° (只扶盘); take18 双腕 21~23°。**腕根不动 + 腕旋转**
  正是坐姿擦盘的真实姿势, 数据没问题。
- ❌ 我没做**尺度记账**就拿"物体动 10cm vs 腕动 0.19cm"当矛盾。实际: 物体被类别先验 ×1.99 放大过, 人手没有。
  杠杆核算 —— 腕→中指尖 15.9cm, 30° 转角 × ~10cm 力臂 ≈ 5.2cm, 与**未放大**的物体位移 4.9cm 吻合。数据自洽。
- ★ 核实中发现: **我们的母带与人的策略相反**。take3 右手 人=平移 0.27cm/转角 21.7°, 母带=平移 13.1cm(路径 96.6cm)/
  转角 **0.0°**。零转角源于造带把海绵朝向设成常量。⟹ 腕朝向这条另案 (用户裁定本轮**只进指列**)。
- ★ 人手层**现成**: `ref_qpos_{left,right}.npz` 就是 DexPilot 重定向到机器人手的产物, 300/300 有效,
  22 指关节峰峰值 18.7~44.8°, 关节顺序与 `GENERIC_JOINT_ORDER` **逐位相同**。造带脚本从没读过它。

**尺度 (用户 2026-09-10 重申)**: 盘 ⌀18cm (重建 ⌀24 × 0.75) / 海绵 7.6×3.8×13.2cm (×1.99) —— 即现役训练用的这套, **不动**。
人手指引只用**角度**, 角度无量纲, 不受 ×1.99 影响 ⟹ 无需任何尺度对齐, 这是"只进指列"路线最省事的地方。

**实现**
- `build_reference.py`: 默认写人手指姿层 (`--no_hand_ref` 可关)。`human_*_f = 母带 squeeze 姿 + (finger_qpos[k] − finger_qpos[0])`,
  按 `source_times` 插值到母带行。**首帧归零焊在 squeeze 上** (pour17 母带同款"手=首帧焊 GraspPose 增量"), 于是 k=0 处
  与现役指列逐位相同, 抓握不被破坏。产出 `clean3_reference_v1h.npz` (sha 57d57e4c) —— **与 v1 在全部原有键上逐位相同**
  (已核, 差 0), 新增 human_right_f/human_left_f/hand_ref_src。
- `task_config.py`: `CLEAN_S2_HAND_REF` / `CLEAN_S2_HAND_W`(默认 1.0) / `CLEAN_S2_CONF_FLAT`。
- `clean_task_env.py`: 覆写 `_ff_fingers` —— 认证后 `f += W × hand_f_delta[k]` (前馈与观测**同一个出口**, 不会
  "喂进去的和看到的不是一回事"); CONF_FLAT 在 tier 计算处一律置 1(黄), 皮筋与观测独热随之常量。
- `train_task.py`: 三键进世界指纹。
- ⚠ **与 Pour 的口径差**: Pour 的 W_HAND 是**奖励项**权重且按档位; Clean 走**前馈**且用常量权重 (逐行切档会让指姿抖)。
  判读时不要当成同一个机制。

**三条**
| 臂 | 旗 | 状态 |
|---|---|---|
| **Base** | `CLEAN_S2_HAND_REF=1` (人手指姿指引) | Denso GPU0 `Clean3_ablBase_s42` 发车 |
| **A1 撤人手指引** | 全关 | = 现役冠军 `Clean3_taskS_s42` (16/16), **直接复用不重跑** (用户裁定) |
| **A2 再拍平置信度** | `CLEAN_S2_CONF_FLAT=1` | Denso GPU1 `Clean3_ablCF_s42` 发车 |
三条共用 `CLEAN_S2_SOFT_REL=1 W=0.5` (冠军承重项) 与母带 v1h; A1 用 v1, 两带在读到的键上逐位相同。

**冒烟 (零动作 8env, Denso)**: Base PASS (cert 0.000 / die_rel 0.375, 人手层认证后才生效故零动作下不参与);
A2 PASS (档位 绿0/黄424/红0 已确认拍平)。

**发车 2026-09-10 ~06:1x UTC**: `denso_clean3_abl.sh NAME GPU SEED [FLAGS]`; 512env / 30M / seed 42;
Base pid 268550 @GPU0 FPS≈594, A2 pid 268873 @GPU1 FPS≈575 ⟹ 各约 **14~15 小时**。日志 `~/<name>.log`。
世界指纹已核: 两条的 S2_HAND_REF/S2_CONF_FLAT 与母带 sha 都记进了 world.json。

**判读 (预注册)**: 评测用 `eval_clean.py --num_envs 16 --release_row 10 --jitter 0`, 与 A1 那次一字不差。
- Base − A1 = 人手指姿指引承重多少 (A1 已知 16/16, 所以 Base 只能持平或更差 —— 若明显更差, 说明人手指姿
  与我们的物体反推腕参考**互相打架**, 那正好呼应"两者策略相反"这条发现);
- A1 − A2 = 置信度分档承重多少, 这个增量可与 Pour 的 NH 10/30 → NHNC 4/30 直接并排。

### 5.16 take 18 倒扣盘变体: 入库→先验→接线→母带 (2026-09-10, **三关只过两关, 未发车**)

**用户裁定两条** (2026-09-10): ① 盘用 Dexonomy 合成时实际吃进去的那份网格入库, **不重跑合成**;
② 洗碗布用 **take 3** 的资产 + 现成 `Clean3_sponge_right.npz` (take18 的布是薄壳: 实心度 0.09 / 环面拓扑 /
长轴短 40%, 6294 条候选里只有 3% 满足"擦盘面零接触", 去重后功能池仅 38 —— Dexonomy 交付 README 自己建议别用)。

**⚠ 我先报错了一次, 这里更正**: 我最初核账说"合成用的网格把**圈足削平了**"。逐档剖面复核推翻了这句 ——
**朝上的那一面 (= 倒扣后被擦的盘底) 两份网格逐档完全相同, 最大差 0.00mm, 圈足凸脊完整**
(r=4.75~5.25cm 处 1.22~1.23cm, 盘心 0.53cm)。差异**全在朝下那一面** (倒扣后朝下的盘内凹面):
r=4.75cm 处 A(入库)=+0.15cm vs B(原网格×0.75)=−1.20cm, 最大差 **13.5mm** —— 即 Dexonomy 那份把
背面圈足内侧的凹槽**填实了** (体积 89.8 vs 76.6 cm³, bbox 厚度 1.90 vs 2.43cm 也由此而来)。
⟹ 影响记账: **擦拭面零损失, "换 take18 就是为了真盘底"这个初衷完全保住**; 打折的是手托的那一侧 ——
指尖少了一圈约 1.2cm 深可勾的沟。先验与 sim 资产同为 A, 自洽。原尺重建网格留档
`objects/object_0/object_mesh_scaled_final_raw24.obj`。

**方位角坑 (踩了两次才踩明白)**: 用户点选的候选 006 (`27_Quadpod__s12_31_11`) 在盘规范系里腕方位角 −49.1°,
take3 先验是 161.2°, 差 149.6°。① 按 take3 配方 (plate_yaw 0) 造带 → **左臂稠密可达 0.00**;
② 我在母带侧用 `--plate_yaw_deg -150` 补偿, 三关的前两关都过了 (双臂 1.00 / 腕距 11.2cm), 但
③ env 起手时 `pregrasp/env.py:_load_grasp_prior` 会把盘按**视频 yaw 0°**摆好再核 IK, 实测 **21.0cm**,
断言拦下: "这个候选在视频 yaw 下够不着, **换候选(别改 yaw)**"。母带侧的 plate_yaw 不进 env 脚手架
(PreGrasp/抬升斜坡/接触分数图那一套用的是自己的摆放), 硬凑 = 两个世界各转各的。
⟹ 按闸的处方换候选: **`27_Quadpod__s14_17_6`** (渲染 `016_g000.5_...`, 方位角 166.5°, 与 take3 差 5.3°)。
盘是圆盘、绕法向自转不可观测, 所以两条候选**是同一种抓法**, 只差一个不可观测的角。006 的先验留档
`Clean18_plate_left_cand006_az-49.npz`。换候选后 env 闸 IK err **0.36cm** 通过。

**判据几何的隐藏 bug (顺带修掉, 与 take18 无关也存在)**: `progress_batch.CleanGeometry` 把 **take3 盘的
顶面剖面写死**在 dataclass 默认值里, 而 `clean_task_env` 三处 (`S_ref`/`S`/`PBj`) 都用默认值构造 ——
换盘后判据仍在拿 take3 的碟形剖面量 take18 的平盘, 零动作 gap 报 **−16.7mm** ("海绵陷进盘里 1.7cm"),
覆盖/接触/落座全部失真。母带本来就存了 `plate_top_profile_r/y` (hold_env 早就是这么读的),
现加 `CleanGeometry.from_reference()` 把判据也接到同一处, **逐项比对、只换真不一样的那项**
(take18 换的是盘, 海绵仍是 take3 那块 ⟹ face_off 不跟着动); 差 <0.5mm 判为同一块盘走默认值,
以免 take3 冠军线的判据尺子被 0.2mm 的手抄舍入差挪动。对照已跑: take3 默认口径逐位不变。

**产物与三关**
| 件 | 路径 / 值 |
|---|---|
| 盘网格 | `datasets/clean_tableware/18/objects/object_0/object_mesh_scaled_final.obj` 17.97×1.90×18.00cm 133210 顶点 |
| 盘先验 | `tasks/pregrasp/priors/Clean18_plate_left.npz` (s14_17_6, 8 接触点) |
| USD | `objects/object_0/object_mesh_scaled_final.usd` + `18/cache/object_1.usd` (布的物理 USD 烘到 18/ 下, 不动 take3 资产指纹) |
| clip | `clips.Clean18_plate` (盘=take18 / 布=take3) |
| 母带 | `L2_Reference/clean18_reference_v1.npz` 465 行 @20Hz (23.2s), `plate_yaw 0 / sponge_yaw 60` |
| **① 稠密 IK** | right **1.00** / left **1.00**, bad_rows 空 ✅ |
| **② 双腕间距** | min **11.6cm** (take3 10.7 / 旧 take18 11.0 同档) ✅ |
| **③ 零动作冒烟** | **✗ 未过 (但闸本身很钝, 见下)**: die 0.917 全是 die_rel, 掉落 0 / 倾覆 0; 回合在 release(第 50 行) 后 1~6 步就死 |
| 参考自身几何 | 覆盖 **1.000** (碟心 60/60, 缘带 156/156) / 行程 167cm / 接触行 1.00 |

⚠ **覆盖率在 take 18 上失去区分度**: 倒扣盘外底近乎平面, 13.2cm 海绵够得着碟心, 参考自身就已经
覆盖 1.000 (take 3 是 0.569 且碟心 0/60), 而 A4 线才 0.45。这条线上要判"擦得好不好"必须另找有头寸的量。
`--geometry_only` 的判据几何也一并接到实测剖面 (原来同样在拿 take3 剖面量别的盘)。

**③ 的现象与已排除项** (8 env, 判据几何修好之后):
- 死因 = **海绵**相对锁存位姿转了 **21.1°** > 20° 死线 (盘自己 12.2°/1.14cm, 在线内);
- 不是"海绵架在圈足上摇": 逐点核过, take18 平盘上海绵足迹 **89/91 点贴面** (take3 碟形只有 52/91), 贴合更好;
- 不是母带的错: 母带自身 gap_min = **−1.8mm** (take3 −1.9mm), 海绵擦盘面与盘法向严格 180°;
- env 里步 0 的 gap 是 **−11.3mm** (母带 −1.8mm) —— 说明**入 env 之后**海绵比母带规定的又陷进去约 1cm,
  与"盘在左手里转了 12°"同量级 (sin12°×6cm≈1.2cm), 指向**新抓法握不稳倒扣盘**这条 (与 §5.13 同类,
  只是从"3s 漂 33cm 脱手"缓和成"转 12°")。
**同代码同会话的 take 3 对照 (跑了两遍, 8 env)**: die **0.440 / 0.125** —— 这个率在 8 env 下抖得厉害,
**不能单看它判读**; README 记的 0.188 落在这两次之间。看连续量才稳:

| 量 | take 3 (两遍) | take 18 | 死线 |
|---|---|---|---|
| 海绵相对锁存转角 | 17.19° / 15.45° | **21.09°** | 20° |
| 盘相对锁存转角 | 4.83° / 5.30° | **12.19°** | 20° |
| 盘相对位移 | 0.67 / 0.72cm | 1.14cm | 3cm |
| 掉落 / 倾覆 | 0 / 0 | 0 / 0 | — |

⟹ 真正的结论不是"take18 崩了", 而是: **take 3 的零动作本来就贴着 20° 死线跑 (15~17°), take 18 只是被
顶过去了一点 (21°)**; 而根在**盘在左手里转 12.2°, 是 take 3 的 2.3 倍** —— 倒扣盘这个握法的抗扭刚度不够。
take 3 拿 15~17° 的起点训到了 16/16, 所以 take 18 未必训不出来, 只是开局有大批回合秒死。

⟹ **未发车**。下一步候选 (待拍板): **(a) 推荐**: 从 325 条池里挑 2~3 条"更包住盘沿"的候选, 各跑一次 probe_hold 静持探针 (3 分钟一条,
不必整轮冒烟), 取**盘转角最小**的那条 —— 直攻 12.2° 这个根; (b) 直接发车, 接受更难的起点 (1 卡 ~15h);
(c) 给倒扣盘线单独放宽 S2 死线 (会让 take18 与 take3 不同尺, 判据分家要在台账里记死, 不推荐)。

发车命令 (三关全过后才可用, 需单独批准):
```bash
bash tasks/Clean/3/C_Wiring/local_clean3_task.sh Clean18_taskS_s42 42 \
  CLEAN_CLIP=Clean18_plate \
  CLEAN_PRIOR_PLATE=/home/lyh/Project/RL_Correction/tasks/pregrasp/priors/Clean18_plate_left.npz \
  CLEAN_REF_NPZ=tasks/Clean/3/A_Design/L2_Reference/clean18_reference_v1.npz \
  CLEAN_S2_SOFT_REL=1
```
`CLEAN_CLIP` / `CLEAN_PRIOR_PLATE` / `CLEAN_PRIOR_SPONGE` 三个环境变量在 `task_config` 与 `hold_env`
**两处同源** (场景是 hold_env 建的, task_config 供 world.json 指纹与外部脚本; 两处必须同一组变量,
否则会出现"指纹写着 take18、场景装的却是 take3"), 三项都进 world.json。
`build_reference.py` 新增 `--sponge_assets_take` 让盘布资产分家; layout 的 `extent_cm` 改成按网格实测。

### 5.17 死线放宽 (2026-09-10 用户裁定): "只要不是抓不住掉落, 手里有一些转动很正常"

**用户裁定原话**: 母带 GUI 目检通过 ("这个母带看着不错"); **死线可以放开一些, 只要不是抓不住掉落 ——
手里有一些转动很正常, 因为原视频洗碗确实有发生一定旋转和偏移**。

**先量了原视频, 结论是"量不出可用的数"** (照 [下结论前先直接核验] 的规矩记在这): 用 Kabsch 从
`replay_world.npz` 的 21 关节手云求手的刚体旋转, 与 `obj_pose_all` 的物体旋转做差:
| | 手内转角 中位 / 最大 | 手内位移 中位 |
|---|---|---|
| take3 盘×左手 | 116.8° / 142.4° | 13.2cm |
| take18 盘×左手 | 67.5° / 179.9° | 17.6cm |
| take18 布×右手 | 25.3° / 38.7° | 10.1cm |
盘转 180° 显然非物理 —— 与台账已记的"盘 conf_rot 只有 3~16 不可用, 母带早已把盘姿态换成常量水平"一致,
**盘那两行是重建噪声, 不能拿来定阈值**; 布那行 (25~39°) 相对可信, 方向上支持用户的判断。
⟹ 阈值不从这里取, 改按"死线只用来抓丢失、不抓转动"来定。

**实现 (默认一律不动)**: `CLEAN_S2_DIE_ROT_DEG` (默认 20°) / `CLEAN_S2_DIE_POS_CM` (默认 3cm) /
`CLEAN_S2_PLATE_TILT_DIE_DEG` (默认 30°) 三个环境变量。**默认保持原值是有意的** —— Denso 在跑的
`Clean3_ablBase/ablCF` 与 take3 冠军都用 20° 这把尺子, 改默认它们立刻不可比。要放宽在发射脚本里显式写。
⚠ 同时把**软罚(H-C3)的归一化跨度与死线解耦**: 新常量 `S2_SOFT_SPAN_POS/ROT_DEG` 恒为 3cm/20°,
`clean_task_env` 的 `tp/tr` 改用它。否则放宽死线会把冠军承重项 (SOFT_REL W=0.5) 的罚斜率一起压平 ——
一次改两个参数, 判读表失效 (台账"一次只改一个参数")。默认路径数学上逐位不变 (span 20 = 旧 die 20)。
两项都进 world.json 指纹。

**take18 零动作三档实测 (8 env)**
| 转角死线 | die | die_rel | 盘转角 | 布转角 | 盘位移 | die_drop / die_table / die_tilt |
|---|---|---|---|---|---|---|
| 20° (原) | 0.917 | 0.917 | 12.2° | 21.1° | 1.14cm | 0 / 0 / 0 |
| 45° | 0.500 | 0.500 | 18.8° | 39.2° | 1.86cm | 0 / 0 / 0 |
| **180° (等于关掉)** | 0.500 | 0.500 | 22.3° | 38.0° | 2.29cm | 0 / 0 / 0 |

**判读**:
- 20°→45° 把死亡率砍半, 但剩下一半仍死在 die_rel (布被放开后继续转到 39°, 一部分越过 45°) ⟹ 45° 这一档
  **转角仍是主要杀手**, 与用户裁定的方向不符。
- 45°→180° (关掉) 死亡率不再下降, 仍是 0.500, 但**死因换了**: 转角已不可能触发 (最大 38°), 剩下的全是
  **位移 > 3cm**。`done_at=[110,106,54,110,107,53,110,53]` —— 一半在第 53 步左右位移超限而死, 另一半活到
  第 110 步 (= release 50 + 认证预算 60) 因**认证超时**结束。
- **三档里 die_drop / die_table / die_tilt 恒为 0** —— 按用户判据 (抓不住掉落) 这些回合**一个真失败都没有**。
- ★ 真正拦住零动作的其实是**认证** (放手后 60 行内要 1cm/5° 连续 10 步), 不是死线: 三档 cert 全是 0.000。
  take 3 零动作同样 cert 0.000~0.080, 它照样训到 16/16 —— 所以这一项零动作过不了**不构成不能训的证据**,
  但要在训练里盯 `sr/cert`: 认证不了时钟就不走, 学不到擦。

**建议 (待拍板)**: take18 线取 **`CLEAN_S2_DIE_ROT_DEG=180` (转角死线关掉)**, 位移 3cm / 掉落 / 落桌 /
盘倾角 30° 四条保留 —— 这正是"只要不是抓不住掉落"的字面实现。**转动仍然被罚**(软罚跨度恒 20°, 斜率没变),
只是不再一刀切死: **整形留着, 铡刀去掉**。取 45° 的话转角仍是主要死因, 不符合裁定。
默认 (take3 线 / Denso 在跑的两条) **一律不动**。

### 5.18 take18 三臂 Base/A1/A2 已备好 (2026-09-10, **待批准发车**)

用户要求把 Base/A1/A2 三臂都准备好。对位 take3 的 §5.15, 三臂共用冠军承重项 `CLEAN_S2_SOFT_REL=1` (W=0.5)
与 take18 母带 `clean18_reference_v1.npz`; **take3 那边 A1 可以复用冠军存档, take18 三臂必须真跑**
(资产分家, 没有现成对照)。

**发射脚本 `tasks/Clean/3/C_Wiring/clean18_abl.sh <base|a1|a2> SEED [GPU] [KEY=VAL ...]`**:
把 take18 的四件套 **焊死在脚本里**, 三臂共享 —— `CLEAN_CLIP=Clean18_plate` /
`CLEAN_PRIOR_PLATE=.../Clean18_plate_left.npz` / `CLEAN_REF_NPZ=.../clean18_reference_v1.npz` /
`CLEAN_S2_DIE_ROT_DEG=180`。§5.16 的坑 (漏一件就静默跑成 take3 的场景) 不给第二次机会。
`CUDA_VISIBLE_DEVICES` 由第三个位置参数给, 本机/Denso/msc 通用。

**零动作冒烟 (8 env, 三臂全 PASS, 转角死线 180)**
| 臂 | 旗 | die | die_rel | drop/table/tilt | 核验行 |
|---|---|---|---|---|---|
| base | `CLEAN_S2_HAND_REF=1` | 0.375 | 0.375 | 0/0/0 | 人手指姿增量 中位 8.63° 最大 42.77°, 首行 0.000° ✅ |
| a1 | (无) | 0.500 | 0.500 | 0/0/0 | die=3cm/180° ✅ |
| a2 | `CLEAN_S2_CONF_FLAT=1` | 0.375 | 0.375 | 0/0/0 | 档位 绿0/黄465/红0 ✅ (拍平生效) |
三臂零动作都没有真掉落; 剩余的 die 全是位移 >3cm (转角已关)。cert 三臂都是 0.000 (与 take3 同, 见 §5.17)。

**⚠ A2 在 take18 上扰动量小得多 (发车前请裁定)**: take18 母带档位 = **绿0/黄333/红132**,
take3 是 绿236/黄110/红78。take18 整条重建置信度够不着绿档门槛 (PROVENANCE: conf_pos 62.0 /
conf_rot 3.0 判 mixed; take3 是 86.5/0.0 判 good)。⟹ A2 在 take18 上只把 132 行红→黄, 另外 333 行本来就是黄的;
在 take3 上它是把 236 绿 + 78 红一起拍平。**同一个消融在两条线上的扰动量差着量级**, take18 的 A1−A2
读出来接近 0 **不能**解释成"置信度分档不承重", 只能说明这条带上本来就没多少档位可分。
三个选项 (待用户定): ① 照跑并把这条限制记死; ② A2 换成随机打乱分档 (对位 Pour 的 `TIER_SHUFFLE` 主对照),
在 465 行上能真造出差别; ③ take18 不做 A2, 只跑 Base/A1。

**排期**: 本机只有 1 张 4080S, 三臂串跑 ≈45h。Denso `Clean3_ablBase/ablCF` 今晚跑完腾出 GPU0/GPU1,
明早可三臂并行 (2080Ti 约 15~16h/条)。

### 5.19 三机部署核验 (2026-09-10, 用户: "不要训练完了跟我说要重新训练")

用户裁定: **Base 在 UCBY, 两条 Ablation 在 Denso**; 发车前先把环境与数据统一。核验做成**两层**, 全部留证:

**第一层 · 文件逐位相同**: `clean18_manifest.sh` / `c18_check.sh` 按一份固定清单 (170 个运行时文件:
rl_rebuild + pregrasp + Clean 全树代码 / 标定 (residual_bound, shell_points) / 先验 / 机器人与手 USD /
母带 / take18 盘 + take3 布的数据) 逐文件 sha256 + 汇总 digest。
同步后 **本机 = Denso = UCBY = `DIGEST=6bbc4e56daa4584f7708c9c03e61a194`**。
(同步前: Denso 缺 9 个, UCBY 缺 45 个 —— UCBY 的仓是 Pour 时代快照, 连 `tasks/Clean` 都没有。)

**世界级资产早先已单独核过** (三处逐位相同): `vega_1p_sharpa_fixedtorso.usd c2bba25a…` /
`arm_residual_bound.json a8c8d5ff…` (A6000 零迁移那次的真凶, 这次没缺) / `arm_shell_points.npz 8575b23b…` /
torch 2.7.0+cu128 / isaaclab 0.54.3。

**第二层 · 同输入跑出同样的数** (8 env 零动作, 三机):
| 项 | 本机 | Denso a1 | Denso a2 | UCBY base |
|---|---|---|---|---|
| 物理覆写 | 0.300kg μ1.00/1.00 指垫1.0 | 同 | 同 | 同 |
| 判据几何 | y∈[0.53,1.23]cm 18档, face_off 1.91cm | 同 | 同 | 同 |
| 档位 | 绿0/黄333/红132 (a2: 绿0/黄465/红0) | ✅ | ✅ | ✅ |
| obs/act/T_EP/die | 364 / 58 / 714 / 3cm·180° | 同 | 同 | 同 |
| 人手层 Δ | 中位 8.63° 最大 42.77° 首行 0.000° | — | — | **同** |
| die | base .375 / a1 .500 / a2 .375 | .375 | .375 | .500 |
| die_drop / table / tilt | 0 / 0 / 0 | 同 | 同 | 同 |
die 率在 8 env 下本来就抖 (§5.17 记过: 同机同配置复跑 take3 得 0.440 / 0.125), 其余**确定性量全部逐位对上**。

**核验过程抓到三个坑 (都属于"不查就会训完才发现")**
1. **清单漏了 `world_fused.npz`** (相机锚定的 c2w, `replay_grasp.load_replay_grasp` 从 mesh/npz 旁找)
   ⟹ Denso 起步 13s 断言崩。改为整树同步 `datasets/clean_tableware/{3,18}` (只排除运行时不用的
   `objects/*/textured/` 与 `masks/`)。教训: 运行时依赖别靠脑补, 用审计钩子 (`trace_open.py`, 记录
   smoke 实际打开的仓内文件, 本次 87 个) 或整树同步。
2. **UCBY 的 `/tmp/isaaclab/logs` 属于别的用户** ⟹ 起步 6s `PermissionError`。IsaacLab 的
   `configure_logging` 默认写 `tempfile.gettempdir()/isaaclab/logs`; 发射脚本加"写不进才换 `TMPDIR`"。
3. **UCBY 的 IsaacLab 是另一棵树** `~/IsaacLab_DexAssemble` (本机/Denso 是 MagicSim/Third_Party/IsaacLab)。
   版本号都报 0.54.3、都是 307 个 py, 但 py-tree digest 不同 (本机/Denso `56016b8f…` vs UCBY `3a86c106…`)。
   逐文件 diff: **只差 5 个** —— `envs/mdp/commands/{__init__,commands_cfg,velocity_command}.py`,
   `envs/mdp/observations.py`, `sim/converters/urdf_converter.py`。这些是 **manager-based env** 的命令/观测项
   与 **URDF→USD 转换器**; 我们走 `DirectRLEnv` + `MeshConverter` + 预烘机器人 USD, 全栈 grep 零引用,
   本机 trace 的 87 个文件里也没有。`isaaclab/envs/__init__.py` 有 `from . import mdp` ⟹ 模块会被加载,
   但其中的函数一个都不会被调用。**⚠ 残余风险**: 上表证明的是"起手横幅 + 8env 零动作"这一层相同,
   不是 30M 步全程相同。要更硬的保证只能把 Base 也放 Denso (等 GPU0/1 今晚腾出)。

**发射脚本** `clean18_abl.sh` 已内置: take18 四件套 + G-A 物理 + 多卡机自动 `RL_ISAAC_NO_GUARD=1`
(flock 按机器不按卡, 会静默排队) + TMPDIR 兜底。

**UCBY 的两个 `build_reference` 僵尸 (feiyang, 9.7 天) 清不掉**: 属 feiyang 用户, 本会话用 yanghong 登录,
权限上杀不了 (且该命令被本地权限层拦下)。**不挡事**: 两张 A6000 各 48GB, 它们共占 1GB / 算力 0%。
要清得 feiyang 本人或 sudo。

### 5.20 take18 三臂发车 (2026-09-10 ~22:13 UTC, 用户选 A: Base 上 UCBY)

用户在 §5.19 给的 A/B 里选 **A** —— 三条今晚同时开跑, 接受"UCBY 的 IsaacLab 差 5 个未被调用的文件"这条残余风险
(§5.19 已论证 + 冒烟对齐, 但只覆盖起手与 8env 零动作, 不是 30M 步全程)。

| 臂 | 机器/卡 | run 名 | pid | 臂旗 |
|---|---|---|---|---|
| Base | UCBY GPU1 | `Clean18_ablBase_s42` | 2075579 | `CLEAN_S2_HAND_REF=1` |
| A1 | Denso GPU2 | `Clean18_ablA1_s42` | 421744 | (无) |
| A2 | Denso GPU3 | `Clean18_ablA2_s42` | 421863 | `CLEAN_S2_CONF_FLAT=1` |

三条共用: `CLEAN_CLIP=Clean18_plate` / `Clean18_plate_left.npz` / `clean18_reference_v1.npz` /
`CLEAN_S2_DIE_ROT_DEG=180` / `CLEAN_S2_SOFT_REL=1` / 物理 0.3kg·μ1·指垫1; 512 env / 30M / seed 42。
发射一律走 `clean18_abl.sh` (四件套焊死 + 多卡机自动 NO_GUARD + TMPDIR 兜底)。
Denso 同时还在跑 take3 的 `Clean3_ablBase/ablCF` (GPU0/1, 今晚完), 四条并存; Denso 24 核, CPU 够。

**判读时必须带上的两条限制** (都在 §5.18):
1. **A2 在 take18 上扰动量小一个量级** (只动 132 行红→黄; take3 动 314 行) ⟹ A1−A2 接近 0
   **不能**读成"置信度分档不承重"。
2. **覆盖率在 take18 上饱和** (参考自身就 1.000, A4 线才 0.45) ⟹ 判"擦得好不好"要看行程 / 接触行占比,
   不能看覆盖。

### 5.21 Denso GPU 故障 → 五条全迁 msc (2026-09-10 ~22:15 UTC 起)

**事故**: §5.20 在 Denso GPU2/3 发 take18 A1/A2 后约 2 分钟, 全机 GPU 掉线 ——
`nvidia-smi` 连枚举都失败 (`Unable to determine the device handle for GPU0000:19:00.0: Unknown Error`),
**四条训练全部挂死** (日志时间戳齐停在 22:14:57~22:15:26, 进程 CPU 126~138% 卡在 GPU 调用上, dmesg 无权限读)。
时间上高度可疑: 同机四个 Isaac 并存是这台机没跑过的负载, 我发车前只算了 CPU 与显存, **没考虑驱动侧压力**。
损失: 两条 take3 消融 (10h) 停在 18.86M / 17.50M, ckpt 落在 21:59, 只丢最后 15 分钟。

**用户裁定**: 四条 + UCBY 上的 take18 Base **全迁 msc 8 卡机**, 统一引擎。

**msc 必须用 `yanghong` 账号, 不是 `msc` 账号** (关键):
| | isaaclab 树 | digest | torch |
|---|---|---|---|
| 本机 / Denso | MagicSim/Third_Party/IsaacLab | `56016b8f…` | 2.7.0+**cu128** |
| **msc yanghong** | `~/MagicSim_IsaacLab` | **`56016b8f…`** ✅ | 2.7.0+**cu128** ✅ |
| msc msc | `~/IsaacLab_DexAssemble` | `3a86c106…` | 2.7.0+cu126 |
| UCBY yanghong | `~/IsaacLab_DexAssemble` | `3a86c106…` | 2.7.0+cu128 |
⟹ msc(yanghong) 与本机/Denso **同一棵 IsaacLab**, take3 两条跨机续跑不换世界。
从 UCBY 可以 `ssh yanghong@169.229.192.185` 跳进 msc (`ProxyJump UCBY` 直接可用)。

**"msc 是另一个世界 (sag 差量级)" 这条旧结论本次不成立**: 同为 512env 的 take18 Base,
msc 与 UCBY 的 `sag(deg)` 11 项里 9 项逐位相同, 2 项差 **0.01°** (沉降实测的舍入噪声)。
旧结论多半就是当时没做环境对齐。判据几何 / 物理覆写 / 档位三项也全同。

**⚠ rsync 的根因 (今天漏文件的总账)**: `rsync -a tasks/Clean DEST/` 取 **basename**, 复制成
`DEST/Clean/` —— 丢了 `tasks/` 前缀。于是"整树同步"实际把三棵树塞到了仓根, 真正的 `tasks/Clean/`
只被 `--files-from` 那几次 (该写法保留路径) 更新过。`ppo_clean.yaml` / `README.md` / 两条母带的
"漏"全是这么来的, **不是 rsync 在过滤**。正确写法: 源与目标都带尾斜杠, 一对一对同步, 事后用
全树 sha256 核。三台机的错位目录 (`./Clean` `./pregrasp` `./clean_tableware`) 已清。

**部署核验一共抓到 5 个坑 (都在发车前挡住)**
| # | 坑 | 若不查会怎样 |
|---|---|---|
| 1 | 清单漏 `world_fused.npz` (相机锚定 c2w) | 起步 13s 断言崩 |
| 2 | UCBY `/tmp/isaaclab/logs` 属别的用户 | 起步 6s PermissionError |
| 3 | UCBY/msc(msc账号) 的 IsaacLab 是另一棵树 (差 5 个文件) | 潜在"另一个世界"; 差异全在 manager-based mdp 与 urdf 转换器, 我们走 DirectRLEnv 零引用 |
| 4 | 清单漏 `ppo_clean.yaml` (只收了 .py/.sh) | **512env 场景全建完、最后一步崩** |
| 5 | `kinematics.py` 的 URDF 路径写死本机 MagicSim 副本 | msc 缺仓内 `datasets/vega_urdf/` 兜底 → FileNotFoundError |
⟹ 清单法 (按后缀猜) 不可靠, 改用**全树逐文件 sha256** (`tree_hash.sh`, 覆盖 rl_rebuild / tasks/Clean /
tasks/pregrasp / datasets/clean_tableware / datasets/vega_urdf / assets)。换机器先对 digest。

**最终布局 (msc, yanghong, 512env/seed42)**
| 卡 | run | 类型 | 目标 |
|---|---|---|---|
| GPU0 | `Clean18_ablBase_s42` (HAND_REF=1) | 新 | 30M |
| GPU0 | `Clean3_ablBase_s42_r2` (HAND_REF=1) | 续 18.86M→ | +11,142,016 |
| GPU2 | `Clean18_ablA1_s42` | 新 | 30M |
| GPU2 | `Clean3_ablCF_s42_r2` (CONF_FLAT=1) | 续 17.50M→ | +12,501,888 |
| GPU3 | `Clean18_ablA2_s42` (CONF_FLAT=1) | 新 | 30M |
共卡四条各 ~310~350 FPS, 独占的 A2 617 FPS。GPU1/4 是 root 容器的 OCIR, GPU5 有人新起的 vLLM(42.5G), 都没动。

**续跑的三个要点** (`clean3_resume.sh` 已固化):
1. `restore_train` **不恢复步数计数器** ⟹ `--max_agent_steps` 必须给**剩余量**, 否则再跑满 30M;
2. 两条的课程早已退火到底 (release_row=10, sr/cert≈0.99), 新加 `CLEAN_RELEASE_START`
   (默认仍 `RELEASE_MAX`, 不传行为不变) 避免被打回 50 重退一遍;
3. 用新名 `_r2`, 不覆盖原目录的 TB 与 ckpt。
核验: 两条 r2 的 world.json 与原 run **9 项全同** (clip/seed/recipe/t_ep/t_ref/anneal_key/
reference.sha/priors.sha/physics), stage1+stage2 参数零差异, 只有 `release_row_start 50→10`。

**★ take3 中间结果 (从被打断的 TB 读出, 未到 30M)**: `Clean3_ablBase_s42` (人手指姿指引) 到 **18.86M
`sr/success` 仍是 0.0000**; `Clean3_ablCF_s42` (拍平置信度) **17.50M 已 0.793**。
方向上印证 §5.15 预注册的那条: 人手指姿与我们的物体反推腕参考**互相打架**。等 30M 定论。

**遗留**: Denso GPU 仍故障 (需 sudo `nvidia-smi -r` 或重启, 本会话无权限); 其上四条残留进程可直接放弃。

### 5.22 take3 消融 Base/A2 终判 (2026-09-11, 两条 r2 续跑跑完后确定性评测)

`--num_envs 16 --release_row 10 --jitter 0` (课程终点档/无抖动/确定性), 母带与消融旗**按各自 world.json 复刻**。

| | **Base** (`CLEAN_S2_HAND_REF=1`) | **A2/CF** (`CLEAN_S2_CONF_FLAT=1`) |
|---|---|---|
| success | **0.000** | **1.000 (16/16)** |
| cert | 1.000 | 1.000 |
| clock_done / clock_frac | 0.000 / **0.556** | 1.000 / 1.000 |
| die | **16/16 全 `rel`** | 0/16 |
| relp 盘/海绵 | 1.88 / 1.96cm | 1.38 / 1.11cm |
| **relrot 盘/海绵** | 12.7° / **22.0°** | 8.5° / **9.1°** |
| coverage / travel | 0.685 / 77.5cm | 0.560 / 105.4cm |

**结论**: `HAND_REF=1` 把**海绵转角推过死线**。它的人手指姿增量中位 5.59°/最大 42.71°, 直接体现在手内
转角上 —— 海绵转角中位 22.0° 而死线 20°, 于是 16/16 每条都在时钟走到一半 (55.6%) 被判死。
A2 的海绵转角只有 9.1°, 不到死线一半。注意 A2 **擦得更不全** (覆盖 0.560 vs 0.685) 但全程没掉;
Base 擦得更狠却每次把海绵转丢。与 §5.11 "交叉与海绵转角是共同瓶颈" 对得上。

**边界**: 单 seed (42)。Clean 的 seed 方差是大的 (见 memory `improvebase-milestone-gh51`), 
"HAND_REF 有害" 要第二个 seed 才坐实。A1 臂未重跑 (= 现役冠军 `Clean3_taskS_s42`, 旧档 16/16)。
⟹ 三臂现状: Base 0/16 · A1 16/16(旧档) · A2 16/16, **Base 是唯一失败的那条**。

### 5.23 跨机发评测的四条坑 (2026-09-11 一次性全踩, 全是"训练脚本里有、发评测时漏配")

发训练用的是 `clean3_resume.sh` 等脚本, 里面早就写好了这些兜底; 我手写 ssh 发评测时逐条漏掉:

| 坑 | 症状 | 兜底 |
|---|---|---|
| Omniverse EULA | `Do you accept the EULA? (Yes/No): Unable to bootstrap inner kit kernel: EOF` —— `nohup` 无 stdin | `OMNI_KIT_ACCEPT_EULA=YES` (+ `< /dev/null`) |
| `/tmp/isaaclab/logs` | `PermissionError [Errno 13]` —— 共享机上该目录属于**别的用户** (msc 上是 `luhr`) | 写不进去才改 `TMPDIR=$HOME/tmp` 并预建 `$TMPDIR/isaaclab/logs` |
| **母带用错** | Base 臂 `AssertionError: CLEAN_S2_HAND_REF=1 需要母带含 human_right_f/human_left_f` | **一律读 `logs/<run>/world.json` 的 `reference.path` + 核 sha**。两条 r2 用的都是 `clean3_reference_v1h.npz` 而非默认的 `v1` |
| `pkill -f` 自杀 | 远端 `pkill -f "eval_clean.py.*_r2_eval16"` 把**它自己所在那条 ssh 命令**也匹配了 (命令行含该串) → 整段没执行, 一度误判成超时 | CLAUDE.md §6b 第④种变体。按 `comm` 过滤取 PID (`awk '$2 ~ /^python/'`), 或把发射写成脚本文件 scp 过去执行 |

⚠ **第三条最危险**: 它不报错、只给错数。第一次发的那条 CF 评测就是用默认母带跑的, 结果无效;
是因为 Base 崩了才暴露。**跨机评测的铁则: 先读 world.json 复刻母带+旗+物理, 再发。**

### 5.24 Clean 训练完整推 GitHub (2026-09-14 02:07~02:13 PDT, 用户: "和 Pour 一样, 让同事能直接看到训练细节, 尤其是一开始退火的学习抓稳盘子和洗碗布的阶段")

分支 **`clean_dp_release_20260914`** (origin = stzabl-png/RL-Correction, 基于 `clean_dp_release_20260913` 代码分支 43d57475), 两次提交 e524be69 + 792fd0d3:

| 内容 | 路径 (分支内) | 说明 |
|---|---|---|
| 数据集 | `datasets/clean_tableware/{3,8,18}` | 202 文件, LFS ~500MB (obj/npz/png/usd) |
| Stage-1 抓稳段 (退火课程) 完整 run | `exports/clean_stage1_hold_20260914/Clean3_hold_s42/` | `stage1_tb/` (release_row 50→10 退火轨迹, sr/success, term/drop_*, hold/dev_*), `train.log`, `launch.log`, **31 个 ckpt 0~19M**, 6 段视频, `world.json`, run README。设计与逐日读数 = 本台账 §4.x (L172~219) |
| Stage-2 九条 (Base/A1/A2 × take3/8/18) | `exports/clean_3x3_ckpts_20260914/<run>/` | `world.json`, `stage1_tb/`, `eval/` (eval10 seed2026 + ev16), `videos/` (3 段), `stage1_nn/last.pth` + `*step_0020M*.pth` (有 20M 节点的 5 条: take18 ×3, take8 Base, take3 taskS; take3 Base/A2 是 r2 续跑无 20M 节点 §5.22; take8 A1/A2 历史只留 last, 见下) |
| 交接文档 | `docs/DP_CLEAN_ROLLOUT_HANDOFF.md` | = 09-13 交接 README + "训练细节在哪看" 附录 |
| 台账 | `tasks/Clean/3/A_Design/DECISIONS.md` | 本文件 |

**坑 (一次)**: 仓库 `.gitignore` 有 `*.pth` / `*.mp4`, 第一次 `git add -A <目录>` 静默丢掉全部 ckpt 与视频 (push 成功、远端核验才发现 pth=0), 第二次 `git add -f` 补齐 (78 个 LFS 对象 160MB)。`.gitattributes` 已补 `*.pth *.mp4` 走 LFS。**以后发布分支的入库核验 = 远端 `git ls-tree` 数 pth/mp4, 不看本地 du**。
对照: Pour 发布分支 `pour_dp_release_20260908` 本身只有代码+数据集+评测日志, ckpt 是走 msc 目录交付的; 这次 Clean 分支比 Pour 多带了 ckpt/TB/视频, 是按用户"完整 + 看训练细节"的要求加的。
