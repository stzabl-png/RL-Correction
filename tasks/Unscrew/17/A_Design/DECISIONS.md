# Unscrew/17 (拔盖变体) —— 规划与台账 (2026-09-01 起稿; B1/B2 已裁定, 待 Phase A 开工许可)

> 实例 = **同事 `unscrew_v5` 的 `tasks/Unscrew/part4/` 代码 + clip 17 + "不拧只拔"变体**。
> 本文件是本实例的全史权威 (照 `tasks/Pour/17/A_Design/DECISIONS.md` 规矩: 拍板/标定/尸检/判据改动逐条落账, 推翻不删旧账)。
> 数据核验结论见记忆 `screw17-data-status`; 同事全史见 `tasks/Unscrew/part4/A_Design/DECISIONS.md` (合并后可读)。

## 0. 用户裁定 (2026-09-01)

| # | 裁定 | 落地含义 |
|---|---|---|
| U1 | **按演示放倒拧** | 左手抓立瓶 → 放倒至近水平 (数据: f24-35 倾到 ~100°, 瓶轴离桌 ~5.5cm ≈ 半径+手厚) → 右手侧向取盖。母带交互段照重建瓶轨迹; 皮筋 3/5/8cm 允许搁桌 |
| U2 | **不拧, 直接拔出**; 接近段生成过程与 pour17 一致 | 脱扣物理改为**轴向拔出**(§2); 机器段走 `tasks/Pour/17/.../build_motion.py` 流程 (cuRobo 站姿→PreGrasp + 成形段 6 级梯→GraspPose), 不用同事的"两腿 cspace" |
| U3 | 盖拿下即可, 手里/桌上看数据 | **数据说放桌上**: f92 右手放盖到桌、松手撤回; f100 盖在桌上, 左手仍持瓶立回 (投影核验 conf 视频 f92/f100)。⟹ 终态 = 盖放桌 (同事 placed+护送三件原样), "拿在手里≥10 步" 为 G3 后中间态 (护送②) |
| U4 | 盖用 kailang 质量 3g, 摩擦 **μ5** (2026-09-01 修订: 原裁定 μ0.4 → 5); 瓶/指垫套 G-A (0.1kg/μ5/μ5) | `apply_phys_rule` 基类只覆写主体物(瓶), 不碰 aux; 盖摩擦唯一生效处 = `clips._screw_from_layout._part` → `screw_assembly.setup_scene` 的 ScrewAux 材质 (average; 指垫 multiply 优先级更高 ⟹ 手↔盖合成 25)。**已改** 0.4→5.0, 质量 0.003 不变。B4 顺带落定: env 桌高 = `correction_env_cfg.table_top_z=0.87` (scene_layout 默认 0.85 只是 CLI 默认, 入库时传 0.87) |
| U5 | 借鉴 `unscrew_v5` | §1 |
| U6 (2026-09-01 B2) | **cuRobo 只给左手**; 左手抓稳后训练进入交互段; 左手把瓶转到右手附近, 右手再接近 GraspPose 取盖; **除 GraspPose 外, 一切参考由人手轨迹或物体轨迹定** | §4 重写; 右手不走 cuRobo 接近; 数据实证: f25→f35 盖走 16.4cm、右 knuckle 0.1cm、右腕朝向 0~1° ⟹ 是左手把盖送到右手 |
| U7 (2026-09-01 B1) | §2 拔出物理 "看着还行" ⟹ **通过**, 无需再定 | `F_pull`/dwell 是探针标定量不是拍板; 旋转脱扣关闭; 脱离后盖自由 (3g) |

## 1. 调研结论: `origin/unscrew_v5` 能借什么 (2026-09-01)

**分支关系**: 与我们在 `49a08125` (L5-30) 分叉; 他们 22 个提交, 我们 9 个; **两边改动文件零重叠**
(含我们未提交的工作树) ⟹ 合并干净。他们改的共享文件: `clips.py`(+`_unscrew_take`)、
`tasks/pregrasp/screw_assembly.py`(+242, 真实螺纹副)、`screw_joint.py`(ScrewSpec 扩字段)、
`tasks/pregrasp/env.py`(+`bypass_lift_scaffold`, 默认关)、`curobo_plan_worker.py`(可移植性)、
`wandb_writer.py`、`recon_kailang/*`。**改了共享代码 ⟹ 合并后必跑 pour 对照冒烟** (横幅 diff)。

**他们的数据 `datasets/unscrew_bottle/17`** = 同一条 take, 但 `replay_world.npz` 是**修补前**的
(1764505 B; 修补后 1781426 B, `obj_verts_local` 已换面积均匀采样); `scene_layout.json` 锚/onset
与我们今天跑的逐字一致 (瓶听左手 f23 / 盖听右手 f32)。入库以**我们修补后的 take** 为准。

| 件 | 原样借 | 要改/换 |
|---|---|---|
| 环境骨架 | `UnscrewEnv(GraspTaskEnv)`: 动作 58 [右臂7,左臂7,右指22,左指22] / 观测 507; 瓶=env.object(左手) 盖=env.aux; 右垫×Aux 接触传感器; 双臂互撞 D6 | 观测"螺旋块 4 维"(screw_frac/released/n_triad/drive_gain) → 拔出块 (§2) |
| clip 注册 | `clips._unscrew_take` (自包含 take, preengaged, upright 投影) | `assembly` 加拔出参数; 关旋转脱扣 |
| 物理 | 解析投影螺旋 + U40/U45 摩擦 (力矩估计三条防幻影: 只看盖侧/矢量差分/零接触真值门) | **无轴向拔出通路**(盖位姿每子步被投影覆盖, 轴向力全被抹掉) → 新增 (§2) |
| 判据 | G1 左手≥3垫 10步 +5 / G2 提升认证 +8 / placed=双物到母带末行 (瓶3cm/15°, 盖5cm/30°) hold15 + **护送三件** (过带永久失败 / 持盖≥10步 / 降幅峰值<4cm/步) / G4 撤退 +15 / 死线 D1-D7 / 三档皮筋 tmix=min(瓶,盖) / W_HCONF / fshape | G3 谓词源: 30° 拧满脱扣 → **拔出脱离锁存**; `r_screw=K·Δθ` → `r_pull`; `probe_thread` → `probe_pull`; **接 G-B 黄窗** (他们没有); D2 "瓶倒 30°" 在放倒变体下须按**母带相对倾角**判 (待核) |
| 母带 | `make_reference.py` 五条规则: 交互窗=layout onset+尾停 / 物体换基到 env 静置 / 瓶 upright 投影 / **盖行由瓶行派生到分离**(4cm) 再走自身轨迹 / **腕平移零依赖** (T0-2b) + IK 三修 (坏行冻结/限速/轻平滑重投影); `build_reference.py` v2 在 Isaac 重锚 (= pour `build_ref_v5` 同款) | 机器段换 pour `build_motion` 流程 (U2); 右腕不再做"抓握自转扫描+6°/行漂移"(不拧); 末段 f92-100 盖位姿垃圾 (z 掉到桌下 23cm) → 末行取 f92 放桌帧并投影到桌面 |
| 先验 | 格式/接法 (`make_prior` 同款) | 他们左=`Screw27_body` 镜像、右=`Screw27_cap_candidates`(左手抓法)镜像, 15 个"无一既可达又位置对", 闭环修 6cm。**换 Dexonomy 新交付**: 左 `screw17_bottle_left` (功能池 12, Isaac lift 12/12, 首选 `1_Large_Diameter__v4_7_14`), 右 `screw17_cap_right` (47, **装配态盖系**, 瓶为障碍, 示范夹角排序; 侧向取盖用 `33_Inferior_Pincer` 族) |
| 探针 | `probe_capgrasp/pinch/grasp/acceptance/rest` 工具家族; `smoke_zero` | 探针接 `world_fingerprint.restore_physics_env` (他们没接) |
| 指纹 | 他们多 `assembly.thread_model/breakaway_*` CRITICAL 键 | 我们多 `robot.pad_friction/switches.tier_*/conf_flat/...` → **双向合并** |
| 训练 | `POUR_UNLOCK=1,2,3` 必带 (T2-9 课程死锁尸检) | 沿用 |

**他们的现状** (HANDOFF 2026-09-01): clip32 链路通; 零动作回放能靠母带拧到 30° 脱扣; **无一次成功训练**
(首训 8.5M 零信号=课程死锁); 未解: t0 缝1 合拢瞬态碰倒瓶 (T2-6e)、右三指只到 2/3、桌高常量 0.87 (我们 scene_layout 0.85)。

## 2. 拔出物理 (✅ 2026-09-01 用户过目无异议 → 落 `ScrewSpec` + `screw_assembly.apply_screw`)

**模型**: 咬合期沿用解析投影 (盖钉在瓶颈); 新增 `breakaway_pull_n`:
沿螺轴的**轴向拉力估计** = 盖侧线速度相对"上一子步写回值"的增量沿轴投影 × m_eff / dt
(与他们的力矩估计同构: 只看盖侧 / 矢量差分再投影 / **零接触强制归零**), EMA + 持续 `unlock_dwell_s`
超阈 → `screw_engaged=False` (脱离锁存 = G3 谓词源)。**拔出方向 = 瓶轴** (盖行由瓶行派生): 盖自身旋转通道不可用 —— 装配态盖轴与瓶轴就差 20°, 交互段 53~86° (f34 盖轴竟离水平 76°), `rotation_usable=true` 是假阳性。旋转脱扣关闭 (`breakaway_torque_nm=∞` 或
`detach_mode="pull"`), 保证**唯一脱扣通路是拔**。脱离后盖为自由刚体 (3g), D1 掉落死线生效。

**初值** (标定量, 非拍板): `F_pull = 3 N` (松盖量级; 盖重 0.03N 的 100 倍, 轻碰不掉)。可传拉力上限 ≈ μ_合成·N =
(指垫 5 × 盖 0.4 = 2) × 实测捏力 5.8~7.6N ≈ 12~15N ⟹ 3N 需真捏但不苛刻。**探针标定后定档**。

**`probe_pull` 四段** (取代 `probe_thread`, 训练前必过): ① 静置不脱 ② 零接触推/拖瓶身不脱 (幻影门)
③ 阈下拉 (1.5N) 不脱 ④ 阈上持续拉 → 脱离且盖随手走。

**假设与证伪** (DESIGN_LOOP 台账):
- **H1 拔出必须靠真捏**: `diag/pull_N` 在 `diag/cap_any≈0` 的步上 ≈0。证伪 (零接触仍有拉力) ⟹ 漏了数值通道 (U44 同款: 瓶被左手带着动, 线速度差分把瓶运动积成拉力), 停训先修。
- **H2 3N 不是天花板**: +8M 时 `pull_N` 95 分位 ≥3N。证伪 ⟹ 降到 1.5N 或加指压课程。
- **H3 放倒后拔出可达**: 母带零学习回放能把盖拔下 (对照他们"母带自己能拧开")。证伪 ⟹ 右手 GraspPose 族/接近时序错, 回 §4。

## 3. 判据与奖励 (继承 + 替换)

| Gate | 谓词 (只认仿真物理量) | 奖 |
|---|---|---|
| G1 抓形成 | 左手 ≥3/5 垫 >0.5N 持续 10 步 | +5 |
| G2 提升认证 | 腕参考 +15mm, 双物 z 升 ≥5mm, 左腕-瓶滑移 <8mm | +8 |
| **G3 拔出脱离** | `screw_engaged` 由拔出锁存翻 False | +10 |
| placed (内部态) | 盖到母带末行 (放桌) 5cm/30° + 瓶末行 3cm/15° hold15, **护送三件全过** | 0 |
| G4 撤退=Success | 双臂贴站姿逐关节 <10° hold15, 物体不被碰歪 | +15 |

稠密项: `adv` / `leash` 三档 / `fshape`(右指 × W_HAND × W_HCONF) / regrip/slope/pen 原样;
`r_screw` → **`r_pull = K·max(0, Δ轴向分离距)`** 直到脱离, 总额 ≈9 与 G3 同量级 (K 随 SEP 距推导, 不手调);
**G-B 黄窗**: 交互行 `[0, r_lift+3]` 档位上限黄 (按母带算)。
死线 D1-D7 原样; **D2 待核**: 放倒变体里瓶本来就横, "瓶倒 30°" 必须相对母带倾角。

## 4. 母带 / 接近段 (按 U6 重写, 2026-09-01)

**数据实证** (支持 U6 的结构): f25→f35 盖原点走 **16.4cm**, 右手 knuckle 质心走 **0.1cm**, 右腕朝向 f0~f35 变化 **0~1°**;
右手指尖质心→盖距 f30 8.7cm → f33 **4.6cm** (接触, contact_auto 右 f34) → f40 12.6cm (拔离)。
⟹ 演示里右手**原地等**, 是左手把盖送过来; 右手只在接触后才动 (腕 f45 起转 13~33°, 拔+放桌)。

| 段 | 谁给参考 | 来源 |
|---|---|---|
| ① 机器段 | cuRobo (pour `build_motion` 流程, 双臂联合规划) | 左臂: 站姿→PreGrasp(瓶 GraspPose 梯顶); 右臂: 站姿→**等待位**(静态目标, 见下), 只是"停车", 不是接近盖 |
| ② 缝1 | 成形段 | 左手 PreGrasp→GraspPose 梯 (ArmIK 逐级) 合拢 → G1 抓稳 |
| ③ 交互段 (RL) 左臂 | **瓶轨迹**反解 IK (手↔瓶 GraspPose 刚性) | 放倒到右手附近; 瓶 conf 在 f30~60 掉到 22~44 → 黄/红档皮筋放松 + 左腕四元数流(活, 与瓶同转 ~60°)做形状指引 (HYB) |
| ③ 右臂, 接触前 (行 < f34) | **人手**: 等待位 = 右 knuckle 质心 f0~f25 静止位换基 + 右腕四元数 f0~f30 | 人手数据直接给 (静腕字段死, 但 knuckle 代理与腕朝向是活的) |
| ③ 右臂, 接触起 (行 ≥ f34) | **盖轨迹 × GraspPose**: PreGrasp→GraspPose 梯锚在盖行上 (盖行由瓶行派生), 切换时刻 = 人手接触起点 f34, 梯进度 = 人手指尖→盖距离曲线 (8.7→4.6cm) | 只有 GraspPose 是先验, 其余按数据 |
| ③ 拔出 | 沿**瓶轴**分离 (盖相对瓶 z 0.19→0.21 再离轴; 盖自身旋转不可用) | 物体轨迹 |
| ③ 放盖 | 盖 rigid ride 右手到 f92 放桌帧 (投影到桌面; f92~100 盖位姿垃圾不用) | 物体轨迹 + 人手接触终点 f92 |
| ④ 撤退 | cuRobo (物体钉终位当障碍) | 同 pour |

**唯一合成的东西** (如实记): 等待位→PreGrasp(盖行) 之间约 6~10 行的插值 (ArmIK, 20°/行限速) —— 因为右腕平移字段是死的, 这几厘米的路径不在数据里; 端点与时刻都由数据定。
**v2**: `build_reference.py` (Isaac 重锚) → `probe_acceptance`。
**默认 (非拍板, 有异议再说)**: 右臂等待位由与左臂同一个 cuRobo 联合规划送到; 若希望右臂机器段原地不动、等待位由 RL 自己走到, 改一个开关即可, 但残差界下 20cm 级的位移会拖慢 G1 后的进度。

## 5. 执行顺序 (需要你 / 不需要你)

**Phase A — 不需要你** (本机 GPU 现空, 4080S 16GB):
- A1 新分支 (off `Step4_RL_Correction`) merge `origin/unscrew_v5`; LFS 只拉 clip 17; **pour 对照冒烟** (共享代码变了)。
- A2 数据入库: `datasets/unscrew_bottle/17` 换修补后 take; 重跑 scene_layout / keyframes; PROVENANCE 记三条坑 (静腕变体 / 桌高 1.175 假象 / 盖末段位姿垃圾)。
- A3 先验: `make_prior.py` (系统 python3) 两侧 → prior npz; RL 侧可达性筛。
- A4 拔出物理 (§2) + `probe_pull` 四段 + 体制自检⑦扩。
- A5 判据 (§3): progress/progress_batch 双版 + 自检五件全绿 + G-B 接入 + 指纹双向合并 + D2 相对化。
- A6 母带 (§4): build_motion 机器段(左 PreGrasp/右等待位) → make_reference (右手两段参考) → build_reference v2 → probe_acceptance。
- A7 `smoke_zero` (机器段死线 0 铁则) + 零学习回放能否拔下盖 (H3)。

**Phase B — 需要你**:
- ~~B1 审 §2~~ ✅ 通过 (U7)。
- ~~B2 右手接近时序~~ ✅ 裁定 (U6): cuRobo 只给左手, 右手参考全由数据定。
- B3 发射批准: 变体 (HYB/OBJ) × seed 数 × 机器 (本机 1024env 显存待确认 / Denso)。
- (B4 桌高 0.85/0.87: 我核实 `cfg.table_top_z` 后报, 只在冲突时找你。)

## 6. 判读针预登记 (开跑前写死)
`sr/gate1` +2M 破 20% (他们 H6.7) · `diag/pull_N` 零接触步 ≈0 (H1) · `ep_rew/pull` >0 ·
`diag/carry_steps` 释放后 ≥10 · `sr_t0/*` 无偏口径 · 成功率只认 `eval_task.py` 确定性评测。

## 7. 风险
合并牵动 GraspTaskEnv 螺旋钩子 (仅 `secondary` clip 激活) 与 `bypass_lift_scaffold` (默认关) → 对照冒烟兜底;
拔出力估计的幻影通道 (瓶被左手带动); 放倒后 D2; 右手接近时序; 他们未解的 t0 缝1 碰瓶。

## 8. GraspPose 选型 (2026-09-01, 离线 IK 筛; 脚本与结果在 `A_Design/L1_Data/GraspPose/`)

**方法**: 纯 numpy `ArmIK` (URDF, 默认躯干/站姿, 名义基座; 结论当排序键, Isaac 冒烟再定死线)。场景 = 相机锚定换基
(瓶底 env (-0.183,-0.016), 桌 0.87); 放倒态取 f40 (瓶轴指 -y = 机器人右侧, 盖心 env (-0.176,-0.142,0.917) 离桌 4.7cm)。
左: 立瓶 yaw 扫描 + 放倒轨迹 f23→f45 逐行可达 (手↔瓶刚性, 滚转 Δ∈{0,±32,±60})。右: 盖系候选 → f40 世界腕位姿, 瓶滚转 psi40 扫描,
抓 + PreGrasp 可达 + 腕离桌 ≥3cm; 决赛 40 随机种子取**限位余量最大**的解。

**结果**:
- 左 12 个候选立瓶都有 ~半圈可达带, 都有一个 yaw 落在人手接近方位 (144°) ±4° 内; 放倒全程可达组合 32~45 组/候选 —— 左手不是约束。
- 右 47 个候选每个只在瓶滚转的 30~90° 窗口里可用 ⟹ 右选型必须联合左的 yaw 与放倒滚转 (= "要不要转物体" 的实质)。
- 单种子 IK 右臂解全部把 `R_arm_j2` 顶在 +26° 限位; 40 种子后发现内部解 (余量 15~40°) —— 是求解器分支问题不是工作空间问题。
  ⚠ 母带 IK 逐行热启动会掉进这种分支 (同事 T2-6 病一同源), A6 造带时右臂 IK 要多种子 + 余量择优。

**选定 (提案, 待用户点头)**:
| 手 | 候选 | 依据 |
|---|---|---|
| 左×瓶 | `1_Large_Diameter__v4_7_14` @ yaw 280 (腕方位 140° = 人手 144°) | demo 20.3° (池内最佳), Isaac lift ✓; 立瓶 IK 0.01cm 余量 24°, PreGrasp(退 4.3cm) 可达; 放倒后 (人手滚转 +32°) IK 0.3cm 余量 38°, 腕离桌 9cm |
| 右×盖 | `33_Inferior_Pincer__1_47` @ psi40≈312 (= yaw280 + 人手滚转 32°) | demo 14.3°, in_region 1.0, 不碰瓶; 从机器人近侧水平捏盖沿 (腕离桌 7.8cm, 盖→腕 [-0.14,-0.02,+0.03]), PreGrasp 沿瓶轴外退 1.6cm; 多种子最优解余量 25°(j7); 滚转容忍 282~322 (±20°); 不滚转 (psi 282) 也有 15° 余量解 |
| 备选右 | `33_Inferior_Pincer__6_44` (demo 15.8°, 余量 40° 但窗 282~312 偏窄) / `8_Prismatic_2_Finger__9_5` (demo 14.4°, 余量 39°, 但 Dexonomy 标 obst_touch=1 且腕离桌仅 3.3cm) | — |

**不需要调整物体轨迹**: 左 yaw 取人手方位, 放倒滚转取人手自己的 +32° (左腕朝向流沿瓶轴分量), 右候选在该滚转处可用且有余量。
**风险**: 右候选滚转窗 ±20° —— 左手策略放倒时的滚转偏差要落在窗内 (闭环右手管位置不管这个); 观察针 `diag/cap_roll_dev`。

## 9. Phase A 执行日志 (2026-09-01 夜, 用户裁定"开工, 本地自训, 早上看")

分支 `unscrew17_pull` (off Step4_RL_Correction): 合并 `origin/unscrew_v5` (零冲突, 合并前后 pour 零动作冒烟横幅逐行只差时间戳 ✅)。
| 步 | 结果 |
|---|---|
| A2 入库 | `datasets/unscrew_bottle/17` 换成修补后 take (replay/ref_qpos/conf/manifest 均更新; world_fused 同); scene_layout 按 env 桌高 0.87 重算, 瓶锚与同事逐字同; 注册 `unscrew17_task` (盖 0.003/μ5, preengaged) |
| A3 先验 | `tasks/pregrasp/priors/Screw17_bottle_left.npz` / `Screw17_cap_right.npz` (+ `Screw17_cap_candidates/`), 转换器 `A_Design/L1_Data/GraspPose/make_prior_screw17.py` **含 com_offset 平移** (通用 make_prior 假设无平移, 这批不成立); 腕位与离线筛选逐位对拍 ✅ |
| A4 拔出物理 | `ScrewSpec.detach_mode/breakaway_pull_n/mass_eff_kg`; `screw_assembly`: pull 模式旋转永不解锁, 轴向拉力估计 (盖侧线速度增量·轴, 接触门, EMA, dwell) → 脱扣; 咬合期盖质量 m_eff, 脱扣还原; 反作用力回瓶。开关 `UNSCREW_DETACH=pull` (默认 twist = 同事口径不变)。探针 `B_SmokeTest/probe_pull.py` 五段 (静置/零接触门/阈下/阈上脱扣+质量还原/向内推) |
| A5 判据 | `task_env`: r_pull (K=9.4, φ=拉力/阈值 双向差分, 脱扣 φ≡1), 观测拔出块, diag `pull_N/pull_detach/pull_phantom_N/cl_on`; G-B 黄窗接入 progress+progress_batch (r_lift 按瓶行); 指纹加 `assembly.detach_mode/breakaway_pull_n/mass_eff_kg`, `robot.pad_friction`, `switches.tier_*`; 自检五件全绿 (variants ⑤ 测试前提修: 本 clip 母带末行盖离桌 0.6cm 在放下带内, 自由落体起点改带顶+5cm) |
| A6 母带 | `make_reference`: 原生先验不镜像/不补原点; **瓶滚转规范化** (回转体滚转不可观 → 常量局部滚转让接触行 盖→腕 落到 §8 方向; 实测未规范前右腕在盖**下方**离桌 2cm, 规范后 [-0.137,-0.014,+0.036] 腕离桌 12.4cm); 左抓方位按**世界方位角** 140±20° 择优 (yaw*=345 于滚转 290 系); 右手接触前等待位 (行 0~10 静止于接触行盖上抓姿退 4cm); 右手净空梯抬升 10~20cm (站位腕离桌仅 3cm, cuRobo 保守碰撞球判桌碰 — 首轮 6 档全灭的根因) |
| U9 闭环 | `task_env._right_cl_servo`: `UNSCREW_RIGHT_CL=1` 时接触触发 (行≥接触行−3 且 实际盖离母带接触行盖位 ≤8.7cm) 后右臂前馈 = 朝 实际盖×盖系 GraspPose 梯 的一步阻尼最小二乘 (PhysX 雅可比, 步长 2cm/10°, 20°/步), 脱扣后退回母带行 |
| env 实测 | 瓶静置 env (-0.124, 0.089, 0.87); 交互窗 f23..f100 (78 行); 盖脱离行 17 (f40); 右接触行 11 (f34) |

**偏离台账原计划 (如实记)**: 机器段用同事 `plan_machine_segs` (两腿 cspace) 而非 pour `build_motion` 成形段 —— 时间所限先用已接线的; U8 的"接触前跟人手全通道"简化为"接触行盖上抓姿静止等待" (人手 knuckle 代理只动 3cm, 等价于静止; 未做人手系换基)。

### 9.1 发射前把关与首发 (2026-09-01 04:00~06:15)
| 关 | 结果 / 修的坑 |
|---|---|
| cuRobo 机器段 | Approach 第 2 档净空通 (左径向 12cm/抬 6cm, 右抬 12cm), Retreat 通; "第二腿"(贴物腿) 两次都判碰 → 缝1 笛卡尔进刀桥接 (9 行进刀 + 10 行合拢)。**坑**: 右站位解单种子落在贴限分支 (余量 0°), 同分支抬不起净空点 (5~7cm 解不出) → 站位解改 48 种子按限位余量择优 (0→7.4°), 净空点同分支热启; 机器段终点距站位 R22°/L26° (原 152°) |
| smoke_zero | 静置对账 瓶 0.20cm / 盖 0.33cm ✅; **机器段死线 0 (铁则)** ✅; 缝1 零动作合拢碰倒瓶 1 次 (记账项) |
| probe_pull | 首跑 4/9 FAIL: 静置就读出 -1.96N = **重力幻影** (盖每子步被写回瓶速后重力仍给 g·dt 增量; 力矩估计无此项). 减去重力沿轴分量后 **9/9 PASS** (阈下 1.50N 估 1.50N; 阈上脱扣; 质量 0.2→0.003 还原; 向内推不脱扣) |
| v2 重铸 | 交互 IK pos>1cm 1/156, 关键窗坏行 0, 右臂限位余量中位 3.6° (紧), 左 24° |
| probe_acceptance | 稳定性 4/4 ✅; **可训练性告警: 站位垫 L0/R0** (零动作左手没抓上瓶, 瓶被推倒滑 18~34cm) —— 同事 T2-4 口径"reference 是 correction 先验不是零动作答案", 不阻塞, 但这是首发最大的风险 (G1 全靠 RL 残差把手合上) |
| 世界指纹 | 首发被拦: 验收记盖质量 0.2 (咬合态 m_eff), 训练开场读 0.003 → 指纹改记实物质量 (`_cap_mass_orig`), 重出凭据 |
| **首发** | `U17_pull_cl_s51` 06:13 本地 512 env, HYB, `UNSCREW_DETACH=pull UNSCREW_RIGHT_CL=1 POUR_UNLOCK=1,2,3`, FPS ~2500; `logs/U17_pull_cl_s51/` |

**左站位抓握几何 (probe_grasp, 瓶钉住)**: 指垫瓶系径向 index 2.8 / middle 3.6 / thumb 3.5 / ring 4.2 / pinky 4.6cm (瓶半径 3.25) → index 已穿入、pinky 差 1.3cm: 手绕掌法线有 ~7° 偏转, 不是整体径向差 (径向收紧 ±6mm 扫描都 L0/5)。离线 FK (prior grasp 角) 五垫径向都 4.3~4.7cm (贴面), 说明是 **squeeze 剂量不均 + 接触后偏转**, 下一轮试 βL=0.5 或按垫分配 squeeze。

### 9.2 左站位抓握沙盒扫描 (2026-09-01 06:30~07:25, 与训练并行, `B_SmokeTest/trim_sweep.sh` / `beta_sweep.sh`)
- 径向收紧 ±6/+12mm (沙盒 v1, 离线锚): **不可比** —— 沙盒 v1 比正式 v2 整体深 1.3cm, 五垫全穿入 0 力; 只证明径向平移不是杠杆。
- βL squeeze 剂量 (正式 v2, 瓶钉住): 0.5 → L1 (仅小指 1.7N); **0.7 → L3** (拇 2.4 / 无名 1.2 / 小指 3.9N); 1.0 → L3 (拇 5.6 / 无名 2.1 / 小指 7.9N, index 穿入 0.45cm); 1.3 → L3 (拇 9.7 / 小指 11.8N)。index/middle 在所有档都 0 力。
- 判读: 手绕掌法线偏转 ~7° (index 高且深, pinky 低且远), 合拢先由拇/小指顶到瓶, 自由瓶被推倒 → 零动作 L0。**下一轮杠杆 = 腕姿态闭环对准** (同事右手 `probe_capgrasp --fit` 的左手版: 量五垫径向 → 绕指列轴微转腕 → 重解 IK → 复测), 不是径向/剂量。βL=0.7 作为更温和的备选剂量。

### 9.3 首发尸检 + 换左候选 (2026-09-01 07:00~07:40)
- 首发 `U17_pull_cl_s51` 4M 步: G1 t0 口径 0.9 (残差能合手), **G2 认证 0/24** (`probe_certfail`: 瓶/盖 Δz ≈0, 滑移 0.5cm ok, 垫 3~4 ok, **左腕实际抬 −0.4cm vs 参考 +1.5cm**) → 时钟冻在 G2 前, 右手/拔盖全程没开始 (`cap_any=0`)。
- 根因 (`probe_seam` 零动作逐行): 缝1 进刀"先对高度"要把左腕从 0.964 降到 0.906m, **实际腕停在 0.96~0.99 下不去** (关节差 12°), v2 重铸把整段交互重锚到这个假站位 (v2 站位比 v1 高 **8.5cm**, 右手只差 0.2cm)。认证行 IK 本身正确 (+14mm), 但臂被顶住抬不起来。
- 再往下: 1_Large_Diameter 族 (选型 §8 的 v4_7_14) 腕在瓶底 3.3cm、拇指朝上, 小指侧掌缘/腕链悬在 C_MC 下 ~9cm —— **Dexonomy 是浮手**, 桌面净距滤的是手本身, 装到 DexMate 臂上就撞桌。§8 的离线 IK 筛只查腕可达, 没查"手/腕链最低点离桌" —— 选型判据缺一条。
- 换 `11_Power_Sphere__v3_4_12` (腕 10.6cm, 接触 5~12cm, demo 44°, 放倒可达组合 40): 新 v1 左站位腕离桌 10.7cm, 左 IK 100%/余量 18.8°。以 `UNSCREW_LEFT_PRIOR=Screw17_bottle_left_PS.npz` 起第二发 `U17_pull_PS_s51` (首发保留到第二发起来再停)。
- 首发产物归档: `launch_logs/run2/{Approach,Retreat}_run1.npz`, `reference_v{1,2}_run1.npz`, `acceptance_v2_run1.json`; 录像 `logs/U17_pull_cl_s51/videos/U17_pull_cl_s51_2M.mp4`。

### 9.4 第二发 `U17_pull_PS_s51` (2026-09-01 07:48 发射)
- 母带: 左 `11_Power_Sphere__v3_4_12`, v1 站位左腕离桌 10.7cm; **v2 重锚与 v1 站位差 0.4cm** (首发 8.5cm) ⟹ 缝1 进刀到位。冒烟铁则 0; probe_pull 9/9; v2 交互 IK 1/156 坏行; 验收 4/4 (零动作站位垫仍 L0~1: 抓握靠 RL 残差合上, 与首发同)。
- 首发 `U17_pull_cl_s51` 6M 步止 (G1 t0 0.9 / G2 0 / 认证 0/24, 结构性; 保留 ckpt+录像供对照)。
- 第二发 0.59M 步首读: **sr/cert_pass 0.10~0.17, sr/gate2 0.06** (首发全程 0), `cap_any`/`cl_on` 开始出现非零 (右手已零星触盖)。

### 9.5 第二发尸检 (2026-09-01 08:30~10:00): Power_Sphere 在 Isaac 里指垫不上力
- `U17_pull_PS_s51` 5M 步: **G1=0 全程** (t0 口径也 0), 认证偶发 (g1/g2 出生点带来的), 右手零星触盖。
- 站位探针 (瓶钉住, v2 母带): 五垫径向 4.2~5.4cm (贴面距 ~4.3), 力全 0 → L0。径向收紧 +10/+16mm 不改径向 (环抱抓法指尖在瓶远侧, 腕平移只横滑); βL 1.5/2.0 无效 (该先验 squeeze 层仅 0~4°); 新加 `LEFT_CURL_DEG` (站位额外捏合) 12/20/30°: 五垫径向收到 4.1~4.5 后**不再变**, 力 0~0.7N (L0~1) —— 手指被瓶挡住但**指垫传感器读不到力**, 判断是指节/指腹先于弹性垫触瓶 (垫在指尖内侧, 这种"抓球状"手型垫不朝瓶)。⟹ G1 (≥3 垫 >0.5N) 对这族抓法结构性不可达, 与训练 5M 步 G1=0 一致。
- 对照: 1_Large_Diameter (首发) 垫朝瓶, 零动作 L3 (瓶钉住)、训练 t0 G1 0.9 —— 但腕高 3.3cm 装臂撞桌。
- 试验: `Screw17_bottle_left_LDup6.npz` = LD 先验沿瓶轴整体上移 6cm (腕 9.3cm, 接触 8~18cm, 瓶身圆柱到 15cm、肩 15~17cm) —— 沙盒看指垫与腕能否到位; 成则起第三发。

### 9.6 第三发 `U17_pull_LDup6_s51` (2026-09-01 10:23 发射)
- 左先验 `Screw17_bottle_left_LDup6.npz` (= 1_Large_Diameter__v4_7_14 沿瓶轴上移 6cm): 沙盒瓶钉住零动作 **L3/5** (拇 10.8N/无名 0.9/小指 0.6), 腕实际到位 (离桌 8.8cm); v1 站位左腕离桌 9.5cm, 左 IK 100%/余量 28°; **v2 与 v1 站位差 0.4cm**; 冒烟铁则 0; probe_pull 9/9; 验收 4/4。
- 第二发 `U17_pull_PS_s51` 5M 步止 (G1=0 结构性, §9.5)。
- 运维教训 (今晨两次): ①改正在运行的 bash 脚本会让它执行错行; ②`kill` 用含脚本名的 `-f` 模式会杀到自己的 shell (CLAUDE.md 6b 变体③) — 清理一律按 `comm` 过滤; ③规划器 ✅ 后必卡 app.close, 收割器要按"产物文件 mtime 新于本轮起始"判, 否则白等 40 分钟。
