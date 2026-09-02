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

### 9.7 第三发尸检 (2026-09-01 11:00~11:45): 母带自己就把瓶推倒, 训练开场即死
- `U17_pull_LDup6_s51` 3.2M 步止: `sr/cert_pass` 0.73→0.06 逐窗衰减, `sr_t0/gate1` 0.005 平 (首发同期 0.49), 回合长 90~120 步 (首发 211~305)。
- `B_SmokeTest/probe_death.py` (钩 `_reset_idx`, 重置前抓死状态): 确定性策略 **48/48 死于 D2 瓶倒, 行 94** (缝1 合拢第 3 行, 站位前), 瓶倾 44°, 瓶被推 2cm, 左垫 0, 手最低点离桌 +1.3cm。
  **零动作 12/12 同行同因** (瓶倾 33~38°) ⟹ 母带的合拢本身就把瓶推倒; 策略只是推得更狠。视频 `logs/U17_pull_LDup6_s51/videos/U17_pull_LDup6_s51_3M.mp4` 末帧同景。
- 机制: LD 先验接触带 z 2~12cm (瓶系), 上移 6cm 后 8~18cm —— 瓶身圆柱只到 15cm, index/middle 落在肩/颈够不到 (沙盒钉住: 拇 10.8N, index 0, middle 0.4), 合拢变成拇指单边推; 0.1kg/底半径 3.25cm 的立瓶在 12cm 高处 **0.27N 就倾倒** (μ5 不滑只倒)。首发的低抓法推在 3cm 高处 (需 1N) 且指列箍住瓶底, 所以首发 t0 G1 能到 0.9。
- **把关链漏判**: 验收 `baseline.envs[].bottle_tilt_error_deg=89.6` + `station_pads_l=0` 被同事口径 "reference 是先验不是零动作答案" 放行; `smoke_zero` 的 "机器段死线 0" 与训练 env 零动作行 94 D2 死不一致 (待查口径)。**新硬门 (待接)**: 零动作到站位行瓶倾 < 15° 且不位移 > 2cm, 否则不许发射。
- 出路 = 接触带留在瓶身、腕链抬离桌面 ⟹ 把 LD 抓法**绕接触质心、绕切向轴俯仰 (腕上翘)**: 离线 IK (`A_Design/L1_Data/GraspPose/pitch_screen.py`, 立瓶 yaw 扫描, 3 种子取余量最大):
  | θ | 腕高(瓶底上) | 接触带 | 可达 yaw | 最佳余量 |
  |---|---|---|---|---|
  | 0 (原版) | 3.3cm | 2.0~11.9 | 14/36 | 36° |
  | −15 | 6.3 | 2.9~11.3 | 15/36 | 55° |
  | **−30** | **9.3** | **3.7~10.4** | **17/36** | **41°** |
  | −45 | 12.2 | 3.3~10.2 | 19/36 | 33° |
  | −60 | 14.7 | 3.1~9.9 | 17/36 | 24° |
  先验 `Screw17_bottle_left_{P30,P45,P30u2}.npz` (`A_Design/L1_Data/GraspPose/make_pitch_priors.py`); 沙盒 `B_SmokeTest/pitch_sweep.sh` (钉住量垫力/手最低点 + 自由瓶看合拢推不推倒; `probe_grasp` 补印 `手最低点-桌`)。
- **把关链改口径 (2026-09-01 11:50, 三发共同教训)**: ① `smoke_zero` 的 "缝1 交接段死线 (记账, 不入铁则)" 改为**铁则** —— 同事口径 "残差在此段对策略开放, 属 correction 的学习对象" 被三发证伪 (首发/三发 smoke 都记了 `row=94 D2瓶倒`, 我也按记账项放行了); 零动作合拢把瓶推倒 = 策略从没见过"瓶立着的站位", G1 无从学起。② `probe_acceptance` 加硬门 "站位行零动作瓶倾 < 15° 且位移 < 2cm, ≥3/4 env" (`station_bottle_ok`, 进凭据 JSON)。两条都在 `stage_chain.sh` 里阻塞发射。
- **离线"装臂最低点"筛落地** (`A_Design/L1_Data/GraspPose/lowest_point_screen.py`, 全臂+手 FK 链接原点, 未含网格半径—用沙盒标定): 12 个左候选里 §8 当初按 demo 角选的 `v4_7_14` 链最低 **−0.1cm**(人手方位处, 即撞桌), 而**没试过的三个 LD 同族** `v3_2_27`(链+6.4/指+2.7/余量22°) `v3_7_29`(+6.3/+2.7/29°) `v4_4_7`(+5.4/+2.2/25°) 干净得多 —— demo 角 39~43° 比 v4_7_14 的 20° 差, 但按 [[模板选择原则]] 接触位置合理性优先。**刚性俯仰路线实测失败**: 绕水平轴转动让上排垫外移 Δz·sinθ、下排垫内穿 (P45 只剩 ring/pinky, P30u2 只剩 pinky), 多高度包裹不能刚性倾斜, 只能重新合成 → 放弃自改抓法, 回到原生候选。三个新候选先验已产 (`Screw17_bottle_left_{LD227,LD729,LD447}.npz`), 沙盒队列中 (钉住垫力 + 接近段钉住缝1放开)。

### 9.8 LD 三候选沙盒判 (2026-09-01 11:47) + LD227 剂量扫描
- 判据三条: 钉住 ≥3 垫有力且两侧都有 / 左手最低点 >0 / 缝1 放开瓶倾 <15°。
- **左手最低点全过** (+1.7/+2.0/+2.2cm, 都是 pinky elastomer 最低 —— 离线筛的链余量兑现, 判据有效)。
- 钉住垫力: **LD227 L2/5** (index 2.1N/middle 2.4N/ring 0.3, 拇/小指 0, 五垫径向 4.1~4.7 ≈ 贴面) | LD729 L0/5 | LD447 L0/5 (径向同带, 差在毫米级)。
- 缝1 自由 (接近段钉住, `--pin_approach` 新开关): 三者同样 row64 起倾, 终态 30.1/30.1/30.8° —— 零动作合拢仍推瓶, 但比 LDup6 (33~38° 且继续倒) 轻, 瓶斜靠在指列上稳住。
- → LD227 是唯一活口: 差的是拇指/下二指的 squeeze 剂量, 不是几何。`betaLD227_sweep.sh`: βL ∈ {1.3,1.6,2.0} 钉住量五垫 (probe_grasp 自己施加 βL 缩放, 不用重铸母带), 最佳剂量跑缝1自由看 <15° 能不能达成。达不成 → 本地手段穷尽, C1 交用户 (Dexonomy 按双约束重出)。
- **收口 (12:15)**: βL 上行单调变差 (1.0→L2 / 1.3→L1 / 1.6→L1 / 2.0→L0, 过卷让垫转离表面); βL=1.6 缝1自由 终态 28.6° —— **<15° 不达成**。本地手段穷尽 (原候选×12/上移/刚性俯仰/剂量), 全部路线×三判据矩阵见战报页。**C1 交用户二选一**: (A, 推荐) Dexonomy 按双约束重出候选 —— 装臂后链最低点 ≥+1cm (筛子 `lowest_point_screen.py` 可直接当交付过滤器) 且 指垫朝瓶身 z≤15cm; (B) 留 LD227 做"腕姿态闭环对准"工程 (量五垫→毫米/度级调腕→重解 IK→复测, 同事右手 probe_capgrasp --fit 的左手版, 估半天, 收敛不保证)。四发暂停待 C1。

## 10. Pour/17 路线重建规划 (2026-09-01 下午, 用户指示"看 Pour/17 结构 + 完整路线参考搭建, 写规划"; **待审, 未开工**)

### 10.0 读完 Pour/17 结构后的三个结论
1. **判据机是同族, 不用换**。同事的 `tasks/Unscrew/part4` 本来就是 Pour/17 v5 谱系的移植 (G1/G2认证/皮筋三档/RSI/单时钟/指纹全同构)。"换到 Pour 路线"的实质是三件别的事: **参考搭建纪律 (L1-4/L2-6)、L5-36 的两味药、把关哲学**。
2. **三发死掉的病, Pour/17 得过且治好了**。Pour 台账 36.3.1 原文: "12~25° 的倾斜是**合拢推挤期** (G1 之前, 缝1+IA0 前段) 形成的" —— 与我们 row94 D2 同病。Pour 的解法不是"零动作必须不推瓶"(我 §9.7 加的 15° 硬门比 Pour 的实践更严), 而是: ①铁则只要求**放音零死线触发**; ②`POUR_HOLD_POSE=2` 从缝1合拢起到 G2 给**绝对倾角/漂移的渐进罚** (5°→60°, 5mm→5cm, K=0.05/步), 认证加硬条件 (倾角≤10°∧漂移≤1cm); ③`POUR_GRIP_SHAPE` 给认证期**跟腕比 r_follow** (结果信号, 实测比几何代理 r_opp 好爬)。三级阶梯确定性评测 0.0000 → 0.0273 → **0.5586** (gh s51, 历史 9 倍)。
3. **LD227 在 Pour 把关哲学下是可训候选**。它零动作合拢把瓶推到 28.6° (βL1.6) / 30.1° (βL1.0) —— 我的 15° 硬门枪毙它, 但 Pour 铁则 (无死线) 下 28.6°<D2 机器段 30° 过线 (余量仅 1.4°, 见 10.4 拍板④)。C1-A (Dexonomy 重出) 降级为备份线, 不再是唯一出路。

### 10.1 Pour/17 结构地图 (逐层, 供对照)
| 层 | 内容 | 关键文件 | 用户拍板点 |
|---|---|---|---|
| L1-1/2 数据 | 物轨+双置信度 / 人手轨迹+conf 口径 | `L1_Data/{Object_Traj,Human_Traj}` | conf 制度两条 |
| L1-3 GraspPose | 候选→thumbfix→**GUI 目检** | `L1_Data/GraspPose` + view_grasppose | ✅ 目检拍板 |
| L1-4 接近/撤离 | cuRobo 满障碍联合规划(站姿→Dexonomy 原生 pregrasp 梯级0, 手指 GENERIC_OPEN 几何做碰撞检查) → **PCHIP 六级梯腕指同步成形** → 合拢(close_backoff 抵消穿透伪影); **撤退=接近精确倒放** ("接近≠倒放撤退"单向性入档) | `Motion_Planning/build_motion.py`, `curobo_plan_worker` (联合/ cspace 两模式), view_motion GUI | ✅ 双向 GUI 定稿 |
| L2-6/7/8 拼装 | 编舞 Approach(165)+缝1(25)+交互窗(135)+缝2(25)+Retreat(165); 交互窗=[物体开始动,停止动] 逐物体判; 人手**首帧焊接**(借运动不借位置); 物体逐物换基+RTS+conf 三档; **人手净空钳制 v2 (FK 全连杆最低点≥桌+2cm)**; 单时钟 frame_of_row + source 标记; 进度条置信门控双参考 | `view_reference.py` (活样机+--export+--record), `build_ref_v5.py` (交互段物轨反解 IK 重铸 + **四道出厂检查**: 连续性/峰值保全/IK 精度/末态-判据一致) | ✅ 全链活样机+mp4 验收 |
| L3 学习 | 四 Gate+认证/死线全表 ("**凡参考会路过的状态绝不能是死区**")/RSI 三桶自动推导/观测/残差锚; **铁则: 参考放音零死线 + 反向验证(闸门必须能证明自己会红)**; L5-36: GRIP_SHAPE + HOLD_POSE v2 | `L3_Learning/progress*.py` + 16 个 selftest, REWARD_DOC.md | 判据逐项拍板 |
| B/C | 探针族+REPORT; env 只做接线; smoke_zero A~D; 确定性评测 t0 | `B_SmokeTest/`, `C_Wiring/` | 发射批准 |

### 10.2 Unscrew/17 重建方案 (逐段映射)
- **L1-3 (改)**: 候选筛 = 双约束 (`lowest_point_screen.py` 装臂最低点 ≥+1cm ∧ 指垫朝瓶身 z≤15cm) + 零动作放音铁则。现役候选 **LD227** (`1_Large_Diameter__v3_2_27`); 给用户出 GUI 目检包 (render + 沙盒垫力表 + 缝1 自由回放读数)。
- **L1-4 (重造, 弃同事 plan_machine_segs 两腿桥接)**: cuRobo 满障碍**联合**规划 站姿→{左=瓶 pregrasp 梯级0 (Dexonomy 原生, 半张开), 右=等待位 (盖上抓姿退 4cm)}; 左手 PCHIP 六级梯成形 + close_backoff 标定; 撤退 = placed 后 (盖已放桌/在手) 右先松盖倒放、双臂回站姿 —— 按数据 f92 盖放桌口径设计, view_motion GUI 走查。
- **L2 (重拼, 换 Pour schema)**: 编舞 Approach + 缝1(squeeze 渐入, 同现制) + 交互窗 78 行 (f23..f100, 分段器复核) + 缝2 + Retreat; 母带加 `frame_of_row`/`source`/conf 三档着色; 保留瓶滚转规范化与 U9 右手闭环; 净空钳制 v2 **扩到臂链** (手+腕链最低点, 三发根因判据入厂检); build_ref v2 重锚 + 四道出厂检查照搬; 右手接触前 = 等待位静止焊接 (静腕数据的"借运动"=零运动, 口径写死)。
- **L3 (移植两味药, 判据机不动)**: ①`UNSCREW_HOLD_POSE=2` — 缝1 合拢起至 G2, 瓶绝对倾角/xy 漂移渐进罚 (5°→60° / 5mm→5cm, K=0.05); 认证硬条件 倾角≤10°∧漂移≤1cm; ②`UNSCREW_GRIP_SHAPE` — 左手 r_follow (认证斜坡瓶升/腕升比, earn-only K=1) + r_opp (K=2, 按 Pour 教训预期贡献小但留着); 右手×盖同款 (G3 前的护送段)。指纹/digset 口径照 Pour 规矩 (旗开才变 digest)。
- **把关哲学 (改我 §9.7 的过严门)**: 铁则回到 Pour 口径 = 零动作全链放音**无死线触发** + 反向验证; 站位行瓶态从"硬门 <15°"降为**记账+判读针** (`probe/seam_tilt`)。
- **发射体制**: 2 seed (51/52), 预注册: 点火 = `sr_t0/gate1 ≥0.15 @3M ∧ ema_cert ≥0.15`; 判死 = `sr_t0/gate1 <0.03 @3M ∧ 认证瓶升中位 <1mm` ⟹ 4M 停; 判读针预登记 (缝1 倾角分布 / r_follow 账本 / pull_N / cl_on)。

### 10.3 施工顺序 (每步产物可 GUI 走查; ★=拍板点, 不拍不往下走)
| 步 | 内容 | 量 | 产物 |
|---|---|---|---|
| U-P0 | 分段器过 unscrew take 出五段帧界复核; LD227 按 Pour 铁则重判 (训练 env 零动作全链, βL∈{1.0,1.3,1.6}, 记死线/倾角) | 半天 | 帧界表 + 放音报告 + **★拍板① LD227 目检** |
| U-P1 | L1-4 重造 (cuRobo 联合 + PCHIP 六级梯 + close_backoff + 净空钳制v2含臂链) + 撤退设计 | 1 天 | Approach/Retreat npz + view_motion 走查 + **★拍板②** |
| U-P2 | L2 重拼 (Pour schema 母带 + 四道出厂检查) + 活样机 + 全链 mp4 | 半天 | reference npz + mp4 + **★拍板③** |
| U-P3 | L3 两味药移植 + 死线口径复审 (**★拍板④: 缝1 段瓶 D2 30°→45°?** 理由: LD227 参考自己 28.6°, 余量 1.4° 违背"参考路过≠死区"原则; HOLD_POSE 渐进罚接管梯度, D2pre 60° 仍兜底) + selftest/铁则/反向验证全绿 | 1 天 | 自检报告 |
| U-P4 | **★拍板⑤ 发射批准** → 2 seed 本地/Denso, 预注册判读, 守夜 | — | 第四发 |
| 备份线 | C1-A: Dexonomy 双约束重出 (与 U-P0~P2 并行不冲突, 有更好候选随时换入) | 用户侧 | — |

### 10.4 开放问题 (随拍板点一起答)
① LD227 demo 角 43° (vs 烧掉的 v4_7_14 20°) —— 按模板选择原则接触位置优先, 但请目检确认; ② 右手接触前"等待位静止"口径是否接受为正式设计 (静腕数据下"首帧焊接借运动"退化为静止, 非简化而是数据事实); ③ 拍板④ 的 45° 数字; ④ HOLD_POSE 数字取 Pour 默认 (10°/1cm, 罚窗 5°→60°) 还是按瓶重标 (0.1kg 瓶更易倒, 可收紧起罚点到 3°)。

### 10.5 U-P0/P2 首件: LD227 完整参考 + 全链活样机 (2026-09-01 下午, 用户指示"合成完整参考+Isaac 查看")
- 造带链 `C_Wiring/build_full_ref.sh` (只造不发射): v1 pre-plan → cuRobo Approach(81行)/Retreat(81行, 倒放兜底) → v1 终版 (`d54405f1`, 291 行 = app81/seam1 25/ia78/seam2 25/ret81; 左 IK 100% 余量中位 15.0°, 缝1 进刀差 R18.4°/L18.0°) → **v2 重锚 `e8099b14`** (交互 IK pos>1cm 1/156, 关键窗坏行 0, 余量中位 R3.6°/L14.3°, 认证行 IK 亚毫米)。左先验 = `Screw17_bottle_left_LD227.npz`。⚠ 右抓站位行扫描标"不可达"后由 48 种子择优救回 (0→7.4°), 与前几发同口径。
- 活样机 `B_SmokeTest/view_reference.py` + `view_ref.sh` (照 Pour L2-6 物理体制): approach/retreat = 机器人碰撞开(108 体)+物体钳静置; seam1/interact/seam2 = 碰撞关+物体逐帧钳到母带物轨行+手走 ref58 行; conf 三档轨迹曲线; 终端回车暂停/继续; `--selftest --headless` 抽帧冒烟。
- 修运维: 造带脚本给规划器配"✅ 即收割" (接近段规划曾卡 app.close 白等满 40min timeout, 教训③再现)。

### U10 裁定 (2026-09-02): 盖装在瓶上, 预留 30° 拧转即脱 —— 取代 U2 的"纯拔出"
- 用户原话: "瓶盖要放在瓶子上 预留30度的旋转就能拿下"。装配 preengaged 不变 (盖本就在瓶上, 母带 row0 盖-瓶 Δz=18.0cm ✓)。
- **机制是同事 T2-11 现成的** (commit 200d0808): `SCREW_TURNS=UNSCREW_TURNS 默认 30/360`, 经 `screw_turns_override` 进 runtime ScrewSpec (env init 断言), K_SCREW 按 30° 预算缩放, 母带 meta/世界判据/selftest_regime⑦ 全查 turns; `DETACH_MODE` 默认本就是 twist —— 是我们 U2 期的脚本用 `UNSCREW_DETACH=pull` 盖掉了它。
- 落实 = 17 侧全部脚本 `UNSCREW_DETACH=pull` → `${UNSCREW_DETACH:-twist}`; 把关链螺纹探针按模式选 (twist→probe_thread A~E 五段, pull→probe_pull); r_pull 退役、r_screw (拧转势) 复位, 拔出物理整套保留为 `UNSCREW_DETACH=pull` 变体开关。现役 LD227 母带 turns 已是 30° (T2-11 之后造的), 不用重造。

### U11 裁定 (2026-09-02, 用户看活样机反馈) + 两个查看器修复
- 用户看全链活样机: "在用右手接近物体 GraspPose, 动作一卡一卡"; 看静态目检: "瓶盖貌似在瓶子里面"。
- **U11 右臂在家** (`UNSCREW_RIGHT_HOME` 默认开, make_reference 烙进母带, 参考 md5 覆盖不加指纹开关): 机器段/缝1 右臂 = 站姿 (cuRobo 右腿退化为原地, 右净空梯/站位悬停整段消失 —— 实测原站位右腕悬停离盖 21.6cm、接近段右臂摆 71°+68°, 就是用户看到的"右手在接近"); 交互行 0..k_contact(11) 平滑飞向接触行抓姿。训练时钟 G2 前不走 ⟹ 右手起飞在左手认证抓稳之后, 正合 U6 时序。
- 盖掉进瓶里 = view_grasp bug: 钉位姿在 30 步自由静置**之后**抓, 而静置没走螺纹装配, 盖自由落体 (~0.1s ≈7cm) 掉进瓶口。修: reset 后立即抓钉、静置期也钉。世界本身没病 (check_u10 reset 即读 Δz=18.0cm ✓)。
- 卡顿 = 查看器每行 12 子步渲染完再 sleep。修: 子步间参考行线性插值 + 匀速渲染; 另加 `--record` (replicator rgb, Pour 同款)。
- build_full_ref.sh 规划段补"✅ 落盘即收割" (接近段曾卡 app.close 白等满 40min)。

### 10.6 U-P1 首件: 左臂 站姿→GraspPose 路径 (2026-09-02, 用户指示"按我之前的设计 plan 路径")
- `17/A_Design/L1_Data/Motion_Planning/build_left_approach.py` 照 Pour L1-4 用户定稿: 站姿 --cuRobo cspace 满障碍(桌+立瓶+盖, 充气1cm)--> Dexonomy 原生 pregrasp 梯级0 --PCHIP 单调样条六级梯(腕指同步, 单次缓入缓出)--> GraspPose(母带站位行)。右臂全程站姿 (U11)。梯级位姿 = 站位腕 FK ∘ (prior grasp)⁻¹ ∘ pregrasp_k 刚性相对变换 (免场景换基)。
- 产物 `LeftApproach_LD227.npz` (81+84 行): 六级 IK 全站位同分支 (距站位 ≤5°, 误差 0.01~0.46cm), cuRobo 目标候选 #0 一把过, 成形段峰值行跳 **0.9°**, 终点腕误差 0.00cm。
- `view_grasp --path` 播放模式: 循环 = 路径 → squeeze 渐入 → 保持 → 复位。自由瓶全周期实测 **倾 1.6°** (L2/5: index 6.1N/middle 3.2N/ring 0.4N) —— 比 40 步瞬移合拢 (2.6°) 更稳, 佐证 10.5 的"合拢速度分水岭": 六级梯的慢合拢天然不推瓶。
- 沿途修三坑: UnscrewEnv 无 `_anchor_T` (用 env_rest.json 的 anchor_T_left, 与 make_reference._mk_ik 同源); worker 产物列名 = joint_names/traj (照 _load_plan); 梯级顺序按退距降序排 (不可信任 npz 原序)。

### 10.7 U12 (2026-09-02 用户裁定"左手不错" + 编舞体制指示): 左接近段剪进母带
- `make_reference` 加 `UNSCREW_LEFT_APPROACH_NPZ` 剪接口: 接近段 = LeftApproach 产物 165 行 (cuRobo 满障碍 81 + 六级梯 84, **合拢在段内缓慢完成**); 缝1 从"进刀+快合拢"降级为 **25 行静持 + squeeze 渐入窗** (28.6° 推倒病灶随旧快合拢一起切除); 右臂在家语义保持。17 侧 5 个脚本默认导出。
- 母带 375 行 (app165/s1 25/ia78/s2 25/ret82), v1 `6ba797d1` / v2 `139d26dd`, 关键窗坏行 0。录像逐帧核验: 接近开物理贴面合拢立瓶 → 编舞段瓶放倒、右手抓姿骑盖、脱离放桌、扶正、撤离。
- **U-P3 待办旗**: 手指门冻结窗 `[0, APP_END)` 现在盖住了合拢段 (APP_END 81→165) —— 训练时手指残差在合拢期被冻结, 是否放开 (改成冻到 cuRobo 末行 81) 待拍板④一起议。

### 10.8 U13 (2026-09-02 用户抓包"右手轨迹到桌下"): 桌面净空钳制升级 + 采样陷阱
- 排查顺序: 右手全连杆 FK 最低点全程 ≥+9.4cm (不是手); 盖原点行 0/78 桌下 (原点钳制在); **网格最低点口径一量: 盖穿桌 19/26 采样行、最深 −1.4cm, 瓶末段 −0.4cm** —— 同事版只钳了盖**原点** z (+6mm), 盖躺倒时网格底比原点低; 用户看到的"右手轨迹扎桌下"= 右手骑着穿桌的盖轨走。
- 修 = Pour L2-6 官方口径**网格最低点钳制** (≥桌+2mm, 9 帧平滑不低于必要抬升) 进 make_reference, 瓶在盖行派生**之前**钳 (咬合段盖跟瓶一起抬, 螺旋刚性不变), 盖在自由段平滑后钳。首铸: 盖 63/78 行抬, 最大 +1.7cm。
- **采样陷阱 (U13.1)**: 钳制器/审计各自随机采 600/800 顶点 + trimesh process 开关不一, 互相漏底圈极值顶点 → "钳过了"与"还穿 −0.57cm"同时为真。改**全顶点**求最低 (78 行×全网格矩阵乘, 秒级)。审计口径同步。
- 连锁: 物体行变 → 规划摘要变 → 出处断言拦旧 cuRobo 产物 (设计内, 正常工作) → 整链重规划。

### 10.9 U14 (2026-09-02 用户裁定): Sim 物体全部上图案纹理
- 资产现状: 重建 OBJ 无 UV/无 mtl/无贴图, cache USD 无材质 —— 真贴图无从映射。
- 方案: 回转体**运行时圆柱投影生成 primvars:st** + UsdPreviewSurface×UsdUVTexture 绑程序化图案 PNG (`datasets/unscrew_bottle/17/cache/textures/{bottle,cap,table}.png`): 瓶=棋盘+四色竖带+轴向细线; 盖=四象限大色块+白指针条纹 (**拧 30° 一眼可见**); 桌=木色细网格 (平面投影)。实现 `B_SmokeTest/texture_objects.py::apply_textures(E)`, 两个查看器场景 build 后调用 (try/except 降级)。
- 已知限制: ①环向 UV 接缝一条色带 (vertex 插值 atan2 回绕), 只影响观感; ②桌若是 Cube 非 Mesh 则跳过; ③训练 env 不渲染, 不受影响; 录像器要同款上妆时 import 同一 helper 即可。
- **U15 补 (2026-09-02 用户更正: 要 SAM3D 自带纹理, 不是程序图案)**: 带纹理产物在 `egodex_auto/.../17/objects/object_*/textured/*.glb` (UV+2048² 贴图; part4 树没有这层)。几何是另一套细分且 GLTF y-up 居中 (盖高还差 3mm) → **帧对齐 + 最近邻 UV 迁移**: `extract_real_textures.py` 离线对齐 (上下符号按 NN 距离中位硬判 —— 剖面 L2 判据实测会把瓶选倒), `texture_objects.apply_textures` 运行时对 USD 顶点 KDTree 取 UV 绑真贴图 (瓶 NN 中位 5.2mm, 盖 1.5mm), 缺档退程序图案。回转体 yaw 偏置无害。
- **U16 补 (2026-09-02 用户抓包"GUI 里纹理是碎的")**: UV 迁移的病根 = SAM3D 贴图是**图集** (atlas 分块), 逐顶点最近邻把另一套网格的 UV 搬过来, 跨块三角形在图集里插值出乱码色块 —— 远景录像看不出, GUI 凑近全是碎花。**换法: 视觉网格直接挂 SAM3D 自己的 `retarget/object_*_textured.usd`** (管线现成, z-up 壳/几何 y 轴长居中/无物理 API), 作为物体 prim 的纯视觉子节点跟随刚体, 对齐 4×4 (NN 定符号同款) 存 `real_*_visual.json`; 原 CAD 网格 MakeInvisible, **碰撞保留 = 物理零改动**。UV 迁移/程序图案退为二三级 fallback。四物体 json 已产 (unscrew 瓶5.1/盖1.5, pour 瓶5.2/杯11.9mm), 抽取器 `extract_real_textures.py` 一并产出可复现。
- **U16.1 (纹理挂载三连坑, 全离线破案)**: ①pour 配对错 —— `datasets/pour17` staged 网格是 **y-up 居中**(非 z-up 落底), 固定 y→z 假设产出 rs=3.41/sz=0.30 的坏矩阵; 对齐核心改**数据驱动**: 对称轴=与三向延伸中位数偏离最大的轴 (长轴 argmax 对碟形退化, 盖抓过) + 两条硬断言 (NN<15mm/延伸比∈[0.7,1.4]) 拒绝静默错配。重产后 pour 0.3mm (auto 与 staged 实为同套几何)。②递法约定: 内存 stage 合成探针定案 **p·M 行向量, 递 M^T** (unscrew 瓶 6.1mm ✓ vs 78mm 躺倒 ✗)。③教训: 两个 bug 叠加时"经验试错"会互为烟雾弹 —— 第一次去转置的"修复"其实是错上加错; 破案靠的是离线可复现探针 + 亚毫米真值, 不是再录一遍看。
