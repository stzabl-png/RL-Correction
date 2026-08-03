# 参考条件残差修正 RL — 设计总览

> 项目：**把 noisy 视频 HOI 重建，在 IsaacSim 里 RL 修正成物理可行的抓取+操作轨迹**（reference-conditioned residual RL / DeepMimic·D-Grasp·PhysHOI·ManipTrans 一脉）。
> 本文档是收敛后的**单一权威设计**，取代此前所有分层修订。配套参考文档：
> [FRAME_ALIGNMENT.md](FRAME_ALIGNMENT.md)（四系统帧对齐）、[DEXMATE_WORKSPACE.md](DEXMATE_WORKSPACE.md)（可达区）、[POLICY_DEEP_DIVE.md](POLICY_DEEP_DIVE.md)（可复用的 SharpaWave 地基）。
> 状态标记：✅已定 / 🔶待调数字 / 💤后续阶段。

---

## 0. 项目定位：一个"数据引擎"

最终愿景 = **从人类视频学机器人 Policy**。当前瓶颈 = 视频重建物理不可用（穿模/悬空/抖动/物体滑穿手指；magicdexmate 把 video-recon 重定向到 Sharpa 手后**直接执行成功率极低**，这就是本项目动机）。本项目是桥：

```
人类视频 ─►(noisy)HOI重建 ─►【RL 修正(本项目)】─► 物理可行的手+物体轨迹 ─► 干净数据学 Video→Policy
```

收紧的三条要求：**①推理快**（策略紧凑可摊还）、**②可泛化**（几何/参考是泛化载体）、**③reward 可被 VLM 参数化**（后续元层）。

**目标是 (b) 忠实修正"那次操作轨迹"**，不是单纯生成抓取数据；**视频里的人手操作绝不砍**。

---

## 1. 核心架构：按"可信度分通道"处理参考 ✅

**关键原则**：*规划器/干净先验的价值 ∝ 1 / 参考可信度*。
ManipTrans 用 MoCap（干净）→ 贴着追、不用规划器即可。**我们用 video-recon（噪声大）→ 不能贴着追噪声**，因此需要独立于视频的干净信号（OCIR grasp、cuRobo 可行轨迹）来兜住不可信的通道。于是参考**分通道**处理：

| 通道 | 来源 | 可信度 | 角色 |
|---|---|---|---|
| **物体粗轨迹** | 视频①(FoundationPose) | 中 | **松追踪目标**——"那次操作"的核心内容(b)，λ 低 |
| **手腕粗运动/接近方向** | 视频① | 中 | 松追踪 + 接近段引导（保留人手意图 b） |
| **手指细节/精确接触/抓取位姿** | 视频①(retarget) | **低** | **不追**——交 OCIR② 给干净目标、RL 修可行性 |
| **抓取构型+接触点** | OCIR② | **高**(独立于视频) | 抓取段**可信目标**，RL 收敛到它 |
| **可行接近轨迹** | cuRobo③ | **高** | **残差锚点 + RSI 初始态 + 可行性脚手架** |

**这同时满足两条硬约束**：① 人手不砍——人手在**可信通道**（物体轨迹+粗手运动）上是实质追踪目标，保留视频那次操作(b)；② 视频太噪——**不可信通道**（手指/接触细节）交 OCIR/cuRobo 兜底，不追噪声。

**残差锚点 = cuRobo③**（干净），非原始噪声重定向、非 imitator。**imitator 预训练 MVP 不做**（ManipTrans 48%→58% 的收益是在干净目标上测的，噪声参考下未知，还多一大步；卡住再上）。

---

## 2. MDP 规格

### 2.1 动作（28-DOF 残差）✅
```
腕 6-DOF (SE(3) wrench-PD 施加到浮动 root):  wrist_tgt = ③_wrist[phase] ⊕ scale_w·a_wrist   (a_wrist: 3 pos + 3 轴角)
指 22-DOF (力矩PD):                          q_finger_tgt = ③_finger[phase] + scale_f·a_finger
```
- **残差锚在 cuRobo③**；残差初始化 ≈0 + warmup（从可行轨迹起步，只轻推）。
- **腕 = 6-DoF wrench-PD**（Option A，D-Grasp/ManipTrans 验证）：手 root 本就是自由浮动基座（USD 无 FixedJoint，只驱动不加 DOF），朝向残差用手系轴角（无万向锁）。
- **手的重力打开**；wrench 每步 clamp 到 `F_max/T_max`≈DexMate 右臂末端能力（保可迁移，见 DEXMATE_WORKSPACE.md）。
- 复用：指残差插 `sharpa_wave_env.py:236`；外部力/力矩施加已有先例(`:245-252`)。新建：腕 SE(3) PD 层。
- 🔶 待定：kp/kd、F_max/T_max、scale_w/scale_f 初值（从 URDF effort 反推）。

### 2.2 观测 ✅
原则：**喂相对量(误差/delta)**，腕误差手系+轴角。
- 本体感觉：指 q/target、**腕位姿+速度(新)**、触觉 5 指力、手系重力。
- **参考通道（按可信度）**：残差锚 ③ 在 {t, t+1, t+5} 的腕/指相对误差 + 完成度标量；追踪目标 ①物体粗轨迹 + ②抓取目标（相对当前）。**不喂视频①的手指细节**。
- **物体信号**：`goal_obj ⊖ obj`。actor 物体位姿 = MVP 先真值、后加噪（物体位姿"共享·不同精度"；纯特权量只 critic 看）。
- **几何/affordance 输入 = 单物体 MVP 不给**（几何烤进权重）；affordance 只在 OCIR 上游；Phase-3 跨物体泛化才加点云。
- 归一化交现有 RunningMeanStd。

### 2.3 特权信息 → critic ✅
穿模深度、接触法向/点（`enable_contact_pos` **只给 critic 开**或指尖 FK 算）、物体真实线/角速度、追踪误差(**对③非①**)、当前接触点 vs 目标距离、物体到目标距离。原则：sim 知道、部署拿不到。网络结构不变，只改喂 critic 的内容。

### 2.4 奖励：乘法式 imitation × 接触 ✅结构 / 🔶权重
采 **PhysHOI 乘法结构**（掉某项→整奖归零，逼每帧都对）：
```
r = r_track_hand  ·  r_track_object  ·  r_contact   ( × 物理正则因子 / − 加法惩罚, 组合方式=调参项)
```
- `r_track_hand = exp(-k_h·err)`：手/指追**可信目标**——抓取段=OCIR②/cuRobo③（**严格**，λ 高；D-Grasp 指尖权重 4×）；操作段=①粗手运动。
- `r_track_object = exp(-k_o·err)`：物体追**视频①粗轨迹**（**松**，λ 约 1/50 于手；PhysHOI 手法）——这就是"忠实还原操作"(b)。
- `r_contact`（**主导项**）：达成的**接触力 / 目标接触集**匹配（取自 OCIR②，非穿模深度/邻近）；**抓力封顶 5×物重**（D-Grasp `r_c`：`(g̃ᶜᵀ·1_{f>0})/(g̃ᶜᵀg̃ᶜ) + min(g̃ᶜᵀf, 5·m·g)`）。
- 物理正则：穿模、物体线/角速度、动作平滑、腕残差越界（工作区）。
- 全部用有界核 `exp(-k·err²)∈[0,1]`，k 编码优先级（接触/抓取紧、物体/人手松）。
- **物理松弛课程**：重力 0→真、摩擦 高→真（补偿重定向残余穿模，ManipTrans 手法）、阈值 松→紧。
- 🔶 待定：各 k、λ、乘法/加法组合、课程节奏（扩展离线单测：静止→0 / touch-and-go→衰减 / fling→掉分 / co-drift→不付）。

### 2.5 复位：RSI ✅
从 cuRobo③ 随机相位 `t` 起，手/物体状态写成③在 `t`；**物体全程动态**（非 teleport）；**速度从参考差分给**（零速度 RSI 学不会动态技能，DeepMimic 消融）；**相位受限**：早期只从接触前相位起，抓稳了才采持握相位（避免运动学锚点在空中持握相位第 0 步甩飞）。可选 **PhysHOI 接触标定**：把参考在 sim settle 一次读回无穿模版当 ① 预处理。

### 2.6 相位：按完成度推进 ✅
子目标达成才推进（抓取段：接触数≥阈值且物体稳；操作段：物体到位）。没抓稳冻结相位。Phase-1 可先时间锁定跑通，再上完成度门控。

### 2.7 终止 / 成功 ✅
- **硬 ET**（比软惩罚强，RSI+ET 成对）：掉落 / 穿模过大 / 腕越界 / "该抓但接触力=0 持续 K 步"。
- **成功 = 撤支撑测试**（移走支撑，5s 内物体不滑落/掉落，D-Grasp）。

---

## 3. 端到端数据流

```
┌─ 离线数据准备 (per case) ─────────────────────────────────────────────────────────┐
│ 视频 ─►[Reconstruct_and_Retarget recon Path B: ViPE+HaWoR+FoundationPose]─► ①(手 MANO + 物体6D, 噪声)   │
│                        │                                                            │
│                        ├─►[magicdexmate 重定向]─► ①-手指(Sharpa 22关节)              │
│                        └─►(物体6D轨迹 + 手腕粗运动)─► ①-可信通道                       │
│ 物体mesh ─►[Affordance]─► 热力图 ─►[OCIR/BODex(sharpa_right)]─► ②grasp+接触(干净可信)   │
│                                        └─►[cuRobo(sharpa_right)]─► ③可行接近轨迹(锚点)  │
│              ↓↓↓ 全部经 frames.py 对齐到 IsaacLab env-local 系 (见 FRAME_ALIGNMENT.md) │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                     ▼
┌─ 训练 (RL, IsaacLab 768env) ──────────────────────────────────────────────────────┐
│ 残差策略(锚③) + wrench-PD浮动腕 + 重力松弛课程 + 相位受限RSI + 乘法imitation×接触reward │
│ 物体全程动态 · PPO + 非对称critic(特权) · 硬ET · 物理松弛课程                          │
│ 选型: probe 打硬指标(撤支撑成功率/穿模改善/物体轨迹保真) — 不信训练reward                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                     ▼
┌─ 推理 = 数据引擎 ─────────────────────────────────────────────────────────────────┐
│ 新视频 → ①②③ → 策略 rollout(从 t=0 完整跑) → 重采样到视频时间轴 → 修正后可行(手+物体)轨迹 │
│ 评估: 物理改善量(穿模↓/滑移/撤支撑) + 物体6D对 HOI4D GT 保真 (非手关节匹配)              │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                     ▼
              干净物理可行数据 ─► 下游学 Video→Policy   │   Phase-2: 换 DexMate 真机
```

---

## 4. 模块 I/O 契约

三个层次，别混：

| 层 | 输入 | 输出 |
|---|---|---|
| **A 策略**(每步) | obs(本体+参考误差+物体位姿+完成度) | 28-DOF 残差 |
| **B 修正函数**(每episode) | {③锚点, ②目标+接触, ①可信通道(物体轨迹+粗手), 物体位姿} | **一条物理可行 rollout(手+物体结果轨迹)** |
| **C 数据引擎**(每视频) | 视频→①②③ | **过滤后的干净轨迹**(撤支撑测试为准入门) |

- **输出 = 物理仿真结果轨迹，不是策略动作。**
- **训练/收割分离**：训练用 RSI 随机相位；**产出数据从 t=0 完整 rollout 跑一遍、重采样到视频时间轴 + 低通滤波去抖**。
- **输出 schema（要定死）**：`{手根6D位姿, 22关节角, 物体6D位姿, 接触点/法向/力, 每帧时间戳, 成功flag}` + 世界系 z-up + wxyz + 米 + 固定帧率。

---

## 5. 数据管线 ref builders（状态 + 契约）

| Producer | 产出 | 状态 | 帧/契约要点 |
|---|---|---|---|
| **Reconstruct_and_Retarget recon (Path B)** | ①视频 HOI：手 MANO(axis-angle) + 物体6D | 代码在，未接 | 用 `world_fused.npz`/`replay_world.npz`（**非** ego_pipeline Path A）；gravity_z_up_world，原点=ViPE gauge(要重锚)，30fps手/15fps物体。见 FRAME_ALIGNMENT |
| **magicdexmate 重定向** | **仅 22 手指关节角**(弧度,SDK序) | 已可用 | **无手腕**(base_rot_from_joints 推导或 npz)、**无物体**(数据集)；只用腕+5指尖约束→中节/外展/CMC 是优化器猜的(不可行来源) |
| **OCIR grasp**(实验室自研 https://github.com/jiaka1chen/ocir-grasp-synthesis.git) | ②grasp+接触 | **未 clone**；基于 BODex | BODex_modified 已配 sharpa_right；输出 `pregrasp/grasp/squeeze_qpos + contact_point/frame/force`；浮动手 robot_pose=world位姿1:1映射；物体 +90°about-X 对账 |
| **cuRobo(sharpa_right)** | ③可行接近轨迹(含手腕) | 配置在，未接 | 规划到 ②grasp；world系 z-up/米/wxyz；MVP 可先用 `poses/n4_pose1.npy`(002aa185真实包络抓取)当 ②-goal |

**本地现有 bootstrap 资产**（002aa185）：真实 mesh(1045顶点) + `poses/n4_pose1.npy`(真5指包络抓取) + `cache/graspxl_enclosing_*`(2807抓取)+`_contacts.npy`(每指接触标志)。

---

## 6. 算法 & 网络（复用 SharpaWave 引擎）

- **PPO + 非对称 actor-critic (v3net)**：actor 看 obs+priv瓶颈嵌入；critic 看 obs+原始priv(作弊)。768 并行 env、自适应 KL、GAE、输入/值归一化、ckpt 格式——**全部直接复用**（`ppo.py`, `models.py`, `ppo_cfg_v3net_p8.yaml`），只改 priv 语义(§2.3)。
- **选型 probe**：打硬指标（撤支撑成功率/穿模改善/物体轨迹保真），argmax，**不信训练 reward**；确定性、DR关、长评估窗。
- 教师-学生蒸馏(stage-2 ProprioAdapt)：💤 暂缓，先跑通 stage-1。

---

## 7. Phase 路线

- **Phase-1 (MVP)**：Flying Hand(28-DOF) · 单物体(002aa185) · **抓取+抬升的忠实修正**（物体轨迹追①到抬升；② 干净抓取；③ 锚点）· 手写乘法 reward · 重力松弛课程。目标=**证明手动 RL 能修正出可行抓取+操作**（去风险）。
- **Phase-1.5**：**"后面的轨迹"**（更长操作）；完成度相位门控；① 物体轨迹追踪延伸到全程。
- **Phase-2**：换 DexMate 手臂（cuRobo IK 映射腕目标→7关节；reward 不变）。
- **Phase-3+**：多物体/多任务泛化（残差锚每物体参考 + 点云几何 + 可选 affordance 通道消融）；VLM 生成 reward 权重 + simulator-in-the-loop 验证。

---

## 8. 与参考工作的关系

| 工作 | 我们采纳 | 我们不同 |
|---|---|---|
| **ManipTrans** (最接近, 人手→灵巧手→物理RL→数据集) | 残差-on-参考PD、6DoF wrench腕、物理松弛课程、一视频→一轨迹收割、"可行性由物理+残差涌现" | 它 MoCap 干净→无规划器；**我们 video 噪声→保留 OCIR/cuRobo 兜不可信通道**；imitator MVP 不做 |
| **PhysHOI** | 乘法 imitation×接触结构、接触项主导、物体松/手严、接触标定预处理 | 它单clip整体imitation；我们残差修正 + 分通道可信度加权 |
| **D-Grasp** | 浮动手 wrench控制、`r_c`接触力封顶、撤支撑成功测试、残差-on-参考+wrist-guidance、数据引擎(一label→多轨迹) | — |
| **DeepMimic** | RSI(含速度)、硬ET、有界exp核reward、PD目标动作 | 完成度相位(非时间)、非对称特权critic、接触项(它没有) |

---

## 9. 关键设计原则 & 陷阱

1. **可信度决定处理方式**：噪声参考不能贴着追；不可信通道交干净先验(OCIR/cuRobo)兜底。
2. **"零动作=物理OK"不成立**（cuRobo运动学锚点无接触动力学）→ 重力松弛课程 + 相位受限 RSI。
3. **接触项必须持续+主导**（PhysHOI 消融 95%→27%）；乘法结构让掉接触=整奖归零。
4. **门控/ET 防甩飞刷分**（SharpaWave 逃逸自旋教训）。
5. **reward 可自由用仿真真值**（不部署）；state(actor) 只放部署拿得到的。
6. **选型只信 held/撤支撑 probe，不信训练 reward**；评估窗 ≥2–3× 回合。
7. **HOI4D GT 不是干净手部目标**（GT人手带穿模）→ 评物理改善量 + 物体6D保真 + 接触IoU对OCIR，**不做手关节匹配**。
8. **帧对齐单测先行**（frames.py + 002aa185），两 doc 旧物体位置数字作废（env 实际 `(-0.0956,-0.0052,0.619)`）。
9. **多物体会稀释**（SharpaWave 34.4→1.1）；单物体先深，残差锚每物体参考缓解。

---

## 10. Build items（按依赖）& 待调数字

- [ ] **B1 frames.py** ✅设计：单一约定(z-up/wxyz/米/手根=Sharpa root) + T_recon→isaac(重锚) / xyzw→wxyz / 30→15fps / T_bodexscene→isaac(object相对+90°X) + **002aa185 单测**。
- [ ] **B2 OCIR/BODex 出 ②grasp**：MVP 可先用 `poses/n4_pose1.npy` 顶替；OCIR clone 后替换。
- [ ] **B3 cuRobo(sharpa_right) 规划 ③**：到 ②grasp 的 open→grasp(→lift) 轨迹(含手腕)。
- [ ] **B4 magicdexmate→① 轨迹拼装**：手指(magicdexmate)+手腕6D(推导/npz)+物体6D(Path B) → 对齐到 env 系；可选接触标定。
- [ ] **B5 `_gx_traj_wrist` 缓存**(No,Pmax,Lmax,7) + 载入 env + RSI 写手腕 root（**新建非复用**：现有张量只有 joint+object 无手腕）。
- [ ] **B6 RL env 改造**：wrench-PD腕 + 重力松弛课程 + 相位受限RSI + RSI速度 + 乘法imitation×接触reward(接触持续/ET/有界核) + critic-only contact_pos。
- [ ] **B7 输出收割+schema**：t=0 rollout→重采样→低通→写数据产品。
- [ ] 🔶 **调参**：#5 reward 各 k/λ/组合；#7 PD/F_max/T_max/scale 初值。
- [ ] 💤 Phase-3: 点云几何输入、affordance 通道消融、VLM reward、HOI4D 评估桥。

---

## 附：关键决策日志（可追溯）

1. 任务=残差修正 noisy HOI（非 SharpaWave 无参考旋转）；四根本改动：残差动作/参考进obs/追踪奖励/RSI复位/物体全程动态。
2. 手腕必动 → Phase-1 Flying Hand(28-DOF)去风险，reward 全任务空间→切 DexMate 零重写；工作区约束=限残差幅度。
3. 腕驱动 = SE(3) wrench-PD(Option A) + 重力打开 + wrench clamp（非运动学 teleport=方案C）。
4. obs 相对量/手系轴角；前瞻{t,t+1,t+5}；actor 物体位姿先真值后加噪。
5. 抓取成功稠密代理 → 后被 §2.4 乘法 imitation×接触 取代；成功=撤支撑测试。
6. affordance 只在 OCIR 上游、不进 RL；单物体先做。
7. **(b) 忠实修正是目标、人手不砍**（推翻 (a)→(b) 分期）。
8. **ManipTrans 定论**：可行性由物理+残差涌现，MoCap 参考不用规划器。
9. **用户纠正（决定性）**：我们是 **video-recon 噪声参考**，与 ManipTrans 的 MoCap 不同 → **OCIR/cuRobo 回来**（按可信度分通道），锚点=cuRobo③，人手在可信通道(物体轨迹+粗手)实质追踪，imitator MVP 不做。→ 收敛到 §1 架构。
