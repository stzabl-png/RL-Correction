# Unscrew V5 强化学习环境 — 物理建模与奖励配置总览

**分支** `unscrew_v5` · **任务** 双手拧瓶盖（左手持瓶 / 右手拧盖 → 把盖放到桌上 → 撤回）
**代码根** `tasks/Unscrew/part4/` · **全史台账** `A_Design/DECISIONS.md`

这份文档写给**只需要 RL 环境（物理 + 奖励）的人**：看完知道每个文件负责什么、
关键参数是什么口径、想改哪里去改、以及怎么验证自己没改坏。

---

## 0. 三十秒上手

```bash
git checkout unscrew_v5 && git lfs pull
export PY=/path/to/isaac/python

# 判据/奖励的离线自检 (不开 Isaac, 十几秒)
UNSCREW_CLIP=32 PYTHONPATH=. $PY tasks/Unscrew/part4/A_Design/L3_Learning/selftest_regime.py
UNSCREW_CLIP=32 PYTHONPATH=. $PY tasks/Unscrew/part4/A_Design/L3_Learning/selftest_variants.py

# 建环境 + 零动作跑一遍 (需要 Isaac; 只需 reference_v1.npz, 仓库里已有)
CUDA_VISIBLE_DEVICES=0 TMPDIR=$HOME/tmp UNSCREW_CLIP=32 SHARPA_WANDB=0 \
OMNI_KIT_ACCEPT_EULA=YES POUR_SQUEEZE_FF=1 PYTHONPATH=. \
  $PY -u tasks/Unscrew/part4/C_Wiring/smoke_zero.py --steps 400 --headless
```

> 只有**发射训练**才额外需要 `reference_v2.npz` + `acceptance_v2.json`
> （见 `docs/HANDOFF_20260901_unscrew_v5.md`）。看环境/奖励不需要它们。

---

## 1. 文件职责一览

### 环境与接线 `C_Wiring/`
| 文件 | 负责什么 |
|---|---|
| **`task_config.py`** | **所有可调参数的单一来源**：路径、物体几何、螺纹口径、β 剂量、抓取先验与标定值。改配置只改这里 |
| **`task_env.py`** | Isaac 环境接线：观测 507 / 动作 58、残差界与门控、奖励求和、死线、RSI 出生、诊断台账。**判据本身不在这里**（在 L3_Learning） |
| `train_task.py` | PPO 入口（含渐进 RSI 解锁、TB 台账、母带/世界指纹预检） |
| `eval_task.py` | 确定性评测（**成功率只认这个**，训练期 TB 有分母问题） |
| `smoke_zero.py` | 零动作冒烟：静置对账 / 机器段死线 / 各出生点行为 |
| `world_fingerprint.py` | 世界指纹：机器人/母带/判据摘要/螺纹/物性逐项比对，防"换了东西还接着训" |
| `ppo_task.yaml` | PPO 超参 |
| `launch_remote.sh` / `verify_deploy.sh` | 发射 / 部署自检 |

### 判据与奖励 `A_Design/L3_Learning/`
| 文件 | 负责什么 |
|---|---|
| **`progress.py`** | **判据的唯一事实源**（标量版）：四级 Gate、护送三件、死线、置信度档位、奖励项定义。所有阈值集中在文件头 |
| `progress_batch.py` | 同一套判据的 torch 批量版（训练用），与标量版**逐位一致**由自检保证 |
| `REWARD_DOC.md` | 奖励表的设计说明（每一项为什么这么设） |
| `selftest_*.py` | 五件自检：放音 / 批量一致 / RSI 全点位 / 合法变体+反向 / 体制 |

### 参考轨迹（母带）`A_Design/L1_Data` + `L2_Reference`
| 文件 | 负责什么 |
|---|---|
| `plan_machine_segs.py` | cuRobo 规划机器段（站姿↔操作位置），产出 `Approach.npz`/`Retreat.npz` |
| `make_reference.py` | 离线构建母带 v1：物体换基、腕参考、抓取先验锚定、机器段拼接 |
| `B_SmokeTest/build_reference.py` | 在 Isaac 内重铸 v2（交互段腕参考重锚到实测站位） |

### 探针 `B_SmokeTest/`（标定值都是**量**出来的，不是猜的）
| 探针 | 用途 |
|---|---|
| `probe_thread.py` | **真实螺纹副五段物理验证**：静置不转 / 阈下锁死 / 阈上稳态转速 / 撤力回锁 / 拧满脱扣 |
| `probe_capgrasp.py` | 盖抓取候选逐个实测接触；`--fit` 闭环对准、`--curl` 逐级捏合、`--pick` 钉候选 |
| `probe_pinch.py` | 站位行逐级捏合，报三指接触/指力/组件扰动 + 实际腕位对账 |
| `probe_grasp.py` | 左手抓握几何 + 三方对账（命令/实际/离线 IK）+ 逐行盯瓶 |
| `probe_acceptance.py` | 训练可用性验收，写世界指纹凭据 |
| `probe_rest.py` | env 实测静置位/锚（母带换基的输入） |

---

## 2. 物理建模：真实螺纹副（U40 + U45）

**要解决的问题**：早期用"接触门 × ω 阻尼"当摩擦替身——**有接触就白给转动**，
于是碰一下瓶盖它就自己转开脱落，策略学到的是"戳"而不是"握着拧"。

**现在的模型**（`tasks/pregrasp/screw_assembly.py` + `ScrewSpec`）：

| 机制 | 值 | 含义 |
|---|---|---|
| 咬合期盖惯量 | `I_eff = 5e-4 kg·m²`（球形） | 指尖→盖的可传扭矩由 **PhysX 摩擦锥**裁决（≤μ·N·r，捏得紧才传得多） |
| 静锁 breakaway | `0.04 N·m` | 已破封松盖的量级；轻拍/轻擦永远不解锁 |
| 库仑 / 粘滞 | `0.015 N·m` / `0.03 N·m·s` | τ=0.06N·m 时稳态 **1.5 rad/s**（≈人手拧速） |
| 转动律 | `ω = sign(τ)·(|τ|−kinetic)⁺ / viscous` | **准静态（过阻尼）**：不积分存动量，手停即停；"戳转"物理上失效 |
| 回锁 | `|ω|<0.05` 且 `|τ|<breakaway` | 天然的静/动摩擦滞环 |
| 反作用扭矩 | 回施于瓶身 | 左手持瓶必须抗扭 |
| 拧开角 | **`turns = 30°`** | 已破封松盖；拧满即 detach（`screw_detach_at_full`） |
| 安全夹 | `2.5 rad/s` | 只是安全栏，整形交给摩擦模型 |

**力矩估计的三条防幻影**（三次尸检换来的，改动前务必读 `screw_assembly.py` 注释）：
① 只看盖侧；② 按**矢量**差分再投影（螺轴变向会漏）；③ **零接触真值门**
（指尖没碰盖就不可能有力矩）。

**验证**：`probe_thread.py` 五段全过 —— 静置力矩 0.00 mN·m、阈下 20.0 vs 外加
20 mN·m、阈上稳态 1.50 vs 解析 1.50 rad/s、撤力滑行 1.2°、拧满脱扣。

---

## 3. 奖励与判据

### 3.1 两层结构
- **稀疏任务核**：四级里程碑 Gate，只认仿真物理量
- **稠密引导**：跟踪演示的物体轨迹与人手轨迹，按**逐帧置信度**加权

### 3.2 Gate 与死线（`progress.py` 文件头集中定义）

| Gate | 判据 | 奖励 |
|---|---|---|
| G1 抓形成 | 左手 ≥3 垫触瓶，持续 10 步 | +5 |
| G2 提升认证 | 腕参考+15mm，双物 z 升≥5mm 且左腕-瓶滑移<8mm | +8 |
| G3 拧开释放 | 螺纹拧满 30° 物理脱扣（**仿真量，不用重建的盖转角**） | +10 |
| placed | 双物到母带末行位姿（瓶3cm/15°、盖5cm/30°）hold15，**且护送三件全过** | 0（内部态） |
| G4 撤退=Success | 双臂贴站姿逐关节<10° hold15，物体不被碰歪 | +15 终局 |

**护送三件**（成功必须是"右手**拿着**盖**放**到桌上"）：
① 过带：盖无右垫接触且单步降>8mm 地穿过桌面+3cm 带顶 ⇒ 永久失败
② 持盖：释放后 ≥2 右垫触盖累计 ≥10 步
③ 受控下放：释放后单步降幅峰值 <4cm/步（自由落体 ~2m/s 一票否决）

**死线**：D1 掉落 5cm / D2 瓶倒 30° / D3 偏离母带 35cm / D4 滑移 5cm /
D5 插桌 / D6 双臂互撞 / D7 超时。机器段（cuRobo 背书那段）**零误触是铁则**。

### 3.3 稠密项（`task_env.py` 求和，逐项进 TB `ep_rew/*`）

| 项 | 含义 |
|---|---|
| `adv` | 时钟推进（棘轮） |
| `leash` | 物体跟踪皮筋，按置信度档放松（绿3cm/黄5cm/红8cm；红档 rot 禁判） |
| `ms` | Gate 里程碑一次性收入 |
| `screw` | **拧转势** `K_SCREW×Δθ`（双向计，回拧扣分）。总额固定 ≈9.4，与 G3 的 +10 同量级 —— **`K_SCREW` 随 `turns` 自动推导**，改 turns 不会稀释引导 |
| `fshape` | 手指形状指引（人手指流方向余弦，受人手置信度门控） |
| `regrip`/`slope`/`pen` | 反射奖 / 斜坡罚 / 贴实罚 |

### 3.4 置信度门控（这批数据的关键）
物体侧 `tmix = min(瓶, 盖)` 三档：绿 `W_OBJ=1.0/W_HAND=0`、黄 `0.5/0.5`、红 `0.2/0.8`；
人手侧 `W_HCONF` 绿1/黄0.5/红0。
**原则：判定层只认物理量，置信度只调引导权重。**
clip32 实测：物体位置置信度中位 85/83（绿档 67%/74%），**旋转只有 54/50**（绿档
18%/23%）——所以盖/瓶绕自身轴的自转被当作**规范自由度**解出来，而不是照抄。

---

## 4. 人手→机械手的重定向（这版的重点）

演示数据有两个"高置信度也不能直接用"的通道：
- **腕平移是死数据**（clip32 全程位移 0.61cm，同期盖走了 66cm）→ 腕位由物体轨迹/先验推导
- **手指是人手关节角**，直接搬到 SharpaWave 不构成握 → 必须走 GraspPose 先验

| 手 | 先验 | 标定值（都在 `task_config.py`，注释写了怎么量的） |
|---|---|---|
| 左手持瓶 | `Screw27_body`（右手约定 → **镜像**给左手） | `BETA_L=1.0`、`PRIOR_RADIAL_TRIM=0.007`、物体系原点补 +9.85cm |
| 右手拧盖 | `Screw27_cap_candidates`（**左手**抓法 → **镜像**给右手） | `CAP_GRASP_PICK=8_0_calib`、`CAP_GRASP_TRIM=(0.0046,0.0171,-0.0417)`（**盖坐标系**）、`CAP_PINCH_DEG=25`、`BETA_R=1.0` |

**两个反复咬人的坑**（换 clip / 换手必看）：
1. **物体系原点**：数据集 mesh 原点在**底部**，Dexonomy 先验的物体系是**居中**的——
   直接搬会差半个物体高（左手曾因此锚在桌面以下、被物理顶高 10cm 去抓空气）。
2. **自转规范**：腕姿由先验确定后，早期那套"绕螺轴自转扫描"会把先验姿态再转一档
   （实测正好 60°）——腕位对得上而姿态全错。已关闭。

**零动作实测（当前配置）**：左手站位 4 垫抓稳；右手 **三指 2/3、指力 7.58N、
螺纹角 13.7°**，再合 10° 即拧满 30° 脱扣。

---

## 5. 观测 / 动作 / 残差

- **观测 507** = 框架 503 + 螺旋块 4（`screw_frac`、`released`、`n_triad/3`、`drive_gain`）
- **动作 58** = [右臂7, 左臂7, 右指22, 左指22]，**残差**叠加在母带前馈上
- 残差界：臂 机器段 ±0.05 rad / 交互段 ±0.08 rad；手指 ±5.7°/步、累计 ±68.8°
- 门控：机器段冻结手指、Retreat 冻指开臂；缝1 与交互段全开

---

## 6. 改了东西怎么验证没改坏

```bash
# 1) 判据/奖励改动 -> 五件自检必须全绿 (十几秒, 不开 Isaac)
for f in progress progress_batch rsi variants regime; do
  UNSCREW_CLIP=32 PYTHONPATH=. $PY tasks/Unscrew/part4/A_Design/L3_Learning/selftest_$f.py
done
# 2) 物理改动 -> 五段探针必须全过
... $PY -u tasks/Unscrew/part4/B_SmokeTest/probe_thread.py --headless
# 3) 抓握/几何改动 -> 探针复测接触 (别靠 IK 可达率猜, 可达 ≠ 抓得住)
... $PY -u tasks/Unscrew/part4/B_SmokeTest/probe_pinch.py --headless
```

**纪律**（`docs/DESIGN_LOOP.md`）：一次只改一个参数；改动上线前把假设与证伪条件
写进 `A_Design/DECISIONS.md`；判据阈值一改，世界指纹的 `criteria.digest` 就变，
旧的验收凭据自动失效——这是特性不是麻烦。

---

## 7. 已知边界（别当成已解决）

1. **奖励结构尚未经过一次成功训练验证**。上一轮 8.5M 步是在"右手零接触"的旧配置上
   跑的（零信号）；右手重定向刚打通，新一轮训练的判读针见 HANDOFF 文档。
2. **t0 出生点在缝1 合拢瞬态可能碰倒瓶**（`DECISIONS` T2-6e）。机器段改成端到端
   到操作位置后可能已缓解，待冒烟复核。
3. **右手三指只到 2/3**，拇指/中指还差约 1cm 收拢，交给 RL 手指残差。
4. **发射训练必须带 `POUR_UNLOCK=1,2,3`**，否则渐进 RSI 会死锁（解锁 g1 要求
   t0 口径的 G1≥30%，而 t0 够不着 ⇒ 永不解锁，实测 8.5M 步零信号）。
