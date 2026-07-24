# SharpaWave 手内旋转冠军策略 — 完整拆解与改造指南

> 目标读者：想逐块理解这套策略、然后改成自己策略的人。
> 覆盖对象：**Pipeline 冠军** = gym 任务 `Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-Net-v1`，checkpoint `checkpoints/pipeline_fullg_adjrot.pth`（README: +21.8 rad 持握旋转/回合，≈3.5 圈，≈90 s 回合）。
> 全部数值、公式、`file:line` 均来自源码核对。所有路径相对 `/home/lyh/Project/RL_Correction`。

---

## 0. 总览：这套策略到底是什么

一句话：**用 HORA/RMA 风格的 PPO（非对称 actor-critic）+ 一个精心设计的"带通旋转奖励(band-pass)"，训练 22-DOF 灵巧手把一个 7 cm 光滑球在手内持续旋转**。它的核心创新几乎全在**奖励设计**和**复位分布**上，网络和 PPO 本身是标准的。

整条流水线（每一层后面章节详解）：

| 层 | 关键内容 | 主文件 |
|---|---|---|
| 环境 | 22 关节力矩PD控制；obs=195（3帧本体感觉+触觉+手系重力）；priv=8（特权信息给critic） | `sharpa_wave_env.py`, `sharpa_wave_env_cfg.py` |
| 奖励 | 带通旋转收入 + 持握门控 + 接触数缩放 + 突发计credit + 稀疏棘轮 + 温和居中 | `sharpa_wave_graspxl_bandrot.py`, `..._posebank.py` |
| 复位 | 50% 原始pinch + 50% 包络抓取姿态库（教"先调整再旋转"）；重力课程 −0.05→−9.81；物体力扰动 | `..._adjustrotate_diverse.py`, `graspxl_env.py` |
| 算法 | PPO + v3net（分离critic、priv-8嵌入、σ下限0.1、熵1e-3） | `algo/ppo/ppo.py`, `algo/models/models.py`, `agents/ppo_cfg_v3net_p8.yaml` |
| 选型 | **训练reward不是选型信号**；快照扫描 argmax held-gated probe | `scripts/snapshot_probe.py`, `rot_held_probe.py` |

**冠军关键常量速查**：obs=195 · priv=8 · action=22 · 力矩PD · `action_scale=1/24` · `clip_actions=1.0` · `dof_limits_scale=0.9` · sim dt=1/240 · decimation=12 → 控制20Hz · 训练回合40s(800步)/评估120s · 物体固定0.10kg · 重力课程封顶−10 · DR: PD[0.5,2]/摩擦[0.5,2]/CoM±0.01/力扰动scale2.0；**质量DR关、尺寸DR关**。

**两个冠军别混淆**（都在同一个 env，只是 env-var 不同）：
- **Pipeline 冠军**（本文档主角，`train_pipeline_champion.sh`，`-Net-v1`，+21.8 rad *持握* 旋转）：带通+持握门控+n缩放+突发credit+稀疏棘轮+温和居中+力扰动+pinch混合复位。**不设** `NETPROG_W`/`DROP_PENALTY`/`BAND_SCALE_END`/`PC`。
- **Net-progress 冠军**（`train_minimal_design_netprog.sh`，`-NetPC-v1`，+34.4 rad *物理净* 旋转）：加了高水位净进度项、终末掉落罚、脚手架退火、点云条件；`CENTER_SCALE=0.0`。这些项在 Pipeline 冠军里**全部为 0**。两个指标口径不同，**不可直接比较**（+21.8 是持握口径，+34.4 是物理净口径，更严格）。

---

## 1. 环境 / 观测 / 动作 / 物理

**类继承链（env）**：`SharpaWaveGraspXLBandRotPoseBankEnv`(`..._posebank.py:219`) → `SharpaWaveGraspXLBandRotEnv`(`..._bandrot.py:337` = `_BandRotMixin` + `...AdjustRotateDiverseEnv`) → `...AdjustRotateEnv` → `SharpaWaveGraspXLEnv`(`graspxl_env.py`) → `SharpaWaveInhandRotateEnv`(`sharpa_wave_env.py`, DirectRLEnv)。
**cfg 链**：`...PoseBankCfg`(`..._posebank.py:14`) → `...BandRotCfg`(`..._bandrot.py:308`) → `...AdjustRotateDiverseCfg` → ... → `SharpaWaveGraspXLSustainedCfg`(`sharpa_wave_env_cfg.py:566`) → ... → `SharpaWaveEnvCfg`(`:50`)。

### 1.1 观测向量

`_get_observations()` 返回一个 dict，最多 5 个 key（`sharpa_wave_env.py:263-277`，GraspXL 层扩展 `graspxl_env.py:200-211`）：

| dict key | shape | 消费者 | 说明 |
|---|---|---|---|
| `policy` | `(N, 195)` | actor（+critic） | 本体感觉/触觉 3 帧堆叠(192) + **手系重力(3)** |
| `priv_info` | `(N, 8)` | actor 特权嵌入 **和** 分离 critic | HORA 特权外因(privileged) |
| `proprio_hist` | `(N, 30, 64)` | 仅 stage-2 自适应 TConv（冠军不用） | 最近 `prop_hist_len=30` 帧 |
| `pointcloud` | `(N, P+B, 5)` | PointNet 分支 | 仅 `SHARPA_PC=1`（默认关） |
| `obj_pose` | `(N, 7)` | world-model 头 | 默认关 |

**`policy` 单帧布局（64 维/帧）**（`sharpa_wave_env.py:494-501`；`proprio_frame_dim=64`，`sharpa_wave_env_cfg.py:584`）：

| 切片 | 维 | 内容 | 计算 |
|---|---|---|---|
| `[0:22]` | 22 | **关节位置**，加噪并 unscale 到 [-1,1] | `unscale(q + U(-1,1)*0.02)`；噪声 `joint_noise_scale=0.02`(`:291`)；`unscale(x)=(2x-u-l)/(u-l)`(`:545`) |
| `[22:44]` | 22 | **当前位置目标** `cur_targets`（原始弧度） | `:499` |
| `[44:49]` | 5 | **触觉/接触力** 每指一标量（拇/食/中/环/小） | 平滑+延迟滞后的接触力范数 `sensed_contacts`(`:456-472`) |
| `[49:64]` | 15 | **接触位置**（5指×xyz） | 弹性体系；此任务 `enable_contact_pos=False` → **置零**(`:484-485`) |

policy obs = 最近 **3 帧** flatten = `3×64=192`（`:515`），再由 GraspXL 层**追加手系重力向量(3)** → **195**（`graspxl_env.py:205-210`）：
```python
g = self.physics_sim_view.get_gravity(); gvec = tensor([g0,g1,g2])/9.81
g_hand = quat_apply(quat_conjugate(hand.root_quat_w), gvec.expand(N,3))
obs["policy"] = cat([obs["policy"], g_hand], -1)   # 192 -> 195
```
> **为什么加手系重力**：本体感觉+触觉是自我中心的（手转到任何姿态数值都一样），手系里的重力方向是唯一能区分手朝向的线索 —— SO(3) 等变性修复。滑窗历史 `obs_buf_lag_history`=`(N,80,64)`，复位时用当前状态回填避免零填充瞬变(`:505-514`)。

**触觉计算核心**（`sharpa_wave_env.py:456-472`）：5 个弹性体接触传感器力范数 → `contact_smooth=0.5` 平滑 → `contact_latency=0.005` 延迟滞后 → 持久量 `self.last_contacts`(`(N,5)`)，被奖励/持握门复用。`binary_contact=False` → 模拟力值而非二值。

**`priv_info`（8 维）特权外因**（`priv_info_buf`, `sharpa_wave_env.py:76`）：

| 切片 | 内容 | 位置 |
|---|---|---|
| `[0:3]` | 物体位置位移 `object_pos − object_default_pose[:3]`（逐步实时） | `:518` |
| `[3]` | 摩擦随机化标量 | `:147` |
| `[4]` | 物体质量 | `:155`（**冠军 `randomize_mass=False` → 恒 0，无信息**）|
| `[5:8]` | 物体质心(x,y,z) | `:151` |

**actor/critic 非对称**（"separate critic；priv 8 raw→critic；actor embeds first 8"）：
- `separate_critic:True` → 独立 critic MLP `[512,256,128]`，输入 `195+8=203` **原始** priv（`models.py:176-178`）。
- `actor_priv_dim:8` → actor 把前 8 维 priv 过 `priv_mlp=[256,128,8]` 再 tanh 压成 8 维 extrinsic 拼到 obs → actor 输入 203（`models.py:167-169`）。
- 两者都看到 8 维 priv，但 **actor 经过瓶颈嵌入（可部署/可 HORA 蒸馏），critic 看原始**（Dactyl 式非对称值函数）。

### 1.2 动作空间

- **`action_space=22`**（`sharpa_wave_env_cfg.py:53`），每关节一维。
- **控制模式：力矩(computed PD)**，`torque_control=True`(`:64`)。init 时把关节 PD 刚度/阻尼清零，让 env 自算力矩(`sharpa_wave_env.py:101-107`)。
- **动作是增量位置目标 delta**（`:236-241`）：
```python
actions = saturate(actions, -1, +1)
targets = prev_targets + action_scale * actions     # action_scale = 1/24 ≈ 0.0417 rad/单位
cur_targets[:,act] = saturate(targets, lower, upper)
```
- **目标→力矩**（`:254-261`）：`torques = p_gain*(cur_targets - q) - d_gain*qd`，`set_joint_effort_target(...)`。
- PD 增益取 USD 默认再逐 env 随机化；`clip_actions=1.0`(`:62`)；`dof_limits_scale=0.9`(`:303`) 把 URDF 限位乘 0.9。
- **22 关节顺序** `actuated_joint_names`(`:211-234`)：拇指(CMC_FE,CMC_AA,MCP_FE,MCP_AA,IP)、食/中/环(MCP_FE,MCP_AA,PIP,DIP)、小指(CMC,MCP_FE,MCP_AA,PIP,DIP)。

### 1.3 物理

| 量 | 值 | 位置 |
|---|---|---|
| sim `dt` | `1/240 s` | `:67` |
| `decimation` | 12 → 控制步 `0.05s`(20Hz) | `:60` |
| `episode_length_s` | 20（训练用 `SHARPA_EPISODE_S=40`→800步；评估120s） | `:52` |
| 基础 `gravity` | `(0,0,-0.05)`（课程爬升） | `:69` |
| 求解器 | TGS，pos-iter 8，vel-iter 0 | `:70-77` |
| 手 | gravity **禁用**、**自碰撞开** `enabled_self_collisions=True` | `:87-96` |
| 物体 | GraspXL 球，质量 **0.10kg 固定**，convexDecomposition 碰撞，摩擦 combine=`max` | `object_cfg.py:13,37-90`；`adjustrotate_diverse.py:50-59` |
| 接触传感器 | 5 弹性体（history=3，读取用）+ 5 DP（配置但不读） | `sharpa_wave_env_cfg.py:146-209` |

**掉落判定**（冠军走位移口径 `_orient_dones`, `graspxl_env.py:273-299`）：`‖object_pos − object_default_pose[:3]‖ > 0.10m` 且非 settle 窗 → drop（唯一失败终止；驱动重力课程的 `drop_rate`）。

### 1.4 域随机化(DR)

| DR | 冠军生效 | 参数 | 位置 |
|---|---|---|---|
| PD 增益 | 开 | scale[0.5,2.0] 逐 env 逐 dof | `sharpa_wave_env.py:338-360` |
| 摩擦 | 开 | scale[0.5,2.0]，→`priv[3]` | `:137-147` |
| CoM | 开 | ±0.01m，→`priv[5:8]` | `:148-151` |
| 质量 | **关**（GraspXLCfg `:401`） | (本会 U(0.01,0.25)) | 固定 0.10kg |
| 尺寸 | **关** `[1,1,1]` | | `:394` |
| 物体力扰动 | **开 scale=2.0**（脚本设） | prob 0.2，`randn*mass*2.0`，decay 0.9/0.08s | `:245-252`；`posebank.py:118-127` |

> **评估时**：probe 关掉 DR（`randomize_*=False`）、关力扰动、固定 g=−9.81、关课程。

### 1.5 影响环境层的 SHARPA_* 变量（选摘）

| 变量 | 默认 | 作用 | 位置 |
|---|---|---|---|
| `SHARPA_POSE_CACHE` | "" | `(N,29)` 复位姿态缓存路径 | `posebank.py:21-24` |
| `SHARPA_GRASPXL_OBJECT` | `002aa185...` | import 期选默认物体 | `sharpa_wave_env_cfg.py:23` |
| `SHARPA_PC` | 关 | 开点云 obs 分支 | `posebank.py:80-83` |
| `SHARPA_GRAVITY_Z` | 课程 | 固定重力、关课程 | `posebank.py:106-113` |
| `SHARPA_EPISODE_S` | 20 | 回合长度(s) | `posebank.py:114-117` |
| `SHARPA_ENC_FRAC` | 1.0 | 包络缓存 vs pinch 复位比例 | `posebank.py:97-105` |
| `SHARPA_ROT_AXIS` | (0,0,1) | 旋转轴（同时改奖励轴） | `posebank.py:154-163` |
| `SHARPA_DATASET_ROOT` | 作者路径 | 数据集根 | `sharpa_dataset.py:24-29` |

---

## 2. 奖励系统（核心中的核心）

每步总奖励（`sharpa_wave_graspxl_bandrot.py:282`）：
```
total = base_total + r_rot + sparse_reward + netprog_reward + drop_reward
```
其中 `netprog_reward`、`drop_reward` 对 Pipeline 冠军 **恒为 0**（对应 env-var 未设）。全局：`step_dt=0.05s`，`rot_axis=(0,0,1)` 世界-z，`last_contacts` 是每指一标量的 5 维接触力。

### 2.1 基础栈 `base_total`（`sharpa_wave_env.py:279-303, 549-562`）

```python
object_angvel = axis_angle_from_quat(quat_mul(object_rot, quat_conj(object_rot_prev)))/step_dt  # rad/s
rotate_reward = saturate((object_angvel*rot_axis).sum(-1), -0.5, +0.5)
object_linvel_penalty = ||Δobject_pos||_1 / step_dt
torque_penalty = Σ τ²
work_penalty   = (Σ τ·q̇)²
_disp = ||object_pos − object_default_pose[:3]||
object_pos_diff = exp(−_disp / 0.03)    # 有界居中，center_reward_bounded=True
```

| 项 | 冠军权重 | 说明 |
|---|---|---|
| rotate(基础clip) | **0.0（关）** | 被带通取代；`bandrot.py:317` |
| object_linvel 罚 | −0.5（强制反平移） | `env_cfg:589` |
| pos_diff 罚 | 0.0（AdjustRotate 清零，给调整自由） | `adjustrotate.py:31` |
| torque 罚 | −0.1 | `env_cfg:591` |
| work 罚 | −0.5 | `env_cfg:592` |
| 居中 object_pos | **0.3**（`SHARPA_CENTER_SCALE`，默认 1.0） | `env_cfg:595` |
| readiness-Φ 整形 | **关**（`use_readiness_potential=False`） | `bandrot.py:319` |

> **关键教训**：基础 clip-rotate 是"被奖励的抖动吸引子"（signed clip 带 0 下限 → ω≈0.037 抖动也拿正奖励）；带通把它关掉。居中默认 1.0 时给静止死握的年金 3× 于可学的旋转 → 必须砍到 0.3。有界 `exp(−disp/σ)` 而非无界 `1/(disp+ε)`（后者死握时爆到 ~400）。

### 2.2 带通旋转收入 `r_rot`（`bandrot.py:77-208`）—— **策略的灵魂**

**① 瞬时角速度**（`:78-81`）：quat delta → axis-angle → /dt → 投影到 `rot_axis` = `omega_signed`。
**② EMA 平滑**（`:84-86`）：`omega_ema = a·omega_ema + (1−a)·omega_signed`，`a=ema_alpha=0.85`。±0.5 方波收缩到 `|ema|≈(1−a)/(1+a)·0.5≈0.041` < 0.06 下限 → 振荡不给钱。冠军用 0.85（非默认 0.9），因 0.9 会把带通开启延迟 ~10 步，欠付短突发。
**③ 带通形状**（`:89-126`）：
```python
mag  = clip((|ema| − floor)/(omax − floor), 0, 1)         # floor=0.06, omax=3.0
over = clip((|ema| − c0)/((c0+0.8) − c0), 0, 1)           # c0 = ceil = 1.2
mag  = mag * (1 − over)                                    # 1.2 以上逐渐减到 0（2.0 处全零）
band = band_scale * mag * sign(ema)                       # band_scale=8.0
```
- **下限 0.06**：0.06 以下**精确零**付款 → 杀死抖动吸引子（实测抖动 0.037）。默认 0.15 会在 0.03–0.09 区间零付款，蠕动到驱动之间无梯度；0.06 保留抖动死区又恢复其上斜率。
- **上限 1.2**：无界饱和斜坡会付 MAX 给任意暴力自旋 → "甩飞机器"。1.2 以上把付款渐减到 0（2.0 处全零），让 ~1 rad/s 可控区成为最优。
- **scale 8.0**：线性付款系数（默认 3.0）。

**④ 自适应上调**（`:129-132`）：`adaptive_mult = clip(1 + 0.5·max(0,mean_revs_per_ep), 1, 3.0)`；`mean_revs_per_ep` 每 2000 步提交一次已完成回合的净圈数均值。
**⑤ 持握门控**（`SHARPA_BAND_HELD=1`, `:139-153`）：
```python
n_eng_h = (last_contacts > 0.15).sum(-1)                  # band_held_thresh=0.15
disp_h  = ||object_pos − object_default_pose[:3]||
held_gate = ((n_eng_h >= 2) & (disp_h < 0.05)).float()    # min_fingers=2, held_disp=0.05
r_rot = r_rot · held_gate
```
> "持握" ⇔ **≥2 指接触(力>0.15) 且 物体距座位<0.05m**。关闭"逃逸自旋漏洞"（掉落/滚动的球也累积 ω 却未被握住 —— 零重力下观察到自旋刷分）。
**⑥ 接触数缩放**（`SHARPA_BAND_NSCALE=3.0`, `:173-185`）：`nmult = clip(n_eng/3, 0, 5/3)` → n2:0.67 n3:1.0 n4:1.33 n5:1.67，n0:0。连续无台阶。

**冠军 `r_rot` 全式**：
```
r_rot = [8.0·clip((|ema|−0.06)/2.94,0,1)·(1−over)·sign(ema)] · clip(1+0.5·revs,1,3) · held_gate · clip(n/3,0,1.67)
```

### 2.3 稀疏棘轮 `sparse_reward`（`bandrot.py:210-230`）

```python
omega_eff = sign(ema)·max(|ema|−0.06,0)·held_gate         # 供超下限、持握门控的有效速度
omega_eff = clip(omega_eff, −1.2, +1.2)                   # 上限也夹棘轮积分
cum_angle += omega_eff·step_dt                            # 净有效角度
new_level  = floor(cum_angle / 0.15)                      # sparse_theta=0.15
gained     = max(0, new_level − _sparse_level)            # 只有净前进付款
sparse_reward = 2.5 · gained                              # sparse_bonus=2.5
_sparse_level = max(_sparse_level, new_level)             # 棘轮：后退不付、回到旧地不重付
```
突发计 credit（EMA 0.85 + 细棘轮 0.15/2.5）单独一项就把持握速率从 0.15 拉到 0.94 rad/s。

### 2.4 高水位净进度 `netprog_reward`（`bandrot.py:232-249`）—— **仅 Net-progress 冠军**

`SHARPA_NETPROG_W=15.0`；`in_play = (disp<0.10)`；`_phys_cum += omega_signed·dt·in_play`（**物理净、所有步、非门控 → 滑退真的扣分**）；`netprog_reward = 15.0·max(_phys_cum − _phys_hw, 0)`；`_phys_hw = max(_phys_hw, _phys_cum)`。
> **跑步机免疫**（5→3→5 循环重走旧地 → 付 0）+ **恢复安全**（掉落再爬不罚，只是不挣钱直到破纪录）。**冠军里 = 0。**

### 2.5 终末掉落罚 `drop_reward`（`bandrot.py:268-281`）—— **仅 Net-progress 冠军**

`SHARPA_DROP_PENALTY=15.0`；`drop_reward = −15.0·((disp>0.10) & ~settling)`。**冠军里 = 0。**

### 2.6 完整 SHARPA_* 奖励旋钮表（"C"=Pipeline冠军 "NP"=Net-progress）

| 变量 | cfg 属性 | 默认 | 作用 | C | NP |
|---|---|---|---|---|---|
| `SHARPA_BAND_HELD` | `band_held_gate` | False | 持握门控 band+sparse | 1 | 1 |
| `SHARPA_BAND_NSCALE` | `band_n_scale_div` | 关 | 连续 n 缩放 | 3.0 | 3.0 |
| `SHARPA_BAND_FLOOR` | `band_omega_floor` | 0.15 | 零付款下限 | 0.06 | 0.06 |
| `SHARPA_BAND_SCALE` | `band_scale` | 3.0 | 带通线性系数 | 8.0 | 8.0 |
| `SHARPA_BAND_CEIL` | `band_omega_ceil` | 关 | 带通上限 | 1.2 | 1.2 |
| `SHARPA_CENTER_SCALE` | `object_pos_reward_scale` | 1.0 | 居中权重 | 0.3 | 0.0 |
| `SHARPA_BAND_EMA` | `ema_alpha` | 0.9 | ω EMA 系数 | 0.85 | 0.85 |
| `SHARPA_SPARSE_THETA` | `sparse_theta` | 0.3 | 棘轮粒度 | 0.15 | 0.15 |
| `SHARPA_SPARSE_BONUS` | `sparse_bonus` | 5.0 | 每级付款 | 2.5 | 2.5 |
| `SHARPA_NETPROG_W` | `band_netprog_w` | 关 | 高水位净进度 | — | 15.0 |
| `SHARPA_DROP_PENALTY` | `drop_penalty` | 关 | 终末掉落罚 | — | 15.0 |
| `SHARPA_BAND_SCALE_END`/`_SCAFFOLD_ANNEAL` | `band_scale_end`/`band_scaffold_anneal` | 关/40000 | 脚手架退火 band_scale | — | 2.0/50000 |
| `SHARPA_ENC_FRAC` | `diverse_enclosing_frac` | 1.0 | 姿态库 vs pinch 复位比 | 0.5 | — (=0) |
| `SHARPA_FORCE_SCALE`/`_PROB` | `force_scale`/`random_force_prob_scalar` | —/0.2 | 物体力扰动 | 2.0/0.2 | 2.0/0.2 |
| `SHARPA_EPISODE_S` | `episode_length_s` | 20 | 回合长度 | 40 | 40 |
| `SHARPA_PC` | `enable_pointcloud` | False | 点云分支 | — | 1 |
| `SHARPA_NUM_POSES` | `graspxl_num_poses` | 3 | 每物体起始姿态数 | — | 40 |

（还有一堆被超越/关闭的杠杆：`SHARPA_BAND_N4`/`_LOWN`/`_HOLD_INC`/`SHARPA_BACKSLIP_W`/`SHARPA_BAND_ROTSTREAK` 两个冠军都不用。）

### 2.7 六条设计法则 → 代码映射（`docs/FULL_REPORT...md:91-111`）

1. **定价即命运**：每个观察到的均衡（2指转、死握、层camp、突发甩飞）都是完整收入栈的正确最优。连接型目标（握×转）需**结构性**定价而非调数值：任何对瞬时速度单调的付款都有边界最优（camp 或 fling）。→ **带通同时带下限0.06和上限1.2**。还有居中年金 3× 于旋转的修复（`CENTER_SCALE=0.3`）。
2. **占用律**：旋转速度随接触数单调升，n=max 时完美定向。→ **连续 n 缩放 `r_rot·n/3`**。
3. **搅动律 + 收割**：快速盆地在持续探索训练下是终末快照（5/5 自毁）；扰动训练是例外。→ **物体力扰动 `FORCE_SCALE=2.0`** + 快照扫描选型。
4. **训练reward≠选型**：reward 128→372 上升同时真实持握质量 0.88→0.09 崩溃。→ 只信持握门控确定性 probe。
5. **评估窗右删失**：用 ≥2–3× 回合上限的窗评估回合级目标（训练40s评估120s）；同一 ckpt 因窗不同得 10.1 vs 3.9 rad。→ `EPISODE_S=40` 训练 / 120 评估。
6. **复位多样教恢复**：pinch混合复位使耐久翻倍。→ **`ENC_FRAC=0.5`**：半数回合从原始 pinch 起，逼迫抓取构建+中途座位恢复。这也是策略会做"调整"而非只旋转的原因。

---

## 3. 复位 / 姿态库 / 重力课程 / 扰动

### 3.1 复位分布（`SHARPA_ENC_FRAC=0.5`）

每 env 独立伯努利抽（`adjustrotate_diverse.py:102`）：`enc = ids[rand < 0.5]`。

| 复位类型 | 冠军比例 | 设置什么 |
|---|---|---|
| **PINCH（replay）** | ~0.5 | 从数据集抓取轨迹回放（`graspxl_env.py:310-359`）；settle 若干步再交给 RL；指尖捏 |
| **ENCLOSING（"adjusted"）** | ~0.5 | 直接从姿态库缓存行写入 22 关节+物体位姿（`adjustrotate_diverse.py:111-166`）；无 settle，立即 RL 控制 |

**姿态缓存格式 `(N,29)`**（`adjustrotate_diverse.py:79`）：**列序 = [0:22] 22关节 · [22:25] 物体位置(env相对) · [25:29] 物体四元数(wxyz)**（磁盘上是**关节在前**）。

**ENCLOSING 写入核心**（`adjustrotate_diverse.py:111`）：
```python
rows = enclosing_cache[randint(0, N, (m,))]
dof=rows[:,:22]; obj_pos_rel=rows[:,22:25]; obj_quat=rows[:,25:29]
object.write_root_pose_to_sim([obj_pos_rel+origins, obj_quat])   # 零速度
object_default_pose[enc,:3] = obj_pos_rel                        # 居中/掉落参考 = 此姿态
hand.write_joint_state_to_sim(dof, 0); prev_targets=cur_targets=dof
_gx_settle[enc]=0                                                # 立即 RL 控制
```
冠军 `SHARPA_POSE_CACHE=poses/n4_pose1.npy` 是 **(1,29)** 单个 5 指包络抓取 —— 所有 enclosing env 都从这一个姿态复位。

### 3.2 姿态库 / 驱动库 / cache 文件（均 `(N,29)` 同布局）

| 文件 | shape | 含义 |
|---|---|---|
| `poses/n4_pose1.npy` | (1,29) | 5指 n≥4 包络抓取（`regrip_pose_bank.py` 从 R4RLN4 策略收割）→ **冠军复位** |
| `poses/drive_bank/pose{1,2,3}.npy` | (1,29) | 近旋转"驱动"态（+7.3rad/s@n3）→ 逆课程复位种子 |
| `cache/graspxl_enclosing_<obj>.npy` | (2807,29) | 掉落静置抓取生成的包络抓取（`SHARPA_POSE_CACHE` 未设时的默认） |
| `cache/graspxl_adjusted_<obj>{,_v2,_v3a,_perturb}.npy` | (199~256,29) | 从 regrip 提取的"调整"抓取 |
| `cache/graspxl_handoff_<obj>.npy`+`_meta` | (105,29)+(105,4) | ≥3指交接态 |
| `cache/sharpa_grasp_linspace_0.5-0.5-1.npy` | (50000,29) | 圆柱 pinch 缓存（基础任务用，非冠军） |

### 3.3 重力课程

基础 −0.05 起（`sharpa_wave_env_cfg.py:69`），`gravity_curriculum=True`(`:330`)，**掉落率门控、每次 −0.05、封顶 −10**。冠军走位移路径（`graspxl_env.py:293-299`）：
```python
rotate_ok = extras["rotate_reward"] > -1.0            # BandRot 门=-1.0 → 实际只看掉落率
if drop_rate < 0.01 and rotate_ok and gravity_curriculum and step>1000:
    if |g|<10: set_gravity(0,0,-|g|-0.05)
```
`graspxl_curr_drop_thresh=0.01`（松，边转边升）。
> **历史**：纯 pinch 的 3A 在 g≈−0.40 卡住（pinch 撑不住重力→门永不开）；`ENC_FRAC=0.5` 混合让包络 env 撑得住，门才推进。**评估固定 g=−9.81** 因课程状态不在 ckpt 里（probe: `env_cfg.sim.gravity=(0,0,-9.81); gravity_curriculum=False`）。

### 3.4 物体力扰动（`force_scale=2.0`）

每 `_pre_physics_step`（`sharpa_wave_env.py:245-252`）：~20% env 获得新高斯力 `randn·mass·2.0`，两次刷新间按 `0.9^(dt/0.08)` 衰减，复位清零。HORA 式随机物体力矩 —— 鲁棒边界/防掉落课程（day3_perturb：回合寿命≈翻倍、掉落减半、优化更稳）。**probe 关掉。**

---

## 4. 算法 & 网络架构

栈 = HORA/RLGames 派生 PPO + "v3net" 非对称 actor-critic。冠军 ckpt 是 **裸 stage-1 PPO**（含 `env_mlp/actor_mlp/critic_mlp/mu/value/sigma`，无 `adapt_tconv/pc_encoder/wm_head`）。

### 4.1 PPO 超参（`agents/ppo_cfg_v3net_p8.yaml`，`algo/ppo/ppo.py` 消费）

| 超参 | 值 | yaml行 |
|---|---|---|
| network.mlp.units（actor&critic干） | **[512,256,128]** | :19 |
| network.priv_mlp.units（外因嵌入） | **[256,128,8]** | :21 |
| separate_critic | True | :22 |
| actor_priv_dim | 8 | :23 |
| sigma_floor | **0.1** | :24 |
| normalize_input/value | True/True | :30-31 |
| value_bootstrap | True | :32 |
| **gamma γ** | **0.99** | :35 |
| **tau (GAE λ)** | **0.95** | :36 |
| **learning_rate 初始** | **5e-3**（自适应KL后） | :37 |
| **kl_threshold** | **0.02** | :38 |
| **horizon_length** | **32** | :40 |
| **mini_epochs** | **5** | :42 |
| critic_coef | 1（loss 里再 ×0.5） | :45 |
| **entropy_coef** | **0.001** | :46 |
| **e_clip** | **0.2** | :47 |
| bounds_loss_coef | 0.0001 | :48 |
| grad_norm | 1.0 | :51 |
| priv_info_dim | **8** | :58 |

- **minibatch** 被 `train.py:72` 覆盖为 `min(num_envs*8, 32768)`。
- **LR 调度**：自适应 KL（`ppo.py:436-449`）—— `KL>2·kl_thr`→`lr/1.5`；`KL<0.5·kl_thr`→`lr·1.5`，夹 [1e-6,1e-2]。
- **优势归一化**：整批 `(adv−mean)/(std+1e-8)`（`experience.py:116`）。
- **奖励缩放**：`shaped = 0.01·rewards`（`ppo.py:385`）+ 超时值自举。
- **GAE**：标准 `δ=r+γV'·nonterm−V`，`A=δ+γλ·nonterm·A'`。

### 4.2 网络（"v3net"，`algo/models/models.py:72-206`）

冠军维度（对 ckpt 核对）：obs=195，priv=8，action=22，PC/WM 关。
- **外因嵌入 `env_mlp`** = MLP([256,128,8], in=8) → forward 末端 `tanh`。
- **actor 干 `actor_mlp`** = MLP([512,256,128], in=195+8=**203**) → `mu=Linear(128→22)`。
- **分离 critic 干 `critic_mlp`** = MLP([512,256,128], in=195+8=**203**，用**原始** priv) → `value=Linear(128→1)`。
- **`sigma`** = `Parameter(zeros(22))`，**状态无关学习 log-std**，forward 夹下限 `log(0.1)`。
- 激活全 ELU；采样 `Normal(mu, exp(logstd)).sample()`；推理返回 `mu` 夹 [-1,1]。

**forward 数据流**（`models.py:153-186`）：
```python
obs=obs_dict['obs']; obs_raw=obs
extrin = tanh(env_mlp(priv_info[...,:8])); obs=cat([obs,extrin])   # 203
# (PC 分支：use_pc 时 cat pc_encoder(pointcloud) 128 维)
x = actor_mlp(obs)                                                 # 128 latent
xc = critic_mlp(cat([obs_raw, priv_info])); value = value(xc)      # 分离 critic
mu = mu(x); logstd = clamp(sigma, min=log(0.1))
```

**PointNet 分支**（冠军不用，`models.py:28-42`）：逐点 MLP(5→64→128→128) + **点轴 max-pool**（置换不变）→ 128 维拼到 actor 干输入。点云 `(N,P,5)`=xyz(3)+mask(2)。
**ProprioAdapt（stage-2 蒸馏，冠军不用）**：`adapt_tconv` 时序卷积从 30 帧本体感觉历史回归出 8 维外因，蒸馏损失 `mean((tanh(tconv) − tanh(env_mlp(priv)).detach())²)`，Adam lr 3e-4，只训 tconv。评估 `--algorithm ProprioAdapt` 需额外加载 `sa_mean_std`。

**ckpt 格式**（`ppo.py:233-241`）：`{'model':..., 'running_mean_std':..., 'value_mean_std':...}`。obs 归一化 RunningMeanStd（Welford，float64，`y=clamp((x−μ)/√(var+1e-5), −5, 5)`），评估冻结。

### 4.3 PPO 更新（`ppo.py:305-343`）

```python
ratio = exp(old_logp − logp)
a_loss = max(−adv·ratio, −adv·clamp(ratio,1−0.2,1+0.2))          # PPO clip
c_loss = max((V−ret)², (V_clipped−ret)²)                         # value clip 用 e_clip
b_loss = clamp_max(mu−1.1,0)² + clamp_max(−mu+1.1,0)²            # 边界损失
loss = a_loss + 0.5·c_loss·critic_coef − entropy·0.001 + b_loss·1e-4
clip_grad_norm_(params, 1.0)
```

### 4.4 包装器

- **GymStyleEnvWrapper**（`wrapper/sharpa_wave_env_wrapper.py`）：IsaacLab DirectRLEnv → RLGames API。动作夹 [−1,1]；`obs_dict['obs']=obs_dict['policy']`；`time_outs=truncated` 供无限时域值自举；`num_obs=195, num_actions=22, prop_hist_len=30`。
- **ConfigWrapper**：容器，暴露 `.train/.task/.test`。

---

## 5. 训练流程 & 模型选择

### 5.1 冠军训练命令（`train_pipeline_champion.sh`，~2h RTX4090，24M 步）

```bash
SHARPA_SELF_COLLISION=1 SHARPA_POSE_CACHE=poses/n4_pose1.npy \
SHARPA_BAND_HELD=1 SHARPA_BAND_NSCALE=3.0 SHARPA_BAND_FLOOR=0.06 SHARPA_BAND_SCALE=8.0 \
SHARPA_BAND_CEIL=1.2 SHARPA_CENTER_SCALE=0.3 SHARPA_BAND_EMA=0.85 \
SHARPA_SPARSE_THETA=0.15 SHARPA_SPARSE_BONUS=2.5 SHARPA_FORCE_SCALE=2.0 \
SHARPA_ENC_FRAC=0.5 SHARPA_EPISODE_S=40 \
python rl_rebuild/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-Net-v1 \
  --max_agent_steps 24000000 --num_envs 768 --headless
```
**从零训练**（无 `--resume`）。**pinch→adjust→sustained-rotate "流水线"概念**：50% 原始 pinch 复位逼策略先把捏姿*调整*成支撑/包络抓取、再持握*旋转*、并中途*恢复座位*；另 50% 姿态库直接搭建旋转技能。这就是"调整"能力（不只旋转）的来源，把突发旋转变成 ~90s 耐久。

**stage-1 vs stage-2**：`train.py` 按 `agent_cfg["algo"]` 用 `eval()` 分派。stage-1=`PPO`（冠军，ckpt 存 `logs/<exp>/<ts>/stage1_nn/ep_*.pth` 每 100 iter）；stage-2=`ProprioAdapt`（蒸馏，加载 stage-1，强制关课程）。

### 5.2 模型选择（**训练reward≠质量**）

1. **`snapshot_probe.py`**：一次启动，restore 每个 `ep_*.pth`，用**持握门控**指标打分，取 **argmax**（绝不用 `last.pth`/wandb reward）。持握定义 `held := active & ~done & started & (n_eng≥2) & (disp<0.05)`。
2. **评估窗 ≥2–3× 回合上限**（`--eval_steps 2800`≈120s，训练 40s）——否则右删失掉正好要奖励的长回合。
3. **确定性**（`mu=clamp(act_inference,-1,1)`）、DR 关、g 固定、课程关。
4. `rot_held_probe.py` 额外报**物理净旋转**（+34.4 口径）和**制动曲线**（ω 按瞬时接触数分箱，证明 n=4–5 是驱动模式）。
> **搅动律**：快旋盆地是终末快照，σ-0.1 续训会毁掉 —— probe 一显示快盆地就立刻存档，只靠更好定价的新draw改进。

---

## 6. 改造成你自己的策略 —— 实操指南

这一节是给你"对齐修改"用的。先分清**结构性设计**（乱动会塌）和**可调旋钮**（安全实验）。

### 6.1 什么绝对不要乱动（结构性，动了整套逻辑就变）

- **带通"下限+上限"双约束的存在**（不是具体数值，是"两端都有边界"这件事）。这是"定价即命运"的核心：任何对速度单调的付款都会收敛到 camp 或 fling。你可以调 floor/ceil 数值，但**不能只留一端**。
- **持握门控 `held_gate` 对 band 和 sparse 的乘法**。去掉它 → 逃逸自旋刷分，尤其低重力/大扰动下。
- **分离 critic + actor 特权嵌入瓶颈**（非对称值函数）。这是可部署性（HORA蒸馏）的前提。
- **复位多样性**（pinch + 包络混合）。纯包络会学不到"调整/恢复"，纯 pinch 会卡在低重力（撑不住课程）。
- **模型选择用 held-gated probe argmax，评估窗 ≥2–3× 回合**。这条错了你会选出假冠军。

### 6.2 想换目标时改哪里（按意图查表）

| 你想要 | 改什么 | 位置 |
|---|---|---|
| 换旋转轴（如绕 x/y） | `SHARPA_ROT_AXIS="1,0,0"` | `posebank.py:154-163`（同时改 reward 轴与复位轴） |
| 换物体 | `SHARPA_OBJ_ID=<id>` 或 `SHARPA_OBJ_IDS=a,b,c`（多物体会**稀释**单物体深度，且强制纯 pinch 复位） | `posebank.py:70-96` |
| 更快/更慢的目标转速 | 调 `SHARPA_BAND_CEIL`（上限=可控转速上界）和 `SHARPA_BAND_FLOOR` | `posebank.py:43-49,198-209` |
| 更强持握（少掉落） | 提 `SHARPA_CENTER_SCALE`、加 `SHARPA_DROP_PENALTY`、提 `min_fingers`；但注意会压旋转 | `posebank.py:169-172,210-216` |
| 换成"净进度"口径（物体泛化） | 开 `SHARPA_NETPROG_W`、`SHARPA_CENTER_SCALE=0`、脚手架退火 `SHARPA_BAND_SCALE_END` | `posebank.py:173-184` |
| 加视觉/点云条件 | `SHARPA_PC=1` + 换 agent yaml 到 `ppo_cfg_v3net_p8_pc.yaml` | `posebank.py:80-83` |
| 改探索强度 | agent yaml 的 `sigma_floor`(0.1)、`entropy_coef`(1e-3) —— 这俩是当年破 camp/burst 盆地的关键杠杆 | `ppo_cfg_v3net_p8.yaml:24,46` |
| 改控制频率/精细度 | `decimation`、`action_scale`(1/24) | `sharpa_wave_env_cfg.py:60,63` |
| 改回合/评估长度 | `SHARPA_EPISODE_S`（训练）、probe `--eval_steps` | `posebank.py:114-117` |

### 6.3 加你自己的奖励项（推荐做法）

模式已经很清晰：在 `_BandRotMixin._get_rewards`（`bandrot.py:~232-282`）里，`total = super()._get_rewards() + 你的项`，每项用一个 `SHARPA_*` env-var 守卫（默认关，保持旧行为不变），在 `posebank.py.__post_init__` 里读取。**务必**：
- 新项若涉及"持握 vs 未握"，复用现成的 `held_gate` / `disp` / `n_eng` 几何，别自己重算一套阈值。
- 新的逐 env 累积量（像 `_phys_cum`）要在 `_reset_idx`（`bandrot.py:294-298`）里清零。
- 加完先用 `scripts/bandrot_reward_unit_test.py` 那种离线单测验证：常速抖动→0、单调、方波→0、棘轮只在净前进付款。

### 6.4 常见陷阱（作者踩过的）

1. **单调速度奖励必塌**：加任何"转得快就多给"的无界项 → camp 或 fling。要嘛带通，要嘛净进度高水位。
2. **无界居中 `1/(disp+ε)`**：死握时爆到 ~400，压死旋转。用有界 `exp(−disp/σ)` 且权重 ≤0.3。
3. **零重力/强扰动下的逃逸自旋**：没有 held_gate 时，滚落的球累积 ω 刷分。守卫必须门控在有物理意义的区间。
4. **用训练 reward 选 ckpt**：reward 会一路涨而真实质量崩。只信 probe。
5. **短评估窗**：右删失长回合，同一 ckpt 分数能差 2–3×。
6. **多物体训练稀释**：`137c1d2f` 在 6 物体 run 里从 34.4 掉到 1.1（预算/干扰效应，不是配置错）。单物体先做深，再考虑泛化。
7. **质量 DR 关着**：`priv[4]`（质量）恒 0 —— 若你开 `randomize_mass`，记得这维才有信息。

### 6.5 建议的改造流程

1. **先复现**：跑通冠军 probe（见 §5.2 命令），确认 baseline 数值/行为。
2. **一次只动一个旋钮**：改一个 `SHARPA_*`，从零训一个短 run（几 M 步），用 snapshot_probe 扫描看曲线，别信最后一个 ckpt。
3. **改奖励用守卫式 env-var**：默认关，能随时回退到冠军配方。
4. **结构性改动（换 obs/网络/复位机制）要重训**，且注意 obs 维度变化会让旧 ckpt 不兼容（冠军 obs=195）。
5. **选型纪律不能省**：held-gated probe + 长窗 + argmax。

---

## 附：关键文件索引

| 主题 | 文件 |
|---|---|
| 基础 env / obs / action / 物理 / DR | `rl_rebuild/tasks/inhand_rotate/sharpa_wave_env.py`, `sharpa_wave_env_cfg.py` |
| GraspXL replay 复位 / 重力课程 / 力扰动 | `.../sharpa_wave_graspxl_env.py` |
| 包络姿态库复位 / ENC_FRAC 混合 | `.../sharpa_wave_graspxl_adjustrotate_diverse.py` |
| 带通奖励系统（核心） | `.../sharpa_wave_graspxl_bandrot.py` |
| 全部 SHARPA_* 旋钮读取 | `.../sharpa_wave_graspxl_bandrot_posebank.py` |
| PPO / 经验缓冲 / GAE | `rl_rebuild/algo/ppo/ppo.py`, `experience.py` |
| v3net 网络 / PointNet / TConv | `rl_rebuild/algo/models/models.py` |
| 超参 | `rl_rebuild/tasks/inhand_rotate/agents/ppo_cfg_v3net_p8.yaml` |
| 训练入口 / stage1-2 | `rl_rebuild/scripts/train.py` |
| 选型 probe | `rl_rebuild/scripts/snapshot_probe.py`, `rot_held_probe.py` |
| 离线奖励单测 | `rl_rebuild/scripts/bandrot_reward_unit_test.py`, `d0_reward_unit_test.py` |
| 训练脚本 | `train_pipeline_champion.sh`, `train_minimal_design_netprog.sh` |
| 数据集加载 / (N,29) 格式 | `rl_rebuild/graspxl/sharpa_dataset.py` |
| 六设计法则 / run 账本 | `docs/FULL_REPORT_2026-07-02_to_07-05.md`, `docs/RUN_LEDGER_2026-07-03_to_07-05.md` |

> 一处已知微妙点：stage-2 ProprioAdapt 的 TConv 用 `input_shape[0]//3 = 195//3 = 65` 作为每帧维度，而基础本体感觉帧其实是 64（195 = 192 + 3 手系重力）。冠军是 stage-1，不走 TConv，所以不受影响；但你若做 ProprioAdapt 蒸馏，注意这个 64/65 的切分。
