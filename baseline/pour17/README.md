# Pour17 Baseline Bundle（给 adapted H2S2R）

目标：让 baseline 在**与 ours 相同的 simulator / robot / scene / reset / evaluator** 下重训，
但保留它自己的 observation、action、reward、FABRICS controller。

本目录只装 **baseline 需要的**东西。ours 专属的 `eval_best.pth`、DP 348D 数据、
GraspPose、CuRobo guide、confidence reward、503D obs、58D residual action **一律不放**。

**状态：世界+感知+evaluator 已齐（2026-08-29 与 RL session 对齐后补全）。**
**USD 自包含性已验证通过（见 §2.1），无需 flatten。**
**统一评测器已写好并自检通过（见 §4.1），只等 baseline 接两个回调。**
本 bundle 四块（world / perception / reference / evaluator）**已全部就位**。
**母带已定版：`2ed81358`（对齐 ours 当前训练）—— 见 `reference/TAPE_DECISION.md`。**
**起跑线已冻结：`world/canonical_reset_v1.json`（horizon 903 步 / 初始 q/qd / 物体位姿 / G2 认证规范）。**

### 2026-08-29 第二轮（按 baseline 侧验收反馈修）

| 问题 | 处理 |
|---|---|
| horizon 三个数并存 | ✅ **冻结 903 步**（`int(653×1.2)+120`）。`episode_length_s=13.125` 对 Pour17 **无效**，已在 manifest 与文档标注；**评测器自己强制上限，不看 baseline env 的 `truncated`** |
| 缺 canonical reset | ✅ `world/canonical_reset_v1.json`：58 关节初始 q（母带行 0，已自验与末行逐关节差 **0.000°**）+ qd 全零 + 两物体位姿/零速度 + 无随机化 + root pose 处置 |
| **G2 无法通过** | ✅ 新增第三个回调 `env.apply_certification_offset(alpha)`，**由评测器发起**。规范口径定为**笛卡尔**"双腕世界 +Z 抬 15mm、姿态不变"，ours 的关节空间实现已 FK 反算验证等价（右腕 +14.65mm / 左腕 +13.81mm，姿态变化 0.1° 级） |
| 焊接热身未披露 | ⚠ ours 训练默认 `POUR_HOLD_K=15`（前 15 步钳位物体）。**评测协议建议设 0**，★需两边确认 |
| `pour_env.py.reference` / `REWARD_DOC.md` 与红线矛盾 | ✅ **已撤出**（含 reward 权重与残差界分档，等于附送配方） |
| `self_collision` 记录不可信 | ✅ 已重导。**真值 `False`**（来源 `USD:/World/envs/env_0/Robot/root_joint`，本 bundle 用 `pxr` 独立复核过 authored 值）。⚠ 注意 `correction_env_cfg.py:121` 写的 `enabled_self_collisions=True` **对本任务未生效** —— 三处都叫 self_collision，只有 USD 那个是 PhysX 真读的。原来那个 `False` 是采集器默认值**蒙对的**，故新增 `self_collision_source` 字段：**值对不等于记录可信** |
| `progress.py` 里含 ours 奖励常量 | ⚠ **未解决，待拍板**。撤掉 `REWARD_DOC.md` 只是换了容器 —— `W_OBJ`/`W_HAND`/`LEASH_*`/`MS_REWARD`/`WAGE*` 原样在评测器必需的 `progress.py` 里。当前按 **(a) 显式声明**处理（见 `evaluator/INTEGRATION.md` 顶部），终态是 **(b) ours 侧拆成 `criteria.py` + `reward_weights.py`**。不做 (c) 裁剪版：会破坏唯一真源 |
| 率值无分母 | ✅ `eval_result.json` 里任何"率"都带 `*_denominator`，分母为 0 给 `null` 不给 `0.0`（RL session 曾因此把"本窗零回合"误判成"成功率 0%"） |
| UTF-8 / 文件数 | ✅ `read_text/write_text` 全部显式 `encoding="utf-8"`；文件数以 `HASHES.txt` 为准 |

## 🔴 拿到 bundle 先看这四条（都是实测纠正，不看会复现错）

1. **机器人 USD 要换一个：合作方点名的 `stance0803` 是别的任务的世界，与 Pour17 无关。**
   - 运行时实际加载 `vega_1p_sharpa_fixedtorso.usd`（**md5 143385ad…**）；
   - `..._stance0803.usd`（md5 f77f235d…）是 **AAG-F 那条线的"出生世界"存档**：
     8/26 主 USD 换过站姿后，AAG-F 的 ckpt 回放直接 0/64（而 obs 维度一字没变 348 ——
     "同维异义"事故），于是把旧站姿另存并加 `DEXMATE_FIXED_USD` 覆写让它能回放。
     **Pour17 从来没用过它**，只有 `docs/DP_AAGF_RUNBOOK.md` 引用。
   - ⟹ 它**不是"Pour17 的旧版世界"**，别当同一条线的历史版本看。复现 Pour17 baseline
     **唯一正确选择是 `__RUNTIME_LOADED`（143385ad）**。两份都在 `world/`，按后缀区分。
2. **瓶和杯的物理材质差 6 倍摩擦**，且加载路径不同：
   | | mass | friction | restitution | 加载的 USD | 角色 |
   |---|---|---|---|---|---|
   | 瓶 object_1 | **0.530 kg** | **3.0 / 3.0** | 0 | `objects/object_1.usd` | primary（运行时刚体化，材质被覆写成 3.0） |
   | 杯 object_0 | **0.150 kg** | **0.5 / 0.5** | 0 | `cache/object_0.usd` | aux（烘焙物理，128 hulls + shrink_wrap） |
   ⚠ `clips.py` 里 ObjectSemantics **声明**瓶 friction=0.5，**运行时实际是 3.0** —— 以
   `world_manifest.json` 的运行时值为准，不要读声明值。
3. **物理桌面只有一个数：`table_top_z = 0.87`**。`scene_layout.json` 里的
   `rl_table_height=0.85` / `scene_table_z=0.8638` / `shift=0.0138` 是**重建侧摆放阶段**的量，
   **不进物理**。⚠ `evaluator/progress.py` 里的 `TABLE_Z=0.87` 是**独立硬编码副本**，
   与 cfg 无代码联动 —— **改桌高必须两处同步改**。
4. **母带（reference v2）今天换代了，而 evaluator 会读母带**（`_mouth_local()` 从
   `obj_quat_*` 取瓶口/杯口方向定义 G3）⟹ **母带版本会改变判据本身，必须钉版**。
   见下方 §3。

---

## ⚠ 首要更正：v1/v2 都不是"方法中立"的

原需求写「v1 可用于核对原始视频轨迹和输入转换」——**这个前提不成立**。实测字段：

| 字段 | v1 (515 行) | v2 (543 行) |
|---|---|---|
| `right_q` / `left_q` | **(T,7) 机器人臂关节角**（ours IK 结果） | 同（float64） |
| `right_f` / `left_f` | (T,22) Sharpa 手指 | 同 |
| `human_right_q` / `human_left_q` | **无** | **(T,7) 有** |
| 人手腕位姿 / MANO | **无** | **无** |
| `conf_pos_*` / `conf_rot_*` | 有（ours 置信度） | 有 |

即 **v1 的 ours 污染比 v2 更彻底**：它连人侧原始量都没保留。两者都不能当"原始感知数据"。

**→ 真正方法中立的源头是 `perception/`（重建产物 + 原始视频），不是 `reference/`。**

---

## 1. `perception/` —— 方法中立的感知数据 ✅ 完整

| 文件 | 内容 |
|---|---|
| `pour17_raw_video.mp4` | 原始 EgoDex demonstration（`pour/17.mp4`，30fps） |
| **`pour17_perception.npz`** | 下表字段，米制 / wxyz / left→object_0(杯) right→object_1(瓶) |
| `world_fused.npz` | 重建全量（相机 K/c2w、双手 MANO、两物体 6DoF、对齐元数据） |
| `object_valid_measured.npz` | 逐帧"真话版"物体可信标志（判据 `ovm_v1_20260826`） |
| `conf_pour_17.mp4` | 可信度叠加视频（核对重建贴不贴观测） |
| `recon_render_pour_17.mp4` | 纯重建渲染（"RL 眼中的场景"） |
| `make_pour17_perception.py` | 生成脚本（可复现） |

`pour17_perception.npz` 字段（T=142）：

```
frame_ids (142,)          timestamps (142,)  ← 全 NaN，见下方 unavailable 说明
K (3,3)                   c2w (142,4,4)      ← 设备标定/SLAM，非估计
left/right_wrist_pose_wxyz (142,7)           ← 位置+四元数 wxyz
left/right_hand_q (142,22)                   ← Sharpa DexPilot 重定向，物体无关
hand_joint_names (22,)
object_0_pose_wxyz (142,7)   object_1_pose_wxyz (142,7)
left_valid / right_valid / object_0_valid / object_1_valid / *_scored (142,)
object_valid_criteria                        ← 判据版本串
# MANO 备份路线（想自行 retarget 就用这组，绕开我们的 DexPilot）
left/right_mano_{trans,rot,pose45,betas,valid}
meta                                         ← JSON：单位/约定/坐标系/各字段来源
```

**明确标注 unavailable 的字段**：
- `timestamps`：EgoDex 不提供绝对时戳，只能由 `frame_ids / fps` 推导；
- **两个时基**：源视频 30fps，重建产物 **15fps / 142 帧**（`meta` 里都写了）。

**已满足的约定**：位置米制 ✅ 四元数 wxyz ✅ left→cup/object_0、right→bottle/object_1 ✅
（后者与 `world/scene_layout.json` 里 `object_0.anchor_hand: "left"` 一致）。

## 2. `world/` —— 冻结的仿真世界 ✅ 资产 + 环境指纹齐（仅缺 flatten）

| 文件 | 状态 |
|---|---|
| `vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd` | ✅ **训练实际加载的这个**（24MB，md5 143385ad…）。**已验证自包含** |
| `vega_1p_sharpa_fixedtorso_stance0803__NOT_USED_BY_TRAINING.usd` | ⚠️ 合作方点名要的，但**代码零引用**，仅存档对照 |
| **`world_manifest.json`** | ✅ **环境指纹**，运行时 dump（生成器 `export_world_manifest.py` 可重跑） |
| `objects/object_{0,1}.usd` + `objects/object_{0,1}/object_mesh_scaled_final.obj` | ✅ |
| `cache/object_{0,1}.usd` + `cache/config.yaml` + `.asset_hash` | ✅ **只有杯(aux)用这版**烘焙物理；瓶(primary)用 `objects/`。见上方第 2 条 |
| `scene_layout.json` | ✅ 两物体静置位姿/朝向理由/尺寸。⚠ 里面的三个桌高数**不进物理**，见上方第 3 条 |
| `DEXMATE_ARTICULATION_JOINTS__STALE_67line.txt` | ⚠️ **67 行与运行时对不上**（运行时 articulation 就是 58 关节且全受控），已改名标记，**别用**；以 manifest 的 58 名有序表为准 |
| `DEXMATE_JOINTS.md` / `DEXMATE_BODY_POSES.json` | ✅ body 参考 |
| `world_json_samples/*.world.json` | ⚠️ 只有 8 个键，**不是环境指纹**；已被 `world_manifest.json` 取代，仅存档。⚠ 里面 `usd:"default"` 的意思是**"没设 DEXMATE_FIXED_USD 覆写 ⟹ 加载了默认那份 `vega_1p_sharpa_fixedtorso.usd`"**，不是"没用 USD"——该字段只记有无覆写、不记路径，信息量不足 |

### 2.1 USD 自包含性 ✅ 已验证（2026-08-29）

用 `flatten_usd.sh` + `flatten_usd.py`（本目录，可重跑）对全部 5 个 USD 做依赖分析：

| USD | 外部 layer | 资产引用 | prims / joints / massAPI / collisionAPI | 结论 |
|---|---|---|---|---|
| `vega_..._RUNTIME_LOADED.usd` | 0 | 0 | 1557 / 104 / 104 / 108 | ✅ 自包含 |
| `objects/object_0.usd`（杯，视觉） | 0 | 0 | 2 / 0 / 0 / 0 | ✅ |
| `objects/object_1.usd`（瓶，primary 加载） | 0 | 0 | 2 / 0 / 0 / 0 | ✅ |
| `cache/object_0.usd`（杯，aux 加载，烘焙物理） | 0 | 0 | 3 / 0 / 1 / 1 | ✅ |
| `cache/object_1.usd` | 0 | 0 | 3 / 0 / 1 / 1 | ✅ |

**全部零外部依赖，不需要 flatten。** 两点说明：

1. 机器人 USD 有 **1 个未解析引用 `OmniPBR.mdl`** —— 那是 Omniverse **标准材质库**里的
   shader，Isaac 自带，不是本地缺文件，且**只影响渲染外观、不影响物理**。
   合作方只要用 Isaac Sim 就有；换渲染器则外观会变、物理不变。
2. ⚠ **关节数有三个数并存，不解释一定有人搞错**：
   | 来源 | 数量 | 含义 |
   |---|---|---|
   | `DEXMATE_ARTICULATION_JOINTS__STALE_67line.txt` | 67 | **过期表，别用** |
   | 机器人 USD 里的 joint prim | 104 | 含固定/被动关节 |
   | **运行时 articulation DOF** | **58** | **受控关节，以此为准** |
   以 `world_manifest.json` 的 58 名有序表为唯一真源。

⚠ 这只是**结构自检**，不能替代物理一致性验证。要严格证明，需在 Isaac 里跑零动作回放对拍
（四个出生点的累计奖励与终止步逐位一致，口径见 ours 的 `smoke_zero.py`）——
但既然零外部依赖、文件逐字节就是训练在用的那份，这一步的必要性已大幅下降。

### `world_manifest.json` 关键值（全部运行时读出）

```
时间     physics_dt = 1/240 s   control_dt = 0.05 s (20 Hz)
         decimation = 12        render_interval = 2      episode_length_s = 13.125
桌子     table_top_z = 0.87     size (1.2192, 1.8288, 0.04)   friction 0.5/0.5
机器人   articulation 58 关节, 全部受控      self_collision = False
         顺序 [R_arm_j1..j7, L_arm_j1..j7, right 22 指, left 22 指]（完整名单+软限位在 manifest）
开关     approach_only=True   obj_jitter_xy=0.0（无位置随机化）
         friction_curriculum=False（hi=6.0/lo=3.0 定义了但未启用）   POUR_NO_D6_effective=false
场景     env_spacing = 2.0 m   replicate_physics = False
```

**11 个接触传感器**（不是 10 个）：
- 10 个指垫 `{right,left}_{thumb,index,middle,ring,pinky}_elastomer`，
  `history_length=1`、`update_period=0.0`、力阈 `PAD_FTH=0.5N`、`PADS_MIN=3`；
  **右手 5 个 filter → `/World/envs/env_.*/Object`（瓶）；左手 5 个 → `/World/envs/env_.*/Aux`（杯）**
- 第 11 个 = **D6 跨侧互撞**：右侧远端外壳
  `(R_arm_l5|R_arm_l7|R_arm_l8|right_hand_C_MC|right_.*_elastomer)`
  × 左侧同名 5 条，判力不判距；训练默认装（`POUR_NO_D6` 未设）。

⚠️ `eval_best.pth` 对应的最新 run（`logs/E2EL_pour17/2026-08-26_*`）没有 `world.json`，
只有 `auto_stop.json` —— 所以**环境指纹以 `world_manifest.json` 为准**，不要用 world.json。

## 3. `reference/` —— ✅ 实体文件，非 LFS 指针

**已按 md5 钉版**（因为 evaluator 会读母带 → 母带版本改变判据本身）：

| 文件 | 行数 | 交互段 | 说明 |
|---|---|---|---|
| `pour17_reference_v1__9f2dbeb6__515rows.npz` | 515 | — | v1 母带 |
| `pour17_reference_v2__b95546e9__543rows.npz` | 543 | 163 | **旧版**：现有 `logs/` 里 eval_best 那批 run 用的 |
| `pour17_reference_v2__2ed81358__653rows.npz` | 653 | **273** | **新版（2026-08-29 换代）**：ours 当前训练在用，`world_manifest.json` 指向它 |

新版改动（RL session 说明）：修了重建朝向噪声（倒水段 conf_rot 全红却被照单全收 →
IK 被翻译成 72°/帧 的关节跳）+ 腕奇异 + 时间扩张，交互段 163→273 行。

### ✅ 已定版：`2ed81358`（2026-08-29 拍板）

**权威母带 = `pour17_reference_v2__2ed81358__653rows.npz`**，理由与另两版的地位见
**`reference/TAPE_DECISION.md`**。`run_eval.py` 默认值与 `world_manifest.json` 均指向它，
自检会核对两者一致性。

另两版仅作存档：`b95546e9`（现有 eval_best 那批 run 用的旧版）、
`9f2dbeb6`（v1，倒水段有 42°+72°/帧的臂角跳变，**不要用来训练**）。

⚠ **换母带 = 换判据**（evaluator 从母带取瓶口/杯口方向与静置位姿），
所以报告里必须写明用了哪一版。

**用途（按上文更正后）**：
- 两者都**只能**用于场景 / 时序 / evaluator 对齐与 debugging oracle；
- 都**不能**用于"核对原始视频轨迹"——那要用 `perception/`；
- 训练时**不要读**：`right_q`/`left_q`/`right_f`/`left_f`（ours IK+重铸）、
  `conf_pos_*`/`conf_rot_*`（ours 置信度）、`cert_arm7_*`（ours 认证位形）。

## 4. `evaluator/` —— ✅ 判据齐全，且**天然与 reward 解耦**

`progress.py` / `progress_batch.py` 是判据本体（已核：`progress_batch.py` 内**无任何 reward 引用**）。
阈值与需求描述完全一致：

```
G1: PADS_MIN=3 (双手各≥3/5垫), PAD_FTH=0.5N, G1_HOLD=10 步
G2: CERT_RISE=0.005 (5mm 提升), CERT_SLIP=0.008 (8mm 滑移限), CERT_RAMP/HOLD/RET=8/5/8
G3: M2_TILT=90°, mouth_gate=0.12 (12cm 口距), M2_HOLD=25 步
Placed: M3_POS=0.03, M3_ROT=15°, M3_HOLD=15 步
G4/Success: M4_ARM=10° 逐关节, M4_DIST_POS/ROT=0.05/30°, M4_HOLD=15 步
死线: D1_DROP=0.05, D2_TILT=30°, D3_DEV=0.35, TABLE_Z=0.87 …（D4-D7 在 env 侧）
```

### 4.1 `run_eval.py` —— 统一评测器 ✅ 已写好并自检通过

```bash
cd evaluator && python run_eval.py            # 自检（不需要 Isaac）
python run_eval.py --entry your_pkg.mod:make_env,policy --episodes 512
```

自检结果（2026-08-29）：**世界指纹三项全部一致**（table_top_z 0.87 / 力阈 0.5N /
垫数 3），母带 `2ed81358` 交互行 273，瓶口局部 `[0,0.087,0]` 杯口 `[0,0.066,0]`，
512 个 seed 由 `--seed-base` **纯函数**生成（任何机器同值）。

**协议固定死**：t0 reset / 无 RSI / deterministic mean / 512 episodes；
报 success rate、environment steps、wall-clock、GPU 数、各关卡到达数、D1–D8 失败分布，
并落盘 `reset_manifest.json`。

**三条红线已在代码里落实**：不 import ours 任何训练件；判据阈值全部
`from progress import`（一个数字都没复制）；D1–D8 判因不读 ours 的 `pour_env.tb`
（批量版的 `fail` 只是 bool 张量不带原因，所以按 progress.py 同一组条件自行判因）。

**已通过 RL session 的红线 review**（扫 import 与敏感词，唯一命中的是文档里"本文件不做什么"
的声明本身）。review 提出的两处已改：死线归因改为**返回全部命中、每条各自计数**
（不跟 `progress.py` 那个"最后命中覆盖"的任意顺序）；删掉 `m3_snap` 的死代码回退分支
（`hasattr` 恒为 True，该分支永不执行）并加断言。

接入契约（两个回调 + `info` 里 7 个实测量 + 三条易错依赖）见
**`evaluator/INTEGRATION.md`**。

## 5. 完整性

`HASHES.txt`（sha256）/ `MD5.txt` 覆盖本目录**全部文件**（数量以文件本身为准，不在文档里重复写死）。

## 还缺什么

见 **`docs/MISSING_FOR_RL_AGENT.md`**。2026-08-29 已完成：任务 1（环境 manifest）、
任务 2（USD 自包含性验证 —— 结论是本来就自包含）、任务 4 关键部分（human_* 字段定性）。

**四项全部完成**（任务 1 环境 manifest / 任务 2 USD 自包含 / 任务 3 评测器 / 任务 4 字段定性）。

剩下的都在 baseline 侧或需要合作方拍板：
1. baseline 实现 `make_env` / `policy` 两个回调（契约见 `evaluator/INTEGRATION.md`）；
2. ~~母带用哪一版~~ —— **已定 `2ed81358`**（`reference/TAPE_DECISION.md`）；
3. 可选：在 Isaac 里跑一次零动作回放对拍，做物理一致性的严格证明
   （四个出生点的累计奖励与终止步逐位一致；USD 已验证零外部依赖，此步必要性已大幅下降）。

另需合作方拍板：**母带用 `2ed81358`（对齐 ours 当前训练）还是 `b95546e9`（复现现有 eval_best）**。
