# 世界指纹（World Fingerprint）——为什么你的 ckpt "突然用不了了"

> 面向：**Kailang**（baseline 对照实验）、**Dubang**（DP 数据采集/训练）、
> 以及任何要跨机器、跨时间复现一次 rollout 的人。
> 2026-08-29 首版。本文档描述的问题在本项目已发生过至少三次。

---

## 一、症状：ckpt 没坏、场景没错，但分数是 0

典型现场：

- 同事给你一个 `eval_best.pth`，在他机器上 100%，在你机器上 **0/64**；
- **不报任何错**。维度对得上（还是 348 维），加载成功，跑得很顺，就是不得分；
- 你去看渲染，机器人动作"像那么回事"，但抓空、或者往桌子里插。

## 二、病根：ckpt 背的是**绝对数值**，不是"看图行事"

这些策略**不看图像，执行时也不重新感知**。物体位置、抓握目标点、参考轨迹，
全部作为**具体数值**烧进了权重里——相当于肌肉记忆走位。

所以只要"世界的数值"变了，记忆就对不上：

| 变了什么 | 后果 |
|---|---|
| 机器人 USD 换版（站姿变了） | 同一组关节角对应的手掌位置全变 → 抓空 |
| 桌高改了 | 下压深度全错 → 插进桌子或悬空 |
| 物体出生位姿变了 | 伸手方向全错 |
| 参考母带换代 | 时序/判据基准变了 |

### ★ 为什么维度闸救不了你

**"同维异义"**：换了配置但**形状没变**。obs 还是 348 维、action 还是 58 维，
所有 assert 全绿——但每一维的**含义**变了。

> **维度闸对"世界换版"是瞎的。它的绿灯零信息量。**

这是本项目 2026-08-26 的真实事故：主 USD 换新站姿后，AAG-F 的 ckpt 回放直接 0/64，
而 obs 维度一字没变。

## 三、解法：跑之前先对"世界指纹"

**世界指纹 = 一次 rollout 所依赖的全部世界数值的快照。**
ckpt 存盘时记下来，加载时对一遍，不匹配就拒跑并告诉你差在哪 ——
把"跑完 64 局拿 0 分再猜"变成"启动前 1 秒报错并指名"。

### 最小版（人工，30 秒，现在就能做）

```bash
md5sum assets/vega_1p_sharpa_fixedtorso.usd
# f77f235df53494508fefdf1d1518af32  = 旧站姿（AAG-F 的出生世界）
# 143385ad217f10f3fe730338d748246d  = 新站姿（Pour17 当前世界）
```

**不匹配时不要改 ckpt，要把世界切回去**：

```bash
export DEXMATE_FIXED_USD=$PWD/assets/vega_1p_sharpa_fixedtorso_stance0803.usd
```

> `stance0803` **不是"病文件"，是版本错配**：同一份 USD 对 Pour17（新站姿）是错的，
> 对 AAG-F（旧站姿出生）是**唯一正确世界**。没有绝对的病文件，只有相对权重的版本。
> **永久禁删。**

### 完整版（机器可读）

```bash
OMNI_KIT_ACCEPT_EULA=YES python baseline/pour17/world/export_world_manifest.py
# -> world_manifest.json
```

内容（**全部运行时读出，不是抄常量**）：

| 段 | 关键字段 |
|---|---|
| `time` | `physics_dt_s` / `control_dt_s` / `decimation` / `render_interval` |
| `table` | `table_top_z_m` / `table_size_m` / 摩擦 |
| `objects` | 逐物体 `mass_kg` / `static_friction` / `restitution` / **实际加载的 `usd_loaded`** / `rest_pose_env_frame_xyz_wxyz` |
| `robot` | `usd` / `num_controlled_joints` / **58 名有序关节** / 软限位 / `self_collision` + `self_collision_source` |
| `sensors` | 11 个接触传感器的 `prim_path` / `history_length` / `filter_prim_paths_expr` / 力阈 |
| `switches` | `approach_only` / `obj_jitter_xy` / `friction_curriculum` / `POUR_NO_D6_effective` |
| `reference_tape` | 母带 `path` / `md5` / `sha256` |

配套的**起跑线**在 `baseline/pour17/world/canonical_reset_v1.json`：
58 关节初始 q/qd、两物体初始位姿与速度、有无随机化、**episode horizon**、G2 认证规范。

---

## 四、三条从血里学来的纪律

### 纪律 1：值对 ≠ 记录可信 —— 物理量必须带 `*_source`

真实案例（2026-08-29）：`robot.self_collision` 记作 `False`。

- 它来自采集器 `getattr(..., False)` 的**默认值**，不是任何一次读取；
- `correction_env_cfg.py:121` 写着 `enabled_self_collisions=True`；
- USD 里 authored 的真值确实是 **`False`**（`/vega_1p_sharpa/root_joint`）。

⟹ **那个值恰好蒙对了，但记录本身完全不可信。** 三处都叫 self_collision，三个来源，
只有 USD 那个是 PhysX 真读的。

**做法**：多路取值 + 标注来源，读不到记 `null` 而不是默认值。

```json
"self_collision": false,
"self_collision_source": "USD:/World/envs/env_0/Robot/root_joint"
```

没有 `*_source`，你无法区分"**读到了** False"和"**默认成了** False"。

### 纪律 2：任何"率"旁边必须有它的分母

真实案例（2026-08-29）：监控里 `rate = hit / max(n, 1)`，**分母为 0 时把"本窗零回合"
渲染成"成功率 0%"**——据此差点判定两条健康训练线"每回合开局即死"，
真相只是回合太长还没结算。

**做法**：分母为 0 发 `NaN`/`null`，**并把分母本身作为指标发出去**。

```json
"success_rate": null,          // 不是 0.0
"success_rate_denominator": 0
```

### 纪律 3：同名字段可能有多个来源，写清哪个生效

Pour17 现存的三组同名歧义：

| 名字 | 出现处 | 哪个算数 |
|---|---|---|
| 桌高 | `scene_layout.rl_table_height=0.85` / `scene_table_z=0.8638` / `cfg.table_top_z=0.87` / `progress.TABLE_Z=0.87` | **物理只认 0.87**；前两个是重建侧摆放量，不进物理。⚠ `progress.py` 的 0.87 是**独立硬编码副本，与 cfg 无联动**——改桌高必须两处同步改 |
| 关节数 | `DEXMATE_ARTICULATION_JOINTS.txt` 67 行 / USD 里 104 个 joint prim / 运行时 58 | **58**（USD 含固定/被动关节；那份 67 行的表已过期） |
| episode 时长 | `world_manifest.time.episode_length_s = 13.125` / env 侧 `D7 = int(T_ROW*1.2)+120 = 903` 步 | **903 步（45.15s）**。`episode_length_s` 对 Pour17 **无效**——ours 从不用 env 的 `truncated` 走超时 |

---

## 五、按角色的用法

### Dubang（DP 数据采集 / 训练）

1. **采集前跑第 0 步硬闸**（见 `docs/DP_AAGF_RUNBOOK.md`）：md5 不对不许跑；
2. **存 ckpt 时把 `world_manifest.json` 一起存进 ckpt 目录**；
3. **加载 ckpt 时对指纹**，不匹配拒跑并列出差异项；
4. 采集出的 DP 数据在 README 里注明它的世界指纹 —— 否则半年后没人知道这批数据出生在哪。

> ⚠ 当前现状：`logs/*/world.json` 只有 8 个键，而且把 `usd` 记成 `"default"`
> （意思是"没设覆写"，**不是路径**）；**最新那批 `eval_best.pth` 连 `world.json` 都没有**。
> 这正是"用不了了也查不出为什么"的根源。

### Kailang（baseline 对照实验）

1. 用 `baseline/pour17/` 里的 `world_manifest.json` + `canonical_reset_v1.json` 搭环境；
2. 跑 `evaluator/run_eval.py` 自检 —— 它会把你的世界和指纹逐项对拍，**不通过别往下做**；
3. 报告里写明用的**母带版本**（换母带 = 换判据）、**horizon**、**焊接热身设置**。

接入细节见 `baseline/pour17/evaluator/INTEGRATION.md`。

### 任何人：新增一个世界依赖时

问自己三句：
1. 这个量**运行时**是多少（不是 cfg 里写的多少）？
2. 它的**来源**是哪里，读不到时会不会悄悄用默认值？
3. 同名的量在别处还有吗，哪个生效？

---

## 六、相关文件

| 路径 | 用途 |
|---|---|
| `baseline/pour17/world/export_world_manifest.py` | 世界指纹导出器（运行时读真值） |
| `baseline/pour17/world/world_manifest.json` | Pour17 当前世界指纹 |
| `baseline/pour17/world/canonical_reset_v1.json` | 起跑线：初始状态 + horizon + G2 认证规范 |
| `baseline/pour17/evaluator/run_eval.py` | 统一评测器（启动即对拍世界指纹） |
| `baseline/pour17/evaluator/INTEGRATION.md` | baseline 接入契约 |
| `docs/DP_AAGF_RUNBOOK.md` | AAG-F → DP 采集手册（第 0 步硬闸） |
| `baseline/pour17/world/flatten_usd.{sh,py}` | USD 依赖分析 / 自包含性验证 |
