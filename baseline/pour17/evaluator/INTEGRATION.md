# `run_eval.py` 接入契约（给 baseline 侧）

评测器已写好并自检通过。baseline 只需实现两个回调，**其余全部固定**——
协议、判据阈值、seed 生成、世界指纹核对都在 `run_eval.py` 里写死，谁也改不了。

## 先跑自检（不需要 Isaac）

```bash
cd baseline/pour17/evaluator && python run_eval.py
```

会打印并落盘 `eval_result.json`：协议、G1–G4/死线全部阈值、世界指纹核对、
512 个 seed、GPU 信息。**世界指纹核对不通过就别往下做** —— 说明你的世界和我们的不是同一个。

当前自检结果（2026-08-29，本机）：

```
母带      pour17_reference_v2__2ed81358__653rows.npz  md5=2ed81358…  交互行=273
瓶口局部  [0.0, 0.087, 0.0]      杯口局部 [0.0, 0.066, 0.0]
世界指纹  ✅ 全部一致
   ✅ table_top_z           manifest=0.87  evaluator=0.87
   ✅ pad_force_threshold_N manifest=0.5   evaluator=0.5
   ✅ pads_min_per_hand     manifest=3     evaluator=3
   robot USD   .../assets/vega_1p_sharpa_fixedtorso.usd
   control_dt  0.05   decimation 12
协议      t0 reset / 无 RSI / deterministic mean / 512 episodes
seeds     20260829..20261340（由 --seed-base 纯函数生成，任何机器同值）
```

## ⚠ G2 认证：必须实现的第三个回调（否则 baseline 永远过不了 G2）

G2「抬升认证」**不是被动观测，是主动动作**：把双腕抬起来、保持、看物体有没有跟着升、再放回。
`policy` 无从知道"现在该抬手了"，所以**由统一评测器发起**：

```python
def apply_certification_offset(self, alpha: float) -> None:
    """alpha 取 [0,1]：把双腕从当前抓握站位沿【世界 +Z】抬 alpha * 15 mm，姿态保持不变。
    用你自己的控制器/IK 达成即可 —— 这正是它 method-agnostic 的地方。"""
```

- **规范口径**：双腕沿**世界 +Z 抬 15 mm，姿态不变**；
- **时间表由评测器给**：斜升 8 步 → 保持 5 步 → 斜降 8 步
  （`CERT_RAMP/CERT_HOLD/CERT_RET`，import 自 `progress.py`），你只管执行传进来的 `alpha`；
- **参考点是"抓握站位"**（母带交互首行 IA0=190 的腕位姿），**不是**母带末行 stance；
- **通过判据**：双物体各升 ≥5 mm（`CERT_RISE`）且手物相对滑移 <8 mm（`CERT_SLIP`）。

**ours 的实现与之等价（已 FK 反算验证）**：ours 在关节空间插值到预存的
`cert_arm7_{right,left}`，而那组值就是"腕位 +15 mm 世界 Z、姿态不变"的离线 IK 解：

| | dx | dy | dz | 姿态变化 |
|---|---|---|---|---|
| 右腕 | +2.37 mm | +0.51 mm | **+14.65 mm** | 0.154° |
| 左腕 | +2.92 mm | −0.05 mm | **+13.81 mm** | 0.117° |

主分量就是 +Z 约 14–15 mm，横向 2–3 mm 是 IK 残差（pos_err 0.96/0.89 mm）。
⚠ **ours 在关节空间插值、你若在笛卡尔空间直线插值**，15 mm 尺度下差异是亚毫米级 ——
可以接受，但**写在这里以免日后拿它解释成绩差异**。

## ⚠ Episode horizon = 903 步（不是 13.125 秒）

```
MAX_CONTROL_STEPS = 903 = int(T_ROW * 1.2) + 120,  T_ROW = 母带全长 653
                        = 45.15 s @ control_dt 0.05
```

**`world_manifest.time.episode_length_s = 13.125` 对 Pour17 无效** —— ours 从不用 env 的
`truncated` 走超时，用的是 env 侧自算的 D7。**评测器自己强制这个上限，不看你的 `truncated`**，
所以你的 env 把 horizon 设成多少都不影响可比性（但建议设得 ≥903 免得提前截断）。

## ⚠ 焊接热身：评测协议设 0（需两边确认）

ours 训练时 `POUR_HOLD_K=15`：复位后前 15 个控制步把两物体**钳位**在初始位姿，
让手指先压实成形再放开物理。**这是训练脚手架，不是任务定义**，且前 0.75 s 物体不受力会影响成绩。

**✅ 已裁定（2026-08-29，项目负责人）：评测协议一律 `POUR_HOLD_K=0`，两边都不做。**
理由：那 15 步钳位是 ours 的训练脚手架，不属于任务定义；且前 0.75 s 物体不受力会影响成绩。
`run_eval.py` 的 `WARMUP_CLAMP_STEPS = 0` 已固化此裁定，自检会打印出来。

完整起跑线定义见 **`world/canonical_reset_v1.json`**（58 关节初始 q/qd、物体初始位姿与速度、
无随机化、root pose 处置、horizon、认证规范）。

## 要实现的两个回调

```python
def make_env(num_envs: int, seed: int, device: str) -> EnvLike: ...
def policy(obs): ...        # deterministic mean，不要加探索噪声
# 外加 EnvLike.apply_certification_offset(alpha)  <- 见上方 G2 一节（必须实现）
```

`EnvLike` 需要：

| 成员 | 说明 |
|---|---|
| `reset() -> obs` | 从 t0 复位（**不要 RSI**） |
| `step(action) -> (obs, terminated, truncated, info)` | 标准 gym 风格 |
| `reset_state()`（可选） | 返回本 episode 初始状态，写进 `reset_manifest.json`；没有就标 unavailable |
| `apply_certification_offset(alpha)` | **必须**：G2 认证，见上方 G2 一节 |
| `close()`（可选） | |

`info` 每步必须带这 7 个**实测量**（evaluator 只吃这些，不碰你的 obs/action）：

| key | 形状/类型 | 含义 |
|---|---|---|
| `object_0_pose` | (7,) xyz + wxyz | 杯，世界系，米 |
| `object_1_pose` | (7,) xyz + wxyz | 瓶 |
| `arm_q_right` | (7,) | 右臂关节角 |
| `arm_q_left` | (7,) | 左臂关节角 |
| `pads3` | bool | **双手各 ≥3/5 指垫接触力 >0.5N** |
| `wrist_right` | (3,) | 右腕位置，世界系 |
| `wrist_left` | (3,) | 左腕位置 |

> `pads3` 的口径：10 个指垫 body 是
> `{right,left}_{thumb,index,middle,ring,pinky}_elastomer`；
> 右手 5 个对**瓶**(`/World/envs/env_.*/Object`)、左手 5 个对**杯**(`.../Aux`) 测力；
> 力阈 0.5N、每手至少 3 个 —— 全部见 `world/world_manifest.json` 的 `sensors` 段。

## 跑评测

```bash
python run_eval.py --entry your_pkg.your_module:make_env,policy \
                   --episodes 512 --device cuda:0
```

产出 `eval_result.json`（success rate / environment steps / wall-clock / GPU 数 /
各关卡到达数 / D1–D8 失败分布 / 逐 episode 明细）+ `reset_manifest.json`。

## 三条设计红线（已在代码里落实）

1. **不 import ours 的任何训练件** —— 没碰 `eval_pour.py`、PPO wrapper、503D obs、
   58D residual action、GraspPose、CuRobo、confidence reward。只依赖冻结的
   `progress.py` + `world_manifest.json` + 母带。
2. **判据阈值全部 `from progress import`，一个数字都没复制** ——
   `progress.py` 是唯一真源，改它评测器自动跟。
3. **D1–D8 原因分类不读 ours 的 `pour_env.tb`**（那是 ours 的 env 成员）。
   批量版 `PourProgressBatch` 的 `fail` 只是 bool 张量、不带原因，所以
   `classify_failure()` 按 `progress.py` 的**同一组条件**自行判因，阈值仍来自 import。

## ⚠⚠ `progress.py` / `progress_batch.py` 里含 ours 的奖励常量 —— 请勿使用

评测器需要 `progress.py` 来算 G1–G4 与死线，但**判据常量和奖励常量住在同一个文件**。
以下这些**属于 ours 的方法，评测路径完全不读它们**：

| 常量 | 是什么 |
|---|---|
| `W_OBJ` / `W_HAND` | ours HYB 变体的**置信门控双参考权重** |
| `LEASH_POS` / `LEASH_ROT` | ours 的皮筋容差（奖励侧） |
| `MS_REWARD` | ours 各关卡的奖励额度 |
| `WAGE` / `WAGE_CAP` | ours 的站位维持费与封顶 |
| `GATE_POS` / `GATE_ROT` / `RED_GATE_POS` / `TIER_HI` / `TIER_LO` | ours 的时钟门与置信分档 |

**请勿 import、复制或用它们复现 ours 的奖励。** H2S2R 应当只用自己的 reward。

评测器实际 import 的只有判据量：`G1_HOLD`、`CERT_RAMP/HOLD/RET`、`CERT_RISE`、`CERT_SLIP`、
`M2_TILT`、`M2_HOLD`、`M3_POS/ROT/HOLD`、`M4_ARM/HOLD`、`M4_DIST_POS/ROT`、
`D1_DROP`、`D2_TILT`、`D3_DEV`、`TABLE_Z`，以及 `PourProgress` / `_axis_tilt`。

> 说明：之所以没有给一个"裁剪版 progress.py"，是因为那会**破坏唯一真源** ——
> 裁剪版会和 ours 的主版漂移，而两边判据一旦不同，谁都不会发现。
> 宁可带着不该看的常量，也要保证判据逐字节同源。

## 物理量的 `*_source` 纪律

manifest 里凡是物理量，尽量带一个 `*_source` 字段说明取值来源。

**为什么**（2026-08-29 真实案例）：`robot.self_collision` 一度记作 `False`，
而它是采集器 `getattr(..., False)` 的**默认值**，不是任何一次读取 —— 后来发现
`correction_env_cfg.py:121` 写的 `enabled_self_collisions=True` **对本任务的 articulation 未生效**，
而 USD 里 authored 的真值确实是 `False`。

也就是说：**那个值恰好蒙对了，但记录本身不可信**。三处都叫 self_collision，三个来源
（cfg 声明 / 采集器默认 / USD authored），只有第三个是 PhysX 真正读的。

⟹ 没有 `*_source`，你无法区分"读到了 False"和"默认成了 False"。
当前 `self_collision_source = "USD:/World/envs/env_0/Robot/root_joint"`，
本 bundle 也独立用 `pxr` 复核过该属性 authored 值为 `False`。

## 三条依赖关系（2026-08-29 与 RL session 核实，容易搞错）

**1. 瓶口/杯口的 `half` 是几何量，换物体必须改**
`mouth_local()` 里 `up = [0, half, 0]` —— **物体局部 Y 轴即长轴**，`half` 是沿长轴的半长：
`0.087` = 瓶高 17.4cm / 2，`0.066` = 杯高 13.2cm / 2。
- 换物体 ⟹ **必须改这两个数**（它们是几何量，不是标定量）；
- 换母带（同物体）⟹ 数值不用改；
- ⚠ 但**"口"取长轴哪一端，是由母带首个交互行的四元数决定的**
  （取静置时世界 z 更高的那端）。母带换代若首行朝向变了，符号可能翻。
  当前平滑保证首行是锚点（可信帧），所以稳定 —— 但这是一条真实依赖，换母带后值得核一眼。

**2. 评测不传 `no_hand_ref`**
它**只影响 ours 训练侧的奖励权重**（`out["w_obj"]` / `out["w_hand"]`），
**对判据 / 成败 / 死线零影响**。
⚠ `progress.py` 模块头那句"no_hand_ref=True → 红档宽物门 8cm 无手接管"是 **L5-6 之前的
过期描述**（时钟门早已统一为 `gp = GATE_POS if tier > 0 else RED_GATE_POS`，与它无关）；
RL session 已修掉该注释。以代码为准。

**3. `leash_rot_tilt` / `kcap` 保持默认**，评测口径不动它们。

## 死线归因口径 ⚠（与 progress.py 不同，有意为之）

`classify_failure()` **返回全部命中的死线**，分布按**每条各自计数**（一个 episode 可计入多条）。

为什么不跟 `progress.py` 的 `out["fail"]`：那是**反复赋值、最后命中覆盖前面的**，
顺序是写终止逻辑时的副产物，对归因没有意义。评测报告要的是分布，所以这里用无歧义口径。
**对"是否终止"的判定无影响**（任一成立即终止）。

结果 JSON 里两种归因都留着，可互相对拍：
- `deadlines_fired`：本评测器口径（全部命中）
- `fail_reason_progress`：`progress.py` 自己的单条归因

## 换母带 = 换判据 ⚠

evaluator 从母带取**瓶口/杯口方向**与**静置位姿**，所以 `--tape` 换版会改变 G3/Placed 的判定。
默认用 `2ed81358`（ours 当前训练版）。若与 `world_manifest.json` 记录的版本不一致，
自检会打印警告。**报告里必须写明用了哪一版。**
