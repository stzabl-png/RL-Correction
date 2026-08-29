# 任务书：补齐 Pour17 Baseline Bundle 的缺口（给 RL Agent）

## 背景与边界

我们要给合作方一个**自包含的 Pour17 baseline bundle**，让 adapted H2S2R 在与 ours
**相同的 simulator / robot / scene / reset / evaluator** 下重新训练。baseline 保留它自己的
observation、action、reward 和 FABRICS controller —— 我们只统一"世界"和"评判"。

**已完成部分在 `baseline/pour17/`**（世界资产、感知数据、reference、evaluator 本体、hash）。
先读那里的 `README.md` 再动手，不要重复已有工作。

**红线：不要把 ours 的方法泄漏给 baseline。** 以下内容不进 bundle、也不要在导出物里引用：
`eval_best.pth`、dp_collect 的 348D 数据、GraspPose、CuRobo guide、confidence reward、
ours 的 reward weights、503D observation、58D residual action。

---

## ✅ 任务 1 已完成（2026-08-29，RL session）
产物 `world/world_manifest.json` + 生成器 `world/export_world_manifest.py`（可重跑）。
下面原始需求保留作存档。

## ~~任务 1（最高优先）：写一个真正的环境指纹导出器~~

**现状**：`logs/*/world.json` 只有 8 个键
（`variant/beta_r/beta_l/ref_npz/ref_md5/usd/obs_dim/act_dim`），**不是环境指纹**；
而且 `eval_best.pth` 对应的最新 run（`logs/E2EL_pour17/2026-08-26_*`）连 `world.json` 都没有，
只有 `auto_stop.json`。

**要做**：新增 `baseline/pour17/world/export_world_manifest.py`，构造一次 Pour17 env
（`tasks/Pour/17/C_Wiring/pour_env.py::build_cfg` 是入口）后，从**实际生效的 cfg / sim / 场景对象**
里读出下列值，写成 `world_manifest.json`。**必须是运行时读出的真值，不许手抄常量**；
读不到的字段显式写 `"unavailable"` 并注明原因。

需要的字段：

1. **时间步**：`sim.dt`、control dt、`decimation`、`render_interval`、physx substeps、solver 迭代数
2. **桌子**：最终生效的 table height 与 table pose
   （注意 `scene_layout.json` 里有两个数：`rl_table_height=0.85` 与 `scene_table_z=0.8638`，
   差 `scene_to_rl_shift_z=0.0138`；而 `progress.py` 里另有 `TABLE_Z=0.87`。
   **三个数必须说清哪个在物理里生效、哪个只是判据阈值**）
3. **两个物体**（杯 object_0 / 瓶 object_1）：mass、static/dynamic friction、restitution、
   collision approximation（convexHull / convexDecomposition / SDF …）、密度或质量来源
4. **机器人**：默认站姿（各关节角）、**58 个受控关节的准确名称与顺序**
   （布局已知 = `[R臂7, L臂7, R指22, L指22]`，名称运行时在 `self.hand.joint_names`；
   注意 `DEXMATE_ARTICULATION_JOINTS.txt` 那 67 行是全 articulation，**不是**这 58 个）
5. **接触传感器**：10 个指垫的 body 名与阈值
   （已知 body 名 `{left,right}_{thumb,index,middle,ring,pinky}_elastomer`，
   力阈 `PAD_FTH=0.5N`、`PADS_MIN=3`；还需 `history_length`、`filter_prim_paths_expr`、
   以及 D6 跨侧互撞传感器的完整配置）
6. **self-collision 是否启用**，以及其它影响物理结果的开关
   （`friction_curriculum` 默认值、`obj_jitter_xy`、`approach_only`、`POUR_NO_D6` 等环境变量的默认行为）
7. **资产指向**：实际加载的是 `datasets/pour17/objects/*.usd` 还是 `datasets/pour17/cache/*.usd`
   （`cache/` 下有 `.asset_hash` 与 `config.yaml`，怀疑 cache 版才是训练实际用的 —— **必须确认**）

产物：`baseline/pour17/world/world_manifest.json` + 追加 hash 到 `HASHES.txt`。

## ✅ 任务 2 已完成（2026-08-29）——结论：**本来就自包含，不需要 flatten**

全部 5 个 USD 依赖分析结果：外部 layer 0 / 资产引用 0。工具在
`world/flatten_usd.sh` + `flatten_usd.py`（可重跑）。详见 README §2.1。

★ **顺带记一条环境事实**（我一开始误判成"本地没有 USD 工具"）：
本机**有** USD 24.05，在 Isaac 的 `extscache/omni.usd.libs` 扩展里。引出方式：
```
PYTHONPATH      = <该扩展根目录>          # 里面有 pxr/
LD_LIBRARY_PATH = <该扩展>/bin : <libpython3.11.so.1.0 所在目录>
```
**注意 .so 在 `bin/` 不在 `lib/`**，且不需要起 SimulationApp、不需要 EULA 变量。

## ~~任务 2 原文~~（**已完成**）

★ **验证判据（RL session 给的，比看文件结构可靠）**：flatten 前后各跑一次
**零动作回放**，**四个出生点的累计奖励与终止步必须逐位一致**。
口径直接用 `tasks/Pour/17/C_Wiring/smoke_zero.py`。
★ 注意 flatten 的对象是 **`vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd`**，
不是 `stance0803`（那是 AAG-F 的世界，与 Pour17 无关）。
⚠ 本机 `pxr` 只在 Isaac 环境里，重建侧 venv 没有 —— 需在装了 USD 的环境执行。

## ~~任务 2 原文~~

1. 检查 `baseline/pour17/world/vega_1p_sharpa_fixedtorso_stance0803.usd` 有没有外部引用
   （sublayers / references / payloads / 贴图），列出全部依赖；
2. 产出 **flattened / self-contained** 版本（`usdcat --flatten` 或等价），放
   `baseline/pour17/world/vega_1p_sharpa_fixedtorso_stance0803.flat.usd`；
3. 验证 flatten 后能在干净环境里加载并与原版**物理一致**（至少：关节数、质量属性、碰撞体一致）；
4. 对 `objects/*.usd` 与 `cache/*.usd` 做同样检查；确认两版差异并说明哪版训练在用。

## ✅ 任务 3 已完成（2026-08-29）
产物 `evaluator/run_eval.py` + `evaluator/INTEGRATION.md`，自检通过（世界指纹三项一致）。
三条红线已落实。剩下的是 baseline 侧接两个回调。

## ~~任务 3 原文~~：评测协议（evaluator 已有，协议缺）

`baseline/pour17/evaluator/` 里的 `progress.py` / `progress_batch.py` 是判据本体，
已核实**与 reward 解耦**（`progress_batch.py` 内无任何 reward 引用），直接冻结即可，不要改判据。

★ **RL session 的红线提醒**：**不要复用 ours 的 `eval_pour.py`** —— 它 import 我们的
PPO/wrapper/503D obs，会把 ours 的方法带进 bundle。baseline 自己写，只依赖三样：
world（USD + manifest）、evaluator（progress.py，已冻结且与 reward 解耦）、reset（从 t0，不用 RSI）。
★ **D1–D8 统计不要读 `pour_env.tb`**（那是 ours 的 env 成员）——
改为按 `progress.py` 的 `out["fail"]` 字符串分类（D1_drop / D2_tilt / D3_dev / D8_disturb…），
成功判定读 `PourProgress.g[4]`（批量版 `PB.g4`）。这样完全不碰 ours 的 env。

**要做**：写 `baseline/pour17/evaluator/run_eval.py`，固定评测协议：

- 全部 episode **从 t0 reset**，**不使用 RSI 初始化**
- **不加 exploration noise**，用 deterministic policy mean
- **至少 512 episodes**
- 产出并冻结 **固定 seed list** 与 **reset manifest**
  （每个 episode 的初始物体位姿/机器人位形，可复现）
- 报告：success rate、达到指定成功率所需 environment steps、wall-clock time、GPU 数量
- 输出完整 **failure / timeout reason 分布** 与 **D1–D8 逐条计数**
  （`pour_env.py` 里 `self.tb` 已在记账 `term/D1..D8`，直接复用）

产物：`run_eval.py` + `eval_seeds.json` + `reset_manifest.json` + 一次基线跑的结果 JSON。

## ✅ 任务 4 关键部分已完成
`human_right_q`/`human_left_q` 已定性：= **v1 的 `right_q` 重采样**，即机器人臂角，
**不含人手原始信息**（`build_ref_v5.py:373`）。**不进 bundle 的训练输入**。
剩余：给三份母带各写 `FIELDS.md` 逐字段可用/不可用标注。

## ~~任务 4 原文~~：reference npz 的字段级污染清单

给 `reference/pour17_reference_v{1,2}.npz` 各写一份 `FIELDS.md`，逐字段标注
**baseline 可用 / 不可用**，理由一句话。已知必须标"不可用（ours 专属）"的：
`right_q`、`left_q`、`right_f`、`left_f`、`conf_pos_*`、`conf_rot_*`、`cert_arm7_*`。
`human_right_q` / `human_left_q`（仅 v2 有）需要说明它是"人侧 7 维臂角"还是别的东西 ——
**去代码里确认它的生成路径**，不要猜。

---

## 交付验收

全部完成后：
1. 更新 `baseline/pour17/README.md` 的状态表（把 ⚠️ 改成 ✅）；
2. 重新生成 `HASHES.txt` / `MD5.txt`；
3. 在 `docs/` 下留一份 `HANDOFF.md`：合作方拿到 bundle 后的最小复现步骤。

## 已经确认、不要重复调查的结论

- `pour17_reference_v1/v2.npz` 是**实体文件，不是 Git-LFS 指针**（165KB / 363KB）；
- **v1 里没有任何人手原始量**，它的 `right_q/left_q` 是 (T,7) 机器人臂角。
  原需求以为"v1 可核对原始视频轨迹"是错的，方法中立数据在 `perception/`；
- `perception/pour17_perception.npz` 已产出（重建仓侧），含腕位姿、22 维手指 q、
  物体位姿、逐帧有效标志、以及 MANO 备份路线，米制 + wxyz + left→cup/right→bottle；
- EgoDex **不提供绝对时戳**，`timestamps` 只能标 unavailable；
- 源视频 30fps、重建产物 15fps/142 帧，**两个时基**已在 `meta` 写明。
