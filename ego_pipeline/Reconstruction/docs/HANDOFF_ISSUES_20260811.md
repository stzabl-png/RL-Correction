# 重建管线待修清单（2026-08-11，来自 ARCTIC 真值校准实验）

背景：拿 ARCTIC 数据集（有物体 6DoF / 双手 MANO / 逐帧相机的真值）跑我们的全自动重建，
第一次能把误差量化。过程中撞到的问题列在这里，**未修的部分交给重建管线负责人**。
证据都给了文件行号或实测数字，不是推测。

---

## A. 未修 —— 建议改

### A1. 批量入口无法透传步骤参数 → `register-each` 事实上不可达

`fp_pose/run_sequence.py:122` 提供 `--pose-mode {track, register-each}`（默认 `track`），
但 `run_batch_queue.py:768-790` 拼步骤命令是**写死的参数列表**，没有任何透传机制
（`grep pose_mode|fp_mode|extra_args run_batch_queue.py` 零命中）。

**后果**：走 `reconstruct.sh` 或批量队列时**永远只能是 `track`**。`register-each`
（逐帧用 SAM2 mask 重新 register）只有直接调 `run_sequence.py` 才能用到，
所以它从未在真实批量里被评估过。

**建议**：给 `run_batch_queue.py` 加一个按步骤的额外参数透传口
（例如 `--step-args fp_pose=--pose-mode=register-each`），让这类模式可评估、可对比。

### A2. `_run_subprocess` 吞掉子进程 stderr → vipe 失败无法诊断

`_legacy/vipe/_common.py:552` 的 `subprocess.run(...)` 没有把子进程输出落盘。
vipe 失败时，worker 日志里只有一句 `CalledProcessError: ... returned non-zero exit status 1`，
**真正的错误完全看不到**。

**实测**：2026-08-11 02:08 在 GPU0 上重跑 laptop 的 vipe 失败，日志里只有 19 行，
没有任何可用于定位的信息，只能放弃该支线。

**建议**：把子进程 stdout/stderr 重定向到 step 的 worker 日志（或单独的 `vipe.child.log`）。

### A3. 自动标注"实例发现"耗时与文档严重不符

`auto_label_v17a.py` docstring 写 `2/3 实例发现 + SAM2 传播 (~1-2 分钟 GPU)`。
**实测**（同为 4 个 interaction episode）：
| 视频 | 帧数 | 实例发现耗时 |
|---|---|---|
| `s05/laptop_grab_01` | 828 | **16 分钟**（23:25→23:41） |
| `s07/ketchup_grab_01` | 703 | **>75 分钟** |

差 4.7 倍，且都远超文档。这一步是端到端耗时的**主要瓶颈**（重建九步本身
828 帧只要 20 分钟、100 帧只要 8 分钟）。

**建议**：至少把 docstring 改成实测区间；如果可能，查一下为何同样 4 个 episode 会差 5 倍
（怀疑与每个 episode 内候选部件数 / seed frame 重试次数有关）。

### A4. ★ 选帧策略：`sam3d` / `sam3d_scale` / `fp_pose` 三步继承**同一帧**，是单点故障

实测（`s07/ketchup_grab_01`，laptop 同构，帧号 642）：
```
label_prompt.json  frame_idx            347
sam3d              reconstruction_frame 347
sam3d_scale        reference_frame_idx  347
fp_pose            anchor_frame         347
```
一帧同时决定了**网格几何、米制尺度、位姿锚点**三件事，且三者串联、误差不抵消。

`auto_label_v17a.py` 的选帧规则是"该实例最早的 accepted 帧"，理由写的是
"通常手尚未接触、遮挡最小"。**但遮挡最小 ≠ 几何信息最全**：

`s05/laptop_grab_01` 的 f642 是**正上方俯视、笔记本平放**的一帧，遮挡确实极小
（`median_occl` 0.031），但那个视角**厚度方向完全不可观测** →
重建网格厚度多估 **36%**（长宽只差 6–7%），进而物体被放到远 **23%** 的位置。

这解释了为什么误差是**整条 take 一个常数**（laptop 深度比 1.358、ketchup 1.516）
而不是逐帧噪声 —— 它源于一次性的单帧决策，然后被双向跟踪带到全片。

**建议**：选帧判据里加入"几何可观测性"（例如物体主轴在像平面上的展开程度 /
mask 的长宽比是否退化），而不只看遮挡。这比在下游加判据更接近根因。
⚠ 注意 `track` 模式下逐帧 SAM2 mask **是被用的**（每帧质心+卡尔曼定初始位姿，
`fp_common.py:296-317`），所以问题不在"没用全片 mask"，而在**锚点那一帧选得不好**。

---

## B. 已修 —— 请知悉，勿回退

| commit | 问题 | 证据 |
|---|---|---|
| `8b33283` | **崩溃残留目录会永久毒化该视频**：`run_instance_video_segmentation` 拒绝写入非空目录（`FileExistsError`），一次 CUDA 崩溃留下的半截 episode 目录让之后**每次重试都在同一秒失败**。实测 9 条视频因此全部报废、机器空转 6 小时 | 已改为：确认无任何 ready manifest 时先清目录再跑 |
| `5c59763` | `reconstruct.sh` 批量里**一条视频标注失败会 `exit 1` 中止整批** | 改为跳过并记账，全败才失败 |
| `c3683e6` | ① `--gpu-ids` 落在 `-*` 万能分支：空格写法的值被 `[0-9]*` 吃成"跑前 N 条"，且**自动标注看不到它、固定吃 0 号卡**；② 实例发现的**最后阶段**（跨周期身份链接）失败时整步判失败，丢掉完好的逐 episode 产物 | `--gpu-ids` 提为一等公民；后段失败但前段可用则继续，`--instance all` 降级时双行警告 + `label_prompt_degraded.json` 存档 |
| （更早） | `run_batch_queue.py` 里"confidence 的 marker 在 final 目录"这条规则只写在 `_step_done()`，另外两处（跑完汇报、`_collect_lightweight_step_metadata`）无条件查 interim → **必然**报 `no completion marker`，且 step_metadata 永远没有 confidence | 抽出 `_marker_dir(job, step)` 统一三处 |
| （更早） | `retarget_isaacsim.py:833` 对 `nargs="+"` 的 `--object-usd` 用 `os.path.basename` → **多物体 Isaac 回放一次都没跑通过** | 已修 |
| （更早） | `auto_label_v17a.py` 多物体必须喂**视频级** `video_mask_sequence.json`；喂 episode 级会**静默塌成单物体** | 已接 `--instance all`，并写进 CLAUDE.md 第 6 条 |

---

## C. 不是管线的问题 —— 是我（分析侧）记错了，一并更正

我的记忆里曾有这条，**是错的**，已更正：
> ~~fp_pose 默认 `--fp-mode track`：1 帧配准 + 前后向跟踪；`register-each` 才用逐帧 mask~~

两处错：
1. **`track` 模式每帧都用逐帧 SAM2 mask**（质心 + 6D 卡尔曼 → 每帧初始位姿），
   `fp_common.py` 文件头 docstring 写得很清楚。正确说法是"`register-each` 才用逐帧 mask
   **做重新配准**"。
2. 参数名是 **`--pose-mode`**，不是 `--fp-mode`。

管线的 docstring 和 help 文本本身是准确的，误导来自我的笔记。

---

## D. confidence 侧（`RL_Correction/steps/step2_reconstruction/`，不属于重建管线）

列在这里只为完整，不需要重建负责人处理：
- `pose_audit.py` 的 `max_frames=400`：828 帧的视频只审前 400 帧，而实测这条数据
  **恰好前半段更差**（对称感知旋转误差 前400帧 38–66° vs 后半段 22–33°）
- `rot_factor` 惩罚不足：laptop 三轴可观测性 `[0.276,0.264,0.388]`（两轴低于阈值 0.28）
  却仍给 0.918；ketchup `[0.185,0.195,0.195]` 给 0.524、`rotation_usable` 仍为 True
- **缺一整条深度/尺度判据**：现有判据（`explained`/`d_cent_norm`/`lag`）全是图像内轮廓比对，
  而实测主误差在深度（laptop 深度分量 236mm vs 横向 38mm，投影质心只差 33px）
  —— 这类误差在图像里**结构性不可见**
