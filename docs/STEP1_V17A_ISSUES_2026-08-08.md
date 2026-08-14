# Step1 感知层批量跑通实录与问题清单

**日期** 2026-08-08 　**作者** 杨泓（stzabl@berkeley.edu）　**主要收件人** Kailang
**涉及代码** `jiaka1chen/HumanVideo2RobotData` 分支 `STEP_3_V17`，commit `be0791b`
（v17A 管线由 commit `767e455` 一次性引入，作者 DennisXu626）
**本地副本** `RL_Correction/steps/step1_perception/masks/v17a/`（与上游 `experiments/hoi_detr` 逐字节一致，未改动）

---

## 摘要

我用 v17A + sam3_hands 批量处理 egodex/test 的 200 条视频，想建一个"重建前的 video prior"数据库。
跑到第 86 条时机器被拖死，中断。实测结果：

| | |
|---|---|
| 已尝试 | 86 / 200 |
| v17A 成功 | 35 条（其中 32 条手 mask 也齐全） |
| v17A 失败 | 51 条 |
| **成功率** | **约 37%** |

**结论先说三句：**

1. **v17A 的设计取向没有问题**，它是 precision 优先、"宁可失败也不输出错的"。这个判断对下游三维重建是对的。
2. **失败率高主要来自两个可调阈值的标定**，不是模型能力问题 —— SAM2 在 465 次候选生成里**一次都没有失手**（原始候选中位 12 个，从未为 0），是后面的过滤链把 81% 砍掉了。
3. **有 3 个是实打实的 bug 或过严硬约束**，会让整条视频直接失败，建议优先修。

另外有一条是**我们这边用错了**：v17A README 的流程是三步，第三步 `render_mask_sequence_preview` 是人工验收环节，我的批处理脚本把它整个跳过了，所以质量问题积累到第 86 条才被发现。

---

## 一、基础设施问题（我们这边，供参考）

批处理跑了 11 小时后整机死机，需要硬重启。根因是内存。

`run_instance_video_segmentation.py:197` 和 `:222` 两处都传了：

```python
init_state_kwargs={"offload_video_to_cpu": True}
```

SAM2 会把整段视频按 1024×1024 float32 全部驻留在内存，约 **12.6 MB/帧**；而这个脚本一条视频里会建**至少两份** inference state（`propagate_multi_object_logits` + `object_seed_from_box`）。

egodex 的长视频正好踩线：

| take | 帧数 | 单份 state ≈ |
|---|---|---|
| sort_beads/0 | 1715 | 21.6 GB |
| setup_cleanup_table/0 | 1341 | 16.9 GB |
| set_up_clean_up_chessboard/0 | 1323 | 16.6 GB |
| sleeve_unsleeve_cards/0 | 1098 | 13.8 GB |

我的机器只有 32 GB 内存。OOM killer 先后杀了 **6 次**，被杀的全是 `run_instance_video_segmentation`
（`01:06 / 01:23 / 03:14 / 09:22 / 09:29 / 09:42`），单进程峰值 `anon-rss 29 GB + swap 26 GB ≈ 55 GB`。
最后在换页颠簸中触发内核调度器 Oops（`RIP: 0x0`，CPU12 hard LOCKUP），整机僵死。

**这不算 v17A 的问题**（在大内存机器上不会暴露），但如果要在 <64 GB 的机器上批跑长视频，建议：

- 给这一步套内存上限，让它自己死而不是拖垮整机：
  `systemd-run --user --scope -p MemoryMax=18G -p MemorySwapMax=0 -- python -m experiments.hoi_detr.run_instance_video_segmentation ...`
- 或者对 >800 帧的视频先分段/抽帧
- 长期看，`offload_video_to_cpu` 之外可以考虑 SAM2 的 `offload_state_to_cpu` + 分段 propagate

---

## 二、失败分布（86 条实测）

顶层 `summary.json` 的 status（63 条有产出 summary 的）：

| status | 条数 |
|---|---|
| `success` | 35 |
| `failed_episode_segmentation` | 17 |
| `failed_global_identity_linking` | 11 |

下层各阶段的状态计数（跨所有 take 累计）：

| status | 次数 |
|---|---|
| `failed_no_candidates` | 300 |
| `failed_seed_discovery` | 37 |
| `failed_no_candidates_in_interaction_roi` | 21 |
| `failed_fatal_error` | 19 |
| `failed_ambiguous_visual_memory` | 38 |

（另有 6 条是被前述 OOM 杀掉的，与 v17A 无关，不计入。）

---

## 三、三层失败的机理与证据

### 3.1 候选筛选：判据④和判据⑤自相矛盾 ★最高优先级

**漏斗实测**（465 次 AMG 调用）：

| | 中位 | 均值 | 为 0 的次数 |
|---|---|---|---|
| SAM2 原始候选 `raw_candidate_count` | 12 | 13.3 | **0 次** |
| 过滤后 `filtered_candidate_count` | 1 | 2.5 | **118 次（25%）** |

**SAM2 一次都没失手，全部损耗发生在过滤链**，平均淘汰 81%。

过滤链在 `run_sam2_amg_probe.py:run()`，5 道硬性淘汰：

```python
hand_region = _hand_region(shape, detection_frame)   # ← 手的检测【框】，不是手 mask

for candidate in candidates:
    if area < 500 or area > 100000:                       continue   # ① 面积带
    if mask[0].any() or mask[-1].any() \
       or mask[:,0].any() or mask[:,-1].any():            continue   # ② 碰画面四边就扔（硬编码，无 CLI 开关）
    if largest_component_fraction < 0.98:                 continue   # ③ 必须 98% 单连通域
    if (mask & hand_region).sum()/area > 0.1:             continue   # ④ 与手框重叠 >10% 就扔
```

然后 `run_instance_seed_discovery.py:_candidate_pool()` 再筛一次：

```python
interaction_roi = _box_region(shape, _linked_object_boxes(frame))  # 与手有 hf 链接的物体框
if (mask & interaction_roi).sum()/area < 0.15:            continue   # ⑤ 必须 15% 落在交互 ROI
```

**矛盾点：**

- ④ 要求候选**不能**压在手框上
- ⑤ 要求候选**必须**落在交互 ROI 里，而交互 ROI 就是"**与手有链接的那个物体**"的框

第一人称操作视频里，手正握着物体，手框和物体框在几何上必然大面积重叠。

**实测（173 个 seed 帧样本）：**

```
物体 ROI 落在手检测框内的比例:   中位 22.8%
超过 ④ 的 0.1 阈值的 seed 帧:    82%
手检测框平均覆盖画面:            5.8%
```

也就是说 **82% 的 seed 帧上，任何真正盖住交互物体的候选都会被判据④当场淘汰。** 对照数据完全吻合：

| take | seed 帧 | 物 ROI 在手框内 | raw → filtered |
|---|---|---|---|
| insert_remove_furniture_bench_cabinet__0 | 25 | 100.0% | 7 → **0** |
| roll_ball__0 | 128 | 100.0% | 9 → 1 |
| clean_cups__0 | 3 | 96.8% | 13 → 1 |
| fold_unfold_paper_origami__0 | 299 | 85.4% | 9 → **0** |
| dry_hands__0 | 299 | 82.4% | 10 → **0** |

**我理解判据④的本意是"排除手本身"** —— SAM2 AMG 是无类别的，一定会把手分出来当候选；而这一步管线里还拿不到手 mask（`sam3_hands` 在下游才跑），唯一可用的信号就是 HOI-DETR 的手框。这个动机完全正当。

问题在于：

1. **阈值 0.1 把"这坨是手"和"这坨被手握着"混为一谈了。** 手 mask 大约只填满手框的一半，用框相当于把手的势力范围翻倍。
2. **判据②"碰边就扔"是硬编码的**，没有 CLI 开关。第一人称视频里物体贴边很常见。

**建议**（都只是改默认值，不动代码）：

| 参数 | 现值 | 建议 | 理由 |
|---|---|---|---|
| `--max-hand-overlap` | 0.1 | **0.5 ~ 0.6** | 这一条就能解掉 82% 的误杀 |
| `--min-largest-component-fraction` | 0.98 | **0.85** | 手一遮物体就碎成两块 |
| `--points-per-side` | 16 | **24 / 32** | 小物体采样密度 |
| 判据② | 碰一下就扔 | 改成"边缘接触像素占比 > X%"才扔，并暴露成 CLI 参数 | 需要改代码 |

失败重灾区也完全符合"小物体成堆 + 手重度遮挡"的画像：
`sort_beads__0` 被滤光 50 次、`gather_roll_dice__0` 8 次、`basic_fold__0` 7 次，
其余 `play_mancala` / `assemble_disassemble_tiles` / `stack_remove_jenga` / `assemble_disassemble_legos` / `crumple_flatten_paper` 各 2~3 次。

---

### 3.2 视觉身份记忆：阈值 0.80 对该相似度度量不可达

`visual_instance_memory.py:64-66`：

```python
min_reuse_similarity = 0.80   # ≥0.80 判"还是同一个物体"
max_new_similarity   = 0.45   # ≤0.45 判"是个新物体"
min_reuse_margin     = 0.08
```

落在 (0.45, 0.80) 死区 → `failed_ambiguous_visual_memory` → 冒泡成 `failed_global_identity_linking`（11 条 take）。

**实测 38 条 ambiguous 记录的 best_score：**

```
最高 0.795    中位 0.712    最低 0.491
≥0.80 的:  0 条        ← 一条都没到线
```

而 margin 中位 **0.266**、24/29 条 ≥0.08 —— 说明"最佳匹配 vs 次佳"区分得非常清楚，
**并不是真的分不清谁是谁，而是绝对分数永远够不着 0.80。**

原因在度量本身（`descriptor_similarity`）：

```
相似度 = 0.7 × HSV直方图交集 + 0.2 × 面积项 + 0.1 × 长宽比项
```

后两项即使都满分（=1.0），也要求 HSV 直方图交集 ≥ **0.714** 才能凑到 0.80。
同一物体跨帧（光照变化、被手遮挡、姿态旋转）的 HSV 直方图交集实测只有 0.5~0.75。
**这个阈值和这个度量在标定上是不匹配的。**

**降阈值的收益（保持 `min_reuse_margin=0.08` 不变）：**

| `min_reuse_similarity` | 可判定 |
|---|---|
| 0.80（现状） | **0 / 38** |
| 0.75 | 9 / 38 (23%) |
| **0.70** | **21 / 38 (55%)** |
| 0.65 | 25 / 38 (65%) |

我理解「证据有歧义时不强制伪造结果」是刻意设计（README 和 docstring 都写了），这点我同意。
这里想说的只是：**0.80 这个具体数值使得"歧义"成了默认状态，判据实际上没有起到区分作用。**
建议要么降到 0.70，要么改用对光照/姿态更鲁棒的描述子（比如加入 SAM2 自身的 mask embedding，
或把 HSV 直方图换成 CLIP/DINO patch 特征的余弦相似度）。

---

### 3.3 `conditioning masks overlap` — 一个像素就整条视频失败 ★建议优先修

`failed_fatal_error` 共 19 次，真实异常分布：

| 异常 | 次数 | 涉及 take |
|---|---|---|
| `ValueError: conditioning masks overlap: instance_0001 和 instance_0002` | 14 | furniture_bench_chair、legos、clean_cups、clean_tableware、clip_unclip_papers … |
| 同上（0001/0003、0002/0003） | 3 | build_unstack_lego、load_dispense_ice、open_close_insert_remove_case |
| `ValueError: duplicate object conditioning frame` | 1 | furniture_bench_stool |
| `FileExistsError: output directory is not empty` | 1 | add_remove_lid__0 |

**17/19 是同一个原因。** 抛出点在 `run_persistent_mask_sequence.py:_apply_conditioning_ownership()`：

```python
for index, first_id in enumerate(object_ids):
    for second_id in object_ids[index + 1:]:
        if np.any(conditioning_masks[first_id] & conditioning_masks[second_id]):
            raise ValueError(f"conditioning masks overlap: {first_id} and {second_id}")
```

判据是 `np.any(...)` —— **两个 conditioning mask 只要共享 1 个像素，整条视频就直接异常退出。**

而受影响的 take 全是"零件紧贴在一起"的任务：乐高、叠着的杯子和餐具、夹在一起的纸。
SAM2 在相邻零件边界上产生 1~2 像素的重叠几乎是必然的。

**建议**：把零容忍改成有容差，并且降级处理而不是抛异常。例如：

```python
inter = (m1 & m2).sum()
if inter / min(m1.sum(), m2.sum()) > TOL:      # TOL 建议 0.02~0.05
    raise ...
else:
    # 边界像素按距离归属，或直接从两者中都剔除
```

另外 `FileExistsError: output directory is not empty`（`run_instance_seed_discovery.py:230`）
会让**断点续跑直接炸**：重跑失败条目时如果没先清目录就报错退出。建议加 `--overwrite` 开关。

---

## 四、"成功"的 take 里也有问题

我给 32 条成功的 take 做了叠加可视化（物体 mask + 双手 mask + 接触区间投影回原视频），
看完发现"成功"这个标签比想象的虚。

### 4.1 逐帧采纳率只有 41%

32 条成功 take 的逐帧实例观测：

```
accepted  6190
rejected  8816        ← 被 v17A 的时序离群过滤器拒掉
采纳率      41%
```

极端案例：

| take | 采纳率 | 说明 |
|---|---|---|
| `clean_surface__0` | **1%** | 447 个观测只留下 3 个，名义成功、实际是空的 |
| `insert_remove_shirt_in_tube__0` | 6% | |
| `put_toothpaste_on_toothbrush__0` | 7% | |
| `slot_batteries__0` | 13% | |

被拒的主因是 `area_too_small` + `fragmented_mask` + `area_collapse_vs_history` 三连。
我查了具体现场（`screw_unscrew_bottle_cap__0` 的瓶盖）：

- `raw_mask`（冲突消解后）只剩 **21~143 像素**
- `unresolved_mask`（消解前）有 **3665~10017 像素**

也就是说**瓶盖的 mask 在冲突消解阶段被瓶身吃掉了**，然后才触发 `area_collapse_vs_history`。
所以这里的根因可能不在时序过滤器本身，而在它上游的 mask ownership 消解。这条我没有继续深挖，
如果你要看现场，`unresolved_mask` 那份是有信息的，`raw_mask` 那份已经退化了。

### 4.2 sam3_hands 的三个问题

| 问题 | 数量 | 说明 |
|---|---|---|
| `RuntimeError: No hand detected in entire video` | 9 条 | 整条视频检不出任何手（另有 6 条是 OOM 连累的，不算） |
| **整条视频只出一只手** | 2 条 | `fold_unfold_paper_basic__0`、`insert_remove_shirt_in_tube__0` 全程只有 `left_hand_0.png`，右手一张没有。折纸和套衣管都是双手任务，这是漏检 |

检不出任何手的 9 条：
`fry_bread/0`、`clip_unclip_papers/0`、`crumple_flatten_paper/0`、`load_dispense_ice/0`、
`peel_place_sticker/0`、`insert_remove_tennis_ball/0`、`measure_objects/0`、`sort_beads/0`、`boil_serve_egg/0`

（被 OOM 连累、需要重跑确认的另外 6 条：`assemble_disassemble_soft_legos/0`、
`assemble_disassemble_structures/0`、`declutter_desk/0`、`set_up_clean_up_chessboard/0`、
`setup_cleanup_table/0`、`sleeve_unsleeve_cards/0`）
| **手 mask 包含整条小臂和袖子** | 普遍 | `basic_pick_place__0` 的右手 mask 有 **19 万像素**（占画面 9%） |

第三条对下游接触判定影响很大：我们的接触判据是「手∩膨胀物体 / **手 mask 面积**」，
手 mask 越大分母越大，实打实的抓取也会被摊薄到阈值以下。这条我这边也会改判据（改成
除以 `min(手面积, 物体面积)`，或先取手掌区域），但 sam3 侧如果能把小臂截掉会更干净。

### 4.3 案例：`basic_pick_place__0` —— interaction 窗口是"靠近"不是"抓取"

这条最能说明问题，建议直接看视频：

- v17A 只跟踪了物体 **38/126 帧**（frame 38~85），且这段被标为 `interaction_keyframes`
- 但在整个跟踪窗口内，手离物体最近只到 **86 像素**（接触判据的膨胀半径是 26 像素）
- 画面上看得很清楚：INT 时间轴亮着"交互中"，而手在画面另一头，手机静静躺在桌上
- 另外 sam3 在第 62 帧后把右手跟丢（mask 从 129k 像素塌到 <1k）

这印证了我们内部文档早先记录的担心：**v17A 的 hand-object link 语义是"关联/靠近"而非"抓取"**，
直接当抓取区间用会有假阳性。下游如果要用 `interaction_keyframes` 当抓取相位，需要再加一道
手物协同运动的闸门。

### 4.4 ★案例：`screw_unscrew_bottle_cap/27` —— 同视频同判据的 A/B，接触召回只剩 2%

这条是我们 RL 训练在用的那个 clip（`Screw27`）的原视频，2026-08-08 单独补跑了一次完整 Step1。
它值得单独拿出来，因为**同一条视频已经有一份人工点标路线的产物，可以直接对拍**。

跑通耗时 149s（HOI-DETR 42s + SAM2 传播 42s + sam3_hands 65s），v17A 报 `status: success`。

**但产出是这样的：**

```
object_ids          ['object_0001']          ← 只有瓶身，瓶盖从未被注册成独立零件
跟踪帧范围           17 ~ 156（140 帧）
逐帧状态             accepted 5 帧 / rejected 135 帧
被拒原因             fragmented_mask ×135（100%）
被采纳的 5 帧         152, 153, 154, 155, 156  ← 全在视频末尾
```

**A/B 对拍**（同一条视频、同一个 `contact/detect.py`、同一套参数 `dilation_frac=0.012 / frac_thr=0.02`，
唯一变量是物体 mask 的来源）：

| 物体 mask 来源 | 左手接触 | 右手接触 |
|---|---|---|
| 现有人工点标 + 完整重建路线 | `[[16, 110]]` → **95 帧** | `[[12, 154]]` → **143 帧** |
| v17A 自动路线 | `[]` → **0 帧** | `[[152, 154]]` → **3 帧** |

**右手接触召回 3/143 ≈ 2%，左手 0%。**

**原因**：这条视频是坐姿双手拧盖，两条穿着毛衣的手臂全程横在瓶子前面，把瓶子的 mask 切成多块，
于是逐帧校验的 `min_largest_component_fraction = 0.98` 把 135 帧全部判成 `fragmented_mask`。
只有当人拧完把手撤开（152~156 帧），瓶子才重新变成一个干净的连通块被采纳 —— 也就是说
**恰恰是"正在交互"的那 135 帧被丢掉了，只留下"交互结束后"的 5 帧**。
而接触检测只吃 accepted mask，所以整个拧盖动作产生了零接触信号。

**这暴露了一个比失败率更值得注意的问题：静默失败。**

v17A 的设计原则是"宁可失败也不输出错的"（README 明写），但这一条是 `status: success` 出来的。
`segmentation_report.md` 其实老实报了：

```
| episode_00 | 1 | object_0001 | 5 |          ← take 27，5 帧
| episode_00 | 2 | object_0001 | 126 |        ← take 0 对照，126 帧
| episode_00 | 2 | object_0002 | 36  |
```

**信息在，但没有任何东西拦它** —— 逐帧的 rejected 不会上升为视频级失败。
建议加一道视频级质量门：`accepted 帧数 / interaction 窗口帧数` 低于某个比例（比如 30%）时，
把 status 降级为 `failed_low_coverage` 或至少 `success_low_confidence`，让下游能判别。

（这一条我这边也有责任：我的 `build_db.sh` 判成功的条件只有"manifest 存在 + 手 mask 帧数 > 0"，
所以 take 27 被记成 ✓。我会改成同时要求最低采纳率。）

**关于瓶盖**：瓶盖始终没被分出来，这属于"可分离物体"，本来就是我们计划走 retrieval / 静态扫描的类别
（现有 RL 用的 `bottle_body.obj` + `bottle_cap.obj` 来自 Kailang 的静态扫描
`datasets/recon_kailang/water_bottle_twist_static`，不是从这条视频重建的）。所以这一点不算 v17A 的缺陷，
但它说明**自动路线目前还不能替代静态扫描路线处理这类物体**。

---

## 五、我们用错的地方

v17A README 的流程是**三步**，第三步是人工验收：

> 3. `render_mask_sequence_preview.py` → `diagnostic_review.mp4`
>
> **视觉质量由人工查看预览视频确认。**
> 单元测试通过**不替代**对 `diagnostic_review.mp4` 的人工检查。

**我的批处理脚本只跑了第 1、2 步，第三步整个跳过了。** 这套"精度优先 + 人在环里验收"的管线被
改成了无人值守批跑 200 条，然后把 pass/fail 当成最终结论 —— 质量问题因此积累到第 86 条才暴露。
这个是我的问题，已经在补：我另外写了一个叠加可视化工具
（`RL_Correction/steps/step1_perception/viz_overlay.py`，额外叠了双手 mask、接触区域和时间轴），
批处理脚本也会把第三步补回去。

---

## 六、建议的优先级

| 优先级 | 事项 | 改动量 |
|---|---|---|
| **P0** | `conditioning masks overlap` 改成有容差 + 降级处理（§3.3） | 改代码，~10 行 |
| **P0** | `--max-hand-overlap` 默认 0.1 → 0.5~0.6（§3.1） | 改默认值 |
| **P0** | 加视频级质量门：采纳率过低时 status 降级，别再静默返回 `success`（§4.4） | 改代码，小 |
| **P1** | `min_reuse_similarity` 0.80 → 0.70（§3.2） | 改默认值 |
| **P1** | `--min-largest-component-fraction` 0.98 → 0.85 | 改默认值 |
| **P1** | 输出目录非空改成 `--overwrite` 开关，别让断点续跑炸掉 | 改代码，小 |
| **P2** | 判据②"碰边就扔"改成占比阈值并暴露成 CLI 参数 | 改代码，小 |
| **P2** | sam3_hands：小臂/袖子截断；双手漏检排查 | 需要看 sam3 侧 |
| **P2** | mask ownership 消解：瓶盖被瓶身吃掉的现象（§4.1） | 需要定位 |

**P0 的两条改完，我可以立刻在 86 条已有数据上做一次 A/B 对拍**（改前 vs 改后的 filtered 候选数、
最终成功率、逐帧采纳率），跑一轮大约十几分钟，能直接量出收益。

---

## 七、另一个视角：失败原因本身可能就是我们要的东西

我们原本想解决的问题是：**"能不能光看 mask 就判断这个交互物体该走完整重建，还是该走 retrieval？"**

v17A 的失败原因分类，其实已经是这个判别器了：

| v17A 失败类型 | 对应的物理情形 | 建议走向 |
|---|---|---|
| `failed_no_candidates`（小物体成堆、手重度遮挡） | 珠子、骰子、乐高、积木 | retrieval |
| `conditioning masks overlap`（零件紧贴） | 叠杯、餐具、贴合零件 | retrieval 或分件重建 |
| `failed_ambiguous_visual_memory`（跨 episode 认不出同一个） | 可分离/可形变物体 | retrieval |
| `success` + 高采纳率 | 单体刚性物体 | 完整重建 |

所以调阈值这件事有个取舍，取决于目标：

- 目标是**多拿重建数据** → 按 §6 放宽阈值，接受更多噪声，靠人工看预览片挑
- 目标是**做 retrieval 分流判别器** → **一个参数都不用动**，直接把失败原因当标签用

这两条不冲突，可以并行。我这边打算先做后者（零成本，且是我们原本要的东西）。

---

## 附：数据与复现路径

```
数据库根目录
  /home/lyh/Project/Reconstruct_and_Retarget/Data/VideoPrior/

  build_db.sh                    批处理脚本（三步：HOI-DETR → SAM2 传播 → sam3_hands）
  videos.txt                     200 条清单（101 个任务，每个任务前 2 条）
  logs/build_db_main.log          逐条进度
  logs/<take>.log                 每条的完整 stdout/stderr
  takes/<任务>/<视频号>/
      video.mp4       → 原视频软链
      v17a            → video_mask_sequence 目录软链
      hands           → sam3_hands mask 目录软链
      contact_auto.json           接触区间（我这边生成）
      overlay.mp4                 叠加可视化（我这边生成）
      .done / .failed             状态标记
  viz/<take>_overlay.mp4          32 条叠加视频（87 MB）
  viz/QUALITY.md                  32 条的可用性排名
  _v17a_work/interim/egodex_test/<ID>/    v17A 全部中间产物（含各级 summary.json）
  _recon_work/egodex_test/<ID>/           sam3_hands 产物
```

**建议 Kailang 优先看这三条叠加视频**（问题最典型）：

```
viz/basic_pick_place__0_overlay.mp4              interaction 窗口 ≠ 抓取（§4.3）
viz/clean_surface__0_overlay.mp4                 采纳率 1%（§4.1）
viz/screw_unscrew_bottle_cap__0_overlay.mp4      瓶盖被瓶身吃掉（§4.1）
```

**再看这三条作为对照**（质量好的）：`play_piano__0` / `pour__0` / `push_pop_toy__0`。

本文里所有统计数字都可以从 `_v17a_work` 下的 `summary.json` 复算，需要脚本我可以一并发。
