# 视频 → 重建 的整合状态（2026-08-12）

**给三方看的对账表。** 目前自动 mask（Kailang）、选帧/尺度（杜邦）、选帧追踪（Kailang）
各自在自己分支上推进，整合方（用户）在合流时反复撞到问题却不知道该找谁改。
本文把**今天实测踩到的每一个坑**写清楚：现象、根因、证据、该谁改。

一句话现状：**整条链能跑通（laptop 已端到端验证），但接口约定不统一，
每次合流都要手工适配；而失败时报错指向的位置往往不是真正的病灶。**

---

## 0. 现在的实际链路（以 `reconstruct.sh` 为准）

```
mp4
 ├─ [标注] 默认全自动 auto_label_v17a  ← Kailang（HOI-DETR + SAM2 传播）
 │         产物: label_prompt.json / frame_plan.json / video_mask_sequence.json
 │         人工点选只是 fallback（--no-auto-label + 手写 label_prompt）
 └─ [九步] vipe → sam3_hands → sam2_object → hawor
            → sam3d → sam3d_scale → fp_pose → fuse → confidence
```

选帧器（杜邦，`scale_develop/`）目前**不在九步之内**，是外挂：读 v17A 的
`video_mask_sequence.json` → 写 `frame_plan.json` 的 `sam3d_frame` → 九步再消费。
这个接口是对的（`frame_plan.py` 就是为此设计的），但**没有接进 `reconstruct.sh`**，
每次要人工跑三条命令并手工搭 run 目录。

---

## 1. ★ 最需要统一的：单一入口

现在跑一条视频要在三个地方切换：

| 环节 | 入口 | 工作目录约定 |
|---|---|---|
| 九步重建 | `ego_pipeline/reconstruct.sh` | `Output/ReconstructOutput/interim/<dataset>/<video_id>/` |
| v17A 自动标注 | `ego_pipeline/bin/auto_label_v17a.py` | `experimental/hoi_detr_v17a/data/interim/<dataset>/<video_id>/` |
| 选帧器 | `scale_develop/select_frame_v2.py --run <dir>` | `runs/<id>/`（第三套布局）|

**三套目录布局互不相同**，靠 `scale_develop/adapt_run_dir.py`（软链适配器）粘合。
建议：把选帧器作为九步之间的一步接进 `reconstruct.sh`（它已支持 `--steps=`），
由管线统一给路径，取消第三套布局。

---

## 2. 今天实测踩到的坑（按"该谁改"分组）

### A. 给 Kailang（自动 mask / v17A）

**A1. ★ SAM2 传播在多色物体上整段失效，且失败被下游误报成别的原因**

实测 `s07/ketchup_grab_01`（番茄酱瓶：红盖 + 绿身 + 深色标签）：

| | ketchup | laptop（对照，单色木板）|
|---|---|---|
| 视频总帧 | 703 | 828 |
| manifest 覆盖 | 301 | 733 |
| **raw_mask 全空（0px）** | **265** | **0** |
| 有内容 | 36 | 862 |
| 面积中位 | **0** | 73710px |

**88% 的帧 SAM2 没有输出任何 mask。** 质量门如实报告了这件事，但下游看到的是
`rejected_temporal_outlier` 和 `INSUFFICIENT_TRACK`，**读起来像"门太严"，实际是"没数据"**。
排查这个坑花了整整一轮。

建议：传播步骤在**空 mask 比例超阈值时显式失败并给出原因**，而不是把空 mask 写进 manifest
让下游自己发现。

**A2. 跨周期身份链接失败时，同一 `instance_id` 在不同 episode 指不同物体**

`s05/laptop_grab_01` 实测（可视化 `laptop_v17a_overlay.mp4`）：

| | instance_0001 | instance_0002 |
|---|---|---|
| episode_00 | **白色桌垫** | 木板 |
| episode_01 | 木板 | — |
| episode_02 | **白色桌垫** | — |
| episode_03 | 木板 | **白色桌垫** |

**身份链接判失败是正确的**（它们本来就不是同一个东西），不要通过放松阈值去"修"——
那会把木板和桌垫强行连成一个物体。真正的问题是 episode 级种子选错了物体。

`adapt_run_dir.py` 会在链接失败时合成 `*_synth` 视频级 manifest，
provenance 里写明"假设跨 episode 同名 instance 是同一物体"——**该假设在此不成立**，
使用合成 manifest 的下游需要知道这一点。

**A3. interim 产物被清理后，重跑会静默走到错误分支**

`s05/laptop` 的 `interim/.../vipe/` 和 `sam2_object/video_segmentation/masks/` 都被清空，
但最终目录 `Output/ReconstructOutput/arctic15/s05/laptop_grab_01/masks/` 保留了 414 帧。
重跑 `--force` 时 sam3d 报 `FileNotFoundError: .../frame_000383_masks/object_0.png`，
指向 interim，**看不出"最终目录其实有"**。

建议：步骤在 interim 缺失但最终产物存在时，要么自动回读，要么明确提示两处不一致。

### B. 给杜邦（选帧 / 尺度）

**B1. 硬编码路径（已在我们这边改好，建议上游一并改）**

`scale_develop/*.sh` 与 `select_frame_v2.py` 里有 `/home/bangdu/...`、`/media/msc-auto/...`
共 8 处。我们的对应物是：

| 他的 | 我们的 |
|---|---|
| `/home/bangdu/RL-Correction-recon` | `~/Reconstruct_and_Retarget` |
| `/home/bangdu/HOI-DETR` | `third_party/HOI-DETR`（ckpt `epoch_5.pth` 已在）|
| `/home/bangdu/HumanVideo2RobotData/third_party/sam2` | `third_party/sam2` |
| `envs/hoidetr` | 我们的 v17A 跑在 **`hawor`** 环境 |
| `envs/sam3` | 同名 `sam3` ✓ |

**环境名不同不等于缺环境** —— 我一开始据此判断"缺三样，跑不起来"，是错的：
HOI-DETR 权重、SAM2、全部环境我们都有，远端已有 89 条 v17A 产物为证。

**B2. `eval_baseline_frame_selection.py` 的 schema 白名单拒绝 `*_synth` 变体**

断言只接受 `persistent_video_mask_sequence_v1` / `persistent_mask_sequence_v1`，
而身份链接失败时合成出来的是 `..._v1_synth`，**结构逐字段一致**。
我们这边已放开并加注释。建议上游直接接受该变体，或由合成方改用正式 schema 名。

**B3. `select_frame_v2.py` 在空 mask 上崩溃（无守卫）**

绕过 v17A 质量门让所有帧进候选后：
`ValueError: zero-size array to reduction operation minimum which has no identity`。
正常情况下质量门保证非空，但一旦上游变化就会崩。建议加空 mask 跳过。

**B4. 选帧器只逐 track 选帧，"哪个 track 是目标"由第三段 Qwen 决定 —— 这一点容易被误解**

我一度据此判断"第 1 步塌了所以选帧器没用"，**是错的**。
`qwen_final_arbiter.py` 的主体仲裁正是为此设计（`interaction_target` 字段）。
建议在 README 顶部显式写清三段各自回答什么问题。

### C. 给整合方（我们自己）

**C1. `reconstruct.sh` 的 flag 透传：空格写法的值会被吞掉**

`--gpu-mem-budget-mb 38885` → 只传了 flag，值被 `[0-9]*` 分支吃成"跑 N 条"。
**必须用 `--gpu-mem-budget-mb=38885` 等号写法。** 与 CLAUDE.md 记的 `--gpu-ids` 同一个坑，
建议直接修 `reconstruct.sh` 的参数解析。

**C2. 批量队列的标注闸不看请求了哪些步骤**

只跑 `--steps=vipe,sam3_hands,hawor`（纯手部实验，不需要物体）也会被
`Missing labels with --skip-label` 拦住。我们用占位 `label_prompt.json` 绕过。
建议闸只在请求的步骤含 `sam2_object` 时生效。

**C3. 队列的 GPU 预算写死 43000MB，不看别人占了多少**

共享机上导致 sam3d_scale 的 nvdiffrast `cudaMalloc` 失败（`Cuda error 2` = OOM），
一夜 5 次。已用"按实际空闲算预算"绕过。建议启动时读实际 free。

**C4. 帧率没有统一，帧号跨模块传递时不带出身**

`arctic`(30fps) 与 `arctic15`(15fps) 两批产物并存，选帧器输出的 f768 属于 30fps，
我们的真值验证全在 15fps —— 换算一次才能对上。
`label_prompt.json` / `frame_plan.json` / `contact_auto.json` / `video_mask_sequence.json`
存的都是裸帧号，**没有 fps 字段**。
另外这些常量是以"帧数"表达的时间阈值，换 fps 会静默变含义：
`FP_ONSET_OFFSET=10`、`min_positive_frames=3`、`confirmation_window=5`、
`max_gap_frames=1`、`phase.detect` 的 `--min-len/--max-gap`、`smooth_sigma=2.0`。
正确做法在仓里已有样板：`interaction_episodes.py` 的 `keyframe_delay_seconds` 用秒表达
再乘 fps。建议统一到 15fps + 各 json 加 `fps`/`source_video` 字段。

---

## 3. 已验证能跑通的部分（laptop 端到端）

| 步骤 | 结果 |
|---|---|
| v17A 自动标注 | 862 个 mask 条目，无空帧 |
| `filter_tracks` | 2 个 track 保留（hand_link 89.5% / 76.0%）|
| `select_frame_v2` | 选出 f768 / f705 |
| **`qwen_final_arbiter`** | **`interaction_target=instance_0001`（木板），桌垫降为次要** |

Qwen 对选中帧的描述："一个浅色木质盒子（或木盖）"，`source=qwen_accept`（非兜底）。
主体仲裁的对比图存在 `qwen_audit/target_{A,B}_*.jpg`，可审计。

**结论：自动认物体这条路是通的，卡点在 SAM2 传播的稳定性，不在选帧或仲裁。**

---

## 4. 复现用的东西

| 用途 | 位置 |
|---|---|
| mask 叠加视频（看传播在哪帧跟丢）| `/tmp/overlay_video.py`（远端）|
| v17A 逐 episode 对照图 | `/tmp/viz_v17a.py`（远端）|
| run 目录适配器 | `scale_develop/adapt_run_dir.py`（仓内已有）|
| 绕过质量门做对照 | `/tmp/bypass_gate.py`（远端）|

vLLM（Qwen 终审用）：`GPU=7 bash ~/bin/start_vlm.sh`，端口 8807，
`QWEN_BASE_URL=http://127.0.0.1:8807/v1` + `QWEN_MODEL=vlm`。
⚠ `sam3` 环境原本缺 `openai` 包，已装 3.0.0。
