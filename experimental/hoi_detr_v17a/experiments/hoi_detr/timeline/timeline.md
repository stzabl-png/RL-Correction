# timeline：交互三段分割使用说明

输入一条 raw MP4，依次运行下面四个脚本，为**每只手**输出一条「接近 — 交互 — 离开」三段可视化视频 `three_phase_timeline.mp4`。规则要点：每只手独立处理；手和**任意合理物体**有 hf 白线即计入交互（只过滤明显不合理的大背景框）；交互窗口取该手最早连线段起点到最晚连线段终点，中间空隙一并归入交互；没有连线帧的手不产出结果（单手任务自然只出一条）。

Qwen 只参与两处：判定左右手（投票，平均 x 兜底）、输出任务类型和每只手的主要物体名（仅作元数据展示，不参与过滤）。每条视频固定 2 次调用，与视频长度无关。

## 环境准备

前提：已具备 `docs: document v17A segmentation pipeline`（be0791b）之前的全部环境
（`hoidetr` / `sam3` 两个 conda env、HOI-DETR checkout 与 checkpoint、SAM2 子模块）。
在此基础上只需增量安装一次：

```bash
/home/bangdu/miniforge3/envs/sam3/bin/pip install openai   # Qwen 客户端依赖
```

另外确认系统有 `ffmpeg`（转 H.264 用）。Qwen API key 已内置默认值，如需覆盖：
`export DASHSCOPE_API_KEY=<key>`。

## 运行（以 data/EgoDex/test/insert_remove_usb/0.mp4 为例）

```bash
cd <repo根目录>
DS=insert_remove_usb  VID=0
VIDEO=$(pwd)/data/EgoDex/test/$DS/${VID}.mp4
BASE=data/interim/$DS/$VID

# ① HOI-DETR 检测 —— hoidetr 环境（py3.7），物理 1 号卡
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
/home/bangdu/miniforge3/envs/hoidetr/bin/python -m experiments.hoi_detr.run_sequence \
  --dataset $DS --video-id $VID --video $VIDEO --gpu 0 \
  --hoi-detr-root /home/bangdu/HOI-DETR \
  --checkpoint /home/bangdu/HOI-DETR/checkpoints/epoch_5.pth --checkpoint-authorized \
  --source-revision 1b367292f3833afd64a204bd4d9d84519541d035 \
  --checkpoint-revision 85719ac7bf20b8b67e26206faddf0d9582052046 \
  --frame-stride 1 --visualize

PY=/home/bangdu/miniforge3/envs/sam3/bin/python   # ②③④ 均用 sam3 环境，无 GPU

# ② 左右手拆分 + 任意物体连线过滤（Qwen 2 次调用）
$PY -m experiments.hoi_detr.timeline.dual_hand_frame_split \
  --video $VIDEO --detections $BASE/hoi_detr_probe/detections.json \
  --output-dir $BASE/anylink_split

# ③④ 每只手：合并交互段，再压成三段（left_hand / right_hand 目录存在才跑）
for side in left right; do
  TRK=$BASE/anylink_split/${side}_hand/filtered_track.json
  [ -f "$TRK" ] || continue    # 该手无连线帧则跳过
  $PY -m experiments.hoi_detr.timeline.mark_interaction_segments \
    --video $VIDEO --filtered-track $TRK \
    --output-dir $BASE/anylink_timeline/${side}_hand
  $PY -m experiments.hoi_detr.timeline.mark_three_phase_timeline \
    --video $VIDEO \
    --segments $BASE/anylink_timeline/${side}_hand/interaction_segments.json \
    --filtered-track $TRK \
    --output-dir $BASE/anylink_three_phase/${side}_hand
done
```

## 输出

| 文件 | 说明 |
|---|---|
| `anylink_three_phase/{side}_hand/three_phase_timeline.mp4` | **最终可视化**（H.264+faststart）：横幅与边框按相位着色，蓝=接近、绿=交互、橙=离开；底部时间轴标交互起止帧号 |
| `anylink_three_phase/{side}_hand/three_phase.json` | 三段边界 + 合并前的原始交互段 |
| `anylink_timeline/{side}_hand/segments_timeline.mp4` / `interaction_segments.json` | 细粒度交互段（三段合并前），绿=段、黄=桥接、灰=段外 |
| `anylink_split/dual_hand_analysis.json` | 左右手判定依据、Qwen 元数据（task_type、每手物体名）、每手连线帧清单 |
| `anylink_split/{side}_hand/kept_frames.mp4` / `frames/*.jpg` | 该手连线帧检查视频/图片 |

## 注意

- ① 有完成标记，重跑自动跳过；确要重新推理加 `--force`。
- GPU 必须用 `CUDA_VISIBLE_DEVICES=<物理卡号>` + `--gpu 0` 的组合指定；直接 `--gpu 1` 会触发混设备 CUDA 错误。
- 框过滤仅两条：面积 > 70% 画幅（`--max-frame-area-fraction 0.7`），或框贴住 ≥3 条画面边缘；其余物体框全部保留。
- ③ 的断口规则：≤4 帧自动桥接，>4 帧切段，跨度 <6 帧的段丢弃；Qwen 断口仲裁默认关闭，
  需要时加 `--qwen-arbitration`（适用于把细粒度段本身当交付物的场景）。
- Qwen 的 task_type / 物体名只是元数据；哪只手出结果由数据决定（有无连线帧）。
- 单手旧管线（`qwen_interaction_object_filter.py`，按单一任务物体过滤）仍保留可用，见 git 历史版本说明。

---

## 多段版扩展（本仓新增，2026-08）

原版④把全片压成**单个**交互窗口；多循环任务（如 stock_unstock_fridge 反复取放）
改用下面两个新脚本，保留**每个**交互循环各自的「靠近-交互-离开」。

### ②' 全长左右手交互视频 `render_per_hand_interaction_video.py`

与②同样的手轨迹 + Qwen 判手逻辑，但输出**全长**视频（非仅连线帧）：每帧只画指定手
的手框、其 hf-link 物体框与连线，另一只手的交互完全过滤；该手交互时加绿(右)/蓝(左)边框。

```bash
$PY -m experiments.hoi_detr.timeline.render_per_hand_interaction_video \
  --video $VIDEO --detections $BASE/hoi_detr_probe/detections.json \
  --output-dir <out>          # [--no-qwen] 跳过 Qwen 判手，用平均 x 兜底
# 产物: {left,right}_hand_interactions.mp4 + per_hand_links.json(逐帧 link + 判手依据)
```

### ③' 多段三相位分割 `mark_multi_phase_segments.py`

消费 ②' 的 `per_hand_links.json`。断口三级规则：
**≤4 帧（`--bridge-gap`）自动桥接 → [5,30] 帧（`--qwen-gap-min/max`）Qwen 遮挡仲裁 →
更长直接切段**；跨度 <6 帧（`--min-seg`）的段丢弃。仲裁对每个断口抽帧发 Qwen 两问
（q1 手是否仍持物且物被遮挡；q2 断口前后是否同一物体/部件-整体），**双 yes 才桥接**，
API 失败保守不桥接；每手调用上限 `--qwen-max-calls`(默认 8)，`--no-qwen-arbitration`
可整体关闭（回退纯 link 基线）。每两个相邻交互段之间的空档从中点切开：前半=上一循环
「离开」，后半=下一循环「靠近」。

```bash
$PY -m experiments.hoi_detr.timeline.mark_multi_phase_segments \
  --video $VIDEO --detections $BASE/hoi_detr_probe/detections.json \
  --per-hand-links <out>/per_hand_links.json --output-dir <out>/multi_phase
# 产物: {left,right}_hand_multi_phase.mp4  可视化（顶部横幅按相位变色+循环编号；
#         底部醒目三色进度条：蓝=靠近/绿=交互/橙=离开，带图例、相位分隔线、
#         交互段起止帧号、三角游标）
#       {left,right}_hand_multi_phase.json 各循环三相位边界 + qwen_gap_records
#         （每个仲裁断口的 q1/q2 答案与是否桥接，查询图存 qwen_gap_queries/，可审计）
```

实测参考：11 个 [5,30] 断口人工核验，仲裁裁决 11/11 正确——sweep_dustpan 真放手
不误桥（右手 3→1 循环、左手保持切分），lock_unlock_key 钥匙插锁的部件融合正确桥接。
