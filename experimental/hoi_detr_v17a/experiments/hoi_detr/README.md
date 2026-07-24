# Automatic interaction-object segmentation (v17A)

本目录实现一条纯视觉、无需文本提示的交互物体分割管线：HOI-DETR 负责检测手及交互物体框，SAM2 负责零件级 mask 初始化与双向传播，video registry 负责在多个 interaction episode 之间复用固定实例 ID。

当前分支：`feat/step3-auto-seg-v17a`。

## 目标与输出语义

- 只处理 HOI-DETR 判定为与手交互的区域，不分割场景中的全部物体。
- 每个可区分的交互零部件使用独立、固定的全局 ID；不输出杯盖、杯身、support 等关系或类别语义。
- mask 是当前帧可见区域的 modal mask，不推测被遮挡区域的 amodal mask。
- 公共 mask 只在 interaction 的开始帧到结束帧之间输出；episode 间隙可以保留内部视觉记忆用于 ID 匹配，但正式输出隐藏 mask。
- 相同物体再次被交互时复用已有 ID；明确不同的物体才创建新 ID；证据有歧义时不强制伪造结果。
- 最终 `segmentation_report.md` 只汇报 episode 总数、每个 episode 的 instance 数量，以及每个 instance 有非空输出 mask 的帧数。视觉质量由人工查看预览视频确认。

管线不读取 task text、caption、HDF5 标注或其他文本元数据来影响推理、阈值或身份判断。

## 当前流程

```text
RGB MP4
  -> HOI-DETR hand / firstobject / secondobject 多框与 hf / fs link
  -> 框裁剪、退化框和异常大背景框过滤
  -> interaction episode 开始/结束帧
  -> SAM2 候选 mask 与零件注册
  -> episode 内正向 + 反向传播
  -> 视频级视觉身份映射与固定 ID
  -> video_mask_sequence.json
  -> segmentation_report.md + 可视化 MP4
```

多框会分别参与验证；不会只保留单一最大框，也不会要求“小框和大框必须同时出现”。候选 mask 与已有实例的时序轨迹和像素重合用于判断复用或新增 ID。

## v17A 的遮挡恢复

v17A 针对按压、组装或手部遮挡造成的短暂 mask 缩小/消失：

1. 低面积 partial mask 不再立即删除 ID，而是将该 ID 标记为 `temporarily_occluded`，冻结最后一个可靠状态。
2. 冻结期间的低质量 mask 不更新 registry、空间参考或 `known_union`；完全遮挡帧允许公共可见 mask 暂时为空。
3. 已确认的完整 HOI 区域作为 private interaction envelope 继续传播，但永远不会成为公共实例或视频标签。
4. 在 episode 边界内最多搜索 1 秒；找到连续 3 帧可靠候选后，用新锚点重新 conditioning 原 ID，并正反向回填冻结区间。
5. 默认验证门槛为：flow-warp continuity `0.50`、forward/backward cycle IoU `0.60`、运动补偿后质心步长不超过当前 mask 对角线的 `0.50`。

传播安全机制仍然保留：跨 ID 像素冲突、无效 mask、错误框和有歧义的重捕获不会为了强制产出而被填补或合并。

## 当前边界

- 本分支是 v17A：重点是已注册 ID 的有界遮挡恢复。
- 通用的“双向时序新部件登记”属于后续 v17B；当前版本不能声称已经解决所有先合并后分离、先分离后合并的新增部件场景。
- 当前没有训练、微调或 VLM；种子选择来自 HOI 框、episode 锚点和 SAM2 候选。
- 本模块不进行 pose estimation，也不输出物体关系；下游只消费 mask、固定 ID 和 interaction keyframes。

## 外部资产

仓库只包含适配和管线代码，不提交模型权重或外部 HOI-DETR checkout。

| 资产 | 要求 |
|---|---|
| HOI-DETR source | `AhmadDarKhalil/HOI-DETR`，commit `1b367292f3833afd64a204bd4d9d84519541d035` |
| HOI-DETR checkpoint | `epoch_5.pth`，HF revision `85719ac7bf20b8b67e26206faddf0d9582052046` |
| HOI checkpoint size | `5,855,053,598` bytes |
| HOI checkpoint SHA-256 | `4708fd0ddc5c3d386bad67c31152de58676840b911f6d01219e0092a603277d3` |
| SAM2 source | 仓库子模块 `third_party/sam2` |
| SAM2 default config | `configs/sam2.1/sam2.1_hiera_l.yaml` |

HOI-DETR checkpoint 的模型卡要求获得作者授权。runner 不会下载 checkpoint；只有在操作者已经获得授权时才能传入 `--checkpoint-authorized`。模型权重、测试视频、mask 和运行输出均不得提交到 Git。

## 环境准备

推荐在 Linux x86_64 + NVIDIA GPU 环境运行。HOI-DETR 已验证的 ABI 组合为 Python 3.10、PyTorch 2.11/CUDA 13.0、`mmcv-full==1.7.2`；SAM2 必须能在同一环境正常 import。

从仓库根目录执行：

```bash
git submodule update --init third_party/sam2

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from mmcv.ops import RoIAlign; from sam2.sam2_image_predictor import SAM2ImagePredictor; print('runtime imports ok')"
```

准备以下路径。尖括号内容必须替换为实际路径：

```bash
export VIDEO="<absolute-path-to-input.mp4>"
export DATASET="custom"
export VIDEO_ID="example_0001"
export HOI_DETR_ROOT="<absolute-path-to-HOI-DETR>"
export HOI_CHECKPOINT="<absolute-path-to-epoch_5.pth>"
export SAM2_ROOT="$(pwd)/third_party/sam2"
export SAM2_CHECKPOINT="<absolute-path-to-sam2.1_hiera_large.pt>"
export INSTANCE_OUTPUT="$(pwd)/data/interim/${DATASET}/${VIDEO_ID}/instance_pipeline_v17a"
```

`INSTANCE_OUTPUT` 必须不存在或为空。需要保留旧结果时，请为新运行使用新的输出目录，不要覆盖旧目录。

## 运行完整管线

### 1. 生成 HOI-DETR 多框检测

```bash
python -m experiments.hoi_detr.run_sequence \
  --dataset "$DATASET" \
  --video-id "$VIDEO_ID" \
  --video "$VIDEO" \
  --gpu 0 \
  --hoi-detr-root "$HOI_DETR_ROOT" \
  --checkpoint "$HOI_CHECKPOINT" \
  --checkpoint-authorized \
  --source-revision 1b367292f3833afd64a204bd4d9d84519541d035 \
  --checkpoint-revision 85719ac7bf20b8b67e26206faddf0d9582052046 \
  --frame-stride 1 \
  --visualize
```

检测结果固定写入：

```text
data/interim/{dataset}/{video_id}/hoi_detr_probe/
  detections.json
  upstream_predictions.json
  hoi_detr_report.json
  vis/{video_id}.mp4
  hoi_detr_probe_complete.json
```

同一输入和配置已完成时 runner 会复用 completion marker；确实需要重新推理时显式添加 `--force`。

### 2. 运行 episode、零件注册和 SAM2 传播

```bash
python -m experiments.hoi_detr.run_instance_video_segmentation \
  --video "$VIDEO" \
  --detections "data/interim/${DATASET}/${VIDEO_ID}/hoi_detr_probe/detections.json" \
  --output-dir "$INSTANCE_OUTPUT" \
  --sam2-root "$SAM2_ROOT" \
  --checkpoint "$SAM2_CHECKPOINT" \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --gpu 0
```

v17A 的五个时序参数已有以下默认值，通常无需重复传入：

```text
--max-temporal-validation-seconds 1.0
--min-temporal-confirmation-frames 3
--min-flow-warp-continuity 0.50
--min-cycle-iou 0.60
--max-compensated-centroid-step-diagonals 0.50
```

只有在独立开发实验中才应修改这些阈值；正式比较时必须记录完整命令。

### 3. 生成供人工检查的完整视频

```bash
python -m experiments.hoi_detr.render_mask_sequence_preview \
  --manifest "$INSTANCE_OUTPUT/video_mask_sequence/video_mask_sequence.json" \
  --video "$VIDEO" \
  --output "$INSTANCE_OUTPUT/diagnostic_review.mp4" \
  --watermark "v17A"
```

## 关键输出

```text
${INSTANCE_OUTPUT}/
  summary.json
  box_observations.json
  interaction_episodes.json
  visual_identity_map.json
  episode_XX/
    box_events.json
    sequence_attempt_XX/
      masks/
      raw_masks/
      mask_sequence.json
      diagnostic_mask_sequence.json
      frozen_intervals / reacquisition audit fields
  video_registry/
    reconstruction_registry.json
  video_mask_sequence/
    video_mask_sequence.json
    segmentation_report.md
    summary.json
  diagnostic_review.mp4                # 第 3 步生成
```

`video_mask_sequence.json` 是下游正式消费的固定 ID manifest；mask 仅在 interaction 有效窗口内公开。`diagnostic_mask_sequence.json`、冻结区间、重捕获锚点和 private-envelope 审计用于开发诊断，不应当作额外公共实例。

`segmentation_report.md` 的表格只包含：

| Episode | Instance count | Instance ID | Duration (frames) |
|---|---:|---|---:|

其中 duration 是该 ID 在对应 episode 内具有非空公共输出 mask 的帧数。

## 测试

安装 `pytest` 后，从仓库根目录运行：

```bash
python -m pytest experiments/hoi_detr/tests -q
```

测试覆盖 HOI adapter、episode、组件决定、固定 ID、跨 episode bridge、mask ownership、遮挡冻结/重捕获、private envelope、视频级 manifest 和 segmentation report。单元测试通过不替代对 `diagnostic_review.mp4` 的人工检查。

## 失败与排查

- `output directory is not empty`：为新运行指定一个全新目录。
- `failed_no_interaction_episodes`：HOI 检测没有形成有效 interaction episode；先查看 HOI 可视化和 `box_observations.json`。
- `failed_fatal_error`：查看 `fatal_summary.json`；不要把异常视频伪装成成功输出。
- 某 ID 在遮挡帧暂时为空：允许；检查之后是否以同一 ID 重捕获。
- 两个公共 ID 大面积重叠：传播安全机制会拒绝冲突 mask，不会自动合并。
- 视频与报告只保存在 `data/` 下，默认被 `.gitignore` 排除；每次测试都应保留并向人工 reviewer 汇报准确路径。
