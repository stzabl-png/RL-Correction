# recon_kailang

该目录集中存放 Kailang 为静态物体 `video → reconstruction → Step4 RL` 链路新增的独立功能。
它与 `recon_pipeline/` 的既有实现隔离：必须修改通用行为的兼容性补丁仍原位更新；新入口、
测试和后续扩展统一放在这里，便于组内识别作者和 review 范围。

## 当前内容

```text
recon_kailang/
  v17_mask_adapter/
    import_v17a_masks.py
  tests/
    test_v17_mask_adapter.py
```

相关的原位通用更新包括：

- `recon_pipeline/sam3_hands/run_sequence.py`：允许显式传入本地 SAM3/SAM3.1 checkpoint；
- `recon_pipeline/sam3d_scale/run_sequence.py`：最终 mesh 固定采用 SAM3D visible-surface 尺度；
- `.gitattributes`：shell script 固定 LF，避免 Linux runner 被 Windows 行尾破坏。

## v17A mask adapter

当上游 Step3 已选定 interaction episode、global object ID 和最佳重建帧时，adapter 将
`persistent_mask_sequence_v1` 转为重建管线原有的 `sam2_object` 目录。它不自行选择
episode、物体或帧，不再次运行 SAM2，也不打开人工标注 UI。

从 `ego_pipeline/Reconstruction/` 执行：

```bash
python3 recon_kailang/v17_mask_adapter/import_v17a_masks.py \
  --dataset egodex \
  --video-id test__basic_pick_place__1 \
  --video /path/to/basic_pick_place/1.mp4 \
  --manifest /path/to/v17a/video_mask_sequence.json \
  --source-object-id object_2 \
  --reconstruction-frame 45 \
  --object-name phone
```

导入器会验证 manifest 状态、视频帧数和分辨率，只复制所选 ID 的 accepted mask，其他帧
写入同尺寸零 mask，并生成兼容的 `label_prompt.json`、`v17a_import.json` 和 completion
marker。导入完成后从 HaWoR 继续，不要再次运行 `sam2_object/run_sequence.py`：

```bash
python3 recon_pipeline/run_pipeline.py \
  --dataset egodex --video-id test__basic_pick_place__1 \
  --steps vipe,sam3_hands,hawor,sam3d,sam3d_scale,fp_pose,fuse --gpu 0
```

## 最终尺度与姿态契约

`object_mesh_scaled_final.obj` 的生产尺度采用 SAM3D 朝向、object mask 和 ViPE depth 得到的
visible-surface principal-axis scale。FoundationPose stage 2 只保留诊断信息，不再次缩放
mesh；逐帧 6D 姿态仍由后续 `fp_pose` 提供。元数据应包含：

```text
scale_method = sam3d_visible_surface_principal_axis
foundationpose_stage2_diagnostic_only = true
```

## 测试

在 Linux/HV2RD 环境、仓库根目录运行：

```bash
python -m unittest discover \
  -s ego_pipeline/Reconstruction/recon_kailang/tests \
  -p 'test_*.py'
```

模型权重、视频、mask、OBJ/USD、NPZ 和运行输出不属于本目录，不得提交到 Git。
