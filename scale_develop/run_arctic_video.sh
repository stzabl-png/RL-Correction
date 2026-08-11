#!/bin/bash
# ARCTIC 单视频端到端: ego 帧序列 -> mp4 -> v17A stage A(HOI-DETR) -> stage B(实例分割/SAM2)
#   -> 基线自动选帧评测(pick_best_frame 原代码)
# 用法: bash run_arctic_video.sh s01/box_grab_01
# 前提: ARCTIC 已下载解压(见 /media/msc-auto/HDD/dataset/arctic/download_10seq.sh)
set -e
SEQ=${1:?用法: bash run_arctic_video.sh s01/box_grab_01}
VIDEO_ID=$(echo "$SEQ" | tr / _)

ARCTIC=/media/msc-auto/HDD/dataset/arctic/arctic_repo/unpack
RECON=/home/bangdu/RL-Correction-recon
WORK=$RECON/scale_develop/runs/$VIDEO_ID
V17A=$RECON/experimental/hoi_detr_v17a
HOI_DETR_ROOT=/home/bangdu/HOI-DETR
HOI_CHECKPOINT=$HOI_DETR_ROOT/checkpoints/epoch_5.pth
SAM2_ROOT=/home/bangdu/HumanVideo2RobotData/third_party/sam2
SAM2_CHECKPOINT=$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt
PY_HOIDETR=/home/bangdu/miniforge3/envs/hoidetr/bin/python
PY_SAM3=/home/bangdu/miniforge3/envs/sam3/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-1}   # 默认 1 号卡,避开训练

mkdir -p "$WORK"

# ── 0. ego 视角(相机 0)帧序列 -> mp4(30fps, ARCTIC 官方帧率) ──
VIDEO=$WORK/$VIDEO_ID.mp4
if [ ! -f "$VIDEO" ]; then
  EGO_DIR=$ARCTIC/images/$SEQ/0
  [ -d "$EGO_DIR" ] || { echo "找不到 ego 帧目录 $EGO_DIR (检查解压结构后改本行)"; exit 1; }
  FIRST=$(ls "$EGO_DIR" | head -1)
  ffmpeg -y -framerate 30 -pattern_type glob -i "$EGO_DIR/*.${FIRST##*.}" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$VIDEO"
fi

# ── A. HOI-DETR 探测 ──
if [ ! -f "$WORK/hoi_detr_probe/detections.json" ]; then
  cd "$V17A"
  DATA_ROOT=$WORK $PY_HOIDETR -m experiments.hoi_detr.run_sequence \
    --dataset arctic --video-id "$VIDEO_ID" --video "$VIDEO" --gpu 0 \
    --hoi-detr-root "$HOI_DETR_ROOT" --checkpoint "$HOI_CHECKPOINT" --checkpoint-authorized \
    --source-revision 1b367292f3833afd64a204bd4d9d84519541d035 \
    --checkpoint-revision 85719ac7bf20b8b67e26206faddf0d9582052046 \
    --frame-stride 1 --visualize
  # 探测产物默认落在 V17A 的 data/interim/arctic/$VIDEO_ID/,拷回 WORK
  cp -r "$V17A/data/interim/arctic/$VIDEO_ID/hoi_detr_probe" "$WORK/" 2>/dev/null || true
fi
DET=$WORK/hoi_detr_probe/detections.json
[ -f "$DET" ] || DET=$V17A/data/interim/arctic/$VIDEO_ID/hoi_detr_probe/detections.json

# ── B. 实例分割 + SAM2 传播 ──
MANIFEST=$WORK/instance/video_mask_sequence/video_mask_sequence.json
if [ ! -f "$MANIFEST" ]; then
  cd "$V17A"
  $PY_SAM3 -m experiments.hoi_detr.run_instance_video_segmentation \
    --video "$VIDEO" --detections "$DET" --output-dir "$WORK/instance" \
    --sam2-root "$SAM2_ROOT" --checkpoint "$SAM2_CHECKPOINT" \
    --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml --gpu 0 ${STAGE_B_EXTRA:-}
fi

# ── C. 基线自动选帧评测 ──
$PY_SAM3 "$RECON/scale_develop/eval_baseline_frame_selection.py" \
  --manifest "$MANIFEST" --video "$VIDEO" --out "$WORK/frame_selection_baseline"

echo "== 完成: $WORK/frame_selection_baseline/ (看 contact_sheet.jpg + report.json) =="
