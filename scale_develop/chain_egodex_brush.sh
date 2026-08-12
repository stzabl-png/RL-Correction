#!/bin/bash
# EgoDex sweep_dustpan/1(扫把 brush)mask+选帧链:HOI-DETR -> 实例分割/SAM2 -> v2.1 选帧 -> Qwen 终审。
# 幂等可重入。之后由 chain_egodex_brush_recon.sh 跑重建(需人工确认物体是 brush 而非簸箕)。
set -e
cd "$(dirname "$0")"
VIDEO_ID=sweep_dustpan_1
SRC_MP4=/home/bangdu/RL-Correction/Data/EgoDex/test/sweep_dustpan/1.mp4
RECON=/home/bangdu/RL-Correction-recon
WORK=$PWD/runs/$VIDEO_ID
V17A=$RECON/experimental/hoi_detr_v17a
HOI_DETR_ROOT=/home/bangdu/HOI-DETR
HOI_CHECKPOINT=$HOI_DETR_ROOT/checkpoints/epoch_5.pth
SAM2_ROOT=/home/bangdu/HumanVideo2RobotData/third_party/sam2
SAM2_CHECKPOINT=$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt
PY_HOIDETR=/home/bangdu/miniforge3/envs/hoidetr/bin/python
PY_SAM3=/home/bangdu/miniforge3/envs/sam3/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}   # 默认 0 号卡(当前空闲)

mkdir -p "$WORK"
VIDEO=$WORK/$VIDEO_ID.mp4
[ -f "$VIDEO" ] || cp "$SRC_MP4" "$VIDEO"

# ── A. HOI-DETR 探测 ──
if [ ! -f "$WORK/hoi_detr_probe/detections.json" ]; then
  cd "$V17A"
  DATA_ROOT=$WORK $PY_HOIDETR -m experiments.hoi_detr.run_sequence \
    --dataset egodex --video-id "$VIDEO_ID" --video "$VIDEO" --gpu 0 \
    --hoi-detr-root "$HOI_DETR_ROOT" --checkpoint "$HOI_CHECKPOINT" --checkpoint-authorized \
    --source-revision 1b367292f3833afd64a204bd4d9d84519541d035 \
    --checkpoint-revision 85719ac7bf20b8b67e26206faddf0d9582052046 \
    --frame-stride 1 --visualize
  cp -r "$V17A/data/interim/egodex/$VIDEO_ID/hoi_detr_probe" "$WORK/" 2>/dev/null || true
  cd - > /dev/null
fi
DET=$WORK/hoi_detr_probe/detections.json
[ -f "$DET" ] || DET=$V17A/data/interim/egodex/$VIDEO_ID/hoi_detr_probe/detections.json
[ -f "$DET" ] || { echo "== FAILED: 找不到 detections.json =="; exit 1; }

# ── B. 实例分割 + SAM2 传播 ──
MANIFEST=$WORK/instance/video_mask_sequence/video_mask_sequence.json
if [ ! -f "$MANIFEST" ]; then
  if [ -d "$WORK/instance" ]; then echo "== 清理 stage B 半成品 =="; rm -rf "$WORK/instance"; fi
  cd "$V17A"
  $PY_SAM3 -m experiments.hoi_detr.run_instance_video_segmentation \
    --video "$VIDEO" --detections "$DET" --output-dir "$WORK/instance" \
    --sam2-root "$SAM2_ROOT" --checkpoint "$SAM2_CHECKPOINT" \
    --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml --gpu 0 ${STAGE_B_EXTRA:-}
  cd - > /dev/null
fi

# ── C. 基线 + track 过滤 + v2.1 选帧 + Qwen 终审 ──
$PY_SAM3 eval_baseline_frame_selection.py \
  --manifest "$MANIFEST" --video "$VIDEO" --out "$WORK/frame_selection_baseline"
$PY_SAM3 filter_tracks.py --run "$WORK" || echo "== track_filter 失败(非致命) =="
$PY_SAM3 select_frame_v2.py --run "$WORK" --hand-mode pixel
$PY_SAM3 qwen_final_arbiter.py --run "$WORK"
echo "== MASK CHAIN DONE $(date +%H:%M:%S): 看 $WORK/final_selection.json,确认 brush 的 object_id 后跑 recon =="
