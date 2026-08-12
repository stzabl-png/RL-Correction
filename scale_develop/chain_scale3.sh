#!/bin/bash
# scale 实验前置:laptop/microwave 的 mask 管线 + 选帧(幂等,可反复重入)
cd "$(dirname "$0")"
PY=/home/bangdu/miniforge3/envs/sam3/bin/python
for s in laptop microwave; do
  R=runs/s01_${s}_grab_01
  [ -f "$R/final_selection.json" ] && { echo "== $s 已完成,跳过 =="; continue; }
  if [ -d "$R/instance" ] && [ ! -f "$R/instance/video_mask_sequence/video_mask_sequence.json" ]; then
    echo "== $s 清理 stage B 半成品 =="; rm -rf "$R/instance"
  fi
  echo "== $s 开始 $(date +%H:%M:%S) =="
  bash run_arctic_video.sh s01/${s}_grab_01 || { echo "== $s FAILED(管线) =="; continue; }
  $PY filter_tracks.py --run "$R" || echo "== $s FAILED(track_filter,非致命) =="
  CUDA_VISIBLE_DEVICES=1 $PY select_frame_v2.py --run "$R" --hand-mode pixel || { echo "== $s FAILED(v2.1) =="; continue; }
  $PY qwen_final_arbiter.py --run "$R" || echo "== $s FAILED(qwen) =="
done
echo "== CHAIN DONE $(date +%H:%M:%S) =="
