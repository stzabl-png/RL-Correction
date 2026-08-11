#!/bin/bash
# 5 条新 ARCTIC 视频全链条(幂等,可反复重入):
#   stage A/B+基线(run_arctic_video.sh 自带跳过逻辑) -> v2.1 -> Qwen 终审
# 每条视频:final_selection.json 已存在则整条跳过;stage B 半成品自动清理。
cd "$(dirname "$0")"
PY=/home/bangdu/miniforge3/envs/sam3/bin/python
for s in capsulemachine laptop microwave mixer waffleiron; do
  R=runs/s01_${s}_grab_01
  [ -f "$R/final_selection.json" ] && { echo "== $s 已完成,跳过 =="; continue; }
  # stage B 半成品(有 instance 目录但没有最终 manifest)清掉重来
  if [ -d "$R/instance" ] && [ ! -f "$R/instance/video_mask_sequence/video_mask_sequence.json" ]; then
    echo "== $s 清理 stage B 半成品 =="; rm -rf "$R/instance"
  fi
  echo "== $s 开始 $(date +%H:%M:%S) =="
  bash run_arctic_video.sh s01/${s}_grab_01 || { echo "== $s FAILED(管线) =="; continue; }
  CUDA_VISIBLE_DEVICES=1 $PY select_frame_v2.py --run "$R" --hand-mode pixel || { echo "== $s FAILED(v2.1) =="; continue; }
  $PY qwen_final_arbiter.py --run "$R" || echo "== $s FAILED(qwen) =="
done
echo "== CHAIN DONE $(date +%H:%M:%S) =="
