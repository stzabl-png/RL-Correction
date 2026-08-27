#!/bin/bash
# 每 1M agent steps 用 last.pth 录一集 (从训练 stdout 解析 "Agent Steps: NNNNM")。
# 用法: autorecord_pour.sh <train.log> <ckpt_dir> <out_dir> <name> <python>
# 注意: 调用方需先 export 好该线的 POUR_* 环境变量 + CUDA_VISIBLE_DEVICES。
LOG=$1; CKPT=$2; OUT=$3; NAME=$4; PY=$5
HERE="$(cd "$(dirname "$0")" && pwd)"
last=0
while true; do
  m=$(grep -oE "Agent Steps: *[0-9]+M" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  if [ -n "$m" ]; then m=$((10#$m)); else m=0; fi
  if [ "$m" -gt "$last" ] && [ -f "$CKPT/last.pth" ]; then
    echo "[autorecord] ${NAME} @ ${m}M"
    SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 "$PY" "$HERE/record_pour.py" \
      --checkpoint "$CKPT/last.pth" --out "$OUT/${NAME}_${m}M.mp4" \
      --headless --enable_cameras >> "$OUT/record.log" 2>&1
    last=$m
  fi
  # 训练进程没了且无新里程 -> 收尾再录一集后退出
  if ! grep -q "" "$LOG" 2>/dev/null; then sleep 60; continue; fi
  if [ -n "$(find "$LOG" -mmin +30 2>/dev/null)" ]; then
    echo "[autorecord] 日志静默>30min, 退出"; break
  fi
  sleep 300
done
