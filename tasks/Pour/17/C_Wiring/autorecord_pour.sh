#!/bin/bash
# 每 1M agent steps 用 last.pth 录一集。读数源: 进度文件(纯数字步数) 或 旧式stdout日志。
# 用法: autorecord_pour.sh <progress_or_log> <ckpt_dir> <out_dir> <name> <python> [parent_pid]
SRC=$1; CKPT=$2; OUT=$3; NAME=$4; PY=$5; PPID_W=${6:-}
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
last=0
while true; do
  if [ -n "$PPID_W" ] && ! kill -0 "$PPID_W" 2>/dev/null; then
    echo "[autorecord] 父进程已退, 收尾退出"; break
  fi
  if [[ "$SRC" == *progress_steps.txt ]]; then
    s=$(cat "$SRC" 2>/dev/null | tr -dc 0-9); m=$(( ${s:-0} / 1000000 ))
  else
    m=$(grep -oE "Agent Steps: *[0-9]+M" "$SRC" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    m=$(( 10#${m:-0} ))
  fi
  if [ "$m" -gt "$last" ] && [ -f "$CKPT/last.pth" ]; then
    echo "[autorecord] ${NAME} @ ${m}M"
    SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 "$PY" "$HERE/record_pour.py" \
      --checkpoint "$CKPT/last.pth" --out "$OUT/${NAME}_${m}M.mp4" \
      --headless --enable_cameras >> "$OUT/record.log" 2>&1
    last=$m
  fi
  sleep 300
done
