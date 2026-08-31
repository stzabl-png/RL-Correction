#!/usr/bin/env bash
# Record one immutable policy rollout for every 3M diagnostic checkpoint.
set -uo pipefail
CHECKPOINTS=${1:?logs/checkpoints dir}; VIDEOS=${2:?outputs_video dir}
PREFIX=${3:?artifact prefix}
PY=${4:?isaac python}; PARENT=${5:?training pid}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="$ROOT" SHARPA_WANDB=0 CUDA_VISIBLE_DEVICES=0
LOCK_DIR=${RL_ISAAC_LOCK_DIR:-$HOME/.cache/rl_correction}
PAUSE="$LOCK_DIR/pause.request"
mkdir -p "$LOCK_DIR"
cleanup_pause() {
  if [ -f "$PAUSE" ] && [ "$(cat "$PAUSE" 2>/dev/null)" = "$$" ]; then
    rm -f "$PAUSE"
  fi
}
trap cleanup_pause EXIT INT TERM
mkdir -p "$CHECKPOINTS" "$VIDEOS"
SEEN="$CHECKPOINTS/.${PREFIX}.recorded"; touch "$SEEN"
while kill -0 "$PARENT" 2>/dev/null; do
  for meta in "$CHECKPOINTS"/"${PREFIX}"_*M/metrics.json; do
    [ -e "$meta" ] || continue
    node=$(dirname "$meta"); base=$(basename "$node"); tag=${base#${PREFIX}_}
    grep -qxF "$tag" "$SEEN" && continue
    ckpt="$node/checkpoint.pth"
    [ -f "$ckpt" ] || continue
    video_dir="$VIDEOS/${PREFIX}_${tag}"
    video="$video_dir/policy.mp4"
    frames="$video_dir/topdown_frames"
    trace="$node/rollout.npz"
    mkdir -p "$video_dir"
    echo "[autorecord] $tag checkpoint=$ckpt" >&2
    printf '%s\n' "$$" > "$PAUSE"
    timeout -k 30 ${SWEEP_REC_TIMEOUT:-2400} "$PY" -u \
      "$HERE/record_sweep.py" --checkpoint "$ckpt" --out "$video" \
      --topdown_frames_dir "$frames" --topdown_tail_on_failure \
      --trace "$trace" --headless --enable_cameras \
      >> "$node/record.log" 2>&1
    status=$?
    sleep "${RL_ISAAC_SETTLE:-10}"
    cleanup_pause
    if [ -s "$video" ] && [ -s "$trace" ] \
       && [ "$(find "$frames" -maxdepth 1 -name 'frame_*.png' | wc -l)" -gt 0 ]; then
      echo "$tag" >> "$SEEN"
    else
      echo "[autorecord] $tag failed status=$status; will retry" >&2
    fi
  done
  sleep 30
done
