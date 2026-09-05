#!/usr/bin/env bash
# Record one immutable policy rollout at the requested checkpoint interval.
set -uo pipefail
CHECKPOINTS=${1:?logs/checkpoints dir}; VIDEOS=${2:?outputs_video dir}
PREFIX=${3:?artifact prefix}
PY=${4:?isaac python}; PARENT=${5:?training pid}
RECORD_EVERY=${6:?record interval in agent steps}; METHOD=${7:?ablation method}
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="$ROOT" SHARPA_WANDB=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
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
    tag_m=${tag%M}; tag_steps=$((10#$tag_m * 1000000))
    [ $((tag_steps % RECORD_EVERY)) -eq 0 ] || continue
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
      "$HERE/record_sweep.py" --checkpoint "$ckpt" --method "$METHOD" --out "$video" \
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
