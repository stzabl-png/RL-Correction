#!/bin/bash
# Detached: wait for the OrientCurr training to finish, then EVAL (full g) + RENDER a video, with
# retries for the intermittent carb "Recursion not allowed" boot crash. Launch with setsid:
#   setsid nohup bash watch_and_render.sh >/tmp/oc_watch_outer.log 2>&1 </dev/null & disown
set -u
cd /home/magics/mt_dir/Dynrotate/sharpa_rl_rebuild
PY=/home/magics/miniconda3/envs/sharpa/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
RUN=logs/debug/2026-06-28_18-39-23
TASK=Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-OrientCurr-v0
ST=ORIENTCURR_FINAL_STATUS.txt; : > "$ST"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$ST"; }

log "watcher started; waiting for training (rl_rebuild/scripts/train.py) to finish..."
while pgrep -f "rl_rebuild/scripts/train.py" >/dev/null 2>&1; do sleep 60; done
sleep 10
CK="$RUN/stage1_nn/best.pth"; [ -f "$CK" ] || CK=$(ls -t "$RUN"/stage1_nn/*.pth 2>/dev/null | head -1)
log "training ended. using checkpoint: $CK"

# ---- EVAL at full gravity (retry the intermittent carb crash) ----
for try in 1 2 3 4; do
  log "eval attempt $try"
  $PY rl_rebuild/scripts/eval_metrics.py --task "$TASK" --num_envs 64 --load_path "$CK" > ORIENTCURR_eval.log 2>&1
  if grep -qiE "held fraction|yaw angvel|mean .yaw" ORIENTCURR_eval.log; then
    log "eval OK:"; grep -iE "held fraction|yaw angvel|mean .yaw|cumulative" ORIENTCURR_eval.log >> "$ST"; break
  fi
  log "eval attempt $try crashed/failed; retrying"; sleep 8
done

# ---- RENDER a video (retry) ----
for try in 1 2 3 4; do
  log "render attempt $try"
  $PY rl_rebuild/scripts/render_video.py --task "$TASK" --num_envs 4 --video_length 320 \
     --load_path "$CK" --device cuda:0 --out_dir videos_gx_orientcurr \
     --eye="0.6,0.6,0.75" --lookat="0.0,0.0,0.55" > ORIENTCURR_render.log 2>&1
  if ls videos_gx_orientcurr/*.mp4 >/dev/null 2>&1; then
    log "render OK -> $(ls -t videos_gx_orientcurr/*.mp4 | head -1)"; break
  fi
  log "render attempt $try crashed/failed; retrying"; sleep 8
done
log "WATCHER DONE"
