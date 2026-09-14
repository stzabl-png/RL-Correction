#!/usr/bin/env bash
# Clean3 Stage-2 (最简六项配方, 台账 §5.7) 发车: bash denso_clean3_task.sh NAME GPU SEED [extra args]
#   例: bash denso_clean3_task.sh Clean3_task_s42 0 42
set -euo pipefail
NAME=$1; GPU=$2; SEED=$3; shift 3
REPO=$HOME/RL_Correction
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH="$REPO" RL_ISAAC_NO_GUARD=1 CUDA_VISIBLE_DEVICES=$GPU
export TMPDIR=${TMPDIR:-$HOME/tmp/clean3}; mkdir -p "$TMPDIR" logs
export POUR_OBJ_MASS=0.3 POUR_OBJ_FRIC=1.0 POUR_PAD_FRIC=1.0
for f in tasks/Clean/3/A_Design/L2_Reference/clean3_reference_v1.npz tasks/pregrasp/priors/Clean3_plate_left.npz \
         tasks/pregrasp/priors/Clean3_sponge_right.npz tasks/Clean/3/C_Wiring/clean_task_env.py tasks/Clean/3/A_Design/L3_Learning/progress_batch.py; do
  [[ -s "$f" ]] || { echo "缺部署产物: $f" >&2; exit 4; }
done
echo "[denso_clean3_task] $NAME GPU=$GPU seed=$SEED PY=$PY"
nohup "$PY" -u tasks/Clean/3/C_Wiring/train_task.py --headless --num_envs "${NUM_ENVS:-512}" --seed "$SEED" --name "$NAME" \
  --max_agent_steps "${MAX_STEPS:-30000000}" "$@" > "$HOME/$NAME.log" 2>&1 &
echo "[denso_clean3_task] pid=$! log=$HOME/$NAME.log"
