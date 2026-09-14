#!/usr/bin/env bash
# 本机 (4080S 16GB) Clean3 Stage-2 发车: bash local_clean3_task.sh NAME SEED [FLAGS...]
set -euo pipefail
NAME=$1; SEED=$2; shift 2
REPO=/home/lyh/Project/RL_Correction
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH="$REPO"
export POUR_OBJ_MASS=0.3 POUR_OBJ_FRIC=1.0 POUR_PAD_FRIC=1.0
mkdir -p launch_logs
echo "[local_clean3_task] $NAME seed=$SEED flags=[$*]"
setsid env "$@" nohup "$PY" -u tasks/Clean/3/C_Wiring/train_task.py --headless --num_envs "${NUM_ENVS:-512}" --seed "$SEED" --name "$NAME" \
  --max_agent_steps "${MAX_STEPS:-30000000}" > "launch_logs/$NAME.log" 2>&1 < /dev/null &
echo "[local_clean3_task] pid=$! log=launch_logs/$NAME.log"
