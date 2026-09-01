#!/usr/bin/env bash
# Unscrew/17 拔盖变体 本地发射 (4080S 16GB). 用法: bash tasks/Unscrew/17/C_Wiring/launch_local.sh NAME SEED [NUM_ENVS] [HYB|OBJ] [RIGHT_CL 0/1]
#   台账 tasks/Unscrew/17/A_Design/DECISIONS.md §5/§6. 开关: UNSCREW_DETACH=pull (U2), 原生先验 (§8), U9 闭环 (RIGHT_CL).
#   POUR_UNLOCK=1,2,3 必带 (同事 T2-9 课程死锁尸检). autorec 关 (它绕过 gpu_guard 并行录像, 16GB 顶不住), 录像事后补.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
NAME=${1:?name}; SEED=${2:-51}; NE=${3:-512}; VAR=${4:-HYB}; CL=${5:-1}
PY=${PY:-/home/lyh/luhr/MagicSim/.venv/bin/python}
mkdir -p logs launch_logs
export UNSCREW_CLIP=17 UNSCREW_DETACH=pull UNSCREW_RIGHT_CL=$CL UNSCREW_PULL_N=${UNSCREW_PULL_N:-3.0}
export SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. TMPDIR=${TMPDIR:-$HOME/tmp}
export POUR_UNLOCK=1,2,3 POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1 POUR_BONUS_DIST=1 POUR_VARIANT=$VAR
mkdir -p "$TMPDIR"
echo "[launch] $NAME seed=$SEED envs=$NE var=$VAR right_cl=$CL detach=pull pull_n=$UNSCREW_PULL_N" | tee -a launch_logs/${NAME}.log
exec $PY -u tasks/Unscrew/part4/C_Wiring/train_task.py --name "$NAME" --seed "$SEED" --num_envs "$NE" --no_autorec --headless "${@:6}" >> launch_logs/${NAME}.log 2>&1
