#!/usr/bin/env bash
# 本地录像: record_local.sh NAME CKPT OUT_MP4 [HYB|OBJ] [RIGHT_CL]  (环境变量与 launch_local.sh 同套, 世界核对 strict)
set -euo pipefail
cd /home/lyh/Project/RL_Correction
NAME=$1; CKPT=$2; OUT=$3; VAR=${4:-HYB}; CL=${5:-1}
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
export UNSCREW_CLIP=17 UNSCREW_DETACH=pull UNSCREW_RIGHT_CL=$CL UNSCREW_PULL_N=${UNSCREW_PULL_N:-3.0} UNSCREW_LEFT_PRIOR=${UNSCREW_LEFT_PRIOR:-Screw17_bottle_left_LDup6.npz}
export SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. TMPDIR=${TMPDIR:-$HOME/tmp}
export POUR_UNLOCK=1,2,3 POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1 POUR_BONUS_DIST=1 POUR_VARIANT=$VAR
export RL_ISAAC_NO_GUARD=${RL_ISAAC_NO_GUARD:-1}
mkdir -p "$(dirname "$OUT")"
echo "[rec] $NAME ckpt=$CKPT out=$OUT $(date +%H:%M)"
$PY tasks/Unscrew/part4/C_Wiring/record_task.py --checkpoint "$CKPT" --out "$OUT" --headless --enable_cameras 2>&1 | tee "${OUT%.mp4}.log" | grep -v "^\[Warning\]\|^\[Info\]\|\[gpu.foundation\|Warning\] \[omni" | tail -30
echo "[rec] done $(date +%H:%M): $(ls -la "$OUT" 2>/dev/null | awk '{print $5" bytes"}')"
