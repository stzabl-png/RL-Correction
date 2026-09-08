#!/bin/bash
# msc 单线自动录像: msc_autorec_arm.sh <ARM> <GPU> [POUR_flag=val ...]
ARM=$1; GPU=$2; shift 2
cd ~/RL_Correction
source ~/RL_Correction/env_a6000.sh >/dev/null 2>&1 || true
export VEGA_URDF=$HOME/RL_Correction/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa.urdf
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=$HOME/RL_Correction
export CUDA_VISIBLE_DEVICES=$GPU
for kv in "$@"; do export "$kv"; done
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
NAME=E2E_Pour17_$ARM
mkdir -p logs/$NAME/videos
nohup bash tasks/Pour/17/C_Wiring/autorecord_pour.sh \
  logs/${NAME}.out logs/$NAME/stage1_nn logs/$NAME/videos $NAME $PY \
  > logs/${NAME}_autorec.out 2>&1 &
echo "autorec $NAME pid $!"
