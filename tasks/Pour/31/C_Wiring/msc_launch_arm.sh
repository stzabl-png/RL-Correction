#!/bin/bash
# msc 单线发射: msc_launch_arm.sh <ARM字母> <GPU号> [POUR_flag=val ...]
# 例: msc_launch_arm.sh B 0 POUR_SQUEEZE_FF=1
set -e
ARM=$1; GPU=$2; shift 2
cd ~/RL_Correction
source ~/RL_Correction/env_a6000.sh >/dev/null 2>&1 || true
export VEGA_URDF=$HOME/RL_Correction/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa.urdf
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=$HOME/RL_Correction
export RL_ISAAC_NO_GUARD=1 CUDA_VISIBLE_DEVICES=$GPU
for kv in "$@"; do export "$kv"; done
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
NAME=E2E_Pour17_$ARM
mkdir -p logs
nohup $PY -u tasks/Pour/17/C_Wiring/train_pour.py \
  --name $NAME --num_envs 512 --headless \
  > logs/${NAME}.out 2>&1 &
echo "launched $NAME on GPU$GPU pid $!"
