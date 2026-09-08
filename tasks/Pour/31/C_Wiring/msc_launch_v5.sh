#!/bin/bash
# v5 单线发射(msc): msc_launch_v5.sh <NAME> <GPU> <SEED> <VARIANT:HYB|OBJ>
set -e
NAME=$1; GPU=$2; SEED=$3; VAR=$4
cd ~/RL_Correction
source ~/RL_Correction/env_a6000.sh >/dev/null 2>&1 || true
export VEGA_URDF=$HOME/RL_Correction/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa.urdf
export TMPDIR=$HOME/tmp; mkdir -p $TMPDIR
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=$HOME/RL_Correction
export RL_ISAAC_NO_GUARD=1 CUDA_VISIBLE_DEVICES=$GPU
export POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1 POUR_BONUS_DIST=1
export POUR_VARIANT=$VAR
PY=$HOME/miniconda3/envs/env_isaaclab/bin/python
mkdir -p logs
nohup $PY -u tasks/Pour/17/C_Wiring/train_pour.py \
  --name $NAME --num_envs 512 --seed $SEED --headless \
  > logs/${NAME}.out 2>&1 &
echo "launched $NAME on GPU$GPU seed$SEED $VAR pid $!"
