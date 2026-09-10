#!/usr/bin/env bash
set -euo pipefail
cd "/home/msc-auto/RL_sweep"
export PYTHONPATH=.
export SHARPA_WANDB=0
export CUDA_VISIBLE_DEVICES=1
export RL_ISAAC_LOCK_DIR=/home/msc-auto/.cache/rl_correction_gpu1
T=/home/msc-auto/RL_sweep/logs/expert/transitions_fullinside
exec /home/msc-auto/miniconda3/envs/isaac/bin/python -u tasks/Sweep/2/C_Wiring/train_sweep.py \
  --task_config tasks/Sweep/new_data/configs/take9_fixed_cube15_human_v2.json \
  --name Task3Take9FixedCube15HumanV2_20260910 \
  --artifact_prefix Task3Take9FixedCube15HumanV2__20260910_policy \
  --method full --num_envs 1024 --seed 42 --max_agent_steps 24000000 \
  --actor_epochs 0 --critic_epochs 0 \
  --diag_every_steps 3000000 --record_every_steps 6000000 \
  --expert25 "$T/sweep2_entry25_transitions.npz" \
  --expert40 "$T/sweep2_entry40_transitions.npz" \
  --expert_full "$T/sweep2_full15_transitions.npz" \
  --failure "$T/sweep2_canonical_failure_transitions.npz" \
  --headless
