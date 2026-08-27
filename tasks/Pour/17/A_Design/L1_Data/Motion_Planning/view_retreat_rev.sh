#!/usr/bin/env bash
# Retreat 试验版: Approach 精确倒放 (完整抓形起手 -> 六级梯倒放松开边退 -> cuRobo 倒放回站姿)
cd /home/lyh/Project/RL_Correction || exit 1
SHARPA_WANDB=0 PYTHONPATH=. exec /home/lyh/luhr/MagicSim/.venv/bin/python -u \
  tasks/Pour/17/A_Design/L1_Data/Motion_Planning/view_motion.py \
  --file tasks/Pour/17/A_Design/L1_Data/Motion_Planning/Retreat_approach_rev.npz "$@"
