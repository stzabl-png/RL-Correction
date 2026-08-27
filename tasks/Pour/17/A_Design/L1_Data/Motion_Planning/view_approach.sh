#!/usr/bin/env bash
# GUI 回放 Approach_pour17.npz (运动学, 循环播放)
cd /home/lyh/Project/RL_Correction || exit 1
SHARPA_WANDB=0 PYTHONPATH=. exec /home/lyh/luhr/MagicSim/.venv/bin/python -u \
  tasks/Pour/17/A_Design/L1_Data/Motion_Planning/view_motion.py \
  --file tasks/Pour/17/A_Design/L1_Data/Motion_Planning/Approach_pour17.npz "$@"
