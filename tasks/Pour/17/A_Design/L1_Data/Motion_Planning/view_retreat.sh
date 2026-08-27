#!/usr/bin/env bash
# Retreat 现场版 GUI: 静止GraspPose同源摆位 -> 定格等 Enter -> cuRobo退5cm(指前45帧绷直根部钉住)
cd /home/lyh/Project/RL_Correction || exit 1
SHARPA_WANDB=0 PYTHONPATH=. exec /home/lyh/luhr/MagicSim/.venv/bin/python -u \
  tasks/Pour/17/A_Design/L1_Data/Motion_Planning/view_retreat_live.py "$@"
