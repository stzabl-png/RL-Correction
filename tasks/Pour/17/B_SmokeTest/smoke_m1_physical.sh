#!/usr/bin/env bash
cd /home/lyh/Project/RL_Correction || exit 1
SHARPA_WANDB=0 PYTHONPATH=. exec /home/lyh/luhr/MagicSim/.venv/bin/python -u \
  tasks/Pour/17/B_SmokeTest/smoke_m1_physical.py "$@"
