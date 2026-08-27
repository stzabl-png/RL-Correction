#!/usr/bin/env bash
# L2 参考拼接查看器: 重建人手21关节(红=左 绿=右)+双物体原始位姿, 无conf无RTS
cd /home/lyh/Project/RL_Correction || exit 1
SHARPA_WANDB=0 PYTHONPATH=. exec /home/lyh/luhr/MagicSim/.venv/bin/python -u \
  tasks/Pour/17/A_Design/L2_Reference/view_reference.py "$@"
