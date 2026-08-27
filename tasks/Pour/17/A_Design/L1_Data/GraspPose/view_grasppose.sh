#!/usr/bin/env bash
# GraspPose 检查 GUI: 当前场景(黑桌6x4ft/87cm/新站姿), 物体按 prior 摆放,
# 双手 IK 直接钉在各自 GraspPose 上。绿球=右手靶(瓶), 红球=左手靶(杯)。
# 用法: bash view_grasppose.sh            # GUI
#       bash view_grasppose.sh --headless # 只打印 IK 误差自检
cd /home/lyh/Project/RL_Correction || exit 1
SHARPA_WANDB=0 PYTHONPATH=. exec /home/lyh/luhr/MagicSim/.venv/bin/python -u \
  -m tasks.pregrasp.view_grasp_pose \
  --prior_a tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz --yaw_a 19.5 \
  --prior_b tasks/pregrasp/priors/Pour17_cup_thumbfix.npz --yaw_b 90 "$@"
