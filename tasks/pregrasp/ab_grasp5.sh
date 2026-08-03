#!/bin/bash
# Grasp5 GraspPose prior A/B: 串行跑两条 20M 步的 run (电源红线: 不并发).
#   A = 基线 (v2.8 配方原样, 从零)      -> logs/AB_Grasp5_base
#   B = A + Dexonomy GraspPose prior    -> logs/AB_Grasp5_prior
# 判读: 比两边 milestones.json 的首达 1%/10%/50% 步数 + 训完的确定性评测.
set -e
cd /home/lyh/Project/RL_Correction
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
STEPS=20000000

echo "=== A 组 (无 prior) 开始 $(date) ==="
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.train --headless \
  --clip Grasp5 --name AB_Grasp5_base --num_envs 1024 --max_agent_steps $STEPS
echo "=== A 组结束, 静置 15s 释放 GPU ==="
sleep 15
echo "=== B 组 (GraspPose prior) 开始 $(date) ==="
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.train --headless \
  --clip Grasp5 --name AB_Grasp5_prior --num_envs 1024 --max_agent_steps $STEPS \
  --grasp_prior
echo "=== A/B 全部结束 $(date) ==="
