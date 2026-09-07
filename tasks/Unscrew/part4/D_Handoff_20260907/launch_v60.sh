#!/bin/bash
# v60: v59 配方 + PINCH_WAGE 语义改"捏且转"每步工资 0.05 (T2-36, 唯一变量)。
# 前提: GPU0 已清空 (v58 已停, 对照使命完成: 71.9M align 0.26 对 v59 0.96)。
# 注意: UNSCREW_PINCH_WAGE 语义已变 —— v59 里是回合封顶(2.0), v60 起是每步
# 工资(0.05, 只在 >=2 右垫在盖 且 螺角当步爬过回合新高>0.05° 时发, 不封顶)。
cd /home/feiyang/WorkSpace/RL-Correction
setsid nohup env SHARPA_WANDB=0 UNSCREW_CLIP=17 \
  POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1 POUR_BONUS_DIST=1 POUR_PAD_FRIC=6.0 \
  POUR_VARIANT=HYB POUR_UNLOCK=1,2,3 UNSCREW_STAGE_C_FRAC=0.25 \
  UNSCREW_CAP_GRASP=user_thumb_index UNSCREW_CAP_PINCH=8 UNSCREW_MAX_TILT=80 \
  UNSCREW_CERT_WAGE=3.0 UNSCREW_CAP_WAGE=3.0 UNSCREW_CAP_APPROACH=3.0 UNSCREW_PINCH_WAGE=0.05 \
  CUDA_VISIBLE_DEVICES=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=/home/feiyang/WorkSpace/RL-Correction \
  /home/feiyang/isaacsim/python.sh -u tasks/Unscrew/part4/C_Wiring/train_task.py \
  --name Unscrew17HYB_v60 --num_envs 512 --seed 42 --headless --no_autorec \
  > /home/feiyang/WorkSpace/RL-Correction/logs/Unscrew17HYB_v60.out 2>&1 < /dev/null &
echo $! > /home/feiyang/WorkSpace/RL-Correction/logs/Unscrew17HYB_v60.pid
disown
echo "v60 launched pid=$(cat /home/feiyang/WorkSpace/RL-Correction/logs/Unscrew17HYB_v60.pid)"
