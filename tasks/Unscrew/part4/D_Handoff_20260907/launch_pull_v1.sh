#!/bin/bash
# PULL_v1: 拔盖新环境 (T2-40, 用户预案 "唯一的不同 = 盖不用拧, 拔开")。
# 配方 = v61 逐字不变 (新母带 04c2beb1 + 捏且转工资 0.05) + UNSCREW_CAP_MODE=pull。
# 用户裁定与 v60 同卡 (GPU0, 24GB: v60 3.7GB + 本 run ~3.7GB)。拧盖环境不删。
cd /home/feiyang/WorkSpace/RL-Correction
GPU=${GPU:-0}
setsid nohup env SHARPA_WANDB=0 UNSCREW_CLIP=17 \
  POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1 POUR_BONUS_DIST=1 POUR_PAD_FRIC=6.0 \
  POUR_VARIANT=HYB POUR_UNLOCK=1,2,3 UNSCREW_STAGE_C_FRAC=0.25 \
  UNSCREW_CAP_GRASP=user_thumb_index UNSCREW_CAP_PINCH=8 UNSCREW_MAX_TILT=80 \
  UNSCREW_CERT_WAGE=3.0 UNSCREW_CAP_WAGE=3.0 UNSCREW_CAP_APPROACH=3.0 UNSCREW_PINCH_WAGE=0.05 \
  UNSCREW_CAP_MODE=pull UNSCREW_PULL_FULL_MM=15 UNSCREW_PULL_BREAKAWAY_N=2.0 \
  UNSCREW_PULL_KINETIC_N=0.8 UNSCREW_PULL_VISCOUS=40 UNSCREW_PULL_VMAX=0.15 \
  CUDA_VISIBLE_DEVICES=$GPU RL_ISAAC_NO_GUARD=1 PYTHONPATH=/home/feiyang/WorkSpace/RL-Correction \
  /home/feiyang/isaacsim/python.sh -u tasks/Unscrew/part4/C_Wiring/train_task.py \
  --name Unscrew17PULL_v1 --num_envs 512 --seed 42 --headless --no_autorec \
  > /home/feiyang/WorkSpace/RL-Correction/logs/Unscrew17PULL_v1.out 2>&1 < /dev/null &
disown
echo "PULL_v1 launching on GPU$GPU (pid 见 logs/Unscrew17PULL_v1.pid, 起动后由发射者回填)"
