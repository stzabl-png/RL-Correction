#!/bin/bash
# v61: v60 配方逐字不变 + 母带重铸 (T2-39 右前臂离桌约束, 行 206~230 只抬腕 ≤44mm)。
# 唯一变量 = 母带几何 (母带 md5 变, 新世界指纹)。对照 = v60 (旧母带, 同配方)。
# 发射前提: 验收链全过 (外壳探针 ≥+1.2cm / 稳态漂移 ~2mm / probe_acceptance 凭据重签)
#          且 GPU 槽位已清空 (见台账 T2-39 编成)。
cd /home/feiyang/WorkSpace/RL-Correction
GPU=${GPU:-1}
setsid nohup env SHARPA_WANDB=0 UNSCREW_CLIP=17 \
  POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1 POUR_BONUS_DIST=1 POUR_PAD_FRIC=6.0 \
  POUR_VARIANT=HYB POUR_UNLOCK=1,2,3 UNSCREW_STAGE_C_FRAC=0.25 \
  UNSCREW_CAP_GRASP=user_thumb_index UNSCREW_CAP_PINCH=8 UNSCREW_MAX_TILT=80 \
  UNSCREW_CERT_WAGE=3.0 UNSCREW_CAP_WAGE=3.0 UNSCREW_CAP_APPROACH=3.0 UNSCREW_PINCH_WAGE=0.05 \
  CUDA_VISIBLE_DEVICES=$GPU RL_ISAAC_NO_GUARD=1 PYTHONPATH=/home/feiyang/WorkSpace/RL-Correction \
  /home/feiyang/isaacsim/python.sh -u tasks/Unscrew/part4/C_Wiring/train_task.py \
  --name Unscrew17HYB_v61 --num_envs 512 --seed 42 --headless --no_autorec \
  > /home/feiyang/WorkSpace/RL-Correction/logs/Unscrew17HYB_v61.out 2>&1 < /dev/null &
disown
sleep 20
pgrep -f "kit/python/bin/python3 -u tasks/Unscrew/part4/C_Wiring/train_task.py --name Unscrew17HYB_v61" | head -1 \
  > /home/feiyang/WorkSpace/RL-Correction/logs/Unscrew17HYB_v61.pid
echo "v61 launched on GPU$GPU pid=$(cat /home/feiyang/WorkSpace/RL-Correction/logs/Unscrew17HYB_v61.pid)"
