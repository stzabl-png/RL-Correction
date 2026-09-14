#!/usr/bin/env bash
# take3 消融续跑 (Denso GPU 故障后迁 msc). 用法: clean3_resume.sh <base|cf> GPU
# 配方逐项照搬原 run 的 world.json: 母带 v1h / SOFT_REL W0.5 / die 20° (默认) / seed 42 /
# 课程已退火到底 -> CLEAN_RELEASE_START=10; 步数按剩余量给 (restore_train 不恢复计数器)。
set -euo pipefail
ARM=${1:?base|cf}; GPU=${2:?GPU}
REPO=$HOME/RL_Correction; PY=$HOME/miniconda3/envs/isaac/bin/python
cd "$REPO"
case "$ARM" in
  base) OLD=Clean3_ablBase_s42; FLAGS=(CLEAN_S2_HAND_REF=1);  REMAIN=11142016 ;;
  cf)   OLD=Clean3_ablCF_s42;   FLAGS=(CLEAN_S2_CONF_FLAT=1); REMAIN=12501888 ;;
  *) echo "未知臂 $ARM"; exit 2 ;;
esac
NEW=${OLD}_r2
CKPT=logs/$OLD/stage1_nn/last.pth
[ -f "$CKPT" ] || { echo "缺 ckpt $CKPT"; exit 2; }
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH="$REPO"
export POUR_OBJ_MASS=0.3 POUR_OBJ_FRIC=1.0 POUR_PAD_FRIC=1.0
export CUDA_VISIBLE_DEVICES=$GPU
[ "$(nvidia-smi -L | wc -l)" -gt 1 ] && export RL_ISAAC_NO_GUARD=1
if ! mkdir -p /tmp/isaaclab/logs 2>/dev/null || [ ! -w /tmp/isaaclab/logs ]; then
  export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR/isaaclab/logs"; fi
mkdir -p launch_logs
echo "[clean3_resume] $ARM: $OLD -> $NEW | GPU$GPU | 从 $(cat logs/$OLD/progress_steps.txt) 步续, 再跑 $REMAIN 步"
setsid env CLEAN_REF_NPZ=tasks/Clean/3/A_Design/L2_Reference/clean3_reference_v1h.npz \
  CLEAN_S2_SOFT_REL=1 CLEAN_RELEASE_START=10 "${FLAGS[@]}" \
  nohup "$PY" -u tasks/Clean/3/C_Wiring/train_task.py --headless --num_envs 512 --seed 42 \
    --name "$NEW" --max_agent_steps "$REMAIN" --load_path "$CKPT" \
    > "launch_logs/$NEW.log" 2>&1 < /dev/null &
echo "[clean3_resume] pid=$! log=launch_logs/$NEW.log"
