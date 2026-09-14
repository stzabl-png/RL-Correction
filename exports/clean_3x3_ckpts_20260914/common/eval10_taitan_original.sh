#!/usr/bin/env bash
# Clean 九条 × 10 回合 (Pour eval10 同款协议)。用法: bash eval10.sh <GPU> <seed> <run1,run2,...>
# 配方**逐条从各自 world.json 读**, 不硬编码 —— 昨天硬编码漏了 Base 臂的 HAND_REF, 被世界核对拦下。
set -u
cd ~/RL_Correction
GPU=$1; SEED=$2; IFS=',' read -ra RUNS <<< "$3"
if ! mkdir -p /tmp/isaaclab/logs 2>/dev/null || [ ! -w /tmp/isaaclab/logs ]; then
  export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR/isaaclab/logs"; fi
PY=""
for c in $HOME/miniconda3/envs/env_isaaclab/bin/python $HOME/miniconda3/envs/isaac/bin/python \
         $HOME/miniforge3/envs/isaac/bin/python; do
  [ -x "$c" ] && "$c" -c "import isaaclab" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "❌ 无 isaaclab 解释器"; exit 1; }
L2=$PWD/tasks/Clean/3/A_Design/L2_Reference
PR=$PWD/tasks/pregrasp/priors
mkdir -p logs/eval10
for n in "${RUNS[@]}"; do
  # 从 world.json 读配方
  read -r CLIP REF PLATE HAND CONF <<< "$("$PY" - "$n" <<'PYEOF'
import json, os, sys
d = json.load(open(f"logs/{sys.argv[1]}/world.json"))
s2 = d.get("stage2", {})
print(d.get("clip"), os.path.basename(d["reference"]["path"]),
      os.path.basename(d["priors"]["plate"]["path"]),
      1 if s2.get("S2_HAND_REF") else 0, 1 if s2.get("S2_CONF_FLAT") else 0)
PYEOF
)"
  echo "[eval10] $n  clip=$CLIP 母带=$REF 盘先验=$PLATE HAND_REF=$HAND CONF_FLAT=$CONF"
  env TMPDIR="${TMPDIR:-/tmp}" OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 \
    PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="$GPU" CLEAN_IGNORE_WORLD=1 \
    CLEAN_CLIP="$CLIP" CLEAN_REF_NPZ="$L2/$REF" CLEAN_PRIOR_PLATE="$PR/$PLATE" \
    POUR_OBJ_MASS=0.3 POUR_OBJ_FRIC=1.0 POUR_PAD_FRIC=1.0 \
    CLEAN_S2_SOFT_REL=1 CLEAN_S2_SOFT_W=0.5 CLEAN_S2_DIE_ROT_DEG=180 \
    CLEAN_S2_HAND_REF="$HAND" CLEAN_S2_CONF_FLAT="$CONF" \
    "$PY" tasks/Clean/3/C_Wiring/eval_clean.py --checkpoint "logs/$n/stage1_nn/last.pth" \
      --num_envs 10 --release_row 10 --jitter 0 --seed "$SEED" --tag "${n}_e10s${SEED}" \
    > "logs/eval10/eval10_${n}_s${SEED}.log" 2>&1 </dev/null
  echo "[eval10] $n 完成: $(grep -oE '★ .*' logs/eval10/eval10_${n}_s${SEED}.log | tail -1 | cut -c1-110)"
done
echo "ALLDONE"
