#!/usr/bin/env bash
# Denso (4×2080Ti) 发射: denso_clean3.sh NAME GPU SEED [extra train_clean args...]
#   例: bash tasks/Clean/3/C_Wiring/denso_clean3.sh Clean3_hold_s42 0 42
# 物理规矩 (台账 §5.0⑦, 显式写, 发车后 grep 日志 "难度覆写"): 盘 0.3kg / 海绵 0.05kg(USD) / μ1 / 指垫 1
# 认证阈值 (2026-09-08 用户裁定): 1.0cm / 5°
set -euo pipefail
NAME=$1; GPU=$2; SEED=$3; shift 3
REPO=$HOME/RL_Correction
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
cd "$REPO"
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH="$REPO" RL_ISAAC_NO_GUARD=1 CUDA_VISIBLE_DEVICES=$GPU
export TMPDIR=${TMPDIR:-$HOME/tmp/clean3}; mkdir -p "$TMPDIR" logs
export POUR_OBJ_MASS=0.3 POUR_OBJ_FRIC=1.0 POUR_PAD_FRIC=1.0
export CLEAN_CERT_POS_CM=${CLEAN_CERT_POS_CM:-1.0} CLEAN_CERT_ROT_DEG=${CLEAN_CERT_ROT_DEG:-5}
for f in tasks/Clean/3/A_Design/L2_Reference/clean3_reference_v1.npz tasks/pregrasp/priors/Clean3_plate_left.npz \
         tasks/pregrasp/priors/Clean3_sponge_right.npz datasets/clean_tableware/3/retarget/object_0_textured.usd \
         datasets/clean_tableware/3/objects/object_1/object_mesh_scaled_final.obj datasets/clean_tableware/3/scene_layout.json; do
  [[ -s "$f" ]] || { echo "缺部署产物: $f" >&2; exit 4; }
done
echo "[denso_clean3] $NAME GPU=$GPU seed=$SEED cert=${CLEAN_CERT_POS_CM}cm/${CLEAN_CERT_ROT_DEG}deg PY=$PY"
nohup "$PY" -u tasks/Clean/3/C_Wiring/train_clean.py --headless --num_envs "${NUM_ENVS:-512}" --seed "$SEED" --name "$NAME" "$@" \
  > "$HOME/$NAME.log" 2>&1 &
echo "[denso_clean3] pid=$! log=$HOME/$NAME.log"
