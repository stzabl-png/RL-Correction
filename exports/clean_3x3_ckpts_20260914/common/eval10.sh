#!/usr/bin/env bash
# Clean 九条 × 10 回合复现 (包内自洽版)。配方**逐条从各自 world.json 读**, 不硬编码。
# 用法: bash common/eval10.sh <REPO根> <GPU> <seed> <run1,run2,...>
#   REPO根 = 你的 RL_Correction 仓 (代码 + datasets/clean_tableware 已按 README §1 放好)
#   本脚本所在的包目录 = 母带/先验/ckpt 的来源
set -u
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO=$1; GPU=$2; SEED=$3; IFS=',' read -ra RUNS <<< "$4"
cd "$REPO"
if ! mkdir -p /tmp/isaaclab/logs 2>/dev/null || [ ! -w /tmp/isaaclab/logs ]; then
  export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR/isaaclab/logs"; fi
PY=""
for c in $HOME/miniconda3/envs/env_isaaclab/bin/python $HOME/miniconda3/envs/isaac/bin/python \
         $HOME/anaconda3/envs/isaac/bin/python $HOME/miniforge3/envs/isaac/bin/python; do
  [ -x "$c" ] && "$c" -c "import isaaclab" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "❌ 找不到带 isaaclab 的解释器, 手动改 PY="; exit 1; }
mkdir -p "$PKG/eval10_out"
for n in "${RUNS[@]}"; do
  W="$PKG/runs/$n/world.json"
  [ -f "$W" ] || { echo "❌ 无 $W"; continue; }
  read -r CLIP REF PLATE SPONGE HAND CONF <<< "$("$PY" - "$W" <<'PYEOF'
import json, os, sys
d = json.load(open(sys.argv[1])); s2 = d.get("stage2", {})
pr = d["priors"]
plate = next(os.path.basename(v["path"]) for k, v in pr.items() if "plate" in k.lower() or "plate" in v["path"])
sponge = next(os.path.basename(v["path"]) for k, v in pr.items() if "sponge" in k.lower() or "sponge" in v["path"])
print(d.get("clip"), os.path.basename(d["reference"]["path"]), plate, sponge,
      1 if s2.get("S2_HAND_REF") else 0, 1 if s2.get("S2_CONF_FLAT") else 0)
PYEOF
)"
  echo "[eval10] $n  clip=$CLIP 母带=$REF 盘=$PLATE 布=$SPONGE HAND_REF=$HAND CONF_FLAT=$CONF"
  env TMPDIR="${TMPDIR:-/tmp}" OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 \
    PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="$GPU" CLEAN_IGNORE_WORLD=1 \
    CLEAN_CLIP="$CLIP" CLEAN_REF_NPZ="$PKG/assets/tapes/$REF" \
    CLEAN_PRIOR_PLATE="$PKG/assets/priors/$PLATE" CLEAN_PRIOR_SPONGE="$PKG/assets/priors/$SPONGE" \
    POUR_OBJ_MASS=0.3 POUR_OBJ_FRIC=1.0 POUR_PAD_FRIC=1.0 \
    CLEAN_S2_SOFT_REL=1 CLEAN_S2_SOFT_W=0.5 CLEAN_S2_DIE_ROT_DEG=180 \
    CLEAN_S2_HAND_REF="$HAND" CLEAN_S2_CONF_FLAT="$CONF" \
    "$PY" tasks/Clean/3/C_Wiring/eval_clean.py --checkpoint "$PKG/runs/$n/stage1_nn/last.pth" \
      --num_envs 10 --release_row 10 --jitter 0 --seed "$SEED" --tag "${n}_e10s${SEED}" --headless \
    > "$PKG/eval10_out/eval10_${n}_s${SEED}.log" 2>&1 </dev/null
  echo "[eval10] $n → $(grep -oE '★ .*' "$PKG/eval10_out/eval10_${n}_s${SEED}.log" | tail -1 | cut -c1-120)"
done
echo ALLDONE
