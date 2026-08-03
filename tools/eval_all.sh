#!/usr/bin/env bash
# 对一组 run 的 last.pth 串行跑确定性评测 —— 定论只认这个 (CLAUDE.md 第 2 条:
# TB 的 success_rate 分母是"这一步恰好 reset 的 env 数", 只能看趋势).
#
#   tools/eval_all.sh Grasp2 dexmate run_dir1 run_dir2 ...
#
# 串行, 每条之间静置 SETTLE 秒; 与训练互斥由 gpu_guard 的 flock 保证.
set -uo pipefail
CLIP=${1:?用法: eval_all.sh <clip> <robot> <run_dir>...}
ROBOT=${2:?}
shift 2
cd "$(dirname "$0")/.."
PY=${ISAAC_PYTHON:-/home/lyh/luhr/MagicSim/.venv/bin/python}
SETTLE=${RL_ISAAC_SETTLE:-10}
OUT=${EVAL_OUT:-/tmp/eval_all_$$.txt}

flock -n -E 99 "$HOME/.cache/rl_correction/isaac.lock" true
if [ $? -eq 99 ]; then
  echo "[eval_all] ⚠ GPU 槽位被占 (训练在跑). 评测会一直阻塞等锁, 先等训练结束."
  exit 1
fi

: > "$OUT"
for RUN in "$@"; do
  CK="$RUN/stage1_nn/last.pth"
  [ -e "$CK" ] || { echo "[eval_all] 跳过 (无 last.pth): $RUN"; continue; }
  NAME=$(basename "$(dirname "$RUN")")
  echo "[eval_all] ===== $NAME ====="
  OMNI_KIT_ACCEPT_EULA=YES ISAAC_SIM_ACCEPT_EULA=1 SHARPA_WANDB=0 PYTHONPATH=. \
    "$PY" -u -m rl_rebuild.correction.eval_policy \
      --checkpoint "$CK" --clip "$CLIP" --robot "$ROBOT" \
      --num_envs 1024 --episodes 2 --headless 2>&1 | tee -a "$OUT" \
    | grep -a --line-buffered -E "成功率|success|抬升|接触|===" || true
  sleep "$SETTLE"
done
echo
echo "[eval_all] 汇总 (完整输出在 $OUT):"
grep -aE "成功率|success_rate" "$OUT" || true
