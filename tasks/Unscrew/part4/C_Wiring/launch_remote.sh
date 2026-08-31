#!/usr/bin/env bash
# 远端单线发射:
#   launch_remote.sh NAME GPU SEED HYB|OBJ [CLIP]
# 可选环境变量: PY / NUM_ENVS / MAX_AGENT_STEPS / AUTO_RECORD=1 /
# UNSCREW_TMPDIR。
set -euo pipefail

usage() {
  echo "usage: $0 NAME GPU SEED HYB|OBJ [CLIP]" >&2
}

if (( $# < 4 || $# > 5 )); then
  usage
  exit 2
fi

NAME=$1
GPU=$2
SEED=$3
VAR=${4^^}
CLIP=${5:-32}
NUM_ENVS=${NUM_ENVS:-512}

case "$VAR" in
  HYB|OBJ) ;;
  *) echo "variant 必须是 HYB 或 OBJ, got: $VAR" >&2; exit 2 ;;
esac
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU 必须是非负整数" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED 必须是非负整数" >&2; exit 2; }
[[ "$CLIP" =~ ^[0-9]+$ ]] || { echo "CLIP 必须是数字" >&2; exit 2; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
cd "$REPO_ROOT"

# env_a6000.sh 是**另一台机器**的环境 (PY/VEGA_URDF/RR_ROOT/AFFORDANCE_ROOT/
# PYTHONPATH 全指向 $HOME/data 与 $HOME/RL_Correction)。只有那台机器上才 source,
# 否则一堆变量被设成不存在的路径, 而下游的 ${VAR:-默认} 因为"变量已非空"永远
# 用不上默认值 —— 2026-08-31 实测连着炸了两次 (找不到 Isaac Python / 找不到
# vega URDF)。
if [[ -z "${PY:-}" && -f "$REPO_ROOT/env_a6000.sh" && -d "$HOME/data/vega_urdf" ]]; then
  source "$REPO_ROOT/env_a6000.sh" >/dev/null 2>&1 || true
fi
# 判据是"能不能执行", 不是"变量空不空": env_a6000.sh 会把 PY 设成别的机器的
# 路径 (~/miniconda3/...), 变量一非空就跳过探测 => 本机永远报"找不到 Isaac Python"。
if [[ -z "${PY:-}" || ! -x "${PY:-}" ]]; then
  for candidate in "$HOME/miniforge3/envs/isaac/bin/python" "$HOME/miniconda3/envs/isaac/bin/python" "$HOME/miniconda3/envs/env_isaaclab/bin/python"; do
    if [[ -x "$candidate" ]]; then
      PY=$candidate
      break
    fi
  done
fi
[[ -n "${PY:-}" && -x "$PY" ]] || {
  echo "找不到 Isaac Python；请 export PY=/path/to/env/bin/python" >&2
  exit 3
}

# 同 PY: 判据是"文件在不在", 不是"变量设没设" —— env_a6000.sh 会把 VEGA_URDF
# 指到别的机器的路径 (~/data/vega_urdf/...), 变量一非空就用不上仓库自带资产。
_VEGA_REPO="$REPO_ROOT/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa.urdf"
if [[ -z "${VEGA_URDF:-}" || ! -f "${VEGA_URDF:-}" ]]; then
  VEGA_URDF=$_VEGA_REPO
fi
export VEGA_URDF
TMP_BASE=${TMPDIR:-"$HOME/tmp"}
export TMPDIR=${UNSCREW_TMPDIR:-"$TMP_BASE/unscrew-${USER:-user}"}
mkdir -p -- "$TMPDIR" logs
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# 远端独占卡时关掉排队闸没问题; 但共享机上必须留得下开关 ——
# 默认改成可覆盖 (RL_ISAAC_NO_GUARD=0 即恢复 flock 独占槽位)。
export RL_ISAAC_NO_GUARD=${RL_ISAAC_NO_GUARD:-1} CUDA_VISIBLE_DEVICES=$GPU
export POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1 POUR_BONUS_DIST=1
export POUR_VARIANT=$VAR UNSCREW_CLIP=$CLIP

for required in "assets/vega_1p_sharpa_fixedtorso.usd" "tasks/Unscrew/part4/A_Design/L2_Reference/$CLIP/reference_v2.npz" "tasks/Unscrew/part4/A_Design/L2_Reference/$CLIP/acceptance_v2.json"; do
  [[ -s "$required" ]] || { echo "缺少部署产物: $required" >&2; exit 4; }
  if grep -a -m 1 -q '^version https://git-lfs.github.com/spec/v1' "$required"; then
    echo "LFS 文件仍是指针: $required；请先 git lfs pull" >&2
    exit 4
  fi
done
"$PY" -c "import sys; sys.path.insert(0, '$REPO_ROOT/tasks/Unscrew/part4/C_Wiring'); import task_config as tc; tc.require_training_reference(tc.REF_V2)"

TRAIN_ARGS=(--name "$NAME" --num_envs "$NUM_ENVS" --seed "$SEED" --headless)
if [[ -n "${MAX_AGENT_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--max_agent_steps "$MAX_AGENT_STEPS")
fi
if [[ "${AUTO_RECORD:-0}" != 1 ]]; then
  TRAIN_ARGS+=(--no_autorec)
fi

LOG="logs/${NAME}.out"
nohup "$PY" -u tasks/Unscrew/part4/C_Wiring/train_task.py "${TRAIN_ARGS[@]}" >"$LOG" 2>&1 &
PID=$!
echo "$PID" >"logs/${NAME}.pid"
echo "launched $NAME clip$CLIP on GPU$GPU seed$SEED $VAR pid $PID"
echo "log: $REPO_ROOT/$LOG"
