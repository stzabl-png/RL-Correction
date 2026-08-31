#!/usr/bin/env bash
# 新服务器部署验收: verify_deploy.sh [CLIP] [--isaac]
set -euo pipefail

CLIP=${1:-32}
MODE=${2:-}
[[ "$CLIP" =~ ^[0-9]+$ ]] || { echo "CLIP 必须是数字" >&2; exit 2; }
[[ -z "$MODE" || "$MODE" == "--isaac" ]] || {
  echo "usage: $0 [CLIP] [--isaac]" >&2
  exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
cd "$REPO_ROOT"

if [[ -z "${PY:-}" ]]; then
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

export UNSCREW_CLIP=$CLIP
export VEGA_URDF=${VEGA_URDF:-"$REPO_ROOT/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa.urdf"}
TMP_BASE=${TMPDIR:-"$HOME/tmp"}
export TMPDIR=${UNSCREW_TMPDIR:-"$TMP_BASE/unscrew-${USER:-user}"}
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p -- "$TMPDIR"

git lfs version >/dev/null
required=(
  "assets/vega_1p_sharpa_fixedtorso.usd"
  "datasets/unscrew_bottle/$CLIP/replay_world.npz"
  "datasets/unscrew_bottle/$CLIP/contact/affordance_bottle_left.npz"
  "tasks/Unscrew/part4/A_Design/L1_Data/Motion_Planning/$CLIP/Approach.npz"
  "tasks/Unscrew/part4/A_Design/L1_Data/Motion_Planning/$CLIP/Retreat.npz"
  "tasks/Unscrew/part4/A_Design/L2_Reference/$CLIP/env_rest.json"
  "tasks/Unscrew/part4/A_Design/L2_Reference/$CLIP/reference_v2.npz"
  "tasks/Unscrew/part4/A_Design/L2_Reference/$CLIP/acceptance_v2.json"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "缺少部署产物: $path" >&2; exit 4; }
  if grep -a -m 1 -q '^version https://git-lfs.github.com/spec/v1' "$path"; then
    echo "LFS 文件仍是指针: $path；请先 git lfs pull" >&2
    exit 4
  fi
done

"$PY" -c "import importlib.metadata as m, torch; print('python/torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('isaacsim', m.version('isaacsim'), 'isaaclab', m.version('isaaclab'))"
"$PY" -c "import sys; sys.path.insert(0, '$REPO_ROOT/tasks/Unscrew/part4/C_Wiring'); import task_config as tc; tc.require_training_reference(tc.REF_V2)"
echo "[deploy] ✅ LFS、依赖元数据、clip$CLIP 正式母带与训练稳定性凭据通过"

if [[ "$MODE" == "--isaac" ]]; then
  GPU=${GPU:-0}
  SMOKE_STEPS=${SMOKE_STEPS:-20}
  CUDA_VISIBLE_DEVICES=$GPU RL_ISAAC_NO_GUARD=${RL_ISAAC_NO_GUARD:-0} "$PY" -u tasks/Unscrew/part4/C_Wiring/smoke_zero.py --steps "$SMOKE_STEPS" --headless
  echo "[deploy] ✅ Isaac clip$CLIP smoke 完成 ($SMOKE_STEPS steps)"
fi
