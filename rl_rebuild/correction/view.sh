#!/usr/bin/env bash
# 打开 GUI 看训练环境 —— 跑的就是训练用的那个 env, 所见即 RL 所见.
#
#   ./rl_rebuild/correction/view.sh                      # DexMate 当前设定 (默认)
#   ./rl_rebuild/correction/view.sh --robot flying       # 和飞手对照
#   ./rl_rebuild/correction/view.sh --clip Grasp7        # 换物体
#   ./rl_rebuild/correction/view.sh --speed 0.3          # 慢放看细节
#   GRASP_APPROACH=40 ./rl_rebuild/correction/view.sh    # 打开接近段 (目前是坏的)
#
# 关窗口或 Ctrl-C 退出.
set -uo pipefail
cd "$(dirname "$0")/../.."
export SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES ISAAC_SIM_ACCEPT_EULA=1
PY=${ISAAC_PYTHON:-/home/lyh/luhr/MagicSim/.venv/bin/python}
exec "$PY" -u -m rl_rebuild.correction.view_env "$@"
