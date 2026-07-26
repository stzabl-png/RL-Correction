#!/usr/bin/env bash
# 环境查看器守护壳: env_viewer.py --watch 检测到 env 代码改动会以码 42 退出,
# 本脚本把它自动拉起来 —— 效果是"改 cfg/env 代码 -> GUI 自动重建成新场景".
#
#   ./rl_rebuild/correction/viewer_loop.sh [--clip Grasp2] [其它 env_viewer 参数]
#
# 正常退出 (Ctrl+C / 关窗口) 不会重启. 只有码 42 才重启.
set -uo pipefail
cd "$(dirname "$0")/../.."                      # 仓库根
PY=${ISAAC_PYTHON:-/home/lyh/luhr/MagicSim/.venv/bin/python}

# 训练用的几何参数 — 必须和 checkpoint 一致, 否则看到的场景不是训练时的场景
export GRASP_PALM_OFF=${GRASP_PALM_OFF:-0.12}
export GRASP_HOVER=${GRASP_HOVER:-0.02}
export GRASP_CLOSE_SCALE=${GRASP_CLOSE_SCALE:-1.7}
export GRASP_APPROACH=${GRASP_APPROACH:-15}
export GRASP_CONTACT_CLOSE=${GRASP_CONTACT_CLOSE:-1}
export GRASP_START_JITTER=${GRASP_START_JITTER:-0}
export SHARPA_WANDB=0
export SHOW_DEXMATE=${SHOW_DEXMATE:-1}
export SHOW_HUMAN_TRAJ=${SHOW_HUMAN_TRAJ:-1}   # 画人手轨迹细线 + 按双手均衡摆放物体          # 查看器里显示 DexMate 视觉参考
export MAGICSIM_ASSETS=${MAGICSIM_ASSETS:-/home/lyh/luhr/MagicSim/Assets}

echo "[viewer_loop] 几何: PALM_OFF=$GRASP_PALM_OFF CLOSE_SCALE=$GRASP_CLOSE_SCALE APPROACH=$GRASP_APPROACH CONTACT_CLOSE=$GRASP_CONTACT_CLOSE JITTER=$GRASP_START_JITTER"

while true; do
  "$PY" -m rl_rebuild.correction.env_viewer --watch ${VIEWER_FOLLOW:+--follow} ${VIEWER_PLAY:+--play} ${VIEWER_REALTIME:+--realtime} "$@"
  rc=$?
  if [ $rc -eq 42 ]; then
    echo "[viewer_loop] env 代码已改, 10s 后重建场景 (carb 需要时间释放)..."
    sleep 10                                    # CLAUDE.md 第 5 条: 不等会触发 carb mutex 崩溃
    continue
  fi
  echo "[viewer_loop] 查看器退出 (码 $rc), 不重启"
  break
done
