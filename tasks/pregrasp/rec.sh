#!/usr/bin/env bash
# 录一段 ckpt 的视频, 走 gpu_guard 的"录像让出"协议:
#   落 PAUSE 标记 -> 训练在 epoch 边界(刚存完 ckpt)让出显卡并挂起
#   -> 录像独占跑完并**完全退出** -> 静置 SETTLE 秒 -> 删标记 -> 训练拿回继续。
# 训练不丢进度, 只是墙钟变长。
#
# 用法:
#   bash tasks/pregrasp/rec.sh <ckpt路径> [额外参数...]
#   bash tasks/pregrasp/rec.sh logs/Minimal_L/*/stage1_nn/last.pth
#
# 默认按 Minimal_L 的场景录 (--approach --approach_only --minimal + Grasp3/8_5/yaw215)。
# 录别的 run 就把场景参数作为额外参数覆盖。
set -euo pipefail
CKPT="${1:?用法: rec.sh <ckpt路径> [额外参数]}"; shift || true
[ -f "$CKPT" ] || { echo "找不到 ckpt: $CKPT"; exit 1; }

PY="${RLCORR_PY:-/home/lyh/luhr/MagicSim/.venv/bin/python}"
CD="${RLCORR_ROOT:-/home/lyh/Project/RL_Correction}"
LOCK_DIR="${RL_ISAAC_LOCK_DIR:-$HOME/.cache/rl_correction}"
PAUSE="$LOCK_DIR/pause.request"
SETTLE="${RL_ISAAC_SETTLE:-10}"

mkdir -p "$LOCK_DIR"
cleanup() { _rc=$?; rm -f "$PAUSE"; echo "[rec] 已删 PAUSE 标记, 训练可以拿回显卡"; exit $_rc; }
trap cleanup EXIT INT TERM

echo "[rec] 落 PAUSE 标记 -> 等训练在 epoch 边界让出显卡 (可能要等一个 epoch)"
touch "$PAUSE"

cd "$CD"
# ⚠ 底座命令**不许**放 store_true 开关 (--approach_only 之类): argparse 的开关出现
#   即 True, 调用方追加参数盖不掉 —— 2026-08-17 实测 L5(26维) 录像被它钉成 14 维,
#   load_state_dict 崩。模式开关一律由调用方传 (autorec 的 EXTRA 直通)。
SHARPA_WANDB=0 PYTHONPATH=. "$PY" -u -m tasks.pregrasp.record \
  --checkpoint "$CKPT" --headless \
  --clip Grasp3 \
  --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215 \
  --num_envs 4 "$@" && RC=0 || RC=$?   # set -e 会在失败时直接退出, 这里接住
# Isaac 的异常钩子会吞退出码 (崩了也返回 0) —— 只认产出物: 没有新 mp4 就是失败
VD="$(dirname "$CKPT")/../videos"
TAG="$(basename "$CKPT" .pth)"
if ! ls "$VD"/"$TAG"-*.mp4 >/dev/null 2>&1; then
  echo "[rec] ❌ 没有产出 mp4 (退出码 $RC 不可信, Isaac 钩子会吞) -> 判失败"
  RC=3
fi
if [ "$RC" != "0" ]; then echo "[rec] ❌ 录像失败 (退出码 $RC), 见上面的报错"; fi
echo "[rec] 录像进程已退出, 静置 ${SETTLE}s 让显存与 carb 释放干净"
sleep "$SETTLE"
[ "$RC" = "0" ] && echo "[rec] ✅ 视频在 <run_dir>/videos/"
exit "$RC"   # 让失败真的失败, 不被 trap 吃掉
