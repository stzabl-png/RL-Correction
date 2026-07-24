#!/usr/bin/env bash
# 自动录像监视器: 训练每存一个周期快照 (ep_N, save_frequency=100), 若 N 是 VIDEO_EVERY
# 的倍数就用它录一段回放视频到 <run_dir>/videos/. 训练结束后自动退出.
#
# 用法 (第二个终端):
#   ./rl_rebuild/correction/autorecord.sh logs/correction_clip11/<时间戳目录> [间隔迭代数=1000]
set -uo pipefail
RUN=${1:?用法: autorecord.sh <run_dir> [video_every=1000] [clip=clip11] [python]}
EVERY=${2:-1000}
CLIP=${3:-clip11}                                # 必须与训练一致, 否则录到默认 clip11 的物体
cd "$(dirname "$0")/../.."                       # 仓库根 (模块导入需要)
# python: train.py 传自己的 sys.executable (跨机通用); 缺省回退本地 venv
PY=${4:-${ISAAC_PYTHON:-${RR_ROOT:-/home/lyh/Project/Reconstruct_and_Retarget}/third_party/MagicDexMate/.venv-isaac/bin/python}}
mkdir -p "$RUN/videos"
SEEN="$RUN/videos/.recorded"
touch "$SEEN"
echo "[autorecord] 监视 $RUN, 每 $EVERY 迭代录一段"

while true; do
  for f in "$RUN"/stage1_nn/ep_*_step_*.pth; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    ep=$(sed -E 's/ep_([0-9]+)_.*/\1/' <<< "$base")
    if [ $((ep % EVERY)) -eq 0 ] && ! grep -qxF "$base" "$SEEN"; then
      echo "[autorecord] 迭代 $ep -> 录像 ($base)"
      OMNI_KIT_ACCEPT_EULA=YES ISAAC_SIM_ACCEPT_EULA=1 $PY -u -m rl_rebuild.correction.record \
        --checkpoint "$f" --out_dir "$RUN/videos" --clip "$CLIP" --headless \
        && echo "$base" >> "$SEEN" \
        || echo "[autorecord] 录像失败: $base (下轮重试)"
    fi
  done
  if ! pgrep -f "rl_rebuild.correction.train" > /dev/null; then
    echo "[autorecord] 训练进程已结束, 退出"
    break
  fi
  sleep 60
done
