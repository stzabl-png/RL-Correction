#!/usr/bin/env bash
# 评测点自动录像守护 —— 盯着一个 run 的日志, 每出现一次确定性评测就录一段。
#
# 为什么是守护脚本而不是训练里的钩子:
#   训练进程启动时代码就定死了, 给**在跑的** run 加录像只能重起。守护脚本在进程外面,
#   随时可开可关, 不碰训练。录像本身走 gpu_guard 的让出协议 (rec.sh), 训练不丢进度。
#
# 用法:
#   bash tasks/pregrasp/autorec.sh Minimal_L ~/minimal.log
#   nohup bash tasks/pregrasp/autorec.sh Minimal_L ~/minimal.log > ~/autorec.log 2>&1 &
#
# 停: kill 掉它, 或 touch ~/.cache/rl_correction/autorec.stop
set -uo pipefail
RUN="${1:?用法: autorec.sh <run名> <训练日志> [record.py 场景覆盖参数...]}"
LOG="${2:?用法: autorec.sh <run名> <训练日志> [record.py 场景覆盖参数...]}"
EXTRA=("${@:3}")   # 直传 rec.sh -> record.py; argparse 同名旗子后出现的赢, 可覆盖默认场景
CD="${RLCORR_ROOT:-/home/lyh/Project/RL_Correction}"
STOP="${RL_ISAAC_LOCK_DIR:-$HOME/.cache/rl_correction}/autorec.stop"
POLL="${AUTOREC_POLL:-60}"

cd "$CD"
rm -f "$STOP"
seen=0
echo "[autorec] 盯着 $RUN (日志 $LOG), 每次确定性评测后录一段。停: touch $STOP"

while :; do
  [ -f "$STOP" ] && { echo "[autorec] 收到停止标记, 退出"; exit 0; }
  # 训练没了就退出, 免得留个空转的守护
  if ! pgrep -f "name $RUN" > /dev/null 2>&1; then
    echo "[autorec] 训练 $RUN 已不在, 退出"; exit 0
  fi
  n=$(grep -ac "确定性成功率" "$LOG" 2>/dev/null || echo 0)
  if [ "${n:-0}" -gt "$seen" ]; then
    seen="$n"
    line=$(grep -a "确定性成功率" "$LOG" | tail -1)
    ep=$(printf "%s" "$line" | grep -o "ep[0-9]*" | head -1)
    ck=$(ls -t logs/"$RUN"/*/stage1_nn/last.pth 2>/dev/null | head -1)
    if [ -z "$ck" ]; then echo "[autorec] 找不到 ckpt, 跳过"; continue; fi
    echo "[autorec] 第 $seen 次评测 ($ep) -> 录像"
    echo "[autorec]   $line"
    # 录完把文件按 epoch 改名存档, 否则下一次会覆盖掉 (RecordVideo 用固定文件名)
    if bash tasks/pregrasp/rec.sh "$ck" ${EXTRA[@]+"${EXTRA[@]}"}; then
      vd=$(dirname "$ck")/../videos
      # 只改**这次新录的**(RecordVideo 固定前缀 last-*), 别把已归档的再套一层前缀
      for f in "$vd"/last-*.mp4; do
        [ -e "$f" ] || continue
        mv "$f" "$vd/${ep:-ep?}_$(basename "$f")" 2>/dev/null || true
      done
      echo "[autorec] ✅ 存档到 $vd/${ep}_*.mp4"
    else
      echo "[autorec] ❌ 这次录像失败, 继续盯下一次"
    fi
  fi
  sleep "$POLL"
done
