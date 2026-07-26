#!/usr/bin/env bash
# 自动录像监视器: 训练每存一个周期快照 (ep_N, save_frequency=100), 若 N 是 VIDEO_EVERY
# 的倍数就用它录一段回放视频到 <run_dir>/videos/. 训练结束后自动退出.
#
# 用法 (第二个终端):
#   ./rl_rebuild/correction/autorecord.sh logs/correction_clip11/<时间戳目录> [间隔迭代数=2000]
#
# ⚡ 功耗防护 (2026-07-24 整机断电事故后加, 见 rl_rebuild/utils/gpu_guard.py):
#   录像开的是**带渲染**的第二个 Isaac Sim. 它和训练同时满载时的功耗尖峰会触发本机
#   电源 OCP, 整机瞬断 (已发生两次: 2026-07-22 14:46 / 2026-07-24 16:15).
#   所以录像前先落 PAUSE 标记 -> 训练在 epoch 边界释放 GPU 槽位并挂起 -> 录像进程
#   独占 GPU 跑完 -> **完全退出** -> 静置 SETTLE 秒让显存和 carb 释放干净 -> 删标记
#   -> 训练重新拿槽位继续. 全程 GPU 上只有一个进程在算.
set -uo pipefail
RUN=${1:?用法: autorecord.sh <run_dir> [video_every=2000] [clip=clip11] [robot=flying] [python]}
EVERY=${2:-2000}
CLIP=${3:-clip11}                                # 必须与训练一致, 否则录到默认 clip11 的物体
# 必须与训练时一致 —— 动作维度不同(飞手28 / DexMate29), 载错直接维度不匹配
ROBOT=${4:-flying}
cd "$(dirname "$0")/../.."                       # 仓库根 (模块导入需要)
# python: train.py 传自己的 sys.executable (跨机通用); 缺省回退本地 venv
PY=${5:-${ISAAC_PYTHON:-${RR_ROOT:-/home/lyh/Project/Reconstruct_and_Retarget}/third_party/MagicDexMate/.venv-isaac/bin/python}}
mkdir -p "$RUN/videos"
SEEN="$RUN/videos/.recorded"
touch "$SEEN"

# ---- 让出协议 (路径/变量必须与 rl_rebuild/utils/gpu_guard.py 一致) ----
LOCK_DIR=${RL_ISAAC_LOCK_DIR:-$HOME/.cache/rl_correction}
PAUSE="$LOCK_DIR/pause.request"
SETTLE=${RL_ISAAC_SETTLE:-10}
mkdir -p "$LOCK_DIR"
# 本脚本被 Ctrl-C / kill 带走时必须保证标记不残留, 否则训练会一直挂到
# RL_ISAAC_MAX_PAUSE(默认 900s) 超时才自愈.
cleanup() { rm -f "$PAUSE"; }
trap cleanup EXIT INT TERM

echo "[autorecord] 监视 $RUN, 每 $EVERY 迭代录一段 (录像期间训练让出 GPU)"

record_one() {                                   # $1 = ckpt 路径
  local f=$1 rc
  echo "[autorecord] 请求训练让出 GPU 槽位 ..."
  # 标记里写自己的 PID: 下面的 trap 只是尽力而为 (bash 在前台子进程运行期间会推迟信号,
  # kill -9 更是直接绕过), 所以训练侧还会主动探这个 PID 活没活, 死了就立刻放行.
  echo $$ > "$PAUSE"                             # 落标记: 训练在下个 epoch 边界释放槽位
  # record.py 自己会 flock 排队, 拿到槽位才启动 Isaac —— 不会和训练撞车
  OMNI_KIT_ACCEPT_EULA=YES ISAAC_SIM_ACCEPT_EULA=1 $PY -u -m rl_rebuild.correction.record \
    --checkpoint "$f" --out_dir "$RUN/videos" --clip "$CLIP" --robot "$ROBOT" --headless
  rc=$?
  echo "[autorecord] 录像进程已退出 (rc=$rc), 静置 ${SETTLE}s 等显存/carb 释放干净"
  sleep "$SETTLE"                                # CLAUDE.md 第 5 条: 不等会触发 carb mutex 崩溃
  rm -f "$PAUSE"                                 # 解除让出 -> 训练重新拿槽位继续
  echo "[autorecord] 已解除让出, 训练恢复"
  return $rc
}

while true; do
  for f in "$RUN"/stage1_nn/ep_*_step_*.pth; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    ep=$(sed -E 's/ep_([0-9]+)_.*/\1/' <<< "$base")
    if [ $((ep % EVERY)) -eq 0 ] && ! grep -qxF "$base" "$SEEN"; then
      echo "[autorecord] 迭代 $ep -> 录像 ($base)"
      if record_one "$f"; then
        echo "$base" >> "$SEEN"
      else
        echo "[autorecord] 录像失败: $base (下轮重试)"
      fi
    fi
  done
  if ! pgrep -f "rl_rebuild.correction.train" > /dev/null; then
    echo "[autorecord] 训练进程已结束, 退出"
    break
  fi
  sleep 60
done
