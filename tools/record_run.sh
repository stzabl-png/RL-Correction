#!/usr/bin/env bash
# 训练**结束后**串行补录一个 run 的所有快照 —— 判读闭环的默认录像方式.
#
#   tools/record_run.sh <run_dir> <clip> [robot=dexmate] [--closeup]
#   tools/record_run.sh logs/correction_Grasp2_dexmate_accum/2026-07-27_23-09-46 Grasp2
#
# ## 为什么不用训练中的 autorecord
#
# 并发本身两者都安全: 所有 Isaac 入口在 AppLauncher 之前先抢 gpu_guard 的 flock,
# 任何时刻 GPU 上只可能有一个 Isaac 进程 (2026-07-24 整机断电事故后加的防护).
# 区别在**故障面**:
#   autorecord 要走"让出协议" (落 PAUSE 标记 -> 训练在 epoch 边界释放槽位挂起 ->
#   录完删标记), 多一个失败模式: 标记残留会让训练空转到 RL_ISAAC_MAX_PAUSE(900s) 才自愈.
#   训练后补录完全不碰这套东西, 而且不占训练墙钟.
# 代价是看不到"实时"进展 —— 但 1 小时量级的 run 不需要实时.
#
# 每条录完让进程**完全退出**再录下一条, 中间静置 SETTLE 秒 (CLAUDE.md 第 5 条:
# 杀完 Isaac 不等 10s 会触发 carb mutex 崩溃).
set -uo pipefail

RUN=${1:?用法: record_run.sh <run_dir> <clip> [robot=dexmate] [--closeup]}
CLIP=${2:?必须指定 clip (与训练一致, 否则录到别的物体)}
ROBOT=${3:-dexmate}
shift 3 2>/dev/null || shift $#
CLOSEUP=""
for a in "$@"; do [ "$a" = "--closeup" ] && CLOSEUP=1; done

cd "$(dirname "$0")/.."                          # 仓库根
PY=${ISAAC_PYTHON:-/home/lyh/luhr/MagicSim/.venv/bin/python}
SETTLE=${RL_ISAAC_SETTLE:-10}
OUT="$RUN/videos"
mkdir -p "$OUT"

# 近景: 让 7.5cm 的小物体在 720x540 里看得见 (默认视角 7.5m 外, 物体只有几个像素).
CAM=()
if [ -n "$CLOSEUP" ]; then
  CAM=(--eye "0.55,0.72,1.18" --lookat "0.06,0.14,0.87")
fi

# 有 Isaac 占着槽位就别录 —— 不是不安全 (flock 会挡), 而是会白等一个长跑训练的锁.
#
# ⚠ 这里**故意不用 pgrep**: `pgrep -f "rl_rebuild.correction.train"` 会匹配到**自己的
#   命令行** —— 任何 `until ! pgrep -f "...train"; do sleep; done` 的等待循环, 其命令行
#   本身就含这个字符串, 于是永远等不到"退出" (2026-07-27 实测踩到, 训练早已结束却判成在跑).
#   flock 是权威信号: 谁真的占着 GPU 槽位, 与进程叫什么名字无关.
LOCK_DIR=${RL_ISAAC_LOCK_DIR:-$HOME/.cache/rl_correction}
LOCK="$LOCK_DIR/isaac.lock"
mkdir -p "$LOCK_DIR"; touch "$LOCK"
flock -n -E 99 "$LOCK" true
if [ $? -eq 99 ]; then
  echo "[record_run] ⚠ 已有 Isaac 进程占着 GPU 槽位 (多半是训练还在跑)."
  echo "[record_run]   本脚本是**训练后**补录用的, 现在录会一直阻塞等锁."
  echo "[record_run]   要边训边录请用 autorecord.sh 的让出协议."
  exit 1
fi

shopt -s nullglob
CKPTS=("$RUN"/stage1_nn/ep_*_step_*.pth)
CKPTS+=("$RUN/stage1_nn/last.pth")
echo "[record_run] $RUN"
echo "[record_run] 待录 ${#CKPTS[@]} 个快照, 串行 (同一时刻只有一个 Isaac)"

n=0
for f in "${CKPTS[@]}"; do
  [ -e "$f" ] || continue
  n=$((n + 1))
  base=$(basename "$f" .pth)
  echo "[record_run] ($n/${#CKPTS[@]}) $base"
  OMNI_KIT_ACCEPT_EULA=YES ISAAC_SIM_ACCEPT_EULA=1 SHARPA_WANDB=0 PYTHONPATH=. \
    "$PY" -u -m rl_rebuild.correction.record \
      --checkpoint "$f" --out_dir "$OUT" --clip "$CLIP" --robot "$ROBOT" \
      --num_envs 1 --rsi_prob 0 --no_stop_on_done --headless "${CAM[@]}" \
      > "$OUT/$base.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[record_run]   ✗ 失败 (rc=$rc), 详见 $OUT/$base.log"
  else
    echo "[record_run]   ✓ $(ls -t "$OUT"/*.mp4 2>/dev/null | head -1)"
  fi
  sleep "$SETTLE"                                # 等显存/carb 释放干净再起下一个
done
echo "[record_run] 完成. 视频在 $OUT/"
ls -la "$OUT"/*.mp4 2>/dev/null
