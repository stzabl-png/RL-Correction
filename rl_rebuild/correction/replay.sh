#!/usr/bin/env bash
# 一条命令回放重建轨迹 —— DexMate 手臂 + Sharpa 手用 IK 跟踪, 启动即播放.
#
#   ./rl_rebuild/correction/replay.sh            # 默认 Grasp2
#   ./rl_rebuild/correction/replay.sh Grasp7     # 换 clip (Grasp0 ~ Grasp19)
#
# 相机默认斜俯视全景 (0.9,0.9,1.35 -> 0,0,0.90). 改机位加 --eye/--lookat.
# 默认按源帧率(15fps)实时节拍播放, 和录像/原视频速度一致.
# 不需要开控制器 —— 控制器(dexmate_ctl)只是调机器人关节角用的.
set -uo pipefail
CLIP="${1:-Grasp2}"; shift || true
export VIEWER_FOLLOW=1 VIEWER_PLAY=1 VIEWER_REALTIME=1
exec "$(dirname "$0")/viewer_loop.sh" --clip "$CLIP" "$@"
