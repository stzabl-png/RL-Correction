#!/usr/bin/env bash
# 从把关链的某一级续跑 (START_FROM=pull|v2|accept), 过了就发射
set -uo pipefail; cd "$(dirname "$0")/../../../.."
NAME=${1:-U17_pull_cl_s51}; SEED=${2:-51}; export START_FROM=${START_FROM:-pull}
echo "[night] $(date +%H:%M) resume chain from $START_FROM"
bash tasks/Unscrew/17/C_Wiring/stage_chain.sh > launch_logs/u17_chain2.log 2>&1; rc=$?
tail -20 launch_logs/u17_chain2.log
[ $rc -eq 0 ] || { echo "[night] ❌ 把关链未过, 不发射"; exit 1; }
echo "[night] $(date +%H:%M) LAUNCH $NAME"; nohup bash tasks/Unscrew/17/C_Wiring/launch_local.sh "$NAME" "$SEED" 512 HYB 1 > /dev/null 2>&1 &
sleep 5; echo "[night] launched pid $!"
