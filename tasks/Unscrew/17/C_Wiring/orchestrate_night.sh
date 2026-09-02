#!/usr/bin/env bash
# 一夜编排: 规划机器段 -> 母带 v1 -> 把关链 -> 发射. 用法: bash orchestrate_night.sh NAME SEED
set -uo pipefail
cd "$(dirname "$0")/../../../.."
NAME=${1:-U17_pull_cl_s51}; SEED=${2:-51}
PY=/home/lyh/luhr/MagicSim/.venv/bin/python; D=tasks/Unscrew/part4
export UNSCREW_CLIP=17 UNSCREW_DETACH=${UNSCREW_DETACH:-twist} UNSCREW_RIGHT_CL=1 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. TMPDIR=$HOME/tmp
mkdir -p launch_logs $TMPDIR
reap() { P=$(ps -u $USER -o pid=,comm=,args= | awk -v pat="$1" '$2 ~ /^python/ && $0 ~ pat {print $1}'); [ -n "$P" ] && kill -9 $P 2>/dev/null; sleep 10; }
# 先按当前先验/配置重建 v1 (规划摘要以它为基底; 2026-09-01 10:00 实测: 漏这一步会拿旧站位规划 → 摘要不符断言)
rm -f $D/A_Design/L1_Data/Motion_Planning/17/Approach.npz $D/A_Design/L1_Data/Motion_Planning/17/Retreat.npz
echo "[night] $(date +%H:%M) make_reference (pre-plan)"; timeout 1500 $PY -u $D/A_Design/L2_Reference/make_reference.py > launch_logs/u17_v1_preplan.log 2>&1 || { echo "[night] ❌ v1 (pre-plan)"; exit 1; }
echo "[night] $(date +%H:%M) plan approach"; timeout 2400 $PY -u $D/A_Design/L1_Data/Motion_Planning/plan_machine_segs.py --headless > launch_logs/u17_plan_app.log 2>&1; reap plan_machine_segs
grep -q "\[plan\] ✅" launch_logs/u17_plan_app.log || { echo "[night] ❌ approach 规划失败"; exit 1; }
echo "[night] $(date +%H:%M) plan retreat"; timeout 2400 $PY -u $D/A_Design/L1_Data/Motion_Planning/plan_machine_segs.py --retreat --headless > launch_logs/u17_plan_ret.log 2>&1; reap plan_machine_segs
grep -q "\[plan\] ✅\|反向兜底成功\|倒放" launch_logs/u17_plan_ret.log || { echo "[night] ❌ retreat 规划失败"; exit 1; }
echo "[night] $(date +%H:%M) make_reference"; timeout 1500 $PY -u $D/A_Design/L2_Reference/make_reference.py > launch_logs/u17_v1_final.log 2>&1 || { echo "[night] ❌ v1"; exit 1; }
grep -n "Approach =\|Retreat =\|机器段终点\|right IK\|已写" launch_logs/u17_v1_final.log | cut -c1-160
echo "[night] $(date +%H:%M) stage chain"; bash tasks/Unscrew/17/C_Wiring/stage_chain.sh > launch_logs/u17_chain.log 2>&1; rc=$?
tail -25 launch_logs/u17_chain.log
[ $rc -eq 0 ] || { echo "[night] ❌ 把关链未过, 不发射"; exit 1; }
echo "[night] $(date +%H:%M) LAUNCH $NAME"; nohup bash tasks/Unscrew/17/C_Wiring/launch_local.sh "$NAME" "$SEED" 512 HYB 1 > /dev/null 2>&1 &
sleep 5; echo "[night] launched pid $!"
