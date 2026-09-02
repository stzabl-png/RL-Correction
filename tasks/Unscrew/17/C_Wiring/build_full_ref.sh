#!/usr/bin/env bash
# 只造参考不发射: v1(pre-plan) -> cuRobo approach/retreat -> v1(final) -> v2 重锚. 供活样机查看 (2026-09-01 §10 U-P0/P2)
set -uo pipefail
cd "$(dirname "$0")/../../../.."
PY=/home/lyh/luhr/MagicSim/.venv/bin/python; D=tasks/Unscrew/part4
export UNSCREW_LEFT_APPROACH_NPZ=${UNSCREW_LEFT_APPROACH_NPZ:-tasks/Unscrew/17/A_Design/L1_Data/Motion_Planning/LeftApproach_LD227.npz}
export UNSCREW_CLIP=17 UNSCREW_DETACH=${UNSCREW_DETACH:-twist} UNSCREW_RIGHT_CL=1 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. TMPDIR=$HOME/tmp RL_ISAAC_NO_GUARD=1
export UNSCREW_LEFT_PRIOR=${UNSCREW_LEFT_PRIOR:-Screw17_bottle_left_LD227.npz}
mkdir -p launch_logs $TMPDIR
reap() { P=$(ps -u $USER -o pid=,comm=,args= | awk -v pat="$1" '$2 ~ /^python/ && $0 ~ pat {print $1}'); [ -n "$P" ] && kill -9 $P 2>/dev/null; sleep 10; }
rm -f $D/A_Design/L1_Data/Motion_Planning/17/Approach.npz $D/A_Design/L1_Data/Motion_Planning/17/Retreat.npz
echo "[ref] $(date +%H:%M) v1 pre-plan (prior=$UNSCREW_LEFT_PRIOR)"
timeout 1500 $PY -u $D/A_Design/L2_Reference/make_reference.py > launch_logs/u17_ref_v1pre.log 2>&1 || { echo "[ref] ❌ v1 pre-plan"; exit 1; }
plan_stage() { # $1 log $2.. extra args — ✅ 落盘即收割 (卡 app.close 白等 40min 的教训)
  local LOG=$1; shift
  timeout 2400 $PY -u $D/A_Design/L1_Data/Motion_Planning/plan_machine_segs.py "$@" --headless > $LOG 2>&1 &
  local wp=$!; while kill -0 $wp 2>/dev/null; do grep -qE "\[plan\] ✅|反向兜底成功|倒放|❌" $LOG 2>/dev/null && { sleep 20; break; }; sleep 10; done
  reap plan_machine_segs; wait $wp 2>/dev/null || true
}
echo "[ref] $(date +%H:%M) plan approach"; plan_stage launch_logs/u17_ref_planapp.log
grep -q "\[plan\] ✅" launch_logs/u17_ref_planapp.log || { echo "[ref] ❌ approach 规划失败"; exit 1; }
echo "[ref] $(date +%H:%M) plan retreat"; plan_stage launch_logs/u17_ref_planret.log --retreat
grep -q "\[plan\] ✅\|反向兜底成功\|倒放" launch_logs/u17_ref_planret.log || { echo "[ref] ❌ retreat 规划失败"; exit 1; }
echo "[ref] $(date +%H:%M) v1 final"; timeout 1500 $PY -u $D/A_Design/L2_Reference/make_reference.py > launch_logs/u17_ref_v1.log 2>&1 || { echo "[ref] ❌ v1 final"; exit 1; }
grep -E "left IK|right IK|已写|站位|G-B" launch_logs/u17_ref_v1.log | tail -6 | cut -c1-150
echo "[ref] $(date +%H:%M) v2 重锚"; timeout 2400 $PY -u $D/B_SmokeTest/build_reference.py --headless > launch_logs/u17_ref_v2.log 2>&1 &
wp=$!; while kill -0 $wp 2>/dev/null; do grep -q "已写\|❌" launch_logs/u17_ref_v2.log && { sleep 20; break; }; sleep 10; done; reap build_reference; wait $wp 2>/dev/null || true
grep -q "已写" launch_logs/u17_ref_v2.log || { echo "[ref] ❌ v2"; tail -3 launch_logs/u17_ref_v2.log; exit 1; }
grep -E "已写|IK|余量|站位" launch_logs/u17_ref_v2.log | tail -5 | cut -c1-150
echo "[ref] ★完整参考已就绪: $D/A_Design/L2_Reference/17/reference_v2.npz"
