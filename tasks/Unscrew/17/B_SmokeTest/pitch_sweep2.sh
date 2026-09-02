#!/usr/bin/env bash
# 左先验变体扫描 (沙盒 v1, 机器段占位): 每个变体 → 瓶钉住量指垫几何/力 + 手最低点; 再瓶自由看合拢会不会推倒
set -uo pipefail; cd "$(dirname "$0")/../../../.."
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
export UNSCREW_CLIP=17 UNSCREW_DETACH=${UNSCREW_DETACH:-twist} UNSCREW_RIGHT_CL=1 POUR_UNLOCK=1,2,3 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. RL_ISAAC_NO_GUARD=1 UNSCREW_NO_PLAN=1
run_probe() {  # $1 log  $2.. args
  local LOG=$1; shift
  timeout 900 $PY -u tasks/Unscrew/part4/B_SmokeTest/probe_grasp.py "$@" --headless > $LOG 2>&1 &
  local wp=$!; while kill -0 $wp 2>/dev/null; do grep -q "站位垫 L" $LOG && { sleep 15; break; }; sleep 10; done
  local P=$(ps -u $USER -o pid=,comm=,args= | awk '$2 ~ /^python/ && /probe_grasp/ {print $1}'); [ -n "$P" ] && kill -9 $P; sleep 5
}
for V in ${VARIANTS:-P30 P45 P30u2}; do
  OUT=launch_logs/sandbox/v1_$V.npz
  UNSCREW_LEFT_PRIOR=Screw17_bottle_left_$V.npz timeout 900 $PY -u tasks/Unscrew/part4/A_Design/L2_Reference/make_reference.py --out $OUT > launch_logs/sandbox/v1_$V.log 2>&1 || { echo "[pitch] $V v1 失败: $(grep -i 'error\|失败\|Traceback' launch_logs/sandbox/v1_$V.log | tail -2 | cut -c1-150)"; continue; }
  echo "[pitch] $V v1: $(grep -E '站位.*腕|左腕.*离桌|IK.*左|余量' launch_logs/sandbox/v1_$V.log | tail -2 | tr '\n' ' ' | cut -c1-220)"
  export POUR_REF_NPZ=$OUT UNSCREW_LEFT_PRIOR=Screw17_bottle_left_$V.npz
  run_probe launch_logs/sandbox/grasp_${V}_pin.log --pin_bottle
  echo "[pitch] $V 钉住: $(grep '站位垫 L' launch_logs/sandbox/grasp_${V}_pin.log | tail -1) | $(grep '最低点' launch_logs/sandbox/grasp_${V}_pin.log | tail -1) | $(grep -E '^   (thumb|index|middle|ring|pinky) +瓶系' launch_logs/sandbox/grasp_${V}_pin.log | awk '{printf "%s r%s F%s; ", $1, $6, $NF}')"
  run_probe launch_logs/sandbox/grasp_${V}_free.log --pin_approach
  echo "[pitch] $V 缝1自由(接近段钉住): $(grep '站位垫 L' launch_logs/sandbox/grasp_${V}_free.log | tail -1) | $(grep '瓶开始倾倒' launch_logs/sandbox/grasp_${V}_free.log | head -1 | cut -c1-90) | $(grep -E '^\[grasp\] 瓶 pos' launch_logs/sandbox/grasp_${V}_free.log | tail -1 | cut -c1-80)"
done
echo "[pitch] done"
