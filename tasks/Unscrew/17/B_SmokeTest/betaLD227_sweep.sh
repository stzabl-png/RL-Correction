#!/usr/bin/env bash
# LD227 squeeze 剂量扫描: 复用沙盒 v1_LD227.npz, 钉住量五垫力; 最佳剂量再跑缝1自由看推不推倒
set -uo pipefail; cd "$(dirname "$0")/../../../.."
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
export UNSCREW_CLIP=17 UNSCREW_DETACH=pull UNSCREW_RIGHT_CL=1 POUR_UNLOCK=1,2,3 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. RL_ISAAC_NO_GUARD=1 UNSCREW_NO_PLAN=1
export POUR_REF_NPZ=launch_logs/sandbox/v1_LD227.npz UNSCREW_LEFT_PRIOR=Screw17_bottle_left_LD227.npz
run_probe() { local LOG=$1; shift
  timeout 900 $PY -u tasks/Unscrew/part4/B_SmokeTest/probe_grasp.py "$@" --headless > $LOG 2>&1 &
  local wp=$!; while kill -0 $wp 2>/dev/null; do grep -q "站位垫 L" $LOG && { sleep 15; break; }; sleep 10; done
  local P=$(ps -u $USER -o pid=,comm=,args= | awk '$2 ~ /^python/ && /probe_grasp/ {print $1}'); [ -n "$P" ] && kill -9 $P; sleep 5; }
for B in 1.3 1.6 2.0; do
  UNSCREW_BETA_L=$B run_probe launch_logs/sandbox/grasp_LD227_b${B}_pin.log --pin_bottle
  echo "[b227] βL=$B 钉住: $(grep '站位垫 L' launch_logs/sandbox/grasp_LD227_b${B}_pin.log | tail -1) | $(grep -E '^   (thumb|index|middle|ring|pinky) +瓶系' launch_logs/sandbox/grasp_LD227_b${B}_pin.log | awk '{print $1"="$NF}' | tr '\n' ' ')"
done
BEST=$(for B in 1.3 1.6 2.0; do L=$(grep -o '站位垫 L[0-9]' launch_logs/sandbox/grasp_LD227_b${B}_pin.log | tail -1 | grep -o '[0-9]'); echo "$L $B"; done | sort -rn | head -1 | awk '{print $2}')
echo "[b227] 最佳剂量 βL=$BEST → 缝1自由"
UNSCREW_BETA_L=$BEST run_probe launch_logs/sandbox/grasp_LD227_b${BEST}_free.log --pin_approach
echo "[b227] βL=$BEST 缝1自由: $(grep '站位垫 L' launch_logs/sandbox/grasp_LD227_b${BEST}_free.log | tail -1) | $(grep '瓶开始倾倒' launch_logs/sandbox/grasp_LD227_b${BEST}_free.log | head -1 | cut -c1-100) | $(grep -E '^\[grasp\] 瓶 pos' launch_logs/sandbox/grasp_LD227_b${BEST}_free.log | tail -1 | cut -c1-80)"
echo "[b227] done"
