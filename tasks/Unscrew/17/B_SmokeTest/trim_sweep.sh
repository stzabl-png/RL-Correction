#!/usr/bin/env bash
# 左站位径向收紧扫描 (沙盒: 母带写到 launch_logs/sandbox, 机器段占位, 瓶钉住量指垫几何)
set -uo pipefail; cd "$(dirname "$0")/../../../.."
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
export UNSCREW_CLIP=17 UNSCREW_DETACH=${UNSCREW_DETACH:-twist} UNSCREW_RIGHT_CL=1 POUR_UNLOCK=1,2,3 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. RL_ISAAC_NO_GUARD=1 UNSCREW_NO_PLAN=1
for TR in ${TRIMS:--0.006 0.006 0.012}; do
  OUT=launch_logs/sandbox/v1_trim${TR}.npz
  UNSCREW_RADIAL_TRIM=$TR timeout 900 $PY -u tasks/Unscrew/part4/A_Design/L2_Reference/make_reference.py --out $OUT > launch_logs/sandbox/v1_trim${TR}.log 2>&1 || { echo "[sweep] trim $TR v1 失败"; continue; }
  POUR_REF_NPZ=$OUT UNSCREW_RADIAL_TRIM=$TR timeout 900 $PY -u tasks/Unscrew/part4/B_SmokeTest/probe_grasp.py --pin_bottle --headless > launch_logs/sandbox/grasp_trim${TR}.log 2>&1 &
  wp=$!; while kill -0 $wp 2>/dev/null; do grep -q "站位垫 L" launch_logs/sandbox/grasp_trim${TR}.log && { sleep 15; break; }; sleep 10; done
  P=$(ps -u $USER -o pid=,comm=,args= | awk '$2 ~ /^python/ && /probe_grasp/ {print $1}'); [ -n "$P" ] && kill -9 $P; sleep 5
  echo "[sweep] trim $TR: $(grep '站位垫 L' launch_logs/sandbox/grasp_trim${TR}.log | tail -1) | $(grep -E '^   (thumb|index|middle|ring|pinky) +瓶系' launch_logs/sandbox/grasp_trim${TR}.log | awk '{printf "%s r%s F%s; ", $1, $6, $NF}')"
done
echo "[sweep] done"
