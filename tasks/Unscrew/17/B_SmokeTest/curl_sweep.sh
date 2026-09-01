#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")/../../../.."
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
export UNSCREW_CLIP=17 UNSCREW_DETACH=pull UNSCREW_RIGHT_CL=1 POUR_UNLOCK=1,2,3 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. RL_ISAAC_NO_GUARD=1 UNSCREW_NO_PLAN=1
for C in ${CURLS:-12 20}; do
  OUT=launch_logs/sandbox/v1_curl${C}.npz
  UNSCREW_LEFT_CURL=$C timeout 900 $PY -u tasks/Unscrew/part4/A_Design/L2_Reference/make_reference.py --out $OUT > launch_logs/sandbox/v1_curl${C}.log 2>&1 || { echo "[curl] $C v1 失败"; continue; }
  POUR_REF_NPZ=$OUT UNSCREW_LEFT_CURL=$C $PY -u tasks/Unscrew/part4/B_SmokeTest/probe_grasp.py --pin_bottle --headless > launch_logs/sandbox/grasp_curl${C}.log 2>&1 &
  wp=$!; while kill -0 $wp 2>/dev/null; do grep -q "站位垫 L" launch_logs/sandbox/grasp_curl${C}.log && { sleep 12; break; }; sleep 8; done; kill -9 $wp 2>/dev/null; sleep 5
  echo "[curl] $C°: $(grep '站位垫 L' launch_logs/sandbox/grasp_curl${C}.log | tail -1) | $(grep -E '^   (thumb|index|middle|ring|pinky) +瓶系' launch_logs/sandbox/grasp_curl${C}.log | awk '{printf "%s r%s F%s; ", $1, $6, $NF}')"
done; echo "[curl] done"
