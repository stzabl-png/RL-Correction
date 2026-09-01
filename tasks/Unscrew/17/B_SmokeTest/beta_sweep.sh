#!/usr/bin/env bash
# βL squeeze 剂量扫描 (正式 v2 母带, 瓶钉住, 量站位垫数): 等 trim_sweep 结束再跑
set -uo pipefail; cd "$(dirname "$0")/../../../.."
while pgrep -f trim_sweep.sh >/dev/null; do sleep 30; done
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
export UNSCREW_CLIP=17 UNSCREW_DETACH=pull UNSCREW_RIGHT_CL=1 POUR_UNLOCK=1,2,3 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. RL_ISAAC_NO_GUARD=1
for B in ${BETAS:-0.5 0.7 1.3}; do
  UNSCREW_BETA_L=$B timeout 900 $PY -u tasks/Unscrew/part4/B_SmokeTest/probe_grasp.py --pin_bottle --headless > launch_logs/sandbox/grasp_beta${B}.log 2>&1 &
  wp=$!; while kill -0 $wp 2>/dev/null; do grep -q "站位垫 L" launch_logs/sandbox/grasp_beta${B}.log && { sleep 15; break; }; sleep 10; done
  P=$(ps -u $USER -o pid=,comm=,args= | awk '$2 ~ /^python/ && /probe_grasp/ {print $1}'); [ -n "$P" ] && kill -9 $P; sleep 5
  echo "[beta] βL=$B: $(grep '站位垫 L' launch_logs/sandbox/grasp_beta${B}.log | tail -1) | $(grep -E '^   (thumb|index|middle|ring|pinky) +瓶系' launch_logs/sandbox/grasp_beta${B}.log | awk '{printf "%s r=%s F=%s; ", $1, $(NF-4), $NF}')"
done
# 自由瓶 (不钉) 复核最优一档看能不能真抓住
echo "[beta] done"
