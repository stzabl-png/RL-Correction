#!/usr/bin/env bash
# Unscrew/17 全链参考活样机 (GUI)。用法: bash tasks/Unscrew/17/B_SmokeTest/view_ref.sh [--fps 15] [--selftest --headless]
cd "$(dirname "$0")/../../../.."
export UNSCREW_CLIP=17 UNSCREW_DETACH=pull UNSCREW_RIGHT_CL=1 UNSCREW_LEFT_PRIOR=${UNSCREW_LEFT_PRIOR:-Screw17_bottle_left_LD227.npz}
export SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. TMPDIR=${TMPDIR:-$HOME/tmp} RL_ISAAC_NO_GUARD=1
exec /home/lyh/luhr/MagicSim/.venv/bin/python -u tasks/Unscrew/17/B_SmokeTest/view_reference.py "$@"
