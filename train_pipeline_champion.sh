#!/usr/bin/env bash
# Trains the PIPELINE champion configuration (pinch -> adjust -> sustained rotation, g=-9.81).
# Produced checkpoint class: +21.8 rad held rotation per episode (see README / docs/FULL_REPORT).
# Selection: sweep snapshots afterwards with rl_rebuild/scripts/snapshot_probe.py (argmax, NOT last.pth).
set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SHARPA_SELF_COLLISION=1 \
SHARPA_POSE_CACHE=poses/n4_pose1.npy \
SHARPA_BAND_HELD=1 \
SHARPA_BAND_NSCALE=3.0 \
SHARPA_BAND_FLOOR=0.06 \
SHARPA_BAND_SCALE=8.0 \
SHARPA_BAND_CEIL=1.2 \
SHARPA_CENTER_SCALE=0.3 \
SHARPA_BAND_EMA=0.85 \
SHARPA_SPARSE_THETA=0.15 \
SHARPA_SPARSE_BONUS=2.5 \
SHARPA_FORCE_SCALE=2.0 \
SHARPA_ENC_FRAC=0.5 \
SHARPA_EPISODE_S=40 \
$PY rl_rebuild/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-Net-v1 \
  --max_agent_steps 24000000 --num_envs 768 --headless "$@"
