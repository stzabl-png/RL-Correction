#!/usr/bin/env bash
# 2026-07-08 CHAMPION: minimal design + HIGH-WATER NET-PROGRESS income, point-cloud-conditioned,
# on a SINGLE GraspXL object (137c1d2f, 7.1 cm). Result on that object:
#   PHYSICAL net rotation = +34.442 rad/episode (p90 63.9) — treadmill-proof, ALL-steps metric,
#   held-frac 0.68, ~30 rotation bursts/episode (cyclic re-grip+rotate), closed-loop recovery 0.92.
# NOTE: this is the PHYSICAL-net metric (counts every in-play step, slip-back subtracts). It is a
#   STRICTER measure than the "held rotation/episode" column in the README (+21.8 champion) — do not
#   compare the two numbers directly.
# Champion checkpoint: checkpoints/minimal_design_netprog_phys34.pth  (this run's last.pth == argmax).
# Warm-start:          checkpoints/netprog_warmstart.pth              (Stage-1 net-progress pilot argmax).
#
# This is the SINGLE-object success. Passing more objects via SHARPA_OBJ_IDS=a,b,c trains a
# generalist and DILUTES per-object depth (137c1d2f fell 34.4 -> 1.1 in a 6-object run) — that is a
# budget/interference effect, not this config. Keep it single-object to reproduce 34.4.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
OBJ=137c1d2f391a4b7c9234e035473d0ae3   # single 7.1 cm GraspXL object

OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SHARPA_SELF_COLLISION=1 \
SHARPA_BAND_HELD=1 \
SHARPA_BAND_FLOOR=0.06 SHARPA_BAND_SCALE=8.0 SHARPA_BAND_CEIL=1.2 SHARPA_BAND_EMA=0.85 \
SHARPA_SPARSE_THETA=0.15 SHARPA_SPARSE_BONUS=2.5 \
SHARPA_FORCE_SCALE=2.0 \
SHARPA_EPISODE_S=40 \
SHARPA_PC=1 SHARPA_NUM_POSES=40 \
SHARPA_NETPROG_W=15.0 \
SHARPA_CENTER_SCALE=0.0 \
SHARPA_BAND_NSCALE=3.0 SHARPA_BAND_SCALE_END=2.0 SHARPA_SCAFFOLD_ANNEAL=50000 \
SHARPA_DROP_PENALTY=15.0 \
SHARPA_OBJ_IDS=$OBJ \
$PY rl_rebuild/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-NetPC-v1 \
  --max_agent_steps 50000000 --num_envs 512 --headless \
  --resume --load_path checkpoints/netprog_warmstart.pth "$@"

# Selection discipline (see README): training reward is NOT a model-selection signal. Sweep saved
# snapshots with rl_rebuild/scripts/snapshot_probe.py and take the argmax; measure the PHYSICAL
# net-rot line (not held-cum) with rl_rebuild/scripts/rot_held_probe.py at episodes >= 2-3x the cap.
