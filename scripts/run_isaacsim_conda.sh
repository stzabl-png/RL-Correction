#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${OCIR_ISAACSIM_CONDA_ENV:-env_isaacsim}"
DATA_ROOT="${OCIR_DATA_ROOT:-/data/users/hangkes2/OCIR}"
ISAACSIM_MODE="${OCIR_ISAACSIM_MODE:-webrtc}"
OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-maniskill}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <script.py> [args...]" >&2
  exit 2
fi

if [[ "${OMNI_KIT_ACCEPT_EULA}" != "YES" ]]; then
  echo "OMNI_KIT_ACCEPT_EULA is not YES; Isaac startup may stop at NVIDIA's Omniverse EULA prompt." >&2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH; activate base or use the full conda path first." >&2
  exit 2
fi

OCIR_DATA_ROOT="${DATA_ROOT}" \
OCIR_ISAACSIM_MODE="${ISAACSIM_MODE}" \
OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA}" \
MPLCONFIGDIR="${MPLCONFIGDIR}" \
conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
