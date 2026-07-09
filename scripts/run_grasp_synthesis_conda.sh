#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${OCIR_CUROBO_CONDA_ENV:-${OCIR_GRASP_SYNTHESIS_CONDA_ENV:-env_isaacsim}}"
DATA_ROOT="${OCIR_DATA_ROOT:-/data/users/hangkes2/OCIR}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH_VALUE="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <script.py> [args...]" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH; activate base or use the full conda path first." >&2
  exit 2
fi

OCIR_DATA_ROOT="${DATA_ROOT}" \
PYTHONPATH="${PYTHONPATH_VALUE}" \
conda run --no-capture-output -n "${CONDA_ENV}" python "$@"
