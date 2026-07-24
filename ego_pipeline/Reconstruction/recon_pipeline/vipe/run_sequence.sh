#!/usr/bin/env bash
# Run ViPE step under third_party/vipe uv environment.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash -lc "cd '${REPO_ROOT}/third_party/vipe' && uv run python '${REPO_ROOT}/recon_pipeline/vipe/run_sequence.py' \"\$@\"" \
  _ "$@"
