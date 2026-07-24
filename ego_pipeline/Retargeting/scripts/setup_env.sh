#!/usr/bin/env bash
# Teleop env setup (uv). Re-run safe. From repo root:  bash scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.11 .venv
P=.venv/bin/python
# cpu-only torch is all dex-retargeting needs (and faster to install)
uv pip install -p $P torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p $P -e ../dex-retargeting pyzmq pytest wuji-sdk "sapien==3.0.0b0"

$P scripts/prepare_assets.py
$P scripts/check_env.py
echo "OK. Try: $P scripts/teleop_retarget.py --source mock --viz"
