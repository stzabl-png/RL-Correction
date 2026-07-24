#!/usr/bin/env bash
# Install the retarget stack into an Isaac Sim / Isaac Lab python WITHOUT
# touching its numpy/torch - enables the single-process mode
# (sim/teleop_isaac_single.py) and the zmq consumer (sim/test_env_sharpa.py).
#
# dex-retargeting declares numpy>=2.0 but uses no numpy-2-only APIs; it is
# installed with --no-deps and runs on Isaac's numpy 1.26 (verified, identical
# tracking results - docs/references/03_dex_retargeting.md §6).
#
# Usage:
#   bash scripts/install_into_isaaclab.sh ~/isaacsim/python.sh
#   bash scripts/install_into_isaaclab.sh <isaaclab-conda-env>/bin/python
set -euo pipefail
PY=${1:?usage: install_into_isaaclab.sh <isaac-python, e.g. ~/isaacsim/python.sh>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DEX=$(cd "$REPO/../dex-retargeting" && pwd)

NP=$("$PY" -c "import numpy; print(numpy.__version__)")
C=$(mktemp)
echo "numpy==$NP" > "$C"   # hard-pin Isaac's numpy during resolution

"$PY" -m pip install -c "$C" nlopt pin pytransform3d anytree pyyaml lxml pyzmq wuji-sdk
"$PY" -m pip install --no-deps -e "$DEX"

"$PY" -c "
import numpy, dex_retargeting, pinocchio, nlopt, zmq
assert numpy.__version__ == '$NP', 'numpy got upgraded: ' + numpy.__version__
print('OK: numpy', numpy.__version__, '| pinocchio', pinocchio.__version__)
"
echo "done. try: $PY $REPO/sim/teleop_isaac_single.py --source mock"
