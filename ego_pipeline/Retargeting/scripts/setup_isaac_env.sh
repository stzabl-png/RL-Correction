#!/usr/bin/env bash
# Isaac Sim + Isaac Lab in a uv venv (.venv-isaac) + the retarget stack.
# First run downloads GB-scale isaacsim wheels from pypi.nvidia.com.
#
# isaaclab[isaacsim]==2.3.0 pulls the matching isaacsim wheels automatically
# (2.2.0/2.3.0 are the versions sharpa-rl-lab supports). The retarget stack is
# installed with a numpy constraint so isaacsim's numpy is never touched;
# dex-retargeting goes in with --no-deps (its numpy>=2 pin is declarative only,
# see docs/references/03_dex_retargeting.md §6).
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.11 .venv-isaac
P=$PWD/.venv-isaac/bin/python

# isaaclab pins flatdict==4.0.1 (sdist-only) whose build needs pkg_resources;
# setuptools>=81 removed pkg_resources, so pin it below and build w/o isolation.
uv pip install -p "$P" "setuptools<81" wheel
uv pip install -p "$P" --no-build-isolation --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com "isaaclab[isaacsim]==2.3.0"

NP=$("$P" -c "import numpy; print(numpy.__version__)")
C=$(mktemp); echo "numpy==$NP" > "$C"
uv pip install -p "$P" -c "$C" nlopt pin pytransform3d anytree pyyaml lxml pyzmq wuji-sdk
uv pip install -p "$P" --no-deps -e "$PWD/../dex-retargeting"

"$P" -c "
import numpy as n, dex_retargeting, pinocchio, isaaclab
assert n.__version__ == '$NP', 'numpy got upgraded: ' + n.__version__
print('OK: isaaclab', isaaclab.__version__, '| numpy', n.__version__, '| pinocchio', pinocchio.__version__)
"
echo "done. first launch must accept the Omniverse EULA, e.g.:"
echo "  OMNI_KIT_ACCEPT_EULA=YES $P sim/teleop_isaac_single.py --source mock --headless --duration 5"
