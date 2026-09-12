#!/usr/bin/env bash
# One-shot Python environment for dexmate-calib (no sudo needed).
#
#   scripts/setup_env.sh              # intrinsics + hand-eye + Kinect (all extras)
#   scripts/setup_env.sh --minimal    # intrinsics / offline solving only (numpy, opencv, pyyaml)
#   scripts/setup_env.sh --no-kinect  # robot + FK but skip pyk4a (no Kinect SDK on this box)
#
# Installs uv into ~/.local/bin if missing, pins a uv-managed CPython 3.12 (ships its own
# headers, so pyk4a builds without python3-dev), creates ./.venv and verifies the imports.
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRAS=(--extra dev --extra robot --extra kinect)
case "${1:-}" in
  --minimal)   EXTRAS=(--extra dev);;
  --no-kinect) EXTRAS=(--extra dev --extra robot);;
  "") ;;
  *) echo "usage: $0 [--minimal|--no-kinect]"; exit 2;;
esac

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup] %s\033[0m\n' "$*" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
log "uv $(uv --version | awk '{print $2}')"

if [[ " ${EXTRAS[*]} " == *" kinect "* ]]; then
  if [[ ! -e /usr/include/k4a/k4a.h ]]; then
    die "Kinect SDK headers not found (/usr/include/k4a/k4a.h). Run: sudo scripts/install_kinect_sdk.sh  or use --no-kinect"
  fi
fi

log "installing uv-managed CPython 3.12"
uv python install 3.12 >/dev/null
[[ -f .python-version ]] || echo "3.12" > .python-version

log "uv sync ${EXTRAS[*]}"
uv sync --managed-python "${EXTRAS[@]}"

log "verifying imports"
.venv/bin/python - <<'PY'
import importlib, sys
want = ["numpy", "cv2", "yaml", "pytest", "dexmate_calib"]
opt = ["dexcontrol", "dexmate_urdf", "pinocchio", "pyk4a"]
print("python", sys.version.split()[0], sys.executable)
bad = []
for m in want + opt:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} OK {getattr(mod, '__version__', '')}")
    except Exception as exc:
        (bad if m in want else []).append(m)
        print(f"  {m:14s} -- ({type(exc).__name__})")
if bad:
    raise SystemExit(f"required modules failed: {bad}")
PY
log "done. Activate with:  source .venv/bin/activate   then try:  dexcalib --help"
