#!/usr/bin/env bash
# Vendor the parts of MagicDexMate that Reconstruct_and_Retarget's Retargeting needs.
# One direction: MagicDexMate (your retarget repo) -> Reconstruct_and_Retarget/ego_pipeline/Retargeting.
# Edit retarget code in MagicDexMate, then re-run this to refresh the vendored copy.
#
#   sync_retarget.sh            # dry-run
#   sync_retarget.sh --apply    # vendor for real
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/repo_paths.sh"
SRC="$THIRD_PARTY/MagicDexMate"
DST="$RR_ROOT/ego_pipeline/Retargeting"

# Only the pieces the recon->sim path needs (assets kept for paper release).
INCLUDE=(magicdexmate sim scripts configs assets)
EXCLUDES=(
  --exclude '__pycache__/' --exclude '*.py[cod]' --exclude '.DS_Store'
  --exclude '.venv*'       --exclude '.git*'     --exclude 'RetargetInput/'
  --exclude '.pytest_cache/'
)

mode="${1:-}"
mkdir -p "$DST"
FLAGS=(-a --delete --itemize-changes "${EXCLUDES[@]}")
if [[ "$mode" == "--apply" ]]; then
  echo "[sync_retarget] APPLY : $SRC/{${INCLUDE[*]}} -> $DST"
  for sub in "${INCLUDE[@]}"; do
    [[ -e "$SRC/$sub" ]] && rsync "${FLAGS[@]}" "$SRC/$sub" "$DST/"
  done
  echo "[sync_retarget] done."
else
  echo "[sync_retarget] DRY-RUN (no changes). Re-run with --apply to vendor."
  for sub in "${INCLUDE[@]}"; do
    [[ -e "$SRC/$sub" ]] && rsync "${FLAGS[@]}" -n "$SRC/$sub" "$DST/" || true
  done
fi
