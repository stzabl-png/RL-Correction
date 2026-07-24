#!/usr/bin/env bash
# Vendor-sync the recon pipeline CODE between HV2RD (the git bridge with Jiakai)
# and Reconstruct_and_Retarget's vendored copy. Only pure code is exchanged; heavy assets never move.
#
#   sync_recon.sh pull            # dry-run: show what pulling HV2RD -> Reconstruct_and_Retarget would change
#   sync_recon.sh pull --apply    # actually pull (HV2RD -> Reconstruct_and_Retarget)
#   sync_recon.sh push            # dry-run: show what pushing Reconstruct_and_Retarget -> HV2RD would change
#   sync_recon.sh push --apply    # actually push (Reconstruct_and_Retarget -> HV2RD), then commit+push in HV2RD by hand
#
# Workflow: pull Jiakai's updates = (cd HV2RD && git pull) then `sync_recon.sh pull --apply`.
#           push Reconstruct_and_Retarget recon edits = `sync_recon.sh push --apply` then (cd HV2RD && git commit && git push).
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/repo_paths.sh"
HV2RD="$RECON_PIPELINE/"
BIV2AP="$RR_ROOT/ego_pipeline/Reconstruction/recon_pipeline/"

# Only pure code crosses; exclude caches and anything heavy/generated.
EXCLUDES=(
  --exclude '__pycache__/'  --exclude '*.py[cod]'
  --exclude '.DS_Store'     --exclude '.ipynb_checkpoints/'
  --exclude 'data/'         --exclude 'weights/'  --exclude '*.pt' --exclude '*.pth'
)

dir="${1:-}"; mode="${2:-}"
case "$dir" in
  pull) SRC="$HV2RD";   DST="$BIV2AP" ;;
  push) SRC="$BIV2AP";  DST="$HV2RD"  ;;
  *) echo "usage: $0 <pull|push> [--apply]" >&2; exit 2 ;;
esac

mkdir -p "$DST"
FLAGS=(-a --delete --itemize-changes "${EXCLUDES[@]}")
if [[ "$mode" == "--apply" ]]; then
  echo "[sync_recon] $dir APPLY : $SRC -> $DST"
  rsync "${FLAGS[@]}" "$SRC" "$DST"
  echo "[sync_recon] done."
else
  echo "[sync_recon] $dir DRY-RUN (no changes). Re-run with --apply to sync."
  echo "             $SRC -> $DST"
  rsync "${FLAGS[@]}" -n "$SRC" "$DST" || true
  echo "[sync_recon] (above '*deleting' / 'cd+++/' lines are what --apply WOULD do)"
fi
