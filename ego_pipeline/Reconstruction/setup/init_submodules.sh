#!/usr/bin/env bash
# Populate Reconstruction/third_party with the 6 model repos, pinned to the SAME
# commits HV2RD currently uses (HV2RD is the version reference / git bridge).
# Reconstruct_and_Retarget is not a git repo, so these are plain clones (not git submodules).
#
#   bash init_submodules.sh            # clone any missing repos at HV2RD's pinned commit
#   bash init_submodules.sh --apply    # alias for the same (kept for symmetry)
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/repo_paths.sh"
HV2RD="$HV2RD_ROOT"
TP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party"
mkdir -p "$TP"

# name|url  (commit is read from HV2RD at runtime so we always match the bridge)
REPOS=(
  "vipe|https://github.com/nv-tlabs/vipe.git"
  "sam2|https://github.com/facebookresearch/sam2.git"
  "sam3|https://github.com/jiaka1chen/sam3.git"
  "hawor|https://github.com/ThunderVVV/HaWoR.git"
  "sam-3d-objects|https://github.com/facebookresearch/sam-3d-objects.git"
  "FoundationPose|https://github.com/NVlabs/FoundationPose.git"
)

for entry in "${REPOS[@]}"; do
  name="${entry%%|*}"; url="${entry#*|}"
  dst="$TP/$name"
  commit="$(git -C "$HV2RD/third_party/$name" rev-parse HEAD 2>/dev/null || echo '')"
  if [[ -d "$dst/.git" ]]; then
    echo "[init] $name already cloned -> skip (pinned $commit)"; continue
  fi
  echo "[init] cloning $name ($url) @ ${commit:-HEAD}"
  git clone "$url" "$dst"
  [[ -n "$commit" ]] && git -C "$dst" checkout -q "$commit"
done

echo "[init] third_party ready under $TP"
echo "[init] next: bash setup/build_exts.sh  (build sam2 _C / FP mycpp / ViPE .venv against THESE copies)"
