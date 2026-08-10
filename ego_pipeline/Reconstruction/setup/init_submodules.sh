#!/usr/bin/env bash
# Populate Reconstruction/third_party with the model repos as plain clones,
# pinned to the commits recorded at HV2RD 收编时 (2026-08-10, 见
# docs/hv2rd_archive/submodule_pins.txt)。HV2RD 仓库已废弃删除;开发机上
# third_party 是收编迁移过来的现成目录(含已编译扩展), 本脚本只服务全新 clone 的机器。
#
#   bash init_submodules.sh            # clone any missing repos at the pinned commit
set -euo pipefail

TP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party"
PINS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/hv2rd_archive/submodule_pins.txt"
mkdir -p "$TP"

# name|url|pinned-commit (来源: submodule_pins.txt, 2026-08-10)
REPOS=(
  "vipe|https://github.com/nv-tlabs/vipe.git|"
  "sam2|https://github.com/facebookresearch/sam2.git|2b90b9f5ceec907a1c18123530e92e794ad901a4"
  "sam3|https://github.com/jiaka1chen/sam3.git|a29e1e7733c98cd58c48e047279e4fcdbd523e41"
  "hawor|https://github.com/ThunderVVV/HaWoR.git|66c7d4108d58a716deccd192cb7645170cdc7bd7"
  "sam-3d-objects|https://github.com/facebookresearch/sam-3d-objects.git|f91db411c50efee93d8db7aeb323885650f6f722"
  "FoundationPose|https://github.com/NVlabs/FoundationPose.git|a1b694b83e633c2cb6115b9063d940a687759392"
)

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url commit <<<"$entry"
  dst="$TP/$name"
  if [[ -d "$dst" && -n "$(ls -A "$dst" 2>/dev/null)" ]]; then
    echo "[init] $name already present -> skip"; continue
  fi
  echo "[init] cloning $name ($url) @ ${commit:-HEAD}"
  git clone "$url" "$dst"
  [[ -n "$commit" ]] && git -C "$dst" checkout -q "$commit"
done

echo "[init] third_party ready under $TP  (pin 台账: $PINS)"
echo "[init] next: bash setup/build_exts.sh  (build sam2 _C / FP mycpp / ViPE .venv against THESE copies)"
