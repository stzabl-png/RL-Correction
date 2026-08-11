#!/usr/bin/env bash
# 评测批量跳过了 contact 步(它最贵的是热度图对齐 ~7min/物体/手, 与 confidence 校准无关),
# 但那一步的**前两小步**产出正是候选判据需要的东西:
#   1) bridge  world_fused -> replay_world.npz   (手关节 21×3, 世界系)   ~15s
#   2) detect  2D mask 邻接 -> contact_auto.json (接触区间, 与深度无关)  ~30s
# 这里只补这两步, 纯 CPU, 不与 GPU 批量抢卡。幂等: 产物在就跳过。
set -uo pipefail
RR=/home/lyh/Project/Reconstruct_and_Retarget
EGO=$RR/ego_pipeline
PY=${HAWOR_PYTHON:-/home/lyh/anaconda3/envs/hawor/bin/python}
[ -x "$PY" ] || PY="conda run --no-capture-output -n hawor python"

ok=0; skip=0; fail=0
for take in "$@"; do
  [ -f "$take/world_fused.npz" ] || { echo "  跳过(无 world_fused): $take"; skip=$((skip+1)); continue; }
  if [ -f "$take/replay_world.npz" ] && [ -f "$take/contact_auto.json" ]; then
    echo "  已就位: $(basename $take)"; skip=$((skip+1)); continue
  fi
  echo "== $(basename $(dirname $take))/$(basename $take)"
  if [ ! -f "$take/replay_world.npz" ]; then
    $PY "$EGO/bridge/recon_to_replay.py" --in "$take" --out "$take/" >/dev/null 2>&1 \
      || { echo "    X bridge 失败"; fail=$((fail+1)); continue; }
  fi
  if [ ! -f "$take/contact_auto.json" ]; then
    ( cd "$EGO" && $PY -m phase.detect "$take" >/dev/null 2>&1 ) \
      || { echo "    X detect 失败"; fail=$((fail+1)); continue; }
  fi
  echo "    OK"; ok=$((ok+1))
done
echo "postproc: 新增 $ok, 跳过 $skip, 失败 $fail"
