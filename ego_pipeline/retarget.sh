#!/usr/bin/env bash
# 命令 2：把 ReconstructOutput 转成仿真格式（与 reconstruct.sh 选 take 的方式一致）
#   读 Output/ReconstructOutput/<dataset>/<嵌套 take>/world_fused.npz(+ mesh)
#   写 Output/RetargetOutput/<dataset>/<嵌套 take>/replay_world.npz + object.usd  （结构镜像 recon）
#
#   ./retarget.sh                         # 全部（默认）
#   ./retarget.sh 10                      # 前 10 条
#   ./retarget.sh <Data路径...>           # 指定 take：take 目录 / 父目录(递归) / 视频 / id（同 reconstruct.sh）
#     例: ./retarget.sh Data/HOI4D/HOI4D_release/ZY20210800001/H1/C1/N19/S100/s02
#   --skip-usd 只出 replay_world.npz（不启 Isaac）；--force 重做。
#
# 内部跨环境调用：recon_to_replay.py(hawor 环境跑 MANO) + obj_to_usd.py(.venv-isaac)。
# 路径默认 Output/{Reconstruct,Retarget}Output，可用 $RECON_FINAL_ROOT/$RETARGET_FINAL_ROOT 覆盖。
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo_paths.sh"
BIV2AP="$RR_ROOT"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECON_ROOT="${RECON_FINAL_ROOT:-$BIV2AP/Output/ReconstructOutput}"
DATASET=hoi4d

N=""; TAKES=(); PASS=()
for a in "$@"; do
  case "$a" in
    [0-9]*) N="$a" ;;
    -*)     PASS+=("$a") ;;          # 透传 retarget.py 的 flag（--force / --skip-usd 等）
    *)      TAKES+=("$a") ;;         # take 目录 / 父目录 / 视频 / id
  esac
done

if [[ ${#TAKES[@]} -gt 0 ]]; then
  # 路径/ id 统一解析成 id，再映射到 ReconstructOutput 下的嵌套 take 目录传给 retarget.py
  mapfile -t IDS < <(conda run -n base python "$HERE/bin/ids_from_paths.py" "${TAKES[@]}" | grep .)
  [[ ${#IDS[@]} -gt 0 ]] || { echo "[retarget] 路径没解析出任何 take: ${TAKES[*]}" >&2; exit 1; }
  PATHS=()
  for id in "${IDS[@]}"; do PATHS+=("$RECON_ROOT/$DATASET/${id//__//}"); done
  echo "[retarget] 指定 ${#PATHS[@]} 条 take"
  exec python "$BIV2AP/ego_pipeline/bridge/retarget.py" "${PATHS[@]}" "${PASS[@]}"
else
  exec python "$BIV2AP/ego_pipeline/bridge/retarget.py" ${N:+--limit "$N"} "${PASS[@]}"
fi
