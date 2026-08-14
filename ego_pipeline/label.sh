#!/usr/bin/env bash
# 本地标注（OpenCV 窗口，GPU 预览，绿色 mask）—— 你习惯的那个。
# 结果存进 Output/ReconstructOutput/interim/.../sam2_object/label_prompt.json，
# 之后 ./reconstruct.sh <同样选择> 会用 --skip-label 直接重建。
#
#   ./label.sh                        # 标前 10 条
#   ./label.sh 25                     # 标前 25 条
#   ./label.sh <Data路径...>          # 标指定 take：take 目录 / 父目录(递归) / 视频 / id（同 reconstruct.sh）
#     例: ./label.sh Data/HOI4D/HOI4D_release/ZY20210800001/H1/C6/N24/S248/s01/T2
#
# 窗口操作：左键=物体(绿点) 右键=背景(红点) | s=保存并进下一条 | u=撤销 | c=清空 | a/d 或 ←/→ 换帧 | q/ESC 退出
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo_paths.sh"
BIV2AP="$RR_ROOT"
RECON="$RECON_PIPELINE"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RECON_FINAL_ROOT="$BIV2AP/Output/ReconstructOutput"
export RECON_INTERIM_ROOT="$BIV2AP/Output/ReconstructOutput/interim"
export RECON_FINAL_NESTED=1

N=10; TAKES=(); PASS=()
for a in "$@"; do
  case "$a" in
    [0-9]*) N="$a" ;;
    -*)     PASS+=("$a") ;;
    *)      TAKES+=("$a") ;;
  esac
done

if [[ ${#TAKES[@]} -gt 0 ]]; then
  LIST="$RECON_INTERIM_ROOT/_lists/selected.txt"
  mkdir -p "$(dirname "$LIST")"
  conda run -n base python "$HERE/bin/ids_from_paths.py" "${TAKES[@]}" | grep . > "$LIST" || true
  [[ -s "$LIST" ]] || { echo "[label] 路径没解析出任何 take: ${TAKES[*]}" >&2; exit 1; }
  echo "[label] 本地标注指定 $(wc -l < "$LIST") 条（绿 mask, GPU）。列表: $LIST"
else
  LIST="$RECON_INTERIM_ROOT/_lists/first_${N}.txt"
  mkdir -p "$(dirname "$LIST")"
  conda run -n base python "$HERE/bin/first_n_list.py" "$N" | grep . > "$LIST"
  echo "[label] 本地标注前 $N 条（绿 mask, GPU）。列表: $LIST ($(wc -l < "$LIST") 条)"
fi

# HV2RD env 已废弃(那个仓 2026-08-10 收编删除, env 名一直留到 2026-08-14)。
# codetr 是它的超集: 同 torch/sam2/decord, 另有 HOI-DETR 的 mmcv+transformers。
exec conda run --no-capture-output -n codetr python "$RECON/sam2_object/label_object.py" \
  --dataset hoi4d --dataset-root "$BIV2AP/Data/HOI4D" \
  --video-list "$LIST" --label-mode headed --preview-device cuda --gpu 0 ${PASS[@]+"${PASS[@]}"}
