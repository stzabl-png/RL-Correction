#!/usr/bin/env bash
# 命令 1：重建 -> Reconstruct_and_Retarget/Output/ReconstructOutput/<dataset>/<嵌套路径>/world_fused.npz(+ mesh)
#
# HOI4D（默认数据集）:
#   ./reconstruct.sh 10                                   # 前 10 条
#   ./reconstruct.sh Data/HOI4D/HOI4D_release/.../s02      # 指定 take：目录/父目录(递归)/视频/id
#   ./reconstruct.sh .../s02/T1 .../s02/T2                 # 指定多条
#
# 其它数据集 / 任意 mp4（如 egodex）:
#   ./reconstruct.sh <视频.mp4> --dataset egodex           # egodex root 已内置
#   ./reconstruct.sh <视频.mp4> --dataset X --root /path/to/X
#   ./reconstruct.sh <目录> --dataset egodex               # 目录下所有 *.mp4
#   video_id = mp4 相对 --root 路径(去后缀, / -> __)，输出嵌套。
#
# 标注：默认本地模式(--skip-label，需先 ./label.sh 标好)；加 --web 走网页标注+重建一条龙。
# 其它 flag 透传 run_batch_queue（--force / --dry-run / --workers-per-gpu 2 ...）。
set -euo pipefail

BIV2AP=/home/lyh/Project/Reconstruct_and_Retarget
RECON=/home/lyh/Project/HumanVideo2RobotData/recon_pipeline
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RECON_FINAL_ROOT="$BIV2AP/Output/ReconstructOutput"
export RECON_INTERIM_ROOT="$BIV2AP/Output/ReconstructOutput/interim"
export RECON_FINAL_NESTED=1   # final 目录镜像原数据集嵌套: A__B__C -> A/B/C
# SAM3 版本自动选择:Blackwell/5090 -> sam3;其它显卡 -> sam3.1,零检出则自动回退 sam3。

# 内置数据集 root(其它数据集用 --root 指定)
declare -A DATASET_ROOTS=(
  [hoi4d]="$BIV2AP/Data/HOI4D"
  [egodex]="/home/lyh/Project/V2AP/data/egocentric/egodex"
)

DATASET=hoi4d; ROOT=""; N=10; WEB=0; TAKES=(); PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --root)    ROOT="$2"; shift 2 ;;
    --web)     WEB=1; shift ;;
    [0-9]*)    N="$1"; shift ;;
    -*)        PASS+=("$1"); shift ;;   # 透传 run_batch_queue（--force 等）
    *)         TAKES+=("$1"); shift ;;  # take 路径/父目录/视频/id
  esac
done
[ -n "$ROOT" ] || ROOT="${DATASET_ROOTS[$DATASET]:-}"
[ -n "$ROOT" ] || { echo "[reconstruct] 未知数据集 '$DATASET'，请用 --root 指定 dataset-root" >&2; exit 1; }

LIST="$RECON_INTERIM_ROOT/_lists/selected.txt"; mkdir -p "$(dirname "$LIST")"
if [[ "$DATASET" == "hoi4d" ]]; then
  # 选 take：给了路径就解析成 id 列表，否则前 N 条
  if [[ ${#TAKES[@]} -gt 0 ]]; then
    conda run -n base python "$HERE/bin/ids_from_paths.py" "${TAKES[@]}" | grep . > "$LIST" || true
    [[ -s "$LIST" ]] || { echo "[reconstruct] 路径没解析出任何 take: ${TAKES[*]}" >&2; exit 1; }
    echo "[reconstruct] hoi4d 指定 $(wc -l < "$LIST") 条 take"
  else
    conda run -n base python "$HERE/bin/first_n_list.py" "$N" | grep . > "$LIST"
    echo "[reconstruct] hoi4d 前 $N 条"
  fi
else
  # 通用数据集：TAKES 是 mp4 或目录 -> 把 mp4 绝对路径写进列表(通用 discover 按路径解析)
  [[ ${#TAKES[@]} -gt 0 ]] || { echo "[reconstruct] 数据集 $DATASET 需指定视频或目录" >&2; exit 1; }
  : > "$LIST"
  for t in "${TAKES[@]}"; do
    if [[ -d "$t" ]]; then find "$(realpath "$t")" -name '*.mp4' | sort >> "$LIST"
    elif [[ -f "$t" ]]; then realpath "$t" >> "$LIST"
    else echo "$t" >> "$LIST"; fi   # 也许是 generic id
  done
  [[ -s "$LIST" ]] || { echo "[reconstruct] 没找到视频: ${TAKES[*]}" >&2; exit 1; }
  echo "[reconstruct] $DATASET 指定 $(wc -l < "$LIST") 条视频 (root=$ROOT)"
fi

if [[ "$WEB" == 1 ]]; then
  echo "[reconstruct] 网页标注(GPU 预览) + 重建。打开 http://127.0.0.1:8765/  (远程: ssh -L 8765:127.0.0.1:8765 user@host)"
  exec conda run --no-capture-output -n base python "$RECON/run_batch_queue.py" \
    --dataset "$DATASET" --dataset-root "$ROOT" \
    --video-list "$LIST" --gpu-ids 0 --preview-device cuda --preview-gpu 0 \
    --http-host 0.0.0.0 --http-port 8765 ${PASS[@]+"${PASS[@]}"}
else
  echo "[reconstruct] 本地模式(--skip-label);未标注请先 ./label.sh 或加 --web"
  exec conda run --no-capture-output -n base python "$RECON/run_batch_queue.py" \
    --dataset "$DATASET" --dataset-root "$ROOT" \
    --video-list "$LIST" --skip-label --gpu-ids 0 ${PASS[@]+"${PASS[@]}"}
fi
