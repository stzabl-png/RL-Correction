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
# 标注：**默认全自动**(v17A 自动出物体 mask, 无需人工; --no-auto-label 关闭)。
#      人工点选只是 fallback：--web 网页标注一条龙, 或先 ./label.sh 标好。
# 其它 flag 透传 run_batch_queue（--force / --dry-run / --workers-per-gpu 2 ...）。
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo_paths.sh"
BIV2AP="$RR_ROOT"
RECON="$RECON_PIPELINE"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RECON_FINAL_ROOT="$BIV2AP/Output/ReconstructOutput"
export RECON_INTERIM_ROOT="$BIV2AP/Output/ReconstructOutput/interim"
export RECON_FINAL_NESTED=1   # final 目录镜像原数据集嵌套: A__B__C -> A/B/C
# SAM3 版本: 默认 sam3(2026-08-10 固化)。sam3.1 在 4080S/Ada 检不出手(代码内注释),
# 在 A6000 是 gated 模型会 401, Blackwell 本来就要降级 —— 三种卡全都该用 sam3。
: "${SAM3_VERSION:=sam3}"; export SAM3_VERSION

# 内置数据集 root(其它数据集用 --root 指定)
declare -A DATASET_ROOTS=(
  [hoi4d]="$BIV2AP/Data/HOI4D"
  [egodex]="$EGODEX_ROOT"
)

DATASET=hoi4d; ROOT=""; N=10; WEB=0; AUTOLABEL=1; TAKES=(); PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --root)    ROOT="$2"; shift 2 ;;
    --web)     WEB=1; shift ;;
    --no-auto-label) AUTOLABEL=0; shift ;;
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
  # v17A 自动标注(2026-08-10 接入): 清单里没有标注缓存的视频, 先自动出物体 mask
  # (HOI-DETR 检测 + SAM2 实例传播 + 开朗 adapter 落格式, 幂等)。
  # --no-auto-label 关闭; hoi4d 的清单是 take id 不是视频路径, 不走这条。
  if [[ "$AUTOLABEL" == 1 && "$DATASET" != "hoi4d" ]]; then
    echo "[reconstruct] v17A 自动标注 + VLM 透明门(--no-auto-label 跳过; 门 AUTO_LABEL_VLM_GATE=0 单独关)"
    KEEP="$LIST.keep"; : > "$KEEP"
    while IFS= read -r vp; do
      [[ -f "$vp" ]] || { echo "$vp" >> "$KEEP"; continue; }
      rc=0
      conda run --no-capture-output -n hawor python "$HERE/bin/auto_label_v17a.py" \
        --dataset "$DATASET" --dataset-root "$ROOT" --video "$vp" || rc=$?   # set -e 下必须 ||捕获
      if [[ $rc -eq 3 ]]; then
        echo "[reconstruct] 跳过本视频(全部实例空透明, 规则 v2): $vp"
        continue          # 不进 KEEP -> 不进重建队列
      elif [[ $rc -ne 0 ]]; then
        echo "[reconstruct] 自动标注失败: $vp — 兜底: ./reconstruct.sh <视频> --dataset $DATASET --web 人工标注" >&2
        exit 1
      fi
      echo "$vp" >> "$KEEP"
    done < "$LIST"
    mv "$KEEP" "$LIST"
    [[ -s "$LIST" ]] || { echo "[reconstruct] 清单里所有视频都被透明门过滤, 无事可做"; exit 0; }
  fi
  echo "[reconstruct] 本地模式(--skip-label);自动标注已就位或请先 ./label.sh / --web"
  exec conda run --no-capture-output -n base python "$RECON/run_batch_queue.py" \
    --dataset "$DATASET" --dataset-root "$ROOT" \
    --video-list "$LIST" --skip-label --gpu-ids 0 ${PASS[@]+"${PASS[@]}"}
fi
