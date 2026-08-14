#!/usr/bin/env bash
# 只跑前两步并出验收报告: ① HOI 手/物检测  ② VLM 门(材质 + 分件)
#
#   ./ego_pipeline/verify_step12.sh <video.mp4|目录> [更多视频...] \
#        --dataset <名字> --root <视频根目录> [--gpu-ids 0] [--force]
#
# 干什么:
#   1. HOI-DETR 检测 + **评审视频**(带手/物框和 HF/FS 连线) —— 肉眼确认第一步没错
#   2. 实例发现 + SAM2 传播(VLM 门要按实例采 mask, 只有第 1 步的框还不够)
#   3. VLM 门: 逐实例判材质(透明?) + 整片判分件(需要 Retrieval?)
#   4. 打一张对照表; **不碰** sam3d / 位姿 / 融合 / 重建
#
# 为什么值得单独有这么个脚本:
#   底层错了上面全错。今天就吃过一次 —— 手的位置偏了 400 像素, 而所有下游结论都照常产出,
#   看起来一切正常。前两步是整条链的地基, 应该能独立看、独立验。
#
# VLM 服务在 UCB 8 卡机 GPU7, 本脚本会自动开隧道(见 tools/vlm_tunnel.sh)。
# 服务不可达时**不中断**, 但报告里该条会明确标 "未判定", 不会伪装成"判过了不透明"。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RR="$(cd "$HERE/.." && pwd)"
source "$HERE/repo_paths.sh" 2>/dev/null || true
PY="${HAWOR_PYTHON:-/home/lyh/anaconda3/envs/hawor/bin/python}"

DATASET=""; ROOT=""; GPUS=0; FORCE=0; VIDEOS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --root)    ROOT="$2";    shift 2 ;;
    --gpu-ids) GPUS="$2";    shift 2 ;;
    --force)   FORCE=1;      shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *)         VIDEOS+=("$1"); shift ;;
  esac
done
[ -n "$DATASET" ] || { echo "[verify12] 需要 --dataset" >&2; exit 1; }
[ -n "$ROOT" ]    || { echo "[verify12] 需要 --root(视频根目录)" >&2; exit 1; }
[ ${#VIDEOS[@]} -gt 0 ] || { echo "[verify12] 需要至少一个视频/目录" >&2; exit 1; }

# 展开目录 -> mp4 列表
LIST=()
for v in "${VIDEOS[@]}"; do
  if [ -d "$v" ]; then while IFS= read -r f; do LIST+=("$f"); done < <(find "$v" -name '*.mp4' | sort)
  else LIST+=("$v"); fi
done
echo "[verify12] ${#LIST[@]} 条视频  dataset=$DATASET  gpu=$GPUS"

# ---- VLM 隧道(不通也继续, 只是第 2 步会记"未判定") ----
VLM_OK=1
"$RR/tools/vlm_tunnel.sh" >/dev/null 2>&1 || VLM_OK=0
[ "$VLM_OK" = 1 ] && echo "[verify12] VLM 服务: 可用" \
                  || echo "[verify12] ⚠ VLM 服务不可用 —— 第 2 步会记为未判定(不中断)"

FF=(); [ "$FORCE" = 0 ] || FF=(--force)
for MP4 in "${LIST[@]}"; do
  [ -f "$MP4" ] || { echo "[verify12] ! 找不到 $MP4"; continue; }
  VID="$("$PY" -c "
import sys;sys.path.insert(0,'$RR/ego_pipeline/bin')
from auto_label_v17a import video_id_for
from pathlib import Path
print(video_id_for(Path('$MP4').resolve(), Path('$ROOT')))")"
  echo "=============================================================="
  echo "[verify12] $VID"

  # ① HOI 检测 + 评审视频。--hoi-only 不查 SAM2 权重, 也不会因"重建已完成"而跳过。
  AUTO_LABEL_FORCE=$FORCE conda run --no-capture-output -n hawor python \
    "$HERE/bin/auto_label_v17a.py" --dataset "$DATASET" --dataset-root "$ROOT" \
    --video "$MP4" --gpu "${GPUS%%,*}" --hoi-only --visualize-hoi || \
      { echo "[verify12] ! ① HOI 失败, 跳过本条"; continue; }

  # ② 实例发现 + SAM2 传播 (VLM 门要逐实例 mask; 第①步只有框)
  AUTO_LABEL_INSTANCE=all conda run --no-capture-output -n hawor python \
    "$HERE/bin/auto_label_v17a.py" --dataset "$DATASET" --dataset-root "$ROOT" \
    --video "$MP4" --gpu "${GPUS%%,*}"
  rc=$?
  [ $rc -eq 3 ] && echo "[verify12]   (透明门判本条全部实例为空透明 -> 重建时会跳过)"

  # ③ VLM 门: 材质 + 分件
  "$PY" "$HERE/bin/vlm_gate_step.py" --dataset "$DATASET" --video-id "$VID" \
      --video "$MP4" "${FF[@]}" || echo "[verify12] ! ③ VLM 门失败(不中断)"
done

echo "=============================================================="
"$PY" "$HERE/bin/verify_step12_report.py" --dataset "$DATASET" --root "$ROOT" "${LIST[@]}"
