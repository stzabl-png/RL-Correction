#!/usr/bin/env bash
# 命令：EgoDex 数据集 -> 重建 -> 与主 Pipeline **同格式**的 world_fused.npz(可直接进 retarget)
#
#   ./reconstruct_egodex.sh <task>/<n>            # 单条, 如 screw_unscrew_bottle_cap/0
#   ./reconstruct_egodex.sh <task>                # 该任务全部
#   ./reconstruct_egodex.sh <task>/<n> --variant prod
#
# ─────────────────────────────────────────────────────────────────────────────
# 与主 reconstruct.sh 的差别：EgoDex 自带三样我们本来要**估**的东西，直接注入
#
#   相机位姿  ViPE 估 → 设备 SLAM 给     (12 条实测: 与 ViPE 对齐残差中位 1mm)
#   相机内参  ViPE 估 → 设备标定给       (ViPE 740.18 vs 真值 736.63, 差 0.5%)
#   世界系    ViPE 估重力 + fuse 做 xy 对齐 → 设备已是重力对齐米制
#             ★ 我们的世界尺度实测跨 0.15–1.94, 这条可能比"省掉 HaWoR"更值钱
#   双手      HaWoR 估 → 设备手部追踪给  (--variant dev 时; prod 仍跑 HaWoR)
#
# 仍然要跑的：**深度**(EgoDex 没有, 靠 vipe) + 物体那条线(mask/建模/尺度/位姿, EgoDex 一样都没有)
#
# 变体：
#   dev (默认)  相机+内参+重力+手 全用 EgoDex。开发/评测。
#   prod        只用相机+内参+重力, 手仍跑 HaWoR。
#   两者之差 = HaWoR 的误差对下游的影响 —— 白捡的量化结果。
#
# ⚠ 顺序不能反：先 vipe(拿深度) → 再注入(覆盖 pose/intrinsics/gravity) → 再其余步骤。
#   vipe 会写自己的 pose/intrinsics, 注入必须在它之后。
#
# ⚠ 物体标注：EgoDex 只给物体的**文字名**(llm_objects), 没有 mask。
#   默认走 v17A 全自动标注; 已知它在多色物体上传播会整段失效(见 docs/INTEGRATION_STATUS.md A1),
#   排查用 tools/v17a_debug/overlay_video.py。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo_paths.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RECON_FINAL_ROOT="$RR_ROOT/Output/ReconstructOutput"
export RECON_INTERIM_ROOT="$RR_ROOT/Output/ReconstructOutput/interim"
export RECON_FINAL_NESTED=1
: "${SAM3_VERSION:=sam3}"; export SAM3_VERSION

: "${EGODEX_SRC:=/home/lyh/Project/V2AP/data/egocentric/egodex/test}"
DATASET=egodex_auto
VARIANT=dev
GPUS=0
TARGETS=(); PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --variant)   VARIANT="$2"; shift 2 ;;
    --variant=*) VARIANT="${1#*=}"; shift ;;
    --src)       EGODEX_SRC="$2"; shift 2 ;;
    --gpu-ids)   GPUS="$2"; shift 2 ;;
    --gpu-ids=*) GPUS="${1#*=}"; shift ;;
    -*)          PASS+=("$1"); shift ;;
    *)           TARGETS+=("$1"); shift ;;
  esac
done
[ ${#TARGETS[@]} -gt 0 ] || { grep '^#' "$0" | sed -n '2,12p'; exit 1; }

# ---- 展开目标: <task> 或 <task>/<n> ----
CLIPS=()
for t in "${TARGETS[@]}"; do
  if [ -f "$EGODEX_SRC/${t%/*}/${t##*/}.hdf5" ]; then
    CLIPS+=("$t")
  elif [ -d "$EGODEX_SRC/$t" ]; then
    while IFS= read -r h; do CLIPS+=("$t/$(basename "$h" .hdf5)"); done \
      < <(ls "$EGODEX_SRC/$t"/*.hdf5 2>/dev/null | sort -V)
  else
    echo "[egodex] X 找不到: $EGODEX_SRC/$t{,.hdf5}" >&2; exit 2
  fi
done
echo "[egodex] variant=$VARIANT  共 ${#CLIPS[@]} 条  src=$EGODEX_SRC"

WORK="$RR_ROOT/Data/$DATASET"
mkdir -p "$WORK"

for clip in "${CLIPS[@]}"; do
  TASK="${clip%/*}"; IDX="${clip##*/}"
  VID="${TASK}__${IDX}"
  H5="$EGODEX_SRC/$TASK/$IDX.hdf5"
  MP4_SRC="$EGODEX_SRC/$TASK/$IDX.mp4"
  [ -f "$H5" ] && [ -f "$MP4_SRC" ] || { echo "[egodex] 跳过 $clip (缺 hdf5 或 mp4)"; continue; }

  # 视频进 dataset root。⚠ 硬链不用软链: 批量会 resolve() 路径, 软链会让 video_id
  # 变回原名 -> 判"已完成"直接 0.03 秒退出且不报错(CLAUDE.md 记过这个坑)。
  mkdir -p "$WORK/$TASK"
  MP4="$WORK/$TASK/$IDX.mp4"
  [ -f "$MP4" ] || ln "$MP4_SRC" "$MP4" 2>/dev/null || cp "$MP4_SRC" "$MP4"

  echo "=========== $VID  $(date +%H:%M:%S) ==========="
  # 1) 先跑 vipe 拿深度(它也会写自己的 pose/intrinsics, 下一步覆盖掉)
  "$HERE/reconstruct.sh" "$MP4" --dataset "$DATASET" --root "$WORK" \
      --steps=vipe --keep-interim --no-auto-label --gpu-ids "$GPUS" "${PASS[@]+"${PASS[@]}"}"

  # 2) 注入 EgoDex 的相机/内参/重力(+dev 时的手)
  "$HAWOR_PYTHON" "$HERE/bin/egodex_to_pipeline.py" \
      --hdf5 "$H5" --dataset "$DATASET" --video-id "$VID" \
      --interim-root "$RECON_INTERIM_ROOT" --variant "$VARIANT"

  # 3) ★ 深度尺度：ViPE 深度在其自身尺度下，位姿已换成真实米制 —— 不校正会把物体
  #    放在错误距离上(实测 ViPE 比真值大 17~39%)。上一步已把 s 写进 egodex_source.json。
  DS=$("$HAWOR_PYTHON" -c "
import json,sys
try:
    d=json.load(open('$RECON_INTERIM_ROOT/$DATASET/$VID/egodex_source.json'))
    print(f\"{d.get('depth_scale') or 1.0:.6f}\")
except Exception: print('1.0')")
  # ★ 硬闸: depth_scale 离谱就**终止本条并报错**, 不许带着坏尺度往下跑。
  #   实测 clip 21/22 算出 208 / 156(正常 ~0.29~0.33) —— ViPE 深度整个塌了, 而管线
  #   照样跑完, 产出一条"看起来齐全"的坏数据。坏尺度会把物体放在错误距离上,
  #   下游的接触、可信度、RL 全部跟着错, 而且没有任何一步会报警。
  #   区间取 [0.01, 10]: 比实测正常值宽两个数量级, 只毙掉塌陷这种量级的错。
  if ! "$HAWOR_PYTHON" -c "
import sys
d=float('$DS')
if not (0.01 <= d <= 10.0):
    sys.stderr.write(f'[egodex] X depth_scale={d:g} 超出合理区间 [0.01, 10] —— 本条终止。\n'
                     f'    正常量级 0.29~0.33(EgoDex 实测); 出现 100+ 说明 ViPE 深度塌了。\n'
                     f'    排查: 看 {"$RECON_INTERIM_ROOT/$DATASET/$VID"}/egodex_source.json 与 vipe 输出。\n')
    sys.exit(1)
"; then
    echo "[egodex] X 跳过本条(depth_scale 异常), 继续下一条" >&2
    continue
  fi
  echo "[egodex] depth_scale=$DS -> 传给 sam3d_scale / fp_pose"

  # 4) 手 + 物体 mask。★ AUTO_LABEL_INSTANCE=all: 默认只注册"主实例", 而本任务是
  #    瓶身+瓶盖的配对场景, 两件都要 —— 实测 29 条里 8 条 v17A 找到了 2 个实例但只注册了 1 个,
  #    其中 6 条注册的还是**瓶盖**(按峰值面积排序, 瓶身多数帧被质量门拒掉反而排后)。
  if [ "$VARIANT" = "dev" ]; then PRE=sam3_hands,sam2_object,select_frame; else PRE=sam3_hands,sam2_object,select_frame,hawor; fi
  AUTO_LABEL_INSTANCE=all "$HERE/reconstruct.sh" "$MP4" --dataset "$DATASET" --root "$WORK" \
      --steps="$PRE" --keep-interim --gpu-ids "$GPUS" "${PASS[@]+"${PASS[@]}"}"

  # 5+6) VLM 门与资产检索**已成为正式管线步骤**(vlm_gate / retrieval), 不再在这里手工调 ——
  #      同一件事两个地方各跑一遍会各写各的完成标记, 且只有 EgoDex 这条路享受得到。
  #      现在它们排在 sam3d 之前(分件判定决定"重建网格还是取 CAD"), 所有数据集通用。
  #      retrieval 命中时会替 sam3d/sam3d_scale 写完成标记, 于是那两步自然跳过。

  # 7) VLM门 → 检索 → 位姿 → 融合 → 打分 → 接触。**contact 默认开**。
  # 历史: 这里曾默认关掉 contact —— 旧提取器一条 43.6 分钟, 占全程 97%(实测 clip1:
  #   其余步骤加起来才 90 秒), 因为它内部是**沿视线的多起点深度搜索**, 每个"物体×手"
  #   约 12 分钟。换成 extract_v2(不搜索, 逐帧量一次)后实测 **18.9 秒/条**, 这个默认
  #   就没有道理了 —— 而且接触点是 GraspPose 选模板要的 Video Prior, 默认缺席会让
  #   下游以为"这条没有接触", 而不是"这条没跑过接触"。
  #   要关: NO_CONTACT=1 ./reconstruct_egodex.sh ...
  STEPS="vlm_retrieval,retrieval,sam3d,sam3d_scale,fp_pose,fuse,confidence"
  [ -n "${NO_CONTACT:-}" ] || STEPS="$STEPS,contact"
  # ⚠ --step-arg 用等号写法: reconstruct.sh 的透传对空格写法只传 flag 不传值
  "$HERE/reconstruct.sh" "$MP4" --dataset "$DATASET" --root "$WORK" \
      --steps="$STEPS" --keep-interim --gpu-ids "$GPUS" \
      --step-arg="sam3d_scale:--depth-scale=$DS" \
      --step-arg="fp_pose:--depth-scale=$DS" "${PASS[@]+"${PASS[@]}"}"

  # 8) ★ 数据来源说明: 这条数据里哪些是重建出来的、哪些是数据集直接给的
  FIN="$RECON_FINAL_ROOT/$DATASET/$TASK/$IDX"
  OUT="$FIN/world_fused.npz"
  if [ -f "$OUT" ]; then
    "$HAWOR_PYTHON" "$HERE/bin/write_provenance.py" --final-dir "$FIN" \
        --dataset "$DATASET" --video-id "$VID" --variant "$VARIANT" \
        --interim-root "$RECON_INTERIM_ROOT" || true
    echo "[egodex] ✓ $VID -> $OUT"
  else
    echo "[egodex] X $VID 未产出 world_fused"
  fi
done
echo "[egodex] 全部结束 $(date +%H:%M:%S)"
