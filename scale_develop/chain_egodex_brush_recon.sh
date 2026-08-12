#!/bin/bash
# EgoDex 扫把(sweep_dustpan_1)重建+尺度链:adapter -> vipe -> sam3d -> sam3d_scale -> 尺度护栏。
# 幂等可重入。前置:chain_egodex_brush.sh 已产出 final_selection.json(f295, object_0001=白色扫帚)。
cd "$(dirname "$0")"
VIDEO_ID=sweep_dustpan_1
DATASET=egodex
RECON=/home/bangdu/RL-Correction-recon/ego_pipeline/Reconstruction
CONDA=/home/bangdu/miniforge3/bin/conda
UV=/home/bangdu/.local/bin/uv
PY_SAM3=/home/bangdu/miniforge3/envs/sam3/bin/python
GPU=${GPU:-0}

R=$PWD/runs/$VIDEO_ID
MP4=$R/$VIDEO_ID.mp4
MAN=$R/instance/video_mask_sequence/video_mask_sequence.json
FIN=$R/final_selection.json
OUT=$RECON/data/interim/$DATASET/$VIDEO_ID

SEL=$(/usr/bin/python3 -c "
import json
f = json.load(open('$FIN'))
for oid, e in f.items():
    if e.get('final_frame') is not None:
        print(oid, e['final_frame']); break
")
OBJ=${SEL% *}; FRAME=${SEL#* }
[ -n "$OBJ" ] || { echo "== 无有效 object,退出 =="; exit 1; }
echo "== brush obj=$OBJ frame=$FRAME 开始 $(date +%H:%M:%S) =="

if [ ! -f "$OUT/sam2_object/label_prompt.json" ]; then
  (cd "$RECON" && $CONDA run --no-capture-output -n sam3 python \
    recon_kailang/v17_mask_adapter/import_v17a_masks.py \
    --dataset $DATASET --video-id "$VIDEO_ID" --video "$MP4" --manifest "$MAN" \
    --source-object-id "$OBJ" --reconstruction-frame "$FRAME" \
    --object-name brush --force) || { echo "== FAILED(adapter) =="; exit 1; }
  echo "== adapter ok =="
fi
if [ ! -f "$OUT/vipe/vipe_complete.json" ]; then
  (cd "$RECON/third_party/vipe" && $UV run python \
    "$RECON/recon_pipeline/vipe/run_sequence.py" \
    --dataset $DATASET --video-id "$VIDEO_ID" --video "$MP4" --gpu "$GPU") \
    || { echo "== FAILED(vipe) =="; exit 1; }
  echo "== vipe ok =="
fi
if [ ! -f "$OUT/sam3d/objects/object_0/object_mesh_raw.obj" ]; then
  (cd "$RECON" && $CONDA run --no-capture-output -n sam3d-objects python \
    recon_pipeline/sam3d/run_sequence.py \
    --dataset $DATASET --video-id "$VIDEO_ID" --video "$MP4" --gpu "$GPU") \
    || { echo "== FAILED(sam3d) =="; exit 1; }
  echo "== sam3d ok =="
fi
if [ ! -f "$OUT/sam3d_scale/objects/object_0/object_mesh_scaled_stage1.obj" ]; then
  (cd "$RECON" && $CONDA run --no-capture-output -n foundationpose python \
    recon_pipeline/sam3d_scale/run_sequence.py \
    --dataset $DATASET --video-id "$VIDEO_ID" --video "$MP4" --gpu "$GPU" --depth-scale 1.0) \
    || echo "== sam3d_scale 报错(stage1 mesh 若已导出仍可用) =="
  echo "== sam3d_scale done =="
fi

# ── 尺度护栏 ──
OBJDIR=$OUT/sam3d_scale/objects/object_0
MESH=$OBJDIR/object_mesh_scaled_final.obj
[ -f "$MESH" ] || MESH=$OBJDIR/object_mesh_scaled_stage1.obj
[ -f "$MESH" ] || { echo "== FAILED: 无 scaled mesh =="; exit 1; }
META=$(ls "$OBJDIR"/*metadata*.json 2>/dev/null | head -1)
$PY_SAM3 scale_sanity_qwen.py --run "$R" --scaled-mesh "$MESH" \
  ${META:+--scale-metadata "$META"} || { echo "== FAILED(guardrail) =="; exit 1; }
echo "== BRUSH RECON CHAIN DONE $(date +%H:%M:%S) =="
