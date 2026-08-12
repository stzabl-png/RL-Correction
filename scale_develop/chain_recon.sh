#!/bin/bash
# 三条视频的重建链(adapter -> vipe -> sam3d -> sam3d_scale),幂等可重入。
# 先等 mask 链(chain_scale3.sh)退出再开跑,避免抢 GPU 1。
cd "$(dirname "$0")"
RECON=/home/bangdu/RL-Correction-recon/ego_pipeline/Reconstruction
CONDA=/home/bangdu/miniforge3/bin/conda
UV=/home/bangdu/.local/bin/uv
GPU=${GPU:-1}

while pgrep -f chain_scale3.sh > /dev/null; do sleep 60; done

for s in ketchup laptop microwave; do
  VID=s01_${s}_grab_01
  R=$PWD/runs/$VID
  MP4=$R/$VID.mp4
  MAN=$R/instance/video_mask_sequence/video_mask_sequence.json
  FIN=$R/final_selection.json
  OUT=$RECON/data/interim/arctic/$VID
  [ -f "$FIN" ] || { echo "== $s 无 final_selection,跳过 =="; continue; }
  SEL=$(/usr/bin/python3 -c "
import json
f = json.load(open('$FIN'))
valid = {o: e for o, e in f.items() if e.get('final_frame') is not None}
targets = {o: e for o, e in valid.items() if e.get('interaction_target')}
for oid, e in (targets or valid).items():
    print(oid, e['final_frame']); break
")
  OBJ=${SEL% *}; FRAME=${SEL#* }
  [ -n "$OBJ" ] || { echo "== $s 无有效 object,跳过 =="; continue; }
  echo "== $s obj=$OBJ frame=$FRAME 开始 $(date +%H:%M:%S) =="

  if [ ! -f "$OUT/sam2_object/label_prompt.json" ]; then
    (cd "$RECON" && $CONDA run --no-capture-output -n sam3 python \
      recon_kailang/v17_mask_adapter/import_v17a_masks.py \
      --dataset arctic --video-id "$VID" --video "$MP4" --manifest "$MAN" \
      --source-object-id "$OBJ" --reconstruction-frame "$FRAME" \
      --object-name "$s" --force) || { echo "== $s FAILED(adapter) =="; continue; }
    echo "== $s adapter ok =="
  fi
  if [ ! -f "$OUT/vipe/vipe_complete.json" ]; then
    (cd "$RECON/third_party/vipe" && $UV run python \
      "$RECON/recon_pipeline/vipe/run_sequence.py" \
      --dataset arctic --video-id "$VID" --video "$MP4" --gpu "$GPU") \
      || { echo "== $s FAILED(vipe) =="; continue; }
    echo "== $s vipe ok =="
  fi
  if [ ! -f "$OUT/sam3d/objects/object_0/object_mesh_raw.obj" ]; then
    (cd "$RECON" && $CONDA run --no-capture-output -n sam3d-objects python \
      recon_pipeline/sam3d/run_sequence.py \
      --dataset arctic --video-id "$VID" --video "$MP4" --gpu "$GPU") \
      || { echo "== $s FAILED(sam3d) =="; continue; }
    echo "== $s sam3d ok =="
  fi
  if [ ! -f "$OUT/sam3d_scale/objects/object_0/object_mesh_scaled_stage1.obj" ]; then
    (cd "$RECON" && $CONDA run --no-capture-output -n foundationpose python \
      recon_pipeline/sam3d_scale/run_sequence.py \
      --dataset arctic --video-id "$VID" --video "$MP4" --gpu "$GPU" --depth-scale 1.0) \
      || echo "== $s sam3d_scale 报错(stage1 mesh 若已导出仍可用) =="
    echo "== $s sam3d_scale done =="
  fi
  # ── 尺度:多帧跨度比 + 三路共识融合(修正 mesh 落在 runs/<id>/scale_fuse/) ──
  $CONDA run --no-capture-output -n foundationpose python scale_extent_v1.py \
    --run "$R" --dataset arctic --video-id "$VID" || echo "== $s FAILED(extent) =="
  $CONDA run --no-capture-output -n sam3 python scale_fuse_qwen.py \
    --run "$R" --dataset arctic --video-id "$VID" || echo "== $s FAILED(fuse) =="
  echo "== $s RECON DONE $(date +%H:%M:%S) =="
done
echo "== RECON CHAIN DONE =="
