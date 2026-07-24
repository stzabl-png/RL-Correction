#!/bin/bash
# Render a close-up mp4 for each rebuild stage (sequential; single GPU).
cd /home/magics/mt_dir/Dynrotate/sharpa_rl_rebuild
PY=/home/magics/miniconda3/envs/sharpa/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0

render () {  # task ckpt outdir nenvs
  echo "=== rendering $3 ($1) ==="
  $PY rl_rebuild/scripts/render_video.py --task "$1" --num_envs "$4" --video_length 250 \
      --load_path "$2" --device cuda:0 --out_dir "$3" 2>&1 | tail -3
}

render Isaac-Inhand-Rotate-Sharpa-Wave-v0        results/run_logs/2026-06-27_13-31-07/stage1_nn/best.pth results/videos/stage1_baseline 4
render Isaac-Inhand-Rotate-Sharpa-Wave-PC-v0     results/run_logs/2026-06-27_13-57-28/stage1_nn/best.pth results/videos/stage2_pointcloud 4
render Isaac-Inhand-Rotate-Sharpa-Wave-WM-v0     results/run_logs/2026-06-27_14-25-35/stage1_nn/best.pth results/videos/stage3_worldmodel 4
render Isaac-Inhand-Rotate-Sharpa-Wave-WM-Multi-v0 results/run_logs/2026-06-27_14-54-36/stage1_nn/best.pth results/videos/stage4_multiscale 8
echo "ALL RENDERS DONE"
ls -la results/videos/*/*.mp4
