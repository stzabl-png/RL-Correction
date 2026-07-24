#!/bin/bash
cd /home/magics/mt_dir/Dynrotate/sharpa_rl_rebuild
PY=/home/magics/miniconda3/envs/sharpa/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
# 1) GOOD rotation reference: single-object Replay policy (palm-up), yaw ~0.79
$PY rl_rebuild/scripts/render_video.py --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-v0 \
  --num_envs 4 --video_length 320 --load_path logs/debug/2026-06-28_02-02-07/stage1_nn/last.pth \
  --device cuda:0 --out_dir videos_orient_single --eye=0.45,0.45,0.78 --lookat=-0.05,0.0,0.60 2>&1 | tail -2
# 2) OrientCurr policy at FULL SO(3) (the shaking the user caught; now correctly forced to full cap)
SHARPA_ORIENT_CAP=3.14159 $PY rl_rebuild/scripts/render_video.py --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-OrientCurr-v0 \
  --num_envs 4 --video_length 320 --load_path logs/debug/2026-06-28_18-39-23/stage1_nn/best.pth \
  --device cuda:0 --out_dir videos_orient_shaking --eye=0.65,0.65,0.78 --lookat=0.0,0.0,0.55 2>&1 | tail -2
echo "RENDERS DONE"; ls -la videos_orient_*/*.mp4
