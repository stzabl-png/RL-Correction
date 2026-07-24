#!/bin/bash
cd /home/magics/mt_dir/Dynrotate/sharpa_rl_rebuild
PY=/home/magics/miniconda3/envs/sharpa/bin/python
export OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
rn () { echo "=== $3 ==="; $PY rl_rebuild/scripts/render_video.py --task "$1" --num_envs "$4" --video_length 320 \
        --load_path "$2" --device cuda:0 --out_dir "$3" --eye="$5" --lookat="$6" 2>&1 | tail -2; }
rn Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-v0 $(cat /tmp/gxreplay_run.txt)stage1_nn/best.pth videos_gx_single 4 "0.45,0.45,0.78" "-0.05,0.0,0.60"
rn Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Multi-v0  $(cat /tmp/gxmulti_run.txt)stage1_nn/best.pth  videos_gx_multi  4 "0.55,0.55,0.85" "-0.05,0.0,0.60"
rn Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Orient-v0 $(cat /tmp/gxorient_run.txt)stage1_nn/best.pth videos_gx_orient 4 "0.6,0.6,0.75" "0.0,0.0,0.55"
echo "ALL GX RENDERS DONE"; ls -la videos_gx_*/*.mp4
