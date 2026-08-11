export PATH=$HOME/miniconda3/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh
cd $HOME/Reconstruct_and_Retarget/ego_pipeline
R=$HOME/Reconstruct_and_Retarget/Data/arctic15
V=""
for f in s05/laptop_grab_01 s07/ketchup_grab_01 s05/box_grab_01 s04/capsulemachine_grab_01 \
         s05/espressomachine_grab_01 s05/microwave_grab_01 s05/mixer_grab_01 \
         s06/notebook_grab_01 s10/phone_grab_01 s02/scissors_grab_01 s05/waffleiron_grab_01; do
  V="$V $R/$f.mp4"
done
# 标注已完成 -> 走 --skip-label(不再起标注服务)。去掉 contact 步:它 36% 的耗时里贵的是
# 接触点热度图对齐, 与 confidence 校准无关; 需要的 bridge+接触区间用 postproc 单独补(45秒/条)。
# --keep-interim: 保住 vipe depth 与 sam2 masks, 之后换参考帧只需重跑下游四步。
S=vipe,sam3_hands,sam2_object,hawor,sam3d,sam3d_scale,fp_pose,fuse,confidence
./reconstruct.sh $V --dataset arctic15 --root $R --gpu-ids 0 --steps=$S --keep-interim --no-auto-label
echo "RECON15_DONE $(date +%H:%M)"
