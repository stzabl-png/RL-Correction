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
# --web: 网页标注 + 标完接重建, 且跳过 v17A 自动标注
# cuda4 == 物理 GPU5(本机 CUDA 编号比 nvidia-smi 小 1)
./reconstruct.sh $V --dataset arctic15 --root $R --web --gpu-ids 4
