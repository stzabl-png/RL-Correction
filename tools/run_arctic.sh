export PATH=$HOME/miniconda3/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh
cd $HOME/Reconstruct_and_Retarget/ego_pipeline
export AUTO_LABEL_INSTANCE=all
R=$HOME/Reconstruct_and_Retarget/Data/arctic_ego
./reconstruct.sh $R/s05__laptop_grab_01.mp4 --dataset arctic --root $R --gpu-ids=5
echo "EXIT=$?"
