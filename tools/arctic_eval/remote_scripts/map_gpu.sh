export PATH=$HOME/miniconda3/bin:$PATH; source $HOME/miniconda3/etc/profile.d/conda.sh
echo "nvidia-smi 各卡已用(GiB):"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '{printf "  smi%-2s %.2f\n",$1,$2/1024}'
echo "torch 在各 CUDA_VISIBLE_DEVICES 下看到的已用(GiB):"
for g in 0 1 2 3 4 5 6 7; do
  v=$(CUDA_VISIBLE_DEVICES=$g conda run --no-capture-output -n vllm python -c "
import torch
try:
    f,t=torch.cuda.mem_get_info(0); print(round((t-f)/2**30,2))
except Exception as e: print('FAIL')
" 2>/dev/null | tail -1)
  echo "  CVD=$g -> $v"
done
