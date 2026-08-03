# RL_Correction 在 A6000 (128.32.164.89, 用户 yanghong) 的环境变量
# 用法: source ~/RL_Correction/env_a6000.sh
export RR_ROOT=$HOME/data/RR
export AFFORDANCE_ROOT=$HOME/data/AffordanceModel
export VEGA_URDF=$HOME/data/vega_urdf/vega_1p_sharpa.urdf
export MAGICSIM_ASSETS=$HOME/data/magicsim_assets      # 仅回退用; 仓库内已有躯干锁死 USD
export SHARPA_WANDB=0
export PYTHONPATH=$HOME/RL_Correction
export PY=$HOME/miniconda3/envs/isaac/bin/python
alias rlcd='cd $HOME/RL_Correction'
echo "[env] RL_Correction A6000 环境已加载 (RR_ROOT=$RR_ROOT)"
export OMNI_KIT_ACCEPT_EULA=YES   # 首次运行需接受 Omniverse EULA
