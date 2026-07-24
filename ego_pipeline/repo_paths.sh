#!/usr/bin/env bash
# 仓内/外部根路径的统一出口(shell 版,与 repo_paths.py 对应)。
#
#   source "$(dirname "${BASH_SOURCE[0]}")/repo_paths.sh"
#
# RR_ROOT 自动从本文件位置推导 -> clone 到任何位置都能跑,无需配置。
# 其余外部依赖保留开发机默认值,均可用同名环境变量覆盖。
#
# 历史:本仓原名 Bi-V2AP,2026-07 改名为 Reconstruct_and_Retarget;旧绝对路径已失效。

_RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 本仓(自动推导) ----------
: "${RR_ROOT:=$(cd "$_RP_DIR/.." && pwd)}"
: "${RR_DATA_ROOT:=$RR_ROOT/Data}"
: "${RR_OUTPUT_ROOT:=$RR_ROOT/Output}"
: "${RECON_OUTPUT:=$RR_OUTPUT_ROOT/ReconstructOutput}"
: "${RETARGET_OUTPUT:=$RR_OUTPUT_ROOT/RetargetOutput}"
: "${THIRD_PARTY:=$RR_ROOT/third_party}"   # 注意:third_party 不入 git,需自行准备

# ---------- 外部依赖(可用同名环境变量覆盖) ----------
: "${HV2RD_ROOT:=/home/lyh/Project/HumanVideo2RobotData}"
: "${RECON_PIPELINE:=$HV2RD_ROOT/recon_pipeline}"
: "${V2AP_ROOT:=/home/lyh/Project/V2AP}"
: "${EGODEX_ROOT:=$V2AP_ROOT/data/egocentric/egodex}"
: "${HAWOR_PYTHON:=/home/lyh/anaconda3/envs/hawor/bin/python}"
: "${ISAAC_PYTHON:=$THIRD_PARTY/MagicDexMate/.venv-isaac/bin/python}"

export RR_ROOT RR_DATA_ROOT RR_OUTPUT_ROOT RECON_OUTPUT RETARGET_OUTPUT THIRD_PARTY
export HV2RD_ROOT RECON_PIPELINE V2AP_ROOT EGODEX_ROOT HAWOR_PYTHON ISAAC_PYTHON
unset _RP_DIR
