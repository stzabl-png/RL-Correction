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
: "${RECON_PIPELINE:=$RR_ROOT/ego_pipeline/Reconstruction/recon_pipeline}"
: "${V2AP_ROOT:=/home/lyh/Project/V2AP}"
: "${EGODEX_ROOT:=$V2AP_ROOT/data/egocentric/egodex}"
# hawor 解释器: **自动探测**, 不写死。本地是 anaconda3, 远端 UCB 是 miniconda3,
# 写死本地路径会让远端在 vipe 之后立刻 "No such file or directory" 退出 ——
# 而 vipe 本身是成功的, 所以看起来像"每条都跑了 2 分钟然后失败"(2026-08-12 空烧 10 条)。
if [ -z "${HAWOR_PYTHON:-}" ]; then
  for _c in "$HOME/anaconda3/envs/hawor/bin/python" "$HOME/miniconda3/envs/hawor/bin/python" \
            /home/lyh/anaconda3/envs/hawor/bin/python; do
    [ -x "$_c" ] && { HAWOR_PYTHON="$_c"; break; }
  done
  : "${HAWOR_PYTHON:=python3}"
  unset _c
fi
: "${ISAAC_PYTHON:=$THIRD_PARTY/MagicDexMate/.venv-isaac/bin/python}"

: "${A2G_ROOT:=/home/lyh/Project/Affordance2Grasp}"
: "${EGODEX_RAW_ROOT:=$A2G_ROOT/data_hub/RawData/EgoRawData/egodex/test}"
: "${HAWOR_DATA:=$A2G_ROOT/third_party/hawor/_DATA}"
: "${VIPE_ROOT:=$THIRD_PARTY/vipe}"
: "${HOI4D_RELEASE:=$V2AP_ROOT/data/egocentric/hoi4d/HOI4D_release}"
export A2G_ROOT EGODEX_RAW_ROOT HAWOR_DATA VIPE_ROOT HOI4D_RELEASE

export RR_ROOT RR_DATA_ROOT RR_OUTPUT_ROOT RECON_OUTPUT RETARGET_OUTPUT THIRD_PARTY
export RECON_PIPELINE V2AP_ROOT EGODEX_ROOT HAWOR_PYTHON ISAAC_PYTHON
unset _RP_DIR
