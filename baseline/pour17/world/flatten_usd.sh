#!/usr/bin/env bash
# USD 自包含化 —— 依赖分析 + flatten。
#
# ★ 本机 pxr 的正确引出方式(2026-08-29 摸出来的, 别再以为"本地没有 USD 工具"):
#   USD 核心在 Isaac 的 extscache/omni.usd.libs 扩展里, 需要三样才 import 得动:
#     PYTHONPATH      -> 该扩展根目录(里面有 pxr/)
#     LD_LIBRARY_PATH -> 该扩展的 bin/ (libusd_*.so 在 bin 不在 lib) + libpython3.11 所在目录
#   不需要起 SimulationApp, 也不需要 OMNI_KIT_ACCEPT_EULA。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V=/home/lyh/Project/Reconstruct_and_Retarget/third_party/MagicDexMate/.venv-isaac
P="$V/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
LIBPY="$(dirname "$(find /home/lyh/anaconda3/pkgs -name libpython3.11.so.1.0 2>/dev/null | head -1)")"
export PYTHONPATH="$P"
export LD_LIBRARY_PATH="$P/bin:$LIBPY:${LD_LIBRARY_PATH:-}"
exec "$V/bin/python" "$HERE/flatten_usd.py" "$@"
