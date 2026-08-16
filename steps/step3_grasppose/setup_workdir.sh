#!/usr/bin/env bash
# 新机器上初始化 Step3 的运行环境。只做两件事, 都不可省:
#
#   ① assets/object 软链 —— 11G 物体资产不入库(DGN_5k 库 + 运行时生成的
#      processed_data/scene_cfg)。hydra 的 asset_root 写死是相对 CWD 的 `assets`,
#      所以必须以软链的形式出现在仓内, 而不是靠环境变量绕开。
#   ② pip install -e —— dexrun 是 console script, 装了才有。
#      ★只设 PYTHONPATH 不够: take_grasp_pipeline.sh 调的是 `dexrun` 命令本身。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OBJ="$(python3 -c "import sys;sys.path.insert(0,'$HERE');import paths;print(paths.STEP3_OBJ_ROOT)")"
ENV="$(python3 -c "import sys;sys.path.insert(0,'$HERE');import paths;print(paths.DEXO_ENV)")"

mkdir -p "$OBJ"
ln -sfn "$OBJ" "$HERE/assets/object"
echo "① assets/object -> $OBJ"

if [ -x "$ENV/bin/pip" ]; then
  "$ENV/bin/pip" install -e "$HERE" --no-deps -q && echo "② dexrun 已装到 $ENV"
else
  echo "② 跳过: 找不到 $ENV/bin/pip —— 先建 conda 环境 dexonomy, 或设 DEXO_ENV"
fi
"$ENV/bin/python" "$HERE/paths.py" 2>/dev/null || python3 "$HERE/paths.py"
