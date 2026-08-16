"""Step3 的外部路径出口 —— 重资产不入库, 一律从这里解析。

沿用 `rl_rebuild/correction/paths.py` 的写法: 环境变量 > 本机默认(存在才用) > 兜底。

    export RR_ROOT=/path/to/Reconstruct_and_Retarget     # Step2 重建产物所在仓
    export STEP3_OBJ_ROOT=/path/to/object_assets         # 物体资产工作区(11G, 见下)
    export DEXO_ENV=/path/to/conda/envs/dexonomy         # 带 dexrun 的解释器环境

⚠ **不要信任重建产物内部烘着的绝对路径。** Step2 交接件里的 `mesh` /
`contact_cloud_npz` / `poseqa_root` 写的是产出那台机器的路径(实测 pour/17 里是
`/home/yanghong/...`), 换机器必然失效。定位文件一律用显式传入的 take 目录重建,
只从产物里读**数值**。这条与 `rl_rebuild/correction/paths.py` 顶部同源。

物体资产为什么不入库: `assets/object/` 有 11G —— DGN_5k 上游物体库 6.2G、
运行时由 `tools/import_object.py` 生成的 `custom/processed_data` 2.6G、recon 1.8G。
仓内 `assets/object` 是一个**软链**(已 gitignore), 由 `setup_workdir.sh` 建。
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _root(env_var: str, *candidates: str) -> str:
    """环境变量 > 第一个真实存在的候选 > 最后一个候选(保留原报错)。"""
    p = os.environ.get(env_var)
    if p:
        return p
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[-1]


# Step2 重建产物仓 (take 目录在 <RR_ROOT>/Output/ReconstructOutput/<dataset>/<task>/<take>)
RR_ROOT = _root("RR_ROOT", "/home/lyh/Project/Reconstruct_and_Retarget")
RR_OUTPUT = os.path.join(RR_ROOT, "Output", "ReconstructOutput")

# 物体资产工作区。默认优先复用旧 Dexonomy 检出里已生成的 2.6G, 免得重跑 import_object;
# 那份退休后回落到仓外的中立工作目录。
STEP3_OBJ_ROOT = _root(
    "STEP3_OBJ_ROOT",
    "/home/lyh/Project/Dexonomy/assets/object",
    os.path.join(os.path.dirname(REPO), "_step3_work", "object"),
)

# 带 dexrun 的 conda 环境 (steps/README.md 的约定: 8 个环境不因合仓而合并)
DEXO_ENV = _root("DEXO_ENV",
                 os.path.expanduser("~/anaconda3/envs/dexonomy"),
                 os.path.expanduser("~/miniconda3/envs/dexonomy"))
DEXO_PYTHON = os.path.join(DEXO_ENV, "bin", "python")

# 仓内固定位置
HAND_DIR = os.path.join(HERE, "assets", "hand")
TEMPLATE_PRIOR = os.path.join(HERE, "assets", "template_prior.json")
TOOLS = os.path.join(HERE, "tools")


def take_dir(dataset: str, task: str, take: str) -> str:
    return os.path.join(RR_OUTPUT, dataset, task, str(take))


if __name__ == "__main__":
    for k in ("RR_ROOT", "RR_OUTPUT", "STEP3_OBJ_ROOT", "DEXO_ENV", "HAND_DIR"):
        v = globals()[k]
        print(f"{k:<16} {v}   {'✓' if os.path.exists(v) else '✗ 不存在'}")
