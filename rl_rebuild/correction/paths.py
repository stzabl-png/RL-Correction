"""外部依赖仓库的根路径 — 统一出口, 可用同名环境变量覆盖.

上游仓库改名历史: Bi-V2AP -> Reconstruct_and_Retarget (2026-07). 旧路径已不存在,
所有引用必须走这里, 不要在别处硬编码.

    export RR_ROOT=/path/to/Reconstruct_and_Retarget
    export AFFORDANCE_ROOT=/path/to/AffordanceModel

⚠️ 重要约束: **不要信任重建/retarget 产物内部烘着的绝对路径**
(`world_fused.npz` 的 video 字段、`world_summary.json`、`*_complete.json` 等仍写着
改名前的旧路径, 二进制 npz 无法批量替换). 定位源数据一律用显式传入的目录参数,
只从产物里读数值数组.
"""
from __future__ import annotations

import os

# 重建 + retarget 流水线仓 (原 Bi-V2AP)
RR_ROOT = os.environ.get("RR_ROOT", "/home/lyh/Project/Reconstruct_and_Retarget")
# affordance 预测模型输出仓
AFFORDANCE_ROOT = os.environ.get("AFFORDANCE_ROOT", "/home/lyh/Project/AffordanceModel")
# OCIR 抓姿合成仓 (设定 A 的 grasp_pose / curobo_traj 源)
OCIR_ROOT = os.environ.get("OCIR_ROOT", "/home/lyh/Project/ocir-grasp-synthesis")

RR_OUTPUT = os.path.join(RR_ROOT, "Output")                 # ReconstructOutput / RetargetOutput
RR_RETARGETING = os.path.join(RR_ROOT, "ego_pipeline", "Retargeting")
ISAAC_PYTHON = os.environ.get(
    "ISAAC_PYTHON", os.path.join(RR_ROOT, "third_party", "MagicDexMate",
                                 ".venv-isaac", "bin", "python"))
