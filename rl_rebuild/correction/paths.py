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

# 仓库自带的**最小数据集** (datasets/, 走 LFS): 只含复现 Grasp3 那条训练所需的文件.
# 上游仓不在这台机器上时自动回落到它 —— 外部协作者 clone 完就能直接开训, 不用配环境变量.
# 要训别的 clip 仍需完整上游产物, 见 docs/DEPLOY_NEW_MACHINE.md §2B.
_BUNDLED = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))


def is_bundled(path: str) -> bool:
    """这个文件是不是仓库自带最小数据集里的?

    用途: `datasets/` 里的几份产物是**作为一组冻结快照一起提交**的, 彼此必然配套。
    而 git **不保留 mtime** —— clone 之后所有文件的 mtime 都是克隆时刻, 于是上游
    那些"按 mtime 判新鲜度"的检查 (ref_qpos 的 source_mtime 断言、stable_poses 的
    缓存 key) 在别人机器上必然误判。对自带快照跳过 mtime 判定, 其余路径照旧。
    """
    try:
        return os.path.commonpath([os.path.abspath(path), _BUNDLED]) == _BUNDLED
    except ValueError:          # 跨盘符 (Windows) 时 commonpath 会抛
        return False


def _root(env_var: str, default: str, bundled: str) -> str:
    """环境变量 > 本机默认路径(存在才用) > 仓库自带最小集 > 本机默认路径(保留原报错)."""
    p = os.environ.get(env_var)
    if p:
        return p
    if os.path.isdir(default):
        return default
    b = os.path.join(_BUNDLED, bundled)
    return b if os.path.isdir(b) else default


# 重建 + retarget 流水线仓 (原 Bi-V2AP)
RR_ROOT = _root("RR_ROOT", "/home/lyh/Project/Reconstruct_and_Retarget", "RR")
# affordance 预测模型输出仓
AFFORDANCE_ROOT = _root("AFFORDANCE_ROOT", "/home/lyh/Project/AffordanceModel",
                        "AffordanceModel")
# OCIR 抓姿合成仓 (设定 A 的 grasp_pose / curobo_traj 源)
OCIR_ROOT = os.environ.get("OCIR_ROOT", "/home/lyh/Project/ocir-grasp-synthesis")

RR_OUTPUT = os.path.join(RR_ROOT, "Output")                 # ReconstructOutput / RetargetOutput
RR_RETARGETING = os.path.join(RR_ROOT, "ego_pipeline", "Retargeting")
ISAAC_PYTHON = os.environ.get(
    "ISAAC_PYTHON", os.path.join(RR_ROOT, "third_party", "MagicDexMate",
                                 ".venv-isaac", "bin", "python"))
