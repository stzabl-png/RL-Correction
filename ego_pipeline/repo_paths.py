"""仓内/外部根路径的**统一出口** —— 不要在别处硬编码绝对路径。

设计:
- `RR_ROOT`(本仓根)**自动从本文件位置推导**,所以 clone 到任何机器/任何目录都能直接跑,
  无需设任何环境变量。
- 外部依赖(HumanVideo2RobotData / V2AP / hawor 环境)保留开发机的默认值,
  但都可用**同名环境变量覆盖**,别人 clone 后按自己机器 export 即可。

用法:
    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # -> ego_pipeline/
    from repo_paths import RR_ROOT, RECON_PIPELINE, RECON_OUTPUT

Shell 脚本用同目录的 `repo_paths.sh`(`source` 它即可)。

历史:本仓原名 Bi-V2AP,2026-07 改名为 Reconstruct_and_Retarget;旧绝对路径已全部失效。
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default) -> Path:
    """环境变量优先,否则用默认值。"""
    v = os.environ.get(name)
    return Path(v).expanduser().resolve() if v else Path(default)


# ---------- 本仓(自动推导,clone 即用) ----------
# repo_paths.py 位于 <RR_ROOT>/ego_pipeline/ 下
RR_ROOT: Path = _env_path("RR_ROOT", Path(__file__).resolve().parents[1])

EGO_PIPELINE: Path = RR_ROOT / "ego_pipeline"
DATA_ROOT: Path = _env_path("RR_DATA_ROOT", RR_ROOT / "Data")
OUTPUT_ROOT: Path = _env_path("RR_OUTPUT_ROOT", RR_ROOT / "Output")
RECON_OUTPUT: Path = OUTPUT_ROOT / "ReconstructOutput"
RETARGET_OUTPUT: Path = OUTPUT_ROOT / "RetargetOutput"
THIRD_PARTY: Path = RR_ROOT / "third_party"          # 注意:third_party 不入 git,需自行准备

# ---------- 外部依赖(可用同名环境变量覆盖) ----------
# Jiakai 的重建仓(recon_pipeline 的上游)
HV2RD_ROOT: Path = _env_path("HV2RD_ROOT", "/home/lyh/Project/HumanVideo2RobotData")
RECON_PIPELINE: Path = HV2RD_ROOT / "recon_pipeline"

# V2AP(FoundationPose / isaac_ros_ws / egodex 原始数据)
V2AP_ROOT: Path = _env_path("V2AP_ROOT", "/home/lyh/Project/V2AP")
EGODEX_ROOT: Path = _env_path("EGODEX_ROOT", V2AP_ROOT / "data/egocentric/egodex")
FOUNDATIONPOSE_ROOT: Path = _env_path("FOUNDATIONPOSE_ROOT", V2AP_ROOT / "thirdparty/foundationpose")
ISAAC_ROS_WS: Path = _env_path("ISAAC_ROS_WS", V2AP_ROOT / "thirdparty/isaac_ros_ws")

# 解释器
HAWOR_PYTHON: Path = _env_path("HAWOR_PYTHON", "/home/lyh/anaconda3/envs/hawor/bin/python")
ISAAC_PYTHON: Path = _env_path(
    "ISAAC_PYTHON", THIRD_PARTY / "MagicDexMate/.venv-isaac/bin/python")


def describe() -> str:
    """打印当前解析出的所有根路径(排障用):python -m repo_paths"""
    items = [(k, v) for k, v in sorted(globals().items())
             if isinstance(v, Path) and k.isupper()]
    w = max(len(k) for k, _ in items)
    return "\n".join(f"{k:<{w}}  {v}{'' if v.exists() else '   [不存在]'}" for k, v in items)


if __name__ == "__main__":
    print(describe())
