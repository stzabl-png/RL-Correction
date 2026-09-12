#!/usr/bin/env python
"""Copy SharpaWave URDFs + meshes into this repo and fix mesh paths.

Source: ~/luhr/magicsim/Sharpa/sharpa-urdf-usd-xml/wave_01/{right,left}_sharpa_wave/
Dest:   assets/robots/hands/sharpa_wave/{right,left}/

The vendor URDFs reference meshes as `package://<side>_sharpa_wave/meshes/...`
(ROS package style). Pinocchio's kinematic model ignores meshes, but SAPIEN
visualization needs resolvable paths, so we rewrite them to plain relative
`meshes/...` next to the copied URDF.

Usage: .venv/bin/python scripts/prepare_assets.py [--src WAVE_01_DIR]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path("~/magicsim/Sharpa/sharpa-urdf-usd-xml/wave_01").expanduser()
DST = REPO_ROOT / "assets" / "robots" / "hands" / "sharpa_wave"


def prepare_side(src_root: Path, side: str) -> Path:
    src_dir = src_root / f"{side}_sharpa_wave"
    urdf_src = src_dir / f"{side}_sharpa_wave.urdf"
    meshes_src = src_dir / "meshes"
    if not urdf_src.exists():
        sys.exit(f"ERROR: missing {urdf_src}")

    dst_dir = DST / side
    dst_dir.mkdir(parents=True, exist_ok=True)

    text = urdf_src.read_text()
    text, n = re.subn(rf"package://{side}_sharpa_wave/meshes/", "meshes/", text)
    urdf_dst = dst_dir / urdf_src.name
    urdf_dst.write_text(text)

    if meshes_src.exists():
        shutil.copytree(meshes_src, dst_dir / "meshes", dirs_exist_ok=True)
        n_meshes = len(list((dst_dir / "meshes").glob("*")))
    else:
        n_meshes = 0

    n_joints = len(re.findall(r'<joint name="[^"]+" type="revolute"', text))
    print(f"[{side}] urdf -> {urdf_dst.relative_to(REPO_ROOT)}  "
          f"(rewrote {n} mesh refs, {n_meshes} mesh files, {n_joints} revolute joints)")
    if n_joints != 22:
        sys.exit(f"ERROR: expected 22 revolute joints in {urdf_src.name}, found {n_joints}")
    return urdf_dst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="wave_01 directory")
    args = parser.parse_args()

    # 生成好的资产随仓库分发。源目录只在最初制作资产的机器上存在;克隆下来
    # 的仓库没有它,但也不需要它 —— 资产已在。只有源在场时才重新生成。
    have = all((DST / s / f"{s}_sharpa_wave.urdf").exists() for s in ("right", "left"))
    if not args.src.exists():
        if have:
            print(f"[prepare_assets] 资产已在 {DST.relative_to(REPO_ROOT)},"
                  "源目录不在本机,跳过重新生成")
            return
        sys.exit(f"ERROR: 资产缺失且找不到源目录 {args.src}")

    for side in ("right", "left"):
        prepare_side(args.src, side)
    print(f"done. assets at {DST}")


if __name__ == "__main__":
    main()
