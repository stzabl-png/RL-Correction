"""Read camera<->robot extrinsics written by ``RobotCamCalib/extr_calib_vega_kinect.py``.

Files live in ``configs/camera_extrinsics/kinect_<serial>.yaml`` (schema
``dexmate_teleop.camera_extrinsics.v1``). Pure numpy + yaml so it imports from any venv.

    from magicdexmate.camera_extrinsics import load, transform_points
    ext = load("000123456789")                     # serial, or a path
    xyz_base = transform_points(ext["T_base_depth"], xyz_depth_m)   # depth-frame cloud -> base frame
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "configs/camera_extrinsics"
SCHEMA = "dexmate_teleop.camera_extrinsics.v1"
_MATRICES = ("T_base_color", "T_color_depth", "T_base_depth", "T_tagmount_board", "K_color")


def path_for(serial: str) -> Path:
    return DIR / f"kinect_{serial}.yaml"


def available() -> Dict[str, Path]:
    """{serial: path} for every calibrated camera on disk."""
    out: Dict[str, Path] = {}
    if DIR.is_dir():
        for p in sorted(DIR.glob("kinect_*.yaml")):
            out[p.stem[len("kinect_"):]] = p
    return out


def load(serial_or_path: Union[str, Path]) -> dict:
    """Return the yaml as a dict with the 4x4 / 3x3 entries as float ndarrays.

    Raises FileNotFoundError / ValueError with a message that says what to run.
    """
    p = Path(serial_or_path)
    if not p.suffix and not p.exists():
        p = path_for(str(serial_or_path))
    if not p.is_file():
        raise FileNotFoundError(
            f"No camera extrinsics at {p}. Calibrate with "
            "`.venv/bin/python RobotCamCalib/extr_calib_vega_kinect.py --arm right --serial <serial>`."
        )
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{p} is not a {SCHEMA} file (schema={data.get('schema') if isinstance(data, dict) else None}).")
    for k in _MATRICES:
        if data.get(k) is not None:
            data[k] = np.asarray(data[k], dtype=float)
    if data.get("dist_color") is not None:
        data["dist_color"] = np.asarray(data["dist_color"], dtype=float).reshape(-1)
    T = data["T_base_color"]
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError(f"{p}: T_base_color is not a homogeneous 4x4 transform.")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
        raise ValueError(f"{p}: T_base_color rotation is not orthonormal / right-handed.")
    data["path"] = str(p)
    return data


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N,3) points."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    return xyz @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
