"""Loader for ``configs/handeye.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dexmate_calib.boards.config import project_root


def default_handeye_config_path() -> Path:
    return project_root() / "configs" / "handeye.yaml"


def load_handeye_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else default_handeye_config_path()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"{config_path}: expected schema_version 1 hand-eye config")
    for key in ("robot", "camera", "board", "capture", "solve"):
        if key not in data or not isinstance(data[key], dict):
            raise ValueError(f"{config_path}: missing section {key!r}")
    return data
