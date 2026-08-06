"""Asset lookup helpers for grasp synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = REPO_ROOT / "assets"
DEFAULT_SHARPA_WAVE_RIGHT_CONFIG = ASSET_ROOT / "robots/hands/sharpa_wave/sharpa_wave_right.yml"


@dataclass(frozen=True)
class SharpaWaveAsset:
    root: Path
    config_path: Path
    config: dict

    @property
    def urdf_path(self) -> Path:
        return self.root / self.config["urdf_path"]

    @property
    def contact_config_path(self) -> Path:
        return self.root / self.config["contact_points"]["config_path"]

    @property
    def collision_spheres_path(self) -> Path:
        return self.root / self.config["collision_spheres"]["path"]

    @property
    def usd_path(self) -> Path:
        return self.root / self.config["usd_path"]

    def bodex_path(self, key: str) -> Path:
        return self.root / self.config["bodex"][key]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def load_sharpa_wave_right(config_path: str | Path | None = None) -> SharpaWaveAsset:
    path = repo_path(config_path) if config_path else DEFAULT_SHARPA_WAVE_RIGHT_CONFIG
    config = load_yaml(path)
    asset = SharpaWaveAsset(root=path.parent, config_path=path, config=config)
    missing = [
        p
        for p in (
            asset.urdf_path,
            asset.usd_path,
            asset.contact_config_path,
            asset.collision_spheres_path,
            asset.bodex_path("robot_config"),
            asset.bodex_path("collision_spheres"),
            asset.bodex_path("hand_pose_transfer"),
            asset.bodex_path("grasp_synthesis_config"),
        )
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("missing Sharpa asset files: " + ", ".join(str(p) for p in missing))
    return asset
