"""Versioned scene manifest for the two reconstructed pour objects."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SCENE_SCHEMA_VERSION = 1


@dataclass
class PourObjectSpec:
    label: str
    mesh: str
    usd: str
    mass_kg: float | None
    friction: float | None
    initial_pose_wxyz: list[float] | None
    opening_center_local: list[float] | None
    opening_axis_local: list[float] | None
    opening_radius_m: float | None

    def validate(self, *, require_assets: bool, require_geometry: bool) -> None:
        if not require_geometry:
            return
        pose = np.asarray(self.initial_pose_wxyz, dtype=np.float64)
        center = np.asarray(self.opening_center_local, dtype=np.float64)
        axis = np.asarray(self.opening_axis_local, dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"{self.label}: initial_pose_wxyz must contain 7 finite values")
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"{self.label}: opening_center_local must contain 3 finite values")
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 0.99:
            raise ValueError(f"{self.label}: opening_axis_local must be a unit 3-vector")
        if (
            self.mass_kg is None
            or self.mass_kg <= 0.0
            or self.friction is None
            or self.friction <= 0.0
            or self.opening_radius_m is None
            or self.opening_radius_m <= 0.0
        ):
            raise ValueError(f"{self.label}: physical scalars must be positive")
        if require_assets:
            for name, value in (("mesh", self.mesh), ("usd", self.usd)):
                if not value or not Path(value).is_file():
                    raise FileNotFoundError(f"{self.label}: {name} not found: {value}")


@dataclass
class PourSceneManifest:
    demo_id: str
    status: str
    reference_npz: str
    left_grasp_prior: str
    right_grasp_prior: str
    cup: PourObjectSpec
    bottle: PourObjectSpec
    # One rigid transform maps stationary ARKit-world poses into the
    # environment-local simulation frame: [tx, ty, tz, qw, qx, qy, qz].
    video_to_sim_wxyz: list[float] | None = None
    grasp_approval: dict | None = None
    schema_version: int = SCENE_SCHEMA_VERSION

    @property
    def training_ready(self) -> bool:
        return self.status == "ready"

    def validate(self, *, require_assets: bool = False) -> None:
        if self.schema_version != SCENE_SCHEMA_VERSION:
            raise ValueError(
                f"scene schema {self.schema_version} != supported {SCENE_SCHEMA_VERSION}"
            )
        if self.status not in {"pending_reconstruction", "pending_grasp_approval", "ready"}:
            raise ValueError(f"unknown scene status: {self.status}")
        require_geometry = self.status != "pending_reconstruction"
        self.cup.validate(require_assets=require_assets, require_geometry=require_geometry)
        self.bottle.validate(require_assets=require_assets, require_geometry=require_geometry)
        if require_geometry:
            transform = np.asarray(self.video_to_sim_wxyz, dtype=np.float64)
            if transform.shape != (7,) or not np.isfinite(transform).all():
                raise ValueError("video_to_sim_wxyz must contain 7 finite values")
            if not np.isclose(np.linalg.norm(transform[3:7]), 1.0, atol=1.0e-3):
                raise ValueError("video_to_sim_wxyz quaternion must be normalized")
        if require_assets:
            if not isinstance(self.grasp_approval, dict):
                raise RuntimeError("ready scene is missing the manual grasp approval record")
            for side in ("left", "right"):
                if side not in self.grasp_approval:
                    raise RuntimeError(f"manual grasp approval is missing {side}")
            for label, value in (
                ("reference_npz", self.reference_npz),
                ("left_grasp_prior", self.left_grasp_prior),
                ("right_grasp_prior", self.right_grasp_prior),
            ):
                if not value or not Path(value).is_file():
                    raise FileNotFoundError(f"{label} not found: {value}")
            if not self.training_ready:
                raise RuntimeError(f"scene {self.demo_id} is not approved for training")

    def save(self, path: str | Path) -> None:
        self.validate(require_assets=False)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, resolve_relative: bool = True) -> "PourSceneManifest":
        source = Path(path)
        value = json.loads(source.read_text(encoding="utf-8"))
        value["cup"] = PourObjectSpec(**value["cup"])
        value["bottle"] = PourObjectSpec(**value["bottle"])
        manifest = cls(**value)
        if resolve_relative:
            base = source.parent
            for name in ("reference_npz", "left_grasp_prior", "right_grasp_prior"):
                raw = getattr(manifest, name)
                if raw and not Path(raw).is_absolute():
                    setattr(manifest, name, str((base / raw).resolve()))
            for item in (manifest.cup, manifest.bottle):
                for name in ("mesh", "usd"):
                    raw = getattr(item, name)
                    if raw and not Path(raw).is_absolute():
                        setattr(item, name, str((base / raw).resolve()))
        manifest.validate(require_assets=False)
        return manifest
