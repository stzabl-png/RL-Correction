"""PCO-1810 cap assembly and PhysX helical-joint authoring.

The environment applies a GPU-batched pose/velocity constraint only while the
threads are engaged.  This supports both a pre-assembled unscrew task and a
free-cap align/capture/screw-on task without runtime topology changes.  It
avoids both unstable triangle-on-triangle thread contact and the CPU-only
PhysX rack-and-pinion constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from rl_rebuild.correction import frames as F


@dataclass(frozen=True)
class ScrewSpec:
    """Runtime parameters for the right-handed PCO-1810 closure."""

    pitch_m: float = 0.00318
    turns: float = 2.0
    closed_offset_m: float = 0.180
    direction: int = 1
    mode: str = "preengaged"
    capture_radial_m: float = 0.003
    capture_axial_m: float = 0.003
    capture_tilt_deg: float = 10.0
    capture_yaw_deg: float = 30.0
    max_angular_velocity_rad_s: float = 20.0

    def __post_init__(self):
        if self.pitch_m <= 0.0:
            raise ValueError("pitch_m must be positive")
        if self.turns <= 0.0:
            raise ValueError("turns must be positive")
        if self.closed_offset_m <= 0.0:
            raise ValueError("closed_offset_m must be positive")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if self.mode not in ("preengaged", "capture"):
            raise ValueError("mode must be 'preengaged' or 'capture'")
        if min(self.capture_radial_m, self.capture_axial_m) <= 0.0:
            raise ValueError("capture position tolerances must be positive")
        if not 0.0 < self.capture_tilt_deg < 90.0:
            raise ValueError("capture_tilt_deg must be between 0 and 90")
        if not 0.0 < self.capture_yaw_deg <= 180.0:
            raise ValueError("capture_yaw_deg must be between 0 and 180")
        if self.max_angular_velocity_rad_s <= 0.0:
            raise ValueError("max_angular_velocity_rad_s must be positive")

    @property
    def travel_m(self) -> float:
        return self.pitch_m * self.turns

    @property
    def angle_limit_deg(self) -> float:
        return 360.0 * self.turns

    @property
    def ratio_deg_per_m(self) -> float:
        """PhysX rack-and-pinion ratio (angular motion / linear motion)."""

        return self.direction * 360.0 / self.pitch_m

    @classmethod
    def from_mapping(cls, value: Mapping | None) -> "ScrewSpec | None":
        if value is None:
            return None
        return cls(
            pitch_m=float(value["pitch_m"]),
            turns=float(value["turns"]),
            closed_offset_m=float(value["closed_offset_m"]),
            direction=int(value.get("direction", 1)),
            mode=str(value.get("mode", "preengaged")),
            capture_radial_m=float(value.get("capture_radial_m", 0.003)),
            capture_axial_m=float(value.get("capture_axial_m", 0.003)),
            capture_tilt_deg=float(value.get("capture_tilt_deg", 10.0)),
            capture_yaw_deg=float(value.get("capture_yaw_deg", 30.0)),
            max_angular_velocity_rad_s=float(
                value.get("max_angular_velocity_rad_s", 20.0)
            ),
        )


def helical_travel(angle_rad: float | np.ndarray, spec: ScrewSpec):
    """Return signed axial travel for a relative cap rotation."""

    return spec.direction * np.asarray(angle_rad) * spec.pitch_m / (2.0 * np.pi)


def assembled_cap_pose(body_pose_wxyz: np.ndarray, spec: ScrewSpec) -> np.ndarray:
    """Place the closed cap on the bottle axis while preserving body pose."""

    pose = np.asarray(body_pose_wxyz, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"body pose must have shape (7,), got {pose.shape}")
    quat = pose[3:7]
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("body quaternion must be finite and non-zero")
    quat = quat / norm
    offset = F.rot_apply(quat, np.array([0.0, 0.0, spec.closed_offset_m]))
    return np.concatenate([pose[:3] + offset, quat])


def author_screw_metadata(stage, env_path: str, spec: ScrewSpec) -> dict[str, str]:
    """Record one runtime helical mechanism and its collision exclusion.

    The hard relation is enforced by GPU tensors in the environment, not by a
    USD joint.  Detailed bottle/cap collision is disabled only between those
    two assets; contacts with the robot and table remain enabled.
    """

    from pxr import Sdf, UsdGeom, UsdPhysics

    body_path = f"{env_path}/Object"
    cap_path = f"{env_path}/Cap"
    metadata_path = f"{env_path}/BottleScrew"

    for path in (body_path, cap_path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"screw mechanism requires rigid body {path}")

    prim = UsdGeom.Scope.Define(stage, metadata_path).GetPrim()
    prim.CreateAttribute("bottleScrew:pitchM", Sdf.ValueTypeNames.Double).Set(
        spec.pitch_m
    )
    prim.CreateAttribute("bottleScrew:turns", Sdf.ValueTypeNames.Double).Set(
        spec.turns
    )
    prim.CreateAttribute("bottleScrew:mode", Sdf.ValueTypeNames.String).Set(
        spec.mode
    )
    prim.CreateAttribute(
        "bottleScrew:maxAngularVelocityRadS", Sdf.ValueTypeNames.Double
    ).Set(spec.max_angular_velocity_rad_s)

    # The visible threads are render geometry.  Their mutual convex collision
    # would duplicate and fight the analytic helical constraint.
    filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(cap_path))
    filtered.CreateFilteredPairsRel().AddTarget(Sdf.Path(body_path))

    return {"metadata": metadata_path}
