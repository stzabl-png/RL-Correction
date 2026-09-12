"""URDF forward kinematics for Dexmate robots (pinocchio + dexmate-urdf).

Only joint positions are needed as input; the robot itself is never touched here,
so captured sessions can be re-solved offline with a different robot model or link.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUPPORTED_MODELS = ("vega_1", "vega_1p", "vega_1u")


def resolve_urdf_path(model: str) -> Path:
    """Locate ``<model>.urdf`` inside the installed ``dexmate-urdf`` package."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown robot model {model!r}; expected one of {SUPPORTED_MODELS}")
    try:
        from dexmate_urdf import robots
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("dexmate-urdf is not installed; run `uv sync --extra robot`") from exc
    entry = getattr(getattr(robots.humanoid, model), model)
    return Path(str(entry.urdf))


@dataclass
class FrameKinematics:
    """Forward kinematics of one URDF, evaluated from a joint-name → position mapping."""

    model_name: str
    urdf_path: Path
    base_frame: str = "base"

    def __post_init__(self) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pinocchio is not installed; run `uv sync --extra robot`") from exc
        self._pin = pin
        self._model = pin.buildModelFromUrdf(str(self.urdf_path))
        self._data = self._model.createData()
        if not self._model.existFrame(self.base_frame):
            raise ValueError(f"Frame {self.base_frame!r} does not exist in {self.urdf_path.name}")
        self._base_id = self._model.getFrameId(self.base_frame)
        self._joint_index: dict[str, tuple[int, int]] = {}
        for joint_id in range(1, self._model.njoints):
            joint = self._model.joints[joint_id]
            name = self._model.names[joint_id]
            self._joint_index[name] = (joint.idx_q, joint.nq)

    @classmethod
    def from_model(cls, model_name: str, base_frame: str = "base") -> FrameKinematics:
        return cls(
            model_name=model_name, urdf_path=resolve_urdf_path(model_name), base_frame=base_frame
        )

    # ------------------------------------------------------------------ queries
    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_index)

    @property
    def frame_names(self) -> list[str]:
        return [frame.name for frame in self._model.frames]

    def has_frame(self, name: str) -> bool:
        return bool(self._model.existFrame(name))

    def configuration(
        self, joint_positions: dict[str, float], *, strict: bool = True
    ) -> np.ndarray:
        """Build the pinocchio configuration vector from a name → radians mapping.

        Joints missing from ``joint_positions`` stay at their neutral value; with
        ``strict`` unknown joint names raise instead of being ignored.
        """
        q = self._pin.neutral(self._model)
        for name, value in joint_positions.items():
            if name not in self._joint_index:
                if strict:
                    raise KeyError(f"Joint {name!r} not in URDF {self.urdf_path.name}")
                continue
            idx, nq = self._joint_index[name]
            if nq == 1:
                q[idx] = float(value)
            elif nq == 2:  # continuous joints are stored as (cos, sin)
                q[idx] = float(np.cos(value))
                q[idx + 1] = float(np.sin(value))
            else:  # pragma: no cover - not present in Vega URDFs
                raise ValueError(f"Unsupported joint {name!r} with nq={nq}")
        return q

    def frame_pose(
        self, joint_positions: dict[str, float], frame: str, *, strict: bool = True
    ) -> np.ndarray:
        """Return ``T_base_frame`` (4x4) for the given joint positions."""
        if not self._model.existFrame(frame):
            raise ValueError(f"Frame {frame!r} does not exist in {self.urdf_path.name}")
        pin = self._pin
        q = self.configuration(joint_positions, strict=strict)
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        T_world_base = self._data.oMf[self._base_id]
        T_world_frame = self._data.oMf[self._model.getFrameId(frame)]
        return np.asarray((T_world_base.inverse() * T_world_frame).homogeneous, dtype=np.float64)

    def required_joints_for(self, frame: str) -> list[str]:
        """Names of the actuated joints between ``base_frame`` and ``frame``."""
        model = self._model
        fid = model.getFrameId(frame)
        base_joint = model.frames[self._base_id].parentJoint
        joint = model.frames[fid].parentJoint
        chain: list[str] = []
        while joint != 0 and joint != base_joint:
            chain.append(model.names[joint])
            joint = model.parents[joint]
        return list(reversed(chain))
