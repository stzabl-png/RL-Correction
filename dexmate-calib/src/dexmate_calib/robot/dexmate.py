"""Read-only access to Dexmate joint states through ``dexcontrol``.

The calibration tools never command motion.  During hand-eye capture the operator
moves the robot with dexcontrol's own keyboard example (or any other teleop) and
this module only samples joint positions, checking that the robot is stationary
so the photo and the joint reading describe the same pose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_COMPONENTS = ("torso", "left_arm", "right_arm", "head")


@dataclass
class JointSample:
    positions: dict[str, float]
    stationary: bool
    max_motion_rad: float
    read_time_s: float
    checks: int


@dataclass
class DexmateJointReader:
    """Context manager wrapping ``dexcontrol.Robot`` for joint reads only."""

    components: tuple[str, ...] = DEFAULT_COMPONENTS
    settle_checks: int = 3
    settle_interval_s: float = 0.15
    settle_tolerance_rad: float = 2e-3
    _robot: object | None = field(default=None, repr=False)

    def open(self) -> DexmateJointReader:
        try:
            from dexcontrol.robot import Robot
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("dexcontrol is not installed; run `uv sync --extra robot`") from exc
        # auto_shutdown=False: the capture loop owns Ctrl-C handling.
        self._robot = Robot(auto_shutdown=False)
        available = set(self._robot.get_controllable_component_map())  # type: ignore[attr-defined]
        missing = [c for c in self.components if c not in available]
        if missing:
            self.close()
            raise RuntimeError(f"Robot lacks components {missing}; available: {sorted(available)}")
        return self

    def close(self) -> None:
        if self._robot is not None:
            try:
                self._robot.shutdown()  # type: ignore[attr-defined]
            finally:
                self._robot = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def robot_name(self) -> str:
        if self._robot is None:
            raise RuntimeError("DexmateJointReader is not open")
        return str(getattr(self._robot, "_robot_name", "unknown"))

    def component(self, name: str):
        """Return the underlying dexcontrol component object (arm/torso/head)."""
        if self._robot is None:
            raise RuntimeError("DexmateJointReader is not open")
        if name not in self.components:
            raise KeyError(f"Component {name!r} not opened; available: {self.components}")
        return getattr(self._robot, name)

    def read(self) -> dict[str, float]:
        if self._robot is None:
            raise RuntimeError("DexmateJointReader is not open")
        raw = self._robot.get_joint_pos_dict(list(self.components))  # type: ignore[attr-defined]
        return {str(k): float(v) for k, v in raw.items()}

    def read_stationary(self) -> JointSample:
        """Read joints ``settle_checks`` times and report the largest inter-read motion."""
        reads = [self.read()]
        for _ in range(max(0, self.settle_checks - 1)):
            time.sleep(self.settle_interval_s)
            reads.append(self.read())
        names = sorted(reads[0])
        arr = np.array([[r[n] for n in names] for r in reads], dtype=np.float64)
        max_motion = float(np.max(np.ptp(arr, axis=0))) if len(reads) > 1 else 0.0
        return JointSample(
            positions=reads[-1],
            stationary=max_motion <= self.settle_tolerance_rad,
            max_motion_rad=max_motion,
            read_time_s=time.time(),
            checks=len(reads),
        )
