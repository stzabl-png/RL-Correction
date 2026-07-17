"""Closed-loop full-trajectory object-path tracking.

The package is deliberately split into import-safe NumPy reference/control
code and an Isaac-facing runtime module.  Importing :mod:`ocir.full_traj`
never imports Isaac Sim, torch, or cuRobo.
"""

from ocir.full_traj.controller import ControllerConfig, PathFollowingController
from ocir.full_traj.reference import FullTrajectoryReference

__all__ = ["ControllerConfig", "FullTrajectoryReference", "PathFollowingController"]
