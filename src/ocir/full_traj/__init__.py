"""Open-loop full-trajectory carry retargeting.

Generation preserves the completed grasp prefix, transfers the recorded MANO
wrist motion onto the synthesized wrist, and keeps finger targets fixed during
carry.  Importing :mod:`ocir.full_traj` never imports Isaac Sim.
"""

from ocir.full_traj.reference import FullTrajectoryReference

__all__ = ["FullTrajectoryReference"]
