"""Grasp robustness stress tests, independent of the full_traj retarget pipeline.

The rotation gauntlet replaces a proven vertical-lift carry with a scripted
wrist-rotation program about the held object's center, then scores each
rotation leg from the recorded object track.  Importing this package never
imports Isaac Sim.
"""

from ocir.grasp_robustness.rotation_gauntlet import GauntletConfig, build_rotation_gauntlet

__all__ = ["GauntletConfig", "build_rotation_gauntlet"]
