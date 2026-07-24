"""Compatibility re-export: the clearance machinery moved to
:mod:`ocir.grasp_synthesis.clearance` so the synthesis pipelines can use it
too (anchored_bodex computes its four-stage grasp poses with it at synthesis
time). Import from there in new code."""

from ocir.grasp_synthesis.clearance import (  # noqa: F401
    ClearanceChecker,
    build_contact_world,
    load_all_collision_spheres,
)
