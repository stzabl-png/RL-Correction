"""Grasp trajectory generation: turn a synthesized anchored-BODex grasp into a
full approach -> grasp -> carry manipulation, driven by the human video demo.

Two-stage pipeline, split across two conda environments:

- **Stage A** (this package, grasp-synthesis env, torch/CUDA): reads a
  synthesized grasp record + its human demo sequence, retargets the hand
  along the demo, chooses a collision-safe switch-frame from retargeted
  replay to a synthetic approach, and writes a ``trajectory.npz``/``.json``
  pair describing the whole manipulation as per-step hand/object poses and
  finger joint targets, all in the demo's camera frame.
- **Stage B** (``dexisaac.isaac.simulate_grasp_traj``, isaacsim env): loads that
  trajectory and plays it back in Isaac Sim with real PhysX physics on the
  object (gravity, collision, a table, tuned friction) and PD-driven finger
  joints, rendering a video.

``trajectory_schema.py`` is the only module imported by both stages -- it
must never import torch or pxr.
"""
