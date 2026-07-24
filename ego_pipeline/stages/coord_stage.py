"""
Stage 2.5: CoordUnify - convert all 3D data from OpenCV to Z-up right-hand.

OpenCV:  X=right, Y=down,  Z=forward
Z-up:    X=forward, Y=left, Z=up

Transform matrix T (det=+1, preserves handedness):
  T = [[ 0,  0,  1],    X_new =  Z_old (forward)
       [-1,  0,  0],    Y_new = -X_old (left)
       [ 0, -1,  0]]    Z_new = -Y_old (up)
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ego_pipeline.context import EgoContext
from .base import PipelineStage

# OpenCV -> Z-up right-hand system
T_OPENCV_TO_ZUP = np.array(
    [[ 0.,  0.,  1.],
     [-1.,  0.,  0.],
     [ 0., -1.,  0.]],
    dtype=np.float64,
)


class CoordUnifyStage(PipelineStage):
    """Convert all 3D data from OpenCV to Z-up right-hand coordinates."""

    def name(self) -> str:
        return "coord_unify"

    def check_deps(self, ctx: EgoContext) -> list[str]:
        missing = []
        if ctx.mano_trans is None:
            missing.append("mano_trans")
        if ctx.mano_rot is None:
            missing.append("mano_rot")
        return missing

    def run(self, ctx: EgoContext) -> EgoContext:
        T = T_OPENCV_TO_ZUP
        T_T = T.T

        # ── Transform camera poses (N, 4, 4) ────────────────────────────
        if ctx.poses_c2w is not None:
            poses = ctx.poses_c2w.copy()
            for i in range(len(poses)):
                R = poses[i, :3, :3].astype(np.float64)
                t = poses[i, :3, 3].astype(np.float64)
                poses[i, :3, :3] = (T @ R @ T_T).astype(np.float32)
                poses[i, :3, 3] = (T @ t).astype(np.float32)
            ctx.poses_c2w = poses
            print("  Transformed poses_c2w")

        # ── Transform MANO translation (2, N, 3) ────────────────────────
        for h in range(2):
            trans = ctx.mano_trans[h].astype(np.float64)  # (N, 3)
            ctx.mano_trans[h] = (trans @ T_T).astype(np.float32)  # (T @ v)^T = v @ T^T

        side_names = ["left", "right"]
        print(f"  Transformed mano_trans")

        # ── Transform MANO global rotation (2, N, 3) axis-angle ─────────
        for h in range(2):
            for t in range(ctx.mano_rot.shape[1]):
                if ctx.mano_valid is not None and not ctx.mano_valid[h, t]:
                    continue
                aa = ctx.mano_rot[h, t].astype(np.float64)
                R = Rotation.from_rotvec(aa).as_matrix()
                R_new = T @ R @ T_T
                ctx.mano_rot[h, t] = Rotation.from_matrix(R_new).as_rotvec().astype(np.float32)
        print(f"  Transformed mano_rot")

        # mano_pose (joint-local rotations) — NOT transformed
        # mano_betas (shape params) — NOT transformed
        # depth (scalar field) — NOT transformed
        # intrinsics (2D pixel space) — NOT transformed

        print(f"  Coordinate system: OpenCV -> Z-up (X=fwd, Y=left, Z=up)")
        return ctx
