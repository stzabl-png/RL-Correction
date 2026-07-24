"""
EgoContext - shared data structure passed between pipeline stages.

All 3D data is in Z-up right-hand coordinates (X=forward, Y=left, Z=up)
after the CoordUnify stage.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class EgoContext:
    """Shared context flowing through every pipeline stage."""

    # Input
    video_path: str = ""
    output_dir: str = ""

    # Stage 1: ViPE
    frames: np.ndarray | None = None        # (N, H, W, 3) uint8
    intrinsics: np.ndarray | None = None    # (3, 3) float64
    poses_c2w: np.ndarray | None = None     # (N, 4, 4) float32
    depth: np.ndarray | None = None         # (N, H, W) float32 meters

    # Stage 2+2.5+3: HaWoR + CoordUnify + Smooth
    mano_trans: np.ndarray | None = None    # (2, N, 3)
    mano_rot: np.ndarray | None = None      # (2, N, 3) axis-angle
    mano_pose: np.ndarray | None = None     # (2, N, 45)
    mano_betas: np.ndarray | None = None    # (2, N, 10)
    mano_valid: np.ndarray | None = None    # (2, N) bool

    # Object branch (ObjectMask -> ObjectMesh -> ObjectPose). Keys:
    #   "object_mask"     (N,H,W) binary per-frame object mask    [ObjectMaskStage]
    #   "instance_mask"   (N,H,W) 1=left-obj,2=right-obj,3=both   [ObjectMaskStage]
    #   "mesh_path"       canonical .obj                          [ObjectMeshStage]
    #   "mesh_frame_idx"  frame used for reconstruction           [ObjectMeshStage]
    #   "mesh_mask"       (H,W) mask of that frame                 [ObjectMeshStage]
    #   "pose"            (N,4,4) ob_in_cam per frame             [ObjectPoseStage]
    #   "pose_dir"        dir with ob_in_cam/ + track_vis/        [ObjectPoseStage]
    objects: dict[str, Any] | None = None

    # Internal bookkeeping
    _hawor_seq_folder: str = ""
    _hawor_start_idx: int = 0
    _hawor_end_idx: int = 0
