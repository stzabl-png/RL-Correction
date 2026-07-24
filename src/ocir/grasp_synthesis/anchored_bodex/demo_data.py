"""Human-demonstration input loading for anchored BODex grasp synthesis.

Extends the generic sequence-directory format (see
:mod:`ocir.grasp_synthesis.object_surface`) with one required artifact:

    <sequence_dir>/
      human_demo.npz     # MANO hand + object trajectory, object canonical frame
      affordance.npz     # optional: written lazily by anchored_bodex.affordance

``sequence.json`` can override both locations with ``"human_demo"`` /
``"affordance"`` keys (paths relative to the sequence dir, or absolute),
mirroring the existing ``object_mesh``/``points`` override pattern.

``human_demo.npz`` schema (T demo frames):

    frame_ids            (T,)   int32   source frame ids, monotonically increasing
    mano_side            ()     str     "right" / "left"
    betas                (10,)  float   MANO shape parameters
    pose_m_camera        (T,51) float   raw MANO pose (provenance only)
    object_pose_camera   (T,4,4) float  grasped-object pose in camera frame
    hand_joints_object   (T,21,3) float 21 keypoints, object canonical frame
                                        (order: wrist, thumb*4, index*4,
                                        middle*4, ring*4, pinky*4; fingertips
                                        at indices 4/8/12/16/20)
    hand_vertices_object (T,778,3) float MANO vertices, object canonical frame
    vertex_part_ids      (778,) int8    argmax MANO skinning-weight joint per
                                        vertex (0=wrist/palm .. 15), so
                                        synthesis never needs the MANO pkl
    valid_mask           (T,)   bool    frame has valid MANO + object pose;
                                        invalid frames hold NaN rows

All hand data is pre-transformed into the object canonical frame by the
exporter (``ocir.dexycb.export_grasp_sequences``), so this loader needs no
camera calibration and no MANO model files.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

DEMO_FILENAME = "human_demo.npz"
AFFORDANCE_FILENAME = "affordance.npz"

_REQUIRED_KEYS = (
    "frame_ids",
    "mano_side",
    "betas",
    "object_pose_camera",
    "hand_joints_object",
    "hand_vertices_object",
    "vertex_part_ids",
    "valid_mask",
)


def resolve_sequence_file(sequence_dir: Path, meta: dict, key: str, default_name: str) -> Path | None:
    """Resolve an optional per-sequence artifact path (sequence.json override
    first, then the conventional filename)."""

    explicit = meta.get(key)
    if explicit is not None:
        path = Path(explicit)
        path = path if path.is_absolute() else sequence_dir / path
        if not path.exists():
            raise FileNotFoundError(f"sequence.json {key!r} not found: {path}")
        return path
    default = sequence_dir / default_name
    return default if default.exists() else None


@dataclass(frozen=True)
class HumanDemo:
    sequence_dir: Path
    frame_ids: np.ndarray            # (T,) int32
    mano_side: str
    betas: np.ndarray                # (10,)
    object_pose_camera: np.ndarray   # (T, 4, 4)
    hand_joints_object: np.ndarray   # (T, 21, 3)
    hand_vertices_object: np.ndarray  # (T, 778, 3)
    vertex_part_ids: np.ndarray      # (778,) int8
    valid_mask: np.ndarray           # (T,) bool
    metadata: dict

    @property
    def num_frames(self) -> int:
        return int(self.frame_ids.shape[0])

    @property
    def valid_indices(self) -> np.ndarray:
        return np.flatnonzero(self.valid_mask)

    @classmethod
    def from_sequence_dir(cls, sequence_dir: str | Path) -> "HumanDemo":
        sequence_dir = Path(sequence_dir)
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"sequence directory not found: {sequence_dir}")
        meta_path = sequence_dir / "sequence.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        demo_path = resolve_sequence_file(sequence_dir, meta, "human_demo", DEMO_FILENAME)
        if demo_path is None:
            raise FileNotFoundError(
                f"no {DEMO_FILENAME} in {sequence_dir} (and no 'human_demo' override in "
                "sequence.json); anchored BODex requires a human demonstration -- run "
                "scripts/dexycb/export_grasp_sequences.py (or an equivalent exporter) first"
            )
        with np.load(demo_path, allow_pickle=False) as data:
            missing = [key for key in _REQUIRED_KEYS if key not in data.files]
            if missing:
                raise KeyError(f"{demo_path} is missing required keys {missing}")
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int32)
            demo = cls(
                sequence_dir=sequence_dir,
                frame_ids=frame_ids,
                mano_side=str(data["mano_side"]),
                betas=np.asarray(data["betas"], dtype=np.float64),
                object_pose_camera=np.asarray(data["object_pose_camera"], dtype=np.float64),
                hand_joints_object=np.asarray(data["hand_joints_object"], dtype=np.float64),
                hand_vertices_object=np.asarray(data["hand_vertices_object"], dtype=np.float64),
                vertex_part_ids=np.asarray(data["vertex_part_ids"], dtype=np.int8),
                valid_mask=np.asarray(data["valid_mask"], dtype=bool),
                metadata={
                    "demo_path": str(demo_path),
                    "sequence_id": meta.get("sequence_id", sequence_dir.name),
                },
            )
        t = demo.num_frames
        for name, arr, shape in (
            ("object_pose_camera", demo.object_pose_camera, (t, 4, 4)),
            ("hand_joints_object", demo.hand_joints_object, (t, 21, 3)),
            ("hand_vertices_object", demo.hand_vertices_object, (t, 778, 3)),
            ("vertex_part_ids", demo.vertex_part_ids, (778,)),
            ("valid_mask", demo.valid_mask, (t,)),
        ):
            if tuple(arr.shape) != shape:
                raise ValueError(f"{demo_path}: {name} has shape {tuple(arr.shape)}, expected {shape}")
        if not demo.valid_mask.any():
            raise ValueError(f"{demo_path}: no valid demo frames")
        return demo
