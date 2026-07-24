"""Generic per-sequence object input loading for BODex-style grasp synthesis.

A "sequence" is just a directory that provides one object mesh (plus
optional metadata/pre-sampled surface points) -- not tied to DexYCB or any
other specific dataset. This replaces the earlier DexYCB-specific "contact
heatmap" artifact format: the grasp-synthesis algorithm itself never read
the heatmap (it's a visualization-only quantity), only a rough object point
cloud for bounds/center-of-mass estimation, which is now derived directly
from the mesh when a curated point set isn't provided.

Minimal sequence directory:

    <sequence_dir>/
      <object_name>.obj (or .stl)   # required: exactly one mesh file
      points.xyz                    # optional: pre-sampled surface points
      sequence.json                 # optional: explicit overrides

``sequence.json`` (all keys optional) can set ``object_mesh`` and/or
``points`` (paths relative to the sequence dir, or absolute) to disambiguate
when a directory holds more than one mesh, plus arbitrary metadata such as
``sequence_id``/``object_name`` that would otherwise default to the
directory name / mesh filename stem.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import trimesh

#: Surface point count used when a sequence has no curated `points.xyz`.
DEFAULT_SAMPLE_POINTS = 2048


@dataclass(frozen=True)
class ObjectSurface:
    points_object_frame: np.ndarray
    object_mesh_path: Path
    metadata: dict

    @classmethod
    def from_sequence_dir(cls, sequence_dir: str | Path) -> "ObjectSurface":
        sequence_dir = Path(sequence_dir)
        if not sequence_dir.is_dir():
            raise FileNotFoundError(f"sequence directory not found: {sequence_dir}")

        meta_path = sequence_dir / "sequence.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        object_mesh_path = _resolve_object_mesh(sequence_dir, meta)
        points_object_frame = _resolve_points(sequence_dir, meta, object_mesh_path)

        overrides = {"object_mesh", "points"}
        metadata = {
            "sequence_id": meta.get("sequence_id", sequence_dir.name),
            "object_name": meta.get("object_name", object_mesh_path.stem),
            "object_mesh_path": str(object_mesh_path),
            "sequence_dir": str(sequence_dir),
            **{key: value for key, value in meta.items() if key not in overrides},
        }
        return cls(points_object_frame, object_mesh_path, metadata)


def _resolve_object_mesh(sequence_dir: Path, meta: dict) -> Path:
    explicit = meta.get("object_mesh")
    if explicit is not None:
        path = Path(explicit)
        path = path if path.is_absolute() else sequence_dir / path
        if not path.exists():
            raise FileNotFoundError(f"sequence.json object_mesh not found: {path}")
        return path
    candidates = sorted(sequence_dir.glob("*.obj")) + sorted(sequence_dir.glob("*.stl"))
    if not candidates:
        raise FileNotFoundError(
            f"no object mesh (*.obj/*.stl) found in {sequence_dir}; "
            "add one or set 'object_mesh' in sequence.json"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"multiple candidate meshes in {sequence_dir}: {candidates}; "
            "set 'object_mesh' in sequence.json to disambiguate"
        )
    return candidates[0]


def _resolve_points(sequence_dir: Path, meta: dict, object_mesh_path: Path) -> np.ndarray:
    explicit = meta.get("points")
    if explicit is not None:
        path = Path(explicit)
        path = path if path.is_absolute() else sequence_dir / path
        return np.loadtxt(path, dtype=float).reshape(-1, 3)
    default_points = sequence_dir / "points.xyz"
    if default_points.exists():
        return np.loadtxt(default_points, dtype=float).reshape(-1, 3)
    mesh = trimesh.load(str(object_mesh_path), force="mesh", process=False)
    points, _ = trimesh.sample.sample_surface(mesh, DEFAULT_SAMPLE_POINTS)
    return np.asarray(points, dtype=float)
