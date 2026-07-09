"""Object surface loading for BODex-style grasp synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObjectSurface:
    points_object_frame: np.ndarray
    heatmap: np.ndarray | None
    contact_count: np.ndarray | None
    metadata: dict

    @classmethod
    def load(cls, path: str | Path) -> "ObjectSurface":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        points = np.asarray(data["points_object_frame"], dtype=np.float64)
        heatmap = np.asarray(data["heatmap"], dtype=np.float64) if "heatmap" in data.files else None
        contact_count = np.asarray(data["contact_count"], dtype=np.int64) if "contact_count" in data.files else None
        metadata = {key: data[key].item() if data[key].shape == () else data[key] for key in data.files}
        return cls(points, heatmap, contact_count, {**metadata, "source_path": str(path)})

    def estimated_normals(self) -> np.ndarray:
        center = np.mean(self.points_object_frame, axis=0)
        normals = self.points_object_frame - center[None, :]
        return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)


def infer_surface_artifact_path(sequence_dir: str | Path) -> Path:
    candidates = sorted((Path(sequence_dir) / "affordance").glob("*_contact_heatmap.npz"))
    if not candidates:
        raise FileNotFoundError(f"no affordance contact heatmap found under {Path(sequence_dir) / 'affordance'}")
    if len(candidates) > 1:
        raise ValueError(f"multiple affordance artifacts found: {candidates}")
    return candidates[0]
