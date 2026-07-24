"""Point-cloud normalization used at both train and inference time (vendored)."""
from __future__ import annotations

import numpy as np


def normalize_unit_sphere(pts: np.ndarray):
    """Center to centroid, scale to unit sphere. Returns (pts, center, scale)."""
    center = pts.mean(0)
    p = pts - center
    scale = np.linalg.norm(p, axis=1).max()
    scale = max(scale, 1e-6)
    return (p / scale).astype(np.float32), center, scale
