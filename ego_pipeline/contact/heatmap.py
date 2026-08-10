"""Stage 4: the contact heatmap, in the object's own frame.

Geometry decides the heat: after the hand has been placed on the object (stage 3), a
vertex is hot if a tactile pad is close to it. The 2D observation does NOT paint heat --
it only vetoes: object surface the camera plainly sees uncovered cannot be in contact, so
those vertices are forced to zero no matter what the geometry says.

The result lives in the object-local frame, so it stays valid however the object moves
through the video, and is directly the affordance / contact prior that stage 3a of the
pipeline wants.
"""
from __future__ import annotations

import numpy as np


def contact_heatmap(mesh, obs_frames, sigma=0.008, veto_free=True, agg="max") -> dict:
    """Per-vertex contact weight from one or more aligned frames.

    obs_frames: list of dicts with
        T_obj     (4,4)  object-local -> world at that frame
        pads      (N,3)  world-space pad points, ALREADY transformed by the alignment
        link_id   (N,)   which pad link each point belongs to
        free      (V,)   bool, vertices proven untouched at that frame (may be None)
        weight    float  per-frame confidence (0..1)
    """
    from scipy.spatial import cKDTree

    V = np.asarray(mesh.vertices)
    per_frame, nearest_link, used = [], np.full(len(V), -1), []
    best_d = np.full(len(V), np.inf)

    for fr in obs_frames:
        T = np.asarray(fr["T_obj"], dtype=np.float64)
        pads_local = (np.asarray(fr["pads"]) - T[:3, 3]) @ T[:3, :3]     # world -> object
        d, idx = cKDTree(pads_local).query(V)
        w = np.exp(-(d ** 2) / (2.0 * sigma ** 2)) * float(fr.get("weight", 1.0))
        if veto_free and fr.get("free") is not None:
            w = np.where(fr["free"], 0.0, w)                             # hard negative evidence
        per_frame.append(w)
        closer = d < best_d
        best_d[closer] = d[closer]
        nearest_link[closer] = np.asarray(fr["link_id"])[idx[closer]]
        used.append(int(fr["frame"]))

    stack = np.stack(per_frame)
    H = stack.max(0) if agg == "max" else stack.mean(0)
    if H.max() > 0:
        H = H / H.max()

    free_any = np.zeros(len(V), bool)
    for fr in obs_frames:
        if fr.get("free") is not None:
            free_any |= fr["free"]

    return dict(weight=H, distance_m=best_d, nearest_link=nearest_link,
                frames=used, sigma=sigma, agg=agg, vetoed=free_any,
                n_hot=int((H > 0.5).sum()), n_vetoed=int(free_any.sum()),
                hot_area_frac=float((H > 0.5).mean()))


def colorize(weight, cmap="turbo") -> np.ndarray:
    """weight in [0,1] -> uint8 RGB, with the cold end left grey so 'no contact' does
    not read as a low-but-real value."""
    import matplotlib.cm as cm
    rgb = (np.asarray(cm.get_cmap(cmap)(np.clip(weight, 0, 1)))[:, :3] * 255)
    cold = weight < 0.02
    rgb[cold] = np.array([205, 205, 205])
    return rgb.astype(np.uint8)


def export_ply(mesh, weight, path, cmap="turbo"):
    """Object mesh with per-vertex heat colours -- opens in MeshLab / Blender / Isaac."""
    import trimesh
    m = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                        process=False)
    m.visual.vertex_colors = np.hstack([colorize(weight, cmap),
                                        np.full((len(weight), 1), 255, np.uint8)])
    m.export(path)
    return path


def summarize_by_link(hm, link_names) -> dict:
    """How much of the hot region each finger is responsible for."""
    hot = hm["weight"] > 0.5
    out = {}
    for i, name in enumerate(link_names):
        out[name] = int((hot & (hm["nearest_link"] == i)).sum())
    return out
