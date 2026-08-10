"""What the video actually tells us about the hand-object relation at one frame.

IMPORTANT -- what this is NOT. The pixels where the hand mask meets the object mask are
the hand's OCCLUSION BOUNDARY drawn on the object, i.e. the outline of the hand seen
against it. They are not the fingertip contact points: in a real grasp the fingers touch
the object BEHIND the hand, which a single view can never see.

So this module produces POSE EVIDENCE, not contact labels:

  bite2d    object silhouette pixels covered by the hand -> the 2D target that pins the
            two degrees of freedom perpendicular to the viewing ray (depth is nearly
            unconstrained by a silhouette; the object's known 3D pose supplies that).
  free      object surface that is visible AND not covered by the hand -> proof that
            NOTHING touches there. Hard negative evidence for the contact heatmap.
  occluded  object vertices hidden behind the hand -> contact is somewhere in here or
            further behind. A region of uncertainty, not a positive label.
  rim       vertices along the occlusion boundary itself (what the first version of this
            code mislabelled as "the contact region").
"""
from __future__ import annotations

import numpy as np


def project_points(P_world: np.ndarray, c2w: np.ndarray, K: np.ndarray, hw) -> tuple:
    """World points -> integer pixels (OpenCV convention: camera looks down +z).

    Returns (uv int (N,2), depth (N,), in_bounds (N,) bool). Points behind the camera
    keep depth <= 0 and must be filtered by the caller.
    """
    H, W = hw
    R, t = c2w[:3, :3], c2w[:3, 3]
    Pc = (P_world - t) @ R                       # world -> camera  (R^T @ (p - t))
    z = Pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * Pc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / zs + K[1, 2]
    uv = np.stack([u, v], 1)
    inb = np.isfinite(uv).all(1)
    uvi = np.zeros_like(uv, dtype=np.int32)
    uvi[inb] = np.round(uv[inb]).astype(np.int32)
    inb &= (uvi[:, 0] >= 0) & (uvi[:, 0] < W) & (uvi[:, 1] >= 0) & (uvi[:, 1] < H)
    return uvi, z, inb


def splat_depth(uv, z, ok, hw, radius=2) -> np.ndarray:
    """Point cloud -> front-depth image (inf where nothing projects)."""
    from scipy.ndimage import minimum_filter
    H, W = hw
    buf = np.full((H, W), np.inf)
    np.minimum.at(buf, (uv[ok, 1], uv[ok, 0]), z[ok])
    return minimum_filter(buf, size=2 * radius + 1) if radius > 0 else buf


def splat_mask(uv, ok, hw, radius=2, close=9) -> np.ndarray:
    """Point cloud -> filled binary silhouette."""
    import cv2
    H, W = hw
    m = np.zeros((H, W), np.uint8)
    m[uv[ok, 1], uv[ok, 0]] = 1
    if radius > 0:
        m = cv2.dilate(m, np.ones((2 * radius + 1,) * 2, np.uint8))
    if close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close,) * 2, np.uint8))
    return m.astype(bool)


def visible_vertices(V_world, normals_world, c2w, K, hw,
                     depth_tol=0.006, splat=2) -> np.ndarray:
    """Which mesh vertices are seen by the camera (ignoring the hand).

    Two independent tests, both required: the vertex normal must face the camera, and
    the vertex must sit within `depth_tol` of the splatted front depth at its pixel.
    """
    uv, z, inb = project_points(V_world, c2w, K, hw)
    view = V_world - c2w[:3, 3]
    view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-12
    front = (normals_world * view).sum(1) < 0

    ok = inb & (z > 0)
    buf = splat_depth(uv, z, ok, hw, radius=splat)
    vis = np.zeros(len(V_world), dtype=bool)
    vis[ok] = z[ok] <= buf[uv[ok, 1], uv[ok, 0]] + depth_tol
    return vis & front


def contact_band(hand_mask, obj_mask, dilate_px=9) -> np.ndarray:
    """Pixels where the (dilated) hand mask abuts the VISIBLE object mask.

    Careful: this is the seam between the hand and the object's still-visible part, not
    the occluded area. SAM2's object mask already has the hand-covered region removed,
    so pressing harder shrinks this number instead of growing it. Kept for display only
    -- do NOT use it to choose a frame (see best_evidence_frame).
    """
    import cv2
    k = np.ones((2 * dilate_px + 1,) * 2, np.uint8)
    return (cv2.dilate((hand_mask > 0).astype(np.uint8), k)
            & (obj_mask > 0).astype(np.uint8)).astype(bool)


def bite_pixels(take, f: int, V_local, hand_mask=None) -> int:
    """Area of the observed bite at frame f: object silhouette covered by the hand.

    This is exactly the quantity stage 3 maximises the overlap with, so it is also the
    right thing to rank frames by.
    """
    import cv2
    if hand_mask is None:
        hm = cv2.imread(str(take.mask_path("hand", f)), cv2.IMREAD_UNCHANGED)
        if hm is None:
            return -1
        hand_mask = hm > 0
    T = take.obj_T_world[f]
    Vw = V_local @ T[:3, :3].T + T[:3, 3]
    uv, z, inb = project_points(Vw, take.c2w[f], take.K, hand_mask.shape)
    proj = splat_mask(uv, inb & (z > 0), hand_mask.shape, radius=2, close=15)
    return int((proj & hand_mask).sum())


def best_evidence_frame(take, mesh, frames=None, require_valid=True, log=None) -> tuple:
    """Frame whose observed bite is largest -- i.e. where the 2D evidence is strongest.

    The first frame of an annotated grasp interval is the WORST choice: contact has only
    just begun, the hand has barely closed, and the bite is near-empty, which leaves the
    silhouette term with nothing to optimise against.

    Frames whose hand qpos was hole-filled, or whose object pose is invalid, are skipped
    -- aligning against a frame the reconstruction itself flagged as bad is pointless.
    """
    V = np.asarray(mesh.vertices)
    frames = list(range(take.Tv)) if frames is None else list(frames)
    best, table = (-1, frames[0]), []
    for g in frames:
        if require_valid and not (take.hand_valid[g] and take.obj_valid[g]):
            continue
        n = bite_pixels(take, g, V)
        if n < 0:
            continue
        table.append((int(g), n))
        if n > best[0]:
            best = (n, int(g))
    if log and table:
        top = sorted(table, key=lambda r: -r[1])[:5]
        log(f"[frame]  bite area by frame (top 5): {top}  of {len(table)} usable frames")
    return best[1], best[0], table


def hand_occlusion_evidence(take, f: int, mesh, dilate_px=9, falloff_px=25,
                            depth_tol=0.006) -> dict:
    """Everything frame f says about where the hand is, relative to the object."""
    import cv2

    hm = cv2.imread(str(take.mask_path("hand", f)), cv2.IMREAD_UNCHANGED)
    om = cv2.imread(str(take.mask_path("object", f)), cv2.IMREAD_UNCHANGED)
    if hm is None or om is None:
        raise FileNotFoundError(f"missing masks for frame {f}: "
                                f"{take.mask_path('hand', f)} / {take.mask_path('object', f)}")
    hand_mask, obj_mask = hm > 0, om > 0
    H, W = obj_mask.shape

    T = take.obj_T_world[f]
    V = np.asarray(mesh.vertices)
    Vw = V @ T[:3, :3].T + T[:3, 3]
    Nw = np.asarray(mesh.vertex_normals) @ T[:3, :3].T

    uv, z, inb = project_points(Vw, take.c2w[f], take.K, (H, W))
    vis = visible_vertices(Vw, Nw, take.c2w[f], take.K, (H, W), depth_tol=depth_tol)
    ok = inb & (z > 0)

    # object silhouette + its front-depth image: the stage against which the hand is seen
    proj_obj = splat_mask(uv, ok, (H, W), radius=2, close=15)
    obj_depth = splat_depth(uv, z, ok, (H, W), radius=2)

    # the three evidence channels
    bite2d = proj_obj & hand_mask                       # object hidden by the hand
    free2d = proj_obj & obj_mask & ~hand_mask           # object plainly visible => no contact
    occluded = np.zeros(len(V), bool)
    free = np.zeros(len(V), bool)
    sel = vis & inb
    occluded[sel] = bite2d[uv[sel, 1], uv[sel, 0]]
    free[sel] = free2d[uv[sel, 1], uv[sel, 0]]

    # occlusion rim (soft): distance-decayed band along the hand/object boundary
    k = np.ones((2 * dilate_px + 1,) * 2, np.uint8)
    band = (cv2.dilate(hand_mask.astype(np.uint8), k) & obj_mask.astype(np.uint8)).astype(bool)
    dist = cv2.distanceTransform((~band).astype(np.uint8), cv2.DIST_L2, 3)
    wimg = np.clip(1.0 - dist / max(falloff_px, 1), 0.0, 1.0)
    rim_weight = np.zeros(len(V))
    rim_weight[sel] = wimg[uv[sel, 1], uv[sel, 0]]

    return dict(
        frame=int(f), uv=uv, depth=z, in_bounds=inb, visible=vis,
        hand_mask=hand_mask, obj_mask=obj_mask,
        proj_obj=proj_obj, obj_depth=obj_depth,
        bite2d=bite2d, free2d=free2d,
        occluded=occluded, free=free, rim_weight=rim_weight, band=band,
        n_occluded=int(occluded.sum()), n_free=int(free.sum()),
        bite_px=int(bite2d.sum()), free_px=int(free2d.sum()),
        proj_obj_px=int(proj_obj.sum()), band_px=int(band.sum()),
        occlusion_ratio=float(bite2d.sum() / max(proj_obj.sum(), 1)),
        bite_centroid_px=(np.array(np.nonzero(bite2d)).mean(1)[::-1]
                          if bite2d.any() else np.full(2, np.nan)),
    )
