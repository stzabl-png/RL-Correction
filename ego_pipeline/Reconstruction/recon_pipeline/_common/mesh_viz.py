"""OpenCV helpers for projecting meshes, axes, and boxes in the camera frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_VIZ_TRIANGLES = 3000


def face_stride(num_faces: int, *, max_triangles: int = DEFAULT_MAX_VIZ_TRIANGLES) -> int:
    return max(1, int(num_faces) // max(1, int(max_triangles)))


@dataclass(frozen=True)
class TriangleLayer:
    mean_z: float
    uv: np.ndarray  # (3, 2) float
    color_bgr: tuple[int, int, int]
    alpha: float


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] == 3:
        pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    else:
        pts_h = pts
    out = (T @ pts_h.T).T
    return out[:, :3]


def project_cam(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts_cam, dtype=np.float64)
    proj = (K @ pts.T).T
    z = np.maximum(proj[:, 2], 1e-6)
    uv = np.stack([proj[:, 0] / z, proj[:, 1] / z], axis=1)
    return uv, z


def axis_aligned_bbox_corners(verts: np.ndarray) -> np.ndarray:
    v = np.asarray(verts, dtype=np.float64)
    lo = v.min(axis=0)
    hi = v.max(axis=0)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )


_BBOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _clip_point(p: np.ndarray, w: int, h: int) -> tuple[int, int] | None:
    x, y = int(round(float(p[0]))), int(round(float(p[1])))
    if x < -w or x > 2 * w or y < -h or y > 2 * h:
        return None
    return x, y


def draw_bbox_wireframe(
    frame,
    corners_cam: np.ndarray,
    K: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    import cv2

    uv, z = project_cam(K, corners_cam)
    h, w = frame.shape[:2]
    for i0, i1 in _BBOX_EDGES:
        if z[i0] <= 1e-4 or z[i1] <= 1e-4:
            continue
        p0 = _clip_point(uv[i0], w, h)
        p1 = _clip_point(uv[i1], w, h)
        if p0 is None or p1 is None:
            continue
        cv2.line(frame, p0, p1, color_bgr, thickness, cv2.LINE_AA)


def draw_coord_axes(
    frame,
    T_cam: np.ndarray,
    K: np.ndarray,
    *,
    axis_len: float,
    thickness: int = 3,
) -> None:
    import cv2

    axis_pts = np.array(
        [
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
        ],
        dtype=np.float64,
    )
    pts_cam = transform_points(T_cam, axis_pts)
    uv, z = project_cam(K, pts_cam)
    if z[0] <= 1e-4:
        return
    origin = _clip_point(uv[0], frame.shape[1], frame.shape[0])
    if origin is None:
        return
    # OpenCV uses BGR tuples: X=red, Y=green, Z=blue.
    for idx, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)], start=1):
        if z[idx] <= 1e-4:
            continue
        tip = _clip_point(uv[idx], frame.shape[1], frame.shape[0])
        if tip is None:
            continue
        cv2.arrowedLine(frame, origin, tip, color, thickness, tipLength=0.18, line_type=cv2.LINE_AA)


def collect_mesh_triangles(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    alpha: float,
    stride: int = 1,
) -> list[TriangleLayer]:
    verts = np.asarray(verts_cam, dtype=np.float64)
    tris: list[TriangleLayer] = []
    for face in np.asarray(faces, dtype=np.int64)[:: max(1, stride)]:
        tri_cam = verts[face]
        if np.any(tri_cam[:, 2] <= 1e-4):
            continue
        uv, _z = project_cam(K, tri_cam)
        tris.append(
            TriangleLayer(
                mean_z=float(tri_cam[:, 2].mean()),
                uv=uv.astype(np.float64),
                color_bgr=color_bgr,
                alpha=alpha,
            )
        )
    return tris


def draw_sorted_triangles(frame, triangles: list[TriangleLayer], *, w: int, h: int) -> None:
    import cv2

    if not triangles:
        return
    order = np.argsort([-t.mean_z for t in triangles])
    for idx in order:
        tri = triangles[idx]
        poly = np.round(tri.uv).astype(np.int32)
        if (
            np.all((poly[:, 0] < 0) | (poly[:, 0] >= w))
            or np.all((poly[:, 1] < 0) | (poly[:, 1] >= h))
        ):
            continue
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, poly, tri.color_bgr, lineType=cv2.LINE_AA)
        if tri.alpha >= 1.0:
            cv2.fillConvexPoly(frame, poly, tri.color_bgr, lineType=cv2.LINE_AA)
        else:
            cv2.addWeighted(overlay, tri.alpha, frame, 1.0 - tri.alpha, 0, dst=frame)


def render_object_pose_overlay(
    frame,
    *,
    T_cam: np.ndarray,
    mesh_verts_local: np.ndarray,
    mesh_faces: np.ndarray,
    K: np.ndarray,
    mesh_color_bgr: tuple[int, int, int] = (80, 200, 80),
    mesh_alpha: float = 1.0,
    bbox_color_bgr: tuple[int, int, int] = (0, 255, 255),
    face_stride: int = 1,
    draw_mesh: bool = True,
    draw_bbox: bool = True,
    draw_axes: bool = True,
) -> None:
    verts_local = np.asarray(mesh_verts_local, dtype=np.float64)
    verts_cam = transform_points(T_cam, verts_local)
    h, w = frame.shape[:2]

    if draw_mesh:
        tris = collect_mesh_triangles(
            verts_cam,
            mesh_faces,
            K,
            color_bgr=mesh_color_bgr,
            alpha=mesh_alpha,
            stride=face_stride,
        )
        draw_sorted_triangles(frame, tris, w=w, h=h)

    if draw_bbox:
        corners_cam = transform_points(T_cam, axis_aligned_bbox_corners(verts_local))
        draw_bbox_wireframe(frame, corners_cam, K, color_bgr=bbox_color_bgr, thickness=2)

    if draw_axes:
        extent = float(np.linalg.norm(verts_local.max(axis=0) - verts_local.min(axis=0)))
        axis_len = max(extent * 0.45, 0.03)
        draw_coord_axes(frame, T_cam, K, axis_len=axis_len, thickness=3)
