#!/usr/bin/env python3
"""Visualize a final fused reconstruction package.

Default output is a fast raw-video overlay with semi-transparent hand MANO,
hand skeleton, object mesh, and axes. Pass ``--layout side-by-side`` for the
heavier raw-overlay + reconstruction-only diagnostic view.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECON_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RECON_ROOT.parent
HAWOR_ROOT = REPO_ROOT / "third_party" / "hawor"
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.dataset import resolve_video_job  # noqa: E402
from _common.mesh_viz import (  # noqa: E402
    TriangleLayer,
    collect_mesh_triangles,
    draw_coord_axes,
    draw_sorted_triangles,
    face_stride as mesh_face_stride,
    project_cam,
    transform_points,
)
from _common.paths import final_video_dir, resolve_repo_path  # noqa: E402
from _common.viz import _finalize_h264, _open_bgr_writer  # noqa: E402

FINAL_NPZ = "world_fused.npz"
FINAL_MESH = "object_mesh_scaled_final.obj"
FINAL_SCHEMA_VERSION = "recon_world_v4"
DEFAULT_HAWOR_FPS = 30.0

OBJECT_MESH_BGR = (80, 200, 80)
OBJECT_PALETTE_BGR = (
    (80, 200, 80),
    (60, 180, 255),
    (220, 120, 80),
    (180, 100, 220),
    (80, 220, 220),
    (220, 180, 80),
    (120, 220, 120),
    (220, 120, 180),
    (120, 160, 255),
)
LEFT_HAND_BGR = (100, 220, 80)
RIGHT_HAND_BGR = (30, 120, 255)
WORLD_AXIS_LEN_SCALE = 0.18
MANO_HAND_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True)
class ObjectReconstruction:
    object_id: str
    mesh_path: Path
    frame_indices: np.ndarray
    ob_in_cam: np.ndarray
    color_bgr: tuple[int, int, int]


@dataclass(frozen=True)
class FinalReconstruction:
    dataset: str
    video_id: str
    num_frames: int
    k_mat: np.ndarray
    c2w: np.ndarray
    c2w_vipe_world: np.ndarray
    t_gravity_z_up_from_vipe_world: np.ndarray
    object_frame_indices: np.ndarray
    object_ob_in_cam: np.ndarray
    mesh_path: Path
    objects: tuple[ObjectReconstruction, ...]
    hand_params: dict[str, np.ndarray]
    hand_coordinate_frame: str
    hand_source_coordinate_frame: str
    schema_version: str


@dataclass(frozen=True)
class HandMeshSequence:
    faces_left: np.ndarray
    faces_right: np.ndarray
    left_verts_world: np.ndarray
    right_verts_world: np.ndarray
    left_joints_world: np.ndarray
    right_joints_world: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray


@contextlib.contextmanager
def _hawor_runtime():
    prev_cwd = os.getcwd()
    root = str(HAWOR_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _hawor_sample(values: np.ndarray, frame_idx: int, num_video_frames: int, video_fps: float) -> tuple[int, int, float]:
    values = np.asarray(values)
    if values.shape[0] == num_video_frames:
        return frame_idx, frame_idx, 0.0
    if num_video_frames <= 1 or values.shape[0] <= 1:
        idx = min(frame_idx, values.shape[0] - 1)
        return idx, idx, 0.0
    if video_fps > 1e-3:
        t = frame_idx * DEFAULT_HAWOR_FPS / video_fps
    else:
        t = frame_idx * (values.shape[0] - 1) / (num_video_frames - 1)
    t = float(np.clip(t, 0.0, values.shape[0] - 1))
    i0 = int(np.floor(t))
    i1 = min(i0 + 1, values.shape[0] - 1)
    alpha = float(t - i0)
    return i0, i1, alpha


def _sample_sequence(values: np.ndarray, frame_idx: int, num_video_frames: int, video_fps: float) -> np.ndarray:
    values = np.asarray(values)
    i0, i1, alpha = _hawor_sample(values, frame_idx, num_video_frames, video_fps)
    return (1.0 - alpha) * values[i0] + alpha * values[i1]


def _sample_valid(valid: np.ndarray, frame_idx: int, num_video_frames: int, video_fps: float) -> bool:
    valid = np.asarray(valid).astype(bool)
    i0, i1, alpha = _hawor_sample(valid, frame_idx, num_video_frames, video_fps)
    if alpha <= 1e-6:
        return bool(valid[i0])
    if alpha >= 1.0 - 1e-6:
        return bool(valid[i1])
    return bool(valid[i0] and valid[i1])


def _load_final_reconstruction(dataset: str, video_id: str, final_dir: Path | None) -> FinalReconstruction:
    out_dir = final_dir.resolve() if final_dir is not None else final_video_dir(dataset, video_id)
    npz_path = out_dir / FINAL_NPZ
    mesh_path = out_dir / FINAL_MESH
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing final reconstruction: {npz_path}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing final object mesh: {mesh_path}")

    hand_keys = (
        "hand_trans",
        "hand_rot",
        "hand_pose",
        "hand_betas",
        "hand_valid",
        "hand_trans_hawor_world",
        "hand_rot_hawor_world",
    )
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"dataset", "video_id", "num_frames", "K", "c2w", "object_frame_indices", "object_ob_in_cam"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"Final NPZ missing required keys: {missing}")
        schema_version = str(data["schema_version"].item()) if "schema_version" in data.files else ""
        if schema_version != FINAL_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported final schema {schema_version}; expected {FINAL_SCHEMA_VERSION}. "
                "Re-run fuse after the world-frame definition update."
            )
        if {"T_gravity_z_up_from_vipe_world", "gravity_direction_vipe_world", "up_direction_vipe_world"}.issubset(data.files):
            r_zup = np.asarray(data["T_gravity_z_up_from_vipe_world"], dtype=np.float64)[:3, :3]
            gravity = np.asarray(data["gravity_direction_vipe_world"], dtype=np.float64)
            up = np.asarray(data["up_direction_vipe_world"], dtype=np.float64)
            if not np.allclose(r_zup @ gravity, np.array([0.0, 0.0, -1.0]), atol=1e-5):
                raise RuntimeError("Final gravity alignment does not map physical gravity to -Z")
            if not np.allclose(r_zup @ up, np.array([0.0, 0.0, 1.0]), atol=1e-5):
                raise RuntimeError("Final gravity alignment does not map up direction to +Z")
        hand_params = {key: np.array(data[key]) for key in hand_keys if key in data.files}
        if {"object_ids", "object_mesh_filenames", "object_frame_indices_all", "object_ob_in_cam_all", "object_valid_all"}.issubset(
            data.files
        ):
            object_ids = [str(item) for item in np.asarray(data["object_ids"])]
            mesh_filenames = [str(item) for item in np.asarray(data["object_mesh_filenames"])]
            frame_indices_all = np.asarray(data["object_frame_indices_all"], dtype=np.int64)
            ob_in_cam_all = np.asarray(data["object_ob_in_cam_all"], dtype=np.float64)
            valid_all = np.asarray(data["object_valid_all"]).astype(bool)
            objects = []
            for obj_idx, object_id in enumerate(object_ids):
                valid = valid_all[obj_idx]
                obj_mesh = out_dir / mesh_filenames[obj_idx]
                if not obj_mesh.is_file():
                    raise FileNotFoundError(f"Missing final mesh for {object_id}: {obj_mesh}")
                objects.append(
                    ObjectReconstruction(
                        object_id=object_id,
                        mesh_path=obj_mesh,
                        frame_indices=frame_indices_all[obj_idx][valid],
                        ob_in_cam=ob_in_cam_all[obj_idx][valid],
                        color_bgr=OBJECT_PALETTE_BGR[obj_idx % len(OBJECT_PALETTE_BGR)],
                    )
                )
        else:
            objects = [
                ObjectReconstruction(
                    object_id="object_0",
                    mesh_path=mesh_path,
                    frame_indices=np.array(data["object_frame_indices"], dtype=np.int64),
                    ob_in_cam=np.array(data["object_ob_in_cam"], dtype=np.float64),
                    color_bgr=OBJECT_MESH_BGR,
                )
            ]
        return FinalReconstruction(
            dataset=str(data["dataset"].item()),
            video_id=str(data["video_id"].item()),
            num_frames=int(data["num_frames"]),
            k_mat=np.array(data["K"], dtype=np.float64),
            c2w=np.array(data["c2w"], dtype=np.float64),
            c2w_vipe_world=np.array(data["c2w_vipe_world"], dtype=np.float64)
            if "c2w_vipe_world" in data.files
            else np.array(data["c2w"], dtype=np.float64),
            t_gravity_z_up_from_vipe_world=np.array(data["T_gravity_z_up_from_vipe_world"], dtype=np.float64)
            if "T_gravity_z_up_from_vipe_world" in data.files
            else np.eye(4, dtype=np.float64),
            object_frame_indices=np.array(data["object_frame_indices"], dtype=np.int64),
            object_ob_in_cam=np.array(data["object_ob_in_cam"], dtype=np.float64),
            mesh_path=mesh_path,
            objects=tuple(objects),
            hand_params=hand_params,
            hand_coordinate_frame=str(data["hand_coordinate_frame"].item())
            if "hand_coordinate_frame" in data.files
            else "hawor_world",
            hand_source_coordinate_frame=str(data["hand_source_coordinate_frame"].item())
            if "hand_source_coordinate_frame" in data.files
            else "",
            schema_version=schema_version,
        )


def _compute_hand_mesh_sequence(hand_params: dict[str, np.ndarray]) -> HandMeshSequence | None:
    needed = {"hand_trans", "hand_rot", "hand_pose", "hand_betas", "hand_valid"}
    if not needed.issubset(hand_params):
        return None
    try:
        with _hawor_runtime():
            import torch
            from hawor.utils.process import get_mano_faces, run_mano, run_mano_left

            use_cuda = bool(torch.cuda.is_available())
            print(
                f"[visualize] building MANO hand meshes on {'cuda' if use_cuda else 'cpu'}",
                flush=True,
            )
            pred_trans = torch.as_tensor(hand_params["hand_trans"])
            pred_rot = torch.as_tensor(hand_params["hand_rot"])
            pred_pose = torch.as_tensor(hand_params["hand_pose"])
            pred_betas = torch.as_tensor(hand_params["hand_betas"])
            hand_t = int(pred_trans.shape[1])
            full_idx = torch.arange(hand_t, dtype=torch.long)

            left = run_mano_left(
                pred_trans[0:1, full_idx],
                pred_rot[0:1, full_idx],
                pred_pose[0:1, full_idx],
                betas=pred_betas[0:1, full_idx],
                use_cuda=use_cuda,
            )
            right = run_mano(
                pred_trans[1:2, full_idx],
                pred_rot[1:2, full_idx],
                pred_pose[1:2, full_idx],
                betas=pred_betas[1:2, full_idx],
                use_cuda=use_cuda,
            )
            faces = get_mano_faces()
    except Exception as exc:  # noqa: BLE001 - visualization can still render object-only output
        print(f"Warning: MANO hand geometry unavailable: {exc}", file=sys.stderr, flush=True)
        return None

    faces_new = np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ],
        dtype=np.int64,
    )
    faces_right = np.concatenate([np.asarray(faces, dtype=np.int64), faces_new], axis=0)
    return HandMeshSequence(
        faces_left=faces_right[:, [0, 2, 1]],
        faces_right=faces_right,
        left_verts_world=_to_numpy(left["vertices"][0]),
        right_verts_world=_to_numpy(right["vertices"][0]),
        left_joints_world=_to_numpy(left["joints"][0]),
        right_joints_world=_to_numpy(right["joints"][0]),
        left_valid=np.asarray(hand_params["hand_valid"][0]).astype(bool),
        right_valid=np.asarray(hand_params["hand_valid"][1]).astype(bool),
    )


def _world_to_cam(c2w: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    return transform_points(np.linalg.inv(c2w), points_world)


def _transform_hand_sequence_world(hands: HandMeshSequence, transform: np.ndarray) -> HandMeshSequence:
    return HandMeshSequence(
        faces_left=hands.faces_left,
        faces_right=hands.faces_right,
        left_verts_world=transform_points(transform, hands.left_verts_world.reshape(-1, 3)).reshape(
            hands.left_verts_world.shape
        ),
        right_verts_world=transform_points(transform, hands.right_verts_world.reshape(-1, 3)).reshape(
            hands.right_verts_world.shape
        ),
        left_joints_world=transform_points(transform, hands.left_joints_world.reshape(-1, 3)).reshape(
            hands.left_joints_world.shape
        ),
        right_joints_world=transform_points(transform, hands.right_joints_world.reshape(-1, 3)).reshape(
            hands.right_joints_world.shape
        ),
        left_valid=hands.left_valid,
        right_valid=hands.right_valid,
    )


def _draw_hand_skeleton(frame, joints_cam: np.ndarray, k_mat: np.ndarray, *, color_bgr: tuple[int, int, int]) -> None:
    import cv2

    joints = np.asarray(joints_cam, dtype=np.float64)[:21]
    uv, z = project_cam(k_mat, joints)
    h, w = frame.shape[:2]
    for i0, i1 in MANO_HAND_PAIRS:
        if z[i0] <= 1e-4 or z[i1] <= 1e-4:
            continue
        p0 = (int(round(uv[i0, 0])), int(round(uv[i0, 1])))
        p1 = (int(round(uv[i1, 0])), int(round(uv[i1, 1])))
        cv2.line(frame, p0, p1, color_bgr, 2, cv2.LINE_AA)
    for i in range(len(joints)):
        if z[i] <= 1e-4:
            continue
        p = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(frame, p, 3, color_bgr, -1, cv2.LINE_AA)


def _hand_axis_transform(joints_cam: np.ndarray) -> np.ndarray | None:
    joints = np.asarray(joints_cam, dtype=np.float64)
    if joints.shape[0] < 18:
        return None
    origin = joints[0]
    x_axis = joints[9] - origin
    y_axis = joints[5] - joints[17]
    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)
    if x_norm < 1e-6 or y_norm < 1e-6:
        return None
    x_axis = x_axis / x_norm
    y_axis = y_axis - x_axis * float(np.dot(x_axis, y_axis))
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        return None
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return None
    z_axis = z_axis / z_norm
    t_cam = np.eye(4, dtype=np.float64)
    t_cam[:3, 0] = x_axis
    t_cam[:3, 1] = y_axis
    t_cam[:3, 2] = z_axis
    t_cam[:3, 3] = origin
    return t_cam


def _draw_hand_axes(frame, joints_cam: np.ndarray, k_mat: np.ndarray, axis_len: float) -> None:
    t_cam = _hand_axis_transform(joints_cam)
    if t_cam is not None:
        draw_coord_axes(frame, t_cam, k_mat, axis_len=axis_len, thickness=2)


def _draw_triangles_fast(frame, triangles: list[TriangleLayer], *, w: int, h: int) -> None:
    """Fast preview rasterizer: batch fills by alpha and blend once per group."""
    import cv2

    if not triangles:
        return
    groups: dict[float, list[TriangleLayer]] = {}
    for tri in triangles:
        groups.setdefault(round(float(tri.alpha), 2), []).append(tri)
    for alpha, group in sorted(groups.items(), reverse=True):
        overlay = frame.copy()
        for tri in group:
            poly = np.round(tri.uv).astype(np.int32)
            if (
                np.all((poly[:, 0] < 0) | (poly[:, 0] >= w))
                or np.all((poly[:, 1] < 0) | (poly[:, 1] >= h))
            ):
                continue
            cv2.fillConvexPoly(overlay, poly, tri.color_bgr, lineType=cv2.LINE_8)
        cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0, dst=frame)


def _draw_projected_points(
    frame,
    points_cam: np.ndarray,
    k_mat: np.ndarray,
    *,
    color_bgr: tuple[int, int, int],
    alpha: float,
    stride: int,
    radius: int,
) -> None:
    import cv2

    pts = np.asarray(points_cam, dtype=np.float64)[:: max(1, stride)]
    if pts.size == 0:
        return
    uv, z = project_cam(k_mat, pts)
    h, w = frame.shape[:2]
    valid = (
        (pts[:, 2] > 1e-4)
        & (uv[:, 0] >= -radius)
        & (uv[:, 0] < w + radius)
        & (uv[:, 1] >= -radius)
        & (uv[:, 1] < h + radius)
    )
    if not np.any(valid):
        return
    uv = np.round(uv[valid]).astype(np.int32)
    z = z[valid]
    order = np.argsort(-z)
    overlay = frame.copy()
    for idx in order:
        cv2.circle(overlay, tuple(uv[idx]), radius, color_bgr, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)


def _load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


@dataclass(frozen=True)
class ObjectRenderState:
    object_id: str
    verts: np.ndarray
    faces: np.ndarray
    face_stride: int
    pose_by_frame: dict[int, np.ndarray]
    color_bgr: tuple[int, int, int]


def _draw_frame(
    *,
    frame,
    recon: FinalReconstruction,
    frame_idx: int,
    object_states: tuple[ObjectRenderState, ...],
    hands: HandMeshSequence | None,
    scene_axis_len: float,
    hand_axis_len: float,
    layout: str,
    fast_raster: bool,
    hand_face_stride: int,
    object_point_stride: int,
    hand_point_stride: int,
    point_radius: int,
    hand_c2w: np.ndarray,
    hand_video_fps: float,
    draw_world_axis: bool,
    surface_mode: str,
) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    overlay = frame.copy()
    recon_only = np.full_like(frame, 18) if layout == "side-by-side" else None
    k_mat = recon.k_mat
    c2w = recon.c2w[min(frame_idx, len(recon.c2w) - 1)]
    hand_c2w_i = hand_c2w[min(frame_idx, len(hand_c2w) - 1)]
    w2c = np.linalg.inv(c2w)

    overlay_triangles: list[TriangleLayer] = []
    recon_triangles: list[TriangleLayer] = []
    object_poses_this_frame = []
    for obj_state in object_states:
        if frame_idx not in obj_state.pose_by_frame:
            continue
        t_obj_cam = obj_state.pose_by_frame[frame_idx]
        object_poses_this_frame.append(t_obj_cam)
        obj_cam = transform_points(t_obj_cam, obj_state.verts)
        if surface_mode == "points":
            _draw_projected_points(
                overlay,
                obj_cam,
                k_mat,
                color_bgr=obj_state.color_bgr,
                alpha=0.55,
                stride=object_point_stride,
                radius=point_radius,
            )
            if recon_only is not None:
                _draw_projected_points(
                    recon_only,
                    obj_cam,
                    k_mat,
                    color_bgr=obj_state.color_bgr,
                    alpha=0.9,
                    stride=object_point_stride,
                    radius=point_radius,
                )
        else:
            overlay_triangles.extend(
                collect_mesh_triangles(
                    obj_cam,
                    obj_state.faces,
                    k_mat,
                    color_bgr=obj_state.color_bgr,
                    alpha=0.45,
                    stride=obj_state.face_stride,
                )
            )
            if recon_only is not None:
                recon_triangles.extend(
                    collect_mesh_triangles(
                        obj_cam,
                        obj_state.faces,
                        k_mat,
                        color_bgr=obj_state.color_bgr,
                        alpha=0.85,
                        stride=obj_state.face_stride,
                    )
                )

    hand_items: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]] = []
    if hands is not None:
        if _sample_valid(hands.left_valid, frame_idx, recon.num_frames, hand_video_fps):
            left_verts = _sample_sequence(hands.left_verts_world, frame_idx, recon.num_frames, hand_video_fps)
            left_joints = _sample_sequence(hands.left_joints_world, frame_idx, recon.num_frames, hand_video_fps)
            hand_items.append(
                (
                    _world_to_cam(hand_c2w_i, left_verts),
                    _world_to_cam(hand_c2w_i, left_joints),
                    hands.faces_left,
                    LEFT_HAND_BGR,
                )
            )
        if _sample_valid(hands.right_valid, frame_idx, recon.num_frames, hand_video_fps):
            right_verts = _sample_sequence(hands.right_verts_world, frame_idx, recon.num_frames, hand_video_fps)
            right_joints = _sample_sequence(hands.right_joints_world, frame_idx, recon.num_frames, hand_video_fps)
            hand_items.append(
                (
                    _world_to_cam(hand_c2w_i, right_verts),
                    _world_to_cam(hand_c2w_i, right_joints),
                    hands.faces_right,
                    RIGHT_HAND_BGR,
                )
            )

    for verts_cam, _joints_cam, faces, color in hand_items:
        if surface_mode == "points":
            _draw_projected_points(
                overlay,
                verts_cam,
                k_mat,
                color_bgr=color,
                alpha=0.65,
                stride=hand_point_stride,
                radius=max(1, point_radius),
            )
            if recon_only is not None:
                _draw_projected_points(
                    recon_only,
                    verts_cam,
                    k_mat,
                    color_bgr=color,
                    alpha=0.9,
                    stride=hand_point_stride,
                    radius=max(1, point_radius),
                )
        else:
            hand_triangles = collect_mesh_triangles(
                verts_cam,
                faces,
                k_mat,
                color_bgr=color,
                alpha=0.42,
                stride=hand_face_stride,
            )
            overlay_triangles.extend(hand_triangles)
            if recon_only is not None:
                recon_triangles.extend(
                    collect_mesh_triangles(
                        verts_cam,
                        faces,
                        k_mat,
                        color_bgr=color,
                        alpha=0.78,
                        stride=hand_face_stride,
                    )
                )

    if fast_raster:
        _draw_triangles_fast(overlay, overlay_triangles, w=w, h=h)
        if recon_only is not None:
            _draw_triangles_fast(recon_only, recon_triangles, w=w, h=h)
    else:
        draw_sorted_triangles(overlay, overlay_triangles, w=w, h=h)
        if recon_only is not None:
            draw_sorted_triangles(recon_only, recon_triangles, w=w, h=h)

    targets = (overlay,) if recon_only is None else (overlay, recon_only)
    for target in targets:
        for t_obj_cam in object_poses_this_frame:
            draw_coord_axes(target, t_obj_cam, k_mat, axis_len=scene_axis_len, thickness=3)
        if draw_world_axis:
            draw_coord_axes(target, w2c, k_mat, axis_len=scene_axis_len, thickness=3)
        for _verts_cam, joints_cam, _faces, color in hand_items:
            _draw_hand_skeleton(target, joints_cam, k_mat, color_bgr=color)
            _draw_hand_axes(target, joints_cam, k_mat, axis_len=hand_axis_len)
        cv2.putText(
            target,
            f"f{frame_idx}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if recon_only is None:
        return overlay
    return np.concatenate([overlay, recon_only], axis=1)


def visualize_final_reconstruction(
    *,
    recon: FinalReconstruction,
    video_path: Path,
    out_path: Path,
    max_frames: int | None,
    frame_stride: int,
    no_hands: bool,
    layout: str,
    fast_raster: bool,
    object_max_triangles: int,
    hand_face_stride: int,
    object_point_stride: int,
    hand_point_stride: int,
    point_radius: int,
    progress_every: int,
    hand_source: str,
    draw_world_axis: bool,
    surface_mode: str,
) -> Path:
    import cv2

    object_states = []
    object_extents = []
    for obj in recon.objects:
        obj_verts, obj_faces = _load_mesh(obj.mesh_path)
        obj_stride = mesh_face_stride(len(obj_faces), max_triangles=object_max_triangles)
        obj_pose_by_frame = {
            int(frame_idx): np.asarray(pose, dtype=np.float64)
            for frame_idx, pose in zip(obj.frame_indices, obj.ob_in_cam)
        }
        if len(obj_verts):
            object_extents.append(float(np.linalg.norm(obj_verts.max(axis=0) - obj_verts.min(axis=0))))
        object_states.append(
            ObjectRenderState(
                object_id=obj.object_id,
                verts=obj_verts,
                faces=obj_faces,
                face_stride=obj_stride,
                pose_by_frame=obj_pose_by_frame,
                color_bgr=obj.color_bgr,
            )
        )
    obj_extent = max(object_extents) if object_extents else 0.1
    scene_axis_len = max(obj_extent * WORLD_AXIS_LEN_SCALE, 0.04)
    hand_axis_len = max(scene_axis_len * 0.55, 0.025)
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot open video: {video_path}")
    hand_params = dict(recon.hand_params)
    hand_c2w = recon.c2w
    transform_hands_to_final = False
    if hand_source == "raw-hawor" and {"hand_trans_hawor_world", "hand_rot_hawor_world"}.issubset(hand_params):
        hand_params["hand_trans"] = hand_params["hand_trans_hawor_world"]
        hand_params["hand_rot"] = hand_params["hand_rot_hawor_world"]
        hand_c2w = recon.c2w_vipe_world
        print("[visualize] hand source: raw HaWoR globals with ViPE camera", flush=True)
    elif {"hand_trans_hawor_world", "hand_rot_hawor_world"}.issubset(hand_params):
        hand_params["hand_trans"] = hand_params["hand_trans_hawor_world"]
        hand_params["hand_rot"] = hand_params["hand_rot_hawor_world"]
        transform_hands_to_final = True
        print("[visualize] hand source: raw HaWoR MANO geometry transformed to final z-up world", flush=True)
    else:
        print("[visualize] hand source: final fused globals", flush=True)

    if not no_hands and {"hand_valid"}.issubset(hand_params):
        valid = np.asarray(hand_params["hand_valid"])
        print(
            f"[visualize] hand valid frames: left={int(valid[0].sum())}, right={int(valid[1].sum())}; "
            f"video_fps={fps:.3f}",
            flush=True,
        )
    hands = None if no_hands else _compute_hand_mesh_sequence(hand_params)
    if hands is not None and transform_hands_to_final:
        hands = _transform_hand_sequence_world(hands, recon.t_gravity_z_up_from_vipe_world)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_width = width * 2 if layout == "side-by-side" else width
    out_fps = fps / max(1, frame_stride)
    print(
        f"[visualize] source_fps={fps:.3f}, frame_stride={frame_stride}, output_fps={out_fps:.3f}, "
        f"max_frames={max_frames if max_frames is not None else 'all'}, objects={len(object_states)}",
        flush=True,
    )
    writer, raw_path = _open_bgr_writer(out_path, out_fps, out_width, height)

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride == 0:
            vis = _draw_frame(
                frame=frame,
                recon=recon,
                frame_idx=frame_idx,
                object_states=tuple(object_states),
                hands=hands,
                scene_axis_len=scene_axis_len,
                hand_axis_len=hand_axis_len,
                layout=layout,
                fast_raster=fast_raster,
                hand_face_stride=hand_face_stride,
                object_point_stride=object_point_stride,
                hand_point_stride=hand_point_stride,
                point_radius=point_radius,
                hand_c2w=hand_c2w,
                hand_video_fps=fps,
                draw_world_axis=draw_world_axis,
                surface_mode=surface_mode,
            )
            writer.write(vis)
            written += 1
            if progress_every > 0 and written % progress_every == 0:
                print(
                    f"[visualize] wrote {written} frame(s) "
                    f"(source frame {frame_idx}/{recon.num_frames - 1})",
                    flush=True,
                )
            if max_frames is not None and written >= max_frames:
                break
        frame_idx += 1

    cap.release()
    writer.release()
    if written == 0:
        raw_path.unlink(missing_ok=True)
        raise RuntimeError("No frames were written")
    return _finalize_h264(raw_path, out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--video-list", type=Path, default=None, help="Text file of video ids/paths to visualize")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None, help="Raw video path; auto-resolved for known datasets")
    parser.add_argument("--final-dir", type=Path, default=None, help="Override final reconstruction directory")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-hands", action="store_true", help="Render object/camera only without MANO geometry")
    parser.add_argument(
        "--layout",
        choices=("overlay", "side-by-side"),
        default="overlay",
        help="overlay is fast/default; side-by-side also renders the reconstruction-only diagnostic view",
    )
    parser.add_argument(
        "--accurate-raster",
        action="store_true",
        help="Use slower per-triangle depth sorting and alpha blending instead of the fast preview rasterizer",
    )
    parser.add_argument("--object-max-triangles", type=int, default=1200)
    parser.add_argument("--hand-face-stride", type=int, default=4)
    parser.add_argument(
        "--surface-mode",
        choices=("points", "mesh"),
        default="points",
        help="Render object and MANO surfaces as projected point clouds by default; use mesh for triangular overlays",
    )
    parser.add_argument("--object-point-stride", type=int, default=8)
    parser.add_argument("--hand-point-stride", type=int, default=1)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--hand-source",
        choices=("raw-hawor", "final"),
        default="final",
        help="raw-hawor projects original HaWoR MANO globals with the ViPE camera; final uses fused z-up hand globals",
    )
    parser.add_argument("--draw-world-axis", action="store_true", help="Draw the global world-frame axes on the overlay")
    args = parser.parse_args(argv)

    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.object_max_triangles < 1:
        parser.error("--object-max-triangles must be >= 1")
    if args.hand_face_stride < 1:
        parser.error("--hand-face-stride must be >= 1")
    if args.object_point_stride < 1:
        parser.error("--object-point-stride must be >= 1")
    if args.hand_point_stride < 1:
        parser.error("--hand-point-stride must be >= 1")
    if args.point_radius < 1:
        parser.error("--point-radius must be >= 1")
    if args.video_id is None and args.video_list is None:
        parser.error("one of --video-id or --video-list is required")
    if args.video is not None and args.video_list is not None:
        parser.error("--video cannot be combined with --video-list")
    if args.output is not None and args.video_list is not None:
        parser.error("--output cannot be combined with --video-list")
    if args.final_dir is not None and args.video_list is not None:
        parser.error("--final-dir cannot be combined with --video-list")

    if args.video_list is not None:
        entries = [line.strip() for line in resolve_repo_path(args.video_list).read_text().splitlines() if line.strip()]
    else:
        entries = [args.video_id]

    outputs = []
    for idx, entry in enumerate(entries, start=1):
        video_id = resolve_video_job(args.dataset, entry, dataset_root=args.dataset_root).video_id
        print(f"[visualize] video {idx}/{len(entries)}: {video_id}", flush=True)
        recon = _load_final_reconstruction(args.dataset, video_id, args.final_dir)
        if args.video is not None:
            video_path = resolve_repo_path(args.video)
        else:
            video_path = resolve_video_job(args.dataset, video_id, dataset_root=args.dataset_root).video_path
        out_dir = args.final_dir.resolve() if args.final_dir is not None else final_video_dir(args.dataset, video_id)
        if args.output is not None:
            out_path = args.output
        else:
            sampled = args.max_frames is not None or args.frame_stride != 1
            suffix = (
                f"_sampled_stride{args.frame_stride}"
                + (f"_n{args.max_frames}" if args.max_frames is not None else "")
                if sampled
                else ""
            )
            out_path = out_dir / "vis" / f"{video_id}_final_reconstruction{suffix}.mp4"
        out = visualize_final_reconstruction(
            recon=recon,
            video_path=video_path,
            out_path=out_path,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            no_hands=args.no_hands,
            layout=args.layout,
            fast_raster=not args.accurate_raster,
            object_max_triangles=args.object_max_triangles,
            hand_face_stride=args.hand_face_stride,
            object_point_stride=args.object_point_stride,
            hand_point_stride=args.hand_point_stride,
            point_radius=args.point_radius,
            progress_every=args.progress_every,
            hand_source=args.hand_source,
            draw_world_axis=args.draw_world_axis,
            surface_mode=args.surface_mode,
        )
        outputs.append(out)
        print(f"Wrote final reconstruction visualization: {out}", flush=True)
    if len(outputs) > 1:
        print(f"[visualize] completed {len(outputs)} video(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
