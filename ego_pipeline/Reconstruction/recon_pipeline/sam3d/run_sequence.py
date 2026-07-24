#!/usr/bin/env python3
"""Run SAM3D mesh reconstruction from object mask at the label frame."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parents[1]
SAM3D_DIR = Path(__file__).resolve().parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))
if str(SAM3D_DIR) not in sys.path:
    sys.path.insert(0, str(SAM3D_DIR))
SAM2_OBJECT_DIR = RECON_ROOT / "sam2_object"
if str(SAM2_OBJECT_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_OBJECT_DIR))

from _common.artifacts import discard_sam3d_nonessential  # noqa: E402
from _common.dataset import VideoJob  # noqa: E402
from _common.io import read_video_frame  # noqa: E402
from _common.paths import interim_step_dir, is_step_complete, write_step_completion  # noqa: E402
from _common.viz import encode_sam3d_vis_png, vis_png_path  # noqa: E402

SAM3D_ROOT = RECON_ROOT.parent / "third_party" / "sam-3d-objects"
DEFAULT_CONFIG = SAM3D_ROOT / "checkpoints" / "hf" / "pipeline.yaml"


def _load_label_prompt(sam2_object_dir: Path) -> dict:
    from sam2_object_common import load_label_prompt  # noqa: WPS433

    return load_label_prompt(sam2_object_dir).to_json()


def _load_label_frame(sam2_object_dir: Path) -> int:
    from sam2_object_common import load_label_prompt  # noqa: WPS433

    return int(load_label_prompt(sam2_object_dir).frame_idx)


def _object_prompts(sam2_object_dir: Path):
    from sam2_object_common import load_label_prompt  # noqa: WPS433

    return load_label_prompt(sam2_object_dir).objects


def _load_mask(sam2_object_dir: Path, frame_idx: int, object_id: str = "object_0") -> "np.ndarray":
    import numpy as np

    from sam2_object_common import object_mask_filename  # noqa: WPS433

    mask_path = (
        sam2_object_dir
        / "video_segmentation"
        / "masks"
        / f"frame_{frame_idx:06d}_masks"
        / object_mask_filename(object_id)
    )
    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    return mask > 127


def _mesh_from_sam3d_output(output: dict):
    """Convert SAM3D pipeline output to a trimesh.Trimesh."""
    import trimesh

    glb = output.get("glb")
    if glb is not None:
        if isinstance(glb, trimesh.Scene):
            return trimesh.util.concatenate(tuple(glb.geometry.values()))
        return glb

    mesh_out = output.get("mesh")
    if isinstance(mesh_out, list):
        mesh_out = mesh_out[0] if mesh_out else None
    if mesh_out is not None and hasattr(mesh_out, "vertices") and hasattr(mesh_out, "faces"):
        verts = mesh_out.vertices.detach().cpu().numpy()
        faces = mesh_out.faces.detach().cpu().numpy()
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    raise RuntimeError(f"Unsupported SAM3D output keys/types: {list(output.keys())}")


def _keep_clicked_component(mask, points):
    import cv2
    import numpy as np

    binary = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)

    chosen = None
    height, width = mask.shape[:2]
    for point in points or []:
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            label = int(labels[y, x])
            if label > 0:
                chosen = label
                break

    if chosen is None:
        chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == chosen


def _load_hand_mask(step_dir: Path, video_id: str, frame_idx: int, shape) -> "np.ndarray":
    import cv2
    import numpy as np

    hand_dir = (
        step_dir.parent
        / "sam3_hands"
        / video_id
        / "video_segmentation"
        / "masks"
        / f"frame_{frame_idx:06d}_masks"
    )
    hand = np.zeros(shape, dtype=bool)
    for name in ("left_hand_0.png", "right_hand_0.png"):
        path = hand_dir / name
        if not path.is_file():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        hand |= mask > 127
    return hand


def _clean_mask_for_depth(mask, points, step_dir: Path, video_id: str, frame_idx: int):
    import cv2

    component = _keep_clicked_component(mask, points)
    hand = _load_hand_mask(step_dir, video_id, frame_idx, component.shape)
    cleaned = component & ~hand
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(cleaned.astype("uint8"), kernel, iterations=1) > 0
    if int(eroded.sum()) >= 50:
        cleaned = eroded
    return cleaned, component, hand


def _backproject_mask_depth(depth, mask, k_mat):
    import numpy as np

    fx, fy = float(k_mat[0, 0]), float(k_mat[1, 1])
    cx, cy = float(k_mat[0, 2]), float(k_mat[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid intrinsics for scale estimation: fx={fx}, fy={fy}")

    valid = mask & np.isfinite(depth) & (depth > 1e-4)
    if not np.any(valid):
        raise ValueError("No valid positive depth pixels inside cleaned object mask")
    vals = depth[valid]
    lo, hi = np.percentile(vals, [5, 95])
    valid &= (depth >= lo) & (depth <= hi)
    if int(valid.sum()) < 20:
        raise ValueError("Too few valid depth pixels after percentile filtering")

    v, u = np.where(valid)
    z = depth[v, u].astype(np.float64)
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    if len(points) > 50000:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), size=50000, replace=False)]
    return points, valid


def _robust_center(points):
    import numpy as np

    return np.median(np.asarray(points, dtype=np.float64), axis=0)


def _robust_extent(points):
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    lo, hi = np.percentile(pts, [5, 95], axis=0)
    return hi - lo


def _save_pointcloud_ply(path: Path, points) -> None:
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def _rotation_from_sam3d_pose(sam3d_pose: dict):
    import numpy as np

    quat = sam3d_pose.get("rotation")
    if quat is None:
        return np.eye(3, dtype=np.float64), "identity_no_sam3d_rotation"
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.size != 4 or not np.all(np.isfinite(q)):
        return np.eye(3, dtype=np.float64), "identity_invalid_sam3d_rotation"
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64), "identity_degenerate_sam3d_rotation"
    q = q / norm
    w, x, y, z = q
    r_ref = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return r_ref, "sam3d_rotation_quaternion_wxyz"


def _principal_axis_length(points):
    import numpy as np

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 20:
        raise RuntimeError(f"Need at least 20 points for PCA length, got {len(pts)}")
    center = _robust_center(pts)
    centered = pts - center
    lo, hi = np.percentile(centered, [2.5, 97.5], axis=0)
    keep = np.all((centered >= lo) & (centered <= hi), axis=1)
    robust = centered[keep]
    if len(robust) < 20:
        robust = centered
    cov = np.cov(robust.T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    axis /= max(np.linalg.norm(axis), 1e-12)
    proj = centered @ axis
    length = float(np.percentile(proj, 95) - np.percentile(proj, 5))
    if not np.isfinite(length) or length <= 1e-8:
        raise RuntimeError(f"Invalid principal-axis length: {length}")
    return length, axis, center, int(len(robust))


def _project_points(points_cam, k_mat):
    import numpy as np

    proj = (k_mat @ points_cam.T).T
    z = np.maximum(proj[:, 2], 1e-6)
    proj[:, 0] /= z
    proj[:, 1] /= z
    return proj


def _extract_visible_mesh_surface(mesh, r_ref, obs_center, mesh_center, k_mat, image_shape):
    import numpy as np
    import trimesh

    sample_count = min(100000, max(20000, len(mesh.vertices)))
    samples, face_idx = trimesh.sample.sample_surface(mesh, sample_count)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_idx]
    samples = np.asarray(samples, dtype=np.float64)
    normals_cam = (r_ref @ normals.T).T
    points_cam = (r_ref @ (samples - mesh_center).T).T + obs_center

    view_dir = -points_cam
    facing = np.einsum("ij,ij->i", normals_cam, view_dir) > 0
    proj = _project_points(points_cam, k_mat)
    h, w = image_shape[:2]
    inside = (
        facing
        & (proj[:, 2] > 1e-4)
        & (proj[:, 0] >= 0)
        & (proj[:, 0] < w)
        & (proj[:, 1] >= 0)
        & (proj[:, 1] < h)
    )
    if int(inside.sum()) < 100:
        inside = (
            (proj[:, 2] > 1e-4)
            & (proj[:, 0] >= 0)
            & (proj[:, 0] < w)
            & (proj[:, 1] >= 0)
            & (proj[:, 1] < h)
        )
    if int(inside.sum()) < 100:
        raise RuntimeError(f"Visible mesh extraction failed: only {int(inside.sum())} projected samples")

    idx = np.where(inside)[0]
    uv = np.rint(proj[idx, :2]).astype(np.int64)
    linear = uv[:, 1] * w + uv[:, 0]
    z = proj[idx, 2]
    order = np.lexsort((z, linear))
    sorted_idx = idx[order]
    sorted_linear = linear[order]
    keep = np.ones(len(sorted_idx), dtype=bool)
    keep[1:] = sorted_linear[1:] != sorted_linear[:-1]
    visible_idx = sorted_idx[keep]
    if len(visible_idx) < 100:
        raise RuntimeError(f"Visible mesh z-buffer left too few samples: {len(visible_idx)}")
    return samples[visible_idx], points_cam[visible_idx], int(sample_count)


def _estimate_scale_alignment(mesh, obs_points, sam3d_pose: dict | None = None, k_mat=None, image_shape=None):
    import numpy as np

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        raise RuntimeError("SAM3D mesh has no vertices")
    if getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
        raise RuntimeError("SAM3D mesh has no faces")
    if k_mat is None or image_shape is None:
        raise RuntimeError("Camera intrinsics and image shape are required for visible-surface scale estimation")

    r_ref, r_source = _rotation_from_sam3d_pose(sam3d_pose or {})
    c_obs = _robust_center(obs_points)
    c_mesh = _robust_center(verts)
    visible_raw, visible_cam_coarse, sampled_count = _extract_visible_mesh_surface(
        mesh,
        r_ref,
        c_obs,
        c_mesh,
        k_mat,
        image_shape,
    )
    obs_len, obs_axis, _, obs_pca_points = _principal_axis_length(obs_points)
    visible_len, visible_axis, _, visible_pca_points = _principal_axis_length(visible_cam_coarse)
    scale = float(obs_len / visible_len)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid visible-surface principal-axis scale: {scale}")

    warnings = []
    if r_source.startswith("identity"):
        warnings.append("SAM3D orientation unavailable or invalid; used identity orientation.")
    if len(visible_cam_coarse) < 500:
        warnings.append("Visible mesh surface has fewer than 500 points; inspect debug outputs.")
    if scale < 1e-4 or scale > 10.0:
        warnings.append(f"Scale {scale:.6g} is outside the expected sanity range.")

    t_ref = c_obs - scale * (r_ref @ c_mesh)
    v_ref_cam = scale * (r_ref @ verts.T).T + t_ref
    v_metric = scale * verts
    visible_ref_cam = scale * (r_ref @ visible_raw.T).T + t_ref
    return {
        "scale": scale,
        "R_ref": r_ref,
        "R_ref_source": r_source,
        "t_ref": t_ref,
        "mesh_center": c_mesh,
        "obs_center": c_obs,
        "observed_principal_axis_length": obs_len,
        "visible_mesh_principal_axis_length": visible_len,
        "observed_principal_axis": obs_axis,
        "visible_mesh_principal_axis": visible_axis,
        "observed_pca_points": obs_pca_points,
        "visible_mesh_pca_points": visible_pca_points,
        "visible_mesh_points": visible_ref_cam,
        "num_visible_mesh_points": int(len(visible_ref_cam)),
        "num_mesh_surface_samples": sampled_count,
        "warnings": warnings,
        "notes": "Scale = observed visible depth point-cloud principal-axis length / visible SAM3D surface principal-axis length.",
        "v_ref_cam": v_ref_cam,
        "v_metric": v_metric,
    }


def _jsonable_tensor(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _sam3d_pose_metadata(output: dict) -> dict:
    return {
        key: _jsonable_tensor(output.get(key))
        for key in ("rotation", "translation", "scale")
        if key in output
    }


def _write_scale_overlay(
    *,
    rgb_bgr,
    mask,
    clean_mask,
    obs_points,
    mesh,
    v_ref_cam,
    k_mat,
    out_path: Path,
    frame_idx: int,
    scale_meta: dict,
) -> Path:
    import cv2
    import numpy as np

    frame = rgb_bgr.copy()
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    clean_overlay = frame.copy()
    clean_overlay[clean_mask] = (0.65 * clean_overlay[clean_mask] + 0.35 * np.array([0, 170, 0])).astype(np.uint8)
    frame = clean_overlay

    pts = (k_mat @ obs_points.T).T
    z = np.maximum(pts[:, 2], 1e-6)
    uv = pts[:, :2] / z[:, None]
    point_stride = max(1, len(uv) // 6000)
    for x, y in uv[::point_stride].astype(int):
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 0), -1, cv2.LINE_AA)

    proj = (k_mat @ v_ref_cam.T).T
    pz = np.maximum(proj[:, 2], 1e-6)
    proj[:, :2] /= pz[:, None]
    faces = np.asarray(mesh.faces)
    if faces.size:
        face_overlay = frame.copy()
        face_stride = max(1, len(faces) // 5000)
        for tri in faces[::face_stride]:
            if np.any(proj[tri, 2] <= 0):
                continue
            poly = proj[tri, :2].astype(np.int32)
            if (
                np.all((poly[:, 0] < 0) | (poly[:, 0] >= frame.shape[1]))
                or np.all((poly[:, 1] < 0) | (poly[:, 1] >= frame.shape[0]))
            ):
                continue
            cv2.fillConvexPoly(face_overlay, poly, (40, 40, 255), cv2.LINE_AA)
        frame = cv2.addWeighted(face_overlay, 0.28, frame, 0.72, 0)

    edges = mesh.edges_unique
    stride = max(1, len(edges) // 9000)
    for e0, e1 in edges[::stride]:
        if proj[e0, 2] <= 0 or proj[e1, 2] <= 0:
            continue
        p0 = tuple(proj[e0, :2].astype(int))
        p1 = tuple(proj[e1, :2].astype(int))
        cv2.line(frame, p0, p1, (0, 80, 255), 2, cv2.LINE_AA)

    vert_stride = max(1, len(proj) // 7000)
    for x, y, z_val in proj[::vert_stride]:
        if z_val > 0 and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), -1, cv2.LINE_AA)

    lines = [
        f"ref_frame={frame_idx} scale={scale_meta['scale_global']:.5f}",
        "obs_len={:.5f} mesh_visible_len={:.5f}".format(
            scale_meta["observed_principal_axis_length"],
            scale_meta["visible_mesh_principal_axis_length"],
        ),
        f"num_points={scale_meta['num_depth_points']}",
        "cyan=depth points red/orange=projected mesh",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return out_path


def _write_scale_debug_3d(*, obs_points, visible_mesh_points, out_path: Path, scale_meta: dict) -> Path:
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = np.asarray(obs_points, dtype=np.float64)
    vis = np.asarray(visible_mesh_points, dtype=np.float64)
    obs_stride = max(1, len(obs) // 6000)
    vis_stride = max(1, len(vis) // 6000)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(obs[::obs_stride, 0], obs[::obs_stride, 1], obs[::obs_stride, 2], s=2, c="#00d6ff", label="depth points")
    ax.scatter(vis[::vis_stride, 0], vis[::vis_stride, 1], vis[::vis_stride, 2], s=2, c="#ff4b2f", label="visible mesh")
    all_pts = np.concatenate([obs[::obs_stride], vis[::vis_stride]], axis=0)
    center = all_pts.mean(axis=0)
    radius = float(np.max(np.ptp(all_pts, axis=0)) * 0.55)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X cam (m)")
    ax.set_ylabel("Y cam (m)")
    ax.set_zlabel("Z cam (m)")
    ax.set_title(
        "scale={:.5f} | obs_len={:.5f} | mesh_len={:.5f}".format(
            scale_meta["scale_global"],
            scale_meta["observed_principal_axis_length"],
            scale_meta["visible_mesh_principal_axis_length"],
        )
    )
    ax.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def run_sam3d(job: VideoJob, *, gpu: int, visualize: bool, force: bool, config_path: Path) -> dict:
    import cv2
    import numpy as np
    import trimesh

    step_dir = interim_step_dir(job.dataset, job.video_id, "sam3d")
    obj_dir = interim_step_dir(job.dataset, job.video_id, "sam2_object")

    if not force and is_step_complete(step_dir, "sam3d"):
        return {"skipped": True}

    # Remove stale outputs from previous implementations.
    shutil.rmtree(step_dir / "step_scale", ignore_errors=True)
    for stale_name in ("object_mesh_metric.obj",):
        (step_dir / stale_name).unlink(missing_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(SAM3D_ROOT / "notebook"))
    from inference import Inference, load_image, load_single_mask  # type: ignore

    if not config_path.is_file():
        raise FileNotFoundError(
            f"SAM3D config not found: {config_path}. Download checkpoints per third_party/sam-3d-objects README."
        )
    infer = Inference(str(config_path), compile=False)

    object_metas = []
    for obj in _object_prompts(obj_dir):
        object_id = obj.object_id
        obj_step_dir = step_dir / "objects" / object_id
        obj_step_dir.mkdir(parents=True, exist_ok=True)
        frame_idx = int(obj.frame_idx)
        rgb_bgr = read_video_frame(job.video_path, frame_idx)
        mask = _load_mask(obj_dir, frame_idx, object_id)
        if rgb_bgr.shape[:2] != mask.shape[:2]:
            raise ValueError(f"RGB/mask size mismatch: rgb={rgb_bgr.shape[:2]}, mask={mask.shape}")

        work = obj_step_dir / ".work"
        work.mkdir(parents=True, exist_ok=True)
        rgba = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGBA)
        rgba[~mask, 3] = 0
        rgba[mask, 3] = 255
        image_path = work / "input.png"
        mask_path = work / "0.png"
        cv2.imwrite(str(image_path), rgba)
        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))

        image = load_image(str(image_path))
        mask_img = load_single_mask(str(work), index=0)
        output = infer(image, mask_img, seed=42)
        sam3d_pose = _sam3d_pose_metadata(output)

        mesh = _mesh_from_sam3d_output(output)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        raw_mesh_path = obj_step_dir / "object_mesh_raw.obj"
        mesh.export(str(raw_mesh_path))
        pose_path = obj_step_dir / "sam3d_pose.json"
        pose_path.write_text(json.dumps(sam3d_pose, indent=2), encoding="utf-8")
        shutil.rmtree(work, ignore_errors=True)

        meta_i = {
            "object_id": object_id,
            "reconstruction_frame": frame_idx,
            "reference_frame_idx": frame_idx,
            "mesh_path": str(raw_mesh_path),
            "object_mesh_raw": str(raw_mesh_path),
            "sam3d_pose_output": sam3d_pose,
            "sam3d_pose_path": str(pose_path),
            "notes": "Raw SAM3D mesh only. Metric scale is produced by the separate sam3d_scale step.",
        }
        (obj_step_dir / "sam3d_meta.json").write_text(json.dumps(meta_i, indent=2), encoding="utf-8")
        if visualize:
            preview = encode_sam3d_vis_png(
                rgb_bgr=rgb_bgr,
                mask=mask.astype(np.uint8),
                mesh_path=raw_mesh_path,
                out_path=vis_png_path(obj_step_dir, job.video_id, f"{object_id}_recon"),
                frame_idx=frame_idx,
            )
            meta_i["vis_image"] = str(preview)
        object_metas.append(meta_i)

    if not object_metas:
        raise RuntimeError("No labeled objects found for SAM3D")

    primary = object_metas[0]
    # Backward-compatible top-level aliases for existing single-object consumers.
    shutil.copy2(primary["object_mesh_raw"], step_dir / "object_mesh_raw.obj")
    shutil.copy2(primary["sam3d_pose_path"], step_dir / "sam3d_pose.json")
    meta = {
        **primary,
        "num_objects": len(object_metas),
        "object_ids": [item["object_id"] for item in object_metas],
        "objects": object_metas,
    }
    (step_dir / "sam3d_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    discard_sam3d_nonessential(step_dir)

    write_step_completion(step_dir, "sam3d", dataset=job.dataset, video_id=job.video_id, extra=meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    job = VideoJob(dataset=args.dataset, video_id=args.video_id, video_path=args.video.resolve())
    run_sam3d(job, gpu=args.gpu, visualize=args.visualize, force=args.force, config_path=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
