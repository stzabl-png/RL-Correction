#!/usr/bin/env python3
"""Scale a raw SAM3D mesh via clicked-frame depth and single-frame FoundationPose."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

RECON_ROOT = Path(__file__).resolve().parents[1]
STEP_DIR = Path(__file__).resolve().parent
FP_DIR = RECON_ROOT / "fp_pose"
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))
if str(FP_DIR) not in sys.path:
    sys.path.insert(0, str(FP_DIR))

from _common.dataset import VideoJob  # noqa: E402
from _common.io import load_vipe_depth_frame, load_vipe_intrinsics, read_video_frame  # noqa: E402
from _common.paths import interim_step_dir, is_step_complete, write_step_completion  # noqa: E402
from fp_common import _depth_to_meters, _load_foundation_pose  # noqa: E402


def _load_sam3d_helpers():
    path = RECON_ROOT / "sam3d" / "run_sequence.py"
    spec = importlib.util.spec_from_file_location("_sam3d_run_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import SAM3D helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"Mesh has no usable vertices/faces: {path}")
    return mesh


def _load_sam3d_pose(sam3d_dir: Path) -> dict:
    for path in (sam3d_dir / "sam3d_pose.json", sam3d_dir / "sam3d_meta.json"):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "sam3d_meta.json":
            payload = payload.get("sam3d_pose_output", {})
        if payload:
            return payload
    return {}


def _estimate_scale_given_orientation(
    *,
    helpers,
    mesh,
    orientation: np.ndarray,
    orientation_source: str,
    observed_pointcloud: np.ndarray,
    camera_intrinsics: np.ndarray,
    image_shape,
) -> dict:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_center = helpers._robust_center(verts)
    obs_center = helpers._robust_center(observed_pointcloud)

    visible_raw, visible_cam_coarse, sampled_count = helpers._extract_visible_mesh_surface(
        mesh,
        orientation,
        obs_center,
        mesh_center,
        camera_intrinsics,
        image_shape,
    )
    obs_len, obs_axis, _, obs_pca_points = helpers._principal_axis_length(observed_pointcloud)
    mesh_len, mesh_axis, _, mesh_pca_points = helpers._principal_axis_length(visible_cam_coarse)
    scale = float(obs_len / mesh_len)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid scale estimate from {orientation_source}: {scale}")

    t_ref = obs_center - scale * (orientation @ mesh_center)
    v_ref_cam = scale * (orientation @ verts.T).T + t_ref
    visible_ref_cam = scale * (orientation @ visible_raw.T).T + t_ref
    warnings: list[str] = []
    if len(visible_ref_cam) < 500:
        warnings.append(f"{orientation_source}: visible mesh surface has fewer than 500 points.")
    if scale < 1e-4 or scale > 10.0:
        warnings.append(f"{orientation_source}: scale {scale:.6g} is outside expected sanity range.")

    return {
        "scale": scale,
        "orientation": orientation,
        "orientation_source": orientation_source,
        "translation_ref": t_ref,
        "mesh_center": mesh_center,
        "observed_center": obs_center,
        "observed_principal_axis_length": float(obs_len),
        "visible_mesh_principal_axis_length": float(mesh_len),
        "observed_principal_axis": obs_axis,
        "visible_mesh_principal_axis": mesh_axis,
        "observed_pca_points": int(obs_pca_points),
        "visible_mesh_pca_points": int(mesh_pca_points),
        "visible_mesh_points": visible_ref_cam,
        "num_visible_mesh_points": int(len(visible_ref_cam)),
        "num_mesh_surface_samples": int(sampled_count),
        "v_ref_cam": v_ref_cam,
        "warnings": warnings,
    }


def _write_overlay(
    *,
    helpers,
    rgb_bgr,
    mask,
    clean_mask,
    obs_points,
    mesh,
    v_ref_cam,
    k_mat,
    out_path: Path,
    frame_idx: int,
    scale_result: dict,
    label: str,
) -> Path:
    meta = {
        "scale_global": float(scale_result["scale"]),
        "observed_principal_axis_length": float(scale_result["observed_principal_axis_length"]),
        "visible_mesh_principal_axis_length": float(scale_result["visible_mesh_principal_axis_length"]),
        "num_depth_points": int(len(obs_points)),
    }
    out = helpers._write_scale_overlay(
        rgb_bgr=rgb_bgr,
        mask=mask,
        clean_mask=clean_mask,
        obs_points=obs_points,
        mesh=mesh,
        v_ref_cam=v_ref_cam,
        k_mat=k_mat,
        out_path=out_path,
        frame_idx=frame_idx,
        scale_meta=meta,
    )
    import cv2

    img = cv2.imread(str(out))
    if img is not None:
        cv2.putText(img, label, (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (12, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out), img)
    return out


def _project_mesh_mask(mesh, vertices_cam: np.ndarray, k_mat: np.ndarray, shape) -> np.ndarray:
    import cv2

    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    proj = (k_mat @ vertices_cam.T).T
    valid_z = proj[:, 2] > 1e-6
    proj[:, 0] /= np.maximum(proj[:, 2], 1e-6)
    proj[:, 1] /= np.maximum(proj[:, 2], 1e-6)
    faces = np.asarray(mesh.faces)
    for tri in faces:
        if not np.all(valid_z[tri]):
            continue
        poly = proj[tri, :2].astype(np.int32)
        if (
            np.all((poly[:, 0] < 0) | (poly[:, 0] >= w))
            or np.all((poly[:, 1] < 0) | (poly[:, 1] >= h))
        ):
            continue
        cv2.fillConvexPoly(mask, poly, 255, cv2.LINE_AA)
    return mask > 0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(aa, bb).sum() / union)


def _write_foundationpose_overlay(
    *,
    rgb_bgr,
    mask,
    mesh,
    pose: np.ndarray,
    k_mat: np.ndarray,
    out_path: Path,
    frame_idx: int,
) -> Path:
    import cv2

    frame = rgb_bgr.copy()
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts_h = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float64)], axis=1)
    v_cam = (pose @ verts_h.T).T[:, :3]
    proj = (k_mat @ v_cam.T).T
    proj[:, 0] /= np.maximum(proj[:, 2], 1e-6)
    proj[:, 1] /= np.maximum(proj[:, 2], 1e-6)

    face_overlay = frame.copy()
    faces = np.asarray(mesh.faces)
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
    edge_stride = max(1, len(edges) // 9000)
    for e0, e1 in edges[::edge_stride]:
        if proj[e0, 2] <= 0 or proj[e1, 2] <= 0:
            continue
        cv2.line(frame, tuple(proj[e0, :2].astype(int)), tuple(proj[e1, :2].astype(int)), (0, 80, 255), 2)
    cv2.putText(
        frame,
        f"FoundationPose single-frame alignment | ref_frame={frame_idx}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FoundationPose single-frame alignment | ref_frame={frame_idx}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return out_path


def _write_debug_3d(*, obs_points, stage1_points, final_points, out_path: Path, scale1: float, scale2: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = np.asarray(obs_points, dtype=np.float64)
    st1 = np.asarray(stage1_points, dtype=np.float64)
    fin = np.asarray(final_points, dtype=np.float64)
    obs_stride = max(1, len(obs) // 6000)
    st1_stride = max(1, len(st1) // 4000)
    fin_stride = max(1, len(fin) // 4000)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(obs[::obs_stride, 0], obs[::obs_stride, 1], obs[::obs_stride, 2], s=2, c="#00d6ff", label="depth")
    ax.scatter(st1[::st1_stride, 0], st1[::st1_stride, 1], st1[::st1_stride, 2], s=2, c="#ff9f1c", label="stage1")
    ax.scatter(fin[::fin_stride, 0], fin[::fin_stride, 1], fin[::fin_stride, 2], s=2, c="#ff2d55", label="final")
    all_pts = np.concatenate([obs[::obs_stride], st1[::st1_stride], fin[::fin_stride]], axis=0)
    center = all_pts.mean(axis=0)
    radius = max(float(np.max(np.ptp(all_pts, axis=0)) * 0.55), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X cam")
    ax.set_ylabel("Y cam")
    ax.set_zlabel("Z cam")
    ax.set_title(f"SAM3D scale bridge | s1={scale1:.5f} s2={scale2:.5f}")
    ax.legend(loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _run_sam3d_scale_one(
    job: VideoJob,
    *,
    object_id: str,
    obj_prompt,
    gpu: int,
    visualize: bool,
    force: bool,
    depth_scale: float = 1.0,
    register_iters: int = 5,
) -> dict:
    import cv2

    helpers = _load_sam3d_helpers()
    base_step_dir = interim_step_dir(job.dataset, job.video_id, "sam3d_scale")
    base_sam3d_dir = interim_step_dir(job.dataset, job.video_id, "sam3d")
    step_dir = base_step_dir / "objects" / object_id
    sam3d_dir = base_sam3d_dir / "objects" / object_id
    obj_dir = interim_step_dir(job.dataset, job.video_id, "sam2_object")
    vipe_dir = interim_step_dir(job.dataset, job.video_id, "vipe")
    step_dir.mkdir(parents=True, exist_ok=True)

    raw_mesh_path = sam3d_dir / "object_mesh_raw.obj"
    if not raw_mesh_path.is_file():
        raise FileNotFoundError(f"Missing raw SAM3D mesh: {raw_mesh_path}. Run sam3d before sam3d_scale.")
    raw_mesh = _load_mesh(raw_mesh_path)

    frame_idx = int(obj_prompt.frame_idx)
    # 与 sam3d 同帧(frame_plan.sam3d_frame): 尺度估计必须在网格重建的同一帧上做
    from _common.frame_plan import load_frame_plan, planned_frame
    _pf = planned_frame(load_frame_plan(obj_dir), object_id, "sam3d_frame")
    if _pf is not None:
        print(f"[sam3d_scale] frame_plan {object_id}: 参考帧 {frame_idx} -> {_pf}", flush=True)
        frame_idx = int(_pf)
    click_points = obj_prompt.points
    rgb_bgr = read_video_frame(job.video_path, frame_idx)
    mask = helpers._load_mask(obj_dir, frame_idx, object_id)
    depth_raw = load_vipe_depth_frame(vipe_dir, job.video_id, frame_idx)
    depth = _depth_to_meters(depth_raw, depth_scale)
    k_mat, _flip = load_vipe_intrinsics(vipe_dir, job.video_id)
    if rgb_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(f"RGB/mask size mismatch: rgb={rgb_bgr.shape[:2]}, mask={mask.shape}")
    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Depth/mask size mismatch: depth={depth.shape}, mask={mask.shape}")

    clean_mask, component_mask, hand_mask = helpers._clean_mask_for_depth(
        mask,
        click_points,
        step_dir,
        job.video_id,
        frame_idx,
    )
    obs_points, valid_depth_mask = helpers._backproject_mask_depth(depth, clean_mask, k_mat)
    if len(obs_points) < 100:
        raise RuntimeError(f"Too few observed object depth points on clicked frame {frame_idx}: {len(obs_points)}")
    pointcloud_path = step_dir / "observed_object_pointcloud_ref.ply"
    helpers._save_pointcloud_ply(pointcloud_path, obs_points)

    sam3d_pose = _load_sam3d_pose(sam3d_dir)
    r0, r0_source = helpers._rotation_from_sam3d_pose(sam3d_pose)
    warnings: list[str] = []
    if r0_source.startswith("identity"):
        warnings.append("SAM3D orientation missing or invalid; stage-1 scale used identity orientation.")

    stage1 = _estimate_scale_given_orientation(
        helpers=helpers,
        mesh=raw_mesh,
        orientation=r0,
        orientation_source=r0_source,
        observed_pointcloud=obs_points,
        camera_intrinsics=k_mat,
        image_shape=rgb_bgr.shape,
    )
    warnings.extend(stage1["warnings"])
    stage1_mesh = raw_mesh.copy()
    stage1_mesh.vertices = np.asarray(raw_mesh.vertices, dtype=np.float64) * stage1["scale"]
    stage1_mesh_path = step_dir / "object_mesh_scaled_stage1.obj"
    stage1_mesh.export(str(stage1_mesh_path))

    stage1_overlay = _write_overlay(
        helpers=helpers,
        rgb_bgr=rgb_bgr,
        mask=component_mask,
        clean_mask=valid_depth_mask,
        obs_points=obs_points,
        mesh=raw_mesh,
        v_ref_cam=stage1["v_ref_cam"],
        k_mat=k_mat,
        out_path=step_dir / "debug_stage1_scale_overlay.png",
        frame_idx=frame_idx,
        scale_result=stage1,
        label=f"stage 1 | orientation={r0_source}",
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    estimator = _load_foundation_pose(stage1_mesh_path, gpu=gpu, debug_dir=step_dir / "fp_debug_stage1")
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    pose = estimator.register(k_mat, rgb_rgb, depth, component_mask > 0, iteration=register_iters)
    pose = np.asarray(pose, dtype=np.float64)
    pose_path = step_dir / "foundationpose_ref_pose.txt"
    np.savetxt(pose_path, pose, fmt="%.8f")
    fp_overlay = _write_foundationpose_overlay(
        rgb_bgr=rgb_bgr,
        mask=component_mask,
        mesh=stage1_mesh,
        pose=pose,
        k_mat=k_mat,
        out_path=step_dir / "debug_foundationpose_overlay.png",
        frame_idx=frame_idx,
    )

    r1 = pose[:3, :3]
    stage2 = _estimate_scale_given_orientation(
        helpers=helpers,
        mesh=raw_mesh,
        orientation=r1,
        orientation_source="foundationpose_ref_pose_stage1",
        observed_pointcloud=obs_points,
        camera_intrinsics=k_mat,
        image_shape=rgb_bgr.shape,
    )
    warnings.extend(stage2["warnings"])
    ratio = float(stage2["scale"] / stage1["scale"])
    if ratio > 2.0 or ratio < 0.5:
        warnings.append(f"Stage-2/stage-1 scale ratio is unstable: {ratio:.4f}.")

    # The production scale contract uses the SAM3D visible-surface estimate.
    # FoundationPose remains a diagnostic orientation check here and supplies
    # the downstream per-frame pose, but it must not rescale the final mesh.
    final_result = stage1
    final_mesh = stage1_mesh.copy()
    final_mesh_path = step_dir / "object_mesh_scaled_final.obj"
    final_mesh.export(str(final_mesh_path))

    final_overlay = _write_overlay(
        helpers=helpers,
        rgb_bgr=rgb_bgr,
        mask=component_mask,
        clean_mask=valid_depth_mask,
        obs_points=obs_points,
        mesh=raw_mesh,
        v_ref_cam=final_result["v_ref_cam"],
        k_mat=k_mat,
        out_path=step_dir / "debug_final_scale_overlay.png",
        frame_idx=frame_idx,
        scale_result=final_result,
        label=f"final | SAM3D orientation={r0_source}",
    )
    debug_3d = _write_debug_3d(
        obs_points=obs_points,
        stage1_points=stage1["visible_mesh_points"],
        final_points=final_result["visible_mesh_points"],
        out_path=step_dir / "debug_scale_3d.png",
        scale1=stage1["scale"],
        scale2=final_result["scale"],
    )

    projected_final = _project_mesh_mask(raw_mesh, final_result["v_ref_cam"], k_mat, rgb_bgr.shape)
    final_iou = _mask_iou(projected_final, component_mask)
    if final_iou < 0.05:
        warnings.append(f"Projected final mesh has low overlap with object mask: IoU={final_iou:.4f}.")

    meta = {
        "object_id": object_id,
        "reference_frame_idx": frame_idx,
        "scale_stage1": float(stage1["scale"]),
        "scale_final": float(final_result["scale"]),
        "sam3d_orientation_used_stage1": r0.tolist(),
        "sam3d_orientation_source_stage1": r0_source,
        "foundationpose_orientation_used_stage2": r1.tolist(),
        "foundationpose_translation_stage1": pose[:3, 3].tolist(),
        "foundationpose_ref_pose": str(pose_path),
        "num_observed_object_points": int(len(obs_points)),
        "num_valid_depth_pixels": int(valid_depth_mask.sum()),
        "num_clean_mask_pixels": int(clean_mask.sum()),
        "num_component_mask_pixels": int(component_mask.sum()),
        "num_hand_mask_pixels_removed": int(np.logical_and(component_mask, hand_mask).sum()),
        "observed_pointcloud_center": final_result["observed_center"].tolist(),
        "mesh_center_raw": final_result["mesh_center"].tolist(),
        "observed_principal_axis_length_stage1": float(stage1["observed_principal_axis_length"]),
        "visible_mesh_principal_axis_length_stage1": float(stage1["visible_mesh_principal_axis_length"]),
        "observed_principal_axis_length_stage2": float(stage2["observed_principal_axis_length"]),
        "visible_mesh_principal_axis_length_stage2": float(stage2["visible_mesh_principal_axis_length"]),
        "scale_ratio_stage2_over_stage1": ratio,
        "final_projected_mask_iou": final_iou,
        "camera_intrinsics": k_mat.tolist(),
        "depth_source": "vipe",
        "depth_scale": depth_scale,
        "mask_source": "sam2_clicked_frame",
        "raw_mesh": str(raw_mesh_path),
        "object_mesh_scaled_stage1": str(stage1_mesh_path),
        "object_mesh_scaled_final": str(final_mesh_path),
        "observed_object_pointcloud_ref": str(pointcloud_path),
        "debug_stage1_scale_overlay": str(stage1_overlay),
        "debug_foundationpose_overlay": str(fp_overlay),
        "debug_final_scale_overlay": str(final_overlay),
        "debug_scale_3d": str(debug_3d),
        "scale_method": "sam3d_visible_surface_principal_axis",
        "foundationpose_stage2_diagnostic_only": True,
        "warnings": warnings,
    }
    metadata_path = step_dir / "scale_fpalign_scale_metadata.json"
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if warnings:
        print("SAM3D scale warnings:", *warnings, sep="\n  ", flush=True)

    return meta


def run_sam3d_scale(
    job: VideoJob,
    *,
    gpu: int,
    visualize: bool,
    force: bool,
    depth_scale: float = 1.0,
    register_iters: int = 5,
) -> dict:
    from sam2_object.sam2_object_common import load_label_prompt

    step_dir = interim_step_dir(job.dataset, job.video_id, "sam3d_scale")
    step_dir.mkdir(parents=True, exist_ok=True)
    if not force and is_step_complete(step_dir, "sam3d_scale"):
        return {"skipped": True}

    prompt = load_label_prompt(interim_step_dir(job.dataset, job.video_id, "sam2_object"))
    objects = []
    for obj in prompt.objects:
        meta_i = _run_sam3d_scale_one(
            job,
            object_id=obj.object_id,
            obj_prompt=obj,
            gpu=gpu,
            visualize=visualize,
            force=force,
            depth_scale=depth_scale,
            register_iters=register_iters,
        )
        objects.append(meta_i)

    if not objects:
        raise RuntimeError("No labeled objects found for sam3d_scale")

    primary = objects[0]
    primary_dir = step_dir / "objects" / primary["object_id"]
    for name in (
        "object_mesh_scaled_final.obj",
        "object_mesh_scaled_stage1.obj",
        "scale_fpalign_scale_metadata.json",
        "foundationpose_ref_pose.txt",
    ):
        src = primary_dir / name
        if src.is_file():
            shutil.copy2(src, step_dir / name)
    meta = {**primary, "num_objects": len(objects), "object_ids": [obj["object_id"] for obj in objects], "objects": objects}
    write_step_completion(step_dir, "sam3d_scale", dataset=job.dataset, video_id=job.video_id, extra=meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Scale ViPE depth to metres")
    parser.add_argument("--register-iters", type=int, default=5)
    parser.add_argument("--visualize", action="store_true", help="Accepted for pipeline consistency; debug outputs are always written.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    job = VideoJob(dataset=args.dataset, video_id=args.video_id, video_path=args.video.resolve())
    run_sam3d_scale(
        job,
        gpu=args.gpu,
        visualize=args.visualize,
        force=args.force,
        depth_scale=args.depth_scale,
        register_iters=args.register_iters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
