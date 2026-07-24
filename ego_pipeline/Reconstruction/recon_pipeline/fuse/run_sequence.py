#!/usr/bin/env python3
"""Fuse hand MANO + object pose into ViPE world frame; optional test visualization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

RECON_ROOT = Path(__file__).resolve().parents[1]
FUSE_DIR = Path(__file__).resolve().parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))
if str(FUSE_DIR) not in sys.path:
    sys.path.insert(0, str(FUSE_DIR))

from _common.hawor_mano import compute_hand_sequence_geometry, load_hand_params_numpy  # noqa: E402
from _common.gravity import GRAVITY_FRAME, VIPE_FRAME, gravity_camera_forward_alignment_transform, load_vipe_gravity  # noqa: E402
from _common.io import apply_vipe_focal_flip_c2w, count_video_frames, interpolate_c2w_poses, load_vipe_intrinsics, load_vipe_poses  # noqa: E402
from _common.paths import final_video_dir, interim_step_dir, write_step_completion  # noqa: E402
from _common.viz import encode_fuse_vis_mp4, vis_path  # noqa: E402
from fp_pose.fp_common import ob_in_cam_to_world  # noqa: E402
from sam2_object.sam2_object_common import OBJECT_MASK_ID  # noqa: E402

DEFAULT_HAWOR_PYTHON = Path(os.environ.get("HAWOR_PYTHON", "/home/jiakaichen/miniconda3/envs/hawor/bin/python"))
FINAL_SCHEMA_VERSION = "recon_world_v4"
FINAL_NPZ = "world_fused.npz"
FINAL_SUMMARY = "world_summary.json"
FINAL_MESH = "object_mesh_scaled_final.obj"
FINAL_COMPLETION = "reconstruction_complete.json"
FINAL_MASKS_DIR = "masks"
FINAL_MASKS_MANIFEST = "masks_manifest.json"
WORLD_XY_ALIGNMENT_MODE = "middle_camera_forward_projected_xy"
WORLD_XY_ALIGNMENT_TOL = 1e-5
REQUIRED_FINAL_KEYS = {
    "schema_version",
    "dataset",
    "video_id",
    "num_frames",
    "K",
    "c2w",
    "c2w_vipe_world",
    "object_ob_in_cam",
    "object_frame_indices",
    "object_ob_in_world",
    "mesh_filename",
    "T_gravity_z_up_from_vipe_world",
    "gravity_direction_vipe_world",
    "up_direction_vipe_world",
    "world_xy_alignment_mode",
    "world_xy_alignment_frame_index",
    "world_x_direction_vipe_world",
    "world_y_direction_vipe_world",
    "world_z_direction_vipe_world",
}
EXPECTED_HAND_SOURCE_FRAME = "hawor_world_vipe_camera"


def _load_pose_visibility(fp_dir: Path, object_id: str, *, num_frames: int) -> dict:
    path = fp_dir / "objects" / object_id / "pose_visibility.npz"
    if not path.is_file() and object_id == OBJECT_MASK_ID:
        path = fp_dir / "pose_visibility.npz"
    visible_by_frame = np.zeros(num_frames, dtype=bool)
    pose_valid_by_frame = np.zeros(num_frames, dtype=bool)
    if not path.is_file():
        return {
            "path": None,
            "visible_by_frame": visible_by_frame,
            "pose_valid_by_frame": pose_valid_by_frame,
            "visible_frames": 0,
            "pose_valid_frames": 0,
        }
    with np.load(path, allow_pickle=False) as data:
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)
        visible = np.asarray(data["visible"]).astype(bool)
        pose_valid = np.asarray(data["pose_valid"]).astype(bool)
    keep = (frame_indices >= 0) & (frame_indices < num_frames)
    frame_indices = frame_indices[keep]
    visible = visible[keep]
    pose_valid = pose_valid[keep]
    visible_by_frame[frame_indices] = visible
    pose_valid_by_frame[frame_indices] = pose_valid
    return {
        "path": str(path),
        "visible_by_frame": visible_by_frame,
        "pose_valid_by_frame": pose_valid_by_frame,
        "visible_frames": int(visible_by_frame.sum()),
        "pose_valid_frames": int(pose_valid_by_frame.sum()),
    }


def _validate_object_arrays(data: np.lib.npyio.NpzFile, *, final_dir: Path) -> dict:
    if "num_objects" not in data.files:
        return {"num_objects": 1}
    required = {
        "object_ids",
        "object_mesh_filenames",
        "object_frame_indices_all",
        "object_valid_all",
        "object_ob_in_cam_all",
        "object_ob_in_world_all",
        "object_ob_in_vipe_world_all",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise RuntimeError(f"Final multi-object NPZ missing required keys: {missing}")
    num_objects = int(data["num_objects"])
    object_ids = np.asarray(data["object_ids"])
    mesh_filenames = np.asarray(data["object_mesh_filenames"])
    frame_indices_all = np.asarray(data["object_frame_indices_all"])
    valid_all = np.asarray(data["object_valid_all"]).astype(bool)
    ob_cam_all = np.asarray(data["object_ob_in_cam_all"])
    ob_world_all = np.asarray(data["object_ob_in_world_all"])
    ob_vipe_all = np.asarray(data["object_ob_in_vipe_world_all"])
    num_frames = int(data["num_frames"]) if "num_frames" in data.files else None
    if num_objects < 1:
        raise RuntimeError("num_objects must be >= 1")
    if object_ids.shape != (num_objects,):
        raise RuntimeError(f"Invalid object_ids shape {object_ids.shape}; expected {(num_objects,)}")
    if mesh_filenames.shape != (num_objects,):
        raise RuntimeError(f"Invalid object_mesh_filenames shape {mesh_filenames.shape}; expected {(num_objects,)}")
    if frame_indices_all.shape != valid_all.shape:
        raise RuntimeError("object_frame_indices_all and object_valid_all shapes do not match")
    if frame_indices_all.shape[0] != num_objects:
        raise RuntimeError("object_frame_indices_all first dimension does not match num_objects")
    expected_pose_shape = frame_indices_all.shape + (4, 4)
    if ob_cam_all.shape != expected_pose_shape:
        raise RuntimeError(f"Invalid object_ob_in_cam_all shape {ob_cam_all.shape}; expected {expected_pose_shape}")
    if ob_world_all.shape != expected_pose_shape:
        raise RuntimeError(f"Invalid object_ob_in_world_all shape {ob_world_all.shape}; expected {expected_pose_shape}")
    if ob_vipe_all.shape != expected_pose_shape:
        raise RuntimeError(f"Invalid object_ob_in_vipe_world_all shape {ob_vipe_all.shape}; expected {expected_pose_shape}")
    for idx, mesh_filename in enumerate(mesh_filenames):
        mesh_path = final_dir / str(mesh_filename)
        if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty final mesh for object {object_ids[idx]}: {mesh_path}")
    if "object_visible_by_frame" in data.files:
        visible_shape = np.asarray(data["object_visible_by_frame"]).shape
        expected = (num_objects, num_frames)
        if visible_shape != expected:
            raise RuntimeError(f"Invalid object_visible_by_frame shape {visible_shape}; expected {expected}")
    if "object_pose_valid_by_frame" in data.files:
        pose_valid_shape = np.asarray(data["object_pose_valid_by_frame"]).shape
        expected = (num_objects, num_frames)
        if pose_valid_shape != expected:
            raise RuntimeError(f"Invalid object_pose_valid_by_frame shape {pose_valid_shape}; expected {expected}")
    return {"num_objects": num_objects}


def _sample_hand_index(frame_idx: int, num_video_frames: int, num_hand_frames: int) -> int:
    if num_hand_frames <= 1 or num_video_frames <= 1:
        return 0
    t = frame_idx * (num_hand_frames - 1) / (num_video_frames - 1)
    return int(np.clip(round(t), 0, num_hand_frames - 1))



def _validate_final_masks(final_dir: Path, *, num_frames: int) -> dict:
    manifest_path = final_dir / FINAL_MASKS_DIR / FINAL_MASKS_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing final mask manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    object_masks = payload.get("object_masks") or {}
    if not object_masks.get("present"):
        raise RuntimeError("Final object masks are missing from mask manifest")
    object_ids = list(object_masks.get("object_ids") or [])
    expected_object_pngs = num_frames * len(object_ids)
    object_pngs = int(object_masks.get("num_png_files") or 0)
    if expected_object_pngs > 0 and object_pngs < expected_object_pngs:
        raise RuntimeError(f"Final object masks incomplete: {object_pngs} < {expected_object_pngs}")
    object_dir = final_dir / str(object_masks.get("path", ""))
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Missing final object mask directory: {object_dir}")

    hand_masks = payload.get("hand_masks") or {}
    hand_pngs = int(hand_masks.get("num_png_files") or 0)
    if hand_masks.get("present"):
        hand_dir = final_dir / str(hand_masks.get("path", ""))
        if not hand_dir.is_dir():
            raise FileNotFoundError(f"Missing final hand mask directory: {hand_dir}")
        if hand_pngs < num_frames:
            raise RuntimeError(f"Final hand masks incomplete: {hand_pngs} < {num_frames}")
    return {
        "masks_included": True,
        "mask_manifest": _relative_to_final(manifest_path, final_dir),
        "object_mask_pngs": object_pngs,
        "hand_mask_pngs": hand_pngs,
    }


def _validate_hand_roots_in_front(data: np.lib.npyio.NpzFile, *, num_frames: int) -> dict:
    if not {"hand_trans", "hand_valid"}.issubset(data.files):
        return {"hands_included": False, "hand_root_front_samples": 0, "hand_valid_samples": 0}

    if "hand_source_coordinate_frame" in data.files:
        hand_source = str(data["hand_source_coordinate_frame"].item())
        if hand_source != EXPECTED_HAND_SOURCE_FRAME:
            raise RuntimeError(
                f"Unsupported hand_source_coordinate_frame {hand_source}; expected {EXPECTED_HAND_SOURCE_FRAME}. "
                "Re-run fuse or repair the final hand globals."
            )

    hand_trans = np.asarray(data["hand_trans"], dtype=np.float64)
    hand_valid = np.asarray(data["hand_valid"]).astype(bool)
    c2w = np.asarray(data["c2w"], dtype=np.float64)
    if hand_trans.shape[:2] != hand_valid.shape:
        raise RuntimeError("hand_trans and hand_valid frame dimensions do not match")
    if hand_trans.shape[-1] != 3:
        raise RuntimeError(f"Invalid hand_trans shape {hand_trans.shape}; expected (..., 3)")

    valid_samples = 0
    front_samples = 0
    sample_frames = np.unique(np.linspace(0, num_frames - 1, min(32, num_frames), dtype=np.int32))
    for frame_idx in sample_frames:
        hand_idx = _sample_hand_index(int(frame_idx), num_frames, hand_valid.shape[1])
        w2c = np.linalg.inv(c2w[int(frame_idx)])
        for hand_idx_side in range(hand_valid.shape[0]):
            if not hand_valid[hand_idx_side, hand_idx]:
                continue
            valid_samples += 1
            point_world = np.concatenate([hand_trans[hand_idx_side, hand_idx], [1.0]])
            point_cam = w2c @ point_world
            if np.isfinite(point_cam).all() and point_cam[2] > 1e-4:
                front_samples += 1

    if hand_valid.any() and valid_samples > 0 and front_samples == 0:
        raise RuntimeError(
            "All sampled valid MANO hand roots are behind the final camera. "
            "Check HaWoR-to-ViPE hand coordinate conversion."
        )
    return {
        "hands_included": True,
        "hand_root_front_samples": int(front_samples),
        "hand_valid_samples": int(valid_samples),
    }


def _validate_world_xy_alignment(
    data: np.lib.npyio.NpzFile,
    *,
    r_zup_from_vipe: np.ndarray,
    num_frames: int,
) -> dict:
    mode = str(data["world_xy_alignment_mode"].item())
    if mode != WORLD_XY_ALIGNMENT_MODE:
        raise RuntimeError(f"Unsupported world_xy_alignment_mode {mode}; expected {WORLD_XY_ALIGNMENT_MODE}")
    frame_idx = int(np.asarray(data["world_xy_alignment_frame_index"]).item())
    if frame_idx < 0 or frame_idx >= num_frames:
        raise RuntimeError(f"Invalid world_xy_alignment_frame_index {frame_idx}; expected [0, {num_frames})")

    c2w_vipe = np.asarray(data["c2w_vipe_world"], dtype=np.float64)
    if c2w_vipe.shape != (num_frames, 4, 4):
        raise RuntimeError(f"Invalid c2w_vipe_world shape {c2w_vipe.shape}; expected {(num_frames, 4, 4)}")

    axis_targets = {
        "world_x_direction_vipe_world": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "world_y_direction_vipe_world": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "world_z_direction_vipe_world": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    for key, target in axis_targets.items():
        axis_vipe = np.asarray(data[key], dtype=np.float64)
        if axis_vipe.shape != (3,):
            raise RuntimeError(f"Invalid {key} shape {axis_vipe.shape}; expected (3,)")
        if not np.allclose(r_zup_from_vipe @ axis_vipe, target, atol=WORLD_XY_ALIGNMENT_TOL):
            raise RuntimeError(f"{key} is inconsistent with T_gravity_z_up_from_vipe_world")

    camera_forward_final = r_zup_from_vipe @ c2w_vipe[frame_idx, :3, 2]
    camera_forward_xy = camera_forward_final.copy()
    camera_forward_xy[2] = 0.0
    horizontal_norm = float(np.linalg.norm(camera_forward_xy))
    if horizontal_norm <= 1e-6:
        raise RuntimeError("Middle-frame camera focal axis is too close to gravity to validate world +X")
    camera_forward_xy /= horizontal_norm
    if not np.allclose(camera_forward_xy, np.array([1.0, 0.0, 0.0]), atol=WORLD_XY_ALIGNMENT_TOL):
        raise RuntimeError("World +X must match the middle-frame camera focal axis projected into the world XY plane")

    return {
        "world_xy_alignment_mode": mode,
        "world_xy_alignment_frame_index": frame_idx,
    }


def _resolve_hawor_world(hawor_dir: Path, video_id: str) -> Path | None:
    candidates = (
        hawor_dir / video_id / "world_space_res.pth",
        hawor_dir / "world_space_res.pth",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _relative_to_final(path: Path, final_dir: Path) -> str:
    return str(path.relative_to(final_dir))


def _copy_mask_tree(src_dir: Path, dst_dir: Path, final_dir: Path, *, required: bool) -> dict:
    if not src_dir.is_dir():
        if required:
            raise FileNotFoundError(f"Missing mask directory: {src_dir}")
        return {
            "present": False,
            "source": str(src_dir),
            "path": _relative_to_final(dst_dir, final_dir),
            "num_frame_dirs": 0,
            "num_png_files": 0,
        }
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst_dir)
    frame_dirs = sorted(p for p in dst_dir.glob("frame_*_masks") if p.is_dir())
    png_files = sorted(dst_dir.glob("frame_*_masks/*.png"))
    return {
        "present": True,
        "source": str(src_dir),
        "path": _relative_to_final(dst_dir, final_dir),
        "num_frame_dirs": len(frame_dirs),
        "num_png_files": len(png_files),
    }


def _copy_if_present(src: Path, dst: Path, final_dir: Path) -> str | None:
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return _relative_to_final(dst, final_dir)


def _export_final_2d_masks(
    *,
    dataset: str,
    video_id: str,
    out_dir: Path,
    object_prompt_dir: Path,
    sam3_hands_dir: Path,
    object_ids: list[str],
) -> dict:
    masks_dir = out_dir / FINAL_MASKS_DIR
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    object_masks = _copy_mask_tree(
        object_prompt_dir / "video_segmentation" / "masks",
        masks_dir / "objects" / "frames",
        out_dir,
        required=True,
    )
    object_mask_root = object_masks["path"]
    object_mask_files = {
        object_id: f"{object_mask_root}/frame_{{frame_idx:06d}}_masks/{object_id}.png"
        for object_id in object_ids
    }
    object_prompt = _copy_if_present(
        object_prompt_dir / "label_prompt.json",
        masks_dir / "objects" / "label_prompt.json",
        out_dir,
    )
    object_complete = _copy_if_present(
        object_prompt_dir / "sam2_object_complete.json",
        masks_dir / "objects" / "sam2_object_complete.json",
        out_dir,
    )

    hand_masks = _copy_mask_tree(
        sam3_hands_dir / video_id / "video_segmentation" / "masks",
        masks_dir / "hands" / "frames",
        out_dir,
        required=False,
    )
    hand_complete = _copy_if_present(
        sam3_hands_dir / video_id / "hand_masks_complete.json",
        masks_dir / "hands" / "hand_masks_complete.json",
        out_dir,
    )
    hand_step_complete = _copy_if_present(
        sam3_hands_dir / "sam3_hands_complete.json",
        masks_dir / "hands" / "sam3_hands_complete.json",
        out_dir,
    )

    hand_mask_files = {}
    if hand_masks["present"]:
        hand_mask_root = hand_masks["path"]
        hand_mask_files = {
            "left": f"{hand_mask_root}/frame_{{frame_idx:06d}}_masks/left_hand_0.png",
            "right": f"{hand_mask_root}/frame_{{frame_idx:06d}}_masks/right_hand_0.png",
        }

    manifest = {
        "schema_version": "final_2d_masks_v1",
        "dataset": dataset,
        "video_id": video_id,
        "root": FINAL_MASKS_DIR,
        "object_masks": {
            **object_masks,
            "object_ids": object_ids,
            "mask_files": object_mask_files,
            "label_prompt": object_prompt,
            "completion": object_complete,
        },
        "hand_masks": {
            **hand_masks,
            "mask_files": hand_mask_files,
            "completion": hand_complete,
            "step_completion": hand_step_complete,
        },
    }
    manifest_path = masks_dir / FINAL_MASKS_MANIFEST
    manifest["manifest"] = _relative_to_final(manifest_path, out_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _transform_hand_payload_to_zup(hand_payload: dict[str, np.ndarray], r_zup_from_vipe: np.ndarray) -> dict[str, np.ndarray]:
    """Convert raw HaWoR MANO world params into the exported z-up world frame."""
    from scipy.spatial.transform import Rotation

    out = dict(hand_payload)
    if "hand_trans" not in out or "hand_rot" not in out:
        return out
    # HaWoR is run with ViPE camera poses injected as its SLAM trajectory, so
    # its world-space MANO globals already follow the ViPE camera trajectory
    # convention. Do not apply the legacy HaWoR visualization flip here.
    r_zup_from_hawor = np.asarray(r_zup_from_vipe, dtype=np.float64)
    trans = np.asarray(out["hand_trans"], dtype=np.float64)
    rotvec = np.asarray(out["hand_rot"], dtype=np.float64)
    out["hand_trans_hawor_world"] = np.array(out["hand_trans"], copy=True)
    out["hand_rot_hawor_world"] = np.array(out["hand_rot"], copy=True)
    out["hand_trans"] = np.einsum("ij,...j->...i", r_zup_from_hawor, trans).astype(trans.dtype, copy=False)
    root_rot = Rotation.from_matrix(
        np.einsum("ij,...jk->...ik", r_zup_from_hawor, Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix())
    ).as_rotvec()
    out["hand_rot"] = root_rot.reshape(rotvec.shape).astype(rotvec.dtype, copy=False)
    out["hand_coordinate_frame"] = np.array("gravity_z_up_world")
    out["hand_source_coordinate_frame"] = np.array("hawor_world_vipe_camera")
    return out


def _render_fuse_vis_subprocess(
    *,
    dataset: str,
    video_id: str,
    video_path: Path,
    hawor_world: Path,
) -> dict | None:
    script = FUSE_DIR / "render_vis.py"
    hawor_py = DEFAULT_HAWOR_PYTHON
    if not hawor_py.is_file():
        return None
    cmd = [
        str(hawor_py),
        str(script),
        "--dataset",
        dataset,
        "--video-id",
        video_id,
        "--video",
        str(video_path),
        "--hawor-world",
        str(hawor_world),
    ]
    proc = subprocess.run(cmd, cwd=str(RECON_ROOT.parent), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout, flush=True)
        return None
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return payload
    except (json.JSONDecodeError, IndexError):
        return None


def fuse_world_sequence(
    *,
    dataset: str,
    video_id: str,
    video_path: Path,
    visualize: bool = False,
) -> dict:
    vipe_dir = interim_step_dir(dataset, video_id, "vipe")
    hawor_dir = interim_step_dir(dataset, video_id, "hawor")
    fp_dir = interim_step_dir(dataset, video_id, "fp_pose")
    sam3d_scale_dir = interim_step_dir(dataset, video_id, "sam3d_scale")
    object_prompt_dir = interim_step_dir(dataset, video_id, "sam2_object")
    sam3_hands_dir = interim_step_dir(dataset, video_id, "sam3_hands")
    out_dir = final_video_dir(dataset, video_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from sam2_object.sam2_object_common import load_label_prompt

        object_ids = load_label_prompt(object_prompt_dir).object_ids()
    except Exception:
        object_ids = [OBJECT_MASK_ID]
    interim_mesh_path = sam3d_scale_dir / "objects" / object_ids[0] / "object_mesh_scaled_final.obj"
    if not interim_mesh_path.is_file():
        interim_mesh_path = sam3d_scale_dir / "object_mesh_scaled_final.obj"
    if not interim_mesh_path.is_file():
        raise FileNotFoundError(f"Missing final scaled object mesh: {interim_mesh_path}. Run sam3d_scale before fuse.")

    final_mesh_path = out_dir / FINAL_MESH
    shutil.copy2(interim_mesh_path, final_mesh_path)
    final_objects_dir = out_dir / "objects"
    object_mesh_filenames = []
    for object_id in object_ids:
        src_mesh = sam3d_scale_dir / "objects" / object_id / "object_mesh_scaled_final.obj"
        if not src_mesh.is_file() and object_id == object_ids[0]:
            src_mesh = interim_mesh_path
        if not src_mesh.is_file():
            raise FileNotFoundError(f"Missing scaled mesh for {object_id}: {src_mesh}")
        dst_rel = Path("objects") / object_id / FINAL_MESH
        dst_mesh = out_dir / dst_rel
        dst_mesh.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_mesh, dst_mesh)
        object_mesh_filenames.append(str(dst_rel))

    inds, c2w_all = load_vipe_poses(vipe_dir, video_id)
    k_mat, flip = load_vipe_intrinsics(vipe_dir, video_id)
    num_frames = count_video_frames(video_path)
    if num_frames <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    c2w_all = interpolate_c2w_poses(inds, c2w_all, num_frames)
    if flip:
        c2w_all = np.stack([apply_vipe_focal_flip_c2w(p) for p in c2w_all])
    c2w_vipe_world = c2w_all.copy()

    gravity_payload = load_vipe_gravity(vipe_dir, video_id)
    gravity_world = np.asarray(gravity_payload["gravity_world"], dtype=np.float64)
    world_xy_alignment_frame_index = num_frames // 2
    t_zup_from_vipe, world_xy_alignment = gravity_camera_forward_alignment_transform(
        gravity_world,
        c2w_vipe_world[world_xy_alignment_frame_index],
    )
    c2w_zup = np.einsum("ij,tjk->tik", t_zup_from_vipe, c2w_vipe_world)

    objects_summary = []
    object_tracks = []
    object_visibility_tracks = []
    max_track = 0
    for object_id, mesh_filename in zip(object_ids, object_mesh_filenames):
        ob_dir_i = fp_dir / "objects" / object_id / "ob_in_cam"
        if not ob_dir_i.is_dir() and object_id == object_ids[0]:
            ob_dir_i = fp_dir / "ob_in_cam"
        frame_indices_i = sorted(int(p.stem) for p in ob_dir_i.glob("*.txt"))
        object_ob_in_cam_i = []
        ob_world_vipe_i = []
        for fi in frame_indices_i:
            t_obj_cam = np.loadtxt(ob_dir_i / f"{fi:06d}.txt")
            object_ob_in_cam_i.append(t_obj_cam)
            ob_world_vipe_i.append(ob_in_cam_to_world(t_obj_cam, c2w_vipe_world[fi]))
        ob_cam_arr_i = np.stack(object_ob_in_cam_i) if object_ob_in_cam_i else np.zeros((0, 4, 4))
        ob_vipe_arr_i = np.stack(ob_world_vipe_i) if ob_world_vipe_i else np.zeros((0, 4, 4))
        ob_zup_i = np.einsum("ij,njk->nik", t_zup_from_vipe, ob_vipe_arr_i)
        visibility_i = _load_pose_visibility(fp_dir, object_id, num_frames=num_frames)
        if visibility_i["path"] is None:
            visibility_i["pose_valid_by_frame"][frame_indices_i] = True
            visibility_i["visible_by_frame"][frame_indices_i] = True
            visibility_i["pose_valid_frames"] = int(visibility_i["pose_valid_by_frame"].sum())
            visibility_i["visible_frames"] = int(visibility_i["visible_by_frame"].sum())
        object_tracks.append(
            {
                "object_id": object_id,
                "frame_indices": np.asarray(frame_indices_i, dtype=np.int32),
                "ob_in_cam": ob_cam_arr_i,
                "ob_in_vipe_world": ob_vipe_arr_i,
                "ob_in_world": ob_zup_i,
                "mesh_filename": mesh_filename,
            }
        )
        object_visibility_tracks.append(visibility_i)
        max_track = max(max_track, len(frame_indices_i))
        objects_summary.append(
            {
                "object_id": object_id,
                "mesh_filename": mesh_filename,
                "track_frames": len(frame_indices_i),
                "visible_frames": visibility_i["visible_frames"],
                "pose_valid_frames": visibility_i["pose_valid_frames"],
            }
        )
    primary_track = object_tracks[0]
    frame_indices = primary_track["frame_indices"].tolist()
    object_ob_in_cam = list(primary_track["ob_in_cam"])
    ob_world_vipe_arr = primary_track["ob_in_vipe_world"]
    ob_world_zup = primary_track["ob_in_world"]

    n_objects = len(object_tracks)
    object_frame_indices_all = np.full((n_objects, max_track), -1, dtype=np.int32)
    object_valid_all = np.zeros((n_objects, max_track), dtype=bool)
    object_ob_in_cam_all = np.zeros((n_objects, max_track, 4, 4), dtype=np.float64)
    object_ob_in_world_all = np.zeros((n_objects, max_track, 4, 4), dtype=np.float64)
    object_ob_in_vipe_world_all = np.zeros((n_objects, max_track, 4, 4), dtype=np.float64)
    object_visible_by_frame = np.zeros((n_objects, num_frames), dtype=bool)
    object_pose_valid_by_frame = np.zeros((n_objects, num_frames), dtype=bool)
    for idx, track in enumerate(object_tracks):
        n = len(track["frame_indices"])
        visibility_i = object_visibility_tracks[idx]
        object_visible_by_frame[idx] = visibility_i["visible_by_frame"]
        object_pose_valid_by_frame[idx] = visibility_i["pose_valid_by_frame"]
        if n > 0:
            object_frame_indices_all[idx, :n] = track["frame_indices"]
            object_valid_all[idx, :n] = True
            object_ob_in_cam_all[idx, :n] = track["ob_in_cam"]
            object_ob_in_world_all[idx, :n] = track["ob_in_world"]
            object_ob_in_vipe_world_all[idx, :n] = track["ob_in_vipe_world"]

    hawor_world = _resolve_hawor_world(hawor_dir, video_id)
    hand_payload = load_hand_params_numpy(hawor_world) if hawor_world is not None else None
    if hand_payload is not None:
        hand_payload = _transform_hand_payload_to_zup(hand_payload, t_zup_from_vipe[:3, :3])

    masks_manifest = _export_final_2d_masks(
        dataset=dataset,
        video_id=video_id,
        out_dir=out_dir,
        object_prompt_dir=object_prompt_dir,
        sam3_hands_dir=sam3_hands_dir,
        object_ids=object_ids,
    )

    save_kwargs = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "dataset": dataset,
        "video_id": video_id,
        "coordinate_frame": GRAVITY_FRAME,
        "source_coordinate_frame": VIPE_FRAME,
        "T_gravity_z_up_from_vipe_world": t_zup_from_vipe,
        "gravity_direction_vipe_world": gravity_world,
        "up_direction_vipe_world": np.asarray(gravity_payload["up_world"], dtype=np.float64),
        "world_xy_alignment_mode": np.array(WORLD_XY_ALIGNMENT_MODE),
        "world_xy_alignment_frame_index": np.int32(world_xy_alignment_frame_index),
        "world_x_direction_vipe_world": np.asarray(world_xy_alignment["world_x_direction_vipe_world"], dtype=np.float64),
        "world_y_direction_vipe_world": np.asarray(world_xy_alignment["world_y_direction_vipe_world"], dtype=np.float64),
        "world_z_direction_vipe_world": np.asarray(world_xy_alignment["world_z_direction_vipe_world"], dtype=np.float64),
        "world_xy_reference_camera_forward_vipe_world": np.asarray(
            world_xy_alignment["camera_forward_vipe_world"], dtype=np.float64
        ),
        "world_xy_reference_camera_forward_gravity_z_up_pre_yaw": np.asarray(
            world_xy_alignment["camera_forward_gravity_z_up_pre_yaw"], dtype=np.float64
        ),
        "world_xy_reference_camera_forward_xy_pre_yaw": np.asarray(
            world_xy_alignment["camera_forward_xy_pre_yaw"], dtype=np.float64
        ),
        "world_xy_reference_camera_forward_xy_norm": np.float64(world_xy_alignment["camera_forward_xy_norm"]),
        "world_xy_alignment_yaw_angle_rad": np.float64(world_xy_alignment["yaw_angle_rad"]),
        "gravity_sample_frame_indices": np.asarray(gravity_payload["sample_frame_indices"], dtype=np.int32),
        "gravity_cam_samples": np.asarray(gravity_payload["gravity_cam"], dtype=np.float64),
        "up_cam_samples": np.asarray(gravity_payload["up_cam"], dtype=np.float64),
        "mesh_filename": FINAL_MESH,
        "num_objects": np.int32(n_objects),
        "object_ids": np.asarray(object_ids),
        "object_mesh_filenames": np.asarray(object_mesh_filenames),
        "c2w": c2w_zup,
        "c2w_vipe_world": c2w_vipe_world,
        "K": k_mat,
        "object_ob_in_cam": np.stack(object_ob_in_cam) if object_ob_in_cam else np.zeros((0, 4, 4)),
        "object_frame_indices": np.array(frame_indices, dtype=np.int32),
        "object_ob_in_world": ob_world_zup,
        "object_ob_in_vipe_world": ob_world_vipe_arr,
        "object_ob_in_cam_all": object_ob_in_cam_all,
        "object_frame_indices_all": object_frame_indices_all,
        "object_valid_all": object_valid_all,
        "object_ob_in_world_all": object_ob_in_world_all,
        "object_ob_in_vipe_world_all": object_ob_in_vipe_world_all,
        "object_visible_by_frame": object_visible_by_frame,
        "object_pose_valid_by_frame": object_pose_valid_by_frame,
        "mesh_path": FINAL_MESH,
        "num_frames": np.int32(num_frames),
    }
    if hand_payload is not None:
        save_kwargs.update(hand_payload)

    final_npz = out_dir / FINAL_NPZ
    np.savez(final_npz, **save_kwargs)

    summary = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "dataset": dataset,
        "video_id": video_id,
        "num_frames": num_frames,
        "object_track_frames": len(frame_indices),
        "world_npz": FINAL_NPZ,
        "object_mesh": FINAL_MESH,
        "objects": objects_summary,
        "coordinate_frame": GRAVITY_FRAME,
        "source_coordinate_frame": VIPE_FRAME,
        "gravity_alignment": {
            "T_gravity_z_up_from_vipe_world": t_zup_from_vipe.tolist(),
            "gravity_direction_vipe_world": gravity_world.tolist(),
            "up_direction_vipe_world": np.asarray(gravity_payload["up_world"], dtype=np.float64).tolist(),
            "sample_frame_indices": np.asarray(gravity_payload["sample_frame_indices"], dtype=np.int32).tolist(),
            "world_xy_alignment": {
                "mode": WORLD_XY_ALIGNMENT_MODE,
                "frame_index": int(world_xy_alignment_frame_index),
                "world_x_direction_vipe_world": np.asarray(
                    world_xy_alignment["world_x_direction_vipe_world"], dtype=np.float64
                ).tolist(),
                "world_y_direction_vipe_world": np.asarray(
                    world_xy_alignment["world_y_direction_vipe_world"], dtype=np.float64
                ).tolist(),
                "world_z_direction_vipe_world": np.asarray(
                    world_xy_alignment["world_z_direction_vipe_world"], dtype=np.float64
                ).tolist(),
                "reference_camera_forward_vipe_world": np.asarray(
                    world_xy_alignment["camera_forward_vipe_world"], dtype=np.float64
                ).tolist(),
                "reference_camera_forward_xy_norm": float(world_xy_alignment["camera_forward_xy_norm"]),
                "yaw_angle_rad": float(world_xy_alignment["yaw_angle_rad"]),
            },
        },
        "camera": {
            "K": k_mat.tolist(),
            "c2w_shape": list(c2w_zup.shape),
            "vipe_focal_flip_applied": bool(flip),
        },
        "hands_included": hand_payload is not None,
        "masks": masks_manifest,
    }
    if hand_payload is not None:
        summary["hand_param_shapes"] = {k: list(v.shape) for k, v in hand_payload.items()}
    (out_dir / FINAL_SUMMARY).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if visualize:
        object_dir = interim_step_dir(dataset, video_id, "sam2_object")
        primary_object_id = object_ids[0]
        primary_ob_dir = fp_dir / "objects" / primary_object_id / "ob_in_cam"
        if not primary_ob_dir.is_dir():
            primary_ob_dir = fp_dir / "ob_in_cam"
        fuse_vis = vis_path(interim_step_dir(dataset, video_id, "fuse"), video_id)
        hand_geom = None
        if hawor_world is not None:
            hand_geom = compute_hand_sequence_geometry(
                world_result_path=hawor_world,
                c2w_all=c2w_vipe_world,
                num_video_frames=num_frames,
                video_path=video_path,
            )
        if hand_geom is None and hawor_world is not None:
            vis_payload = _render_fuse_vis_subprocess(
                dataset=dataset,
                video_id=video_id,
                video_path=video_path,
                hawor_world=hawor_world,
            )
            if vis_payload and vis_payload.get("vis_video"):
                summary["vis_video"] = vis_payload["vis_video"]
                summary["hands_rendered"] = bool(vis_payload.get("hands_rendered"))
            else:
                vis_path_out = encode_fuse_vis_mp4(
                    video_path=video_path,
                    out_path=fuse_vis,
                    ob_in_cam_dir=primary_ob_dir,
                    frame_indices=frame_indices,
                    k_mat=k_mat,
                    mesh_path=interim_mesh_path,
                    object_masks_dir=object_dir / "video_segmentation" / "masks",
                    object_mask_filename=f"{primary_object_id}.png",
                    hand_geometry=None,
                )
                summary["vis_video"] = str(vis_path_out)
                summary["hands_rendered"] = False
        else:
            vis_path_out = encode_fuse_vis_mp4(
                video_path=video_path,
                out_path=fuse_vis,
                ob_in_cam_dir=primary_ob_dir,
                frame_indices=frame_indices,
                k_mat=k_mat,
                mesh_path=interim_mesh_path,
                object_masks_dir=object_dir / "video_segmentation" / "masks",
                object_mask_filename=f"{primary_object_id}.png",
                hand_geometry=hand_geom,
            )
            summary["vis_video"] = str(vis_path_out)
            summary["hands_rendered"] = hand_geom is not None

    return summary


def validate_final_reconstruction(dataset: str, video_id: str) -> dict:
    out_dir = final_video_dir(dataset, video_id)
    npz_path = out_dir / FINAL_NPZ
    mesh_path = out_dir / FINAL_MESH
    summary_path = out_dir / FINAL_SUMMARY
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing final NPZ: {npz_path}")
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty final mesh: {mesh_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing final summary: {summary_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_FINAL_KEYS - set(data.files))
        if missing:
            raise RuntimeError(f"Final NPZ missing required keys: {missing}")
        schema_version = str(data["schema_version"].item())
        dataset_value = str(data["dataset"].item())
        video_id_value = str(data["video_id"].item())
        if schema_version != FINAL_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported final schema {schema_version}; expected {FINAL_SCHEMA_VERSION}")
        if dataset_value != dataset or video_id_value != video_id:
            raise RuntimeError(
                f"Final identifiers mismatch: ({dataset_value}, {video_id_value}) != ({dataset}, {video_id})"
            )
        num_frames = int(data["num_frames"])
        c2w = data["c2w"]
        k_mat = data["K"]
        obj_cam = data["object_ob_in_cam"]
        obj_world = data["object_ob_in_world"]
        frame_indices = data["object_frame_indices"]
        t_zup = data["T_gravity_z_up_from_vipe_world"]
        gravity = data["gravity_direction_vipe_world"]
        up = data["up_direction_vipe_world"]
        if c2w.shape != (num_frames, 4, 4):
            raise RuntimeError(f"Invalid c2w shape {c2w.shape}; expected {(num_frames, 4, 4)}")
        if k_mat.shape != (3, 3):
            raise RuntimeError(f"Invalid K shape {k_mat.shape}; expected (3, 3)")
        if t_zup.shape != (4, 4):
            raise RuntimeError(f"Invalid T_gravity_z_up_from_vipe_world shape {t_zup.shape}; expected (4, 4)")
        if gravity.shape != (3,):
            raise RuntimeError(f"Invalid gravity_direction_vipe_world shape {gravity.shape}; expected (3,)")
        if up.shape != (3,):
            raise RuntimeError(f"Invalid up_direction_vipe_world shape {up.shape}; expected (3,)")
        r_zup = t_zup[:3, :3]
        if not np.allclose(r_zup @ gravity, np.array([0.0, 0.0, -1.0]), atol=1e-5):
            raise RuntimeError("Gravity alignment must map physical gravity to -Z")
        if not np.allclose(r_zup @ up, np.array([0.0, 0.0, 1.0]), atol=1e-5):
            raise RuntimeError("Gravity alignment must map up direction to +Z")
        world_xy_validation = _validate_world_xy_alignment(data, r_zup_from_vipe=r_zup, num_frames=num_frames)
        if obj_cam.ndim != 3 or obj_cam.shape[1:] != (4, 4) or obj_cam.shape[0] == 0:
            raise RuntimeError(f"Invalid object_ob_in_cam shape {obj_cam.shape}")
        if obj_world.shape != obj_cam.shape:
            raise RuntimeError(f"object_ob_in_world shape {obj_world.shape} does not match {obj_cam.shape}")
        if frame_indices.shape[0] != obj_cam.shape[0]:
            raise RuntimeError("object_frame_indices length does not match object pose count")
        hands_included = {"hand_trans", "hand_rot", "hand_pose", "hand_betas", "hand_valid"}.issubset(data.files)
        if hands_included:
            if data["hand_trans"].shape[:2] != data["hand_valid"].shape:
                raise RuntimeError("hand_trans and hand_valid frame dimensions do not match")
        object_validation = _validate_object_arrays(data, final_dir=out_dir)
        hand_validation = _validate_hand_roots_in_front(data, num_frames=num_frames)
    mask_validation = _validate_final_masks(out_dir, num_frames=num_frames)

    return {
        "schema_version": FINAL_SCHEMA_VERSION,
        "dataset": dataset,
        "video_id": video_id,
        "status": "complete",
        "world_npz": FINAL_NPZ,
        "object_mesh": FINAL_MESH,
        "world_summary": FINAL_SUMMARY,
        "num_frames": num_frames,
        "num_objects": object_validation["num_objects"],
        "object_track_frames": int(frame_indices.shape[0]),
        "hands_included": hands_included,
        "hand_root_front_samples": hand_validation["hand_root_front_samples"],
        "hand_valid_samples": hand_validation["hand_valid_samples"],
        **world_xy_validation,
        **mask_validation,
    }


def write_final_completion(dataset: str, video_id: str) -> Path:
    out_dir = final_video_dir(dataset, video_id)
    payload = validate_final_reconstruction(dataset, video_id)
    marker = out_dir / FINAL_COMPLETION
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def is_final_reconstruction_complete(dataset: str, video_id: str) -> bool:
    marker = final_video_dir(dataset, video_id) / FINAL_COMPLETION
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "complete":
        return False
    try:
        validate_final_reconstruction(dataset, video_id)
    except Exception:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write test overlay MP4 under interim fuse/vis/ (off by default)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    job_video = args.video.resolve()
    out_dir = final_video_dir(args.dataset, args.video_id)
    if not args.force and is_final_reconstruction_complete(args.dataset, args.video_id):
        return 0

    summary = fuse_world_sequence(
        dataset=args.dataset,
        video_id=args.video_id,
        video_path=job_video,
        visualize=args.visualize,
    )
    write_step_completion(
        interim_step_dir(args.dataset, args.video_id, "fuse"),
        "fuse",
        dataset=args.dataset,
        video_id=args.video_id,
        extra=summary,
    )
    write_final_completion(args.dataset, args.video_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
