"""FoundationPose object pose estimation (Python backend).

Default mode uses the original register-once + FP++ tracking path: register on
the labeled frame, then use per-frame SAM2 mask centroids plus a 6D Kalman
filter before each ``track_one`` frame. The tracker runs both forward and
backward from the labeled frame so every video frame receives an object pose.
``register-each`` is available as a slower diagnostic mode that runs
FoundationPose ``register`` on every frame with a non-empty mask.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

RECON_ROOT = Path(__file__).resolve().parents[1]
FP_ROOT = RECON_ROOT.parent / "third_party" / "FoundationPose"

from _common.artifacts import discard_fp_pose_nonessential  # noqa: E402
from _common.io import count_video_frames, iter_vipe_depth_frames, load_vipe_intrinsics  # noqa: E402
from _common.paths import interim_step_dir  # noqa: E402
from fp_pose.kalman_filter_6d import (  # noqa: E402
    KalmanFilter6D,
    adjust_xy,
    mask_centroid,
    mat_from_6d,
    mat_to_6d,
)
from sam2_object.sam2_object_common import OBJECT_MASK_ID, load_label_prompt, object_mask_filename  # noqa: E402

FP_POSE_META_BASENAME = "fp_pose_meta.json"
OB_IN_CAM_DIRNAME = "ob_in_cam"
DEFAULT_POSE_MODE = "track"
POSE_MODES = ("register-each", "track")
BACKEND_BY_POSE_MODE = {
    "track": "foundationpose_python_pp",
    "register-each": "foundationpose_python_register_each",
}


def object_mask_path(object_dir: Path, frame_idx: int, object_id: str = OBJECT_MASK_ID) -> Path:
    return (
        object_dir
        / "video_segmentation"
        / "masks"
        / f"frame_{frame_idx:06d}_masks"
        / object_mask_filename(object_id)
    )


def scaled_mesh_path(dataset: str, video_id: str, object_id: str = OBJECT_MASK_ID) -> Path:
    base = interim_step_dir(dataset, video_id, "sam3d_scale")
    obj_path = base / "objects" / object_id / "object_mesh_scaled_final.obj"
    if obj_path.is_file():
        return obj_path
    return base / "object_mesh_scaled_final.obj"


def ob_in_cam_path(step_dir: Path, frame_idx: int) -> Path:
    return step_dir / OB_IN_CAM_DIRNAME / f"{frame_idx:06d}.txt"


def _ensure_fp_importable() -> None:
    root = str(FP_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_foundation_pose(mesh_path: Path, *, gpu: int, debug_dir: Path):
    _ensure_fp_importable()
    import trimesh
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor  # type: ignore

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    debug_dir.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=None,
        debug=0,
        debug_dir=str(debug_dir),
    )
    return est


def _read_mask(object_dir: Path, frame_idx: int, object_id: str = OBJECT_MASK_ID) -> np.ndarray:
    import cv2

    path = object_mask_path(object_dir, frame_idx, object_id)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((1, 1), dtype=np.uint8)
    return mask


def _depth_to_meters(depth: np.ndarray, scale: float) -> np.ndarray:
    d = depth.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    return d * scale


def _prepare_frame(rgb_bgr: np.ndarray, depth_m: np.ndarray, mask: np.ndarray):
    """Return RGB (HxWx3), depth, mask cropped to even H/W for FoundationPose."""
    import cv2

    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb_rgb.shape[:2]
    if w % 2 or h % 2:
        h2, w2 = h - h % 2, w - w % 2
        rgb_rgb = rgb_rgb[:h2, :w2]
        depth_m = depth_m[:h2, :w2]
        mask = mask[:h2, :w2]
    return rgb_rgb, depth_m, mask


def _mask_has_foreground(mask: np.ndarray) -> bool:
    return bool(np.any(mask > 0))


def _pose_last_as_torch(est, pose: np.ndarray):
    import torch

    if not isinstance(est.pose_last, torch.Tensor):
        est.pose_last = torch.as_tensor(pose, device="cuda", dtype=torch.float32)


def _load_depth_frames(vipe_dir: Path, video_id: str, *, num_frames: int, depth_scale: float) -> dict[int, np.ndarray]:
    depths: dict[int, np.ndarray] = {}
    for idx, depth in iter_vipe_depth_frames(vipe_dir, video_id):
        idx = int(idx)
        if 0 <= idx < num_frames:
            depths[idx] = _depth_to_meters(depth, depth_scale)
    missing = [idx for idx in range(num_frames) if idx not in depths]
    if missing:
        preview = ",".join(str(i) for i in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise FileNotFoundError(f"Missing ViPE depth frames for {video_id}: {preview}{suffix}")
    return depths


def _read_video_frames(video_path: Path, *, num_frames: int) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for frame_idx in range(num_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _fp_pp_init_pose(
    *,
    prev_pose: np.ndarray,
    mask: np.ndarray,
    k_mat: np.ndarray,
    kf: KalmanFilter6D | None,
    mean,
    cov,
) -> tuple[np.ndarray, object, object]:
    """STEP_6_pose ++ init: centroid xy correction with optional 6D Kalman."""
    uv = mask_centroid(mask)
    init = prev_pose
    if uv is None:
        return init, mean, cov

    if kf is not None and mean is not None and cov is not None:
        mean, cov = kf.update(mean, cov, mat_to_6d(init))
        tz = init[2, 3]
        meas_xy = np.array(
            [(uv[0] - k_mat[0, 2]) * tz / k_mat[0, 0], (uv[1] - k_mat[1, 2]) * tz / k_mat[1, 1]]
        )
        mean, cov = kf.update_from_xy(mean, cov, meas_xy)
        init = mat_from_6d(mean[:6])
    else:
        init = adjust_xy(init, k_mat, uv[0], uv[1])
    return init, mean, cov


def _run_fp_pp_track_one(
    *,
    dataset: str,
    video_id: str,
    video_path: Path,
    object_id: str,
    obj_prompt,
    gpu: int,
    depth_scale: float = 1.0,
    use_kf: bool = True,
    kf_scale: float = 0.05,
    register_iters: int = 5,
    track_iters: int = 2,
    pose_mode: str = DEFAULT_POSE_MODE,
) -> dict:
    if pose_mode not in POSE_MODES:
        raise ValueError(f"Unsupported pose_mode={pose_mode!r}; expected one of {POSE_MODES}")

    total_t0 = time.perf_counter()
    vipe_dir = interim_step_dir(dataset, video_id, "vipe")
    object_dir = interim_step_dir(dataset, video_id, "sam2_object")
    base_step_dir = interim_step_dir(dataset, video_id, "fp_pose")
    step_dir = base_step_dir / "objects" / object_id
    step_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = scaled_mesh_path(dataset, video_id, object_id)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing final scaled object mesh: {mesh_path}. Run sam3d_scale before fp_pose.")

    anchor_frame = int(obj_prompt.frame_idx)
    num_frames = count_video_frames(video_path)
    # 帧计划(frame_plan.json): FP 配准帧 = 交互开始帧+10(同事实测更优)。计划只是建议:
    # 越界或该帧 mask 为空 → 回退 prompt 帧并打日志。
    from _common.frame_plan import load_frame_plan, planned_frame
    _pf = planned_frame(load_frame_plan(object_dir), object_id, "fp_register_frame")
    if _pf is not None:
        if not 0 <= _pf < num_frames:
            print(f"[fp_pose] frame_plan {object_id}: 计划帧 {_pf} 越界, 回退 prompt 帧 {anchor_frame}", flush=True)
        elif not _mask_has_foreground(_read_mask(object_dir, _pf, object_id)):
            print(f"[fp_pose] frame_plan {object_id}: 计划帧 {_pf} mask 为空, 回退 prompt 帧 {anchor_frame}", flush=True)
        else:
            print(f"[fp_pose] frame_plan {object_id}: 配准帧 {anchor_frame} -> {_pf}", flush=True)
            anchor_frame = _pf
    if not 0 <= anchor_frame < num_frames:
        raise ValueError(f"Object prompt frame {anchor_frame} is outside video frame range [0, {num_frames})")

    k_mat, _flip = load_vipe_intrinsics(vipe_dir, video_id)

    load_t0 = time.perf_counter()
    est = _load_foundation_pose(mesh_path, gpu=gpu, debug_dir=step_dir / "fp_debug")
    model_load_sec = time.perf_counter() - load_t0
    ob_in_cam_dir = step_dir / OB_IN_CAM_DIRNAME
    if ob_in_cam_dir.exists():
        shutil.rmtree(ob_in_cam_dir)
    ob_in_cam_dir.mkdir(parents=True, exist_ok=True)

    io_t0 = time.perf_counter()
    frames_bgr = _read_video_frames(video_path, num_frames=num_frames)
    depth_by_frame = _load_depth_frames(vipe_dir, video_id, num_frames=num_frames, depth_scale=depth_scale)
    io_prep_sec = time.perf_counter() - io_t0

    register_sec = 0.0
    track_sec = 0.0
    save_sec = 0.0
    num_tracked = 0
    num_registered = 0
    num_track_fallback = 0

    def prepare(frame_idx: int):
        nonlocal io_prep_sec
        io_t = time.perf_counter()
        mask = _read_mask(object_dir, frame_idx, object_id)
        rgb_rgb, depth_m, mask = _prepare_frame(frames_bgr[frame_idx], depth_by_frame[frame_idx], mask)
        io_prep_sec += time.perf_counter() - io_t
        return rgb_rgb, depth_m, mask

    def save_pose(frame_idx: int, pose: np.ndarray) -> None:
        nonlocal num_tracked, save_sec
        save_t = time.perf_counter()
        np.savetxt(ob_in_cam_path(step_dir, frame_idx), np.asarray(pose, dtype=np.float64), fmt="%.8f")
        save_sec += time.perf_counter() - save_t
        num_tracked += 1

    rgb_anchor, depth_anchor, mask_anchor = prepare(anchor_frame)
    if not _mask_has_foreground(mask_anchor):
        raise RuntimeError(f"Empty object mask on FP registration anchor frame {anchor_frame}")

    register_t0 = time.perf_counter()
    anchor_pose = np.asarray(
        est.register(k_mat, rgb_anchor, depth_anchor, mask_anchor > 0, iteration=register_iters),
        dtype=np.float64,
    )
    register_sec += time.perf_counter() - register_t0
    num_registered += 1
    _pose_last_as_torch(est, anchor_pose)
    save_pose(anchor_frame, anchor_pose)

    def track_sequence(frame_indices: list[int], initial_pose: np.ndarray) -> None:
        nonlocal num_registered, num_track_fallback, register_sec, track_sec
        prev_pose = np.asarray(initial_pose, dtype=np.float64)
        mean = cov = None
        kf = KalmanFilter6D(kf_scale) if use_kf else None
        if kf is not None:
            mean, cov = kf.initiate(mat_to_6d(prev_pose))
        est.pose_last = prev_pose
        _pose_last_as_torch(est, prev_pose)

        for frame_idx in frame_indices:
            rgb_rgb, depth_m, mask = prepare(frame_idx)
            if pose_mode == "register-each" and _mask_has_foreground(mask):
                register_t = time.perf_counter()
                pose = est.register(k_mat, rgb_rgb, depth_m, mask > 0, iteration=register_iters)
                register_sec += time.perf_counter() - register_t
                num_registered += 1
                _pose_last_as_torch(est, pose)
                if kf is not None:
                    mean, cov = kf.initiate(mat_to_6d(pose))
            else:
                if pose_mode == "register-each" and not _mask_has_foreground(mask):
                    num_track_fallback += 1
                init, mean, cov = _fp_pp_init_pose(
                    prev_pose=prev_pose,
                    mask=mask,
                    k_mat=k_mat,
                    kf=kf,
                    mean=mean,
                    cov=cov,
                )
                est.pose_last = init
                _pose_last_as_torch(est, init)
                track_t = time.perf_counter()
                pose = est.track_one(rgb_rgb, depth_m, k_mat, iteration=track_iters)
                track_sec += time.perf_counter() - track_t
                if kf is not None and mean is not None and cov is not None:
                    mean, cov = kf.predict(mean, cov)
            prev_pose = np.asarray(pose, dtype=np.float64)
            save_pose(frame_idx, prev_pose)

    track_sequence(list(range(anchor_frame + 1, num_frames)), anchor_pose)
    track_sequence(list(range(anchor_frame - 1, -1, -1)), anchor_pose)

    discard_fp_pose_nonessential(step_dir)

    register_path = ob_in_cam_path(step_dir, anchor_frame)
    meta = {
        "object_id": object_id,
        "anchor_frame": anchor_frame,
        "start_frame": 0,
        "end_frame": num_frames - 1,
        "num_video_frames": num_frames,
        "num_frames_tracked": num_tracked,
        "num_frames_registered": num_registered,
        "num_track_fallback": num_track_fallback,
        "tracking_direction": "bidirectional_from_anchor",
        "mesh_path": str(mesh_path),
        "ob_in_cam_dir": str(ob_in_cam_dir),
        "register_pose_path": str(register_path),
        "backend": BACKEND_BY_POSE_MODE[pose_mode],
        "pose_mode": pose_mode,
        "depth_scale": depth_scale,
        "use_kf": use_kf,
        "kf_scale": kf_scale,
        "register_iters": register_iters,
        "track_iters": track_iters,
        "timing_sec": {
            "total": time.perf_counter() - total_t0,
            "model_load": model_load_sec,
            "io_and_frame_prep": io_prep_sec,
            "foundationpose_register": register_sec,
            "foundationpose_track": track_sec,
            "pose_txt_save": save_sec,
        },
    }
    (step_dir / FP_POSE_META_BASENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def run_fp_pp_track(
    *,
    dataset: str,
    video_id: str,
    video_path: Path,
    gpu: int,
    depth_scale: float = 1.0,
    use_kf: bool = True,
    kf_scale: float = 0.05,
    register_iters: int = 5,
    track_iters: int = 2,
    pose_mode: str = DEFAULT_POSE_MODE,
) -> dict:
    base_step_dir = interim_step_dir(dataset, video_id, "fp_pose")
    base_step_dir.mkdir(parents=True, exist_ok=True)
    object_dir = interim_step_dir(dataset, video_id, "sam2_object")
    prompt = load_label_prompt(object_dir)
    objects = []
    for obj in prompt.objects:
        meta_i = _run_fp_pp_track_one(
            dataset=dataset,
            video_id=video_id,
            video_path=video_path,
            object_id=obj.object_id,
            obj_prompt=obj,
            gpu=gpu,
            depth_scale=depth_scale,
            use_kf=use_kf,
            kf_scale=kf_scale,
            register_iters=register_iters,
            track_iters=track_iters,
            pose_mode=pose_mode,
        )
        objects.append(meta_i)

    if not objects:
        raise RuntimeError("No labeled objects found for fp_pose")

    primary = objects[0]
    primary_dir = base_step_dir / "objects" / primary["object_id"]
    shutil.rmtree(base_step_dir / OB_IN_CAM_DIRNAME, ignore_errors=True)
    src_ob = primary_dir / OB_IN_CAM_DIRNAME
    if src_ob.is_dir():
        shutil.copytree(src_ob, base_step_dir / OB_IN_CAM_DIRNAME)
    src_meta = primary_dir / FP_POSE_META_BASENAME
    if src_meta.is_file():
        shutil.copy2(src_meta, base_step_dir / FP_POSE_META_BASENAME)

    meta = {
        **primary,
        "num_objects": len(objects),
        "object_ids": [item["object_id"] for item in objects],
        "objects": objects,
    }
    (base_step_dir / FP_POSE_META_BASENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def ob_in_cam_to_world(ob_in_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return c2w @ ob_in_cam
