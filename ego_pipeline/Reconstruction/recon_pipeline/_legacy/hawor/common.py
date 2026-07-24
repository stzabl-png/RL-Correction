"""Shared helpers for HumanVideo2RobotData HaWoR pipeline scripts."""

from __future__ import annotations

import json
import os
import shutil
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
HAWOR_ROOT = REPO_ROOT / "third_party" / "hawor"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "testing" / "hawor" / "hoi4d"
DEFAULT_VIPE_DIR = REPO_ROOT / "data" / "testing" / "vipe" / "hoi4d"
DEFAULT_SAM3_DIR = REPO_ROOT / "data" / "testing" / "sam3" / "hoi4d"
DEFAULT_CHECKPOINT = HAWOR_ROOT / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
DEFAULT_INFILLER = HAWOR_ROOT / "weights" / "hawor" / "checkpoints" / "infiller.pt"

VIPE_SCRIPT_DIR = REPO_ROOT / "recon_pipeline" / "_legacy" / "vipe"
RECON_IO_PATH = REPO_ROOT / "recon_pipeline" / "_common" / "io.py"
HAWOR_VIPE_DISP_MAX_SIDE = 256


def _load_vipe_common():
    import importlib.util

    module_path = VIPE_SCRIPT_DIR / "_common.py"
    spec = importlib.util.spec_from_file_location("vipe_pipeline_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ViPE helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_recon_io():
    import importlib.util

    spec = importlib.util.spec_from_file_location("recon_pipeline_common_io", RECON_IO_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load recon IO helpers from {RECON_IO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_vipe = _load_vipe_common()
_recon_io = _load_recon_io()

DEFAULT_HOI4D_ROOT = _vipe.DEFAULT_HOI4D_ROOT
DEFAULT_RGB_ROOT = _vipe.DEFAULT_RGB_ROOT
cleanup_run_metadata = _vipe.cleanup_run_metadata
discover_hoi4d_video_jobs = _vipe.discover_hoi4d_video_jobs
hoi4d_sequence_name_from_video = _vipe.hoi4d_sequence_name_from_video
is_vipe_sequence_complete = _vipe.is_sequence_complete
parse_gpu_list = _vipe.parse_gpu_list
sequence_lock = _vipe.sequence_lock
setup_vipe_script_imports = _vipe.setup_script_imports
stage_hoi4d_video_for_vipe = _vipe.stage_hoi4d_video_for_vipe
video_jobs_from_list_file = _vipe.video_jobs_from_list_file
worker_log_path = _vipe.worker_log_path
write_status = _vipe.write_status
iter_vipe_depth_frames = _recon_io.iter_vipe_depth_frames

CameraSource = Literal["hawor", "vipe"]

COMPLETION_FILE = "world_space_res.pth"
PRE_FILTER_BACKUP = "world_space_res_pre_sam3_filter.pth"
VIS_FILENAME_SUFFIX = "_vis.mp4"
VIS_SMOOTH_FILENAME_SUFFIX = "_vis_smooth.mp4"
SMOOTH_WORLD_FILENAME = "world_space_res_smooth.pth"
SLAM_FILENAME = "slam.npz"
META_FILENAME = "run_meta.json"
WORK_DIRNAME = "_work"


def setup_script_imports() -> Path:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    return SCRIPT_DIR


def _import_hawor_slam():
    """Import hawor_slam without tripping DROID-SLAM's set_start_method in workers."""
    import torch.multiprocessing as torch_mp

    original = torch_mp.set_start_method

    def _safe_set_start_method(method, force: bool = False) -> None:
        try:
            original(method, force=force)
        except RuntimeError:
            pass

    torch_mp.set_start_method = _safe_set_start_method
    try:
        from scripts.scripts_test_video.hawor_slam import hawor_slam

        return hawor_slam
    finally:
        torch_mp.set_start_method = original


def resolve_filter_legacy_negative_focal(*, img_focal: float) -> bool:
    """Extra w2c flip only for legacy runs that stored negative focal without baking flip into slam."""
    return float(img_focal) < 0


def hawor_raw_prediction_mask(
    pred_trans: "torch.Tensor",
    pred_rot: "torch.Tensor",
    pred_hand_pose: "torch.Tensor",
    pred_betas: "torch.Tensor",
    *,
    eps: float = 1e-6,
) -> "torch.Tensor":
    """Frames with stored HaWoR MANO params, ignoring pred_valid."""
    import torch

    energy = (
        pred_trans.abs().sum(dim=-1)
        + pred_rot.abs().sum(dim=-1)
        + pred_hand_pose.abs().sum(dim=-1)
        + pred_betas.abs().sum(dim=-1)
    )
    return (energy > eps).to(dtype=pred_trans.dtype)


def work_root(output_dir: Path) -> Path:
    return output_dir.resolve() / WORK_DIRNAME


def sequence_staging_root(output_dir: Path, sequence_name: str) -> Path:
    return work_root(output_dir) / sequence_name


def staged_video_path(output_dir: Path, sequence_name: str) -> Path:
    return sequence_staging_root(output_dir, sequence_name) / f"{sequence_name}.mp4"


def hawor_work_dir(output_dir: Path, sequence_name: str) -> Path:
    """HaWoR work dir: _work/{name}/{name}/ next to staged {name}.mp4."""
    return sequence_staging_root(output_dir, sequence_name) / sequence_name


def sequence_output_dir(output_dir: Path, sequence_name: str) -> Path:
    return output_dir.resolve() / sequence_name


def vis_smooth_video_path(output_dir: Path, sequence_name: str) -> Path:
    return sequence_output_dir(output_dir, sequence_name) / (
        f"{sequence_name}{VIS_SMOOTH_FILENAME_SUFFIX}"
    )


def vis_video_path(output_dir: Path, sequence_name: str) -> Path:
    return sequence_output_dir(output_dir, sequence_name) / f"{sequence_name}{VIS_FILENAME_SUFFIX}"


def artifact_paths(output_dir: Path, name: str) -> dict[str, Path]:
    base = sequence_output_dir(output_dir, name)
    return {
        "world": base / COMPLETION_FILE,
        "vis": vis_video_path(output_dir, name),
        "meta": base / META_FILENAME,
        "slam": base / SLAM_FILENAME,
    }


def is_sequence_complete(output_dir: Path, name: str, *, require_vis: bool = True) -> bool:
    paths = artifact_paths(output_dir, name)
    if not paths["world"].is_file():
        return False
    if require_vis and not paths["vis"].is_file():
        return False
    return True


def finalize_sequence_outputs(
    *,
    output_dir: Path,
    sequence_name: str,
    work_dir: Path,
    slam_path: Path,
    meta: dict[str, object],
    start_idx: int,
    end_idx: int,
    img_focal: float,
    visualize: bool,
    cleanup_intermediates: bool,
    smooth_camera_sigma: float = 0.0,
) -> dict[str, str]:
    """Move final artifacts to output_dir/{sequence}/ and optionally remove work dir."""
    seq_out = sequence_output_dir(output_dir, sequence_name)
    seq_out.mkdir(parents=True, exist_ok=True)

    world_src = work_dir / COMPLETION_FILE
    if not world_src.is_file():
        raise FileNotFoundError(f"Missing HaWoR result: {world_src}")

    shutil.copy2(world_src, seq_out / COMPLETION_FILE)
    if slam_path.is_file():
        shutil.copy2(slam_path, seq_out / SLAM_FILENAME)

    vis_path = vis_video_path(output_dir, sequence_name)
    if visualize:
        from visualize_outputs import render_hand_overlay

        render_hand_overlay(
            work_dir=work_dir,
            slam_path=slam_path,
            world_result_path=world_src,
            img_focal=float(img_focal),
            start_idx=start_idx,
            end_idx=end_idx,
            output_mp4=vis_path,
            smooth_camera_sigma=smooth_camera_sigma,
        )
        meta["vis_video"] = str(vis_path)
    else:
        meta["vis_video"] = None

    meta_path = seq_out / META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if cleanup_intermediates:
        staging_root = sequence_staging_root(output_dir, sequence_name)
        if staging_root.is_dir():
            shutil.rmtree(staging_root)
        work = work_root(output_dir)
        if work.is_dir() and not any(work.iterdir()):
            work.rmdir()

    return {
        "world_result": str(seq_out / COMPLETION_FILE),
        "meta": str(meta_path),
        "vis_video": str(vis_path) if vis_path.is_file() else "",
        "slam": str(seq_out / SLAM_FILENAME) if (seq_out / SLAM_FILENAME).is_file() else "",
    }


def cleanup_hawor_run_artifacts(
    output_dir: Path,
    *,
    keep_logs: bool = False,
    keep_run_metadata: bool = False,
) -> None:
    """Remove ephemeral parallel-run bookkeeping after a batch finishes."""
    output_dir = output_dir.resolve()
    if not keep_run_metadata:
        cleanup_run_metadata(output_dir)
    if not keep_logs:
        logs_dir = output_dir / ".logs"
        if logs_dir.is_dir():
            shutil.rmtree(logs_dir)
    work_dir = work_root(output_dir)
    if work_dir.is_dir() and not any(work_dir.iterdir()):
        work_dir.rmdir()


def resolve_camera_source(
    *,
    vipe_camera_params: Path | None,
    sequence_name: str,
) -> tuple[CameraSource, Path | None]:
    if vipe_camera_params is None:
        return "hawor", None

    vipe_root = vipe_camera_params.resolve()
    if not is_vipe_sequence_complete(vipe_root, sequence_name):
        raise FileNotFoundError(
            f"--vipe-camera-params requires complete ViPE artifacts for {sequence_name} under {vipe_root}"
        )
    return "vipe", vipe_root


VIPE_FOCAL_FLIP = np.diag([-1.0, -1.0, 1.0])


def _apply_vipe_focal_flip_c2w(c2w: np.ndarray) -> np.ndarray:
    """Match ViPE negative-focal projection using positive focal + flipped c2w."""
    out = c2w.copy()
    out[:3, :3] = out[:3, :3] @ VIPE_FOCAL_FLIP
    return out


def load_vipe_intrinsics(vipe_dir: Path, sequence_name: str) -> tuple[float, np.ndarray, bool]:
    intr_path = vipe_dir / "intrinsics" / f"{sequence_name}.npz"
    data = np.load(intr_path)
    order = np.argsort(data["inds"])
    intr = data["data"][order][0]
    fx, fy, cx, cy = (float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3]))
    # ViPE SLAM can converge to negative fx/fy (sign ambiguity with poses). Keep |f|
    # and apply _apply_vipe_focal_flip_c2w() to the trajectory when flip_xy is True.
    flip_xy = fx < 0 or fy < 0
    fx, fy = abs(fx), abs(fy)
    focal = (fx + fy) / 2.0
    center = np.array([cx, cy], dtype=np.float64)
    return focal, center, flip_xy


def _interpolate_c2w_poses(
    inds: np.ndarray,
    poses: np.ndarray,
    num_frames: int,
    *,
    target_inds: np.ndarray | None = None,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if len(inds) == 0:
        raise ValueError("ViPE pose file has no frames")

    if target_inds is None:
        target_inds = np.arange(num_frames, dtype=np.float64)
    else:
        target_inds = np.asarray(target_inds, dtype=np.float64)
        if target_inds.shape != (num_frames,):
            raise ValueError(f"target_inds shape {target_inds.shape} does not match num_frames={num_frames}")
    target_inds = np.clip(target_inds, float(inds[0]), float(inds[-1]))

    interp_trans = np.stack(
        [np.interp(target_inds, inds.astype(np.float64), poses[:, axis, 3]) for axis in range(3)],
        axis=1,
    )
    rots = R.from_matrix(poses[:, :3, :3])
    slerp = Slerp(inds.astype(np.float64), rots)
    interp_rots = slerp(target_inds).as_matrix()

    out = np.repeat(np.eye(4, dtype=np.float64)[None, ...], num_frames, axis=0)
    out[:, :3, :3] = interp_rots
    out[:, :3, 3] = interp_trans
    return out


def _resize_depth_for_hawor_slam(depth: np.ndarray, *, max_side: int) -> tuple[np.ndarray, float]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D ViPE depth frame, got shape {depth.shape}")
    height, width = depth.shape
    if max_side <= 0 or max(height, width) <= max_side:
        return depth, 1.0

    import cv2

    scale = float(max_side) / float(max(height, width))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    resized = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32, copy=False), scale


def _depth_to_disparity(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    disp = np.zeros(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    disp[valid] = 1.0 / depth[valid]
    return disp


def _load_vipe_disps_for_hawor_timeline(
    *,
    vipe_dir: Path,
    sequence_name: str,
    target_inds: np.ndarray,
    max_side: int = HAWOR_VIPE_DISP_MAX_SIDE,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source_indices = np.rint(target_inds).astype(np.int64)
    source_indices = np.clip(source_indices, 0, None)
    wanted = set(int(i) for i in source_indices.tolist())
    depth_by_index: dict[int, np.ndarray] = {}

    for frame_idx, depth in iter_vipe_depth_frames(vipe_dir, sequence_name):
        idx = int(frame_idx)
        if idx in wanted:
            depth_by_index[idx] = np.asarray(depth, dtype=np.float32)
            if len(depth_by_index) == len(wanted):
                break

    if not depth_by_index:
        raise FileNotFoundError(
            f"No ViPE depth frames found for {sequence_name} in {vipe_dir / 'depth'}"
        )

    available = np.array(sorted(depth_by_index), dtype=np.int64)
    disps: list[np.ndarray] = []
    used_indices: list[int] = []
    original_shape: tuple[int, int] | None = None
    stored_shape: tuple[int, int] | None = None
    depth_resize_scale: float | None = None

    for src_idx in source_indices:
        src_idx_int = int(src_idx)
        if src_idx_int not in depth_by_index:
            nearest = int(available[np.argmin(np.abs(available - src_idx_int))])
        else:
            nearest = src_idx_int
        depth = depth_by_index[nearest]
        if original_shape is None:
            original_shape = tuple(int(v) for v in depth.shape)
        depth_small, resize_scale = _resize_depth_for_hawor_slam(depth, max_side=max_side)
        if stored_shape is None:
            stored_shape = tuple(int(v) for v in depth_small.shape)
            depth_resize_scale = float(resize_scale)
        disps.append(_depth_to_disparity(depth_small))
        used_indices.append(nearest)

    disp_stack = np.stack(disps, axis=0).astype(np.float32, copy=False)
    used = np.asarray(used_indices, dtype=np.int64)
    metadata = {
        "depth_timeline": "hawor_frames_nearest_vipe_source_frame",
        "depth_original_shape": list(original_shape or ()),
        "depth_stored_shape": list(stored_shape or ()),
        "depth_downsample_max_side": int(max_side),
        "depth_resize_scale": depth_resize_scale,
    }
    return disp_stack, used, metadata


def write_hawor_slam_from_vipe(
    *,
    vipe_dir: Path,
    sequence_name: str,
    num_frames: int,
    source_num_frames: int,
    slam_path: Path,
    img_focal: float,
    img_center: np.ndarray,
    focal_flip_xy: bool = False,
) -> None:
    from scipy.spatial.transform import Rotation as R

    pose_path = vipe_dir / "pose" / f"{sequence_name}.npz"
    data = np.load(pose_path)
    order = np.argsort(data["inds"])
    inds = data["inds"][order]
    poses = data["data"][order]
    # HaWoR extracts frames at 30 fps. HOI4D RGB videos here are 15 fps, so
    # HaWoR has twice as many frames as the source video. The injected ViPE
    # trajectory must be sampled by source-video time, not by HaWoR frame index.
    target_inds = np.linspace(
        0.0,
        float(max(source_num_frames - 1, 0)),
        num_frames,
        dtype=np.float64,
    )
    poses = _interpolate_c2w_poses(inds, poses, num_frames, target_inds=target_inds)
    disps, depth_source_indices, depth_meta = _load_vipe_disps_for_hawor_timeline(
        vipe_dir=vipe_dir,
        sequence_name=sequence_name,
        target_inds=target_inds,
    )

    traj = np.zeros((num_frames, 7), dtype=np.float32)
    for frame_idx in range(num_frames):
        c2w = poses[frame_idx]
        if focal_flip_xy:
            c2w = _apply_vipe_focal_flip_c2w(c2w)
        traj[frame_idx, :3] = c2w[:3, 3].astype(np.float32)
        traj[frame_idx, 3:] = R.from_matrix(c2w[:3, :3]).as_quat().astype(np.float32)

    slam_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        slam_path,
        tstamp=np.arange(num_frames, dtype=np.int64),
        disps=disps,
        traj=traj,
        img_focal=img_focal,
        img_center=img_center,
        scale=1.0,
        camera_source="vipe",
        depth_source="vipe",
        depth_source_indices=depth_source_indices,
        vipe_target_inds=target_inds.astype(np.float32),
        **depth_meta,
    )


def _ensure_hawor_importable() -> None:
    root = str(HAWOR_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _count_video_frames(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if frames <= 0:
        raise RuntimeError(f"Could not count source video frames: {video_path}")
    return frames


def hawor_world_predictions_from_observations(
    args: Namespace,
    start_idx: int,
    end_idx: int,
    frame_chunks_all,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert camera-space HaWoR chunks to world space without temporal infilling."""
    import json
    from glob import glob

    import joblib
    import torch
    from natsort import natsorted

    from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam

    file = args.video_path
    video_root = os.path.dirname(file)
    video = os.path.basename(file).split(".")[0]
    seq_folder = os.path.join(video_root, video)
    img_folder = f"{video_root}/{video}/extracted_images"
    imgfiles = np.array(natsorted(glob(f"{img_folder}/*.jpg")))

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    _R_w2c, _t_w2c, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans = torch.zeros(2, len(imgfiles), 3)
    pred_rot = torch.zeros(2, len(imgfiles), 3)
    pred_hand_pose = torch.zeros(2, len(imgfiles), 45)
    pred_betas = torch.zeros(2, len(imgfiles), 10)
    pred_valid = torch.zeros((2, pred_betas.size(1)))

    for idx in (0, 1):
        frame_chunks = frame_chunks_all[idx]
        if len(frame_chunks) == 0:
            continue
        for frame_ck in frame_chunks:
            pred_path = os.path.join(
                seq_folder, "cam_space", str(idx), f"{frame_ck[0]}_{frame_ck[-1]}.json"
            )
            with open(pred_path, encoding="utf-8") as pred_file:
                pred_dict = json.load(pred_file)
            data_out = {key: torch.tensor(value) for key, value in pred_dict.items()}
            R_c2w_sla = R_c2w_sla_all[frame_ck]
            t_c2w_sla = t_c2w_sla_all[frame_ck]
            handedness = "right" if idx > 0 else "left"
            data_world = cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, handedness)
            pred_trans[[idx], frame_ck] = data_world["init_trans"]
            pred_rot[[idx], frame_ck] = data_world["init_root_orient"]
            pred_hand_pose[[idx], frame_ck] = data_world["init_hand_pose"].flatten(-2)
            pred_betas[[idx], frame_ck] = data_world["init_betas"]
            pred_valid[[idx], frame_ck] = 1

    save_path = os.path.join(seq_folder, "world_space_res.pth")
    joblib.dump([pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid], save_path)
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid


@dataclass
class HaworRunConfig:
    output_dir: Path
    checkpoint: Path = DEFAULT_CHECKPOINT
    infiller_weight: Path = DEFAULT_INFILLER
    vipe_camera_params: Path | None = None
    sam3_dir: Path | None = DEFAULT_SAM3_DIR
    img_focal: float | None = None
    visualize: bool = True
    cleanup_intermediates: bool = True
    use_infiller: bool = False
    smooth_poses: bool = False
    smooth_sigma: float = 2.0
    sam3_filter: bool = True
    sam3_filter_strict: bool = False
    sam3_filter_debug_video: bool = True
    sam_dilation: int = 10
    min_mano_area: int = 80
    mano_containment_threshold: float = 0.50
    joint_containment_threshold: float = 0.50
    alignment_mano_containment_threshold: float = 0.60
    alignment_joint_containment_threshold: float = 0.70


def run_hawor_sequence(
    video_path: Path,
    sequence_name: str,
    config: HaworRunConfig,
    *,
    gpu_id: int = 0,
) -> dict[str, object]:
    """Run HaWoR on one HOI4D clip. Optionally inject ViPE camera poses for SLAM."""
    _ensure_hawor_importable()

    output_dir = config.output_dir.resolve()
    sequence_staging_root(output_dir, sequence_name).mkdir(parents=True, exist_ok=True)
    staged_video = stage_hoi4d_video_for_vipe(
        video_path.resolve(), sequence_name, sequence_staging_root(output_dir, sequence_name)
    )
    work_dir = hawor_work_dir(output_dir, sequence_name)

    if not staged_video.is_symlink() and not staged_video.exists():
        raise FileNotFoundError(f"Failed to stage video for {sequence_name}")

    camera_source, vipe_dir = resolve_camera_source(
        vipe_camera_params=config.vipe_camera_params,
        sequence_name=sequence_name,
    )

    img_focal = config.img_focal
    img_center: np.ndarray | None = None
    focal_flip_xy = False
    if camera_source == "vipe" and vipe_dir is not None:
        img_focal, img_center, focal_flip_xy = load_vipe_intrinsics(vipe_dir, sequence_name)

    prev_cwd = os.getcwd()
    os.chdir(HAWOR_ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    args = Namespace(
        video_path=str(staged_video.absolute()),
        input_type="file",
        checkpoint=str(config.checkpoint.resolve()),
        infiller_weight=str(config.infiller_weight.resolve()),
        img_focal=img_focal,
        vis_mode="cam",
    )

    try:
        from scripts.scripts_test_video.detect_track_video import detect_track_video
        from scripts.scripts_test_video.hawor_video import hawor_motion_estimation

        start_idx, end_idx, seq_folder, _imgfiles = detect_track_video(args)
        if str(Path(seq_folder).resolve()) != str(work_dir.resolve()):
            raise RuntimeError(
                f"Unexpected HaWoR work dir {seq_folder}; expected {work_dir}. "
                "Check staged video path and sequence naming."
            )

        if img_focal is not None:
            with open(Path(seq_folder) / "est_focal.txt", "w", encoding="utf-8") as focal_file:
                focal_file.write(str(img_focal))

        frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

        slam_path = Path(seq_folder) / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
        if camera_source == "vipe":
            if vipe_dir is None:
                raise RuntimeError("camera_source=vipe but vipe_dir is None")
            if img_center is None:
                img_focal, img_center, focal_flip_xy = load_vipe_intrinsics(vipe_dir, sequence_name)
            num_frames = end_idx - start_idx
            write_hawor_slam_from_vipe(
                vipe_dir=vipe_dir,
                sequence_name=sequence_name,
                num_frames=num_frames,
                source_num_frames=_count_video_frames(video_path),
                slam_path=slam_path,
                img_focal=float(img_focal),
                img_center=img_center,
                focal_flip_xy=focal_flip_xy,
            )
        elif not slam_path.is_file():
            hawor_slam = _import_hawor_slam()
            hawor_slam(args, start_idx, end_idx)

        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = (
            hawor_world_predictions_from_observations(
                args, start_idx, end_idx, frame_chunks_all
            )
        )
        obs_valid = pred_valid.clone()
        filter_keep_valid = None
        filter_summary: dict | None = None

        if config.sam3_filter:
            if config.sam3_dir is None:
                raise RuntimeError("sam3_filter is enabled but sam3_dir is None")
            import cv2
            import torch
            from pose_smoothing import save_world_poses
            from sam3_mano_filter import (
                FILTER_ARTIFACT,
                FILTER_VIS_SUFFIX,
                Sam3ManoFilterConfig,
                filter_world_poses_with_sam3,
                render_filter_debug_video,
                write_filter_artifact,
            )

            img_files = sorted((Path(seq_folder) / "extracted_images").glob("*.jpg"))
            if not img_files:
                raise RuntimeError(f"No extracted frames under {seq_folder}/extracted_images")
            sample = cv2.imread(str(img_files[0]))
            if sample is None:
                raise RuntimeError(f"Failed to read sample frame: {img_files[0]}")
            height, width = sample.shape[:2]
            filter_cfg = Sam3ManoFilterConfig(
                sam_dilation=config.sam_dilation,
                min_mano_area=config.min_mano_area,
                mano_containment_threshold=config.mano_containment_threshold,
                joint_containment_threshold=config.joint_containment_threshold,
                alignment_mano_containment_threshold=config.alignment_mano_containment_threshold,
                alignment_joint_containment_threshold=config.alignment_joint_containment_threshold,
                write_debug_video=config.sam3_filter_debug_video,
            )
            seq_out = sequence_output_dir(output_dir, sequence_name)
            seq_out.mkdir(parents=True, exist_ok=True)
            pre_filter_path = seq_out / PRE_FILTER_BACKUP
            if not pre_filter_path.is_file():
                save_world_poses(
                    pre_filter_path,
                    pred_trans,
                    pred_rot,
                    pred_hand_pose,
                    pred_betas,
                    obs_valid,
                )
            filter_before = (int(obs_valid[0].sum()), int(obs_valid[1].sum()))
            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, filter_summary = (
                filter_world_poses_with_sam3(
                    pred_trans=pred_trans,
                    pred_rot=pred_rot,
                    pred_hand_pose=pred_hand_pose,
                    pred_betas=pred_betas,
                    pred_valid=pred_valid,
                    slam_path=slam_path,
                    img_focal=float(img_focal),
                    sam3_dir=config.sam3_dir.resolve(),
                    sequence_name=sequence_name,
                    num_frames=end_idx - start_idx,
                    image_size=(height, width),
                    cfg=filter_cfg,
                    candidate_valid=obs_valid,
                    legacy_negative_focal=resolve_filter_legacy_negative_focal(
                        img_focal=float(img_focal)
                    ),
                    strict=config.sam3_filter_strict,
                    source_video=video_path,
                )
            )
            filter_after = (int(pred_valid[0].sum()), int(pred_valid[1].sum()))
            filter_keep_valid = pred_valid.clone()
            save_world_poses(
                Path(seq_folder) / "world_space_res.pth",
                pred_trans,
                pred_rot,
                pred_hand_pose,
                pred_betas,
                pred_valid,
            )
            write_filter_artifact(seq_out / FILTER_ARTIFACT, filter_summary)
            if config.sam3_filter_debug_video:
                render_filter_debug_video(
                    source_video=video_path,
                    sam3_dir=config.sam3_dir.resolve(),
                    sequence_name=sequence_name,
                    filter_summary=filter_summary,
                    output_mp4=seq_out / f"{sequence_name}{FILTER_VIS_SUFFIX}",
                    slam_path=slam_path,
                    world_result_path=Path(seq_folder) / "world_space_res.pth",
                    img_focal=float(img_focal),
                )

        if config.use_infiller:
            from scripts.scripts_test_video.hawor_video import hawor_infiller

            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
                args, start_idx, end_idx, frame_chunks_all
            )
            if config.sam3_filter and filter_keep_valid is not None:
                pred_valid = filter_keep_valid
                from pose_smoothing import save_world_poses

                save_world_poses(
                    Path(seq_folder) / "world_space_res.pth",
                    pred_trans,
                    pred_rot,
                    pred_hand_pose,
                    pred_betas,
                    pred_valid,
                )

        if config.smooth_poses:
            from pose_smoothing import save_world_poses, smooth_world_poses

            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = smooth_world_poses(
                pred_trans,
                pred_rot,
                pred_hand_pose,
                pred_betas,
                pred_valid,
                sigma=config.smooth_sigma,
            )
            save_world_poses(
                Path(seq_folder) / "world_space_res.pth",
                pred_trans,
                pred_rot,
                pred_hand_pose,
                pred_betas,
                pred_valid,
            )

        meta = {
            "sequence": sequence_name,
            "video": str(video_path.resolve()),
            "camera_source": camera_source,
            "infiller": config.use_infiller,
            "smooth_poses": config.smooth_poses,
            "smooth_sigma": float(config.smooth_sigma) if config.smooth_poses else None,
            "smooth_camera_sigma": float(config.smooth_sigma) if config.smooth_poses else None,
            "vipe_camera_params": str(config.vipe_camera_params.resolve())
            if config.vipe_camera_params is not None
            else None,
            "img_focal": float(img_focal),
            "vipe_focal_flip": focal_flip_xy if camera_source == "vipe" else None,
            "depth_source": "vipe" if camera_source == "vipe" else "hawor_metric3d",
            "depth_representation": (
                "inverse_depth_disparity" if camera_source == "vipe" else "hawor_slam_disparity"
            ),
            "depth_downsample_max_side": HAWOR_VIPE_DISP_MAX_SIDE if camera_source == "vipe" else None,
            "sam3_filter": config.sam3_filter,
            "sam3_dir": str(config.sam3_dir.resolve()) if config.sam3_dir is not None else None,
            "sam3_filter_strict": config.sam3_filter_strict,
            "num_frames": int(pred_trans.shape[1]),
        }
        if filter_summary is not None:
            meta["sam3_filter_summary"] = {
                "accepted_frames": filter_summary["accepted_frames"],
                "rejected_frames": filter_summary["rejected_frames"],
                "before_valid_lr": list(filter_before),
                "after_valid_lr": list(filter_after),
                "frame_mapping": filter_summary.get("frame_mapping"),
            }

        finalized = finalize_sequence_outputs(
            output_dir=output_dir,
            sequence_name=sequence_name,
            work_dir=Path(seq_folder),
            slam_path=slam_path,
            meta=meta,
            start_idx=start_idx,
            end_idx=end_idx,
            img_focal=float(img_focal),
            visualize=config.visualize,
            cleanup_intermediates=config.cleanup_intermediates,
            smooth_camera_sigma=config.smooth_sigma if config.smooth_poses else 0.0,
        )

        return {
            "sequence": sequence_name,
            "camera_source": camera_source,
            "world_result": finalized["world_result"],
            "vis_video": finalized["vis_video"],
            "meta": finalized["meta"],
            "pred_shapes": {
                "trans": list(pred_trans.shape),
                "rot": list(pred_rot.shape),
                "hand_pose": list(pred_hand_pose.shape),
                "betas": list(pred_betas.shape),
                "valid": list(pred_valid.shape),
            },
        }
    finally:
        os.chdir(prev_cwd)


__all__ = [
    "CameraSource",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_HOI4D_ROOT",
    "DEFAULT_INFILLER",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SAM3_DIR",
    "DEFAULT_RGB_ROOT",
    "DEFAULT_VIPE_DIR",
    "HAWOR_ROOT",
    "HaworRunConfig",
    "REPO_ROOT",
    "SLAM_FILENAME",
    "SMOOTH_WORLD_FILENAME",
    "artifact_paths",
    "cleanup_hawor_run_artifacts",
    "cleanup_run_metadata",
    "finalize_sequence_outputs",
    "hawor_world_predictions_from_observations",
    "discover_hoi4d_video_jobs",
    "hawor_work_dir",
    "hoi4d_sequence_name_from_video",
    "is_sequence_complete",
    "parse_gpu_list",
    "resolve_camera_source",
    "run_hawor_sequence",
    "sequence_lock",
    "sequence_output_dir",
    "setup_script_imports",
    "staged_video_path",
    "vis_video_path",
    "vis_smooth_video_path",
    "video_jobs_from_list_file",
    "work_root",
    "worker_log_path",
    "write_status",
]
