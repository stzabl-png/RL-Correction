"""Render HaWoR hand meshes overlaid on the source video (PyTorch3D, headless)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from natsort import natsorted

MANO_OPENPOSE_JOINTS = 21


def _mano_face_sets():
    from hawor.utils.process import get_mano_faces

    faces = get_mano_faces()
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
        ]
    )
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_left, faces_right


def _blend_hand(
    background_bgr: np.ndarray,
    renderer,
    faces: np.ndarray,
    verts: torch.Tensor,
    color_rgb: tuple[float, float, float],
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
) -> np.ndarray:
    device = renderer.device
    faces_t = torch.from_numpy(faces).long().to(device)
    colors = torch.tensor([color_rgb], dtype=torch.float32, device=device)
    cameras, lights = renderer.create_camera_from_cv(
        R_w2c.to(device),
        t_w2c.to(device),
    )
    rend, mask = renderer.render_multiple(
        [verts.to(device)],
        faces_t,
        colors,
        cameras,
        lights,
    )
    out = background_bgr.copy()
    rend_bgr = cv2.cvtColor(rend, cv2.COLOR_RGB2BGR)
    out[mask] = rend_bgr[mask]
    return out


def _project_world_joints(
    joints_3d: torch.Tensor,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
    K: torch.Tensor,
) -> np.ndarray:
    from lib.vis.renderer import perspective_projection

    device = K.device
    x3d = joints_3d.unsqueeze(0).to(device)
    x2d = perspective_projection(
        x3d,
        K,
        R=R_w2c.to(device),
        T=t_w2c.unsqueeze(-1).to(device),
    )
    return x2d[0].cpu().numpy()


def _draw_hand_joints(frame_bgr: np.ndarray, joints_2d: np.ndarray) -> np.ndarray:
    from hawor.utils.render_openpose import render_hand_keypoints

    keypoints = np.concatenate(
        [joints_2d[:MANO_OPENPOSE_JOINTS], np.ones((MANO_OPENPOSE_JOINTS, 1), dtype=np.float32)],
        axis=1,
    )
    return render_hand_keypoints(frame_bgr, keypoints)


MANO_OPENPOSE_JOINTS = 21
HAWOR_EXTRACT_FPS = 30
VIPE_FOCAL_FLIP = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))


def _normalize_vipe_focal_and_cameras(
    img_focal: float,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Legacy runs stored negative ViPE focal without flipped poses; fix both."""
    legacy_flip = img_focal < 0
    img_focal = abs(float(img_focal))
    if legacy_flip:
        R_w2c = torch.einsum("ij,njk->nik", VIPE_FOCAL_FLIP, R_w2c)
        t_w2c = torch.einsum("ij,nj->ni", VIPE_FOCAL_FLIP, t_w2c)
    return img_focal, R_w2c, t_w2c


def _extract_video_frames(source_video: Path, *, fps: float = HAWOR_EXTRACT_FPS) -> Path:
    """Extract frames the same way HaWoR detect_track_video does (fps=30)."""
    temp_dir = Path(tempfile.mkdtemp(prefix="hawor_vis_frames_"))
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-vf",
        f"fps={fps}",
        "-start_number",
        "0",
        str(temp_dir / "%04d.jpg"),
    ]
    subprocess.run(command, check=True)
    if not any(temp_dir.glob("*.jpg")):
        raise RuntimeError(f"ffmpeg produced no frames for {source_video}")
    return temp_dir


def _load_background_frames(
    *,
    work_dir: Path | None,
    source_video: Path | None,
    start_idx: int,
    end_idx: int,
) -> tuple[list[np.ndarray], float, Path | None]:
    if (work_dir is None) == (source_video is None):
        raise ValueError("Provide exactly one of work_dir or source_video")

    temp_dir: Path | None = None
    if source_video is not None:
        source_video = source_video.resolve()
        if not source_video.is_file():
            raise FileNotFoundError(f"Source video not found: {source_video}")
        temp_dir = _extract_video_frames(source_video)
        imgfiles = natsorted(glob(str(temp_dir / "*.jpg")))
    else:
        assert work_dir is not None
        imgfiles = natsorted(glob(str(work_dir / "extracted_images" / "*.jpg")))
        if not imgfiles:
            raise FileNotFoundError(f"No extracted frames under {work_dir / 'extracted_images'}")

    if end_idx > len(imgfiles):
        raise RuntimeError(
            f"Requested frames [{start_idx}, {end_idx}) but only {len(imgfiles)} extracted frames exist"
        )

    frames = []
    for img_path in imgfiles[start_idx:end_idx]:
        frame = cv2.imread(img_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read frame: {img_path}")
        frames.append(frame)
    return frames, float(HAWOR_EXTRACT_FPS), temp_dir


def render_hand_overlay(
    *,
    work_dir: Path | None = None,
    source_video: Path | None = None,
    slam_path: Path,
    world_result_path: Path,
    img_focal: float,
    start_idx: int,
    end_idx: int,
    output_mp4: Path,
    fps: float = 30.0,
    smooth_camera_sigma: float = 0.0,
) -> Path:
    """Overlay estimated MANO hands on extracted frames; write MP4 to output_mp4."""
    from lib.eval_utils.custom_utils import load_slam_cam
    from lib.vis.renderer import Renderer
    from hawor.utils.process import run_mano, run_mano_left

    slam_path = slam_path.resolve()
    world_result_path = world_result_path.resolve()
    output_mp4 = output_mp4.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_result_path)
    background_frames, detected_fps, temp_frame_dir = _load_background_frames(
        work_dir=work_dir.resolve() if work_dir is not None else None,
        source_video=source_video,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    if fps <= 0:
        fps = detected_fps

    vis_start = start_idx
    vis_end = end_idx

    try:
        faces_left, faces_right = _mano_face_sets()

        pred_glob_r = run_mano(
            pred_trans[1:2, vis_start:vis_end],
            pred_rot[1:2, vis_start:vis_end],
            pred_hand_pose[1:2, vis_start:vis_end],
            betas=pred_betas[1:2, vis_start:vis_end],
        )
        right_verts = pred_glob_r["vertices"][0]
        right_joints = pred_glob_r["joints"][0]

        pred_glob_l = run_mano_left(
            pred_trans[0:1, vis_start:vis_end],
            pred_rot[0:1, vis_start:vis_end],
            pred_hand_pose[0:1, vis_start:vis_end],
            betas=pred_betas[0:1, vis_start:vis_end],
        )
        left_verts = pred_glob_l["vertices"][0]
        left_joints = pred_glob_l["joints"][0]

        R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(str(slam_path))

        R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
        R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
        t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
        R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
        t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
        if smooth_camera_sigma > 0.0:
            from pose_smoothing import smooth_slam_cameras

            R_w2c_sla_all, t_w2c_sla_all = smooth_slam_cameras(
                R_w2c_sla_all,
                t_w2c_sla_all,
                sigma=smooth_camera_sigma,
            )
        img_focal, R_w2c_sla_all, t_w2c_sla_all = _normalize_vipe_focal_and_cameras(
            img_focal,
            R_w2c_sla_all,
            t_w2c_sla_all,
        )
        left_verts = torch.einsum("ij,tnj->tni", R_x, left_verts.cpu())
        right_verts = torch.einsum("ij,tnj->tni", R_x, right_verts.cpu())
        left_joints = torch.einsum("ij,tnj->tni", R_x, left_joints.cpu())
        right_joints = torch.einsum("ij,tnj->tni", R_x, right_joints.cpu())

        sample = background_frames[0]
        height, width, _ = sample.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        renderer = Renderer(width, height, img_focal, device)
        K = torch.tensor(
            [[img_focal, 0, width / 2], [0, img_focal, height / 2], [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        writer = cv2.VideoWriter(
            str(output_mp4),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_mp4}")

        try:
            for local_i, background in enumerate(background_frames):
                frame_idx = vis_start + local_i

                R_w2c = R_w2c_sla_all[frame_idx : frame_idx + 1]
                t_w2c = t_w2c_sla_all[frame_idx : frame_idx + 1]

                if pred_valid[0, frame_idx]:
                    background = _blend_hand(
                        background,
                        renderer,
                        faces_left,
                        left_verts[local_i : local_i + 1],
                        (0.207, 0.596, 0.792),
                        R_w2c,
                        t_w2c,
                    )
                if pred_valid[1, frame_idx]:
                    background = _blend_hand(
                        background,
                        renderer,
                        faces_right,
                        right_verts[local_i : local_i + 1],
                        (0.804, 0.6, 0.820),
                        R_w2c,
                        t_w2c,
                    )
                if pred_valid[0, frame_idx]:
                    joints_2d = _project_world_joints(
                        left_joints[local_i],
                        R_w2c,
                        t_w2c,
                        K,
                    )
                    background = _draw_hand_joints(background, joints_2d)
                if pred_valid[1, frame_idx]:
                    joints_2d = _project_world_joints(
                        right_joints[local_i],
                        R_w2c,
                        t_w2c,
                        K,
                    )
                    background = _draw_hand_joints(background, joints_2d)
                writer.write(background)
        finally:
            writer.release()
    finally:
        if temp_frame_dir is not None:
            shutil.rmtree(temp_frame_dir, ignore_errors=True)

    if not output_mp4.is_file() or output_mp4.stat().st_size == 0:
        raise RuntimeError(f"Overlay video was not written: {output_mp4}")
    return output_mp4
