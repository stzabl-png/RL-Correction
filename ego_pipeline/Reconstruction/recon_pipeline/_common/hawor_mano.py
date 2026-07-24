"""Load HaWoR MANO predictions and project them into the camera frame."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

RECON_ROOT = Path(__file__).resolve().parents[1]
HAWOR_ROOT = RECON_ROOT.parent / "third_party" / "hawor"
DEFAULT_HAWOR_FPS = 30.0


@dataclass(frozen=True)
class HaworTimeline:
    hawor_fps: float
    video_fps: float
    hawor_num_frames: int
    video_num_frames: int
    source: str = "default"
    interpolation: str = "linear"

    def video_to_hawor_times(self, video_frame_indices: np.ndarray) -> np.ndarray:
        return np.asarray(video_frame_indices, dtype=np.float64) * self.hawor_fps / self.video_fps

    def sample_indices(self, video_frame_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return floor idx, ceil idx, and lerp alpha on the HaWoR timeline."""
        times = self.video_to_hawor_times(video_frame_indices)
        h0 = np.floor(times).astype(np.int64)
        h1 = np.minimum(h0 + 1, self.hawor_num_frames - 1)
        alpha = times - h0
        h0 = np.clip(h0, 0, self.hawor_num_frames - 1)
        return h0, h1, alpha


@dataclass
class HandFrameGeometry:
    left_verts_cam: np.ndarray | None
    right_verts_cam: np.ndarray | None
    left_joints_cam: np.ndarray | None
    right_joints_cam: np.ndarray | None
    left_valid: bool
    right_valid: bool
    hawor_index0: int
    hawor_index1: int
    hawor_alpha: float


@dataclass
class HandSequenceGeometry:
    faces_left: np.ndarray
    faces_right: np.ndarray
    frames: list[HandFrameGeometry]
    timeline: HaworTimeline


def load_hawor_timeline(
    *,
    world_result_path: Path,
    video_path: Path | None,
    num_video_frames: int,
) -> HaworTimeline:
    pred_trans, _, _, _, _ = _load_hand_params(world_result_path)
    hawor_num_frames = int(pred_trans.shape[1])
    hawor_fps = DEFAULT_HAWOR_FPS
    video_fps = DEFAULT_HAWOR_FPS
    source = "default_30fps"
    interpolation = "linear"
    explicit_mapping = False

    meta_path = world_result_path.parent / "run_meta.json"
    if meta_path.is_file():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        mapping = payload.get("sam3_filter_summary", {}).get("frame_mapping", {})
        if mapping:
            explicit_mapping = True
            source = "run_meta.sam3_filter_summary.frame_mapping"
            hawor_fps = float(mapping.get("hawor_fps", hawor_fps))
            video_fps = float(mapping.get("sam3_fps", video_fps))
            interpolation = str(mapping.get("interpolation", interpolation))
            mapped_video_frames = int(mapping.get("sam3_num_frames", num_video_frames))
            if mapped_video_frames > 0:
                num_video_frames = mapped_video_frames

    if not explicit_mapping and video_path is not None and video_path.is_file():
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        measured_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if measured_fps > 1e-3:
            video_fps = measured_fps
            source = "video_metadata"

    if not explicit_mapping and num_video_frames > 1 and hawor_num_frames > 1:
        inferred_hawor_fps = hawor_fps
        inferred_video_fps = video_fps
        duration_from_hawor = (hawor_num_frames - 1) / inferred_hawor_fps
        duration_from_video = (num_video_frames - 1) / inferred_video_fps
        if duration_from_hawor > 1e-6 and duration_from_video > 1e-6:
            # Keep declared fps when close; otherwise trust frame counts.
            if abs(duration_from_hawor - duration_from_video) / duration_from_video > 0.05:
                hawor_fps = inferred_hawor_fps
                video_fps = num_video_frames * hawor_fps / hawor_num_frames
                source = "frame_count_duration_fallback"

    return HaworTimeline(
        hawor_fps=hawor_fps,
        video_fps=video_fps,
        hawor_num_frames=hawor_num_frames,
        video_num_frames=num_video_frames,
        source=source,
        interpolation=interpolation,
    )


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


def _load_hand_params(world_result_path: Path):
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_result_path)
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _world_to_cam(points_world: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    return (w2c @ pts_h.T).T[:, :3]


def _world_to_cam_rt(points_world: np.ndarray, r_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    return (np.asarray(r_w2c, dtype=np.float64) @ pts.T).T + np.asarray(t_w2c, dtype=np.float64).reshape(1, 3)


def _hawor_world_points_for_slam(points_world: np.ndarray) -> np.ndarray:
    """Match HaWoR's SAM3 filter convention before applying SLAM w2c."""
    pts = np.asarray(points_world, dtype=np.float64)
    return pts * np.array([1.0, -1.0, -1.0], dtype=np.float64)


def _interp_world_sequence(values: np.ndarray, h0: np.ndarray, h1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Linearly interpolate (T_hawor, ...) tensors onto video frames."""
    alpha = alpha.reshape(-1, *([1] * (values.ndim - 1)))
    return (1.0 - alpha) * values[h0] + alpha * values[h1]


def _sample_valid(valid: np.ndarray, h0: np.ndarray, h1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Validity must follow the same temporal sample as geometry."""
    v0 = np.asarray(valid[h0]).astype(bool)
    v1 = np.asarray(valid[h1]).astype(bool)
    out = v0 & v1
    out = np.where(alpha <= 1e-6, v0, out)
    out = np.where(alpha >= 1.0 - 1e-6, v1, out)
    return out.astype(bool)


def load_hand_params_numpy(world_result_path: Path) -> dict[str, np.ndarray]:
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = _load_hand_params(world_result_path)
    return {
        "hand_trans": _to_numpy(pred_trans),
        "hand_rot": _to_numpy(pred_rot),
        "hand_pose": _to_numpy(pred_hand_pose),
        "hand_betas": _to_numpy(pred_betas),
        "hand_valid": _to_numpy(pred_valid),
    }


def compute_hand_sequence_geometry(
    *,
    world_result_path: Path,
    c2w_all: np.ndarray,
    num_video_frames: int,
    video_path: Path | None = None,
) -> HandSequenceGeometry | None:
    """Return per-video-frame MANO geometry in the camera frame, or None if MANO is unavailable."""
    if not world_result_path.is_file():
        return None
    try:
        timeline = load_hawor_timeline(
            world_result_path=world_result_path,
            video_path=video_path,
            num_video_frames=num_video_frames,
        )
        num_video_frames = min(num_video_frames, timeline.video_num_frames, len(c2w_all))

        with _hawor_runtime():
            import torch
            from lib.eval_utils.custom_utils import load_slam_cam
            from hawor.utils.process import get_mano_faces, run_mano, run_mano_left

            pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = _load_hand_params(world_result_path)
            hawor_t = int(pred_trans.shape[1])
            full_idx = torch.arange(hawor_t, dtype=torch.long)

            right = run_mano(
                pred_trans[1:2, full_idx],
                pred_rot[1:2, full_idx],
                pred_hand_pose[1:2, full_idx],
                betas=pred_betas[1:2, full_idx],
            )
            left = run_mano_left(
                pred_trans[0:1, full_idx],
                pred_rot[0:1, full_idx],
                pred_hand_pose[0:1, full_idx],
                betas=pred_betas[0:1, full_idx],
            )
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
                ],
                dtype=np.int64,
            )
            faces_right = np.concatenate([faces, faces_new], axis=0)
            faces_left = faces_right[:, [0, 2, 1]]

            right_verts = _to_numpy(right["vertices"][0])
            right_joints = _to_numpy(right["joints"][0])
            left_verts = _to_numpy(left["vertices"][0])
            left_joints = _to_numpy(left["joints"][0])
            valid_full = _to_numpy(pred_valid)

            slam_path = world_result_path.parent / "slam.npz"
            r_w2c_video = t_w2c_video = None
            if slam_path.is_file():
                _r_w2c, _t_w2c, r_c2w, t_c2w = load_slam_cam(str(slam_path))
                r_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
                r_c2w = torch.einsum("ij,njk->nik", r_x, r_c2w)
                t_c2w = torch.einsum("ij,nj->ni", r_x, t_c2w)
                r_w2c = r_c2w.transpose(-1, -2)
                t_w2c = -torch.einsum("bij,bj->bi", r_w2c, t_c2w)
                r_w2c_video = _to_numpy(r_w2c)
                t_w2c_video = _to_numpy(t_w2c)

        video_indices = np.arange(num_video_frames, dtype=np.int64)
        h0, h1, alpha = timeline.sample_indices(video_indices)
        right_verts_v = _interp_world_sequence(right_verts, h0, h1, alpha)
        left_verts_v = _interp_world_sequence(left_verts, h0, h1, alpha)
        right_joints_v = _interp_world_sequence(right_joints, h0, h1, alpha)
        left_joints_v = _interp_world_sequence(left_joints, h0, h1, alpha)
        right_valid = _sample_valid(valid_full[1], h0, h1, alpha)
        left_valid = _sample_valid(valid_full[0], h0, h1, alpha)
    except Exception as exc:
        import warnings

        warnings.warn(f"MANO hand geometry unavailable: {exc}", stacklevel=2)
        return None

    frames: list[HandFrameGeometry] = []
    for local_i in range(num_video_frames):
        lv = rv = lj = rj = None
        if r_w2c_video is not None and t_w2c_video is not None:
            # HaWoR world-space MANO was solved against the 30 fps HaWoR SLAM
            # timeline. Project it with the same sampled camera frame.
            cam_i = min(int(h0[local_i]), len(r_w2c_video) - 1)
            r_w2c_i = r_w2c_video[cam_i]
            t_w2c_i = t_w2c_video[cam_i]
            project = lambda pts: _world_to_cam_rt(pts, r_w2c_i, t_w2c_i)
        else:
            w2c = np.linalg.inv(c2w_all[local_i])
            project = lambda pts: _world_to_cam(pts, w2c)
        if left_valid[local_i]:
            lv = project(_hawor_world_points_for_slam(left_verts_v[local_i]))
            lj = project(_hawor_world_points_for_slam(left_joints_v[local_i]))
        if right_valid[local_i]:
            rv = project(_hawor_world_points_for_slam(right_verts_v[local_i]))
            rj = project(_hawor_world_points_for_slam(right_joints_v[local_i]))
        frames.append(
            HandFrameGeometry(
                left_verts_cam=lv,
                right_verts_cam=rv,
                left_joints_cam=lj,
                right_joints_cam=rj,
                left_valid=bool(left_valid[local_i]),
                right_valid=bool(right_valid[local_i]),
                hawor_index0=int(h0[local_i]),
                hawor_index1=int(h1[local_i]),
                hawor_alpha=float(alpha[local_i]),
            )
        )
    return HandSequenceGeometry(faces_left=faces_left, faces_right=faces_right, frames=frames, timeline=timeline)


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


def draw_hand_skeleton(frame, joints_cam: np.ndarray, K: np.ndarray, *, color_bgr: tuple[int, int, int]) -> None:
    import cv2

    from _common.mesh_viz import project_cam

    joints = np.asarray(joints_cam, dtype=np.float64)[:21]
    uv, z = project_cam(K, joints)
    h, w = frame.shape[:2]

    for i0, i1 in MANO_HAND_PAIRS:
        if z[i0] <= 1e-4 or z[i1] <= 1e-4:
            continue
        p0 = (int(round(uv[i0, 0])), int(round(uv[i0, 1])))
        p1 = (int(round(uv[i1, 0])), int(round(uv[i1, 1])))
        cv2.line(frame, p0, p1, color_bgr, 3, cv2.LINE_AA)

    for i in range(len(joints)):
        if z[i] <= 1e-4:
            continue
        p = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(frame, p, 4, color_bgr, -1, cv2.LINE_AA)
