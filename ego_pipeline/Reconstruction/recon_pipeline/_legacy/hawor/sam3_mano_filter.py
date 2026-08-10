"""Filter HaWoR MANO predictions using SAM3 hand/arm mask containment."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch

FILTER_ARTIFACT = "sam3_mano_filter.json"
FILTER_VIS_SUFFIX = "_sam3_filter_vis.mp4"
HAWOR_EXTRACT_FPS = 30.0

LEFT_HAND_OBJ_ID = "left_hand_0"
RIGHT_HAND_OBJ_ID = "right_hand_0"
HAND_SIDE_OBJ_IDS = (LEFT_HAND_OBJ_ID, RIGHT_HAND_OBJ_ID)


@dataclass
class Sam3ManoFilterConfig:
    sam_dilation: int = 10
    min_mano_area: int = 80
    min_valid_depth_ratio: float = 0.8
    min_inside_image_ratio: float = 0.5
    mano_containment_threshold: float = 0.50
    joint_containment_threshold: float = 0.50
    alignment_mano_containment_threshold: float = 0.60
    alignment_joint_containment_threshold: float = 0.70
    component_score_mano_weight: float = 0.6
    write_debug_video: bool = True


@dataclass
class HandFilterDecision:
    accepted: bool
    mano_containment: float
    joint_containment: float
    matched_sam_component: int | None
    filter_reason: str


@dataclass(frozen=True)
class Sam3TimelineMapping:
    """Map HaWoR extracted-frame indices to continuous SAM3 mask time."""

    hawor_fps: float
    sam3_fps: float
    sam3_num_frames: int

    def hawor_to_sam_time(self, hawor_frame_idx: int | float) -> float:
        return float(hawor_frame_idx) * self.sam3_fps / self.hawor_fps

    def sam_time_in_range(self, hawor_frame_idx: int) -> bool:
        sam_t = self.hawor_to_sam_time(hawor_frame_idx)
        return 0.0 <= sam_t <= float(self.sam3_num_frames - 1)

    def to_summary_dict(self) -> dict[str, float | int | str]:
        return {
            "hawor_fps": self.hawor_fps,
            "sam3_fps": self.sam3_fps,
            "sam3_num_frames": self.sam3_num_frames,
            "interpolation": "linear",
        }

    @classmethod
    def from_summary_dict(cls, payload: dict) -> Sam3TimelineMapping:
        return cls(
            hawor_fps=float(payload["hawor_fps"]),
            sam3_fps=float(payload["sam3_fps"]),
            sam3_num_frames=int(payload["sam3_num_frames"]),
        )


def _load_sam3_helpers():
    import importlib.util
    import sys

    module_path = Path(__file__).resolve().parents[1] / "sam3" / "common.py"
    spec = importlib.util.spec_from_file_location("sam3_pipeline_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SAM3 helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mano_faces(handedness: int) -> np.ndarray:
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
        ],
        dtype=np.int32,
    )
    faces_right = np.concatenate([faces, faces_new], axis=0)
    if handedness == 0:
        return faces_right[:, [0, 2, 1]]
    return faces_right


def _camera_mats_from_slam(
    slam_path: Path,
    img_focal: float,
    *,
    legacy_negative_focal: bool,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    from lib.eval_utils.custom_utils import load_slam_cam

    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(str(slam_path))
    R_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    R_c2w = torch.einsum("ij,njk->nik", R_x, R_c2w)
    t_c2w = torch.einsum("ij,nj->ni", R_x, t_c2w)
    R_w2c = R_c2w.transpose(-1, -2)
    t_w2c = -torch.einsum("bij,bj->bi", R_w2c, t_c2w)
    focal = abs(float(img_focal))
    if legacy_negative_focal:
        flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
        R_w2c = torch.einsum("ij,njk->nik", flip, R_w2c)
        t_w2c = torch.einsum("ij,nj->ni", flip, t_w2c)
    return R_w2c.float(), t_w2c.float(), focal


def _intrinsics_matrix(focal: float, width: int, height: int) -> np.ndarray:
    cx, cy = width / 2.0, height / 2.0
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _project_cam_points(
    points_cam: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    depth_ok = z > 1e-6
    u = np.full(points_cam.shape[0], np.nan, dtype=np.float64)
    v = np.full(points_cam.shape[0], np.nan, dtype=np.float64)
    if np.any(depth_ok):
        pts = points_cam[depth_ok]
        uv = (K @ pts.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        u[depth_ok] = uv[:, 0]
        v[depth_ok] = uv[:, 1]
    inside = (
        depth_ok
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    return np.stack([u, v, z], axis=-1), inside


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    if not np.any(mask):
        return []
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    return [labels == label_idx for label_idx in range(1, num_labels)]


def resolve_sam3_timeline_mapping(
    *,
    sam3_dir: Path,
    sequence_name: str,
    source_video: Path | None = None,
    hawor_fps: float = HAWOR_EXTRACT_FPS,
) -> Sam3TimelineMapping:
    """Resolve SAM3 native-frame timeline vs HaWoR ffmpeg extraction fps."""
    sam3 = _load_sam3_helpers()
    marker = sam3.completion_marker_path(sam3_dir, sequence_name)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    sam3_num_frames = int(payload.get("num_frames") or 0)
    if sam3_num_frames <= 0:
        masks_dir = sam3.masks_root(sam3_dir, sequence_name)
        sam3_num_frames = len(list(masks_dir.glob("frame_*_masks")))
    if sam3_num_frames <= 0:
        raise RuntimeError(f"SAM3 sequence {sequence_name} has no mask frames under {sam3_dir}")

    video_path = source_video
    if video_path is None and payload.get("video"):
        video_path = Path(str(payload["video"]))
    if video_path is not None and video_path.is_file():
        sam3_fps = float(sam3.video_frame_rate(video_path, default=hawor_fps))
    else:
        sam3_fps = hawor_fps

    return Sam3TimelineMapping(
        hawor_fps=float(hawor_fps),
        sam3_fps=sam3_fps,
        sam3_num_frames=sam3_num_frames,
    )


def _resize_bool_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    empty = np.zeros((height, width), dtype=bool)
    if mask.size == 0:
        return empty
    if mask.shape == (height, width):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def _load_sam_masks_at_index(
    masks_dir: Path,
    sam_frame_idx: int,
    height: int,
    width: int,
    load_mask_png,
) -> tuple[np.ndarray, np.ndarray]:
    frame_dir = masks_dir / f"frame_{sam_frame_idx:06d}_masks"
    left = load_mask_png(frame_dir / f"{LEFT_HAND_OBJ_ID}.png")
    right = load_mask_png(frame_dir / f"{RIGHT_HAND_OBJ_ID}.png")
    return (
        _resize_bool_mask(left, height, width),
        _resize_bool_mask(right, height, width),
    )


def _interpolate_bool_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha <= 0.0:
        return mask_a
    if alpha >= 1.0:
        return mask_b
    blended = (1.0 - alpha) * mask_a.astype(np.float32) + alpha * mask_b.astype(np.float32)
    return blended >= 0.5


def _load_interpolated_sam_masks(
    masks_dir: Path,
    hawor_frame_idx: int,
    height: int,
    width: int,
    load_mask_png,
    mapping: Sam3TimelineMapping,
) -> tuple[np.ndarray, np.ndarray]:
    """Load SAM3 left/right masks at the HaWoR timeline via linear interpolation."""
    empty = np.zeros((height, width), dtype=bool)
    if not mapping.sam_time_in_range(hawor_frame_idx):
        return empty.copy(), empty.copy()

    sam_t = mapping.hawor_to_sam_time(hawor_frame_idx)
    idx0 = int(np.floor(sam_t))
    idx1 = min(idx0 + 1, mapping.sam3_num_frames - 1)
    alpha = sam_t - idx0

    left0, right0 = _load_sam_masks_at_index(
        masks_dir, idx0, height, width, load_mask_png
    )
    if idx0 == idx1 or alpha <= 1e-6:
        return left0, right0

    left1, right1 = _load_sam_masks_at_index(
        masks_dir, idx1, height, width, load_mask_png
    )
    return (
        _interpolate_bool_masks(left0, left1, alpha),
        _interpolate_bool_masks(right0, right1, alpha),
    )


def _load_frame_sam_masks(
    masks_dir: Path,
    frame_idx: int,
    height: int,
    width: int,
    load_mask_png,
    mapping: Sam3TimelineMapping | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if mapping is None:
        return _load_sam_masks_at_index(masks_dir, frame_idx, height, width, load_mask_png)
    return _load_interpolated_sam_masks(
        masks_dir, frame_idx, height, width, load_mask_png, mapping
    )


def sam3_hand_presence(
    *,
    sam3_dir: Path,
    sequence_name: str,
    num_frames: int,
    image_size: tuple[int, int],
    source_video: Path | None = None,
    min_area: int = 80,
    hawor_fps: float = HAWOR_EXTRACT_FPS,
) -> np.ndarray:
    """Per-frame, per-hand SAM3 hand-mask presence, independent of HaWoR.

    Returns a bool array (2, num_frames): row 0 = left, row 1 = right; True where
    the SAM3 hand mask for that side at that HaWoR-timeline frame has area
    >= ``min_area``. This is pure image evidence (a mask exists) and does not
    depend on whether HaWoR produced a pose, so it can gate infilled frames:
    only trust a filled hand where SAM3 still sees a hand there (no hallucination).
    """
    sam3 = _load_sam3_helpers()
    masks_dir = sam3.masks_root(sam3_dir, sequence_name)
    timeline = resolve_sam3_timeline_mapping(
        sam3_dir=sam3_dir,
        sequence_name=sequence_name,
        source_video=source_video,
        hawor_fps=hawor_fps,
    )
    height, width = image_size
    present = np.zeros((2, num_frames), dtype=bool)
    for frame_idx in range(num_frames):
        sam_left, sam_right = _load_frame_sam_masks(
            masks_dir, frame_idx, height, width, sam3.load_mask_png, mapping=timeline
        )
        present[0, frame_idx] = int(sam_left.sum()) >= min_area
        present[1, frame_idx] = int(sam_right.sum()) >= min_area
    return present


def _sorted_frame_paths(pattern: str) -> list[str]:
    paths = glob(pattern)

    def _frame_index(path: str) -> int:
        return int(Path(path).stem)

    try:
        return sorted(paths, key=_frame_index)
    except ValueError:
        return sorted(paths)


def _extract_hawor_timeline_frames(
    source_video: Path,
    *,
    hawor_fps: float,
    num_frames: int,
) -> tuple[list[np.ndarray], Path | None]:
    """Extract frames the same way HaWoR detect_track_video does (fps=30)."""
    temp_dir = Path(tempfile.mkdtemp(prefix="hawor_sam3_filter_vis_"))
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video),
        "-vf",
        f"fps={hawor_fps}",
        "-start_number",
        "0",
        str(temp_dir / "%04d.jpg"),
    ]
    subprocess.run(command, check=True)
    imgfiles = _sorted_frame_paths(str(temp_dir / "*.jpg"))
    if not imgfiles:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"ffmpeg produced no frames for {source_video}")
    frames: list[np.ndarray] = []
    for img_path in imgfiles[:num_frames]:
        frame = cv2.imread(img_path)
        if frame is None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise FileNotFoundError(f"Could not read frame: {img_path}")
        frames.append(frame)
    return frames, temp_dir


def _world_hand_geometry(
    pred_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_hand_pose: torch.Tensor,
    pred_betas: torch.Tensor,
    frame_idx: int,
    hand_idx: int,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    from hawor.utils.process import run_mano, run_mano_left

    trans = pred_trans[hand_idx : hand_idx + 1, frame_idx : frame_idx + 1]
    rot = pred_rot[hand_idx : hand_idx + 1, frame_idx : frame_idx + 1]
    hand_pose = pred_hand_pose[hand_idx : hand_idx + 1, frame_idx : frame_idx + 1]
    betas = pred_betas[hand_idx : hand_idx + 1, frame_idx : frame_idx + 1]
    if hand_idx == 0:
        outputs = run_mano_left(trans, rot, hand_pose, betas=betas)
    else:
        outputs = run_mano(trans, rot, hand_pose, betas=betas)
    verts_w = outputs["vertices"][0, 0].detach().cpu().float()
    joints_w = outputs["joints"][0, 0, :21].detach().cpu().float()
    R_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    verts_w = torch.einsum("ij,vj->vi", R_x, verts_w)
    joints_w = torch.einsum("ij,kj->ki", R_x, joints_w)
    R = R_w2c[frame_idx].cpu()
    t = t_w2c[frame_idx].cpu()
    verts_cam = (R @ verts_w.T).T + t
    joints_cam = (R @ joints_w.T).T + t
    return verts_cam.numpy().astype(np.float64), joints_cam.numpy().astype(np.float64)


def _rasterize_mano_mask(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uvz, inside = _project_cam_points(verts_cam, K, width, height)
    mano_mask = np.zeros((height, width), dtype=np.uint8)
    drew = False
    for face in faces:
        if np.any(verts_cam[face, 2] <= 1e-6):
            continue
        tri = uvz[face, :2]
        if not np.all(np.isfinite(tri)):
            continue
        cv2.fillConvexPoly(
            mano_mask,
            tri.reshape(-1, 1, 2).astype(np.int32),
            1,
        )
        drew = True
    if not drew:
        depth_ok = verts_cam[:, 2] > 1e-6
        if np.count_nonzero(depth_ok) >= 3:
            pts2d = uvz[depth_ok, :2].astype(np.float32)
            hull = cv2.convexHull(pts2d)
            cv2.fillConvexPoly(mano_mask, hull, 1)
    return mano_mask > 0, uvz, inside


def _joint_containment(
    uvz: np.ndarray,
    inside_image: np.ndarray,
    sam_mask: np.ndarray,
) -> float:
    valid = inside_image & np.isfinite(uvz[:, 0]) & np.isfinite(uvz[:, 1])
    if not np.any(valid):
        return 0.0
    inside_count = 0
    for idx in np.where(valid)[0]:
        u, v = uvz[idx, 0], uvz[idx, 1]
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < sam_mask.shape[0] and 0 <= ui < sam_mask.shape[1] and sam_mask[vi, ui]:
            inside_count += 1
    return inside_count / float(np.count_nonzero(valid))


def _mano_containment(mano_mask: np.ndarray, sam_mask: np.ndarray) -> float:
    area = int(mano_mask.sum())
    if area == 0:
        return 0.0
    return float(np.logical_and(mano_mask, sam_mask).sum()) / area


def _evaluate_against_components(
    mano_mask: np.ndarray,
    joint_uvz: np.ndarray,
    joints_inside: np.ndarray,
    sam_components: list[np.ndarray],
    cfg: Sam3ManoFilterConfig,
    *,
    strict: bool,
) -> HandFilterDecision:
    if not sam_components:
        return HandFilterDecision(False, 0.0, 0.0, None, "no_sam_mask")

    mano_thresh = (
        cfg.alignment_mano_containment_threshold
        if strict
        else cfg.mano_containment_threshold
    )
    joint_thresh = (
        cfg.alignment_joint_containment_threshold
        if strict
        else cfg.joint_containment_threshold
    )

    best: HandFilterDecision | None = None
    best_score = -1.0
    for comp_idx, comp in enumerate(sam_components):
        mc = _mano_containment(mano_mask, comp)
        jc = _joint_containment(joint_uvz, joints_inside, comp)
        score = cfg.component_score_mano_weight * mc + (1.0 - cfg.component_score_mano_weight) * jc
        if score > best_score:
            accepted = mc >= mano_thresh and jc >= joint_thresh
            reason = "accepted" if accepted else (
                "low_mano_containment" if mc < mano_thresh else "low_joint_containment"
            )
            best = HandFilterDecision(accepted, mc, jc, comp_idx, reason)
            best_score = score
    assert best is not None
    return best


def _evaluate_hand_candidate(
    verts_cam: np.ndarray,
    joints_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    sam_left: np.ndarray,
    sam_right: np.ndarray,
    cfg: Sam3ManoFilterConfig,
    *,
    strict: bool,
) -> HandFilterDecision:
    depth_ratio = float(np.mean(joints_cam[:, 2] > 0))
    if depth_ratio < cfg.min_valid_depth_ratio:
        return HandFilterDecision(False, 0.0, 0.0, None, "invalid_depth")

    joint_uvz, joints_inside = _project_cam_points(joints_cam, K, width, height)
    inside_ratio = float(np.mean(joints_inside))
    if inside_ratio < cfg.min_inside_image_ratio:
        return HandFilterDecision(False, 0.0, 0.0, None, "projected_outside_image")

    mano_mask, _, _ = _rasterize_mano_mask(verts_cam, faces, K, width, height)
    if int(mano_mask.sum()) < cfg.min_mano_area:
        return HandFilterDecision(False, 0.0, 0.0, None, "empty_mano_mask")

    union = np.logical_or(sam_left, sam_right)
    if not np.any(union):
        return HandFilterDecision(False, 0.0, 0.0, None, "no_sam_mask")

    dilated_union = _dilate_mask(union, cfg.sam_dilation)
    components = _connected_components(dilated_union)
    return _evaluate_against_components(
        mano_mask,
        joint_uvz,
        joints_inside,
        components,
        cfg,
        strict=strict,
    )


def _resolve_same_component_conflicts(
    left: HandFilterDecision,
    right: HandFilterDecision,
) -> tuple[HandFilterDecision, HandFilterDecision]:
    if (
        left.matched_sam_component is None
        or right.matched_sam_component is None
        or left.matched_sam_component != right.matched_sam_component
    ):
        return left, right
    if not left.accepted and not right.accepted:
        return left, right
    left_score = 0.6 * left.mano_containment + 0.4 * left.joint_containment
    right_score = 0.6 * right.mano_containment + 0.4 * right.joint_containment
    if left.accepted and right.accepted:
        if left_score >= right_score:
            right = HandFilterDecision(
                False,
                right.mano_containment,
                right.joint_containment,
                right.matched_sam_component,
                "conflict_same_sam_component",
            )
        else:
            left = HandFilterDecision(
                False,
                left.mano_containment,
                left.joint_containment,
                left.matched_sam_component,
                "conflict_same_sam_component",
            )
    return left, right


def filter_world_poses_with_sam3(
    *,
    pred_trans: torch.Tensor,
    pred_rot: torch.Tensor,
    pred_hand_pose: torch.Tensor,
    pred_betas: torch.Tensor,
    pred_valid: torch.Tensor,
    slam_path: Path,
    img_focal: float,
    sam3_dir: Path,
    sequence_name: str,
    num_frames: int,
    image_size: tuple[int, int],
    cfg: Sam3ManoFilterConfig,
    candidate_valid: torch.Tensor | None = None,
    legacy_negative_focal: bool = False,
    strict: bool = False,
    source_video: Path | None = None,
    hawor_fps: float = HAWOR_EXTRACT_FPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    sam3 = _load_sam3_helpers()
    marker = sam3.completion_marker_path(sam3_dir, sequence_name)
    if not marker.is_file():
        raise FileNotFoundError(
            f"SAM3 masks not found for {sequence_name}: expected {marker}. "
            "Run recon_pipeline/sam3_hands first or pass --no-sam3-filter."
        )
    masks_dir = sam3.masks_root(sam3_dir, sequence_name)
    timeline = resolve_sam3_timeline_mapping(
        sam3_dir=sam3_dir,
        sequence_name=sequence_name,
        source_video=source_video,
        hawor_fps=hawor_fps,
    )
    height, width = image_size
    K = _intrinsics_matrix(abs(float(img_focal)), width, height)
    R_w2c, t_w2c, _ = _camera_mats_from_slam(
        slam_path,
        img_focal,
        legacy_negative_focal=legacy_negative_focal,
    )

    if candidate_valid is None:
        candidate_valid = pred_valid.clone()
    else:
        candidate_valid = candidate_valid.clone()

    num_hands, total_frames = pred_valid.shape
    valid_hand = np.zeros((total_frames, num_hands), dtype=bool)
    mano_containment = np.full((total_frames, num_hands), np.nan, dtype=np.float64)
    joint_containment = np.full((total_frames, num_hands), np.nan, dtype=np.float64)
    matched_component = np.full((total_frames, num_hands), -1, dtype=np.int32)
    filter_reason = [["" for _ in range(num_hands)] for _ in range(total_frames)]

    faces = [_mano_faces(0), _mano_faces(1)]

    for frame_idx in range(min(num_frames, total_frames)):
        if not torch.any(candidate_valid[:, frame_idx] > 0):
            continue
        sam_left, sam_right = _load_frame_sam_masks(
            masks_dir,
            frame_idx,
            height,
            width,
            sam3.load_mask_png,
            mapping=timeline,
        )
        frame_decisions: list[HandFilterDecision] = []
        for hand_idx in range(num_hands):
            if candidate_valid[hand_idx, frame_idx] <= 0:
                frame_decisions.append(
                    HandFilterDecision(False, 0.0, 0.0, None, "not_observed")
                )
                continue
            verts_cam, joints_cam = _world_hand_geometry(
                pred_trans,
                pred_rot,
                pred_hand_pose,
                pred_betas,
                frame_idx,
                hand_idx,
                R_w2c,
                t_w2c,
            )
            decision = _evaluate_hand_candidate(
                verts_cam,
                joints_cam,
                faces[hand_idx],
                K,
                width,
                height,
                sam_left,
                sam_right,
                cfg,
                strict=strict,
            )
            frame_decisions.append(decision)

        left_decision, right_decision = _resolve_same_component_conflicts(
            frame_decisions[0],
            frame_decisions[1],
        )
        for hand_idx, decision in enumerate((left_decision, right_decision)):
            if candidate_valid[hand_idx, frame_idx] <= 0:
                continue
            valid_hand[frame_idx, hand_idx] = decision.accepted
            mano_containment[frame_idx, hand_idx] = decision.mano_containment
            joint_containment[frame_idx, hand_idx] = decision.joint_containment
            matched_component[frame_idx, hand_idx] = (
                -1 if decision.matched_sam_component is None else decision.matched_sam_component
            )
            filter_reason[frame_idx][hand_idx] = decision.filter_reason
            if decision.accepted:
                pred_valid[hand_idx, frame_idx] = 1
            else:
                pred_valid[hand_idx, frame_idx] = 0

    summary = {
        "sequence": sequence_name,
        "config": asdict(cfg),
        "strict": strict,
        "sam3_dir": str(sam3_dir.resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "image_size": [height, width],
        "frame_mapping": timeline.to_summary_dict(),
        "frames_evaluated": int(min(num_frames, total_frames)),
        "accepted_frames": {
            "left": int(valid_hand[:, 0].sum()),
            "right": int(valid_hand[:, 1].sum()),
        },
        "rejected_frames": {
            "left": int(np.sum((candidate_valid[0].numpy() > 0) & ~valid_hand[:, 0])),
            "right": int(np.sum((candidate_valid[1].numpy() > 0) & ~valid_hand[:, 1])),
        },
        "valid_hand": valid_hand.tolist(),
        "mano_containment": np.nan_to_num(mano_containment, nan=-1.0).tolist(),
        "joint_containment": np.nan_to_num(joint_containment, nan=-1.0).tolist(),
        "matched_sam_component": matched_component.tolist(),
        "filter_reason": filter_reason,
    }
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, summary


def write_filter_artifact(output_path: Path, summary: dict) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def render_filter_debug_video(
    *,
    source_video: Path,
    sam3_dir: Path,
    sequence_name: str,
    filter_summary: dict,
    output_mp4: Path,
    slam_path: Path,
    world_result_path: Path,
    img_focal: float,
    fps: float = HAWOR_EXTRACT_FPS,
) -> Path:
    import joblib
    import torch
    from hawor.utils.process import run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam
    from lib.vis.renderer import Renderer
    from visualize_outputs import (
        _blend_hand,
        _draw_hand_joints,
        _mano_face_sets,
        _normalize_vipe_focal_and_cameras,
        _project_world_joints,
    )

    sam3 = _load_sam3_helpers()
    masks_dir = sam3.masks_root(sam3_dir, sequence_name)
    mapping_payload = filter_summary.get("frame_mapping")
    if mapping_payload is not None:
        timeline = Sam3TimelineMapping.from_summary_dict(mapping_payload)
    else:
        timeline = resolve_sam3_timeline_mapping(
            sam3_dir=sam3_dir,
            sequence_name=sequence_name,
            source_video=source_video,
            hawor_fps=fps if fps > 0 else HAWOR_EXTRACT_FPS,
        )

    valid_hand = filter_summary["valid_hand"]
    mano_c = filter_summary["mano_containment"]
    joint_c = filter_summary["joint_containment"]
    reasons = filter_summary["filter_reason"]
    num_frames = len(valid_hand)
    if fps <= 0:
        fps = timeline.hawor_fps

    pred_trans, pred_rot, pred_hand_pose, pred_betas, _pred_valid = joblib.load(
        world_result_path.resolve()
    )
    vis_end = min(num_frames, pred_trans.shape[1])

    background_frames, temp_dir = _extract_hawor_timeline_frames(
        source_video.resolve(),
        hawor_fps=timeline.hawor_fps,
        num_frames=vis_end,
    )
    height, width = background_frames[0].shape[:2]

    faces_left, faces_right = _mano_face_sets()
    pred_glob_l = run_mano_left(
        pred_trans[0:1, :vis_end],
        pred_rot[0:1, :vis_end],
        pred_hand_pose[0:1, :vis_end],
        betas=pred_betas[0:1, :vis_end],
    )
    left_verts = pred_glob_l["vertices"][0]
    left_joints = pred_glob_l["joints"][0]

    pred_glob_r = run_mano(
        pred_trans[1:2, :vis_end],
        pred_rot[1:2, :vis_end],
        pred_hand_pose[1:2, :vis_end],
        betas=pred_betas[1:2, :vis_end],
    )
    right_verts = pred_glob_r["vertices"][0]
    right_joints = pred_glob_r["joints"][0]

    _R_w2c, _t_w2c, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(str(slam_path.resolve()))
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    img_focal, R_w2c_sla_all, t_w2c_sla_all = _normalize_vipe_focal_and_cameras(
        img_focal,
        R_w2c_sla_all,
        t_w2c_sla_all,
    )
    left_verts = torch.einsum("ij,tnj->tni", R_x, left_verts.cpu())
    right_verts = torch.einsum("ij,tnj->tni", R_x, right_verts.cpu())
    left_joints = torch.einsum("ij,tnj->tni", R_x, left_joints.cpu())
    right_joints = torch.einsum("ij,tnj->tni", R_x, right_joints.cpu())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = Renderer(width, height, img_focal, device)
    K = torch.tensor(
        [[img_focal, 0, width / 2], [0, img_focal, height / 2], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    output_mp4 = output_mp4.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to open video writer: {output_mp4}")

    side_labels = ("Left", "Right")
    hand_colors = ((0.207, 0.596, 0.792), (0.804, 0.6, 0.820))
    hand_faces = (faces_left, faces_right)
    hand_verts = (left_verts, right_verts)
    hand_joints = (left_joints, right_joints)

    try:
        for frame_idx, frame in enumerate(background_frames):
            sam_left, sam_right = _load_frame_sam_masks(
                masks_dir,
                frame_idx,
                height,
                width,
                sam3.load_mask_png,
                mapping=timeline,
            )
            overlay = frame.copy()
            for mask, color in (
                (sam_left, (80, 220, 100)),
                (sam_right, (255, 120, 30)),
            ):
                if np.any(mask):
                    overlay[mask] = (
                        0.45 * overlay[mask] + 0.55 * np.array(color, dtype=np.uint8)
                    ).astype(np.uint8)

            R_w2c = R_w2c_sla_all[frame_idx : frame_idx + 1]
            t_w2c = t_w2c_sla_all[frame_idx : frame_idx + 1]
            for hand_idx in (0, 1):
                if not bool(valid_hand[frame_idx][hand_idx]):
                    continue
                overlay = _blend_hand(
                    overlay,
                    renderer,
                    hand_faces[hand_idx],
                    hand_verts[hand_idx][frame_idx : frame_idx + 1],
                    hand_colors[hand_idx],
                    R_w2c,
                    t_w2c,
                )
                joints_2d = _project_world_joints(
                    hand_joints[hand_idx][frame_idx],
                    R_w2c,
                    t_w2c,
                    K,
                )
                overlay = _draw_hand_joints(overlay, joints_2d)

            y = 28
            for hand_idx, label in enumerate(side_labels):
                accepted = bool(valid_hand[frame_idx][hand_idx])
                mc = mano_c[frame_idx][hand_idx]
                jc = joint_c[frame_idx][hand_idx]
                reason = reasons[frame_idx][hand_idx]
                status = "valid" if accepted else "reject"
                text = f"{label}: {status} | mano={mc:.2f} joint={jc:.2f}"
                if not accepted and reason:
                    text += f" ({reason})"
                cv2.putText(
                    overlay,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (40, 240, 40) if accepted else (40, 40, 240),
                    2,
                    cv2.LINE_AA,
                )
                y += 28
            sam_t = timeline.hawor_to_sam_time(frame_idx)
            cv2.putText(
                overlay,
                f"hawor={frame_idx} sam_t={sam_t:.2f}",
                (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            writer.write(overlay)
    finally:
        writer.release()
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return output_mp4
