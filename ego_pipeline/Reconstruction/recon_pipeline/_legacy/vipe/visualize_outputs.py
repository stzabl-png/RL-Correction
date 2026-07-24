#!/usr/bin/env python3
"""Render a 2x2 ViPE visualization video from saved artifacts (headless-safe)."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _common import (
    DEFAULT_HOI4D_ROOT,
    DEFAULT_OUTPUT_DIR,
    artifact_paths,
    emit_progress,
    ensure_vipe_importable,
    hoi4d_rgb_video,
    hoi4d_rel_from_name,
    is_sequence_complete,
    setup_script_imports,
)

setup_script_imports()


@dataclass
class TrajectoryView:
    """Fixed orthographic 3D projection computed once from the full trajectory."""

    proj: np.ndarray
    center_uv: np.ndarray
    span: float
    width: int
    height: int
    view_forward: np.ndarray
    pad: int = 28

    @classmethod
    def from_poses(
        cls,
        poses: np.ndarray,
        width: int,
        height: int,
        frustum_depth: float,
        *,
        elevation_deg: float = 32.0,
        azimuth_deg: float = 42.0,
    ) -> TrajectoryView:
        proj, view_forward = _view_projection_matrix(elevation_deg, azimuth_deg)
        points: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]

        for pose in poses:
            points.append(pose[:3, 3])
            points.extend(_frustum_corners_world(pose, frustum_depth))

        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            points.append(np.array(axis, dtype=np.float64) * frustum_depth * 2.5)

        uv = np.stack([proj @ p for p in points], axis=0)
        min_uv = uv.min(axis=0)
        max_uv = uv.max(axis=0)
        center_uv = 0.5 * (min_uv + max_uv)
        half = 0.5 * float(np.max(max_uv - min_uv))
        half = max(half, 1e-3) * 1.14
        return cls(
            proj=proj,
            center_uv=center_uv,
            span=2.0 * half,
            width=width,
            height=height,
            view_forward=view_forward,
        )

    @classmethod
    def from_pose_sets(
        cls,
        pose_sets: Sequence[np.ndarray],
        width: int,
        height: int,
        frustum_depth: float,
        *,
        elevation_deg: float = 32.0,
        azimuth_deg: float = 42.0,
    ) -> TrajectoryView:
        """Build one view whose span fits all pose sets (shared camera-panel scale)."""
        proj, view_forward = _view_projection_matrix(elevation_deg, azimuth_deg)
        points: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]

        for poses in pose_sets:
            for pose in poses:
                points.append(pose[:3, 3])
                points.extend(_frustum_corners_world(pose, frustum_depth))

        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            points.append(np.array(axis, dtype=np.float64) * frustum_depth * 2.5)

        uv = np.stack([proj @ p for p in points], axis=0)
        min_uv = uv.min(axis=0)
        max_uv = uv.max(axis=0)
        center_uv = 0.5 * (min_uv + max_uv)
        half = 0.5 * float(np.max(max_uv - min_uv))
        half = max(half, 1e-3) * 1.14
        return cls(
            proj=proj,
            center_uv=center_uv,
            span=2.0 * half,
            width=width,
            height=height,
            view_forward=view_forward,
        )

    def project(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim == 1:
            return self.proj @ xyz
        return (self.proj @ xyz.T).T

    def to_canvas(self, uv: np.ndarray) -> tuple[int, int]:
        usable_w = max(1, self.width - 2 * self.pad)
        usable_h = max(1, self.height - 2 * self.pad)
        u = int((uv[0] - self.center_uv[0]) / self.span * usable_w + self.width * 0.5)
        v = int((uv[1] - self.center_uv[1]) / self.span * usable_h + self.height * 0.5)
        return u, v

    def view_depth(self, xyz: np.ndarray) -> float:
        xyz = np.asarray(xyz, dtype=np.float64)
        return float(xyz @ self.view_forward)


def _view_projection_matrix(
    elevation_deg: float,
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic projection from OpenCV world XYZ to 2D plot coordinates."""
    elev = np.deg2rad(elevation_deg)
    azim = np.deg2rad(azimuth_deg)
    ce, se = np.cos(elev), np.sin(elev)
    ca, sa = np.cos(azim), np.sin(azim)
    h_axis = np.array([ca, 0.0, sa], dtype=np.float64)
    v_axis = np.array([sa * se, ce, -ca * se], dtype=np.float64)
    view_forward = np.cross(h_axis, v_axis)
    norm = np.linalg.norm(view_forward)
    if norm > 1e-8:
        view_forward /= norm
    return np.stack([h_axis, v_axis], axis=0), view_forward


def _text_banner(text: str, width: int) -> np.ndarray:
    height = 28
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((width - text_w) // 2, 4), text, fill=(0, 0, 0), font=font)
    return np.asarray(img)


def _load_poses(pose_npz: Path) -> np.ndarray:
    data = np.load(pose_npz)
    order = np.argsort(data["inds"])
    return data["data"][order]


def _frustum_depth_for_pose_sets(pose_sets: Sequence[np.ndarray]) -> float:
    centers = np.concatenate([np.asarray(poses)[:, :3, 3] for poses in pose_sets], axis=0)
    if len(centers) == 0:
        return 1e-3
    extent = float(np.linalg.norm(centers - centers[0], axis=1).max())
    return max(extent * 0.15, 1e-3)


def shared_trajectory_view_for_poses(
    pose_sets: Sequence[np.ndarray],
    width: int,
    height: int,
) -> tuple[TrajectoryView, float]:
    """One TrajectoryView and frustum depth shared across multiple pose trajectories."""
    frustum_depth = _frustum_depth_for_pose_sets(pose_sets)
    view = TrajectoryView.from_pose_sets(pose_sets, width, height, frustum_depth)
    return view, frustum_depth


def _frustum_corners_world(c2w: np.ndarray, depth: float, aspect: float = 16 / 9) -> np.ndarray:
    half_h = depth * 0.45
    half_w = half_h * aspect
    corners_cam = np.array(
        [
            [-half_w, -half_h, depth],
            [half_w, -half_h, depth],
            [half_w, half_h, depth],
            [-half_w, half_h, depth],
        ],
        dtype=np.float64,
    )
    rot = c2w[:3, :3]
    trans = c2w[:3, 3]
    return (rot @ corners_cam.T).T + trans


def _depth_range_from_zip(read_depth_artifacts, depth_zip: Path) -> tuple[float, float]:
    depth_range = [np.inf, -np.inf]
    for _, depth_tensor in read_depth_artifacts(depth_zip):
        depth = depth_tensor.numpy()
        valid = np.isfinite(depth) & (depth > 1e-6)
        if not np.any(valid):
            continue
        inv = 1.0 / depth[valid]
        q05, q95 = np.quantile(inv, [0.05, 0.95])
        depth_range[0] = min(depth_range[0], q05)
        depth_range[1] = max(depth_range[1], q95)
    if not np.isfinite(depth_range[0]):
        return 0.0, 1.0
    middle = 0.5 * (depth_range[0] + depth_range[1])
    span = depth_range[1] - depth_range[0]
    return middle - 0.65 * span, middle + 0.65 * span


def _colorize_depth_frame(depth: np.ndarray, depth_min: float, depth_max: float, sky_mask: np.ndarray | None) -> np.ndarray:
    inv = np.zeros_like(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    inv[valid] = 1.0 / depth[valid]
    if sky_mask is not None:
        inv[sky_mask] = depth_min
    inv[~np.isfinite(inv)] = depth_min
    norm = (inv - depth_min) / max(depth_max - depth_min, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _overlay_segmentation(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from vipe.utils.visualization import colorize_mask

    seg = colorize_mask(mask)
    if seg.dtype != np.uint8:
        seg = (seg[..., :3] * 255).astype(np.uint8)
    if seg.shape[:2] != rgb.shape[:2]:
        seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(rgb, 0.5, seg, 0.5, 0)


def _draw_world_axes(canvas: np.ndarray, view: TrajectoryView) -> None:
    origin_uv = view.project(np.zeros(3, dtype=np.float64))
    origin = view.to_canvas(origin_uv)
    axis_len = 0.18 * view.span

    axes = (
        (np.array([1.0, 0.0, 0.0]), (220, 70, 70), "X"),
        (np.array([0.0, 1.0, 0.0]), (70, 170, 80), "Y"),
        (np.array([0.0, 0.0, 1.0]), (70, 90, 220), "Z"),
    )
    cv2.circle(canvas, origin, 4, (30, 30, 30), -1, cv2.LINE_AA)
    for direction, color, label in axes:
        tip = view.to_canvas(view.project(direction * axis_len))
        cv2.arrowedLine(canvas, origin, tip, color, 2, tipLength=0.15, line_type=cv2.LINE_AA)
        cv2.putText(canvas, label, (tip[0] + 4, tip[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "world origin",
        (origin[0] + 6, origin[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )


def _draw_camera_frustum(
    canvas: np.ndarray,
    view: TrajectoryView,
    pose: np.ndarray,
    frustum_depth: float,
    *,
    highlight: bool,
) -> None:
    center = pose[:3, 3]
    center_px = view.to_canvas(view.project(center))

    corners_world = _frustum_corners_world(pose, frustum_depth)
    corners_px = np.array([view.to_canvas(view.project(c)) for c in corners_world], dtype=np.int32)

    plane_color = (40, 170, 230) if highlight else (150, 170, 190)
    line_color = (30, 120, 210) if highlight else (120, 130, 150)
    center_color = (220, 60, 60) if highlight else (100, 100, 100)

    for i in range(4):
        cv2.line(canvas, center_px, tuple(corners_px[i]), line_color, 1, cv2.LINE_AA)
    for i in range(4):
        cv2.line(canvas, tuple(corners_px[i]), tuple(corners_px[(i + 1) % 4]), plane_color, 2, cv2.LINE_AA)
    cv2.circle(canvas, center_px, 4 if highlight else 3, center_color, -1, cv2.LINE_AA)


def _render_pose_panel(
    poses: np.ndarray,
    frame_idx: int,
    view: TrajectoryView,
    frustum_depth: float,
) -> np.ndarray:
    canvas = np.full((view.height, view.width, 3), 255, dtype=np.uint8)
    _draw_world_axes(canvas, view)

    past_centers = poses[:frame_idx]
    if len(past_centers) > 0:
        poly = np.array([view.to_canvas(view.project(p[:3, 3])) for p in past_centers], dtype=np.int32)
        if len(poly) >= 2:
            cv2.polylines(canvas, [poly], False, (170, 170, 170), 1, cv2.LINE_AA)

    past_indices = list(range(frame_idx))
    past_indices.sort(key=lambda idx: view.view_depth(poses[idx][:3, 3]))
    for idx in past_indices:
        _draw_camera_frustum(canvas, view, poses[idx], frustum_depth, highlight=False)

    _draw_camera_frustum(canvas, view, poses[frame_idx], frustum_depth, highlight=True)

    cv2.putText(
        canvas,
        "Camera pose (tilted 3D view, fixed scale)",
        (8, view.height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _resolve_rgb_path(
    paths: dict[str, Path],
    rgb_dir: Path | None,
    *,
    hoi4d_root: Path | None = None,
) -> Path:
    if paths["rgb"].is_file():
        return paths["rgb"]
    if rgb_dir is not None:
        candidate = rgb_dir / f"{paths['pose'].stem}.mp4"
        if candidate.is_file():
            return candidate
    if hoi4d_root is not None:
        rel = hoi4d_rel_from_name(paths["pose"].stem)
        candidate = hoi4d_rgb_video(hoi4d_root, rel)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"RGB video not found for {paths['pose'].stem}")


def render_sequence(
    output_dir: Path,
    name: str,
    *,
    downsample: int = 2,
    overwrite: bool = False,
    rgb_dir: Path | None = None,
    hoi4d_root: Path | None = None,
) -> Path:
    paths = artifact_paths(output_dir, name)
    if not is_sequence_complete(output_dir, name):
        raise FileNotFoundError(f"Missing ViPE artifacts for sequence: {name}")

    viz_path = paths["viz"]
    if viz_path.exists() and not overwrite:
        return viz_path

    ensure_vipe_importable()
    from vipe.utils.io import read_depth_artifacts, read_instance_artifacts, read_instance_phrases
    from vipe.utils.visualization import VideoWriter

    poses = _load_poses(paths["pose"])
    print(f"  {name}: rendering {len(poses)} frames...", flush=True)
    phrases = read_instance_phrases(paths["mask_phrases"]) if paths["mask_phrases"].is_file() else {}
    depth_min, depth_max = _depth_range_from_zip(read_depth_artifacts, paths["depth"])

    rgb_path = _resolve_rgb_path(paths, rgb_dir, hoi4d_root=hoi4d_root)
    reader = imageio.get_reader(str(rgb_path), "ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 15.0))
    size = meta.get("size", None)
    if size is None:
        first = reader.get_data(0)
        height, width = first.shape[:2]
    else:
        width, height = size

    panel_w = max(1, width // downsample)
    panel_h = max(1, height // downsample)

    centers = poses[:, :3, 3]
    path_extent = float(np.linalg.norm(centers - centers[0], axis=1).max()) if len(centers) else 1.0
    frustum_depth = max(path_extent * 0.08, 1e-3)
    traj_view = TrajectoryView.from_poses(poses, panel_w, panel_h, frustum_depth)

    depth_iter = read_depth_artifacts(paths["depth"])
    mask_iter = read_instance_artifacts(paths["mask"]) if paths["mask"].is_file() else None

    viz_path.parent.mkdir(parents=True, exist_ok=True)
    with VideoWriter(viz_path, fps) as writer:
        frame_idx = 0
        while True:
            try:
                rgb = reader.get_data(frame_idx)
            except (IndexError, RuntimeError):
                break

            rgb = cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

            mask_np: np.ndarray | None = None
            if mask_iter is not None:
                mask_idx, mask_tensor = next(mask_iter)
                assert mask_idx == frame_idx, f"Mask frame mismatch: {mask_idx} vs {frame_idx}"
                mask_np = mask_tensor.numpy()
                seg_panel = _overlay_segmentation(rgb, mask_np)
            else:
                seg_panel = rgb.copy()

            depth_idx, depth_tensor = next(depth_iter)
            assert depth_idx == frame_idx, f"Depth frame mismatch: {depth_idx} vs {frame_idx}"
            sky = (mask_np == 1) if mask_np is not None and 1 in phrases else None
            depth_panel = _colorize_depth_frame(depth_tensor.numpy(), depth_min, depth_max, sky)
            depth_panel = cv2.resize(depth_panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

            pose_panel = _render_pose_panel(poses, frame_idx, traj_view, frustum_depth)

            top = np.concatenate([rgb, seg_panel], axis=1)
            bottom = np.concatenate([depth_panel, pose_panel], axis=1)
            grid = np.concatenate([top, bottom], axis=0)

            banner = _text_banner(
                f"Frame {frame_idx:03d} | RGB | Segmentation | Depth | Camera pose",
                grid.shape[1],
            )
            frame = np.concatenate([banner, grid], axis=0)
            writer.write(frame)
            frame_idx += 1

    reader.close()
    if frame_idx != len(poses):
        raise RuntimeError(f"Frame count mismatch for {name}: rgb={frame_idx}, pose={len(poses)}")
    return viz_path


def _render_job(payload: tuple[Path, str, int, bool, Path | None, Path | None]) -> tuple[str, str, bool]:
    output_dir, name, downsample, overwrite, rgb_dir, hoi4d_root = payload
    setup_script_imports()
    paths = artifact_paths(output_dir, name)
    if paths["viz"].exists() and not overwrite:
        return name, str(paths["viz"]), True
    try:
        out = render_sequence(
            output_dir,
            name,
            downsample=downsample,
            overwrite=overwrite,
            rgb_dir=rgb_dir,
            hoi4d_root=hoi4d_root,
        )
        return name, str(out), False
    except Exception as exc:  # noqa: BLE001 - collect per-sequence failures for batch runs
        return name, f"ERROR: {exc}", False


def _discover_sequence_names(output_dir: Path) -> list[str]:
    pose_dir = output_dir / "pose"
    if pose_dir.is_dir():
        return sorted(path.stem for path in pose_dir.glob("*.npz"))
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"ViPE output root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Single sequence name (stem). If omitted, process all complete sequences.",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="Optional flat directory of source RGB MP4s if output rgb/ artifacts were removed",
    )
    parser.add_argument(
        "--hoi4d-root",
        type=Path,
        default=DEFAULT_HOI4D_ROOT,
        help=f"HOI4D dataset root for nested RGB lookup (default: {DEFAULT_HOI4D_ROOT})",
    )
    parser.add_argument("--downsample", type=int, default=2, help="Panel downsample factor (default: 2)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing vis/*.mp4 files")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for batch visualization (default: 1)",
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    output_dir = args.output_dir.resolve()
    rgb_dir = args.rgb_dir.resolve() if args.rgb_dir is not None else None
    hoi4d_root = args.hoi4d_root.resolve()

    if args.sequence:
        names = [args.sequence]
    else:
        names = _discover_sequence_names(output_dir)

    if not names:
        raise SystemExit(f"No sequences found under {output_dir}")

    jobs = [
        (output_dir, name, args.downsample, args.overwrite, rgb_dir, hoi4d_root)
        for name in names
    ]
    failures: list[str] = []
    total = len(jobs)
    print(f"Visualizing {total} sequence(s) with {args.workers} worker(s)...", flush=True)
    completed = 0

    def _report(name: str, result: str, skipped: bool) -> None:
        nonlocal completed
        completed += 1
        if result.startswith("ERROR:"):
            emit_progress(completed, total, "fail", name, detail=result)
            failures.append(f"{name}: {result}")
        elif skipped:
            emit_progress(completed, total, "skip", name, detail="exists")
        else:
            emit_progress(completed, total, "done", name, detail=result)

    if args.workers == 1:
        for job in jobs:
            name, result, skipped = _render_job(job)
            _report(name, result, skipped)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_render_job, job) for job in jobs]
            for future in as_completed(futures):
                name, result, skipped = future.result()
                _report(name, result, skipped)

    print(f"Finished: {completed - len(failures)}/{total} succeeded, {len(failures)} failed", flush=True)

    if failures:
        raise SystemExit("Visualization failed for:\n" + "\n".join(failures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
