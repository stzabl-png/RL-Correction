"""Optional per-step visualization (--visualize only).

Contract (all recon_pipeline steps):
- Off by default; one artifact under ``{step_dir}/vis/`` when enabled.
- Each visualization shows every output this step produces (combined in one file).
- No extra interim PNG sequences or debug dumps are kept for viz.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from _common.mesh_viz import (
    TriangleLayer,
    collect_mesh_triangles,
    draw_sorted_triangles,
    face_stride as mesh_face_stride,
    render_object_pose_overlay,
    transform_points,
)

VIS_DIRNAME = "vis"

# RGB overlay colors (RGB for SAM3 hand helper, BGR for OpenCV writers)
LEFT_HAND_RGB = (80, 220, 100)
RIGHT_HAND_RGB = (255, 120, 30)
OBJECT_MASK_BGR = (40, 120, 255)
OBJECT_MESH_BGR = (80, 200, 80)
OBJECT_MASK_PALETTE_BGR = (
    (40, 120, 255),
    (80, 220, 80),
    (255, 120, 40),
    (220, 80, 220),
    (40, 220, 220),
    (180, 180, 60),
    (255, 80, 120),
    (120, 180, 255),
    (160, 100, 40),
)


def vis_path(step_dir: Path, video_id: str, *, ext: str = "mp4") -> Path:
    return step_dir / VIS_DIRNAME / f"{video_id}.{ext}"


def vis_png_path(step_dir: Path, video_id: str, name: str) -> Path:
    return step_dir / VIS_DIRNAME / f"{video_id}_{name}.png"


def _overlay_mask_bgr(frame, mask, color_bgr: tuple[int, int, int], alpha: float = 0.45):
    import cv2
    import numpy as np

    overlay = frame.copy()
    m = mask > 0 if mask.dtype != bool else mask
    color = np.array(color_bgr, dtype=np.float32)
    overlay[m] = (overlay[m].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def _raw_video_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_raw.avi")


def _open_bgr_writer(out_path: Path, fps: float, width: int, height: int):
    import cv2

    raw_path = _raw_video_path(out_path)
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed for {raw_path}")
    return writer, raw_path


def encode_hands_vis_mp4(
    *,
    video_path: Path,
    masks_dir: Path,
    out_path: Path,
    left_mask_name: str = "left_hand_0.png",
    right_mask_name: str = "right_hand_0.png",
) -> Path:
    """RGB video with left (green) and right (orange) hand masks overlaid together."""
    import cv2
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, raw_path = _open_bgr_writer(out_path, fps, w, h)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_dir = masks_dir / f"frame_{frame_idx:06d}_masks"
        for mask_name, color in (
            (left_mask_name, (100, 220, 80)),
            (right_mask_name, (30, 120, 255)),
        ):
            mask_path = frame_dir / mask_name
            if mask_path.is_file():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.any(mask > 127):
                    frame = _overlay_mask_bgr(frame, mask > 127, color)
        cv2.putText(
            frame,
            f"f{frame_idx} | L=green R=orange",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return _finalize_h264(raw_path, out_path)


def encode_object_mask_vis_mp4(
    *,
    video_path: Path,
    masks_dir: Path,
    mask_filename: str,
    out_path: Path,
    prompt_frame_idx: int | None = None,
    prompt_points: list[tuple[float, float]] | None = None,
    prompt_labels: list[int] | None = None,
) -> Path:
    """RGB + propagated object mask; draws click prompts on the label frame."""
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, raw_path = _open_bgr_writer(out_path, fps, w, h)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask_path = masks_dir / f"frame_{frame_idx:06d}_masks" / mask_filename
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                frame = _overlay_mask_bgr(frame, mask > 127, OBJECT_MASK_BGR)
        if prompt_frame_idx is not None and frame_idx == prompt_frame_idx and prompt_points:
            for (x, y), lbl in zip(prompt_points, prompt_labels or [1] * len(prompt_points)):
                color = (0, 255, 0) if lbl else (0, 0, 255)
                cv2.circle(frame, (int(x), int(y)), 8, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"f{frame_idx} object mask",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return _finalize_h264(raw_path, out_path)


def encode_object_masks_vis_mp4(
    *,
    video_path: Path,
    masks_dir: Path,
    objects: list[dict],
    out_path: Path,
) -> Path:
    """RGB + all propagated object masks; draws click prompts on each label frame."""
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, raw_path = _open_bgr_writer(out_path, fps, w, h)

    object_specs = []
    for idx, obj in enumerate(objects):
        color = OBJECT_MASK_PALETTE_BGR[idx % len(OBJECT_MASK_PALETTE_BGR)]
        object_specs.append(
            {
                "object_id": str(obj["object_id"]),
                "mask_filename": str(obj["mask_filename"]),
                "frame_idx": int(obj.get("frame_idx", obj.get("prompt_frame_idx", 0))),
                "points": obj.get("points") or [],
                "labels": obj.get("labels") or [],
                "color": color,
            }
        )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_dir = masks_dir / f"frame_{frame_idx:06d}_masks"
        for spec in object_specs:
            mask_path = frame_dir / spec["mask_filename"]
            if mask_path.is_file():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    frame = _overlay_mask_bgr(frame, mask > 127, spec["color"])

        for spec in object_specs:
            if frame_idx != spec["frame_idx"] or not spec["points"]:
                continue
            for (x, y), lbl in zip(spec["points"], spec["labels"] or [1] * len(spec["points"])):
                outer = spec["color"] if int(lbl) > 0 else (0, 0, 255)
                cv2.circle(frame, (int(x), int(y)), 8, outer, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 10, (255, 255, 255), 2, cv2.LINE_AA)

        y = 24
        cv2.putText(
            frame,
            f"f{frame_idx} object masks",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 22
        for spec in object_specs[:8]:
            color = spec["color"]
            cv2.rectangle(frame, (8, y - 12), (22, y + 2), color, -1)
            cv2.putText(
                frame,
                spec["object_id"],
                (28, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 18

        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return _finalize_h264(raw_path, out_path)


def encode_fp_pose_vis_mp4(
    *,
    video_path: Path,
    ob_in_cam_dir: Path,
    masks_dir: Path,
    mask_filename: str,
    k_mat,
    mesh_path: Path,
    start_frame: int,
    out_path: Path,
) -> Path:
    """Object mask + projected mesh, 3D bbox, and pose axes for tracked frames."""
    import cv2
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_stride_val = mesh_face_stride(len(faces))

    pose_files = {int(p.stem): p for p in ob_in_cam_dir.glob("*.txt")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, raw_path = _open_bgr_writer(out_path, fps, w, h)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask_path = masks_dir / f"frame_{frame_idx:06d}_masks" / mask_filename
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                frame = _overlay_mask_bgr(frame, mask > 127, OBJECT_MASK_BGR, alpha=0.22)
        if frame_idx in pose_files and frame_idx >= start_frame:
            t_obj_cam = np.loadtxt(pose_files[frame_idx])
            render_object_pose_overlay(
                frame,
                T_cam=t_obj_cam,
                mesh_verts_local=verts,
                mesh_faces=faces,
                K=k_mat,
                mesh_color_bgr=OBJECT_MESH_BGR,
                mesh_alpha=1.0,
                bbox_color_bgr=(0, 255, 255),
                face_stride=face_stride_val,
                draw_mesh=True,
                draw_bbox=True,
                draw_axes=True,
            )
            cv2.putText(
                frame,
                f"track f{frame_idx} | mesh+bbox+axes",
                (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"f{frame_idx} FP++ ob_in_cam",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return _finalize_h264(raw_path, out_path)


def encode_sam3d_vis_png(
    *,
    rgb_bgr,
    mask,
    mesh_path: Path,
    out_path: Path,
    frame_idx: int,
) -> Path:
    """Side-by-side: RGB+mask | raw SAM3D mesh preview (single reconstruction frame)."""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh

    out_path.parent.mkdir(parents=True, exist_ok=True)
    left = rgb_bgr.copy()
    left = _overlay_mask_bgr(left, mask > 0, OBJECT_MASK_BGR, alpha=0.4)
    cv2.putText(
        left,
        f"frame {frame_idx} RGB+mask",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.3, c=verts[:, 2], cmap="viridis")
    ax.set_title("raw SAM3D mesh")
    ax.set_axis_off()
    mesh_png = out_path.with_suffix(".mesh_tmp.png")
    fig.savefig(mesh_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    right = cv2.imread(str(mesh_png))
    mesh_png.unlink(missing_ok=True)
    if right is None:
        right = np.zeros_like(left)
    if right.shape[0] != left.shape[0]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    combo = np.concatenate([left, right], axis=1)
    cv2.imwrite(str(out_path), combo)
    return out_path


def encode_fuse_vis_mp4(
    *,
    video_path: Path,
    out_path: Path,
    ob_in_cam_dir: Path,
    frame_indices: list[int],
    k_mat: np.ndarray,
    mesh_path: Path,
    object_masks_dir: Path | None = None,
    object_mask_filename: str = "object_0.png",
    hand_geometry=None,
) -> Path:
    """Fused preview: RGB + transparent object mesh + MANO hands with depth ordering."""
    import cv2
    import trimesh

    from _common.hawor_mano import HandSequenceGeometry, draw_hand_skeleton

    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    obj_verts = np.asarray(mesh.vertices, dtype=np.float64)
    obj_faces = np.asarray(mesh.faces, dtype=np.int64)
    obj_stride = mesh_face_stride(len(obj_faces))
    fi_set = set(frame_indices)
    hands: HandSequenceGeometry | None = hand_geometry

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, raw_path = _open_bgr_writer(out_path, fps, w, h)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if object_masks_dir is not None:
            mp = object_masks_dir / f"frame_{frame_idx:06d}_masks" / object_mask_filename
            if mp.is_file():
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    frame = _overlay_mask_bgr(frame, mask > 127, OBJECT_MASK_BGR, alpha=0.15)

        triangles: list[TriangleLayer] = []
        if frame_idx in fi_set:
            t_cam = np.loadtxt(ob_in_cam_dir / f"{frame_idx:06d}.txt")
            obj_cam = transform_points(t_cam, obj_verts)
            triangles.extend(
                collect_mesh_triangles(
                    obj_cam,
                    obj_faces,
                    k_mat,
                    color_bgr=OBJECT_MESH_BGR,
                    alpha=1.0,
                    stride=obj_stride,
                )
            )

        hand_triangles: list[TriangleLayer] = []
        if hands is not None and frame_idx < len(hands.frames):
            hf = hands.frames[frame_idx]
            if hf.left_valid and hf.left_verts_cam is not None:
                hand_triangles.extend(
                    collect_mesh_triangles(
                        hf.left_verts_cam,
                        hands.faces_left,
                        k_mat,
                        color_bgr=(100, 220, 80),
                        alpha=1.0,
                        stride=1,
                    )
                )
            if hf.right_valid and hf.right_verts_cam is not None:
                hand_triangles.extend(
                    collect_mesh_triangles(
                        hf.right_verts_cam,
                        hands.faces_right,
                        k_mat,
                        color_bgr=(30, 120, 255),
                        alpha=1.0,
                        stride=1,
                    )
                )

        draw_sorted_triangles(frame, triangles, w=w, h=h)

        if hand_triangles:
            hand_overlay = frame.copy()
            draw_sorted_triangles(hand_overlay, hand_triangles, w=w, h=h)
            cv2.addWeighted(hand_overlay, 0.45, frame, 0.55, 0, dst=frame)

        if hands is not None and frame_idx < len(hands.frames):
            hf = hands.frames[frame_idx]
            if hf.left_valid and hf.left_joints_cam is not None:
                draw_hand_skeleton(frame, hf.left_joints_cam, k_mat, color_bgr=(100, 220, 80))
            if hf.right_valid and hf.right_joints_cam is not None:
                draw_hand_skeleton(frame, hf.right_joints_cam, k_mat, color_bgr=(30, 120, 255))

        if frame_idx in fi_set:
            cv2.putText(
                frame,
                f"fused track f{frame_idx}",
                (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        label = "world fuse | mesh+hands" if hands is not None else "world fuse | mesh only"
        cv2.putText(
            frame,
            f"f{frame_idx} {label}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
    return _finalize_h264(raw_path, out_path)


def _finalize_h264(raw_path: Path, out_path: Path) -> Path:
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing intermediate video: {raw_path}")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        raw_path.unlink(missing_ok=True)
        return out_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        fallback = out_path.with_suffix(".avi")
        if out_path.exists():
            out_path.unlink()
        if raw_path != fallback:
            raw_path.rename(fallback)
            return fallback
        return raw_path
