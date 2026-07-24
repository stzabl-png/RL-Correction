"""Load ViPE / depth / camera artifacts from interim step directories."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

VIPE_FOCAL_FLIP_MATRIX = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]


def _np():
    import numpy as np

    return np


def load_vipe_poses(vipe_dir: Path, video_id: str):
    """Return (frame_indices, c2w 4x4 array) from ViPE pose npz."""
    np = _np()
    data = np.load(vipe_dir / "pose" / f"{video_id}.npz")
    order = np.argsort(data["inds"])
    return data["inds"][order], data["data"][order]


def load_vipe_intrinsics(vipe_dir: Path, video_id: str):
    """Return 3x3 K and focal_flip_xy flag."""
    np = _np()
    data = np.load(vipe_dir / "intrinsics" / f"{video_id}.npz")
    order = np.argsort(data["inds"])
    intr = data["data"][order][0]
    fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
    flip = fx < 0 or fy < 0
    fx, fy = abs(fx), abs(fy)
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return k, flip


def apply_vipe_focal_flip_c2w(c2w):
    np = _np()
    flip = np.array(VIPE_FOCAL_FLIP_MATRIX, dtype=c2w.dtype)
    out = c2w.copy()
    out[:3, :3] = out[:3, :3] @ flip
    return out


def interpolate_c2w_poses(inds, poses, num_frames: int):
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    np = _np()
    target = np.arange(num_frames, dtype=np.float64)
    target = np.clip(target, float(inds[0]), float(inds[-1]))
    trans = np.stack(
        [np.interp(target, inds.astype(float), poses[:, axis, 3]) for axis in range(3)],
        axis=1,
    )
    rots = R.from_matrix(poses[:, :3, :3])
    slerp = Slerp(inds.astype(float), rots)
    out = np.repeat(np.eye(4)[None], num_frames, axis=0)
    out[:, :3, :3] = slerp(target).as_matrix()
    out[:, :3, 3] = trans
    return out


def read_depth_exr_bytes(raw: bytes):
    np = _np()

    try:
        import io as _io

        import Imath
        import OpenEXR

        exr = OpenEXR.InputFile(_io.BytesIO(raw))
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = list(header["channels"].keys())
        channel = "Z" if "Z" in channels else ("Y" if "Y" in channels else channels[0])
        pixel_type = Imath.PixelType(Imath.PixelType.HALF if channel == "Z" else Imath.PixelType.FLOAT)
        dtype = np.float16 if channel == "Z" else np.float32
        arr = np.frombuffer(exr.channel(channel, pixel_type), dtype=dtype).reshape(height, width)
        return arr.astype(np.float32)
    except ImportError:
        pass
    except Exception:
        pass

    import os

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("Failed to decode EXR depth frame")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if np.nanmax(arr) <= 0:
        raise ValueError(
            "Decoded EXR depth is all zero. ViPE depth artifacts use a HALF Z channel; "
            "install/use OpenEXR in this environment to read them correctly."
        )
    return arr.astype(np.float32)


def iter_vipe_depth_frames(vipe_dir: Path, video_id: str):
    """Yield (frame_index, depth float32 HxW) from ViPE depth zip."""
    zpath = vipe_dir / "depth" / f"{video_id}.zip"
    with zipfile.ZipFile(zpath, "r") as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".exr"))
        for name in names:
            stem = Path(name).stem
            try:
                idx = int(stem.split("_")[-1])
            except ValueError:
                idx = int(stem) if stem.isdigit() else 0
            with zf.open(name) as f:
                depth = read_depth_exr_bytes(f.read())
            yield idx, depth


def load_vipe_depth_frame(vipe_dir: Path, video_id: str, frame_idx: int):
    for idx, depth in iter_vipe_depth_frames(vipe_dir, video_id):
        if idx == frame_idx:
            return depth
    raise FileNotFoundError(f"Depth frame {frame_idx} not in {vipe_dir}/depth/{video_id}.zip")


def depth_median_in_mask(depth, mask) -> float:
    np = _np()
    valid = (mask > 0) & np.isfinite(depth) & (depth > 1e-4)
    if not np.any(valid):
        raise ValueError("No valid depth pixels inside mask")
    return float(np.median(depth[valid]))


def export_fp_scene_frame(
    *,
    rgb_bgr,
    depth_m,
    k,
    out_rgb: Path,
    out_depth: Path,
    out_k: Path,
) -> None:
    import cv2

    np = _np()
    h, w = rgb_bgr.shape[:2]
    if w % 2 or h % 2:
        w2, h2 = w - w % 2, h - h % 2
        rgb_bgr = rgb_bgr[:h2, :w2]
        depth_m = depth_m[:h2, :w2]
        k = k.copy()
        # keep principal point; crop is top-left
    out_rgb.parent.mkdir(parents=True, exist_ok=True)
    out_depth.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_rgb), rgb_bgr)
    np.save(out_depth, depth_m.astype(np.float32))
    np.savetxt(out_k, k, fmt="%.8f")


def read_video_frame(video_path: Path, frame_idx: int):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def count_video_frames(video_path: Path) -> int:
    """Frame count via OpenCV, with ffprobe fallback (no numpy required)."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n > 0:
                return n
    except Exception:
        pass

    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip().isdigit():
        return int(proc.stdout.strip())

    raise RuntimeError(
        f"Could not count frames in {video_path}. Install opencv-python or ensure ffprobe is on PATH."
    )
