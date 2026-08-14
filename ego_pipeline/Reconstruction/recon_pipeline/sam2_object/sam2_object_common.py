"""Point-prompt object segmentation helpers backed by SAM2."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _common.artifacts import discard_sam2_object_nonessential
from _common.io import count_video_frames
from _common.paths import REPO_ROOT
from _common.viz import encode_object_mask_vis_mp4, encode_object_masks_vis_mp4, vis_path

LABEL_PROMPT_FILENAME = "label_prompt.json"
LABEL_PROMPT_SCHEMA_VERSION = "sam2_object_prompt_v2"
OBJECT_MASK_ID = "object_0"
SAM2_OBJECT_ID = 1
SAM2_ROOT = REPO_ROOT / "third_party" / "sam2"
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
OBJECT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_object_id(object_id: str) -> str:
    object_id = str(object_id)
    if not object_id or not OBJECT_ID_RE.match(object_id):
        raise ValueError(
            f"Invalid object_id {object_id!r}; use letters, numbers, underscore, dash, or dot only"
        )
    if object_id in {".", ".."}:
        raise ValueError(f"Invalid object_id {object_id!r}")
    return object_id


@dataclass
class ObjectPrompt:
    object_id: str
    frame_idx: int
    points: list[tuple[float, float]]
    labels: list[int]
    name: str | None = None
    locked: bool = False

    def to_json(self) -> dict[str, Any]:
        data = {
            "object_id": self.object_id,
            "frame_idx": self.frame_idx,
            "points": [[float(x), float(y)] for x, y in self.points],
            "labels": [int(v) for v in self.labels],
            "locked": bool(self.locked),
        }
        if self.name is not None:
            data["name"] = self.name
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any], *, default_object_id: str = OBJECT_MASK_ID) -> "ObjectPrompt":
        if not isinstance(data, dict):
            raise ValueError("Each object prompt must be a JSON object")
        try:
            frame_idx = int(data["frame_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Object prompt frame_idx must be an integer") from exc
        if frame_idx < 0:
            raise ValueError(f"Object prompt frame_idx must be non-negative, got {frame_idx}")

        raw_points = data.get("points")
        raw_labels = data.get("labels")
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError("Object prompt points must be a non-empty list")
        if not isinstance(raw_labels, list) or len(raw_labels) != len(raw_points):
            raise ValueError("Object prompt labels must have the same length as points")

        points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"Each prompt point must be [x, y], got {point!r}")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Prompt coordinates must be finite, got {point!r}")
            points.append((x, y))

        labels = [int(value) for value in raw_labels]
        if any(value not in (0, 1) for value in labels):
            raise ValueError(f"Prompt labels must contain only 0 or 1, got {labels}")
        if not any(labels):
            raise ValueError("Each object prompt needs at least one positive point (label=1)")

        return cls(
            object_id=validate_object_id(data.get("object_id") or default_object_id),
            frame_idx=frame_idx,
            points=points,
            labels=labels,
            name=data.get("name"),
            locked=bool(data.get("locked", False)),
        )

    @property
    def mask_filename(self) -> str:
        return f"{self.object_id}.png"


@dataclass
class LabelPrompt:
    objects: list[ObjectPrompt]
    schema_version: str = LABEL_PROMPT_SCHEMA_VERSION

    @property
    def primary(self) -> ObjectPrompt:
        if not self.objects:
            raise ValueError("LabelPrompt has no objects")
        return self.objects[0]

    @property
    def frame_idx(self) -> int:
        return self.primary.frame_idx

    @property
    def points(self) -> list[tuple[float, float]]:
        return self.primary.points

    @property
    def labels(self) -> list[int]:
        return self.primary.labels

    def object_ids(self) -> list[str]:
        return [obj.object_id for obj in self.objects]

    def get_object(self, object_id: str = OBJECT_MASK_ID) -> ObjectPrompt:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(f"Object prompt {object_id!r} not found; available={self.object_ids()}")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objects": [obj.to_json() for obj in self.objects],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LabelPrompt":
        if "objects" not in data:
            return cls(objects=[ObjectPrompt.from_json(data, default_object_id=OBJECT_MASK_ID)])
        objects = [
            ObjectPrompt.from_json(obj, default_object_id=f"object_{idx}")
            for idx, obj in enumerate(data.get("objects") or [])
        ]
        if not objects:
            raise ValueError("Multi-object label prompt has no objects")
        object_ids = [obj.object_id for obj in objects]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError(f"Duplicate object ids in label prompt: {object_ids}")
        return cls(objects=objects, schema_version=str(data.get("schema_version") or LABEL_PROMPT_SCHEMA_VERSION))


def label_prompt_path(step_dir: Path) -> Path:
    return step_dir / LABEL_PROMPT_FILENAME


def has_label_prompt(step_dir: Path) -> bool:
    return label_prompt_path(step_dir).is_file()


def save_label_prompt(step_dir: Path, prompt: LabelPrompt) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    path = label_prompt_path(step_dir)
    path.write_text(json.dumps(prompt.to_json(), indent=2), encoding="utf-8")
    return path


def load_label_prompt(step_dir: Path) -> LabelPrompt:
    return LabelPrompt.from_json(json.loads(label_prompt_path(step_dir).read_text(encoding="utf-8")))


def validate_label_prompt_for_video(
    prompt: LabelPrompt,
    *,
    num_frames: int,
    height: int,
    width: int,
) -> None:
    """Validate frame and point coordinates before loading the SAM2 model."""
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"Invalid video geometry: frames={num_frames}, height={height}, width={width}"
        )
    for obj in prompt.objects:
        if obj.frame_idx >= num_frames:
            raise ValueError(
                f"Prompt frame for {obj.object_id} is outside the video: "
                f"{obj.frame_idx} not in [0, {num_frames - 1}]"
            )
        for x, y in obj.points:
            if not (0.0 <= x < width and 0.0 <= y < height):
                raise ValueError(
                    f"Prompt point for {obj.object_id} is outside the video frame: "
                    f"({x}, {y}) not within {width}x{height}"
                )


def clear_generated_object_outputs(step_dir: Path) -> None:
    """Remove only generated SAM2 outputs; preserve label_prompt.json and frame_plan.json."""
    for dirname in ("video_segmentation", "vis"):
        shutil.rmtree(step_dir / dirname, ignore_errors=True)
    for filename in ("object_masks_vis.mp4", "object_masks_complete.json"):
        (step_dir / filename).unlink(missing_ok=True)


def object_mask_filename(object_id: str) -> str:
    return f"{validate_object_id(object_id)}.png"


def object_output_dir(step_dir: Path, object_id: str) -> Path:
    return step_dir / "objects" / validate_object_id(object_id)


def _load_legacy_sam_video_utils():
    """Load legacy shared utilities for video IO and mask checks only."""
    import importlib.util

    sam3_path = REPO_ROOT / "recon_pipeline" / "_legacy" / "sam3" / "common.py"
    module_name = "sam3_common_recon"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, sam3_path)
    sc = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = sc
    assert spec.loader
    spec.loader.exec_module(sc)
    return sc


def _sam2_checkpoint_hint(checkpoint: Path) -> str:
    return (
        f"SAM2 checkpoint not found: {checkpoint}. "
        "Download checkpoints with: cd third_party/sam2/checkpoints && ./download_ckpts.sh"
    )


def build_object_predictor(
    *,
    gpu_id: int | None,
    device: str | None = None,
    checkpoint: Path | None = None,
    model_cfg: str = DEFAULT_SAM2_MODEL_CFG,
    apply_postprocessing: bool = True,
    **_unused: Any,
):
    if device is None:
        device = "cuda" if gpu_id is not None else "cpu"
    if device == "cuda":
        if gpu_id is None:
            gpu_id = 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        resolved_device = "cuda:0"
    else:
        resolved_device = "cpu"
    checkpoint = (checkpoint or DEFAULT_SAM2_CHECKPOINT).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(_sam2_checkpoint_hint(checkpoint))
    if not SAM2_ROOT.is_dir():
        raise FileNotFoundError(f"SAM2 submodule missing: {SAM2_ROOT}")
    if str(SAM2_ROOT) not in sys.path:
        sys.path.insert(0, str(SAM2_ROOT))

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAM2 CUDA preview requested, but torch.cuda.is_available() is false")
    return build_sam2_video_predictor(
        model_cfg,
        str(checkpoint),
        device=resolved_device,
        apply_postprocessing=apply_postprocessing,
    )


def predictor_device_type(predictor) -> str:
    device = getattr(getattr(predictor, "model", predictor), "device", None)
    return getattr(device, "type", str(device or "cpu"))


def _mask_for_object(obj_ids, mask_logits, obj_id: int = SAM2_OBJECT_ID) -> np.ndarray | None:
    for idx, oid in enumerate(obj_ids):
        if int(oid) == int(obj_id):
            mask = (mask_logits[idx] > 0.0).detach().cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            return mask.astype(bool)
    return None


def ensure_object_preview_video_state(predictor, *, video_path: Path, session_cache: dict[str, Any]):
    """Initialize/reuse the SAM2 interactive state for one video."""
    video_key = str(video_path.resolve())
    state = session_cache.get("sam2_state")
    if state is None or session_cache.get("video") != video_key:
        if state is not None and hasattr(predictor, "reset_state"):
            predictor.reset_state(state)
        state = predictor.init_state(video_path=video_key)
        session_cache["sam2_state"] = state
        session_cache["video"] = video_key
    return state


def preview_object_mask_on_frame(
    predictor,
    *,
    video_path: Path,
    frame_idx: int,
    points: list[tuple[float, float]],
    labels: list[int],
    session_cache: dict[str, Any],
    output_prob_thresh: float = 0.5,
) -> np.ndarray | None:
    """Run SAM2 point prompts on one frame; return boolean mask or None."""
    del output_prob_thresh
    if not points or not any(int(v) for v in labels):
        return None

    state = ensure_object_preview_video_state(predictor, video_path=video_path, session_cache=session_cache)

    import torch

    pts_arr = np.array(points, dtype=np.float32)
    lbl_arr = np.array(labels, dtype=np.int32)
    device_type = predictor_device_type(predictor)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        _, obj_ids, mask_logits = predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=SAM2_OBJECT_ID,
            points=pts_arr,
            labels=lbl_arr,
            clear_old_points=True,
            normalize_coords=True,
        )
    mask = _mask_for_object(obj_ids, mask_logits)
    if mask is None or not np.any(mask):
        return None
    return mask


def encode_object_mask_png(mask: np.ndarray | None) -> bytes:
    """Encode boolean mask as RGBA PNG (alpha=foreground, transparent background)."""
    import cv2

    if mask is None or not np.any(mask):
        raise ValueError("empty mask")
    fg = mask.astype(bool)
    h, w = fg.shape[:2]
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[fg, 3] = 255
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise RuntimeError("Failed to encode mask PNG")
    return buf.tobytes()


def encode_object_mask_preview_jpeg(
    frame_bgr: np.ndarray,
    mask: np.ndarray | None,
    *,
    quality: int = 92,
) -> bytes:
    """Composite SAM2 mask onto a BGR frame and return JPEG bytes."""
    import cv2

    from _common.viz import OBJECT_MASK_BGR, _overlay_mask_bgr

    vis = frame_bgr
    if mask is not None and np.any(mask):
        vis = _overlay_mask_bgr(frame_bgr, mask, OBJECT_MASK_BGR, alpha=0.45)
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode preview JPEG")
    return buf.tobytes()


def read_frame_size(video_path: Path, frame_idx: int) -> tuple[int, int]:
    from _common.io import read_video_frame

    frame = read_video_frame(video_path, frame_idx)
    return frame.shape[:2]


def segment_object_on_video(
    predictor,
    *,
    video_path: Path,
    output_dir: Path,
    object_id: str,
    frame_idx: int,
    points: list[tuple[float, float]],
    labels: list[int],
    output_prob_thresh: float = 0.5,
) -> dict[str, Any]:
    import cv2
    import torch

    del output_prob_thresh
    sc = _load_legacy_sam_video_utils()
    mask_has_foreground = sc.mask_has_foreground
    video_frame_size = sc.video_frame_size

    video_path = video_path.resolve()
    masks_base = output_dir / "masks"
    masks_base.mkdir(parents=True, exist_ok=True)

    height, width = video_frame_size(video_path)
    pts_arr = np.array(points, dtype=np.float32)
    lbl_arr = np.array(labels, dtype=np.int32)
    num_frames = count_video_frames(video_path)

    state = predictor.init_state(video_path=str(video_path))
    device_type = predictor_device_type(predictor)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        predictor.add_new_points_or_box(
            state,
            frame_idx=frame_idx,
            obj_id=SAM2_OBJECT_ID,
            points=pts_arr,
            labels=lbl_arr,
            clear_old_points=True,
            normalize_coords=True,
        )
        outputs_per_frame: dict[int, dict[str, Any]] = {}
        for reverse in (False, True):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=frame_idx,
                reverse=reverse,
            ):
                outputs_per_frame[int(out_frame_idx)] = {
                    "obj_ids": out_obj_ids,
                    "mask_logits": out_mask_logits,
                }

    detected = 0
    for fidx in range(num_frames):
        output = outputs_per_frame.get(fidx)
        mask = None
        if output is not None:
            mask = _mask_for_object(output["obj_ids"], output["mask_logits"])
        frame_dir = masks_base / f"frame_{fidx:06d}_masks"
        frame_dir.mkdir(parents=True, exist_ok=True)
        if mask is not None and mask_has_foreground(mask):
            detected += 1
            arr = (mask.astype(np.uint8) * 255)
        else:
            arr = np.zeros((height, width), dtype=np.uint8)
        cv2.imwrite(str(frame_dir / object_mask_filename(object_id)), arr)

    if hasattr(predictor, "reset_state"):
        predictor.reset_state(state)
    return {
        "num_frames": num_frames,
        "detected_frames": detected,
        "prompt_frame_idx": frame_idx,
        "backend": "sam2",
        "object_id": object_id,
        "mask_filename": object_mask_filename(object_id),
    }


def run_object_masks(
    *,
    video_path: Path,
    step_dir: Path,
    video_id: str,
    gpu_id: int,
    visualize: bool,
    visualize_object_id: str | None = None,
    checkpoint: Path | None = None,
    model_cfg: str = DEFAULT_SAM2_MODEL_CFG,
) -> dict[str, Any]:
    if not has_label_prompt(step_dir):
        raise FileNotFoundError(
            f"Missing {LABEL_PROMPT_FILENAME}. Run label_object.py first for this video."
        )
    prompt = load_label_prompt(step_dir)
    num_frames = count_video_frames(video_path)
    height, width = read_frame_size(video_path, 0)
    validate_label_prompt_for_video(
        prompt,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    # A retry or changed multi-object prompt must never inherit masks from an
    # earlier partial/forced run.  The prompt and frame plan remain untouched.
    clear_generated_object_outputs(step_dir)

    predictor = build_object_predictor(gpu_id=gpu_id, checkpoint=checkpoint, model_cfg=model_cfg)
    object_stats: list[dict[str, Any]] = []
    try:
        for obj in prompt.objects:
            stats_i = segment_object_on_video(
                predictor,
                video_path=video_path,
                output_dir=step_dir / "video_segmentation",
                object_id=obj.object_id,
                frame_idx=obj.frame_idx,
                points=obj.points,
                labels=obj.labels,
            )
            object_stats.append(
                {
                    **stats_i,
                    "frame_idx": obj.frame_idx,
                    "points": obj.points,
                    "labels": obj.labels,
                    "num_points": len(obj.points),
                    "num_positive_points": int(sum(1 for value in obj.labels if int(value) > 0)),
                    "locked": bool(obj.locked),
                    "name": obj.name,
                }
            )
    finally:
        del predictor

    masks_dir = step_dir / "video_segmentation" / "masks"
    primary = prompt.primary
    primary_stats = object_stats[0] if object_stats else {}
    stats = {
        **primary_stats,
        "schema_version": LABEL_PROMPT_SCHEMA_VERSION,
        "num_objects": len(prompt.objects),
        "object_ids": prompt.object_ids(),
        "objects": object_stats,
        "object_mask_id": primary.object_id,
        "masks_dir": str(masks_dir),
        "sam2_checkpoint": str((checkpoint or DEFAULT_SAM2_CHECKPOINT).resolve()),
        "sam2_model_cfg": model_cfg,
    }

    if visualize:
        if visualize_object_id:
            vis_object = prompt.get_object(visualize_object_id)
            out_vis = encode_object_mask_vis_mp4(
                video_path=video_path,
                masks_dir=masks_dir,
                mask_filename=vis_object.mask_filename,
                out_path=vis_path(step_dir, f"{video_id}_{vis_object.object_id}"),
                prompt_frame_idx=vis_object.frame_idx,
                prompt_points=vis_object.points,
                prompt_labels=vis_object.labels,
            )
            stats["vis_object_id"] = vis_object.object_id
        else:
            out_vis = encode_object_masks_vis_mp4(
                video_path=video_path,
                masks_dir=masks_dir,
                objects=object_stats,
                out_path=vis_path(step_dir, f"{video_id}_objects"),
            )
            stats["vis_object_id"] = "all"
        stats["vis_video"] = str(out_vis)

    discard_sam2_object_nonessential(step_dir)
    return stats
