"""Shared helpers for HumanVideo2RobotData SAM3 hand-mask pipeline scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SAM3_ROOT = REPO_ROOT / "third_party" / "sam3"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "testing" / "sam3" / "hoi4d"

VIPE_SCRIPT_DIR = REPO_ROOT / "recon_pipeline" / "_legacy" / "vipe"


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


_vipe = _load_vipe_common()

DEFAULT_HOI4D_ROOT = _vipe.DEFAULT_HOI4D_ROOT
DEFAULT_RGB_ROOT = _vipe.DEFAULT_RGB_ROOT
cleanup_run_metadata = _vipe.cleanup_run_metadata
discover_hoi4d_video_jobs = _vipe.discover_hoi4d_video_jobs
parse_gpu_list = _vipe.parse_gpu_list
sequence_lock = _vipe.sequence_lock
video_jobs_from_list_file = _vipe.video_jobs_from_list_file
worker_log_path = _vipe.worker_log_path
write_status = _vipe.write_status

LEFT_HAND_PROMPT = "left hand"
RIGHT_HAND_PROMPT = "right hand"
LEFT_HAND_OBJ_ID = "left_hand_0"
RIGHT_HAND_OBJ_ID = "right_hand_0"
COMPLETION_FILENAME = "hand_masks_complete.json"
HAND_SPECS = (
    (LEFT_HAND_PROMPT, LEFT_HAND_OBJ_ID),
    (RIGHT_HAND_PROMPT, RIGHT_HAND_OBJ_ID),
)
HAND_MASK_COLORS: dict[str, tuple[int, int, int]] = {
    LEFT_HAND_OBJ_ID: (80, 220, 100),   # green
    RIGHT_HAND_OBJ_ID: (255, 120, 30),  # orange
}
VIS_FILENAME_SUFFIX = "_hand_masks_vis.mp4"


def setup_script_imports() -> Path:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    return SCRIPT_DIR


def sequence_output_root(output_dir: Path, sequence_name: str) -> Path:
    return output_dir.resolve() / sequence_name


def segmentation_output_dir(output_dir: Path, sequence_name: str) -> Path:
    return sequence_output_root(output_dir, sequence_name) / "video_segmentation"


def masks_root(output_dir: Path, sequence_name: str) -> Path:
    return segmentation_output_dir(output_dir, sequence_name) / "masks"


def completion_marker_path(output_dir: Path, sequence_name: str) -> Path:
    return sequence_output_root(output_dir, sequence_name) / COMPLETION_FILENAME


def combined_vis_video_path(output_dir: Path, sequence_name: str) -> Path:
    return sequence_output_root(output_dir, sequence_name) / f"{sequence_name}{VIS_FILENAME_SUFFIX}"


def is_sequence_complete(output_dir: Path, sequence_name: str) -> bool:
    marker = completion_marker_path(output_dir, sequence_name)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "complete":
        return False
    hands = payload.get("hands", {})
    if any(hand.get("detected") for hand in hands.values()):
        vis_path = payload.get("vis_video")
        if vis_path is not None and not Path(vis_path).is_file():
            return False
        return True
    return bool(payload.get("hands_detected"))


def write_completion_marker(
    output_dir: Path,
    sequence_name: str,
    *,
    video_path: Path,
    num_frames: int,
    frame_idx: int,
    version: str,
    hands: dict[str, Any],
    version_fallback: dict[str, Any] | None = None,
    vis_video: Path | None = None,
) -> Path:
    marker = completion_marker_path(output_dir, sequence_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "sequence": sequence_name,
        "video": str(video_path.resolve()),
        "num_frames": num_frames,
        "prompt_frame_idx": frame_idx,
        "sam3_version": version,
        "sam3_requested_version": (
            version_fallback.get("requested_version") if version_fallback else version
        ),
        "sam3_version_fallback": version_fallback,
        "hands": hands,
        "hands_detected": [
            result["text_prompt"]
            for result in hands.values()
            if result.get("detected")
        ],
        "left_hand_prompt": hands.get(LEFT_HAND_OBJ_ID, {}).get("text_prompt"),
        "right_hand_prompt": hands.get(RIGHT_HAND_OBJ_ID, {}).get("text_prompt"),
        "masks_dir": str(masks_root(output_dir, sequence_name)),
        "vis_video": str(vis_video.resolve()) if vis_video is not None else None,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def get_highest_score_obj(outputs: dict[str, Any]) -> tuple[Any, Any, Any]:
    obj_ids = outputs["out_obj_ids"]
    probs = outputs["out_probs"]
    masks = outputs["out_binary_masks"]
    if len(obj_ids) == 0:
        return None, None, None
    best_idx = int(np.argmax(probs))
    return obj_ids[best_idx], probs[best_idx], masks[best_idx]


def mask_has_foreground(mask: np.ndarray | None) -> bool:
    return mask is not None and bool(np.any(mask))


def hand_prompt_sets_from_config(config: "Sam3RunConfig") -> tuple[tuple[tuple[str, ...], str], ...]:
    return (
        (config.left_hand_prompts, LEFT_HAND_OBJ_ID),
        (config.right_hand_prompts, RIGHT_HAND_OBJ_ID),
    )


def video_frame_size(video_path: Path) -> tuple[int, int]:
    """Return (height, width) for an MP4 or JPEG frame directory."""
    import cv2

    video_path = video_path.resolve()
    if video_path.suffix.lower() == ".mp4":
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Failed to read frame size from video: {video_path}")
        return height, width

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    frame_paths = [
        video_path / name
        for name in os.listdir(video_path)
        if (video_path / name).suffix.lower() in image_exts
    ]
    if not frame_paths:
        raise RuntimeError(f"No image frames found under: {video_path}")
    try:
        frame_paths.sort(key=lambda path: int(path.stem))
    except ValueError:
        frame_paths.sort()
    frame = cv2.imread(str(frame_paths[0]))
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")
    height, width = frame.shape[:2]
    return height, width


def video_frame_rate(video_path: Path, default: float = 24.0) -> float:
    import cv2

    video_path = video_path.resolve()
    if video_path.suffix.lower() != ".mp4":
        return default
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return default
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1.0:
        return default
    return fps


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    import cv2

    video_path = video_path.resolve()
    if video_path.suffix.lower() == ".mp4":
        cap = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
    frame_paths = [
        video_path / name
        for name in os.listdir(video_path)
        if (video_path / name).suffix.lower() in image_exts
    ]
    try:
        frame_paths.sort(key=lambda path: int(path.stem))
    except ValueError:
        frame_paths.sort()
    return [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in frame_paths]


def overlay_mask_on_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (30, 144, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    import cv2

    overlay = frame.copy()
    overlay[mask] = (
        (1 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.uint8)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def load_mask_png(mask_path: Path) -> np.ndarray:
    import cv2

    if not mask_path.is_file():
        return np.zeros((0, 0), dtype=bool)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((0, 0), dtype=bool)
    return mask > 0


def write_combined_hand_vis_video(
    video_path: Path,
    masks_dir: Path,
    output_video: Path,
    *,
    fps: float = 24.0,
    num_frames: int | None = None,
) -> Path:
    """Overlay left/right hand masks on the source video and encode an MP4."""
    import cv2

    video_path = video_path.resolve()
    masks_dir = masks_dir.resolve()
    output_video = output_video.resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)

    overlay_dir = masks_dir.parent / "overlays_combined"
    if overlay_dir.exists():
        shutil.rmtree(overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    frames = load_video_frames(video_path)
    if num_frames is not None:
        frames = frames[:num_frames]

    for frame_index, frame in enumerate(frames):
        overlay = frame.copy()
        frame_mask_dir = masks_dir / f"frame_{frame_index:06d}_masks"
        for obj_id, color in HAND_MASK_COLORS.items():
            mask = load_mask_png(frame_mask_dir / f"{obj_id}.png")
            if mask.size and np.any(mask):
                overlay = overlay_mask_on_frame(overlay, mask, color=color)

        cv2.imwrite(
            str(overlay_dir / f"frame_{frame_index:06d}.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(overlay_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ],
        check=True,
    )
    shutil.rmtree(overlay_dir)
    return output_video


def _flash_attn_available() -> bool:
    try:
        from flash_attn_interface import flash_attn_func  # noqa: F401

        return True
    except Exception:
        return False


def _current_cuda_device_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        idx = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(idx)
        return {
            "available": True,
            "index": int(idx),
            "name": torch.cuda.get_device_name(idx),
            "capability": [int(major), int(minor)],
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _is_blackwell_cuda_device(device_info: dict[str, Any]) -> bool:
    if not device_info.get("available"):
        return False
    name = str(device_info.get("name") or "").lower()
    capability = device_info.get("capability") or [0, 0]
    try:
        major = int(capability[0])
    except (TypeError, ValueError, IndexError):
        major = 0

    # Blackwell consumer/server GPUs are SM 12.x in current CUDA releases.
    # Keep name checks as a guard for environments reporting incomplete caps.
    blackwell_name_tokens = ("blackwell", "rtx 50", "rtx 5090", "rtx 5080", "rtx 5070")
    return major >= 12 or any(token in name for token in blackwell_name_tokens)


def resolve_sam3_version_for_device(
    requested_version: str,
    *,
    checkpoint: str | None = None,
    auto_blackwell_fallback: bool = True,
) -> tuple[str, dict[str, Any]]:
    device_info = _current_cuda_device_info()
    fallback = {
        "requested_version": requested_version,
        "effective_version": requested_version,
        "blackwell_detected": _is_blackwell_cuda_device(device_info),
        "auto_blackwell_fallback": bool(auto_blackwell_fallback),
        "device": device_info,
        "reason": None,
    }
    if (
        auto_blackwell_fallback
        and checkpoint is None
        and requested_version == "sam3.1"
        and fallback["blackwell_detected"]
    ):
        fallback["effective_version"] = "sam3"
        fallback["reason"] = "blackwell_sam3_1_precision_issue"
    return str(fallback["effective_version"]), fallback


def build_predictor(
    *,
    version: str = "sam3.1",
    checkpoint: str | None = None,
    compile_model: bool = False,
    use_fa3: bool | None = None,
    auto_blackwell_fallback: bool = True,
):
    from sam3.model_builder import build_sam3_predictor

    if use_fa3 is None:
        use_fa3 = _flash_attn_available()

    version, fallback = resolve_sam3_version_for_device(
        version,
        checkpoint=checkpoint,
        auto_blackwell_fallback=auto_blackwell_fallback,
    )
    if fallback.get("reason"):
        device = fallback.get("device") or {}
        print(
            "[sam3_hands] Blackwell GPU detected "
            f"({device.get('name', 'unknown')}); using sam3 instead of sam3.1",
            flush=True,
        )

    kwargs: dict[str, Any] = {
        "version": version,
        "compile": compile_model,
        "use_fa3": use_fa3,
    }
    if checkpoint is not None:
        kwargs["checkpoint_path"] = checkpoint
    return build_sam3_predictor(**kwargs)


@dataclass
class Sam3RunConfig:
    output_dir: Path
    frame_idx: int = 0
    version: str = "sam3.1"
    checkpoint: Path | None = None
    compile_model: bool = False
    use_fa3: bool | None = None
    auto_blackwell_fallback: bool = True
    save_overlays: bool = False
    save_videos: bool = False
    overlay_fps: int = 24
    visualize: bool = True
    left_hand_prompts: tuple[str, ...] = (LEFT_HAND_PROMPT,)
    right_hand_prompts: tuple[str, ...] = (RIGHT_HAND_PROMPT,)
    output_prob_thresh: float = 0.5
    cleanup_intermediates: bool = True


def segment_hand_on_video(
    predictor,
    *,
    video_path: Path,
    text_prompt: str,
    obj_id: str,
    output_dir: Path,
    frame_idx: int,
    save_overlays: bool,
    save_videos: bool,
    overlay_fps: int,
    output_prob_thresh: float,
) -> dict[str, Any]:
    import cv2

    video_path = video_path.resolve()
    masks_base_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays" / obj_id
    masks_base_dir.mkdir(parents=True, exist_ok=True)
    if save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    response = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(video_path),
        }
    )
    session_id = response["session_id"]
    sam3_obj_id = 1

    try:
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_idx,
                "text": text_prompt,
                "obj_id": sam3_obj_id,
                "output_prob_thresh": output_prob_thresh,
            }
        )

        outputs_per_frame: dict[int, dict[str, Any]] = {}
        for resp in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
            }
        ):
            outputs_per_frame[int(resp["frame_index"])] = resp["outputs"]

        height, width = video_frame_size(video_path)
        video_frames = load_video_frames(video_path) if (save_overlays or save_videos) else None

        sorted_frames = sorted(outputs_per_frame.keys())
        prompt_name = text_prompt.replace(" ", "_")
        out_video_path = output_dir / f"tracked_{obj_id}_{prompt_name}.mp4"
        detected_frames = 0

        for frame_index in sorted_frames:
            outputs = outputs_per_frame[frame_index]
            _, score, best_mask = get_highest_score_obj(outputs)
            if mask_has_foreground(best_mask):
                detected_frames += 1

            frame_mask_dir = masks_base_dir / f"frame_{frame_index:06d}_masks"
            frame_mask_dir.mkdir(parents=True, exist_ok=True)
            if best_mask is not None:
                mask_uint8 = (best_mask.astype(np.uint8) * 255)
            else:
                mask_uint8 = np.zeros((height, width), dtype=np.uint8)
            cv2.imwrite(str(frame_mask_dir / f"{obj_id}.png"), mask_uint8)

            if save_overlays and video_frames is not None and frame_index < len(video_frames):
                frame = video_frames[frame_index]
                if best_mask is not None:
                    overlay = overlay_mask_on_frame(frame, best_mask)
                else:
                    overlay = frame
                cv2.imwrite(
                    str(overlay_dir / f"frame_{frame_index:06d}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )

        if save_videos and save_overlays and detected_frames > 0:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(overlay_fps),
                    "-i",
                    str(overlay_dir / "frame_%06d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(out_video_path),
                ],
                check=True,
            )

        prompt_out = response["outputs"]
        _, prompt_score, prompt_mask = get_highest_score_obj(prompt_out)
        detected = detected_frames > 0 or mask_has_foreground(prompt_mask)
        return {
            "obj_id": obj_id,
            "text_prompt": text_prompt,
            "num_frames": len(sorted_frames),
            "detected": detected,
            "detected_frames": detected_frames,
            "output_prob_thresh": output_prob_thresh,
            "prompt_score": float(prompt_score) if prompt_score is not None else None,
            "tracked_video": str(out_video_path) if save_videos and save_overlays and detected else None,
        }
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def segment_hand_with_fallbacks(
    predictor,
    *,
    video_path: Path,
    prompts: tuple[str, ...],
    obj_id: str,
    output_dir: Path,
    frame_idx: int,
    save_overlays: bool,
    save_videos: bool,
    overlay_fps: int,
    output_prob_thresh: float,
) -> dict[str, Any]:
    """Try prompts in order until one produces a hand mask."""
    deduped_prompts = tuple(dict.fromkeys(prompt.strip() for prompt in prompts if prompt.strip()))
    if not deduped_prompts:
        raise ValueError(f"No prompts provided for {obj_id}")

    last_result: dict[str, Any] | None = None
    for prompt in deduped_prompts:
        result = segment_hand_on_video(
            predictor,
            video_path=video_path,
            text_prompt=prompt,
            obj_id=obj_id,
            output_dir=output_dir,
            frame_idx=frame_idx,
            save_overlays=save_overlays,
            save_videos=save_videos,
            overlay_fps=overlay_fps,
            output_prob_thresh=output_prob_thresh,
        )
        result["prompt_used"] = prompt
        result["prompts_tried"] = list(deduped_prompts)
        last_result = result
        if result.get("detected"):
            return result
    assert last_result is not None
    return last_result


def run_hand_masks_sequence(
    video_path: Path,
    sequence_name: str,
    config: Sam3RunConfig,
    *,
    gpu_id: int | None = None,
    predictor=None,
    own_predictor: bool = False,
) -> dict[str, Any]:
    video_path = video_path.resolve()
    if not video_path.is_file() and not video_path.is_dir():
        raise FileNotFoundError(f"Video not found: {video_path}")

    seg_dir = segmentation_output_dir(config.output_dir, sequence_name)
    seg_dir.mkdir(parents=True, exist_ok=True)

    requested_checkpoint = str(config.checkpoint) if config.checkpoint is not None else None
    if predictor is None:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        effective_version, version_fallback = resolve_sam3_version_for_device(
            config.version,
            checkpoint=requested_checkpoint,
            auto_blackwell_fallback=config.auto_blackwell_fallback,
        )
        if version_fallback.get("reason"):
            device = version_fallback.get("device") or {}
            print(
                "[sam3_hands] Blackwell GPU detected "
                f"({device.get('name', 'unknown')}); using sam3 instead of sam3.1",
                flush=True,
            )
        predictor = build_predictor(
            version=effective_version,
            checkpoint=requested_checkpoint,
            compile_model=config.compile_model,
            use_fa3=config.use_fa3,
            auto_blackwell_fallback=False,
        )
        own_predictor = True
    else:
        effective_version = config.version
        version_fallback = {
            "requested_version": config.version,
            "effective_version": config.version,
            "blackwell_detected": False,
            "auto_blackwell_fallback": False,
            "device": None,
            "reason": None,
            "note": "prebuilt_predictor_supplied",
        }

    hand_results: dict[str, Any] = {}
    num_frames = 0
    prompts_tried: list[str] = []
    try:
        for prompt_set, obj_id in hand_prompt_sets_from_config(config):
            prompts_tried.extend(prompt_set)
            try:
                result = segment_hand_with_fallbacks(
                    predictor,
                    video_path=video_path,
                    prompts=prompt_set,
                    obj_id=obj_id,
                    output_dir=seg_dir,
                    frame_idx=config.frame_idx,
                    save_overlays=config.save_overlays,
                    save_videos=config.save_videos,
                    overlay_fps=config.overlay_fps,
                    output_prob_thresh=config.output_prob_thresh,
                )
            except Exception as exc:
                height, width = video_frame_size(video_path)
                result = {
                    "obj_id": obj_id,
                    "text_prompt": prompt_set[0],
                    "prompt_used": prompt_set[0],
                    "prompts_tried": list(prompt_set),
                    "num_frames": 0,
                    "detected": False,
                    "detected_frames": 0,
                    "prompt_score": None,
                    "error": str(exc),
                    "frame_size": [height, width],
                }
            hand_results[obj_id] = result
            num_frames = max(num_frames, int(result.get("num_frames") or 0))
    finally:
        if own_predictor:
            predictor.shutdown()

    if not any(result.get("detected") for result in hand_results.values()):
        raise RuntimeError(
            "No hand detected in entire video "
            f"(tried: {', '.join(dict.fromkeys(prompts_tried))})"
        )

    vis_video: Path | None = None
    if config.visualize:
        vis_video = write_combined_hand_vis_video(
            video_path,
            masks_root(config.output_dir, sequence_name),
            combined_vis_video_path(config.output_dir, sequence_name),
            fps=video_frame_rate(video_path, default=float(config.overlay_fps)),
            num_frames=num_frames or None,
        )

    marker = write_completion_marker(
        config.output_dir,
        sequence_name,
        video_path=video_path,
        num_frames=num_frames,
        frame_idx=config.frame_idx,
        version=effective_version,
        hands=hand_results,
        version_fallback=version_fallback,
        vis_video=vis_video,
    )
    return {
        "sequence": sequence_name,
        "video": str(video_path),
        "num_frames": num_frames,
        "completion_marker": str(marker),
        "masks_dir": str(masks_root(config.output_dir, sequence_name)),
        "vis_video": str(vis_video) if vis_video is not None else None,
        "sam3_version": effective_version,
        "sam3_requested_version": config.version,
        "sam3_version_fallback": version_fallback,
        "hands": hand_results,
        "hands_detected": [
            result["text_prompt"]
            for result in hand_results.values()
            if result.get("detected")
        ],
    }


def cleanup_sam3_run_artifacts(
    output_dir: Path,
    *,
    keep_logs: bool = False,
    keep_run_metadata: bool = False,
) -> None:
    if not keep_run_metadata:
        cleanup_run_metadata(output_dir)
    if not keep_logs:
        logs_dir = output_dir / ".logs"
        if logs_dir.is_dir():
            shutil.rmtree(logs_dir)
