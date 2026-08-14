#!/usr/bin/env python3
"""Hybrid batch runner with CPU labeling and memory-aware GPU worker slots."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

RECON_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RECON_ROOT.parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.dataset import VideoJob, discover_videos, resolve_video_job, write_video_manifest  # noqa: E402
from _common.paths import INTERIM_ROOT, completion_marker, final_video_dir, interim_step_dir, is_step_complete, resolve_repo_path  # noqa: E402
from sam2_object.sam2_object_common import label_prompt_path  # noqa: E402

AUTO_STEPS = (
    "vipe",
    "sam3_hands",
    "sam2_object",
    "hawor",
    # ★ VLM 门必须在 sam3d **之前**: 它的分件判定决定下一步是"重建网格"还是"取 CAD 资产"。
    #   单刚体网格表达不了"盖相对瓶身转", 放到最后就来不及了。
    "vlm_retrieval",
    "retrieval",
    "sam3d",
    "sam3d_scale",
    "fp_pose",
    "fuse",
    "confidence",
    "contact",
)

STEP_SCRIPTS = {
    "vipe": RECON_ROOT / "vipe" / "run_sequence.py",
    "sam3_hands": RECON_ROOT / "sam3_hands" / "run_sequence.py",
    "sam2_object": RECON_ROOT / "sam2_object" / "run_sequence.py",
    "hawor": RECON_ROOT / "hawor" / "run_sequence.py",
    "vlm_retrieval": RECON_ROOT / "vlm_retrieval" / "run_sequence.py",     # 材质+分件判定, CPU+远端VLM
    "retrieval": RECON_ROOT / "retrieval" / "run_sequence.py",   # 命中则替 sam3d/sam3d_scale 写完成标记
    "sam3d": RECON_ROOT / "sam3d" / "run_sequence.py",
    "sam3d_scale": RECON_ROOT / "sam3d_scale" / "run_sequence.py",
    "fp_pose": RECON_ROOT / "fp_pose" / "run_sequence.py",
    "fuse": RECON_ROOT / "fuse" / "run_sequence.py",
    "confidence": RECON_ROOT / "confidence" / "run_sequence.py",
    "contact": RECON_ROOT / "contact_v2" / "run_sequence.py",
    "contact_align": RECON_ROOT / "contact" / "run_sequence.py",  # 旧: 默认不启用
}

STEP_ENVS = {
    "vipe": "cu128",
    # ★ 两步都用 codetr, **不再用 HV2RD**。
    #   HV2RD 是被收编删除的那个仓留下的 env 名(见 HV2RD_MIGRATION.md: 收编删的是仓,
    #   env 名没人动), 指向一个已经不存在的项目, 留着只会让人以为还有那个依赖。
    #   codetr 实测是 HV2RD 的**超集**: 同样的 torch 2.11.0+cu128 / sam2 / decord 0.6.0,
    #   另外多了 HOI-DETR 那套 mmcv+transformers。缺的只有 sam3(+ftfy==6.1.1), 已补装
    #   (sam3 是仓内 third_party/sam3 的 editable, --no-deps 装, 未动 codetr 原有依赖)。
    #
    #   ⚠ 上游改这行时给的理由是"HV2RD 在 A6000 上缺 decord, init_state 会挂" —— 该理由
    #     **实测不成立**: 两个 env 都有 decord 0.6.0、指向同一个 editable sam2, 在 A6000 上
    #     跑 init_state(video_path=17.mp4) 都成功(4.9s / 4.8s)。改动本身可行, 但别把那条
    #     错误诊断当依据传下去。真正的理由是上面那条: 不再保留已删项目的 env 名。
    "sam3_hands": "codetr",
    "sam2_object": "codetr",
    "hawor": "hawor",
    "vlm_retrieval": "hawor",
    "retrieval": "hawor",
    "sam3d": "biv2ap",
    "sam3d_scale": "biv2ap",
    "fp_pose": "biv2ap",
    "fuse": "hawor",
    "confidence": "hawor",
    "contact": "hawor",
    "contact_align": "hawor",
}

# 只有这些步骤会读 sam2_object/label_prompt.json; 其余步骤不该被"缺标注"拦住。
OBJECT_LABEL_STEPS = {"label", "sam2_object"}


def _steps_need_object_label(steps) -> bool:
    return bool(OBJECT_LABEL_STEPS & set(steps))


GPU_STEPS = {"vipe", "sam3_hands", "sam2_object", "hawor", "sam3d", "sam3d_scale", "fp_pose", "confidence"}
CUDA_VISIBLE_DEVICE_STEPS = {"fp_pose"}
DEFAULT_GPU_MEM_BUDGET_MB = 43000
DEFAULT_STEP_GPU_MEM_MB = {
    "vipe": 33000,
    "sam3_hands": 32000,
    "sam2_object": 9000,
    "hawor": 8000,
    "sam3d": 22000,
    "sam3d_scale": 10000,
    "fp_pose": 6000,
    "confidence": 7000,   # CoTracker 实测 ~6.5GB; audit/rts 是 CPU
}
STEP_EXTRA_ARGS: dict[str, list[str]] = {}   # --step-arg 透传表 (step -> [flags])

DEFAULT_STEP_CPU_THREADS = {
    "vlm_retrieval": "2",     # 只是 HTTP 等远端 VLM
    "retrieval": "2",
    "fp_pose": "4",
    "confidence": "4",
    "contact": "4",
}
FINAL_SCHEMA_VERSION = "recon_world_v4"
HAWOR_CAMERA_TIMELINE = "vipe_time_aligned_v1"
FINAL_NPZ = "world_fused.npz"
FINAL_SUMMARY = "world_summary.json"
FINAL_MESH = "object_mesh_scaled_final.obj"
FINAL_COMPLETION = "reconstruction_complete.json"
FINAL_MASKS_DIR = "masks"
FINAL_MASKS_MANIFEST = "masks_manifest.json"
WORLD_XY_ALIGNMENT_MODE = "middle_camera_forward_projected_xy"
WORLD_XY_ALIGNMENT_TOL = 1e-5
LABEL_CACHE_ROOT = REPO_ROOT / "data" / "object_labels"
LEGACY_LABEL_CACHE_ROOT = REPO_ROOT / "data" / "object_label_backup"
REQUIRED_FINAL_KEYS = {
    "schema_version",
    "dataset",
    "video_id",
    "num_frames",
    "K",
    "c2w",
    "c2w_vipe_world",
    "object_ob_in_cam",
    "object_frame_indices",
    "object_ob_in_world",
    "mesh_filename",
    "T_gravity_z_up_from_vipe_world",
    "gravity_direction_vipe_world",
    "up_direction_vipe_world",
    "world_xy_alignment_mode",
    "world_xy_alignment_frame_index",
    "world_x_direction_vipe_world",
    "world_y_direction_vipe_world",
    "world_z_direction_vipe_world",
}
EXPECTED_HAND_SOURCE_FRAME = "hawor_world_vipe_camera"



def _validate_final_masks(final_dir: Path, *, num_frames: int) -> dict:
    manifest_path = final_dir / FINAL_MASKS_DIR / FINAL_MASKS_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing final mask manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    object_masks = payload.get("object_masks") or {}
    if not object_masks.get("present"):
        raise RuntimeError("Final object masks are missing from mask manifest")
    object_ids = list(object_masks.get("object_ids") or [])
    expected_object_pngs = num_frames * len(object_ids)
    object_pngs = int(object_masks.get("num_png_files") or 0)
    if expected_object_pngs > 0 and object_pngs < expected_object_pngs:
        raise RuntimeError(f"Final object masks incomplete: {object_pngs} < {expected_object_pngs}")
    object_dir = final_dir / str(object_masks.get("path", ""))
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Missing final object mask directory: {object_dir}")

    hand_masks = payload.get("hand_masks") or {}
    hand_pngs = int(hand_masks.get("num_png_files") or 0)
    if hand_masks.get("present"):
        hand_dir = final_dir / str(hand_masks.get("path", ""))
        if not hand_dir.is_dir():
            raise FileNotFoundError(f"Missing final hand mask directory: {hand_dir}")
        if hand_pngs < num_frames:
            raise RuntimeError(f"Final hand masks incomplete: {hand_pngs} < {num_frames}")
    return {
        "masks_included": True,
        "mask_manifest": str(manifest_path.relative_to(final_dir)),
        "object_mask_pngs": object_pngs,
        "hand_mask_pngs": hand_pngs,
    }


def _validate_object_arrays(data: np.lib.npyio.NpzFile, *, final_dir: Path) -> dict:
    if "num_objects" not in data.files:
        return {"num_objects": 1}
    required = {
        "object_ids",
        "object_mesh_filenames",
        "object_frame_indices_all",
        "object_valid_all",
        "object_ob_in_cam_all",
        "object_ob_in_world_all",
        "object_ob_in_vipe_world_all",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise RuntimeError(f"Final multi-object NPZ missing required keys: {missing}")
    num_objects = int(data["num_objects"])
    object_ids = np.asarray(data["object_ids"])
    mesh_filenames = np.asarray(data["object_mesh_filenames"])
    frame_indices_all = np.asarray(data["object_frame_indices_all"])
    valid_all = np.asarray(data["object_valid_all"]).astype(bool)
    ob_cam_all = np.asarray(data["object_ob_in_cam_all"])
    ob_world_all = np.asarray(data["object_ob_in_world_all"])
    ob_vipe_all = np.asarray(data["object_ob_in_vipe_world_all"])
    num_frames = int(data["num_frames"]) if "num_frames" in data.files else None
    if num_objects < 1:
        raise RuntimeError("num_objects must be >= 1")
    if object_ids.shape != (num_objects,):
        raise RuntimeError(f"Invalid object_ids shape {object_ids.shape}; expected {(num_objects,)}")
    if mesh_filenames.shape != (num_objects,):
        raise RuntimeError(f"Invalid object_mesh_filenames shape {mesh_filenames.shape}; expected {(num_objects,)}")
    if frame_indices_all.shape != valid_all.shape:
        raise RuntimeError("object_frame_indices_all and object_valid_all shapes do not match")
    if frame_indices_all.shape[0] != num_objects:
        raise RuntimeError("object_frame_indices_all first dimension does not match num_objects")
    expected_pose_shape = frame_indices_all.shape + (4, 4)
    if ob_cam_all.shape != expected_pose_shape:
        raise RuntimeError(f"Invalid object_ob_in_cam_all shape {ob_cam_all.shape}; expected {expected_pose_shape}")
    if ob_world_all.shape != expected_pose_shape:
        raise RuntimeError(f"Invalid object_ob_in_world_all shape {ob_world_all.shape}; expected {expected_pose_shape}")
    if ob_vipe_all.shape != expected_pose_shape:
        raise RuntimeError(f"Invalid object_ob_in_vipe_world_all shape {ob_vipe_all.shape}; expected {expected_pose_shape}")
    for idx, mesh_filename in enumerate(mesh_filenames):
        mesh_path = final_dir / str(mesh_filename)
        if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing or empty final mesh for object {object_ids[idx]}: {mesh_path}")
    if "object_visible_by_frame" in data.files:
        visible_shape = np.asarray(data["object_visible_by_frame"]).shape
        expected = (num_objects, num_frames)
        if visible_shape != expected:
            raise RuntimeError(f"Invalid object_visible_by_frame shape {visible_shape}; expected {expected}")
    if "object_pose_valid_by_frame" in data.files:
        pose_valid_shape = np.asarray(data["object_pose_valid_by_frame"]).shape
        expected = (num_objects, num_frames)
        if pose_valid_shape != expected:
            raise RuntimeError(f"Invalid object_pose_valid_by_frame shape {pose_valid_shape}; expected {expected}")
    return {"num_objects": num_objects}


def _sample_hand_index(frame_idx: int, num_video_frames: int, num_hand_frames: int) -> int:
    if num_hand_frames <= 1 or num_video_frames <= 1:
        return 0
    t = frame_idx * (num_hand_frames - 1) / (num_video_frames - 1)
    return int(np.clip(round(t), 0, num_hand_frames - 1))


def _validate_hand_roots_in_front(data: np.lib.npyio.NpzFile, *, num_frames: int) -> dict:
    if not {"hand_trans", "hand_valid"}.issubset(data.files):
        return {"hand_root_front_samples": 0, "hand_valid_samples": 0}

    if "hand_source_coordinate_frame" in data.files:
        hand_source = str(data["hand_source_coordinate_frame"].item())
        if hand_source != EXPECTED_HAND_SOURCE_FRAME:
            raise RuntimeError(
                f"Unsupported hand_source_coordinate_frame {hand_source}; expected {EXPECTED_HAND_SOURCE_FRAME}. "
                "Re-run fuse or repair the final hand globals."
            )

    hand_trans = np.asarray(data["hand_trans"], dtype=np.float64)
    hand_valid = np.asarray(data["hand_valid"]).astype(bool)
    c2w = np.asarray(data["c2w"], dtype=np.float64)
    if hand_trans.shape[:2] != hand_valid.shape:
        raise RuntimeError("hand_trans and hand_valid frame dimensions do not match")
    if hand_trans.shape[-1] != 3:
        raise RuntimeError(f"Invalid hand_trans shape {hand_trans.shape}; expected (..., 3)")

    valid_samples = 0
    front_samples = 0
    sample_frames = np.unique(np.linspace(0, num_frames - 1, min(32, num_frames), dtype=np.int32))
    for frame_idx in sample_frames:
        hand_idx = _sample_hand_index(int(frame_idx), num_frames, hand_valid.shape[1])
        w2c = np.linalg.inv(c2w[int(frame_idx)])
        for hand_idx_side in range(hand_valid.shape[0]):
            if not hand_valid[hand_idx_side, hand_idx]:
                continue
            valid_samples += 1
            point_world = np.concatenate([hand_trans[hand_idx_side, hand_idx], [1.0]])
            point_cam = w2c @ point_world
            if np.isfinite(point_cam).all() and point_cam[2] > 1e-4:
                front_samples += 1

    if hand_valid.any() and valid_samples > 0 and front_samples == 0:
        raise RuntimeError(
            "All sampled valid MANO hand roots are behind the final camera. "
            "Check HaWoR-to-ViPE hand coordinate conversion."
        )
    return {"hand_root_front_samples": int(front_samples), "hand_valid_samples": int(valid_samples)}


def _validate_world_xy_alignment(
    data: np.lib.npyio.NpzFile,
    *,
    r_zup_from_vipe: np.ndarray,
    num_frames: int,
) -> dict:
    mode = str(data["world_xy_alignment_mode"].item())
    if mode != WORLD_XY_ALIGNMENT_MODE:
        raise RuntimeError(f"Unsupported world_xy_alignment_mode {mode}; expected {WORLD_XY_ALIGNMENT_MODE}")
    frame_idx = int(np.asarray(data["world_xy_alignment_frame_index"]).item())
    if frame_idx < 0 or frame_idx >= num_frames:
        raise RuntimeError(f"Invalid world_xy_alignment_frame_index {frame_idx}; expected [0, {num_frames})")

    c2w_vipe = np.asarray(data["c2w_vipe_world"], dtype=np.float64)
    if c2w_vipe.shape != (num_frames, 4, 4):
        raise RuntimeError(f"Invalid c2w_vipe_world shape {c2w_vipe.shape}; expected {(num_frames, 4, 4)}")

    axis_targets = {
        "world_x_direction_vipe_world": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "world_y_direction_vipe_world": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "world_z_direction_vipe_world": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    for key, target in axis_targets.items():
        axis_vipe = np.asarray(data[key], dtype=np.float64)
        if axis_vipe.shape != (3,):
            raise RuntimeError(f"Invalid {key} shape {axis_vipe.shape}; expected (3,)")
        if not np.allclose(r_zup_from_vipe @ axis_vipe, target, atol=WORLD_XY_ALIGNMENT_TOL):
            raise RuntimeError(f"{key} is inconsistent with T_gravity_z_up_from_vipe_world")

    camera_forward_final = r_zup_from_vipe @ c2w_vipe[frame_idx, :3, 2]
    camera_forward_xy = camera_forward_final.copy()
    camera_forward_xy[2] = 0.0
    horizontal_norm = float(np.linalg.norm(camera_forward_xy))
    if horizontal_norm <= 1e-6:
        raise RuntimeError("Middle-frame camera focal axis is too close to gravity to validate world +X")
    camera_forward_xy /= horizontal_norm
    if not np.allclose(camera_forward_xy, np.array([1.0, 0.0, 0.0]), atol=WORLD_XY_ALIGNMENT_TOL):
        raise RuntimeError("World +X must match the middle-frame camera focal axis projected into the world XY plane")

    return {
        "world_xy_alignment_mode": mode,
        "world_xy_alignment_frame_index": frame_idx,
    }


@dataclass
class BatchState:
    queued: set[str] = field(default_factory=set)
    running: dict[str, str] = field(default_factory=dict)
    done: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)


class GpuReservation:
    """Thread-safe approximate GPU memory reservation for same-GPU workers."""

    def __init__(self, gpu_ids: list[int], *, budget_mb: int, step_mem_mb: dict[str, int]):
        self.budget_mb = int(budget_mb)
        self.step_mem_mb = dict(step_mem_mb)
        self.used_by_gpu = {gpu_id: 0 for gpu_id in gpu_ids}
        self.condition = threading.Condition()

    def estimate(self, step: str) -> int:
        return int(self.step_mem_mb.get(step, self.budget_mb if step in GPU_STEPS else 0))

    def acquire_any(self, gpu_ids: list[int], step: str, prefix: str) -> tuple[int, int]:
        if step not in GPU_STEPS:
            return gpu_ids[0], 0
        need = self.estimate(step)
        wait_reported = False
        with self.condition:
            while True:
                runnable = [
                    gpu_id
                    for gpu_id in gpu_ids
                    if self.used_by_gpu[gpu_id] + need <= self.budget_mb
                ]
                if runnable:
                    gpu_id = min(runnable, key=lambda x: (self.used_by_gpu[x], x))
                    self.used_by_gpu[gpu_id] += need
                    print(
                        f"{prefix} {step}: reserve GPU {gpu_id} {need} MB "
                        f"({self.used_by_gpu[gpu_id]}/{self.budget_mb} MB)",
                        flush=True,
                    )
                    return gpu_id, need
                if not wait_reported:
                    usage = ", ".join(f"{gpu}:{self.used_by_gpu[gpu]}" for gpu in gpu_ids)
                    print(
                        f"{prefix} {step}: wait for any GPU memory "
                        f"(need {need} MB; used {usage}; budget {self.budget_mb} MB)",
                        flush=True,
                    )
                    wait_reported = True
                self.condition.wait(timeout=10.0)

    def release(self, gpu_id: int, amount_mb: int) -> None:
        if amount_mb <= 0:
            return
        with self.condition:
            self.used_by_gpu[gpu_id] = max(0, self.used_by_gpu[gpu_id] - int(amount_mb))
            self.condition.notify_all()


def _parse_gpu_ids(value: str) -> list[int]:
    gpu_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not gpu_ids:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError(f"duplicate GPU ids are not allowed: {value}")
    return gpu_ids


def _parse_step_mem_overrides(value: str | None) -> dict[str, int]:
    out = dict(DEFAULT_STEP_GPU_MEM_MB)
    if not value:
        return out
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid step memory override {item!r}; expected step=MB")
        step, mem = item.split("=", 1)
        step = step.strip()
        if step not in GPU_STEPS:
            raise argparse.ArgumentTypeError(f"Unknown GPU step in memory override: {step}")
        try:
            mem_mb = int(mem)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid memory MB for {step}: {mem}") from exc
        if mem_mb <= 0:
            raise argparse.ArgumentTypeError(f"Memory override for {step} must be positive")
        out[step] = mem_mb
    return out


def _worker_slots(gpu_ids: list[int], workers_per_gpu: int) -> list[tuple[int, int]]:
    slots = []
    worker_idx = 0
    for _slot_idx in range(workers_per_gpu):
        for gpu_id in gpu_ids:
            slots.append((gpu_id, worker_idx))
            worker_idx += 1
    return slots


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _batch_dir(dataset: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = INTERIM_ROOT / dataset / "batch_queue" / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_label_list(jobs: list[VideoJob], path: Path) -> None:
    path.write_text("\n".join(job.video_id for job in jobs) + "\n", encoding="utf-8")


def _label_ready(job: VideoJob) -> bool:
    step_dir = interim_step_dir(job.dataset, job.video_id, "sam2_object")
    return label_prompt_path(step_dir).is_file()


def _cached_label_prompt_path(job: VideoJob) -> Path:
    return LABEL_CACHE_ROOT / job.dataset / job.video_id / "sam2_object" / "label_prompt.json"


def _flat_cached_label_prompt_path(job: VideoJob) -> Path:
    return LABEL_CACHE_ROOT / job.video_id / "sam2_object" / "label_prompt.json"


def _legacy_cached_label_prompt_path(job: VideoJob) -> Path:
    return LEGACY_LABEL_CACHE_ROOT / job.video_id / "sam2_object" / "label_prompt.json"


def _cache_label_prompt(job: VideoJob) -> bool:
    src = label_prompt_path(interim_step_dir(job.dataset, job.video_id, "sam2_object"))
    if not src.is_file():
        return False
    dst = _cached_label_prompt_path(job)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _restore_cached_label_prompt(job: VideoJob) -> bool:
    dst = label_prompt_path(interim_step_dir(job.dataset, job.video_id, "sam2_object"))
    if dst.is_file():
        return True
    src = _cached_label_prompt_path(job)
    if not src.is_file():
        src = _flat_cached_label_prompt_path(job)
    if not src.is_file():
        src = _legacy_cached_label_prompt_path(job)
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    new_cache = _cached_label_prompt_path(job)
    if src != new_cache:
        new_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, new_cache)
    return True


def _clear_interim_label_prompt(job: VideoJob) -> bool:
    path = label_prompt_path(interim_step_dir(job.dataset, job.video_id, "sam2_object"))
    if not path.is_file():
        return False
    path.unlink()
    return True


def _validate_final_reconstruction(job: VideoJob) -> dict:
    out_dir = final_video_dir(job.dataset, job.video_id)
    npz_path = out_dir / FINAL_NPZ
    mesh_path = out_dir / FINAL_MESH
    summary_path = out_dir / FINAL_SUMMARY
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing final NPZ: {npz_path}")
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing or empty final mesh: {mesh_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing final summary: {summary_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_FINAL_KEYS - set(data.files))
        if missing:
            raise RuntimeError(f"Final NPZ missing required keys: {missing}")
        schema_version = str(data["schema_version"].item())
        dataset_value = str(data["dataset"].item())
        video_id_value = str(data["video_id"].item())
        if schema_version != FINAL_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported final schema {schema_version}; expected {FINAL_SCHEMA_VERSION}")
        if dataset_value != job.dataset or video_id_value != job.video_id:
            raise RuntimeError(
                "Final identifiers mismatch: "
                f"({dataset_value}, {video_id_value}) != ({job.dataset}, {job.video_id})"
            )
        num_frames = int(data["num_frames"])
        c2w = data["c2w"]
        k_mat = data["K"]
        obj_cam = data["object_ob_in_cam"]
        obj_world = data["object_ob_in_world"]
        frame_indices = data["object_frame_indices"]
        t_zup = data["T_gravity_z_up_from_vipe_world"]
        gravity = data["gravity_direction_vipe_world"]
        up = data["up_direction_vipe_world"]
        if c2w.shape != (num_frames, 4, 4):
            raise RuntimeError(f"Invalid c2w shape {c2w.shape}; expected {(num_frames, 4, 4)}")
        if k_mat.shape != (3, 3):
            raise RuntimeError(f"Invalid K shape {k_mat.shape}; expected (3, 3)")
        if t_zup.shape != (4, 4):
            raise RuntimeError(f"Invalid T_gravity_z_up_from_vipe_world shape {t_zup.shape}; expected (4, 4)")
        if gravity.shape != (3,):
            raise RuntimeError(f"Invalid gravity_direction_vipe_world shape {gravity.shape}; expected (3,)")
        if up.shape != (3,):
            raise RuntimeError(f"Invalid up_direction_vipe_world shape {up.shape}; expected (3,)")
        r_zup = t_zup[:3, :3]
        if not np.allclose(r_zup @ gravity, np.array([0.0, 0.0, -1.0]), atol=1e-5):
            raise RuntimeError("Gravity alignment must map physical gravity to -Z")
        if not np.allclose(r_zup @ up, np.array([0.0, 0.0, 1.0]), atol=1e-5):
            raise RuntimeError("Gravity alignment must map up direction to +Z")
        world_xy_validation = _validate_world_xy_alignment(data, r_zup_from_vipe=r_zup, num_frames=num_frames)
        if obj_cam.ndim != 3 or obj_cam.shape[1:] != (4, 4) or obj_cam.shape[0] == 0:
            raise RuntimeError(f"Invalid object_ob_in_cam shape {obj_cam.shape}")
        if obj_world.shape != obj_cam.shape:
            raise RuntimeError(f"object_ob_in_world shape {obj_world.shape} does not match {obj_cam.shape}")
        if frame_indices.shape[0] != obj_cam.shape[0]:
            raise RuntimeError("object_frame_indices length does not match object pose count")
        hands_included = {"hand_trans", "hand_rot", "hand_pose", "hand_betas", "hand_valid"}.issubset(data.files)
        object_validation = _validate_object_arrays(data, final_dir=out_dir)
        hand_validation = _validate_hand_roots_in_front(data, num_frames=num_frames)
    mask_validation = _validate_final_masks(out_dir, num_frames=num_frames)
    return {
        "schema_version": FINAL_SCHEMA_VERSION,
        "status": "complete",
        "dataset": job.dataset,
        "video_id": job.video_id,
        "world_npz": FINAL_NPZ,
        "object_mesh": FINAL_MESH,
        "world_summary": FINAL_SUMMARY,
        "num_frames": num_frames,
        "num_objects": object_validation["num_objects"],
        "object_track_frames": int(frame_indices.shape[0]),
        "hands_included": hands_included,
        "hand_root_front_samples": hand_validation["hand_root_front_samples"],
        "hand_valid_samples": hand_validation["hand_valid_samples"],
        **world_xy_validation,
        **mask_validation,
    }


def _write_final_completion(job: VideoJob) -> Path:
    out_dir = final_video_dir(job.dataset, job.video_id)
    payload = _validate_final_reconstruction(job)
    marker = out_dir / FINAL_COMPLETION
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def _final_complete(job: VideoJob) -> bool:
    marker = final_video_dir(job.dataset, job.video_id) / FINAL_COMPLETION
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "complete":
        return False
    try:
        _validate_final_reconstruction(job)
    except Exception:
        return False
    return True


def _marker_dir(job: VideoJob, step: str) -> Path:
    """Where `step` keeps its completion marker.

    confidence writes to the final dir because cleanup deletes interim and the marker has
    to outlive it.  That rule used to be spelled out only inside _step_done, so the two
    other places that look markers up both probed interim unconditionally: the post-run
    log line always said "no completion marker" for confidence, and
    _collect_lightweight_step_metadata never recorded it.  Both were cosmetic -- gating
    ran through _step_done and was correct -- but the report claimed a step had not
    finished when it had.
    """
    if step in ("confidence", "contact"):
        return final_video_dir(job.dataset, job.video_id)
    return interim_step_dir(job.dataset, job.video_id, step)


def _step_done(job: VideoJob, step: str) -> bool:
    step_dir = _marker_dir(job, step)
    if step in ("confidence", "contact"):
        return is_step_complete(step_dir, step)
    if not is_step_complete(step_dir, step):
        return False
    if step == "hawor":
        marker = completion_marker(step_dir, step)
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("camera_timeline") == HAWOR_CAMERA_TIMELINE
    return True


def _read_json_if_present(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_lightweight_step_metadata(job: VideoJob) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for step in AUTO_STEPS:
        step_dir = interim_step_dir(job.dataset, job.video_id, step)
        marker_payload = _read_json_if_present(completion_marker(_marker_dir(job, step), step))
        if marker_payload is not None:
            metadata.setdefault(step, {})["completion"] = marker_payload
        if step == "fp_pose":
            fp_meta = _read_json_if_present(step_dir / "fp_pose_meta.json")
            if fp_meta is not None:
                metadata.setdefault(step, {})["meta"] = fp_meta
    return metadata


def _preserve_lightweight_step_metadata(job: VideoJob, batch_dir: Path) -> Path | None:
    metadata = _collect_lightweight_step_metadata(job)
    if not metadata:
        return None
    out_dir = batch_dir / "step_metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job.video_id}.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_path


def _cleanup_successful_video(job: VideoJob, batch_dir: Path, *, keep_interim: bool, keep_success_logs: bool) -> str:
    _write_final_completion(job)
    deleted = []
    if _cache_label_prompt(job):
        deleted.append("cached label prompt")
    metadata_path = _preserve_lightweight_step_metadata(job, batch_dir)
    if metadata_path is not None:
        deleted.append("cached step metadata")
    if keep_interim:
        deleted.append("kept interim")
    else:
        interim_dir = INTERIM_ROOT / job.dataset / job.video_id
        if interim_dir.is_dir():
            shutil.rmtree(interim_dir)
        deleted.append("deleted interim")
    if not keep_success_logs:
        shutil.rmtree(batch_dir / "logs" / job.video_id, ignore_errors=True)
        deleted.append("deleted success logs")
    else:
        deleted.append("kept logs")
    return ", ".join(deleted)


def _write_batch_summary(
    *,
    batch_dir: Path,
    jobs: list[VideoJob],
    state: BatchState,
    gpu_ids: list[int],
    steps: list[str],
    keep_interim: bool,
    keep_success_logs: bool,
    workers_per_gpu: int,
    gpu_mem_budget_mb: int,
    step_gpu_mem_mb: dict[str, int],
    elapsed_sec: float,
) -> None:
    videos = []
    for job in jobs:
        final_dir = final_video_dir(job.dataset, job.video_id)
        step_metadata_path = batch_dir / "step_metadata" / f"{job.video_id}.json"
        step_metadata = _read_json_if_present(step_metadata_path)
        if job.video_id in state.failed:
            status = "failed"
        elif _final_complete(job):
            status = "complete"
        else:
            status = "unknown"
        videos.append(
            {
                "video_id": job.video_id,
                "status": status,
                "error": state.failed.get(job.video_id),
                "final_dir": str(final_dir) if final_dir.is_dir() else None,
                "world_npz": str(final_dir / FINAL_NPZ) if (final_dir / FINAL_NPZ).is_file() else None,
                "object_mesh": str(final_dir / FINAL_MESH) if (final_dir / FINAL_MESH).is_file() else None,
                "interim_dir_exists": (INTERIM_ROOT / job.dataset / job.video_id).is_dir(),
                "step_metadata": str(step_metadata_path) if step_metadata_path.is_file() else None,
                "fp_pose_timing_sec": (step_metadata or {}).get("fp_pose", {}).get("meta", {}).get("timing_sec"),
            }
        )
    payload = {
        "status": "failed" if state.failed else "complete",
        "gpu_ids": gpu_ids,
        "steps": steps,
        "elapsed_sec": elapsed_sec,
        "scheduler": {
            "workers_per_gpu": workers_per_gpu,
            "gpu_mem_budget_mb": gpu_mem_budget_mb,
            "step_gpu_mem_mb": step_gpu_mem_mb,
        },
        "cleanup": {
            "keep_interim": keep_interim,
            "keep_success_logs": keep_success_logs,
        },
        "videos": videos,
    }
    (batch_dir / "batch_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Batch Summary", ""]
    lines.append(f"Status: `{payload['status']}`")
    lines.append(f"GPUs: `{','.join(str(x) for x in gpu_ids)}`")
    lines.append(f"Steps: `{','.join(steps)}`")
    lines.append(f"Elapsed seconds: `{elapsed_sec:.1f}`")
    lines.append(f"Workers per GPU: `{workers_per_gpu}`")
    lines.append(f"GPU memory budget MB: `{gpu_mem_budget_mb}`")
    lines.append("")
    lines.append("| Video | Status | Interim exists | FP total sec | Final output |")
    lines.append("|---|---|---:|---:|---|")
    for video in videos:
        final_output = video["world_npz"] or ""
        fp_timing = video.get("fp_pose_timing_sec") or {}
        fp_total = fp_timing.get("total")
        fp_total_text = f"{float(fp_total):.1f}" if fp_total is not None else ""
        lines.append(
            f"| `{video['video_id']}` | {video['status']} | {video['interim_dir_exists']} | "
            f"{fp_total_text} | `{final_output}` |"
        )
    (batch_dir / "batch_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _step_log_path(job: VideoJob, step: str, batch_dir: Path, gpu_id: int | None) -> Path:
    log_dir = batch_dir / "logs" / job.video_id
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"gpu{gpu_id}" if gpu_id is not None else "cpu"
    return log_dir / f"{step}.{suffix}.log"


def _build_step_cmd(
    step: str,
    job: VideoJob,
    *,
    gpu_id: int,
    visualize: bool,
    force: bool,
) -> tuple[list[str], Path]:
    script = STEP_SCRIPTS[step]
    cmd = ["conda", "run", "-n", STEP_ENVS[step]]
    if step == "vipe":
        cmd.extend(["uv", "run", "--no-sync", "python", str(script.resolve())])
        cwd = REPO_ROOT / "third_party" / "vipe"
    else:
        cmd.extend(["python", str(script.relative_to(REPO_ROOT))])
        cwd = REPO_ROOT
    cmd.extend(
        [
            "--dataset",
            job.dataset,
            "--video-id",
            job.video_id,
            "--video",
            str(resolve_repo_path(job.video_path)),
        ]
    )
    if step in CUDA_VISIBLE_DEVICE_STEPS:
        cmd.extend(["--gpu", "0"])
    elif step in GPU_STEPS:
        cmd.extend(["--gpu", str(gpu_id)])
    if visualize:
        cmd.append("--visualize")
    if force:
        cmd.append("--force")
    cmd.extend(STEP_EXTRA_ARGS.get(step, []))
    return cmd, cwd


def _build_label_cmd(
    jobs: list[VideoJob],
    *,
    video_list: Path,
    dataset_root: Path | None,
    label_mode: str,
    http_host: str,
    http_port: int,
    no_browser: bool,
    preview_device: str,
    preview_gpu: int,
    no_mask_preview: bool,
) -> tuple[list[str], Path]:
    cmd = [
        "conda",
        "run",
        "-n",
        STEP_ENVS["sam2_object"],
        "python",
        str((RECON_ROOT / "sam2_object" / "label_object.py").relative_to(REPO_ROOT)),
        "--dataset",
        jobs[0].dataset,
        "--video-list",
        str(video_list),
        "--label-mode",
        label_mode,
        "--http-host",
        http_host,
        "--http-port",
        str(http_port),
        "--preview-device",
        preview_device,
    ]
    if dataset_root is not None:
        cmd.extend(["--dataset-root", str(dataset_root)])
    if preview_device == "cuda":
        cmd.extend(["--gpu", str(preview_gpu)])
    if no_browser:
        cmd.append("--no-browser")
    if no_mask_preview:
        cmd.append("--no-mask-preview")
    return cmd, REPO_ROOT


def _run_step(
    step: str,
    job: VideoJob,
    *,
    worker_idx: int,
    gpu_ids: list[int],
    batch_dir: Path,
    visualize: bool,
    force: bool,
    dry_run: bool,
    gpu_reservation: GpuReservation,
) -> None:
    prefix = f"[worker {worker_idx}] ({job.video_id})"
    if not force and _step_done(job, step):
        print(f"{prefix} {step}: complete, skip", flush=True)
        return
    if dry_run:
        gpu_id = gpu_ids[0]
        cmd, cwd = _build_step_cmd(step, job, gpu_id=gpu_id, visualize=visualize, force=force)
        env_prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if step in CUDA_VISIBLE_DEVICE_STEPS else ""
        print(f"{prefix} {step}: dry-run ({cwd}) {env_prefix}{_quote_cmd(cmd)}", flush=True)
        return
    gpu_id = gpu_ids[0]
    reservation_mb = 0
    if step in GPU_STEPS:
        gpu_id, reservation_mb = gpu_reservation.acquire_any(gpu_ids, step, prefix)
        prefix = f"[gpu {gpu_id} worker {worker_idx}] ({job.video_id})"
    cmd, cwd = _build_step_cmd(step, job, gpu_id=gpu_id, visualize=visualize, force=force)
    log_path = _step_log_path(job, step, batch_dir, gpu_id if step in GPU_STEPS else None)
    device_label = f"GPU {gpu_id}" if step in GPU_STEPS else "CPU"
    if step in CUDA_VISIBLE_DEVICE_STEPS:
        device_label += " (isolated as cuda:0)"
    print(f"{prefix} {step}: start on {device_label}", flush=True)
    env = os.environ.copy()
    cpu_threads = DEFAULT_STEP_CPU_THREADS.get(step, "1")
    env.setdefault("OMP_NUM_THREADS", cpu_threads)
    env.setdefault("MKL_NUM_THREADS", cpu_threads)
    env.setdefault("OPENBLAS_NUM_THREADS", cpu_threads)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if step == "vipe":
        uv_cache_dir = batch_dir / "uv_cache"
        uv_cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("UV_CACHE_DIR", str(uv_cache_dir))
    if step in CUDA_VISIBLE_DEVICE_STEPS:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    start = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            if step in CUDA_VISIBLE_DEVICE_STEPS:
                log.write(f"$ export CUDA_VISIBLE_DEVICES={gpu_id}\n")
            if reservation_mb > 0:
                log.write(f"$ # reserved_gpu_mem_mb={reservation_mb}\n")
            log.write("$ " + _quote_cmd(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    finally:
        gpu_reservation.release(gpu_id, reservation_mb)
    elapsed = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{step} failed for {job.video_id} with code {proc.returncode}; see {log_path}")
    marker = completion_marker(_marker_dir(job, step), step)
    marker_status = "complete" if marker.is_file() else "no completion marker"
    print(f"{prefix} {step}: done in {elapsed:.1f}s ({marker_status})", flush=True)


def _worker(
    *,
    name: str,
    worker_idx: int,
    gpu_ids: list[int],
    work_queue: "queue.Queue[VideoJob | None]",
    state: BatchState,
    state_lock: threading.Lock,
    steps: list[str],
    batch_dir: Path,
    visualize: bool,
    force: bool,
    keep_interim: bool,
    keep_success_logs: bool,
    dry_run: bool,
    gpu_reservation: GpuReservation,
) -> None:
    worker_tag = f"[worker {worker_idx}]"
    while True:
        job = work_queue.get()
        if job is None:
            work_queue.task_done()
            return
        with state_lock:
            state.running[job.video_id] = name
        try:
            skip_steps = False
            if not force and _final_complete(job):
                # 重建已完成的视频: 不重跑重建, 但 confidence 缺了要补
                # (断点场景: fuse 完成后进程中断, interim 已删, 走 steps 循环会整条重跑)
                topups = [s2 for s2 in ("confidence", "contact")
                          if s2 in steps and not _step_done(job, s2)]
                for s2 in topups:
                    print(f"{worker_tag} ({job.video_id}) final complete, top up {s2}", flush=True)
                    _run_step(
                        s2,
                        job,
                        worker_idx=worker_idx,
                        gpu_ids=gpu_ids,
                        batch_dir=batch_dir,
                        visualize=visualize,
                        force=force,
                        dry_run=dry_run,
                        gpu_reservation=gpu_reservation,
                    )
                if not topups:
                    print(f"{worker_tag} ({job.video_id}) final output complete, skip", flush=True)
                cleanup = _cleanup_successful_video(
                    job,
                    batch_dir,
                    keep_interim=keep_interim,
                    keep_success_logs=keep_success_logs,
                )
                skip_steps = True
            else:
                cleanup = None
                if "fuse" not in steps and not keep_interim:
                    raise RuntimeError("aggressive cleanup requires the fuse step in --steps")
            for step in steps:
                if skip_steps:
                    break
                _run_step(
                    step,
                    job,
                    worker_idx=worker_idx,
                    gpu_ids=gpu_ids,
                    batch_dir=batch_dir,
                    visualize=visualize,
                    force=force,
                    dry_run=dry_run,
                    gpu_reservation=gpu_reservation,
                )
            if cleanup is None and "fuse" in steps:
                cleanup = _cleanup_successful_video(
                    job,
                    batch_dir,
                    keep_interim=keep_interim,
                    keep_success_logs=keep_success_logs,
                )
            elif cleanup is None:
                cleanup = "kept interim"
        except Exception as exc:  # noqa: BLE001 - worker records and continues to next video
            with state_lock:
                state.failed[job.video_id] = str(exc)
            print(f"{worker_tag} ({job.video_id}) failed: {exc}", flush=True)
        else:
            with state_lock:
                state.done.add(job.video_id)
            print(f"{worker_tag} ({job.video_id}) all requested steps complete ({cleanup})", flush=True)
        finally:
            with state_lock:
                state.running.pop(job.video_id, None)
            work_queue.task_done()


def _discover_jobs(args: argparse.Namespace) -> list[VideoJob]:
    if args.video_id:
        return [resolve_video_job(args.dataset, args.video_id, dataset_root=args.dataset_root)]
    return discover_videos(
        args.dataset,
        args.input,
        dataset_root=args.dataset_root,
        video_list=args.video_list,
        sample=args.sample,
        limit=args.limit,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("input", type=Path, nargs="?", default=None)
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-ids", type=_parse_gpu_ids, default=[0], help="Comma-separated worker GPU ids")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Reconstruction worker slots per listed GPU. Values >1 use approximate GPU-memory reservations.",
    )
    parser.add_argument(
        "--gpu-mem-budget-mb",
        type=int,
        default=DEFAULT_GPU_MEM_BUDGET_MB,
        help="Approximate per-GPU memory budget for scheduling multiple workers on one GPU",
    )
    parser.add_argument(
        "--step-gpu-mem-mb",
        type=_parse_step_mem_overrides,
        default=None,
        help="Comma-separated per-step GPU memory estimates, e.g. sam2_object=9000,hawor=8000",
    )
    parser.add_argument(
        "--steps",
        default=",".join(AUTO_STEPS),
        help=f"Comma-separated reconstruction steps after labeling. Options: {','.join(AUTO_STEPS)}",
    )
    parser.add_argument("--skip-label", action="store_true", help="Do not start labeling; process videos already labeled")
    parser.add_argument(
        "--force-label",
        action="store_true",
        help="Delete selected videos' interim label prompts before labeling, forcing the UI to collect new labels",
    )
    parser.add_argument("--label-mode", choices=("http",), default="http", help="Batch labeling currently uses one HTTP page")
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--preview-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preview-gpu", type=int, default=0)
    parser.add_argument("--no-mask-preview", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--step-arg", action="append", default=[], metavar="STEP:FLAG",
                        help="给某一步透传额外参数, 可重复。如 --step-arg fp_pose:--pose-mode=register-each "
                             "(此前批量队列没有任何透传机制, register-each 等只能直调步骤脚本)")
    parser.add_argument(
        "--keep-interim",
        action="store_true",
        help="Keep per-video data/interim outputs after successful final reconstruction",
    )
    parser.add_argument(
        "--keep-success-logs",
        action="store_true",
        help="Keep per-step logs for successful videos; failed-video logs are always kept",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.force_label and args.skip_label:
        parser.error("--force-label cannot be combined with --skip-label")
    if args.force_label and not args.force:
        parser.error("--force-label requires --force so reconstruction is rerun with the new labels")
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be >= 1")
    if args.gpu_mem_budget_mb <= 0:
        parser.error("--gpu-mem-budget-mb must be positive")
    step_gpu_mem_mb = _parse_step_mem_overrides(None) if args.step_gpu_mem_mb is None else args.step_gpu_mem_mb
    for step, mem_mb in step_gpu_mem_mb.items():
        if mem_mb > args.gpu_mem_budget_mb:
            parser.error(
                f"GPU memory estimate for {step} ({mem_mb} MB) exceeds --gpu-mem-budget-mb "
                f"({args.gpu_mem_budget_mb} MB)"
            )
    total_start = time.time()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for sa in args.step_arg:
        st, _, flag = sa.partition(":")
        if not flag or st not in STEP_SCRIPTS:
            raise SystemExit(f"--step-arg 格式应为 <step>:<flag>, 未知步骤或缺参数: {sa!r}")
        STEP_EXTRA_ARGS.setdefault(st, []).append(flag)
    if STEP_EXTRA_ARGS:
        print(f"Step extra args: {dict(STEP_EXTRA_ARGS)}", flush=True)
    unknown = set(steps) - set(AUTO_STEPS)
    if unknown:
        parser.error(f"Unknown steps: {sorted(unknown)}")

    jobs = _discover_jobs(args)
    batch_dir = _batch_dir(args.dataset)
    write_video_manifest(jobs, batch_dir / "manifest.json", seed=args.seed)
    label_list = batch_dir / "label_video_list.txt"
    _write_label_list(jobs, label_list)

    print(f"Batch directory: {batch_dir}", flush=True)
    print(f"Videos: {len(jobs)}", flush=True)
    print(f"GPUs: {args.gpu_ids}", flush=True)
    print(f"Worker slots per GPU: {args.workers_per_gpu}", flush=True)
    print(f"GPU memory budget: {args.gpu_mem_budget_mb} MB", flush=True)
    print(f"Steps after label: {','.join(steps)}", flush=True)
    print(
        "Cleanup: "
        + ("keep interim" if args.keep_interim else "delete per-video interim after final validation")
        + (", keep success logs" if args.keep_success_logs else ", delete success logs"),
        flush=True,
    )

    if args.dry_run:
        if not args.skip_label:
            label_jobs = jobs if args.force or args.force_label else [job for job in jobs if not _final_complete(job)]
            _write_label_list(label_jobs or jobs, label_list)
            label_cmd, label_cwd = _build_label_cmd(
                label_jobs or jobs,
                video_list=label_list,
                dataset_root=args.dataset_root,
                label_mode=args.label_mode,
                http_host=args.http_host,
                http_port=args.http_port,
                no_browser=args.no_browser,
                preview_device=args.preview_device,
                preview_gpu=args.preview_gpu,
                no_mask_preview=args.no_mask_preview,
            )
            force_label_note = " (would clear interim label prompts first)" if args.force_label else ""
            print(f"[dry-run] label{force_label_note} ({label_cwd}) {_quote_cmd(label_cmd)}", flush=True)
        worker_slots = _worker_slots(args.gpu_ids, args.workers_per_gpu)
        for idx, job in enumerate(jobs):
            gpu_id, local_worker_idx = worker_slots[idx % len(worker_slots)]
            worker_tag = f"[worker {local_worker_idx}]"
            for step in steps:
                cmd, cwd = _build_step_cmd(step, job, gpu_id=gpu_id, visualize=args.visualize, force=args.force)
                env_prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if step in CUDA_VISIBLE_DEVICE_STEPS else ""
                gpu_note = " runtime GPU selected dynamically;" if step in GPU_STEPS else ""
                print(
                    f"{worker_tag} ({job.video_id}) {step}: dry-run ({cwd});{gpu_note} "
                    f"{env_prefix}{_quote_cmd(cmd)}",
                    flush=True,
                )
        return 0

    label_proc: subprocess.Popen | None = None
    label_jobs = jobs if args.force or args.force_label else [job for job in jobs if not _final_complete(job)]
    if args.force_label and label_jobs:
        cleared = sum(1 for job in label_jobs if _clear_interim_label_prompt(job))
        print(f"Force relabel: cleared {cleared} existing interim label prompt(s)", flush=True)
    if not args.skip_label and label_jobs:
        _write_label_list(label_jobs, label_list)
        label_cmd, label_cwd = _build_label_cmd(
            label_jobs,
            video_list=label_list,
            dataset_root=args.dataset_root,
            label_mode=args.label_mode,
            http_host=args.http_host,
            http_port=args.http_port,
            no_browser=args.no_browser,
            preview_device=args.preview_device,
            preview_gpu=args.preview_gpu,
            no_mask_preview=args.no_mask_preview,
        )
        label_log = batch_dir / "label_server.log"
        label_log_fh = label_log.open("w", encoding="utf-8")
        label_log_fh.write("$ " + _quote_cmd(label_cmd) + "\n\n")
        label_log_fh.flush()
        print(f"Starting labeling server; log: {label_log}", flush=True)
        label_proc = subprocess.Popen(label_cmd, cwd=str(label_cwd), stdout=label_log_fh, stderr=subprocess.STDOUT)
        print(f"Labeling preview device: {args.preview_device}", flush=True)
        print(f"Open labeling UI: http://127.0.0.1:{args.http_port}/", flush=True)
        if args.http_host == "0.0.0.0":
            print(f"Remote access: ssh -L {args.http_port}:127.0.0.1:{args.http_port} user@host", flush=True)
    elif not args.skip_label:
        print("All videos already have valid final outputs; labeling is skipped.", flush=True)
    elif not _steps_need_object_label(steps):
        # ★ 只有真正消费 label_prompt.json 的步骤才需要物体标注。--steps=vipe 之类不需要,
        #   却被这道门拦下 —— 实测 reconstruct_egodex.sh 第一步只跑 vipe(拿深度), 直接
        #   "Missing labels with --skip-label" 退出码 2, 整条 EgoDex 入口起不来。
        #   同样的漏洞早先在 run_pipeline.py 修过, 这个文件漏了。
        print(f"Labeling not required for steps={','.join(steps)}; continuing.", flush=True)
    elif not all(_restore_cached_label_prompt(job) or (not args.force and _final_complete(job)) for job in jobs):
        missing = [
            job.video_id
            for job in jobs
            if not _restore_cached_label_prompt(job) and not (not args.force and _final_complete(job))
        ]
        print("Missing labels with --skip-label:", ", ".join(missing), file=sys.stderr, flush=True)
        return 2

    work_queue: "queue.Queue[VideoJob | None]" = queue.Queue()
    state = BatchState()
    state_lock = threading.Lock()
    gpu_reservation = GpuReservation(
        args.gpu_ids,
        budget_mb=args.gpu_mem_budget_mb,
        step_mem_mb=step_gpu_mem_mb,
    )
    workers = []
    for _gpu_id, worker_idx in _worker_slots(args.gpu_ids, args.workers_per_gpu):
        name = f"worker{worker_idx}"
        thread = threading.Thread(
            target=_worker,
            kwargs={
                "name": name,
                "worker_idx": worker_idx,
                "gpu_ids": args.gpu_ids,
                "work_queue": work_queue,
                "state": state,
                "state_lock": state_lock,
                "steps": steps,
                "batch_dir": batch_dir,
                "visualize": args.visualize,
                "force": args.force,
                "keep_interim": args.keep_interim,
                "keep_success_logs": args.keep_success_logs,
                "dry_run": False,
                "gpu_reservation": gpu_reservation,
            },
            daemon=True,
        )
        thread.start()
        workers.append(thread)

    try:
        with state_lock:
            for job in jobs:
                # 只有"重建完成 且 confidence/contact 也都齐"才在入队前短路——否则放进
                # worker 循环, 由那里的 top-up 分支补缺的后处理步(2026-08-10: 之前只查
                # _final_complete, confidence 失败过的视频会被这里清理+跳过, 永远补不上)
                post_missing = [s2 for s2 in ("confidence", "contact")
                                if s2 in steps and not _step_done(job, s2)]
                if not args.force and _final_complete(job) and not post_missing:
                    cleanup = _cleanup_successful_video(
                        job,
                        batch_dir,
                        keep_interim=args.keep_interim,
                        keep_success_logs=args.keep_success_logs,
                    )
                    print(f"[batch] ({job.video_id}) final output complete ({cleanup})", flush=True)
                    state.done.add(job.video_id)
        while True:
            with state_lock:
                terminal = state.done | set(state.failed)
                known = state.queued | set(state.running) | terminal
            for job in jobs:
                # ★ 不需要物体标注的步骤(如 --steps=vipe)必须能直接入队。
                #   ⚠ 这个坑踩过两次: 上一次只修了上面那道"快速失败"门, **入队这道没修**,
                #     结果不报错也不干活 —— 任务永远进不了队列, worker 空等, 日志停在
                #     "Labeling not required..." 一动不动(实测本次空转 11 分钟才被发现)。
                #     两处判据必须同源, 否则修一处等于没修。
                if not _steps_need_object_label(steps):
                    label_ready = True
                else:
                    label_ready = (_restore_cached_label_prompt(job) if args.skip_label
                                   else _label_ready(job))
                if job.video_id not in known and label_ready:
                    cached_label = _cache_label_prompt(job)
                    print(f"[queue] ({job.video_id}) label ready -> enqueue reconstruction", flush=True)
                    if cached_label:
                        print(f"[queue] ({job.video_id}) cached label prompt", flush=True)
                    with state_lock:
                        state.queued.add(job.video_id)
                    work_queue.put(job)

            if len(terminal) == len(jobs):
                break

            if label_proc is not None and label_proc.poll() is not None:
                if label_proc.returncode != 0:
                    missing = [job.video_id for job in label_jobs if not _label_ready(job)]
                    if missing:
                        print(f"Labeling exited with code {label_proc.returncode}; missing labels: {missing}", flush=True)
                        with state_lock:
                            for video_id in missing:
                                state.failed[video_id] = f"labeling exited before prompt was saved"
                    label_proc = None
                elif all(_label_ready(job) for job in label_jobs):
                    label_proc = None
                else:
                    missing = [job.video_id for job in label_jobs if not _label_ready(job)]
                    print(f"Labeling exited before all prompts were saved; missing labels: {missing}", flush=True)
                    with state_lock:
                        for video_id in missing:
                            state.failed[video_id] = "labeling exited before prompt was saved"
                    label_proc = None

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("Interrupted; terminating label server and workers after current subprocesses exit.", flush=True)
        if label_proc is not None and label_proc.poll() is None:
            label_proc.terminate()
        return 130

    work_queue.join()
    for _ in workers:
        work_queue.put(None)
    for thread in workers:
        thread.join(timeout=1)

    if state.failed:
        _write_batch_summary(
            batch_dir=batch_dir,
            jobs=jobs,
            state=state,
            gpu_ids=args.gpu_ids,
            steps=steps,
            keep_interim=args.keep_interim,
            keep_success_logs=args.keep_success_logs,
            workers_per_gpu=args.workers_per_gpu,
            gpu_mem_budget_mb=args.gpu_mem_budget_mb,
            step_gpu_mem_mb=step_gpu_mem_mb,
            elapsed_sec=time.time() - total_start,
        )
        print("Failed videos:", flush=True)
        for video_id, error in state.failed.items():
            print(f"  {video_id}: {error}", flush=True)
        print(f"Batch logs: {batch_dir}", flush=True)
        return 1
    _write_batch_summary(
        batch_dir=batch_dir,
        jobs=jobs,
        state=state,
        gpu_ids=args.gpu_ids,
        steps=steps,
        keep_interim=args.keep_interim,
        keep_success_logs=args.keep_success_logs,
        workers_per_gpu=args.workers_per_gpu,
        gpu_mem_budget_mb=args.gpu_mem_budget_mb,
        step_gpu_mem_mb=step_gpu_mem_mb,
        elapsed_sec=time.time() - total_start,
    )
    print(f"Batch complete. Logs: {batch_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
