"""Thin runtime wrapper around the official HOI-DETR inference code.

This module deliberately keeps every HOI-DETR/MMCV import behind ``load`` so
the repository-owned schema and tests remain usable without the heavyweight
model environment.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_NAMES = ("hand", "firstobject", "secondobject")
DEFAULT_CONFIG_RELPATH = Path(
    "projects/configs/co_dino_vit/"
    "co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py"
)


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def resolve_upstream_paths(
    source_root: Path,
    checkpoint: Path,
    config: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve and validate the official source, config, and user-supplied weights."""
    root = source_root.expanduser().resolve()
    if not (root / "demo" / "helpers.py").is_file():
        raise FileNotFoundError(
            f"Not an HOI-DETR source tree (missing demo/helpers.py): {root}"
        )
    config_path = config or DEFAULT_CONFIG_RELPATH
    if not config_path.is_absolute():
        config_path = root / config_path
    return root, _require_file(config_path, "HOI-DETR config"), _require_file(
        checkpoint, "HOI-DETR checkpoint"
    )


def _prepend_once(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def _require_module_origin(module: Any, expected_root: Path, label: str) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"Cannot verify {label} module origin")
    resolved_file = Path(module_file).resolve()
    try:
        resolved_file.relative_to(expected_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Imported {label} from {resolved_file}, outside expected root {expected_root}"
        ) from exc


@dataclass
class OfficialHoiDetrRuntime:
    """Loaded official model plus its preprocessing and interaction heads."""

    source_root: Path
    config_path: Path
    checkpoint_path: Path
    device: str
    model: Any
    test_pipeline: Any
    interaction_branch: Any
    helpers: Any
    mmcv: Any

    @classmethod
    def load(
        cls,
        *,
        source_root: Path,
        checkpoint: Path,
        config: Path | None,
        device: str,
    ) -> "OfficialHoiDetrRuntime":
        root, config_path, checkpoint_path = resolve_upstream_paths(
            source_root, checkpoint, config
        )

        # The official helper imports ``configs`` as a top-level module.
        _prepend_once(root)
        _prepend_once(root / "demo")

        try:
            mmcv = importlib.import_module("mmcv")
        except ImportError as exc:
            raise RuntimeError(
                "HOI-DETR requires an isolated environment with MMCV 1.7.2 "
                "built with CUDA ops. See experiments/hoi_detr/README.md."
            ) from exc

        # MMDetection 2.25.3 has a stale MMCV upper-bound. This is the same
        # narrowly scoped import-time compatibility patch used by the official
        # HOI-DETR Hugging Face demo for PyTorch 2.x + MMCV 1.7.2.
        real_mmcv_version = mmcv.__version__
        try:
            mmcv.__version__ = "1.5.0"
            mmdet = importlib.import_module("mmdet")
        finally:
            mmcv.__version__ = real_mmcv_version
        _require_module_origin(mmdet, root / "mmdet", "mmdet")

        try:
            projects = importlib.import_module("projects")
            helpers = importlib.import_module("helpers")
            init_detector = importlib.import_module("mmdet.apis").init_detector
            compose = importlib.import_module("mmdet.datasets.pipelines").Compose
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"Failed to import the official HOI-DETR runtime from {root}"
            ) from exc
        _require_module_origin(projects, root / "projects", "projects")
        _require_module_origin(helpers, root / "demo", "helpers")

        model = init_detector(str(config_path), str(checkpoint_path), device=device)
        model.eval()
        test_pipeline = compose(model.cfg.data.test.pipeline)
        interaction_branch = helpers.find_interaction_branch(model.query_head)
        return cls(
            source_root=root,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            model=model,
            test_pipeline=test_pipeline,
            interaction_branch=interaction_branch,
            helpers=helpers,
            mmcv=mmcv,
        )

    def predict_frame(
        self,
        frame_bgr: Any,
        *,
        scratch_image: Path,
        score_threshold: float,
        nms_iou: float,
        hf_threshold: float,
        fs_threshold: float,
    ) -> dict[str, list[dict[str, Any]]]:
        """Infer one frame and return official-style detections and typed links."""
        if not self.mmcv.imwrite(frame_bgr, str(scratch_image)):
            raise RuntimeError(f"Failed to write scratch frame: {scratch_image}")

        detections, embeddings = self.helpers.run_inference(
            self.model,
            self.test_pipeline,
            str(scratch_image),
            device=self.device,
            class_names=CLASS_NAMES,
            score_thr=score_threshold,
            nms_iou=nms_iou,
        )

        height, width = frame_bgr.shape[:2]
        clipped_count = 0
        dropped_count = 0
        valid_detections = []
        for detection in detections:
            original_box = [float(value) for value in detection["box"]]
            clipped_box = [
                min(max(original_box[0], 0.0), float(width)),
                min(max(original_box[1], 0.0), float(height)),
                min(max(original_box[2], 0.0), float(width)),
                min(max(original_box[3], 0.0), float(height)),
            ]
            if clipped_box != original_box:
                clipped_count += 1
            if clipped_box[0] >= clipped_box[2] or clipped_box[1] >= clipped_box[3]:
                dropped_count += 1
                continue
            normalized = dict(detection)
            normalized["box"] = clipped_box
            valid_detections.append(normalized)

        det_index = {id(det): idx for idx, det in enumerate(valid_detections)}
        hands = [det for det in valid_detections if det["class_id"] == 0]
        firsts = [det for det in valid_detections if det["class_id"] == 1]
        seconds = [det for det in valid_detections if det["class_id"] == 2]

        hf_links: list[dict[str, Any]] = []
        fs_links: list[dict[str, Any]] = []
        for hand in hands:
            for first in firsts:
                interacts, probability = self.helpers.call_interaction(
                    self.interaction_branch,
                    embeddings[hand["query_idx"]],
                    embeddings[first["query_idx"]],
                )
                if interacts and probability >= hf_threshold:
                    hf_links.append(
                        {
                            "a": det_index[id(hand)],
                            "b": det_index[id(first)],
                            "prob": float(probability),
                        }
                    )

        for first in firsts:
            for second in seconds:
                interacts, probability = self.helpers.call_interaction(
                    self.interaction_branch,
                    embeddings[first["query_idx"]],
                    embeddings[second["query_idx"]],
                )
                if interacts and probability >= fs_threshold:
                    fs_links.append(
                        {
                            "a": det_index[id(first)],
                            "b": det_index[id(second)],
                            "prob": float(probability),
                        }
                    )

        serializable_detections = [
            {
                "box": [float(value) for value in det["box"]],
                "score": float(det["score"]),
                "class_id": int(det["class_id"]),
                "class_name": str(det["class_name"]),
            }
            for det in valid_detections
        ]
        return {
            "detections": serializable_detections,
            "hf": hf_links,
            "fs": fs_links,
            "diagnostics": {
                "clipped_boxes": clipped_count,
                "dropped_degenerate_boxes": dropped_count,
            },
        }
