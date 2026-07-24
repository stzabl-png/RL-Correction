"""Discard interim artifacts not required by downstream recon steps."""

from __future__ import annotations

import shutil
from pathlib import Path


def discard_hawor_nonessential(step_dir: Path, video_id: str) -> None:
    """Keep world MANO; drop HaWoR sidecars and optional vis unless under vis/."""
    seq = step_dir / video_id
    if not seq.is_dir():
        return
    for name in ("meta.json", "slam.pkl"):
        (seq / name).unlink(missing_ok=True)


def discard_sam3_hands_nonessential(step_dir: Path, video_id: str) -> None:
    """Keep propagated masks; drop duplicate SAM3 completion marker and legacy vis paths."""
    seq = step_dir / video_id
    marker = seq / "hand_masks_complete.json"
    marker.unlink(missing_ok=True)
    legacy_vis = seq / f"{video_id}_hand_masks_vis.mp4"
    std_vis = step_dir / "vis" / f"{video_id}.mp4"
    if legacy_vis.is_file() and not std_vis.is_file():
        std_vis.parent.mkdir(parents=True, exist_ok=True)
        legacy_vis.rename(std_vis)
    overlays = seq / "video_segmentation" / "overlays"
    if overlays.is_dir():
        shutil.rmtree(overlays)


def discard_sam2_object_nonessential(step_dir: Path) -> None:
    """Keep masks + label_prompt.json; drop duplicate inner completion marker."""
    (step_dir / "object_masks_complete.json").unlink(missing_ok=True)
    legacy_vis = step_dir / "object_masks_vis.mp4"
    std_vis_dir = step_dir / "vis"
    if legacy_vis.is_file():
        std_vis_dir.mkdir(parents=True, exist_ok=True)
        target = std_vis_dir / legacy_vis.name
        if not target.is_file():
            legacy_vis.rename(target)


def discard_sam3d_nonessential(step_dir: Path) -> None:
    """Keep metric mesh + meta; drop temp work dirs and legacy preview paths."""
    shutil.rmtree(step_dir / ".work", ignore_errors=True)
    legacy = step_dir / "mesh_preview.png"
    std = step_dir / "vis" / "mesh_preview.png"
    if legacy.is_file() and not std.is_file():
        std.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(std)


def discard_fp_pose_nonessential(step_dir: Path) -> None:
    """Keep ob_in_cam poses + meta; drop FP debug dirs and legacy Isaac scene exports."""
    for dirname in ("fp_debug", "scene", "masks"):
        path = step_dir / dirname
        if path.is_dir():
            shutil.rmtree(path)
