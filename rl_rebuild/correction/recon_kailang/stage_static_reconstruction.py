"""Stage one static reconstruction for the Step4 training environment.

Example::

    python -m rl_rebuild.correction.recon_kailang.stage_static_reconstruction \
      egodex task1_static_smoke \
      --reconstruction /path/to/ReconstructOutput/.../take \
      --retarget /path/to/RetargetOutput/.../take \
      --src-clip egodex/test/basic_pick_place/1

The destination is created atomically under ``TrainingData/<dataset>/<name>``.
Incomplete staging directories are removed on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from .run_manifest import artifact, finish_manifest, start_manifest


_TD = Path(__file__).resolve().parents[2] / "TrainingData"
_RECON_REQUIRED = ("object_mesh_scaled_final.obj", "world_fused.npz")
_RECON_OPTIONAL = ("world_summary.json", "reconstruction_complete.json")
_RETARGET_REQUIRED = ("replay_world.npz",)
_RETARGET_OPTIONAL = ("ref_qpos.npz",)


def _require_files(directory: Path, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} missing required files: {missing}")


def _copy_selected(source: Path, destination: Path, names: tuple[str, ...]) -> list[str]:
    copied = []
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = source / name
        if src.is_file():
            shutil.copy2(src, destination / name)
            copied.append(name)
    return copied


def stage_static(
    dataset: str,
    name: str,
    reconstruction: str,
    retarget: str,
    src_clip: str,
    *,
    mass_kg: float = 0.2,
    friction: float = 0.5,
    dry_run: bool = False,
) -> Path:
    """Validate and atomically stage one static-object reconstruction."""

    recon = Path(reconstruction).resolve()
    ret = Path(retarget).resolve()
    _require_files(recon, _RECON_REQUIRED, "reconstruction")
    _require_files(ret, _RETARGET_REQUIRED, "retarget")

    destination = (_TD / dataset / name).resolve()
    if destination.exists():
        raise FileExistsError(
            f"destination already exists: {destination}; remove it only after reviewing it"
        )
    if dry_run:
        print(f"[dry-run] {src_clip} -> {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.staging-", dir=destination.parent))
    try:
        provenance = staging / "provenance"
        inputs = {
            "object_mesh_scaled_final": artifact(
                recon / "object_mesh_scaled_final.obj"),
            "world_fused": artifact(recon / "world_fused.npz"),
            "replay_world": artifact(ret / "replay_world.npz"),
        }
        running_manifest = start_manifest(
            provenance,
            final_name="step4_staging_run_manifest.json",
            stage="step4_static_reconstruction_staging",
            entrypoint=__file__,
            command=["python", "-m", __name__],
            inputs=inputs,
            parameters={
                "dataset": dataset,
                "name": name,
                "src_clip": src_clip,
                "mass_kg": mass_kg,
                "friction": friction,
            },
            expected_outputs=[
                "reconstruction/object_mesh_scaled_final.obj",
                "reconstruction/world_fused.npz",
                "retarget/replay_world.npz",
                "meta.json",
            ],
        )
        copied = {
            "reconstruction": _copy_selected(
                recon, staging / "reconstruction", _RECON_REQUIRED + _RECON_OPTIONAL),
            "retarget": _copy_selected(
                ret, staging / "retarget", _RETARGET_REQUIRED + _RETARGET_OPTIONAL),
        }
        (staging / "cache").mkdir()
        upstream_manifests = []
        for source, target_name in (
            (recon / "run_manifest.json", "reconstruction_run_manifest.json"),
            (ret / "run_manifest.json", "retarget_run_manifest.json"),
        ):
            if source.is_file():
                provenance.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, provenance / target_name)
                upstream_manifests.append(target_name)
        meta = {
            "dataset": dataset,
            "object": name,
            "src_clip": src_clip,
            "source": "static_reconstruction",
            "mass_kg": mass_kg,
            "friction": friction,
            "placement": {
                "interaction_frame": "earliest phase_left/right == 1",
                "xy": "rotated mesh AABB centre = aligned wrist XY",
                "rotation": "FoundationPose, frame-converted and normalized only",
                "z": "rotated mesh minimum = table_top_z + 0.002m",
            },
            "manifest": copied,
            "provenance": upstream_manifests,
        }
        (staging / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged_outputs = {
            "mesh": artifact(
                staging / "reconstruction" / "object_mesh_scaled_final.obj",
                label="reconstruction/object_mesh_scaled_final.obj",
            ),
            "world_fused": artifact(
                staging / "reconstruction" / "world_fused.npz",
                label="reconstruction/world_fused.npz",
            ),
            "replay_world": artifact(
                staging / "retarget" / "replay_world.npz",
                label="retarget/replay_world.npz",
            ),
            "meta": artifact(staging / "meta.json", label="meta.json"),
        }
        finish_manifest(
            running_manifest,
            status="ready",
            outputs=staged_outputs,
            validation={
                "required_reconstruction_files_present": True,
                "required_retarget_files_present": True,
                "atomic_destination": True,
            },
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"[static-stage] {src_clip} -> {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("name")
    parser.add_argument("--reconstruction", required=True)
    parser.add_argument("--retarget", required=True)
    parser.add_argument("--src-clip", required=True)
    parser.add_argument("--mass-kg", type=float, default=0.2)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stage_static(
        args.dataset, args.name, args.reconstruction, args.retarget, args.src_clip,
        mass_kg=args.mass_kg, friction=args.friction, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
