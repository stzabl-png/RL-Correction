#!/usr/bin/env python3
"""Prepare a small DexYCB subset for OCIR replay and simulation work."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import shutil
import tarfile
import time
import zipfile

import yaml


DEFAULT_DATA_ROOT = Path("/data/users/hangkes2/OCIR")
DEFAULT_SUBJECT = "20200709-subject-01"
DEFAULT_CAMERA = "932122060861"


YCB_ID_TO_MODEL = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


@dataclass
class SequenceStats:
    name: str
    has_meta: bool = False
    cameras: set[str] = field(default_factory=set)
    label_frames: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    color_frames: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    depth_frames: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))

    def valid_cameras(self) -> list[str]:
        out = []
        for camera in sorted(self.cameras):
            if self.label_frames[camera] and self.color_frames[camera] and self.depth_frames[camera]:
                out.append(camera)
        return out

    def is_valid(self) -> bool:
        return self.has_meta and len(self.valid_cameras()) >= 8

    def min_label_count(self) -> int:
        valid = self.valid_cameras()
        if not valid:
            return 0
        return min(len(self.label_frames[camera]) for camera in valid)

    def to_manifest(self, selected_root: Path, subject: str, preferred_camera: str) -> dict:
        valid = self.valid_cameras()
        camera = preferred_camera if preferred_camera in valid else valid[0]
        frames = sorted(self.label_frames[camera])
        meta_path = selected_root / subject / self.name / "meta.yml"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        ycb_ids = [int(x) for x in meta.get("ycb_ids", [])]
        return {
            "sequence_id": self.name,
            "subject": subject,
            "path": str(selected_root / subject / self.name),
            "meta": str(meta_path),
            "valid_cameras": valid,
            "canonical_camera": camera,
            "frame_ids": frames,
            "num_frames": len(frames),
            "mano_side": meta.get("mano_sides", [meta.get("mano_side", "right")])[0]
            if isinstance(meta.get("mano_sides", None), list)
            else meta.get("mano_side", "right"),
            "mano_calib": meta.get("mano_calib", None),
            "ycb_ids": ycb_ids,
            "ycb_grasp_ind": meta.get("ycb_grasp_ind", None),
            "ycb_models": [
                {
                    "id": ycb_id,
                    "name": YCB_ID_TO_MODEL.get(ycb_id, f"unknown_{ycb_id:03d}"),
                    "mesh": str(selected_root / "models" / YCB_ID_TO_MODEL.get(ycb_id, "") / "textured_simple.obj"),
                }
                for ycb_id in ycb_ids
            ],
        }


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_DEXYCB_PREP {message}", flush=True)


def frame_id_from_name(name: str, prefix: str, suffix: str) -> int | None:
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    stem = name[len(prefix) : -len(suffix)]
    try:
        return int(stem)
    except ValueError:
        return None


def scan_subject_archive(subject_archive: Path, subject: str) -> dict[str, SequenceStats]:
    stats: dict[str, SequenceStats] = {}
    log(f"scanning subject archive {subject_archive}")
    with tarfile.open(subject_archive, "r:gz") as tf:
        for idx, member in enumerate(tf):
            if idx > 0 and idx % 50000 == 0:
                log(f"scanned {idx} archive entries; found {len(stats)} sequences")
            parts = member.name.split("/")
            if len(parts) < 2 or parts[0] != subject or not parts[1]:
                continue
            seq = parts[1]
            seq_stats = stats.setdefault(seq, SequenceStats(name=seq))
            if len(parts) == 3 and parts[2] == "meta.yml":
                seq_stats.has_meta = True
                continue
            if len(parts) < 4:
                continue
            camera = parts[2]
            filename = parts[3]
            if not camera.isdigit():
                continue
            seq_stats.cameras.add(camera)
            frame_id = frame_id_from_name(filename, "labels_", ".npz")
            if frame_id is not None:
                seq_stats.label_frames[camera].add(frame_id)
                continue
            frame_id = frame_id_from_name(filename, "color_", ".jpg")
            if frame_id is not None:
                seq_stats.color_frames[camera].add(frame_id)
                continue
            frame_id = frame_id_from_name(filename, "aligned_depth_to_color_", ".png")
            if frame_id is not None:
                seq_stats.depth_frames[camera].add(frame_id)
    log(f"finished scan; found {len(stats)} sequences")
    return stats


def safe_extract_members(tar: tarfile.TarFile, members: list[tarfile.TarInfo], out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    for member in members:
        dest = (out_dir / member.name).resolve()
        if out_dir != dest and out_dir not in dest.parents:
            raise RuntimeError(f"refusing unsafe archive path: {member.name}")
    tar.extractall(out_dir, members=members)


def extract_tar_prefixes(archive: Path, out_dir: Path, prefixes: list[str], force: bool = False) -> None:
    prefixes = [prefix.rstrip("/") + "/" for prefix in prefixes]
    if not force and all((out_dir / prefix.rstrip("/")).exists() for prefix in prefixes):
        log(f"skip extraction for {archive}; requested prefixes already exist")
        return
    log(f"extracting {archive} prefixes={prefixes}")
    with tarfile.open(archive, "r:gz") as tf:
        batch: list[tarfile.TarInfo] = []
        for member in tf:
            if any(member.name == prefix.rstrip("/") or member.name.startswith(prefix) for prefix in prefixes):
                batch.append(member)
                if len(batch) >= 1000:
                    safe_extract_members(tf, batch, out_dir)
                    batch = []
        if batch:
            safe_extract_members(tf, batch, out_dir)


def extract_mano_models(zip_path: Path, mano_dir: Path, force: bool = False) -> dict:
    mano_dir.mkdir(parents=True, exist_ok=True)
    required = ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]
    if not force and all((mano_dir / name).exists() for name in required):
        log("skip MANO extraction; MANO_RIGHT.pkl and MANO_LEFT.pkl already exist")
    else:
        log(f"extracting MANO model files from {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            for basename in required:
                source = f"mano_v1_2/models/{basename}"
                if source not in names:
                    raise FileNotFoundError(f"{source} not found in {zip_path}")
                target = mano_dir / basename
                with zf.open(source) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return {name: str(mano_dir / name) for name in required}


def select_sequences(stats: dict[str, SequenceStats], count: int, seed: int) -> list[SequenceStats]:
    valid = [seq for seq in stats.values() if seq.is_valid()]
    if len(valid) < count:
        raise RuntimeError(f"only found {len(valid)} valid sequences; need {count}")
    # Prefer complete, longer label tracks, but keep random selection reproducible.
    max_min_labels = max(seq.min_label_count() for seq in valid)
    preferred = [seq for seq in valid if seq.min_label_count() >= max_min_labels - 2]
    pool = preferred if len(preferred) >= count else valid
    rng = random.Random(seed)
    selected = rng.sample(sorted(pool, key=lambda s: s.name), count)
    return sorted(selected, key=lambda s: s.name)


def validate_manifest(manifest: dict) -> list[str]:
    errors = []
    for seq in manifest["sequences"]:
        seq_path = Path(seq["path"])
        if not seq_path.exists():
            errors.append(f"missing sequence path: {seq_path}")
        if not Path(seq["meta"]).exists():
            errors.append(f"missing meta.yml: {seq['meta']}")
        if not seq["frame_ids"]:
            errors.append(f"no frame ids for sequence {seq['sequence_id']}")
        for model in seq["ycb_models"]:
            mesh = Path(model["mesh"])
            if not mesh.exists():
                errors.append(f"missing YCB mesh for sequence {seq['sequence_id']}: {mesh}")
    for side in ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]:
        if not Path(manifest["mano_models"][side]).exists():
            errors.append(f"missing MANO model: {manifest['mano_models'][side]}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preferred-camera", default=DEFAULT_CAMERA)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser()
    dex_raw_root = data_root / "raw_data" / "dex_ycb"
    dex_processed_root = data_root / "processed_data" / "dex_ycb"
    mano_dir = data_root / "raw_data" / "mano_models"
    archives = dex_raw_root / "archives"
    selected_root = dex_processed_root / "selected"
    manifests = dex_processed_root / "manifests"
    selected_root.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    subject_archive = archives / f"{args.subject}.tar.gz"
    calibration_archive = archives / "calibration.tar.gz"
    models_archive = archives / "models.tar.gz"
    mano_zip = mano_dir / "mano_v1_2.zip"
    for path in [subject_archive, calibration_archive, models_archive, mano_zip]:
        if not path.exists():
            raise FileNotFoundError(path)

    mano_models = extract_mano_models(mano_zip, mano_dir, force=args.force)
    extract_tar_prefixes(calibration_archive, selected_root, ["calibration"], force=args.force)
    extract_tar_prefixes(models_archive, selected_root, ["models"], force=args.force)

    stats = scan_subject_archive(subject_archive, args.subject)
    selected = select_sequences(stats, args.count, args.seed)
    log("selected sequences: " + ", ".join(seq.name for seq in selected))
    prefixes = [f"{args.subject}/{seq.name}" for seq in selected]
    extract_tar_prefixes(subject_archive, selected_root, prefixes, force=args.force)

    sequences = [
        seq.to_manifest(selected_root=selected_root, subject=args.subject, preferred_camera=args.preferred_camera)
        for seq in selected
    ]
    manifest = {
        "manifest_version": 1,
        "dataset": "dex_ycb",
        "data_root": str(data_root),
        "dex_ycb_root": str(dex_processed_root),
        "raw_dex_ycb_root": str(dex_raw_root),
        "processed_dex_ycb_root": str(dex_processed_root),
        "selected_root": str(selected_root),
        "subject": args.subject,
        "seed": args.seed,
        "count": args.count,
        "preferred_camera": args.preferred_camera,
        "archives": {
            "subject": str(subject_archive),
            "calibration": str(calibration_archive),
            "models": str(models_archive),
            "mano_zip": str(mano_zip),
        },
        "mano_models": mano_models,
        "sequences": sequences,
    }
    errors = validate_manifest(manifest)
    manifest["validation"] = {"ok": not errors, "errors": errors}
    manifest_path = manifests / "selected_5_sequences.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"wrote manifest {manifest_path}")
    if errors:
        for error in errors:
            log(f"validation error: {error}")
        return 1
    log("validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
