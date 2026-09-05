#!/usr/bin/env python3
"""Generate five small, fixed Sweep2 cube-start variants without changing training.

This tool only writes a JSON manifest and a compact NPZ containing candidate cube
starts.  It never imports Isaac, edits the reference, changes SweepEnv defaults,
or launches a rollout/training process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_CENTER = np.array(
    [-0.0259767957, -0.1788897067, 0.8830000162], dtype=np.float64
)
DEFAULT_COUNT = 5
DEFAULT_PRELUDE_STEPS = 80
COLOR_NAMES = np.asarray(["red", "blue", "green", "yellow", "purple"])
COLORS_RGB = np.asarray(
    [(0.85, 0.12, 0.08), (0.08, 0.25, 0.90), (0.08, 0.70, 0.20),
     (0.90, 0.75, 0.08), (0.55, 0.12, 0.75)], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quat_apply_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate one xyz vector by a batch of wxyz quaternions."""
    quat = np.asarray(quat, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    vx, vy, vz = vec
    return np.stack(
        [
            (1 - 2 * (y * y + z * z)) * vx
            + 2 * (x * y - z * w) * vy
            + 2 * (x * z + y * w) * vz,
            2 * (x * y + z * w) * vx
            + (1 - 2 * (x * x + z * z)) * vy
            + 2 * (y * z - x * w) * vz,
            2 * (x * z - y * w) * vx
            + 2 * (y * z + x * w) * vy
            + (1 - 2 * (x * x + y * y)) * vz,
        ],
        axis=-1,
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=here / "sweep2_reference_v1.npz",
    )
    parser.add_argument("--out-dir", type=Path, default=here)
    parser.add_argument("--name", default="sweep2_cube_cross5_v1")
    parser.add_argument("--lateral-mm", type=float, default=1.0)
    parser.add_argument("--longitudinal-mm", type=float, default=5.0)
    parser.add_argument("--prelude-steps", type=int, default=DEFAULT_PRELUDE_STEPS)
    parser.add_argument("--min-prelude-clearance-mm", type=float, default=30.0)
    parser.add_argument("--min-contact-distance-mm", type=float, default=3.0)
    parser.add_argument("--max-contact-distance-mm", type=float, default=15.0)
    parser.add_argument("--max-contact-speed-mps", type=float, default=0.12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = args.reference.resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if not (0.0 < args.lateral_mm <= 1.0):
        raise ValueError("lateral displacement must be in (0, 1] mm")
    if not (0.0 < args.longitudinal_mm <= 5.0):
        raise ValueError("longitudinal displacement must be in (0, 5] mm")

    with np.load(reference, allow_pickle=False) as ref:
        required = {
            "obj_pos_1",
            "obj_quat_1",
            "obj_quat_0",
            "brush_contact_local",
            "control_hz",
        }
        missing = required.difference(ref.files)
        if missing:
            raise KeyError(f"reference is missing {sorted(missing)}")
        broom_pos = np.asarray(ref["obj_pos_1"], dtype=np.float64)
        broom_quat = np.asarray(ref["obj_quat_1"], dtype=np.float64)
        contact_local = np.asarray(ref["brush_contact_local"], dtype=np.float64)
        control_hz = float(ref["control_hz"])
        pan_quat0 = np.asarray(ref["obj_quat_0"][0], dtype=np.float64)

    if broom_pos.shape != broom_quat.shape[:1] + (3,):
        raise ValueError("broom pose arrays have incompatible shapes")
    if not (0 < args.prelude_steps < len(broom_pos)):
        raise ValueError("prelude steps must lie inside the reference")

    contact_world = broom_pos + quat_apply_wxyz(broom_quat, contact_local)
    contact_speed = np.concatenate(
        [
            np.linalg.norm(np.diff(contact_world, axis=0), axis=1) * control_hz,
            np.zeros(1, dtype=np.float64),
        ]
    )

    lateral = quat_apply_wxyz(pan_quat0[None, :], np.array([1.0, 0.0, 0.0]))[0]
    longitudinal = quat_apply_wxyz(pan_quat0[None, :], np.array([0.0, 0.0, 1.0]))[0]
    lateral[2] = 0.0; longitudinal[2] = 0.0
    lateral /= np.linalg.norm(lateral); longitudinal /= np.linalg.norm(longitudinal)
    lateral *= args.lateral_mm / 1000.0
    longitudinal *= args.longitudinal_mm / 1000.0
    labels = np.asarray(["center", "left", "right", "front", "back"])
    offsets = np.stack([np.zeros(3), -lateral, lateral, longitudinal, -longitudinal])
    positions = DEFAULT_CENTER[None, :] + offsets

    prelude_min_m = args.min_prelude_clearance_mm / 1000.0
    contact_min_m = args.min_contact_distance_mm / 1000.0
    contact_max_m = args.max_contact_distance_mm / 1000.0
    checks = []
    for index, (label, offset, position) in enumerate(
        zip(labels, offsets, positions, strict=True)
    ):
        distance = np.linalg.norm(contact_world - position[None, :], axis=1)
        closest_row = int(np.argmin(distance))
        prelude_row = int(np.argmin(distance[: args.prelude_steps]))
        row_lo = max(0, closest_row - 3)
        row_hi = min(len(contact_speed), closest_row + 4)
        local_speed = float(np.max(contact_speed[row_lo:row_hi]))
        item = {
            "variant_id": f"cube_{label}",
            "color_name": str(COLOR_NAMES[index]),
            "color_rgb": [float(v) for v in COLORS_RGB[index]],
            "semantic_offset": str(label),
            "offset_world_xy_mm": [float(offset[0] * 1000), float(offset[1] * 1000)],
            "cube_start_world_m": [float(value) for value in position],
            "prelude_closest_row": prelude_row,
            "prelude_min_contact_point_distance_mm": float(distance[prelude_row] * 1000),
            "trajectory_closest_row": closest_row,
            "trajectory_min_contact_point_distance_mm": float(distance[closest_row] * 1000),
            "contact_window_max_reference_speed_mps": local_speed,
        }
        failures = []
        if abs(position[2] - DEFAULT_CENTER[2]) > 1e-12:
            failures.append("world z changed")
        if distance[prelude_row] < prelude_min_m:
            failures.append("broom contact point approaches cube during scripted prelude")
        if closest_row < args.prelude_steps:
            failures.append("closest broom approach occurs before policy activation")
        if not (contact_min_m <= distance[closest_row] <= contact_max_m):
            failures.append("closest broom approach is outside the approved contact band")
        if local_speed > args.max_contact_speed_mps:
            failures.append("reference broom speed near contact is too high")
        item["static_gate"] = "PASS" if not failures else "FAIL"
        item["failures"] = failures
        checks.append(item)

    failed = [item for item in checks if item["static_gate"] != "PASS"]
    if failed:
        detail = "\n".join(
            f"{item['variant_id']}: {', '.join(item['failures'])}" for item in failed
        )
        raise RuntimeError(f"candidate generation rejected:\n{detail}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.name}.json"
    npz_path = out_dir / f"{args.name}.npz"
    if not args.force and (json_path.exists() or npz_path.exists()):
        raise FileExistsError("output exists; pass --force only after reviewing the manifest")

    payload = {
        "schema": 1,
        "purpose": "Sweep2 center plus four pan-local cube-start variants",
        "source_reference": str(reference),
        "source_reference_sha256": sha256(reference),
        "center_world_m": [float(value) for value in DEFAULT_CENTER],
        "lateral_mm": float(args.lateral_mm),
        "longitudinal_mm": float(args.longitudinal_mm),
        "count": DEFAULT_COUNT,
        "axes": {"lateral_world_xy": lateral.tolist(),
                 "longitudinal_world_xy": longitudinal.tolist()},
        "plane": "pan-local lateral/longitudinal projected to world XY; Z unchanged",
        "reference_modified": False,
        "training_launched": False,
        "static_gate_only": True,
        "physics_validation_required": True,
        "thresholds": {
            "prelude_steps": int(args.prelude_steps),
            "min_prelude_contact_point_distance_mm": float(
                args.min_prelude_clearance_mm
            ),
            "contact_distance_band_mm": [
                float(args.min_contact_distance_mm),
                float(args.max_contact_distance_mm),
            ],
            "max_contact_window_reference_speed_mps": float(
                args.max_contact_speed_mps
            ),
            "max_lateral_mm": 1.0,
            "max_longitudinal_mm": 5.0,
        },
        "variants": checks,
    }
    np.savez_compressed(
        npz_path,
        variant_ids=np.asarray([item["variant_id"] for item in checks]),
        center_world_m=DEFAULT_CENTER.astype(np.float32),
        offsets_world_m=offsets.astype(np.float32),
        cube_start_world_m=positions.astype(np.float32),
        semantic_offsets=labels,
        lateral_axis_world=lateral.astype(np.float32),
        longitudinal_axis_world=longitudinal.astype(np.float32),
        color_names=COLOR_NAMES,
        colors_rgb=COLORS_RGB,
        lateral_m=np.asarray(args.lateral_mm / 1000.0, dtype=np.float32),
        longitudinal_m=np.asarray(args.longitudinal_mm / 1000.0, dtype=np.float32),
        source_reference_sha256=np.asarray(payload["source_reference_sha256"]),
    )
    payload["npz"] = str(npz_path)
    payload["npz_sha256"] = sha256(npz_path)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"PASS: wrote {json_path}")
    print(f"PASS: wrote {npz_path}")
    for item in checks:
        print(
            f"{item['variant_id']} "
            f"xy_mm={item['offset_world_xy_mm']} "
            f"prelude_clearance={item['prelude_min_contact_point_distance_mm']:.2f}mm "
            f"contact_row={item['trajectory_closest_row']} "
            f"contact_distance={item['trajectory_min_contact_point_distance_mm']:.2f}mm "
            f"speed={item['contact_window_max_reference_speed_mps']:.3f}m/s"
        )


if __name__ == "__main__":
    main()
