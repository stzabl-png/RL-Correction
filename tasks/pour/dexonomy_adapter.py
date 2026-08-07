"""Strict Task-5 adapter for the Step-3 SharpaWave Dexonomy pipeline.

The Step-3 branch contains independent left/right hand models, but its one-shot
shell script does not forward the left-hand flags to trajectory export or the
Dexonomy Isaac batch.  Task 5 therefore uses the shell script only for grasp
synthesis (``SKIP_ISAAC=1``), re-exports with an explicit side, and performs its
authoritative Gate-2 screen in the RL-Correction task physics.

This module deliberately requires a template name.  Selecting the Dexonomy
template remains a human decision.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

import numpy as np


STEP3_COMMIT = "a09b5ab"
HAND_NAME = {"left": "sharpa_wave_left", "right": "sharpa_wave"}


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=None if env is None else dict(env),
        check=True,
        text=True,
    )


def check_step3_checkout(root: str | Path, *, require_commit: bool = True) -> str:
    """Validate the side-specific files that Task 5 relies on."""
    root = Path(root).resolve()
    required = [
        "tools/grasp_pipeline.sh",
        "tools/export_isaac_traj.py",
        "assets/hand/sharpa_wave/right.xml",
        "assets/hand/sharpa_wave_left/left.xml",
        "dexonomy/config/hand/sharpa_wave.yaml",
        "dexonomy/config/hand/sharpa_wave_left.yaml",
        "isaac/assets/robots/hands/sharpa_wave/sharpa_wave_right.yml",
        "isaac/assets/robots/hands/sharpa_wave/sharpa_wave_left.yml",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Step-3 Dexonomy checkout missing: {missing}")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_commit and not revision.startswith(STEP3_COMMIT):
        raise RuntimeError(
            f"Dexonomy must be pinned to Step3 {STEP3_COMMIT}; got {revision}"
        )
    return revision


def synthesize(
    root: str | Path,
    *,
    mesh: str | Path,
    object_id: str,
    side: str,
    template: str,
    epoch: int = 50,
) -> Path:
    """Run the human-selected Dexonomy template and side-correct export."""
    if side not in HAND_NAME:
        raise ValueError("side must be left or right")
    if not template.strip():
        raise ValueError("template is required; Task 5 has no automatic fallback")
    if epoch <= 0:
        raise ValueError("epoch must be positive")

    root = Path(root).resolve()
    mesh = Path(mesh).resolve()
    check_step3_checkout(root)
    if not mesh.is_file():
        raise FileNotFoundError(mesh)

    env = os.environ.copy()
    env.update(
        {
            "HAND": HAND_NAME[side],
            "TMPL": template,
            "EPOCH": str(epoch),
            # The authoritative screen uses RL-Correction physics below.
            "SKIP_ISAAC": "1",
        }
    )
    _run(
        ["bash", "tools/grasp_pipeline.sh", str(mesh), object_id],
        cwd=root,
        env=env,
    )

    experiment = root / "output" / f"{object_id}_{HAND_NAME[side]}"
    # The Step-3 wrapper currently exports right-prefixed joint names even for
    # HAND=sharpa_wave_left.  Re-exporting makes the side explicit and auditable.
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        python = Path(os.environ.get("DEXONOMY_PYTHON", "python"))
    _run(
        [
            str(python),
            "tools/export_isaac_traj.py",
            "--exp-dir",
            str(experiment),
            "--data",
            "grasp_data",
            "--side",
            side,
        ],
        cwd=root,
        env=os.environ.copy(),
    )
    return experiment


def _quat_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("invalid quaternion")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    # Stable eigen solution for a rotation matrix, returned as wxyz.
    K = np.asarray(
        [
            [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], matrix[1, 0] + matrix[0, 1], matrix[2, 0] + matrix[0, 2], matrix[1, 2] - matrix[2, 1]],
            [matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 0] - matrix[0, 2]],
            [matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], matrix[0, 1] - matrix[1, 0]],
            [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2], matrix[0, 1] - matrix[1, 0], matrix[0, 0] + matrix[1, 1] + matrix[2, 2]],
        ],
        dtype=np.float64,
    ) / 3.0
    values, vectors = np.linalg.eigh(K)
    xyzw = vectors[:, np.argmax(values)]
    if xyzw[3] < 0.0:
        xyzw = -xyzw
    return xyzw[[3, 0, 1, 2]]


def _to_input_frame(rows: np.ndarray, input_from_canonical: np.ndarray) -> np.ndarray:
    converted = []
    for row in np.atleast_2d(np.asarray(rows, dtype=np.float64)):
        if row.shape != (29,):
            raise ValueError(f"Dexonomy qpos must have 29 values, got {row.shape}")
        position = input_from_canonical @ row[:3]
        rotation = input_from_canonical @ _quat_to_matrix(row[3:7])
        converted.append(np.concatenate((position, _matrix_to_quat(rotation), row[7:])))
    return np.stack(converted)


def convert_candidate(
    grasp_npy: str | Path,
    info_json: str | Path,
    output_npz: str | Path,
    *,
    side: str,
) -> Path:
    """Convert one side-verified Dexonomy candidate into a Task-5 prior."""
    if side not in HAND_NAME:
        raise ValueError("side must be left or right")
    grasp_npy = Path(grasp_npy).resolve()
    info_json = Path(info_json).resolve()
    output_npz = Path(output_npz).resolve()

    candidate = np.load(grasp_npy, allow_pickle=True).item()
    generated_hand = str(candidate.get("hand_name", ""))
    if generated_hand != HAND_NAME[side]:
        raise ValueError(
            f"candidate hand_name={generated_hand!r}; expected {HAND_NAME[side]!r}"
        )
    info = json.loads(info_json.read_text(encoding="utf-8"))
    canonical_from_input = _quat_to_matrix(
        np.asarray(info["canonical_from_input_rot_wxyz"], dtype=np.float64)
    )
    input_from_canonical = canonical_from_input.T

    contacts = candidate.get("ho_c")
    if not isinstance(contacts, dict) or "pos" not in contacts or "normal" not in contacts:
        raise KeyError("Dexonomy candidate is missing ho_c.pos/normal")
    contact_position = np.asarray(contacts["pos"], dtype=np.float64) @ canonical_from_input
    contact_normal = np.asarray(contacts["normal"], dtype=np.float64) @ canonical_from_input
    if contact_position.ndim != 2 or contact_position.shape[1] != 3 or len(contact_position) == 0:
        raise ValueError("Dexonomy contact positions must be non-empty [N,3]")

    source = str(grasp_npy).encode("utf-8")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        grasp=_to_input_frame(candidate["grasp_qpos"], input_from_canonical)[0],
        squeeze=_to_input_frame(candidate["squeeze_qpos"], input_from_canonical)[0],
        pregrasp=_to_input_frame(candidate["pregrasp_qpos"], input_from_canonical),
        contact_pos=contact_position,
        contact_normal=contact_normal,
        contact_centroid=contact_position.mean(axis=0),
        canon_rot=np.asarray(info["canonical_from_input_rot_wxyz"], dtype=np.float64),
        hand_side=np.frombuffer(side.encode("utf-8"), dtype=np.uint8),
        source=np.frombuffer(source, dtype=np.uint8),
    )
    return output_npz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check")
    check.add_argument("--dexonomy-root", required=True)
    check.add_argument("--allow-newer-commit", action="store_true")

    generate = commands.add_parser("generate")
    generate.add_argument("--dexonomy-root", required=True)
    generate.add_argument("--mesh", required=True)
    generate.add_argument("--object-id", required=True)
    generate.add_argument("--side", required=True, choices=tuple(HAND_NAME))
    generate.add_argument("--template", required=True)
    generate.add_argument("--epoch", type=int, default=50)

    convert = commands.add_parser("convert")
    convert.add_argument("--grasp-npy", required=True)
    convert.add_argument("--info-json", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--side", required=True, choices=tuple(HAND_NAME))

    args = parser.parse_args()
    if args.command == "check":
        revision = check_step3_checkout(
            args.dexonomy_root, require_commit=not args.allow_newer_commit
        )
        print(f"[pour-dexonomy] contract ok revision={revision}")
    elif args.command == "generate":
        experiment = synthesize(
            args.dexonomy_root,
            mesh=args.mesh,
            object_id=args.object_id,
            side=args.side,
            template=args.template,
            epoch=args.epoch,
        )
        print(f"[pour-dexonomy] generated {experiment}")
    else:
        output = convert_candidate(
            args.grasp_npy, args.info_json, args.output, side=args.side
        )
        print(f"[pour-dexonomy] converted {output}")


if __name__ == "__main__":
    main()
