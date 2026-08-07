"""Apply an explicit human Dexonomy selection after Gate-1/2 screening.

This command never selects a fallback.  Each report is a small JSON record
created after the human preview decision and independent Isaac screening:

``schema_version, hand_side, object, template, approved_by, prior_sha256,
gate1_pass, gate2_pass, pads_star, q_star``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tasks.pour.reference import PourReference
from tasks.pour.scene import PourSceneManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_prior(path: Path, side: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        required = {"grasp", "pregrasp", "contact_centroid", "hand_side"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{side} prior missing {sorted(missing)}")
        grasp = np.asarray(data["grasp"])
        pregrasp = np.atleast_2d(data["pregrasp"])
        contact = np.asarray(data["contact_centroid"])
        if grasp.shape != (29,) or pregrasp.shape[1:] != (29,):
            raise ValueError(f"{side} prior grasp/pregrasp must contain 29 values")
        if contact.shape != (3,) or not np.isfinite(contact).all():
            raise ValueError(f"{side} prior contact_centroid must be finite [3]")
        encoded_side = np.asarray(data["hand_side"], dtype=np.uint8).tobytes().decode(
            "utf-8"
        )
        if encoded_side != side:
            raise ValueError(f"{side} prior declares hand_side={encoded_side!r}")


def _validate_report(
    path: Path,
    prior: Path,
    side: str,
    object_name: str,
    *,
    approved_by: str | None = None,
) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError(f"{side} report schema_version must be 1")
    expected = {
        "hand_side": side,
        "object": object_name,
        "prior_sha256": _sha256(prior),
    }
    for name, target in expected.items():
        if value.get(name) != target:
            raise ValueError(f"{side} report {name} does not match prior ({target})")
    if approved_by is not None:
        value["approved_by"] = approved_by
    for name in ("template", "approved_by"):
        if not str(value.get(name, "")).strip():
            raise ValueError(f"{side} report requires {name}")
    if value.get("gate1_pass") is not True or value.get("gate2_pass") is not True:
        raise ValueError(f"{side} prior did not pass Gate-1 and Gate-2")
    if value.get("stable_lift_pass") is not True:
        raise ValueError(f"{side} Gate-2 is missing a stable off-table lift")
    if "joint_gate_pass" in value and value["joint_gate_pass"] is not True:
        raise ValueError(f"{side} prior has no single stable setting passing Gate-2")
    if float(value.get("pads_star", 0.0)) < 4.0 or float(value.get("q_star", 0.0)) <= 0.0:
        raise ValueError(f"{side} Gate-2 requires pads_star>=4 and q_star>0")
    if float(value.get("object_lift_m", 0.0)) < 0.01:
        raise ValueError(f"{side} Gate-2 requires object_lift_m>=0.01")
    return value


def approve_grasps(
    scene_path: str | Path,
    *,
    left_prior_path: str | Path,
    right_prior_path: str | Path,
    left_report_path: str | Path,
    right_report_path: str | Path,
    approved_by: str | None = None,
) -> PourSceneManifest:
    scene_source = Path(scene_path)
    manifest = PourSceneManifest.load(scene_source, resolve_relative=False)
    if manifest.status != "pending_grasp_approval":
        raise RuntimeError(
            f"scene must be pending_grasp_approval, got {manifest.status}"
        )
    reference_path = Path(manifest.reference_npz)
    if not reference_path.is_absolute():
        reference_path = scene_source.parent / reference_path
    if not PourReference.load(reference_path).training_ready:
        raise RuntimeError("reference is not training-ready")

    priors = {
        "left": Path(left_prior_path).resolve(),
        "right": Path(right_prior_path).resolve(),
    }
    reports = {
        "left": Path(left_report_path).resolve(),
        "right": Path(right_report_path).resolve(),
    }
    approval = {}
    for side, object_name in (("left", "cup"), ("right", "bottle")):
        if not priors[side].is_file() or not reports[side].is_file():
            raise FileNotFoundError(f"{side} prior/report is missing")
        _validate_prior(priors[side], side)
        approval[side] = _validate_report(
            reports[side],
            priors[side],
            side,
            object_name,
            approved_by=approved_by,
        )

    manifest.left_grasp_prior = str(priors["left"])
    manifest.right_grasp_prior = str(priors["right"])
    manifest.grasp_approval = approval
    manifest.status = "ready"
    # Resolve all paths once for the strict asset gate.  The saved manifest can
    # retain the reconstruction adapter's path convention.
    checked = PourSceneManifest.load(scene_source, resolve_relative=True)
    checked.left_grasp_prior = manifest.left_grasp_prior
    checked.right_grasp_prior = manifest.right_grasp_prior
    checked.grasp_approval = approval
    checked.status = "ready"
    checked.validate(require_assets=True)
    manifest.save(scene_source)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--left-prior", required=True)
    parser.add_argument("--right-prior", required=True)
    parser.add_argument("--left-report", required=True)
    parser.add_argument("--right-report", required=True)
    parser.add_argument(
        "--approved-by",
        help="explicit human reviewer; required when screen reports do not embed approved_by",
    )
    args = parser.parse_args()
    manifest = approve_grasps(
        args.scene,
        left_prior_path=args.left_prior,
        right_prior_path=args.right_prior,
        left_report_path=args.left_report,
        right_report_path=args.right_report,
        approved_by=args.approved_by,
    )
    print(
        f"[pour-approve] demo={manifest.demo_id} status={manifest.status} "
        f"left={manifest.grasp_approval['left']['template']} "
        f"right={manifest.grasp_approval['right']['template']}"
    )


if __name__ == "__main__":
    main()
