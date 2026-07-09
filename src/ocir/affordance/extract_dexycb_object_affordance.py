#!/usr/bin/env python3
"""Extract hand-contact object affordance point clouds from selected DexYCB clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocir.dexycb.mano_model import load_mano_betas, load_mano_model, pose_m_to_vertices_and_joints


DEFAULT_DATA_ROOT = Path("/data/users/hangkes2/OCIR")
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "processed_data/dex_ycb/manifests/selected_5_sequences.json"
DEFAULT_THRESHOLD_M = 0.005


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_AFFORDANCE {message}", flush=True)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing DexYCB manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_sequence_mano(manifest: dict, sequence: dict):
    side = str(sequence.get("mano_side", "right")).lower()
    model_key = "MANO_LEFT.pkl" if side == "left" else "MANO_RIGHT.pkl"
    model = load_mano_model(manifest["mano_models"][model_key], side=side)

    calib = sequence.get("mano_calib")
    calib_name = calib[0] if isinstance(calib, list) else calib
    if not calib_name:
        raise ValueError(f"sequence {sequence['sequence_id']} has no MANO calibration entry")
    betas_path = Path(manifest["selected_root"]) / "calibration" / f"mano_{calib_name}" / "mano.yml"
    betas = load_mano_betas(betas_path)
    return model, betas, {"side": side, "model_path": manifest["mano_models"][model_key], "betas_path": str(betas_path)}


def read_label(sequence: dict, frame_id: int) -> dict[str, np.ndarray]:
    path = Path(sequence["path"]) / sequence["canonical_camera"] / f"labels_{frame_id:06d}.npz"
    data = np.load(path)
    return {key: data[key] for key in data.files}


def valid_mano_frame(label: dict[str, np.ndarray]) -> bool:
    joints = np.asarray(label["joint_3d"][0], dtype=float)
    pose_m = np.asarray(label["pose_m"], dtype=float)
    if not np.isfinite(joints).all() or np.any(joints < -0.5):
        return False
    if not np.isfinite(pose_m).all() or np.allclose(pose_m, 0.0):
        return False
    return True


def hand_camera_to_object(hand_vertices_cam: np.ndarray, pose_y: np.ndarray) -> np.ndarray:
    """Transform camera-frame hand vertices into the object's canonical model frame."""

    pose_y = np.asarray(pose_y, dtype=float)
    rot = pose_y[:3, :3]
    pos = pose_y[:3, 3]
    return (np.asarray(hand_vertices_cam, dtype=float) - pos[None, :]) @ rot


def nearest_hand_distances(object_points: np.ndarray, hand_points_object: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = np.empty((object_points.shape[0],), dtype=float)
    for start in range(0, object_points.shape[0], batch_size):
        stop = min(start + batch_size, object_points.shape[0])
        diff = object_points[start:stop, None, :] - hand_points_object[None, :, :]
        out[start:stop] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return out


def heatmap_colors(values: np.ndarray) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl

    values = np.asarray(values, dtype=float)
    rgba = mpl.colormaps["inferno"](np.clip(values, 0.0, 1.0))
    return np.asarray(np.round(rgba[:, :3] * 255.0), dtype=np.uint8)


def write_ascii_ply(path: Path, points: np.ndarray, values: np.ndarray) -> None:
    colors = heatmap_colors(values)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float affordance\n")
        f.write("end_header\n")
        for xyz, rgb, value in zip(points, colors, values, strict=True):
            f.write(
                f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} {float(value):.9g}\n"
            )


def set_3d_axes_equal(ax, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = float(np.max(points.max(axis=0) - points.min(axis=0)) / 2.0)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def write_heatmap_png(path: Path, points: np.ndarray, values: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    fig = plt.figure(figsize=(7.5, 6.5), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        s=12,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        linewidths=0,
        depthshade=False,
    )
    contacted = values > 0.0
    if np.any(contacted):
        ax.scatter(
            points[contacted, 0],
            points[contacted, 1],
            points[contacted, 2],
            c=values[contacted],
            s=22,
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.15,
            edgecolors="white",
            depthshade=False,
        )
    set_3d_axes_equal(ax, points)
    ax.view_init(elev=24, azim=-54)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.08, label="normalized contact count")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sequence_by_id(manifest: dict, sequence_id: str) -> dict:
    for sequence in manifest["sequences"]:
        if sequence["sequence_id"] == sequence_id:
            return sequence
    raise KeyError(f"sequence_id not found in manifest: {sequence_id}")


def extract_sequence_affordance(manifest: dict, sequence: dict, threshold_m: float, force: bool) -> dict:
    seq_id = sequence["sequence_id"]
    seq_path = Path(sequence["path"])
    out_dir = seq_path / "affordance"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_idx = int(sequence.get("ycb_grasp_ind", 0) or 0)
    target_info = sequence["ycb_models"][target_idx]
    object_name = target_info["name"]
    object_points_path = Path(target_info["mesh"]).parent / "points.xyz"
    object_points = np.loadtxt(object_points_path, dtype=float).reshape(-1, 3)

    stem = f"{seq_id}_{object_name}"
    counts_path = out_dir / f"{stem}_contact_counts.npz"
    heatmap_path = out_dir / f"{stem}_contact_heatmap.npz"
    ply_path = out_dir / f"{stem}_contact_heatmap.ply"
    png_path = out_dir / f"{stem}_contact_heatmap.png"
    summary_path = out_dir / "summary.json"
    if counts_path.exists() and heatmap_path.exists() and ply_path.exists() and png_path.exists() and not force:
        log(f"{seq_id}: outputs exist; use --force to overwrite")
        return json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"sequence_id": seq_id, "skipped": True}

    mano_model, mano_betas, mano_report = load_sequence_mano(manifest, sequence)

    contact_count = np.zeros((object_points.shape[0],), dtype=np.int32)
    min_distance = np.full((object_points.shape[0],), np.inf, dtype=float)
    valid_frames = 0
    skipped_frames = 0
    frame_contact_counts = []

    for frame_id in sequence["frame_ids"]:
        label = read_label(sequence, int(frame_id))
        if not valid_mano_frame(label):
            skipped_frames += 1
            continue
        pose_y = np.asarray(label["pose_y"], dtype=float)
        if target_idx >= pose_y.shape[0] or not np.isfinite(pose_y[target_idx]).all():
            skipped_frames += 1
            continue
        hand_vertices_cam, _ = pose_m_to_vertices_and_joints(mano_model, label["pose_m"][0], mano_betas)
        hand_vertices_object = hand_camera_to_object(hand_vertices_cam, pose_y[target_idx])
        distances = nearest_hand_distances(object_points, hand_vertices_object)
        contacted = distances <= threshold_m
        contact_count += contacted.astype(np.int32)
        min_distance = np.minimum(min_distance, distances)
        valid_frames += 1
        frame_contact_counts.append(int(np.count_nonzero(contacted)))

    max_count = int(contact_count.max()) if contact_count.size else 0
    heatmap = contact_count.astype(float) / float(max_count) if max_count > 0 else np.zeros_like(contact_count, dtype=float)

    shared = {
        "sequence_id": seq_id,
        "subject": sequence.get("subject"),
        "canonical_camera": sequence.get("canonical_camera"),
        "object_index": target_idx,
        "object_id": target_info["id"],
        "object_name": object_name,
        "threshold_m": float(threshold_m),
        "object_points_path": str(object_points_path),
    }
    np.savez_compressed(
        counts_path,
        points_object_frame=object_points,
        contact_count=contact_count,
        min_distance_m=min_distance,
        **shared,
    )
    np.savez_compressed(
        heatmap_path,
        points_object_frame=object_points,
        heatmap=heatmap,
        contact_count=contact_count,
        min_distance_m=min_distance,
        **shared,
    )
    write_ascii_ply(ply_path, object_points, heatmap)
    write_heatmap_png(png_path, object_points, heatmap, f"{seq_id} {object_name}")

    summary = {
        **shared,
        "num_object_surface_points": int(object_points.shape[0]),
        "num_total_frames": int(len(sequence["frame_ids"])),
        "num_valid_mano_frames": int(valid_frames),
        "num_skipped_frames": int(skipped_frames),
        "num_contact_points": int(np.count_nonzero(contact_count)),
        "max_contact_count": max_count,
        "mean_contact_count_over_contact_points": float(contact_count[contact_count > 0].mean()) if np.any(contact_count > 0) else 0.0,
        "mean_frame_contact_points": float(np.mean(frame_contact_counts)) if frame_contact_counts else 0.0,
        "mano": mano_report,
        "outputs": {
            "contact_counts_npz": str(counts_path),
            "contact_heatmap_npz": str(heatmap_path),
            "contact_heatmap_ply": str(ply_path),
            "contact_heatmap_png": str(png_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(
        f"{seq_id}: {object_name} valid_frames={valid_frames} "
        f"contact_points={summary['num_contact_points']}/{object_points.shape[0]} max_count={max_count}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sequence-id", action="append", default=None, help="DexYCB sequence id. Omit to process all manifest sequences.")
    parser.add_argument("--contact-threshold", type=float, default=DEFAULT_THRESHOLD_M, help="Hand/object surface contact threshold in meters.")
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest.expanduser())
    if args.sequence_id:
        sequences = [sequence_by_id(manifest, seq_id) for seq_id in args.sequence_id]
    else:
        sequences = list(manifest["sequences"])
    summaries = [extract_sequence_affordance(manifest, sequence, float(args.contact_threshold), bool(args.force)) for sequence in sequences]

    processed_root = Path(manifest.get("processed_dex_ycb_root", Path(manifest["selected_root"]).parents[0]))
    aggregate_path = processed_root / "affordance_summary.json"
    aggregate_path.write_text(json.dumps({"threshold_m": float(args.contact_threshold), "sequences": summaries}, indent=2) + "\n", encoding="utf-8")
    log(f"wrote aggregate summary {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
