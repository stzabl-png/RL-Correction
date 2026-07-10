#!/usr/bin/env python3
"""Object affordance heatmap + hand-contact-frame detection for anchored BODex.

Generalizes the earlier standalone DexYCB affordance extractor: it consumes a
sequence directory's ``human_demo.npz`` (already in the object canonical
frame, see :mod:`ocir.grasp_synthesis.anchored_bodex.demo_data`) instead of
raw DexYCB labels, and it is *lazily cached*: :func:`ensure_affordance` loads
``<sequence_dir>/affordance.npz`` when present and otherwise computes and
saves it into the sequence directory, so later runs (and the seed generator)
reuse it.

``affordance.npz`` schema:

    points_object_frame  (N,3)  object surface points (from points.xyz / mesh)
    heatmap              (N,)   contact_count / max(contact_count) in [0, 1]
    contact_count        (N,)   int32, frames each point was within threshold
    min_distance_m       (N,)   min distance to any hand vertex over frames
    frame_ids            (T,)   int32, copied from the demo
    frame_contact_count  (T,)   int32, hand vertices within threshold per frame
                                (0 for invalid demo frames)
    frame_in_contact     (T,)   bool, frame_contact_count >= min_contact_vertices
    contact_threshold_m  ()     float scalar
    min_contact_vertices ()     int scalar

``frame_in_contact`` is the reusable set of demo frames the anchored seed
generator samples from.

Run as a module for batch precompute + debug PLY/PNG:

    python -m ocir.grasp_synthesis.anchored_bodex.affordance \
        --sequences-root /path/to/sequences [--sequence-id ID ...] [--force] [--debug-viz]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from ocir.grasp_synthesis.anchored_bodex.demo_data import (
    AFFORDANCE_FILENAME,
    HumanDemo,
    resolve_sequence_file,
)

DEFAULT_CONTACT_THRESHOLD_M = 0.005
DEFAULT_MIN_CONTACT_VERTICES = 20


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_ANCHORED_AFFORDANCE {message}", flush=True)


def nearest_hand_distances(object_points: np.ndarray, hand_points_object: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = np.empty((object_points.shape[0],), dtype=float)
    for start in range(0, object_points.shape[0], batch_size):
        stop = min(start + batch_size, object_points.shape[0])
        diff = object_points[start:stop, None, :] - hand_points_object[None, :, :]
        out[start:stop] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return out


@dataclass(frozen=True)
class Affordance:
    points_object_frame: np.ndarray  # (N, 3)
    heatmap: np.ndarray              # (N,)
    contact_count: np.ndarray        # (N,) int32
    min_distance_m: np.ndarray       # (N,)
    frame_ids: np.ndarray            # (T,) int32
    frame_contact_count: np.ndarray  # (T,) int32
    frame_in_contact: np.ndarray     # (T,) bool
    contact_threshold_m: float
    min_contact_vertices: int
    path: Path | None

    @property
    def contact_frame_indices(self) -> np.ndarray:
        """Indices (into the demo's frame arrays) where the hand touches the object."""

        return np.flatnonzero(self.frame_in_contact)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            points_object_frame=self.points_object_frame,
            heatmap=self.heatmap,
            contact_count=self.contact_count,
            min_distance_m=self.min_distance_m,
            frame_ids=self.frame_ids,
            frame_contact_count=self.frame_contact_count,
            frame_in_contact=self.frame_in_contact,
            contact_threshold_m=float(self.contact_threshold_m),
            min_contact_vertices=int(self.min_contact_vertices),
        )

    @classmethod
    def load(cls, path: Path) -> "Affordance":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                points_object_frame=np.asarray(data["points_object_frame"], dtype=np.float64),
                heatmap=np.asarray(data["heatmap"], dtype=np.float64),
                contact_count=np.asarray(data["contact_count"], dtype=np.int32),
                min_distance_m=np.asarray(data["min_distance_m"], dtype=np.float64),
                frame_ids=np.asarray(data["frame_ids"], dtype=np.int32),
                frame_contact_count=np.asarray(data["frame_contact_count"], dtype=np.int32),
                frame_in_contact=np.asarray(data["frame_in_contact"], dtype=bool),
                contact_threshold_m=float(data["contact_threshold_m"]),
                min_contact_vertices=int(data["min_contact_vertices"]),
                path=path,
            )


def compute_affordance(
    demo: HumanDemo,
    object_points: np.ndarray,
    *,
    contact_threshold_m: float = DEFAULT_CONTACT_THRESHOLD_M,
    min_contact_vertices: int = DEFAULT_MIN_CONTACT_VERTICES,
) -> Affordance:
    object_points = np.asarray(object_points, dtype=float).reshape(-1, 3)
    t = demo.num_frames
    contact_count = np.zeros((object_points.shape[0],), dtype=np.int32)
    min_distance = np.full((object_points.shape[0],), np.inf, dtype=float)
    frame_contact_count = np.zeros((t,), dtype=np.int32)

    for i in demo.valid_indices:
        hand_vertices = demo.hand_vertices_object[i]
        distances = nearest_hand_distances(object_points, hand_vertices)
        contacted = distances <= contact_threshold_m
        contact_count += contacted.astype(np.int32)
        min_distance = np.minimum(min_distance, distances)
        # Hand-side contact count: hand vertices within threshold of the object
        # (asymmetric to the object-side count above; a fingertip touching a
        # large flat face contacts few object points but many hand vertices).
        hand_to_object = np.sqrt(
            np.min(
                np.sum((hand_vertices[:, None, :] - object_points[None, :, :]) ** 2, axis=2),
                axis=1,
            )
        )
        frame_contact_count[i] = int(np.count_nonzero(hand_to_object <= contact_threshold_m))

    max_count = int(contact_count.max()) if contact_count.size else 0
    heatmap = contact_count.astype(float) / float(max_count) if max_count > 0 else np.zeros_like(contact_count, dtype=float)
    return Affordance(
        points_object_frame=object_points,
        heatmap=heatmap,
        contact_count=contact_count,
        min_distance_m=min_distance,
        frame_ids=demo.frame_ids.copy(),
        frame_contact_count=frame_contact_count,
        frame_in_contact=frame_contact_count >= int(min_contact_vertices),
        contact_threshold_m=float(contact_threshold_m),
        min_contact_vertices=int(min_contact_vertices),
        path=None,
    )


def ensure_affordance(
    sequence_dir: str | Path,
    demo: HumanDemo,
    object_points: np.ndarray,
    *,
    contact_threshold_m: float = DEFAULT_CONTACT_THRESHOLD_M,
    min_contact_vertices: int = DEFAULT_MIN_CONTACT_VERTICES,
    force: bool = False,
) -> Affordance:
    """Load the sequence's cached affordance, or compute and cache it.

    The cache lives in the sequence directory itself (``affordance.npz``, or
    wherever ``sequence.json``'s ``"affordance"`` key points), so both the
    solver and the seed generator share one artifact across runs.
    """

    sequence_dir = Path(sequence_dir)
    meta_path = sequence_dir / "sequence.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    cached = resolve_sequence_file(sequence_dir, meta, "affordance", AFFORDANCE_FILENAME)
    if cached is not None and not force:
        affordance = Affordance.load(cached)
        if affordance.frame_ids.shape == demo.frame_ids.shape and np.array_equal(affordance.frame_ids, demo.frame_ids):
            return affordance
        log(f"{sequence_dir.name}: cached affordance frame_ids mismatch demo; recomputing")

    affordance = compute_affordance(
        demo,
        object_points,
        contact_threshold_m=contact_threshold_m,
        min_contact_vertices=min_contact_vertices,
    )
    out_path = cached if cached is not None else sequence_dir / AFFORDANCE_FILENAME
    affordance.save(out_path)
    log(
        f"{sequence_dir.name}: cached {out_path.name} "
        f"(contact_points={int(np.count_nonzero(affordance.contact_count))}/{affordance.points_object_frame.shape[0]}, "
        f"contact_frames={int(affordance.frame_in_contact.sum())}/{affordance.frame_ids.shape[0]})"
    )
    return Affordance.load(out_path)


# ---------------------------------------------------------------------------
# Debug visualization (migrated from the retired standalone extractor).
# ---------------------------------------------------------------------------


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
        points[:, 0], points[:, 1], points[:, 2],
        c=values, s=12, cmap="inferno", vmin=0.0, vmax=1.0, linewidths=0, depthshade=False,
    )
    contacted = values > 0.0
    if np.any(contacted):
        ax.scatter(
            points[contacted, 0], points[contacted, 1], points[contacted, 2],
            c=values[contacted], s=22, cmap="inferno", vmin=0.0, vmax=1.0,
            linewidths=0.15, edgecolors="white", depthshade=False,
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


def _run_sequence(sequence_dir: Path, args: argparse.Namespace) -> None:
    from ocir.grasp_synthesis.object_surface import ObjectSurface

    demo = HumanDemo.from_sequence_dir(sequence_dir)
    surface = ObjectSurface.from_sequence_dir(sequence_dir)
    affordance = ensure_affordance(
        sequence_dir,
        demo,
        surface.points_object_frame,
        contact_threshold_m=float(args.contact_threshold),
        min_contact_vertices=int(args.min_contact_vertices),
        force=bool(args.force),
    )
    if args.debug_viz:
        debug_dir = sequence_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        stem = f"{sequence_dir.name}_affordance"
        write_ascii_ply(debug_dir / f"{stem}.ply", affordance.points_object_frame, affordance.heatmap)
        write_heatmap_png(debug_dir / f"{stem}.png", affordance.points_object_frame, affordance.heatmap, sequence_dir.name)
        log(f"{sequence_dir.name}: wrote debug viz under {debug_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence-dir", type=Path, default=None)
    group.add_argument("--sequences-root", type=Path, default=None)
    parser.add_argument("--sequence-id", action="append", default=None, help="With --sequences-root: only these subdirectories.")
    parser.add_argument("--contact-threshold", type=float, default=DEFAULT_CONTACT_THRESHOLD_M)
    parser.add_argument("--min-contact-vertices", type=int, default=DEFAULT_MIN_CONTACT_VERTICES)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-viz", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.sequence_dir is not None:
        dirs = [args.sequence_dir.expanduser()]
    else:
        root = args.sequences_root.expanduser()
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if args.sequence_id:
            wanted = set(args.sequence_id)
            dirs = [p for p in dirs if p.name in wanted]
    for sequence_dir in dirs:
        _run_sequence(sequence_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
