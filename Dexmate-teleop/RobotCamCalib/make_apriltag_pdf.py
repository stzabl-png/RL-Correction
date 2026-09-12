#!/usr/bin/env python3
"""Generate a print-ready single AprilTag and its matching board-layout YAML.

The YAML uses the same ``robot_cam_calib.apriltag_board.v1`` schema as
``make_apriltag_grid_pdf.py``, so a one-tag target loads through
``load_apriltag_board_yaml`` and solves through ``estimate_board_pose_bundle_pnp``
with no special-casing. Set ``board_min_tags=1`` on the calibration config, since
the default of 4 can never be met by a single tag.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parent

MM_PER_INCH = 25.4

# Portrait; --landscape swaps them.
PAGE_SIZES_MM = {
    "a0": (841.0, 1189.0),
    "a1": (594.0, 841.0),
    "a2": (420.0, 594.0),
    "a3": (297.0, 420.0),
    "a4": (210.0, 297.0),
}
PAGE_ORDER = ("a4", "a3", "a2", "a1", "a0")

# Printers cannot image closer than roughly this to the sheet edge, and a driver
# that decides the artwork does not fit will silently rescale the page.
MIN_PAGE_MARGIN_MM = 8.0

BORDER_BITS = 1

# Names as pupil_apriltags expects them, mapped to the OpenCV dictionary.
# extr_calib.py forwards the family string straight to pupil_apriltags.Detector,
# so the YAML has to carry the pupil spelling, not the cv2 one.
FAMILIES = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
}


def april_dictionary(family: str):
    if not hasattr(cv2, "aruco"):
        raise SystemExit("cv2.aruco is missing. Install opencv-contrib-python.")
    cv2_name = FAMILIES[family]
    dict_id = getattr(cv2.aruco, cv2_name, None)
    if dict_id is None:
        dict_id = getattr(cv2.aruco, cv2_name.upper(), None)
    if dict_id is None:
        raise SystemExit(f"This OpenCV build has no {cv2_name}.")
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def cell_span(dictionary) -> int:
    """Cells across the printed tag, black border included."""
    return int(dictionary.markerSize) + 2 * BORDER_BITS


def render_marker(dictionary, tag_id: int, marker_px: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, marker_px, borderBits=BORDER_BITS)
    marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, tag_id, marker_px, marker, BORDER_BITS)
    return marker


def snap_marker_px(marker_px: int, cells: int) -> int:
    """Round up to a whole number of pixels per cell so bit edges stay crisp."""
    return int(np.ceil(marker_px / cells) * cells)


def smallest_page_for(size_mm: float) -> Optional[str]:
    for name in PAGE_ORDER:
        w, h = PAGE_SIZES_MM[name]
        if size_mm + 2.0 * MIN_PAGE_MARGIN_MM <= min(w, h):
            return name
    return None


def tag_corners_pupil_order_mm(tag_size_mm: float) -> list[list[float]]:
    """Corner order of pupil_apriltags Detection.corners, in the tag frame."""
    half = tag_size_mm / 2.0
    return [
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ]


def draw_page(
    ax,
    marker: np.ndarray,
    *,
    page_w_mm: float,
    page_h_mm: float,
    tag_size_mm: float,
    quiet_mm: float,
    annotations: Optional[dict],
) -> None:
    ax.set_xlim(0.0, page_w_mm)
    ax.set_ylim(0.0, page_h_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    tag_left = (page_w_mm - tag_size_mm) / 2.0
    tag_bottom = (page_h_mm - tag_size_mm) / 2.0
    ax.imshow(
        marker,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        origin="upper",
        extent=[tag_left, tag_left + tag_size_mm, tag_bottom, tag_bottom + tag_size_mm],
    )

    if annotations is None:
        return

    # Cut line at the edge of the quiet zone. Trimming inside it destroys the
    # white margin the detector needs to close the tag's outer contour.
    ax.add_patch(
        Rectangle(
            (tag_left - quiet_mm, tag_bottom - quiet_mm),
            tag_size_mm + 2.0 * quiet_mm,
            tag_size_mm + 2.0 * quiet_mm,
            fill=False,
            edgecolor="0.75",
            linewidth=0.35,
            linestyle=(0, (4, 3)),
        )
    )

    # Dimension line against the black square, which is the edge to measure.
    dim_y = tag_bottom - quiet_mm - 6.0
    ax.annotate(
        "",
        xy=(tag_left, dim_y),
        xytext=(tag_left + tag_size_mm, dim_y),
        arrowprops=dict(arrowstyle="<->", color="0.35", linewidth=0.6),
    )
    ax.text(
        page_w_mm / 2.0,
        dim_y - 4.5,
        f"measure black square edge = {tag_size_mm:.3f} mm",
        fontsize=7,
        color="0.35",
        ha="center",
        va="top",
    )

    top = tag_bottom + tag_size_mm + quiet_mm
    ax.text(
        page_w_mm / 2.0,
        top + 10.0,
        f"AprilTag {annotations['family']}  id {annotations['tag_id']}  "
        f"edge = {tag_size_mm:.3f} mm",
        fontsize=9,
        color="0.2",
        ha="center",
        va="bottom",
    )
    ax.text(
        page_w_mm / 2.0,
        top + 5.0,
        "Print at 100% / actual size. Do NOT use 'fit to printable area'. "
        "Cut no closer than the dashed line.",
        fontsize=7,
        color="0.35",
        ha="center",
        va="bottom",
    )

    ruler_y = 14.0
    ruler_left = (page_w_mm - 100.0) / 2.0
    ax.plot([ruler_left, ruler_left + 100.0], [ruler_y, ruler_y], color="black", linewidth=0.8)
    for x in (ruler_left, ruler_left + 100.0):
        ax.plot([x, x], [ruler_y - 2.0, ruler_y + 2.0], color="black", linewidth=0.8)
    ax.text(page_w_mm / 2.0, ruler_y + 2.5, "100 mm check", fontsize=7,
            ha="center", va="bottom", color="0.2")


def save_page(
    path_pdf: Optional[Path],
    path_png: Optional[Path],
    marker: np.ndarray,
    *,
    page_w_mm: float,
    page_h_mm: float,
    tag_size_mm: float,
    quiet_mm: float,
    annotations: Optional[dict],
) -> None:
    fig = plt.figure(figsize=(page_w_mm / MM_PER_INCH, page_h_mm / MM_PER_INCH), dpi=300)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_page(
        ax,
        marker,
        page_w_mm=page_w_mm,
        page_h_mm=page_h_mm,
        tag_size_mm=tag_size_mm,
        quiet_mm=quiet_mm,
        annotations=annotations,
    )
    if path_pdf is not None:
        path_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_pdf, format="pdf", facecolor="white")
    if path_png is not None:
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, format="png", dpi=300, facecolor="white")
    plt.close(fig)


def format_mm(value_mm: float) -> str:
    return f"{value_mm:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-size-mm", type=float, default=120.0,
                        help="Edge length of the black outer square, which is what the detector reports.")
    parser.add_argument("--family", default="tag36h11", choices=sorted(FAMILIES))
    parser.add_argument("--page", default="a4", choices=sorted(PAGE_SIZES_MM))
    parser.add_argument("--landscape", action="store_true")
    parser.add_argument("--quiet-fraction", type=float, default=0.25,
                        help="White border around the tag, as a fraction of tag size. Two cells is the safe floor.")
    parser.add_argument("--marker-px", type=int, default=1200,
                        help="Raster resolution. Rounded up to a whole number of pixels per cell.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "assets" / "apriltag_single")
    parser.add_argument("--stem", default=None)
    parser.add_argument("--no-annotations", dest="annotations", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tag_size_mm <= 0:
        raise SystemExit("--tag-size-mm must be positive.")
    if args.quiet_fraction < 0:
        raise SystemExit("--quiet-fraction must be non-negative.")

    dictionary = april_dictionary(args.family)
    capacity = int(dictionary.bytesList.shape[0])
    if not 0 <= args.tag_id < capacity:
        raise SystemExit(
            f"--tag-id {args.tag_id} is out of range for {args.family}, which holds "
            f"ids 0-{capacity - 1}."
        )

    cells = cell_span(dictionary)
    quiet_mm = args.tag_size_mm * args.quiet_fraction
    min_quiet_mm = args.tag_size_mm / cells
    if quiet_mm < min_quiet_mm:
        raise SystemExit(
            f"Quiet zone {quiet_mm:.2f} mm is under one cell ({min_quiet_mm:.2f} mm). "
            "The detector needs white around the black border to close the tag contour; "
            f"use --quiet-fraction {1.0 / cells:.3f} or more."
        )

    outer_mm = args.tag_size_mm + 2.0 * quiet_mm
    page_w_mm, page_h_mm = PAGE_SIZES_MM[args.page]
    if args.landscape:
        page_w_mm, page_h_mm = page_h_mm, page_w_mm

    margin_x_mm = (page_w_mm - outer_mm) / 2.0
    margin_y_mm = (page_h_mm - outer_mm) / 2.0
    if margin_x_mm < MIN_PAGE_MARGIN_MM or margin_y_mm < MIN_PAGE_MARGIN_MM:
        suggestion = smallest_page_for(outer_mm)
        hint = f"Use --page {suggestion}." if suggestion else "Tile it across sheets instead."
        raise SystemExit(
            f"Tag plus quiet zone is {outer_mm:.1f} mm and leaves margins "
            f"{margin_x_mm:.1f}/{margin_y_mm:.1f} mm on {args.page.upper()}, under the "
            f"{MIN_PAGE_MARGIN_MM:.0f} mm a printer needs. The driver would shrink the page "
            f"and quietly rescale the tag. {hint}"
        )

    marker_px = snap_marker_px(args.marker_px, cells)
    marker = render_marker(dictionary, args.tag_id, marker_px)

    stem = args.stem or (
        f"apriltag_{args.family}_id{args.tag_id}"
        f"_tag{format_mm(args.tag_size_mm)}mm_{args.page}"
        f"_{'landscape' if args.landscape else 'portrait'}"
    )

    annotations = None
    if args.annotations:
        annotations = {"family": args.family, "tag_id": args.tag_id}
    save_page(
        args.out_dir / f"{stem}.pdf",
        None,
        marker,
        page_w_mm=page_w_mm,
        page_h_mm=page_h_mm,
        tag_size_mm=args.tag_size_mm,
        quiet_mm=quiet_mm,
        annotations=annotations,
    )
    save_page(
        None,
        args.out_dir / f"{stem}_tag_only.png",
        marker,
        page_w_mm=outer_mm,
        page_h_mm=outer_mm,
        tag_size_mm=args.tag_size_mm,
        quiet_mm=quiet_mm,
        annotations=None,
    )

    # Same schema as make_apriltag_grid_pdf.py, with the board reduced to one tag.
    # The board frame is the tag frame, so T_board_tag is identity and the solved
    # X_CamTag is directly the tag pose.
    corners = tag_corners_pupil_order_mm(args.tag_size_mm)
    board_yaml = {
        "schema": "robot_cam_calib.apriltag_board.v1",
        "name": stem,
        "family": args.family,
        "units": "mm",
        "layout": {
            "rows": 1,
            "cols": 1,
            "tag_id_start": args.tag_id,
            "tag_id_end": args.tag_id,
        },
        "geometry": {
            # tile is the tag plus its quiet zone: the white card you cut out.
            "tag_size_mm": args.tag_size_mm,
            "marker_fraction": args.tag_size_mm / outer_mm,
            "tile_size_mm": outer_mm,
            "explicit_gap_mm": 0.0,
            "pitch_mm": outer_mm,
            "black_marker_edge_gap_mm": 2.0 * quiet_mm,
            "board_size_mm": [outer_mm, outer_mm],
            "quiet_zone_mm": quiet_mm,
            "cells_across_tag": cells,
        },
        "target_frame": {
            "name": "board",
            "origin": "center of the printed tag",
            "x_axis": "toward the tag's right edge as printed",
            "y_axis": "toward the tag's top edge as printed",
            "z_axis": "normal to printed tag plane",
        },
        "detection_corner_order": {
            "source": "pupil_apriltags.Detection.corners",
            "tag_frame_corners_mm": corners,
            "description": "Use each detection's four image corners in this exact order with the matching corners_board_mm entry for bundle PnP.",
        },
        "usage": {
            "board_min_tags": 1,
            "note": (
                "A single tag gives solvePnP only four coplanar points, which admits a "
                "two-fold planar pose ambiguity. Print it as large as the working "
                "distance allows and avoid near-fronto-parallel views, where the two "
                "solutions are hardest to tell apart."
            ),
        },
        "tags": [
            {
                "id": args.tag_id,
                "row": 0,
                "col": 0,
                "center_mm": [0.0, 0.0, 0.0],
                "T_board_tag": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "corners_board_mm": corners,
            }
        ],
    }
    yaml_path = args.out_dir / f"{stem}.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(board_yaml, f, sort_keys=False)

    print(f"family={args.family} (ids 0-{capacity - 1}), id={args.tag_id}")
    print(f"tag_size_mm={args.tag_size_mm:.6f}")
    print(f"quiet_zone_mm={quiet_mm:.6f} ({cells} cells across tag, one cell = {min_quiet_mm:.3f} mm)")
    print(f"cut_size_mm={outer_mm:.6f}")
    print(f"page={args.page}{' landscape' if args.landscape else ''} margins_mm={margin_x_mm:.3f}/{margin_y_mm:.3f}")
    print(f"raster_px={marker_px} ({marker_px // cells} px/cell)")
    print(f"yaml={yaml_path}")


if __name__ == "__main__":
    main()
