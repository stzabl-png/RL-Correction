#!/usr/bin/env python3
"""Generate the print-ready ChArUco board and matching calibration config YAML.

The board raster comes from ``cv2.aruco.CharucoBoard.generateImage`` rather than
from hand-drawn squares, so the printed pattern cannot drift from the board
model the detector solves against.
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

# Long edge first; the board is laid out landscape.
PAGE_SIZES_MM = {
    "a0": (1189.0, 841.0),
    "a1": (841.0, 594.0),
    "a2": (594.0, 420.0),
    "a3": (420.0, 297.0),
    "a4": (297.0, 210.0),
}

# Desktop printers and plotters cannot image closer than roughly this to the
# sheet edge. Leaving less forces the driver into "shrink to fit", and that
# rescale is not guaranteed to be identical along x and y -- which is exactly
# the anisotropy that corrupts the fx/fy ratio.
MIN_PAGE_MARGIN_MM = 8.0

BORDER_BITS = 1


def charuco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise SystemExit("cv2.aruco is missing. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, name):
        raise SystemExit(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_charuco_board(
    squares_x: int,
    squares_y: int,
    square_mm: float,
    marker_mm: float,
    dictionary_name: str,
    legacy_pattern: bool,
):
    dictionary = charuco_dictionary(dictionary_name)
    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (int(squares_x), int(squares_y)),
            float(square_mm),
            float(marker_mm),
            dictionary,
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            int(squares_x),
            int(squares_y),
            float(square_mm),
            float(marker_mm),
            dictionary,
        )
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(bool(legacy_pattern))
    return board, dictionary


def marker_bit_span(dictionary) -> int:
    """Total bits across one printed marker, border included."""
    marker_size = int(getattr(dictionary, "markerSize", 0))
    if marker_size <= 0:
        marker_size = int(dictionary.bytesList.shape[0] and 5)
    return marker_size + 2 * BORDER_BITS


def snap_px_per_square(px_per_square: int, marker_fraction: float, bit_span: int) -> int:
    """Round up until every marker bit lands on a whole number of pixels.

    A marker whose bits straddle pixel boundaries prints with ragged bit edges,
    which costs sub-pixel corner accuracy for no reason.
    """
    for candidate in range(int(px_per_square), int(px_per_square) + 4 * bit_span * 4):
        marker_px = candidate * marker_fraction
        if abs(marker_px - round(marker_px)) > 1e-9:
            continue
        if int(round(marker_px)) % bit_span == 0:
            return candidate
    return int(px_per_square)


def render_board_image(board, squares_x: int, squares_y: int, px_per_square: int) -> np.ndarray:
    size_px = (int(squares_x * px_per_square), int(squares_y * px_per_square))
    if hasattr(board, "generateImage"):
        return board.generateImage(size_px, marginSize=0, borderBits=BORDER_BITS)
    return board.draw(size_px, None, 0, BORDER_BITS)


def board_marker_count(board) -> int:
    ids = board.getIds() if hasattr(board, "getIds") else board.ids
    return int(np.asarray(ids).size)


def draw_page(
    ax,
    board_image: np.ndarray,
    *,
    page_w_mm: float,
    page_h_mm: float,
    board_left_mm: float,
    board_bottom_mm: float,
    board_w_mm: float,
    board_h_mm: float,
    add_crop_marks: bool,
) -> None:
    ax.set_xlim(0.0, page_w_mm)
    ax.set_ylim(0.0, page_h_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    ax.imshow(
        board_image,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        origin="upper",
        extent=[
            board_left_mm,
            board_left_mm + board_w_mm,
            board_bottom_mm,
            board_bottom_mm + board_h_mm,
        ],
    )

    if not add_crop_marks:
        return

    ax.add_patch(
        Rectangle(
            (board_left_mm, board_bottom_mm),
            board_w_mm,
            board_h_mm,
            fill=False,
            edgecolor="0.70",
            linewidth=0.35,
        )
    )
    crop_len_mm = 7.0
    for x in (board_left_mm, board_left_mm + board_w_mm):
        direction = -1.0 if x == board_left_mm else 1.0
        for y in (board_bottom_mm, board_bottom_mm + board_h_mm):
            ax.plot([x, x + direction * crop_len_mm], [y, y], color="0.65", linewidth=0.35)
    for y in (board_bottom_mm, board_bottom_mm + board_h_mm):
        direction = -1.0 if y == board_bottom_mm else 1.0
        for x in (board_left_mm, board_left_mm + board_w_mm):
            ax.plot([x, x], [y, y + direction * crop_len_mm], color="0.65", linewidth=0.35)


def annotate_page(
    ax,
    *,
    page_w_mm: float,
    page_h_mm: float,
    board_bottom_mm: float,
    squares_x: int,
    squares_y: int,
    square_mm: float,
    marker_mm: float,
    dictionary_name: str,
    board_w_mm: float,
    board_h_mm: float,
) -> None:
    top_margin_mm = page_h_mm - (board_bottom_mm + board_h_mm)
    ax.text(
        page_w_mm / 2.0,
        page_h_mm - top_margin_mm * 0.36,
        f"ChArUco {squares_x}x{squares_y}, square = {square_mm:.3f} mm, "
        f"marker = {marker_mm:.3f} mm, {dictionary_name}",
        fontsize=8,
        color="0.2",
        ha="center",
        va="center",
    )
    ax.text(
        page_w_mm / 2.0,
        page_h_mm - top_margin_mm * 0.72,
        "Print at 100% / actual size. Do NOT use 'fit to printable area'. "
        f"Board outer size = {board_w_mm:.2f} mm x {board_h_mm:.2f} mm.",
        fontsize=7,
        color="0.35",
        ha="center",
        va="center",
    )

    # Ruler in the bottom margin. Measuring it is the only check that catches a
    # driver that silently rescaled the page.
    ruler_y_mm = board_bottom_mm * 0.45
    ruler_left_mm = (page_w_mm - 100.0) / 2.0
    ruler_right_mm = ruler_left_mm + 100.0
    ax.plot([ruler_left_mm, ruler_right_mm], [ruler_y_mm, ruler_y_mm], color="black", linewidth=0.8)
    for x in (ruler_left_mm, ruler_right_mm):
        ax.plot([x, x], [ruler_y_mm - 2.0, ruler_y_mm + 2.0], color="black", linewidth=0.8)
    ax.text(
        page_w_mm / 2.0,
        ruler_y_mm + 3.0,
        "100 mm check",
        fontsize=7,
        ha="center",
        va="bottom",
        color="0.2",
    )


def save_page(
    path_pdf: Optional[Path],
    path_png: Optional[Path],
    board_image: np.ndarray,
    *,
    page_w_mm: float,
    page_h_mm: float,
    board_left_mm: float,
    board_bottom_mm: float,
    board_w_mm: float,
    board_h_mm: float,
    annotations: Optional[dict],
) -> None:
    fig = plt.figure(figsize=(page_w_mm / MM_PER_INCH, page_h_mm / MM_PER_INCH), dpi=300)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    draw_page(
        ax,
        board_image,
        page_w_mm=page_w_mm,
        page_h_mm=page_h_mm,
        board_left_mm=board_left_mm,
        board_bottom_mm=board_bottom_mm,
        board_w_mm=board_w_mm,
        board_h_mm=board_h_mm,
        add_crop_marks=annotations is not None,
    )
    if annotations is not None:
        annotate_page(
            ax,
            page_w_mm=page_w_mm,
            page_h_mm=page_h_mm,
            board_bottom_mm=board_bottom_mm,
            board_w_mm=board_w_mm,
            board_h_mm=board_h_mm,
            **annotations,
        )

    if path_pdf is not None:
        path_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_pdf, format="pdf", facecolor="white")
    if path_png is not None:
        path_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path_png, format="png", dpi=300, facecolor="white")
    plt.close(fig)


def format_mm(value_mm: float) -> str:
    text = f"{value_mm:g}".replace(".", "p")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "assets" / "charuco")
    parser.add_argument("--squares-x", type=int, default=10)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--square-mm", type=float, default=75.0)
    parser.add_argument("--marker-fraction", type=float, default=0.75)
    parser.add_argument("--dictionary", default="DICT_5X5_50")
    parser.add_argument("--legacy-pattern", action="store_true")
    parser.add_argument("--page", default="a1", choices=sorted(PAGE_SIZES_MM))
    parser.add_argument("--portrait", action="store_true")
    parser.add_argument(
        "--px-per-square",
        type=int,
        default=448,
        help="Board raster resolution. Rounded up so marker bits land on whole pixels.",
    )
    parser.add_argument("--stem", default=None)
    parser.add_argument("--no-annotations", dest="annotations", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.squares_x <= 1 or args.squares_y <= 1:
        raise SystemExit("--squares-x and --squares-y must be at least 2.")
    if args.square_mm <= 0:
        raise SystemExit("--square-mm must be positive.")
    if args.marker_fraction <= 0 or args.marker_fraction >= 1:
        raise SystemExit("--marker-fraction must be in (0, 1).")

    marker_mm = args.square_mm * args.marker_fraction
    board, dictionary = make_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_mm,
        marker_mm,
        args.dictionary,
        args.legacy_pattern,
    )

    marker_count = board_marker_count(board)
    dictionary_capacity = int(dictionary.bytesList.shape[0])
    if marker_count > dictionary_capacity:
        raise SystemExit(
            f"Board needs {marker_count} markers but {args.dictionary} only holds "
            f"{dictionary_capacity}. Use a larger dictionary or a smaller board."
        )

    page_w_mm, page_h_mm = PAGE_SIZES_MM[args.page]
    if args.portrait:
        page_w_mm, page_h_mm = page_h_mm, page_w_mm

    board_w_mm = args.squares_x * args.square_mm
    board_h_mm = args.squares_y * args.square_mm
    margin_x_mm = (page_w_mm - board_w_mm) / 2.0
    margin_y_mm = (page_h_mm - board_h_mm) / 2.0
    if margin_x_mm < MIN_PAGE_MARGIN_MM or margin_y_mm < MIN_PAGE_MARGIN_MM:
        raise SystemExit(
            f"Board {board_w_mm:.1f}x{board_h_mm:.1f} mm leaves margins "
            f"{margin_x_mm:.1f}/{margin_y_mm:.1f} mm on {args.page.upper()}, below the "
            f"{MIN_PAGE_MARGIN_MM:.0f} mm a printer needs. The driver would shrink the "
            "page to fit and quietly rescale the board. Use a larger sheet or a "
            "smaller --square-mm."
        )

    bit_span = marker_bit_span(dictionary)
    px_per_square = snap_px_per_square(args.px_per_square, args.marker_fraction, bit_span)
    board_image = render_board_image(board, args.squares_x, args.squares_y, px_per_square)

    stem = args.stem
    if stem is None:
        stem = (
            f"charuco_{args.squares_x}x{args.squares_y}"
            f"_square{format_mm(args.square_mm)}mm"
            f"_marker{format_mm(marker_mm)}mm"
            f"_{args.dictionary}_{args.page}"
            f"_{'portrait' if args.portrait else 'landscape'}"
        )

    annotations = None
    if args.annotations:
        annotations = {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_mm": args.square_mm,
            "marker_mm": marker_mm,
            "dictionary_name": args.dictionary,
        }
    save_page(
        args.out_dir / f"{stem}.pdf",
        None,
        board_image,
        page_w_mm=page_w_mm,
        page_h_mm=page_h_mm,
        board_left_mm=margin_x_mm,
        board_bottom_mm=margin_y_mm,
        board_w_mm=board_w_mm,
        board_h_mm=board_h_mm,
        annotations=annotations,
    )

    quiet_margin_mm = args.square_mm / 2.0
    save_page(
        None,
        args.out_dir / f"{stem}_board_only.png",
        board_image,
        page_w_mm=board_w_mm + 2.0 * quiet_margin_mm,
        page_h_mm=board_h_mm + 2.0 * quiet_margin_mm,
        board_left_mm=quiet_margin_mm,
        board_bottom_mm=quiet_margin_mm,
        board_w_mm=board_w_mm,
        board_h_mm=board_h_mm,
        annotations=None,
    )

    config_yaml = {
        "target_type": "charuco",
        # Consumed by intr_calib_charuco.py charuco_mode_defaults(). Metres.
        "charuco": {
            "squares_x": int(args.squares_x),
            "squares_y": int(args.squares_y),
            "square_length": args.square_mm / 1000.0,
            "marker_length": marker_mm / 1000.0,
            "dictionary": str(args.dictionary),
            "legacy_pattern": bool(args.legacy_pattern),
        },
        "board": {
            "name": stem,
            "inner_corners": [args.squares_x - 1, args.squares_y - 1],
            "inner_corner_count": (args.squares_x - 1) * (args.squares_y - 1),
            "marker_count": marker_count,
            "dictionary_capacity": dictionary_capacity,
            "marker_fraction": args.marker_fraction,
            "quiet_zone_per_marker_mm": (args.square_mm - marker_mm) / 2.0,
            "board_size_mm": [board_w_mm, board_h_mm],
            "raster_px_per_square": px_per_square,
            "raster_size_px": [int(board_image.shape[1]), int(board_image.shape[0])],
        },
        "page": {
            "name": args.page,
            "orientation": "portrait" if args.portrait else "landscape",
            "size_mm": [page_w_mm, page_h_mm],
            "margin_mm": [margin_x_mm, margin_y_mm],
        },
        # Single-camera calibration cannot observe scale, so square_length only has
        # to be right in ratio for K and dist. Anisotropic print shrink is the one
        # error that does leak in, straight into fx/fy. Measure both spans.
        "print_check": {
            "print_scale": "100% / actual size, never 'fit to printable area'",
            "measure_x_span_mm": board_w_mm,
            "measure_y_span_mm": board_h_mm,
            "max_anisotropy_fraction": 0.001,
            "note": (
                "Use a laser printer. Absolute scale error only biases translation, "
                "and intrinsics do not use it. A mismatch between the x and y "
                "shrink ratios biases fx/fy directly."
            ),
            "mounting": (
                "Bond to a rigid flat substrate. At this size, 1 mm of paper bow "
                "exceeds the sub-pixel corner noise and becomes the dominant error."
            ),
        },
    }
    config_path = args.out_dir / f"{stem}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump(config_yaml, f, sort_keys=False)

    print(f"squares={args.squares_x}x{args.squares_y}")
    print(f"square_mm={args.square_mm:.6f}")
    print(f"marker_mm={marker_mm:.6f}")
    print(f"inner_corners={(args.squares_x - 1) * (args.squares_y - 1)}")
    print(f"markers={marker_count}/{dictionary_capacity} in {args.dictionary}")
    print(f"board_size_mm={board_w_mm:.3f}x{board_h_mm:.3f}")
    print(f"page={args.page} margins_mm={margin_x_mm:.3f}/{margin_y_mm:.3f}")
    print(f"raster_px={board_image.shape[1]}x{board_image.shape[0]} ({px_per_square} px/square)")
    print(f"config={config_path}")


if __name__ == "__main__":
    main()
