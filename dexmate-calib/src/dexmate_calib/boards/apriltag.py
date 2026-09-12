"""AprilTag grid targets (1x1 single tag up to arbitrary rows x cols).

Profile YAML (``target_type: apriltag_grid``)::

    schema_version: 1
    name: apriltag_36h11_4x4_48mm
    aliases: [robotcamcalib-4x4]
    target_type: apriltag_grid
    apriltag:
      family: tag36h11              # tag36h11 | tag36h10 | tag25h9 | tag16h5
    layout:
      rows: 4
      cols: 4
      tag_id_start: 0               # ids run row-major from the top-left tag
      tag_size_m: 0.048             # outer edge of the black square
      pitch_m: 0.05217              # centre-to-centre spacing (defaults to tag_size_m)
    # or, instead of rows/cols, an explicit list:
    # tags:
    #   - {id: 7, center_m: [0.0, 0.0], size_m: 0.066}

Board frame (same handedness as OpenCV's ChArUco boards): origin at the centre of the
first tag (row 0, col 0 = top-left when the printed sheet is viewed upright), x to the
right (increasing column), y down (increasing row), z into the printed face (away from
the viewer).  Tag corners are stored in the OpenCV detection order top-left, top-right,
bottom-right, bottom-left, so a single tag centred at the origin has corners
``(-s/2,-s/2), (+s/2,-s/2), (+s/2,+s/2), (-s/2,+s/2)``.

Detection uses OpenCV's built-in AprilTag dictionaries (no extra dependency).  A
uniform rotation of every printed tag relative to this convention only rotates the
board frame, which the hand-eye solver absorbs into ``T_link_board``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexmate_calib.intrinsics.detector import Detection

FAMILY_TO_DICT = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
}
FAMILY_CAPACITY = {"tag36h11": 587, "tag36h10": 2320, "tag25h9": 35, "tag16h5": 30}


@dataclass(frozen=True)
class TagSpec:
    tag_id: int
    row: int
    col: int
    center_m: tuple[float, float]
    size_m: float

    def corners_board_m(self) -> np.ndarray:
        cx, cy = self.center_m
        h = self.size_m / 2.0
        return np.array(
            [
                [cx - h, cy - h, 0.0],
                [cx + h, cy - h, 0.0],
                [cx + h, cy + h, 0.0],
                [cx - h, cy + h, 0.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class AprilTagGridProfile:
    path: Path
    data: dict[str, Any]
    sha256: str
    name: str
    family: str
    tags: tuple[TagSpec, ...]
    rows: int
    cols: int

    target_type: str = "apriltag_grid"

    @property
    def dictionary_name(self) -> str:
        return FAMILY_TO_DICT[self.family]

    @property
    def tag_ids(self) -> list[int]:
        return [t.tag_id for t in self.tags]

    @property
    def expected_marker_count(self) -> int:
        return len(self.tags)

    @property
    def expected_corner_count(self) -> int:
        return 4 * len(self.tags)

    @property
    def is_single_tag(self) -> bool:
        return len(self.tags) == 1

    @property
    def tag_size_m(self) -> float:
        return float(np.mean([t.size_m for t in self.tags]))

    def canonical_json(self) -> str:
        return json.dumps(self.data, sort_keys=True, separators=(",", ":"))

    def tag_by_id(self) -> dict[int, TagSpec]:
        return {t.tag_id: t for t in self.tags}

    def object_points(self) -> np.ndarray:
        """(4N, 3) corners of all tags in profile order."""
        return np.concatenate([t.corners_board_m() for t in self.tags], axis=0)

    def create_dictionary(self):
        import cv2

        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary_name))

    def render_image(
        self, pixels_per_m: float = 4000.0, margin_m: float = 0.02
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render the printed board (grayscale) and the pixel↔board-metre transform.

        Returns ``(image, A)`` where ``A`` is a 3x3 affine mapping board (x, y) in metres to
        pixel (u, v).  Used by tests and ``dexcalib board render``.
        """
        import cv2

        pts = self.object_points()
        x_min, y_min = pts[:, 0].min() - margin_m, pts[:, 1].min() - margin_m
        x_max, y_max = pts[:, 0].max() + margin_m, pts[:, 1].max() + margin_m
        width = round((x_max - x_min) * pixels_per_m)
        height = round((y_max - y_min) * pixels_per_m)
        image = np.full((height, width), 255, dtype=np.uint8)
        # board metres -> pixels: u = (x - x_min) * s ; v = (y - y_min) * s  (y down, like the image)
        A = np.array(
            [
                [pixels_per_m, 0.0, -x_min * pixels_per_m],
                [0.0, pixels_per_m, -y_min * pixels_per_m],
                [0, 0, 1],
            ]
        )
        dictionary = self.create_dictionary()
        for tag in self.tags:
            side_px = round(tag.size_m * pixels_per_m)
            marker = cv2.aruco.generateImageMarker(dictionary, tag.tag_id, side_px, borderBits=1)
            corners = tag.corners_board_m()
            uv = (A @ np.c_[corners[:, :2], np.ones(4)].T).T[:, :2]
            u0, v0 = round(uv[0, 0]), round(uv[0, 1])  # top-left
            image[v0 : v0 + side_px, u0 : u0 + side_px] = marker
        return image, A


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def validate_apriltag_grid_data(
    raw: dict[str, Any], source: Path
) -> tuple[list[TagSpec], int, int, str]:
    if raw.get("schema_version") != 1:
        raise ValueError("Only board schema_version: 1 is supported")
    if not isinstance(raw.get("name"), str) or not raw["name"].strip():
        raise ValueError("Board profile requires a non-empty name")
    april = _require_mapping(raw, "apriltag")
    family = str(april.get("family", ""))
    if family not in FAMILY_TO_DICT:
        raise ValueError(f"apriltag.family must be one of {sorted(FAMILY_TO_DICT)}; got {family!r}")
    capacity = FAMILY_CAPACITY[family]

    tags: list[TagSpec] = []
    if "tags" in raw:
        entries = raw["tags"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("tags must be a non-empty list")
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"tags[{i}] must be a mapping")
            tag_id = int(entry["id"])
            size = float(entry["size_m"])
            cx, cy = (float(v) for v in entry.get("center_m", [0.0, 0.0]))
            if size <= 0.0:
                raise ValueError(f"tags[{i}].size_m must be > 0")
            tags.append(
                TagSpec(tag_id, int(entry.get("row", 0)), int(entry.get("col", i)), (cx, cy), size)
            )
        rows = len({t.row for t in tags})
        cols = len({t.col for t in tags})
    else:
        layout = _require_mapping(raw, "layout")
        rows, cols = int(layout["rows"]), int(layout["cols"])
        size = float(layout["tag_size_m"])
        pitch = float(layout.get("pitch_m", size))
        start = int(layout.get("tag_id_start", 0))
        if rows < 1 or cols < 1:
            raise ValueError("layout.rows and layout.cols must be >= 1")
        if size <= 0.0 or pitch < size:
            raise ValueError("Require 0 < tag_size_m <= pitch_m")
        for r in range(rows):
            for c in range(cols):
                tags.append(TagSpec(start + r * cols + c, r, c, (c * pitch, r * pitch), size))
    ids = [t.tag_id for t in tags]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate tag ids in profile")
    if min(ids) < 0 or max(ids) >= capacity:
        raise ValueError(f"Tag ids {min(ids)}..{max(ids)} exceed {family} capacity {capacity}")
    return tags, rows, cols, family


def load_apriltag_grid_profile(path: Path, raw: dict[str, Any]) -> AprilTagGridProfile:
    tags, rows, cols, family = validate_apriltag_grid_data(raw, path)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return AprilTagGridProfile(
        path=path,
        data=raw,
        sha256=hashlib.sha256(canonical).hexdigest(),
        name=str(raw["name"]),
        family=family,
        tags=tuple(tags),
        rows=rows,
        cols=cols,
    )


class AprilTagGridDetector:
    """Detect the profile's tags and expose the same :class:`Detection` interface as ChArUco.

    ``charuco_ids`` are synthetic: ``4 * tag_index + corner_index`` in profile order, so
    :meth:`calibration_points` can map them back to board coordinates.
    """

    def __init__(self, profile: AprilTagGridProfile) -> None:
        import cv2

        self.cv2 = cv2
        self.profile = profile
        self.dictionary = profile.create_dictionary()
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)
        self._index_by_id = {t.tag_id: i for i, t in enumerate(profile.tags)}
        self._object_points = profile.object_points()

    def detect(self, image_bgr: np.ndarray, *, detailed_quality: bool = True) -> Detection | None:
        cv2 = self.cv2
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        marker_corners, marker_ids, _ = self.detector.detectMarkers(gray)
        if marker_ids is None or len(marker_ids) == 0:
            return None
        found: list[tuple[int, np.ndarray]] = []
        for corners, tag_id in zip(marker_corners, marker_ids.reshape(-1)):
            idx = self._index_by_id.get(int(tag_id))
            if idx is None:
                continue
            found.append((idx, np.asarray(corners, dtype=np.float32).reshape(4, 2)))
        if not found:
            return None
        found.sort(key=lambda item: item[0])
        points = np.concatenate([c for _, c in found], axis=0)
        ids = np.concatenate([np.arange(4) + 4 * idx for idx, _ in found]).astype(np.int32)
        tags = [self.profile.tags[idx] for idx, _ in found]
        quality = _tag_quality(gray, points, tags, self.profile)
        return Detection(
            charuco_corners=points.reshape(-1, 1, 2),
            charuco_ids=ids.reshape(-1, 1),
            marker_count=len(found),
            corner_count=len(points),
            coverage_fraction=quality["coverage"],
            sharpness=quality["sharpness"],
            centroid_xy=quality["centroid"],
            orientation_rad=quality["orientation"],
            grid_rows=quality["rows"],
            grid_cols=quality["cols"],
            board_bbox_fraction=quality["bbox_fraction"],
            pixels_per_square=quality["tag_px"],
            rectified_laplacian_var=math.nan,
            rectified_tenengrad_mean=math.nan,
            marker_corners=tuple(c.reshape(1, 4, 2) for _, c in found),
            marker_ids=np.asarray(
                [self.profile.tags[idx].tag_id for idx, _ in found], dtype=np.int32
            ).reshape(-1, 1),
        )

    def draw(self, image_bgr: np.ndarray, detection: Detection | None) -> np.ndarray:
        output = image_bgr.copy()
        if detection is not None and detection.marker_corners:
            self.cv2.aruco.drawDetectedMarkers(
                output, list(detection.marker_corners), detection.marker_ids
            )
        return output

    def calibration_points(self, detection: Detection) -> tuple[np.ndarray, np.ndarray]:
        ids = detection.charuco_ids.reshape(-1)
        if np.any(ids < 0) or np.any(ids >= len(self._object_points)):
            raise ValueError("Detected tag corner ID is outside the profile geometry")
        return self._object_points[ids].copy(), detection.charuco_corners.reshape(-1, 2).astype(
            np.float64
        )


def _tag_quality(
    gray: np.ndarray, points: np.ndarray, tags: list[TagSpec], profile: AprilTagGridProfile
) -> dict:
    import cv2

    pts = points.reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    height, width = gray.shape[:2]
    coverage = float(max(0.0, x_max - x_min) * max(0.0, y_max - y_min) / (width * height))
    pad = 8
    roi = gray[
        max(0, int(y_min) - pad) : min(height, int(y_max) + pad + 1),
        max(0, int(x_min) - pad) : min(width, int(x_max) + pad + 1),
    ]
    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, vecs = np.linalg.eigh(centered.T @ centered)
    axis = vecs[:, -1]
    rows = {t.row for t in tags}
    cols = {t.col for t in tags}
    bbox_fraction = float(
        ((max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1))
        / max(profile.rows * profile.cols, 1)
    )
    quad = pts.reshape(-1, 4, 2)
    side = np.linalg.norm(quad[:, 1] - quad[:, 0], axis=1)
    return {
        "coverage": coverage,
        "sharpness": sharpness,
        "centroid": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
        "orientation": float(math.atan2(float(axis[1]), float(axis[0]))),
        "rows": len(rows),
        "cols": len(cols),
        "bbox_fraction": bbox_fraction,
        "tag_px": float(np.mean(side)),
    }


def identify_markers(image_bgr: np.ndarray) -> list[dict]:
    """Sweep every OpenCV ArUco/AprilTag dictionary and report what is detected.

    Used by ``dexcalib board identify`` to find out which family/id an unknown printed
    tag belongs to.  Returns one entry per (dictionary, id) with the marker side in pixels.
    """
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    names = [
        n
        for n in dir(cv2.aruco)
        if n.startswith("DICT_") and not n.endswith(("H5", "H9", "H10", "H11"))
    ]
    results: list[dict] = []
    for name in sorted(names):
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
        if ids is None:
            continue
        for c, i in zip(corners, ids.reshape(-1)):
            quad = np.asarray(c, dtype=np.float64).reshape(4, 2)
            side = float(np.mean(np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)))
            results.append(
                {
                    "dictionary": name,
                    "id": int(i),
                    "side_px": round(side, 1),
                    "center_px": [round(float(v), 1) for v in quad.mean(axis=0)],
                }
            )
    return results
