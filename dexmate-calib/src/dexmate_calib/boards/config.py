from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import yaml

if TYPE_CHECKING:  # pragma: no cover
    from dexmate_calib.boards.apriltag import AprilTagGridProfile

TARGET_TYPES = ("charuco", "apriltag_grid")


@dataclass(frozen=True)
class BoardProfile:
    path: Path
    data: dict[str, Any]
    sha256: str
    name: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    dictionary_name: str
    legacy_pattern: bool
    first_marker_id: int

    target_type: str = "charuco"

    @property
    def expected_marker_count(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    @property
    def expected_corner_count(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def canonical_json(self) -> str:
        return json.dumps(self.data, sort_keys=True, separators=(",", ":"))

    def create_opencv_board(self):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("opencv-contrib-python is required") from exc
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was built without the aruco module")
        dictionary_id = getattr(cv2.aruco, self.dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {self.dictionary_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            dictionary,
        )
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(self.legacy_pattern)
        if self.first_marker_id:
            ids = board.getIds().reshape(-1) + self.first_marker_id
            if hasattr(board, "setIds"):
                board.setIds(ids)
            else:
                raise ValueError("This OpenCV version cannot set non-zero marker IDs")
        return board, dictionary


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _validated_data(raw: Any, source: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"Board profile must be a mapping: {source}")
    if raw.get("schema_version") != 1:
        raise ValueError("Only board schema_version: 1 is supported")
    if raw.get("target_type") != "charuco":
        raise ValueError("_validated_data only handles target_type: charuco")
    if not isinstance(raw.get("name"), str) or not raw["name"].strip():
        raise ValueError("Board profile requires a non-empty name")

    geometry = _require_mapping(raw, "geometry")
    aruco = _require_mapping(raw, "aruco")
    required_geometry = ("squares_x", "squares_y", "square_length_m", "marker_length_m")
    missing = [key for key in required_geometry if key not in geometry]
    if missing:
        raise ValueError(f"Missing board geometry keys: {missing}")
    for key in ("dictionary", "legacy_pattern"):
        if key not in aruco:
            raise ValueError(f"Missing aruco.{key}")

    squares_x = int(geometry["squares_x"])
    squares_y = int(geometry["squares_y"])
    square = float(geometry["square_length_m"])
    marker = float(geometry["marker_length_m"])
    if squares_x < 2 or squares_y < 2:
        raise ValueError("squares_x and squares_y must both be >= 2")
    if not (0.0 < marker < square):
        raise ValueError("Require 0 < marker_length_m < square_length_m")
    if aruco.get("marker_ids", "sequential") != "sequential":
        raise ValueError("Version 1 currently supports sequential marker IDs only")

    dictionary_name = str(aruco["dictionary"])
    dictionary_capacities = {
        "DICT_4X4_50": 50,
        "DICT_4X4_100": 100,
        "DICT_4X4_250": 250,
        "DICT_4X4_1000": 1000,
        "DICT_5X5_50": 50,
        "DICT_5X5_100": 100,
        "DICT_5X5_250": 250,
        "DICT_5X5_1000": 1000,
        "DICT_6X6_50": 50,
        "DICT_6X6_100": 100,
        "DICT_6X6_250": 250,
        "DICT_6X6_1000": 1000,
        "DICT_7X7_50": 50,
        "DICT_7X7_100": 100,
        "DICT_7X7_250": 250,
        "DICT_7X7_1000": 1000,
        "DICT_ARUCO_ORIGINAL": 1024,
        "DICT_APRILTAG_16h5": 30,
        "DICT_APRILTAG_25h9": 35,
        "DICT_APRILTAG_36h10": 2320,
        "DICT_APRILTAG_36h11": 587,
    }
    capacity = dictionary_capacities.get(dictionary_name)
    if capacity is None:
        raise ValueError(f"Unsupported or unknown dictionary: {dictionary_name}")
    first_id = int(aruco.get("first_marker_id", 0))
    marker_count = (squares_x * squares_y) // 2
    if first_id < 0 or first_id + marker_count > capacity:
        raise ValueError(
            f"Marker IDs [{first_id}, {first_id + marker_count - 1}] exceed "
            f"{dictionary_name} capacity {capacity}"
        )

    physical = raw.get("physical")
    if physical is not None:
        if not isinstance(physical, dict):
            raise ValueError("physical must be a mapping")
        expected_width = squares_x * square
        expected_height = squares_y * square
        for key, expected in (
            ("board_outer_width_m", expected_width),
            ("board_outer_height_m", expected_height),
        ):
            if key in physical and abs(float(physical[key]) - expected) > 1e-9:
                raise ValueError(f"{key} is inconsistent with square geometry")
    return raw


AnyBoardProfile = Union[BoardProfile, "AprilTagGridProfile"]


def load_board_profile(path: str | Path) -> AnyBoardProfile:
    """Load a ChArUco (:class:`BoardProfile`) or AprilTag grid profile from YAML."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Board profile not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Board profile must be a mapping: {resolved}")
    target_type = raw.get("target_type")
    if target_type == "apriltag_grid":
        from dexmate_calib.boards.apriltag import load_apriltag_grid_profile

        return load_apriltag_grid_profile(resolved, raw)
    if target_type != "charuco":
        raise ValueError(f"Unsupported target_type {target_type!r}; expected one of {TARGET_TYPES}")
    data = _validated_data(raw, resolved)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    geometry = data["geometry"]
    aruco = data["aruco"]
    return BoardProfile(
        path=resolved,
        data=data,
        sha256=hashlib.sha256(canonical).hexdigest(),
        name=str(data["name"]),
        squares_x=int(geometry["squares_x"]),
        squares_y=int(geometry["squares_y"]),
        square_length_m=float(geometry["square_length_m"]),
        marker_length_m=float(geometry["marker_length_m"]),
        dictionary_name=str(aruco["dictionary"]),
        legacy_pattern=bool(aruco["legacy_pattern"]),
        first_marker_id=int(aruco.get("first_marker_id", 0)),
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def board_directory() -> Path:
    return project_root() / "configs" / "boards"


def available_board_profiles() -> list[AnyBoardProfile]:
    return [load_board_profile(path) for path in sorted(board_directory().glob("*.yaml"))]


def require_charuco(profile: AnyBoardProfile, purpose: str = "this command") -> BoardProfile:
    """Reject non-ChArUco profiles for the intrinsics pipeline (duck-typed on ``target_type``)."""
    target_type = getattr(profile, "target_type", "charuco")
    if target_type != "charuco":
        name = getattr(profile, "name", "?")
        raise ValueError(f"{purpose} requires a ChArUco board profile; {name!r} is {target_type}")
    return profile  # type: ignore[return-value]


def create_detector(profile: AnyBoardProfile):
    """Return the detector matching the profile type (ChArUco or AprilTag grid)."""
    if isinstance(profile, BoardProfile):
        from dexmate_calib.intrinsics.detector import CharucoDetector

        return CharucoDetector(profile)
    from dexmate_calib.boards.apriltag import AprilTagGridDetector

    return AprilTagGridDetector(profile)


def resolve_board_profile(value: str | Path) -> AnyBoardProfile:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return load_board_profile(candidate)
    requested = str(value)
    for profile in available_board_profiles():
        aliases = profile.data.get("aliases", [])
        if requested == profile.name or requested in aliases:
            return profile
    names = ", ".join(profile.name for profile in available_board_profiles())
    raise FileNotFoundError(f"Unknown board profile {requested!r}; available: {names}")
