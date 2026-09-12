"""Hand skeleton conventions shared across the pipeline.

The pipeline-internal keypoint layout is the MediaPipe 21-landmark order:
0=wrist, 1-4=thumb(CMC,MCP,IP,TIP), 5-8=index(MCP,PIP,DIP,TIP),
9-12=middle, 13-16=ring, 17-20=pinky.

The Wuji glove `hand_skeleton()` stream uses the same landmark set but we
never trust array order coming from the SDK: joints are routed by `name`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

MEDIAPIPE_JOINT_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20])
WRIST_INDEX = 0

# Accept common variants of the canonical names ("pinky_finger_tip", "index_mcp", ...).
_ALIASES = {
    "pinky_finger": "pinky",
    "little_finger": "pinky",
    "little": "pinky",
    "index": "index_finger",
    "middle": "middle_finger",
    "ring": "ring_finger",
}
# Names that must NOT get the finger expansion above.
_CANONICAL = set(MEDIAPIPE_JOINT_NAMES)

_NAME2IDX = {name: i for i, name in enumerate(MEDIAPIPE_JOINT_NAMES)}


def joint_name_to_index(raw_name: str) -> int:
    """Map a (possibly vendor-flavored) joint name to the MediaPipe index.

    Raises KeyError with the offending name so SDK bring-up failures are loud.
    """
    name = raw_name.strip().lower()
    name = re.sub(r"^(left|right|l|r)[_\-]", "", name)
    name = name.replace("-", "_").replace(" ", "_")
    if name in _CANONICAL:
        return _NAME2IDX[name]
    for alias, canonical in _ALIASES.items():
        if name.startswith(alias + "_"):
            candidate = canonical + name[len(alias):]
            if candidate in _CANONICAL:
                return _NAME2IDX[candidate]
    raise KeyError(f"unknown hand joint name: {raw_name!r} (normalized: {name!r})")


@dataclass
class HandFrame:
    """One glove sample: 21 keypoints in the device's wrist-local frame."""

    t_us: int                      # capture timestamp, epoch microseconds
    hand: str                      # "right" | "left"
    kp: np.ndarray                 # (21, 3) float64, meters, MediaPipe order
    conf: np.ndarray = field(default_factory=lambda: np.ones(21))
    wrist_quat: np.ndarray | None = None  # (4,) wxyz, waist->wrist orientation (IMU), optional

    def min_fingertip_conf(self) -> float:
        return float(self.conf[FINGERTIP_INDICES].min())
