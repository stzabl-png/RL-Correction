"""Pure-Python Task-5 protocol enums shared by data and Isaac modules."""
from __future__ import annotations

from enum import Enum, IntEnum


class PourPhase(IntEnum):
    STANCE = 0
    APPROACH = 1
    DUAL_GRASP = 2
    ALIGN = 3
    POUR = 4
    RETURN = 5
    VERIFY = 6

    @classmethod
    def count(cls) -> int:
        return len(cls)


class CurriculumStage(IntEnum):
    SINGLE_GRASP = 0
    DUAL_GRASP = 1
    APPROACH_GRASP = 2
    ALIGN_POUR = 3
    FULL = 4


class AblationMode(str, Enum):
    FULL = "full"
    NO_TRAJECTORY = "no_trajectory"
    NO_CONTACT = "no_contact"
    PURE_RL = "pure_rl"

    @property
    def use_video_trajectory(self) -> bool:
        return self in (AblationMode.FULL, AblationMode.NO_CONTACT)

    @property
    def use_video_contact(self) -> bool:
        return self in (AblationMode.FULL, AblationMode.NO_TRAJECTORY)


class FailureCode(IntEnum):
    NONE = 0
    LEFT_DROP = 1
    RIGHT_DROP = 2
    CUP_TIP = 3
    ALIGN_MISS = 4
    SPILL = 5
    TIMEOUT = 6


FAILURE_NAMES = {
    FailureCode.NONE: "none",
    FailureCode.LEFT_DROP: "left_drop",
    FailureCode.RIGHT_DROP: "right_drop",
    FailureCode.CUP_TIP: "cup_tip",
    FailureCode.ALIGN_MISS: "align_miss",
    FailureCode.SPILL: "spill",
    FailureCode.TIMEOUT: "timeout",
}
