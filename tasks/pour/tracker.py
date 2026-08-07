"""JSONL success/failure tracking for training, evaluation, and ablations."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from tasks.pour.enums import FAILURE_NAMES, FailureCode


@dataclass(frozen=True)
class EpisodeRecord:
    run_name: str
    demo_id: str
    seed: int
    episode: int
    success: bool
    terminal_phase: str
    failure_code: int
    steps: int
    left_grasp: bool
    right_grasp: bool
    cup_fraction: float
    spill_fraction: float
    cup_tilt_deg: float
    bottle_tilt_deg: float
    trajectory_return: float
    contact_return: float
    bottle_fraction: float = 0.0
    cup_pose_wxyz: list[float] = field(default_factory=list)
    bottle_pose_wxyz: list[float] = field(default_factory=list)
    success_subconditions: dict[str, bool] = field(default_factory=dict)

    @property
    def failure_reason(self) -> str:
        return FAILURE_NAMES[FailureCode(self.failure_code)]

    def to_json(self) -> dict:
        value = asdict(self)
        value["failure_reason"] = self.failure_reason
        return value


class SuccessTracker:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, records: Iterable[EpisodeRecord]) -> int:
        count = 0
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_json(), separators=(",", ":")) + "\n")
                count += 1
        return count

    def summary(self) -> dict:
        total = successes = 0
        failure = Counter()
        cup = spill = 0.0
        if not self.path.exists():
            return {"episodes": 0, "success_rate": 0.0, "failure_counts": {}}
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                total += 1
                successes += int(bool(value["success"]))
                cup += float(value["cup_fraction"])
                spill += float(value["spill_fraction"])
                if not value["success"]:
                    failure[value["failure_reason"]] += 1
        return {
            "episodes": total,
            "successes": successes,
            "success_rate": successes / max(total, 1),
            "mean_cup_fraction": cup / max(total, 1),
            "mean_spill_fraction": spill / max(total, 1),
            "failure_counts": dict(sorted(failure.items())),
        }

    def write_summary(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
