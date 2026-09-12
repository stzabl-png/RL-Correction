"""Minimal step-wise keyboard joint teleop for hand-eye capture.

This mirrors dexcontrol's ``examples/advanced_examples/keyboard_joint_control.py``
but is driven by the OpenCV preview window's key events so capture and motion
share one terminal and one focus.  Every ``w``/``s`` key event sends one small
relative joint step through ``move_joint_pos(relative=True)``; nothing moves
unless a key event arrives, so releasing the key stops the robot immediately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

COMPONENT_JOINTS = {"left_arm": 7, "right_arm": 7, "torso": 3, "head": 3}
MAX_STEP_DEG = {"left_arm": 3.0, "right_arm": 3.0, "torso": 1.5, "head": 3.0}


def arm_component_for_link(link: str) -> str:
    """Map a URDF link name to the dexcontrol component that moves it."""
    if link.startswith("L_"):
        return "left_arm"
    if link.startswith("R_"):
        return "right_arm"
    if link.startswith("torso"):
        return "torso"
    if link.startswith(("head", "zed")):
        return "head"
    raise ValueError(f"Cannot infer the controlling component from link {link!r}")


@dataclass
class KeyboardJointTeleop:
    """Handle preview-window keys and translate them into relative joint steps.

    Keys: ``0-9`` select joint, ``w``/``s`` step +/-, ``W``/``S`` step 4x,
    ``t`` cycle component (arm ↔ torso), ``-``/``=`` halve/double the step.
    """

    reader: object
    components: tuple[str, ...]
    step_deg: float = 0.5
    velocity_scale: float = 0.3
    active_component: str = field(init=False)
    joint_index: int = field(init=False, default=0)
    last_command_time: float = field(init=False, default=0.0)
    commands_sent: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("teleop needs at least one component")
        for name in self.components:
            if name not in COMPONENT_JOINTS:
                raise ValueError(f"Unknown component {name!r}")
        self.active_component = self.components[0]
        self.joint_index = COMPONENT_JOINTS[self.active_component] - 1
        self.step_deg = min(self.step_deg, MAX_STEP_DEG[self.active_component])

    # ------------------------------------------------------------------ helpers
    @property
    def joint_count(self) -> int:
        return COMPONENT_JOINTS[self.active_component]

    def seconds_since_command(self) -> float:
        return time.monotonic() - self.last_command_time if self.commands_sent else float("inf")

    def status(self) -> str:
        return (
            f"teleop: {self.active_component} joint {self.joint_index}  step {self.step_deg:.2f} deg"
            "  (w/s move, W/S x4, 0-9 joint, t component, -/= step)"
        )

    def _step(self, direction: int, multiplier: float = 1.0) -> None:
        comp = self.reader.component(self.active_component)  # type: ignore[attr-defined]
        rel = np.zeros(self.joint_count, dtype=np.float64)
        rel[self.joint_index] = np.deg2rad(self.step_deg * multiplier * direction)
        comp.move_joint_pos(rel, relative=True, velocity_scale=self.velocity_scale)
        self.last_command_time = time.monotonic()
        self.commands_sent += 1

    # --------------------------------------------------------------- key events
    def handle_key(self, key: int) -> bool:
        """Return True if the key was consumed by the teleop."""
        if key < 0:
            return False
        ch = chr(key) if 0 <= key < 256 else ""
        if ch == "w":
            self._step(+1)
        elif ch == "s":
            self._step(-1)
        elif ch == "W":
            self._step(+1, 4.0)
        elif ch == "S":
            self._step(-1, 4.0)
        elif ch.isdigit():
            idx = int(ch)
            if idx < self.joint_count:
                self.joint_index = idx
        elif ch == "t":
            pos = self.components.index(self.active_component)
            self.active_component = self.components[(pos + 1) % len(self.components)]
            self.joint_index = min(self.joint_index, self.joint_count - 1)
            self.step_deg = min(self.step_deg, MAX_STEP_DEG[self.active_component])
        elif ch == "-":
            self.step_deg = max(self.step_deg / 2.0, 0.05)
        elif ch in ("=", "+"):
            self.step_deg = min(self.step_deg * 2.0, MAX_STEP_DEG[self.active_component])
        else:
            return False
        return True
