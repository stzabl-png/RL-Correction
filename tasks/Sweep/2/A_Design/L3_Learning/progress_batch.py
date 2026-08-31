"""Vectorized, simulator-independent task contract for Sweep2.

All geometric tests are expressed in the dustpan frame.  The pan mesh uses local
``x`` for width, ``+y`` for its upward normal, and ``+z`` from handle to open lip.
The cube therefore enters across the positive-z lip while moving toward ``-z``.

This file is deliberately free of Isaac imports.  Unit tests and offline expert
generation use exactly the same containment and progress definitions as the env.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SweepGeometry:
    cube_half: float = 0.0125
    pan_half_width: float = 0.060
    pan_inside_z_min: float = 0.015
    pan_mouth_z: float = 0.095
    # Local +y is the pan's upward normal.  Mesh cross-sections through the basin
    # put the load-bearing top surface at y=+8..11 mm; the old -14 mm value was the
    # exterior underside and accepted cubes underneath the pan.  These centre
    # bounds include the 12.5 mm cube half extent plus contact tolerance.
    pan_center_y_min: float = 0.018
    pan_center_y_max: float = 0.030
    # Keep 5 mm beyond the grounded ramp's contact-offset envelope.  The location
    # remains fixed across episodes; this is deterministic task geometry, not
    # domain randomization.
    start_outside: float = 0.065
    stable_speed: float = 0.050
    stable_steps: int = 10
    moved_gate: float = 0.005
    # Retained as a diagnostic only.  Operational success is first valid entry.
    deep_inside_margin: float = 0.020


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply scalar-first quaternion ``q`` to vector ``v``."""
    qw, qv = q[..., :1], q[..., 1:]
    return v + 2.0 * torch.cross(qv, torch.cross(qv, v, dim=-1) + qw * v,
                                 dim=-1)


def point_in_pan_frame(point_w: torch.Tensor, pan_pos_w: torch.Tensor,
                       pan_quat_w: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(pan_quat_w), point_w - pan_pos_w)


def sweep_signals(cube_pos_w: torch.Tensor, cube_vel_w: torch.Tensor,
                  pan_pos_w: torch.Tensor, pan_quat_w: torch.Tensor,
                  pan_vel_w: torch.Tensor, cube_start_w: torch.Tensor,
                  geometry: SweepGeometry = SweepGeometry()) -> dict[str, torch.Tensor]:
    """Return the physical signals used by reward, diagnostics, and termination."""
    c = point_in_pan_frame(cube_pos_w, pan_pos_w, pan_quat_w)
    s = point_in_pan_frame(cube_start_w, pan_pos_w, pan_quat_w)
    half = geometry.cube_half
    fully_inside = (
        (c[:, 0].abs() + half <= geometry.pan_half_width)
        & (c[:, 2] - half >= geometry.pan_inside_z_min)
        & (c[:, 2] + half <= geometry.pan_mouth_z)
        & (c[:, 1] >= geometry.pan_center_y_min)
        & (c[:, 1] <= geometry.pan_center_y_max)
    )
    # Operational task success: the complete lateral footprint is within the
    # mouth and the cube centre has crossed the lip plane.  ``fully_inside``
    # remains the stricter diagnostic that additionally requires the rear face
    # to clear the lip; it is intentionally not the termination condition.
    entered = (
        (c[:, 0].abs() + half <= geometry.pan_half_width)
        & (c[:, 2] - half >= geometry.pan_inside_z_min)
        & (c[:, 2] <= geometry.pan_mouth_z)
        & (c[:, 1] >= geometry.pan_center_y_min)
        & (c[:, 1] <= geometry.pan_center_y_max)
    )
    deep_target_z = geometry.pan_mouth_z - half - geometry.deep_inside_margin
    deep_inside = fully_inside & (c[:, 2] <= deep_target_z)
    rel_speed = torch.linalg.vector_norm(cube_vel_w - pan_vel_w, dim=-1)
    moved = torch.linalg.vector_norm(cube_pos_w - cube_start_w, dim=-1)
    # Potential is 0 at the fixed start and 1 once the cube centre clears the lip.
    denom = (s[:, 2] - geometry.pan_mouth_z).clamp_min(geometry.start_outside * 0.5)
    progress = ((s[:, 2] - c[:, 2]) / denom).clamp(0.0, 1.0)
    lateral_margin = geometry.pan_half_width - half - c[:, 0].abs()
    corridor = (lateral_margin / (geometry.pan_half_width - half)).clamp(0.0, 1.0)
    height_ok = ((c[:, 1] >= geometry.pan_center_y_min)
                 & (c[:, 1] <= geometry.pan_center_y_max))
    deep_progress = ((geometry.pan_mouth_z - c[:, 2]) /
                     max(geometry.pan_mouth_z - deep_target_z, 1.0e-6)).clamp(0.0, 1.0)
    deep_progress = deep_progress * (lateral_margin >= 0.0).float() * height_ok.float()
    deep_margin = deep_target_z - c[:, 2]
    return {
        "cube_pan": c,
        "entered": entered,
        "fully_inside": fully_inside,
        "deep_inside": deep_inside,
        "deep_progress": deep_progress,
        "deep_margin": deep_margin,
        "rel_speed": rel_speed,
        "moved": moved,
        "progress": progress,
        "corridor": corridor,
    }


class SweepProgressBatch:
    """Four monotonic milestones with immediate entry success."""

    def __init__(self, num_envs: int, device: str | torch.device,
                 geometry: SweepGeometry = SweepGeometry()):
        self.geometry = geometry
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.gates = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.stable_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_progress = torch.zeros(self.num_envs, device=self.device)
        self.prev_deep_progress = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor, initial_progress: torch.Tensor | None = None):
        self.gates[env_ids] = False
        self.stable_run[env_ids] = 0
        self.prev_progress[env_ids] = (0.0 if initial_progress is None
                                      else initial_progress)
        self.prev_deep_progress[env_ids] = 0.0

    def step(self, signals: dict[str, torch.Tensor], ready: torch.Tensor,
             broom_near: torch.Tensor) -> dict[str, torch.Tensor]:
        old = self.gates.clone()
        self.gates[:, 0] |= ready
        self.gates[:, 1] |= self.gates[:, 0] & broom_near & (
            signals["moved"] >= self.geometry.moved_gate)
        # Restored 15M contract: first valid entry is operational success.
        self.gates[:, 2] |= self.gates[:, 1] & signals["entered"]
        self.gates[:, 3] |= self.gates[:, 2]
        stable_now = signals["fully_inside"] & (
            signals["rel_speed"] <= self.geometry.stable_speed)
        self.stable_run = torch.where(stable_now, self.stable_run + 1,
                                      torch.zeros_like(self.stable_run))
        # Earn-only shaping: no positive income for holding still or oscillating.
        delta = (signals["progress"] - self.prev_progress).clamp_min(0.0)
        self.prev_progress = torch.maximum(self.prev_progress, signals["progress"])
        # Deep progress remains diagnostics-only under the restored contract.
        deep_delta = torch.zeros_like(delta)
        new_gate = self.gates & ~old
        reward = 4.0 * delta * signals["corridor"]
        reward = reward + 0.5 * new_gate[:, 0] + 1.0 * new_gate[:, 1]
        reward = reward + 4.0 * new_gate[:, 2] + 12.0 * new_gate[:, 3]
        return {
            **signals,
            "gates": self.gates.clone(),
            "new_gate": new_gate,
            "stable_run": self.stable_run.clone(),
            "success": self.gates[:, 3].clone(),
            "task_reward": reward,
            "deep_delta": deep_delta,
        }
