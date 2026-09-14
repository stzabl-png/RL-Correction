"""Per-dataset task specs for ``sweep_env`` (2026-09-13).

Everything in ``sweep_env.py`` that encodes *one particular* dustpan/broom scan
(pan-frame geometry, bristle mask, open pan collider, lip points, fixed cube
start, clip + priors) lives here.  ``SWEEP_TASK_SPEC=sweep2`` (default) is
bit-identical to the 2026-09-03 published runs; ``sweep408`` re-derives the same
quantities from the 408 scans (tasks/Sweep/408 DECISIONS.md §9).

Pan local frame convention, verified on both scans: ``x`` = width, ``+y`` = up,
``+z`` = handle -> open lip.  Broom: ``+z`` toward the head, bristle face at ``y_min``.
"""
from __future__ import annotations

import os
import dataclasses
from dataclasses import dataclass

from progress_batch import SweepGeometry

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_TASK, "..", "..", ".."))


@dataclass(frozen=True)
class SweepTaskSpec:
    name: str
    clip: str                       # clips.CLIPS key (primary = broom / right hand)
    prior_broom: str                # repo-relative
    prior_pan: str
    reference: str                  # default tape (absolute)
    world_task: str
    fixed_cube_start: tuple | None  # env-local world start; None -> tape ``cube_start_w``
    geometry: SweepGeometry
    bristle_y_max: float            # mesh-local mask: y < y_max & z_min < z < z_max
    bristle_z_min: float
    bristle_z_max: float
    pan_boxes: dict                 # name -> (center_xyz, size_xyz, rotate_x_deg), pan-local
    lip_local: tuple                # two pan-local points = lowest edge of the entry wedge
    pan_registration_max_m: float = 0.010
    # 基类 (tasks/pregrasp/env.py) 的先验摆放门: sweep2 用 yaw=0 直接过; 408 的静置面 != Dexonomy 规范面,
    # 必须 canon_rest_override=True + prior_yaw=-1 (自搜可达 yaw), 与 Sweep408 grip_env 同款 (台账 §1)
    canon_rest_override: bool = False
    prior_yaw: float = 0.0
    # 复位审计: 焊好的工具 vs 母带第0行位姿 的容差, 与臂下垂精修轮数。sweep2 = 3mm/4 轮 (原值);
    # 408 工具更重 (0.15/0.20kg) 臂姿不同, 首次审计 右 4.8mm / 左 3.0mm, 放到 8mm + 8 轮
    attach_audit_max_m: float = 0.003
    attach_refine_rounds: int = 4


SWEEP2 = SweepTaskSpec(
    name="sweep2", clip="Sweep2_broom",
    prior_broom="tasks/pregrasp/priors/Sweep2_broom.npz",
    prior_pan="tasks/pregrasp/priors/Sweep2_dustpan.npz",
    reference=os.path.join(_TASK, "A_Design", "L2_Reference", "sweep2_reference_v1.npz"),
    world_task="Sweep2_fixed_cube_fullinside",
    fixed_cube_start=(-0.0259767957, -0.1788897067, 0.8830000162),
    geometry=SweepGeometry(),
    bristle_y_max=-0.050, bristle_z_min=0.020, bristle_z_max=0.090,
    pan_boxes={
        "floor": ((0.0, 0.0065, 0.0475), (0.120, 0.004, 0.065), 0.0),
        "ramp": ((0.0, 0.00625396, 0.09391105), (0.120, 0.002, 0.02811138), 5.102165),
        "side_l": ((-0.062, 0.020, 0.055), (0.004, 0.028, 0.080), 0.0),
        "side_r": ((+0.062, 0.020, 0.055), (0.004, 0.028, 0.080), 0.0),
        "back": ((0.0, 0.020, 0.015), (0.124, 0.028, 0.004), 0.0),
    },
    lip_local=((-0.060, 0.004, 0.108), (0.060, 0.004, 0.108)),
)

# ---- 408 (tasks/Sweep/408 scans, tape v3).  Derived 2026-09-13 from the .obj profiles:
#   pan: handle z<-4.5cm (2.3-2.8cm wide), basin z -4.0..+5.7 (18.2cm wide at the lip),
#        basin top skin y=+5mm (Sweep2: +7..8.5), underside -9..-16mm, scan lip curls up to
#        +8..+15mm over z 5.3..5.8 (same false-lip artefact Sweep2 had) -> mesh collision
#        disabled, open compound collider below.  Along tape v3 the low lip corner sits
#        6.2-8.5mm above the table, floor top ~24-32mm (Sweep2: 10-19mm / 21-37mm).
#   broom: head z 0..11.7cm, bristle face y=-5.2..-4.6, 3.8cm wide (Sweep2 7.8cm).
_G408 = SweepGeometry(
    cube_half=0.0125, pan_half_width=0.070, pan_inside_z_min=-0.030, pan_mouth_z=0.047,
    pan_center_y_min=0.014, pan_center_y_max=0.027, start_outside=0.065)
SWEEP408 = SweepTaskSpec(
    name="sweep408", clip="Sweep408_broom",
    prior_broom="tasks/pregrasp/priors/Sweep408_broom.npz",
    prior_pan="tasks/pregrasp/priors/Sweep408_dustpan.npz",
    reference=os.path.join(_ROOT, "tasks/Sweep/408/A_Design/L2_Reference/sweep408_sweep2fmt_v3.npz"),
    world_task="Sweep408_fixed_cube_fullinside",
    fixed_cube_start=None,
    geometry=_G408,
    bristle_y_max=-0.046, bristle_z_min=0.005, bristle_z_max=0.110,
    pan_boxes={
        # floor top = y+5mm over z -3.5..+3.0; ramp drops to +2.5mm at the lip z=+5.7
        "floor": ((0.0, 0.003, -0.0025), (0.150, 0.004, 0.065), 0.0),
        "ramp": ((0.0, 0.00275, 0.0435), (0.150, 0.002, 0.0272), 5.29),
        "side_l": ((-0.076, 0.020, 0.005), (0.004, 0.030, 0.090), 0.0),
        "side_r": ((+0.076, 0.020, 0.005), (0.004, 0.030, 0.090), 0.0),
        "back": ((0.0, 0.020, -0.040), (0.156, 0.030, 0.004), 0.0),
    },
    lip_local=((-0.070, 0.0005, 0.057), (0.070, 0.0005, 0.057)),
    canon_rest_override=True, prior_yaw=-1.0,
    attach_audit_max_m=0.008, attach_refine_rounds=8,
)

# ---- 175 (2026-09-13): 同一批实物, 网格/先验/几何全部沿用 408, 只换 clip 与母带
SWEEP175 = dataclasses.replace(
    SWEEP408, name="sweep175", clip="Sweep175_broom",
    reference=os.path.join(_ROOT, "tasks/Sweep/175/A_Design/L2_Reference/sweep175_sweep2fmt_v1.npz"),
    world_task="Sweep175_fixed_cube_fullinside", prior_yaw=110.0)   # 408 自搜到的 yaw; 175 自搜 150° 后 IK 不可达 (台账 175 §6)

SPECS = {"sweep2": SWEEP2, "sweep408": SWEEP408, "sweep175": SWEEP175}


def resolve_spec(name: str | None = None) -> SweepTaskSpec:
    key = (name or os.environ.get("SWEEP_TASK_SPEC") or "sweep2").strip().lower()
    if key not in SPECS:
        raise ValueError(f"unknown SWEEP_TASK_SPEC {key!r}; choose {sorted(SPECS)}")
    return SPECS[key]
