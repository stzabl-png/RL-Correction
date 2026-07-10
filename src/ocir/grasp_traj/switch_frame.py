"""Dynamic switch-frame selection: walk backward from a default
approach-duration frame until the retargeted hand pose is collision-safe.

Replaces any custom motion-planning machinery for the approach segment: the
approach itself (see ``segments.py``) is a plain standoff + straight-line
interpolation, so all the "safely reach the grasp pose without touching the
object too early" work happens here, once, by picking a good starting frame.
"""

from __future__ import annotations

import numpy as np
import torch

from ocir.grasp_synthesis.anchored_bodex.affordance import Affordance
from ocir.grasp_synthesis.anchored_bodex.demo_analysis import DemoGraspAnalysis
from ocir.grasp_synthesis.anchored_bodex.demo_data import HumanDemo
from ocir.grasp_synthesis.anchored_bodex.retarget import RetargetResult
from ocir.grasp_synthesis.bodex_curobo_v2.contact_world import SingleObjectContactWorld

from ocir.grasp_traj.clearance import ClearanceChecker

DEFAULT_APPROACH_CLEARANCE_M = 0.01


def select_switch_frame(
    demo: HumanDemo,
    analysis: DemoGraspAnalysis,
    retarget_all: RetargetResult,
    clearance_checker: ClearanceChecker,
    world: SingleObjectContactWorld,
    affordance: Affordance,
    *,
    approach_seconds: float,
    fps: float,
    clearance_m: float = DEFAULT_APPROACH_CLEARANCE_M,
    override_joints: np.ndarray | None = None,
) -> tuple[int, dict]:
    """Pick the demo frame at which retargeted replay hands off to the
    synthetic approach segment.

    ``retarget_all.frame_indices`` must cover (at least) every valid frame up
    to and including ``analysis.grasp_frame_index`` -- the candidate pool this
    function walks backward through.

    ``override_joints`` (len == the action's joint tail), when given, replaces
    each candidate's retargeted joints for the clearance check -- pass the
    wide-open pregrasp posture so the chosen frame is one where the hand can
    already be FULLY OPEN without touching the object.
    """

    grasp_frame = int(analysis.grasp_frame_index)
    default_frame = grasp_frame - int(round(float(approach_seconds) * float(fps)))
    default_frame = max(0, min(default_frame, grasp_frame - 1 if grasp_frame > 0 else 0))

    frame_to_row = {int(f): i for i, f in enumerate(retarget_all.frame_indices)}
    candidates = [f for f in range(default_frame, -1, -1) if f in frame_to_row]
    if not candidates:
        raise ValueError(
            f"no retargeted candidate frames at or before the default switch frame {default_frame} "
            f"(retarget_all covers frames {sorted(frame_to_row)})"
        )

    first_contact_frame = None
    if affordance.frame_in_contact.any():
        first_contact_frame = int(np.argmax(affordance.frame_in_contact))

    actions_np = np.stack([retarget_all.ref_actions[frame_to_row[f]] for f in candidates])
    if override_joints is not None:
        actions_np = actions_np.copy()
        actions_np[:, 7:] = np.asarray(override_joints, dtype=actions_np.dtype)[None]
    actions = torch.as_tensor(
        actions_np,
        device=clearance_checker.device_cfg.device,
        dtype=clearance_checker.device_cfg.dtype,
    )
    clearances = clearance_checker.compute_clearances(actions, world).detach().cpu().numpy()

    chosen = None
    chosen_clearance = None
    walked = 0
    for walked, (frame, clearance) in enumerate(zip(candidates, clearances)):
        contact_ok = first_contact_frame is None or frame < first_contact_frame
        if clearance >= clearance_m and contact_ok:
            chosen = frame
            chosen_clearance = float(clearance)
            break

    satisfied = chosen is not None
    if not satisfied:
        chosen = default_frame
        chosen_clearance = float(clearances[0]) if len(clearances) else None
        walked = 0

    report = {
        "default_frame": int(default_frame),
        "grasp_frame_index": grasp_frame,
        "chosen_frame": int(chosen),
        "clearance_m": chosen_clearance,
        "clearance_threshold_m": float(clearance_m),
        "frames_walked_back": int(walked),
        "first_contact_frame": None if first_contact_frame is None else int(first_contact_frame),
        "clearance_satisfied": bool(satisfied),
        "num_candidates_checked": int(len(candidates)),
    }
    return int(chosen), report
