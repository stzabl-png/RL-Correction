"""Three-stage grasp poses, computed at synthesis time.

Every written grasp record carries three wrist+finger poses instead of one:

- ``pregrasp``: the mid-optimization snapshot taken the moment the staged
  contact cost enters its middle (1cm-standoff) stage -- i.e. the pose
  optimized under the initial ~2cm-standoff target, the loosest of the three
  staged targets and the end of the optimizer's first (force-closure-scored)
  phase. This reproduces BODex's own ``save_qpos``/``mid_result`` mechanism;
  since the optimizer loop lives in the frozen ``bodex_curobo_v2`` module,
  the snapshot is taken by *subclassing* its optimizer core
  (:class:`SnapshotBodexNewtonOpt`), never by modifying it. The raw snapshot
  is captured at ``opt_progress == 0.6`` where the in-optimization penetration
  penalty has not yet ramped in, so it frequently penetrates the object
  (measured: object-dependent, some seeds 5-18mm inside). It is therefore
  *opened out of collision in joint space, per finger*: the wrist pose is
  kept EXACTLY as optimized (it is the approach pose the whole trajectory is
  built around) and each finger's ``_pregrasp_open_mask`` channels -- the
  flexion channels (``_FE``/``_PIP``/``_DIP``/``_IP``) plus BOTH thumb-CMC
  DoFs (``thumb_CMC_AA`` opens alongside ``thumb_CMC_FE``); the remaining
  spread/AA channels frozen -- are scaled toward 0 rad *only as far as that
  finger needs* to clear the object by ``pregrasp_clearance_m`` (default
  5mm). Each finger is searched independently against only the spheres it
  moves; the palm is ignored (no joint opens it), so a finger that already
  clears keeps its grasp posture and a penetrating palm never splays the
  fingers. The squeeze delta below is computed from the UN-opened snapshot
  so this opening never inflates the squeeze extrapolation.
- ``grasp``: the fully optimized final action, unmodified (identical to the
  record's ``action``). Usually still slightly penetrates the object (the
  staged contact cost's final target distance is 0; the in-optimization
  penetration penalty bounds but does not eliminate the overshoot) -- the
  simulation's soft drives absorb the overlap as contact force.
- ``squeeze``: Articulation-BODex extrapolation --
  ``grasp + clamp(grasp - snapshot, min=squeeze_min_rad)`` per joint
  (snapshot = the pregrasp BEFORE the joint-space opening above): the delta
  is the optimizer's OWN full closing motion (its intended force direction),
  the floor applied to flexion channels only (never abduction/adduction),
  so joints that barely moved to reach contact (typically the thumb) still
  get a minimum drive-through force request while joints that closed a lot
  get proportionally more.

All poses are full 29-D actions ``[pos(3), quat_wxyz(4), joints(22)]`` in
the object canonical frame, full asset joint order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ocir.grasp_synthesis.anchored_bodex.seed_generator import relax_joint_mask
from ocir.grasp_synthesis.bodex_curobo_v2.newton_opt import (
    BodexNewtonOpt,
    BodexProgressGradientOptCore,
)
from ocir.grasp_synthesis.clearance import ClearanceChecker

DEFAULT_SQUEEZE_MIN_RAD = 0.15
#: Target SDF clearance for the opened pregrasp pose: an approach pose that
#: must sit comfortably clear of the object, not skim it.
DEFAULT_PREGRASP_CLEARANCE_M = 0.005
_PREGRASP_OPEN_SAMPLES = 101


_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _finger_of(name: str) -> str | None:
    """Finger a joint/link name belongs to, or None for the palm base."""

    for finger in _FINGERS:
        if finger in name:
            return finger
    return None


def _pregrasp_open_mask(joint_order: list[str]) -> np.ndarray:
    """Channels the pregrasp clearance search opens toward 0 rad: the flexion
    channels plus BOTH thumb-CMC DoFs -- ``thumb_CMC_AA`` opens alongside
    ``thumb_CMC_FE`` so the whole thumb column can move off the object (user
    decision, 2026-07-15). Distinct from ``relax_joint_mask`` on purpose:
    that mask also drives the squeeze stage's flexion-only floor and the
    seed relaxation, whose semantics are unchanged."""

    mask = relax_joint_mask(joint_order)
    for i, name in enumerate(joint_order):
        if name.endswith("thumb_CMC_AA"):
            mask[i] = True
    return mask


class _SnapshotGradientOptCore(BodexProgressGradientOptCore):
    """Optimizer core that records the per-problem iteration centers the
    first time ``opt_progress`` reaches ``snapshot_progress`` -- the same
    quantity BODex appends to ``mid_result`` via its ``save_qpos`` hook."""

    snapshot_progress: float = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_actions: torch.Tensor | None = None

    def _opt_step(self, iteration_state):
        if (
            self.snapshot_actions is None
            and getattr(self, "_bodex_progress", 0.0) >= self.snapshot_progress
        ):
            # (num_problems, action_horizon, action_dim) -- the current
            # centers, before this step moves them under the new stage's cost.
            self.snapshot_actions = iteration_state.action.detach().clone()
        return super()._opt_step(iteration_state)

    def _prepare_initial_iteration_state(self, q):
        self.snapshot_actions = None
        return super()._prepare_initial_iteration_state(q)


class SnapshotBodexNewtonOpt(BodexNewtonOpt):
    """Drop-in ``BodexNewtonOpt`` whose core snapshots the pregrasp actions.

    Reimplements only ``__init__`` (the parent hard-codes its core class);
    everything else -- step direction, momentum, the delegated protocol --
    is inherited unchanged.
    """

    def __init__(self, config, rollout_list: list, use_cuda_graph: bool = False, *, snapshot_progress: float):
        self._momentum_buf = None
        self._core = _SnapshotGradientOptCore(
            config,
            rollout_list,
            step_direction_fn=self._get_step_direction_impl,
            on_reinitialize=self._on_reinitialize,
            on_initial_state=None,
            on_resize=self._on_resize,
            on_shift=None,
            use_cuda_graph=use_cuda_graph,
        )
        self._core.snapshot_progress = float(snapshot_progress)
        self._core.update_num_problems(config.num_problems)
        self._core.finish_init()

    @property
    def pregrasp_actions(self) -> torch.Tensor | None:
        """(num_problems, action_horizon, action_dim) snapshot, or None if
        the optimization never reached ``snapshot_progress``."""

        return self._core.snapshot_actions


def _open_pregrasp(
    pregrasp_action: np.ndarray,
    open_mask: np.ndarray,
    joint_order: list[str],
    clearance_checker: ClearanceChecker,
    world,
    *,
    clearance_target_m: float,
    num_samples: int = _PREGRASP_OPEN_SAMPLES,
) -> tuple[np.ndarray, dict[str, float], float, float]:
    """Open the pregrasp out of collision, **per finger**: each finger's own
    ``open_mask`` joints are scaled toward 0 rad -- wrist pose and every other
    finger frozen -- only as far as that finger needs to clear the object.

    Each finger is searched independently (the wrist is fixed, so opening one
    finger cannot move another's spheres): for finger F we sweep only F's
    open-mask joints from the snapshot toward 0, reduce the SDF clearance over
    **only the spheres F actually moves** (identified from the sweep, so a
    metacarpal that a frozen CMC leaves in place is excluded), and take the
    smallest fraction whose min clearance reaches ``clearance_target_m``.
    Palm spheres never enter any finger's reduction (no joint opens them), so
    a penetrating palm -- a wrist-placement problem -- does not force the
    fingers open. Fingers that already clear stay at their grasp values.

    Returns ``(opened_action, {finger: open_fraction}, snapshot_clearance_m,
    opened_clearance_m)`` where the two clearances are the min over all
    openable spheres, before and after."""

    joints = pregrasp_action[7:]
    joint_finger = [_finger_of(name) for name in joint_order]
    device = clearance_checker.device_cfg

    opened = pregrasp_action.copy()
    open_fractions: dict[str, float] = {}
    movable_any = np.zeros(len(clearance_checker.sphere_link_names), dtype=bool)
    ts = np.linspace(0.0, 1.0, int(num_samples))

    for finger in _FINGERS:
        jidx = np.asarray(
            [i for i in range(len(joint_order)) if joint_finger[i] == finger and open_mask[i]],
            dtype=int,
        )
        if jidx.size == 0:
            open_fractions[finger] = 0.0
            continue
        # Sweep: only this finger's joints scale toward 0; all else frozen.
        path = np.tile(pregrasp_action[None], (ts.size, 1)).astype(np.float64)
        path[:, 7 + jidx] = joints[jidx][None] * (1.0 - ts[:, None])
        sph = (
            clearance_checker.compute_sphere_clearances(
                device.to_device(path.astype(np.float32)), world
            )
            .cpu()
            .numpy()
        )  # (num_samples, N)
        # Spheres this finger actually moves (their clearance varies across the
        # sweep); kinematically these are exactly F's descendant spheres.
        moved = sph.std(axis=0) > 1e-6
        movable_any |= moved
        cols = np.where(moved)[0]
        if cols.size == 0:
            open_fractions[finger] = 0.0
            continue
        finger_clear = sph[:, cols].min(axis=1)
        ok = np.where(finger_clear >= float(clearance_target_m))[0]
        if ok.size == 0:
            idx = int(ts.size - 1)
            print(
                f"[anchored_bodex] WARNING: {finger} finger cannot clear the object even fully "
                f"open (best {finger_clear[-1]:.4f} m); likely a wrist-placement problem -- using "
                "fully open"
            )
        else:
            idx = int(ok[0])
            if idx > 0:
                print(
                    f"[anchored_bodex] {finger} finger opened {ts[idx]:.2f} of the way to clear "
                    f"the object ({finger_clear[0]:.4f} m -> {finger_clear[idx]:.4f} m)"
                )
        open_fractions[finger] = float(ts[idx])
        opened[7 + jidx] = joints[jidx] * (1.0 - ts[idx])

    # Report clearance over the openable spheres only (palm excluded), before
    # and after -- a whole-hand min that a frozen penetrating palm can't skew.
    def _movable_min(action: np.ndarray) -> float:
        sph = (
            clearance_checker.compute_sphere_clearances(
                device.to_device(action[None].astype(np.float32)), world
            )
            .cpu()
            .numpy()[0]
        )
        return float(sph[movable_any].min()) if movable_any.any() else float(sph.min())

    return opened, open_fractions, _movable_min(pregrasp_action), _movable_min(opened)


@dataclass(frozen=True)
class GraspStages:
    pregrasp: np.ndarray  # (29,) full action, opened clear of the object
    grasp: np.ndarray     # (29,) the fully optimized final action, as-is
    squeeze: np.ndarray   # (29,)
    report: dict


def compute_grasp_stages(
    raw_action: np.ndarray,
    pregrasp_action: np.ndarray,
    joint_order: list[str],
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
    clearance_checker: ClearanceChecker,
    world,
    *,
    squeeze_min_rad: float = DEFAULT_SQUEEZE_MIN_RAD,
    pregrasp_clearance_m: float = DEFAULT_PREGRASP_CLEARANCE_M,
) -> GraspStages:
    """Derive the opened ``pregrasp`` and extrapolated ``squeeze`` stages
    from the raw final action and the pregrasp snapshot (both full 29-D,
    object canonical frame). ``grasp`` is the raw final action verbatim."""

    grasp_action = np.asarray(raw_action, dtype=np.float64).copy()
    pregrasp_snapshot = np.asarray(pregrasp_action, dtype=np.float64).copy()
    flex_mask = relax_joint_mask(joint_order)

    # -- Pregrasp opening: the stage-0 snapshot is captured before the
    # penetration penalty ramps in and frequently sits inside the object.
    # The wrist pose is kept EXACTLY as optimized (it is the approach pose
    # the trajectory is built around); each finger's _pregrasp_open_mask
    # channels (flexion + both thumb-CMC DoFs) are opened toward 0 rad,
    # per-finger, only as far as that finger needs to clear the object. --
    pregrasp_opened, pregrasp_open_fractions, snapshot_clearance_m, pregrasp_clearance = _open_pregrasp(
        pregrasp_snapshot,
        _pregrasp_open_mask(joint_order),
        joint_order,
        clearance_checker,
        world,
        clearance_target_m=pregrasp_clearance_m,
    )

    grasp_clearance_m = clearance_checker.min_clearance_m(grasp_action.astype(np.float32), world)

    # -- Squeeze: Articulation-BODex extrapolation with a flexion-only floor.
    # The delta is the optimizer's FULL closing motion (grasp - snapshot),
    # computed from the UN-opened snapshot so the pregrasp opening above
    # never inflates the squeeze extrapolation. --
    delta = grasp_action[7:] - pregrasp_snapshot[7:]
    delta = np.where(flex_mask, np.clip(delta, squeeze_min_rad, None), delta)
    squeeze_action = grasp_action.copy()
    squeeze_action[7:] = np.clip(grasp_action[7:] + delta, joint_limits_lower, joint_limits_upper)

    report = {
        "pregrasp_snapshot_clearance_m": snapshot_clearance_m,
        "pregrasp_open_fractions": {k: float(v) for k, v in pregrasp_open_fractions.items()},
        "pregrasp_open_fraction_max": float(max(pregrasp_open_fractions.values(), default=0.0)),
        "pregrasp_clearance_target_m": float(pregrasp_clearance_m),
        "pregrasp_clearance_m": float(pregrasp_clearance),
        "grasp_clearance_m": float(grasp_clearance_m),
        "squeeze_min_rad": float(squeeze_min_rad),
    }
    return GraspStages(
        pregrasp=pregrasp_opened,
        grasp=grasp_action,
        squeeze=squeeze_action,
        report=report,
    )


def stage_pose_dict(action: np.ndarray, joint_order: list[str]) -> dict:
    """Record-JSON form of one stage: named joints, wxyz orientation."""

    action = np.asarray(action, dtype=float)
    return {
        "position": action[:3].tolist(),
        "orientation": action[3:7].tolist(),
        "joints": {name: float(v) for name, v in zip(joint_order, action[7:])},
    }
