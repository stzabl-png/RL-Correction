"""Four-stage grasp poses, computed at synthesis time.

Every written grasp record carries four wrist+finger poses instead of one:

- ``raw_grasp``: the fully optimized final action, unmodified. Usually
  penetrates the object by several mm (the staged contact cost's final
  target distance is 0 and the optimizer overshoots into the SDF).
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
  *retreated along the palm approach axis* -- the whole hand translated
  straight back, joints frozen -- to the smallest backoff whose full-hand SDF
  clearance reaches ``pregrasp_clearance_m`` (default 5mm). The result is a
  guaranteed collision-free pre-grasp pose; only the wrist position moves, so
  the squeeze delta (``raw_grasp - pregrasp`` joints, below) is unchanged.
- ``grasp``: the raw grasp retreated along the pregrasp -> raw_grasp
  interpolation path (lerp position/joints + slerp orientation) to the
  largest fraction whose full-hand SDF clearance is still >=
  ``contact_clearance_m`` (default 2mm) -- deliberately SHY of the contact
  boundary, matching UltraDexGrasp's stance that the tightest converged
  pose is never a configuration to physically reach. Actual contact force
  comes from the squeeze stage's drives, not from commanding a
  zero-clearance position (which also matches the simulation's summed
  hand+object rest offsets).
- ``squeeze``: Articulation-BODex extrapolation --
  ``grasp + clamp(raw_grasp - pregrasp, min=squeeze_min_rad)`` per joint:
  the delta is the optimizer's OWN full closing motion (its intended force
  direction, including the part the retreat removed), the floor applied to
  flexion channels only (never abduction/adduction), so joints that barely
  moved to reach contact (typically the thumb) still get a minimum
  drive-through force request while joints that closed a lot get
  proportionally more.

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
DEFAULT_CONTACT_CLEARANCE_M = 0.002
#: Target SDF clearance for the retreated pregrasp pose. Deliberately larger
#: than the grasp's contact clearance: the pregrasp is an approach pose that
#: must sit comfortably clear of the object, not skim it.
DEFAULT_PREGRASP_CLEARANCE_M = 0.005
#: Palm-frame approach axis; ``-(R @ axis)`` is the world-frame direction that
#: backs the wrist away from the object (matches grasp_traj's
#: PALM_APPROACH_AXIS_LOCAL and _project_wrist_pose).
_PALM_APPROACH_AXIS_LOCAL = np.asarray([1.0, 0.0, 0.0])
_DEFAULT_MAX_PREGRASP_BACKOFF_M = 0.06
_PREGRASP_BACKOFF_STEP_M = 0.002


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


def _wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Single wxyz quaternion -> 3x3 rotation matrix (same convention as
    grasp_traj.trajectory_schema.quat_wxyz_to_matrix; kept local so the
    synthesis package carries no dependency on the trajectory package)."""

    w, x, y, z = np.asarray(quat, dtype=np.float64) / max(np.linalg.norm(quat), 1e-12)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _retreat_pregrasp(
    pregrasp_action: np.ndarray,
    clearance_checker: ClearanceChecker,
    world,
    *,
    clearance_target_m: float,
    max_backoff_m: float,
    step_m: float,
) -> tuple[np.ndarray, float, float, float]:
    """Back the pregrasp wrist straight along the palm approach axis (joints
    frozen) to the smallest offset whose full-hand SDF clearance reaches
    ``clearance_target_m``. Returns ``(retreated_action, backoff_m,
    snapshot_clearance_m, retreated_clearance_m)``; caps at ``max_backoff_m``
    with a warning if even that does not clear."""

    back_dir = -(_wxyz_to_matrix(pregrasp_action[3:7]) @ _PALM_APPROACH_AXIS_LOCAL)
    offsets = np.arange(0.0, max_backoff_m + 1e-9, step_m)
    pos = pregrasp_action[:3][None] + offsets[:, None] * back_dir[None]
    quat = np.tile(pregrasp_action[3:7][None], (offsets.size, 1))
    joints = np.tile(pregrasp_action[7:][None], (offsets.size, 1))
    actions = np.concatenate([pos, quat, joints], axis=-1).astype(np.float32)
    clearances = (
        clearance_checker.compute_clearances(clearance_checker.device_cfg.to_device(actions), world)
        .cpu()
        .numpy()
    )
    snapshot_clearance = float(clearances[0])
    ok = np.where(clearances >= float(clearance_target_m))[0]
    if ok.size == 0:
        print(
            "[anchored_bodex] WARNING: pregrasp snapshot still penetrates after backing the wrist "
            f"off {max_backoff_m:.3f} m (clearance {clearances[-1]:.4f} m); using the cap"
        )
        idx = int(offsets.size - 1)
    else:
        idx = int(ok[0])
        if idx > 0:
            print(
                f"[anchored_bodex] pregrasp wrist backed off {offsets[idx] * 1000:.0f} mm to clear "
                f"the object (snapshot {snapshot_clearance:.4f} m -> {float(clearances[idx]):.4f} m)"
            )
    retreated = pregrasp_action.copy()
    retreated[:3] = pos[idx]
    return retreated, float(offsets[idx]), snapshot_clearance, float(clearances[idx])


def _slerp_wxyz(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2, dot = -q2, -dot
    if dot > 0.9995:
        out = q1 + t * (q2 - q1)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(min(dot, 1.0))
    q2_orth = q2 - q1 * dot
    q2_orth = q2_orth / np.linalg.norm(q2_orth)
    return q1 * np.cos(theta0 * t) + q2_orth * np.sin(theta0 * t)


def _interp_action(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    out = a * (1.0 - t) + b * t
    out[3:7] = _slerp_wxyz(a[3:7], b[3:7], t)
    return out


@dataclass(frozen=True)
class GraspStages:
    pregrasp: np.ndarray   # (29,) full action
    raw_grasp: np.ndarray  # (29,)
    grasp: np.ndarray      # (29,) retreated, non-penetrating contact
    squeeze: np.ndarray    # (29,)
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
    contact_clearance_m: float = DEFAULT_CONTACT_CLEARANCE_M,
    pregrasp_clearance_m: float = DEFAULT_PREGRASP_CLEARANCE_M,
    num_retreat_samples: int = 101,
) -> GraspStages:
    """Derive the retreated ``grasp`` and extrapolated ``squeeze`` stages
    from the raw final action and the pregrasp snapshot (both full 29-D,
    object canonical frame)."""

    raw_action = np.asarray(raw_action, dtype=np.float64).copy()
    pregrasp_snapshot = np.asarray(pregrasp_action, dtype=np.float64).copy()

    # -- Pregrasp retreat: the stage-0 snapshot is captured before the
    # penetration penalty ramps in and frequently sits inside the object;
    # back the wrist straight along the palm approach axis (joints frozen) to
    # a guaranteed-clear pre-grasp pose. Only the wrist position moves, so the
    # squeeze delta below (raw - pregrasp joints) is unaffected. --
    pregrasp_action, pregrasp_backoff_m, snapshot_clearance_m, _ = _retreat_pregrasp(
        pregrasp_snapshot,
        clearance_checker,
        world,
        clearance_target_m=pregrasp_clearance_m,
        max_backoff_m=_DEFAULT_MAX_PREGRASP_BACKOFF_M,
        step_m=_PREGRASP_BACKOFF_STEP_M,
    )

    # -- Retreat: largest t on the pregrasp -> raw path whose clearance is
    # still >= contact_clearance_m (shy of the contact boundary; squeeze
    # supplies the actual contact force) --
    ts = np.linspace(0.0, 1.0, int(num_retreat_samples))
    path = np.stack([_interp_action(pregrasp_action, raw_action, float(t)) for t in ts], axis=0)
    clearances = (
        clearance_checker.compute_clearances(
            clearance_checker.device_cfg.to_device(path.astype(np.float32)), world
        )
        .cpu()
        .numpy()
    )
    ok = np.where(clearances >= float(contact_clearance_m))[0]
    if ok.size == 0:
        if clearances[0] >= 0.0:
            # Expected with the in-optimization penetration penalty: the whole
            # pregrasp -> raw path skims the surface inside the margin, so the
            # snapshot itself becomes the commanded grasp (squeeze supplies
            # the contact force) -- the UltraDexGrasp stance exactly.
            print(
                "[anchored_bodex] grasp stage = pregrasp snapshot "
                f"(path clearance {clearances[0]:.4f}..{clearances[-1]:.4f} m, all under the "
                f"{contact_clearance_m:.4f} m margin)"
            )
        else:
            print(
                "[anchored_bodex] WARNING: even the pregrasp snapshot penetrates the object "
                f"(clearance {clearances[0]:.4f} m); using it as the contact grasp anyway"
            )
        retreat_t = 0.0
        grasp_action = pregrasp_action.copy()
    else:
        retreat_t = float(ts[ok[-1]])
        grasp_action = path[ok[-1]].copy()

    # -- Squeeze: Articulation-BODex extrapolation with a flexion-only floor.
    # The delta is the optimizer's FULL closing motion (raw - pregrasp), not
    # (grasp - pregrasp): the retreat may have removed most of the latter,
    # but the intended force direction is the whole pregrasp -> raw sweep. --
    flex_mask = relax_joint_mask(joint_order)
    delta = raw_action[7:] - pregrasp_action[7:]
    delta = np.where(flex_mask, np.clip(delta, squeeze_min_rad, None), delta)
    squeeze_action = grasp_action.copy()
    squeeze_action[7:] = np.clip(grasp_action[7:] + delta, joint_limits_lower, joint_limits_upper)

    report = {
        "retreat_fraction": retreat_t,
        "pregrasp_snapshot_clearance_m": snapshot_clearance_m,
        "pregrasp_backoff_m": pregrasp_backoff_m,
        "pregrasp_clearance_target_m": float(pregrasp_clearance_m),
        "pregrasp_clearance_m": float(clearances[0]),
        "raw_grasp_clearance_m": float(clearances[-1]),
        "grasp_clearance_m": float(clearances[ok[-1]]) if ok.size else float(clearances[0]),
        "contact_clearance_m": float(contact_clearance_m),
        "squeeze_min_rad": float(squeeze_min_rad),
        "wrist_retreat_m": float(np.linalg.norm(raw_action[:3] - grasp_action[:3])),
    }
    return GraspStages(
        pregrasp=pregrasp_action,
        raw_grasp=raw_action,
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
