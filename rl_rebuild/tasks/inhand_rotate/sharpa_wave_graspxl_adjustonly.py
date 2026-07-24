# ADJUSTMENT-ONLY validation task (additive; NO baseline edits). Trains a policy to CONVERT the imported
# GraspXL fingertip PINCH into a rotation-READY (enclosing) grasp with *NO rotation reward at all* -- the
# policy is paid ONLY by the validated rotation-readiness potential Phi (plus the inherited survival /
# centering / regularization terms). This isolates the pure "adjustment" behavior so we can later EXTRACT
# the learned adjusted grasps (rl_rebuild/scripts/extract_adjusted_poses.py) and rotate FROM them vs FROM
# the pinch, to test whether adjustment actually helps rotation.
#
# This file is SELF-CONTAINED: a new cfg variant that subclasses SharpaWaveGraspXLAdjustRotateCfg. It adds
# NO new env class -- the registered task reuses the existing SharpaWaveGraspXLAdjustRotateEnv (pinch reset
# via graspxl_init="replay" + readiness-Phi shaping). No baseline cfg/env is edited.
#
# Design vs the parent SharpaWaveGraspXLAdjustRotateCfg:
#   - rotate_reward_scale 3.0 -> 0.0   : NO rotation reward. Pure adjustment. This is the whole point of
#     the experiment: the ONLY task-relevant signal is the readiness potential Phi.
#   - gravity_curriculum   -> False    : adaptive gravity curriculum OFF (mirrors the Diverse-FixedG variant
#     in sharpa_wave_graspxl_adjustrotate_diverse.py).
#   - sim.gravity          -> -4.9     : FIXED half-g (same value + same override pattern as Diverse-FixedG)
#     so that a good (enclosing) grasp actually matters during adjustment and the later rotation probe is
#     gravity-matched.
#   INHERITED, unchanged: readiness-Phi potential (use_readiness_potential=True, w_phi=3.0,
#   phi_contact_thresh=0.1, phi_smooth=0.5, gamma_phi=0.99); pos_diff_penalty_scale=0.0; torque control;
#   obs dim 195 (192 proprio/tactile + hand-frame gravity 3); GraspXL REPLAY (PINCH) reset -- the
#   AdjustRotate env's DEFAULT reset (NO diverse enclosing mix, NO friction combine="max").
# Train FROM SCRATCH (obs dim 195, same as AdjustRotate).
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjustrotate import SharpaWaveGraspXLAdjustRotateCfg


@configclass
class SharpaWaveGraspXLAdjustOnlyFixedGCfg(SharpaWaveGraspXLAdjustRotateCfg):
    """Adjustment-only cfg: readiness-Phi reward with rotate_reward_scale=0 (NO rotation reward), at a
    FIXED gravity of -4.9 with the adaptive gravity curriculum OFF. Reset is from the PINCH (inherited
    GraspXL replay init); readiness-Phi potential + pos_diff=0 are inherited from AdjustRotate. The
    registered task reuses SharpaWaveGraspXLAdjustRotateEnv. Train FROM SCRATCH (obs dim 195)."""
    # ---- NO rotation reward: the policy is driven ONLY by the readiness-Phi potential + survival terms ----
    rotate_reward_scale = 0.0
    # ---- FIXED gravity, curriculum OFF (same rationale + same override pattern as Diverse-FixedG) ----
    # With gravity_curriculum=False, BOTH curriculum blocks (sharpa_wave_env._get_dones and
    # sharpa_wave_graspxl_env._orient_dones) are skipped, so gravity stays at the fixed sim value below and
    # the SE(3) orientation cap stays at graspxl_orient_start (~0.3 rad, near palm-up).
    gravity_curriculum = False
    # Override the inherited SimulationCfg gravity (base = -0.05) to a FIXED -4.9. Same dt / render_interval
    # / physx as the base; ONLY gravity_z changes. configclass converts this class-level value into a
    # per-instance default_factory deepcopy, so it overrides the parent without contaminating it. The
    # __post_init__ below re-asserts the same values as a belt-and-suspenders guarantee.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -4.9),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608,   # 2**23
            gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )

    def __post_init__(self):
        # Guaranteed override even if the class-level redefines above were ever a no-op. self.sim is already
        # a per-instance deepcopy here, so this does not touch any base/shared default.
        self.sim.gravity = (0.0, 0.0, -4.9)
        self.gravity_curriculum = False
        self.rotate_reward_scale = 0.0


@configclass
class SharpaWaveGraspXLAdjustOnlyCurrCfg(SharpaWaveGraspXLAdjustRotateCfg):
    """Adjustment-only cfg with a DROP-GATED gravity CURRICULUM (user request: 'learn adjustment until
    -4.9 g or beyond'). rotate_reward_scale=0 (NO rotation reward; driven only by the readiness-Phi
    potential + survival). Unlike the FixedG variant, gravity is NOT fixed: the adaptive gravity curriculum
    is ON (inherited) and starts at the base -0.05, ramping toward -10 as the policy learns to HOLD.

    CRITICAL FIX vs the inherited (rotation-gated) curriculum: the GraspXL gravity curriculum normally
    advances only while drop_rate < graspxl_curr_drop_thresh AND rotate_reward > graspxl_orient_rotate_gate
    (sharpa_wave_graspxl_env.py ~L254-255). With NO rotation reward, rotate_reward stays ~0, so gravity
    would never advance. Setting graspxl_orient_rotate_gate = -1.0 makes rotate_ok always True, so gravity
    ramps on the DROP RATE ALONE -- it advances exactly as fast as the learned adjustment can hold the
    object at rising gravity. The FINAL gravity reached is a direct measure of adjustment robustness.
    (This same rotation-gate is why 3A/3B's gravity stalled: their rotate_reward 0.08-0.13 < the 0.15 gate.)
    Reset is from the PINCH (inherited replay init); readiness-Phi + pos_diff=0 inherited. Train FROM SCRATCH."""
    rotate_reward_scale = 0.0
    graspxl_orient_rotate_gate = -1.0   # disable rotation gate -> gravity curriculum gated on drop-rate ONLY
    # gravity_curriculum=True and sim.gravity=(0,0,-0.05) are inherited base defaults -> ramps from near-zero.


@configclass
class SharpaWaveGraspXLAdjustOnlyCurrV2Cfg(SharpaWaveGraspXLAdjustOnlyCurrCfg):
    """v2 regrip reward (fix the 2-finger-cradle failure): MULTIPLICATIVE readiness potential
    Φ = spread * coverage (coverage GATES spread), so the policy can no longer max Φ with 2 opposed
    fingers while parking the rest lifted — it must recruit more fingers into a supportive grasp.
    Everything else inherited from AdjustOnlyCurr (rotate_reward_scale=0, drop-gated gravity curriculum
    with the rotation-gate disabled, pinch reset). Train FROM SCRATCH."""
    phi_mode = "mult"
