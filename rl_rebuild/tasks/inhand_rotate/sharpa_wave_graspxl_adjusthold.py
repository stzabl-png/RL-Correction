# v3a: ADJUSTHOLD-PERSISTENT -- persistent per-step hold + finger-coverage reward.
#
# Root diagnosis (MUST RESPECT):
#   PBRS (Ng-1999) is optimum-preserving. It cannot shift the optimum of the adjustment-only
#   objective (survive + centering + energy). v1 (linear Phi) converged to a 2-finger opposition
#   cradle -- the true cheapest optimum. v2 (mult Phi) shrank the n=2 gradient (Phi 0.575->0.225)
#   so the policy collapsed to n=0 open-palm-rest + drops. No Phi formula can fix this; potential
#   shaping only guides exploration toward what is already optimal.
#
#   Fix: NON-POTENTIAL per-step terms whose STEADY-STATE value makes a secure multi-finger
#   enclosing grasp the UNIQUE optimum:
#
#       r_hold = w_hold * held
#       r_cov  = w_cov * clip(n_engaged / K, 0, 1) * held
#
#   where held = (disp < hold_disp_gate), and disp = ||object_pos - object_default_pose[:3]||.
#
#   Per-step offline ordering (proven; see experiment plan):
#     4-finger enclosing (2.00) > 2-finger cradle (1.84) > palm-rest (1.72) > drop (-0.09)
#
#   Anti-hacking:
#     (1) GATE: r_cov = 0 whenever held = 0. Light finger taps that don't support the ball
#         → disp grows → held = 0 → no coverage reward.
#     (2) CAP: r_hold ≤ w_hold; r_cov ≤ w_cov. No unbounded accumulation.
#     (3) Torque + work penalties (inherited): penalise excessive squeezing.
#     (4) force_scale = 0.5 random perturbations (inherited): dislodge palm-rests and
#         2-finger cradles that are not load-bearing; genuine enclosing is perturbation-stable.
#
#   PBRS DISABLED (use_readiness_potential=False) to avoid the v2 gradient-shrink failure.
#   Rotation DISABLED (rotate_reward_scale=0.0): pure adjustment objective.
#   Gravity curriculum gated on DROP RATE ONLY (graspxl_orient_rotate_gate=-1.0).
#
#   Additive-only: new @configclass (subclass of SharpaWaveGraspXLAdjustRotateCfg) + new env
#   (subclass of SharpaWaveGraspXLAdjustRotateEnv, overrides only _get_rewards). No baseline
#   code or config is touched.

import torch
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjustrotate import (
    SharpaWaveGraspXLAdjustRotateCfg,
    SharpaWaveGraspXLAdjustRotateEnv,
)


@configclass
class SharpaWaveGraspXLAdjustHoldPersistentCfg(SharpaWaveGraspXLAdjustRotateCfg):
    """v3a: persistent per-step HOLD + COVERAGE reward to fix the cradle (v1) and
    palm-rest-drop (v2) failures of the PBRS-only variants.

    Core changes vs SharpaWaveGraspXLAdjustRotateCfg:
      - use_readiness_potential = False  (PBRS disabled; avoids v2 gradient-shrink failure)
      - rotate_reward_scale     = 0.0   (pure adjustment; same as AdjustOnly* variants)
      - graspxl_orient_rotate_gate=-1.0 (gravity curriculum gated on drop-rate ONLY; same fix
                                         as AdjustOnlyCurr -- without this, rotate_reward=0
                                         keeps gravity stalled forever at the 0.15 gate)
      - w_hold, w_cov, cov_K, hold_disp_gate  (new non-potential per-step reward parameters)

    Inherited unchanged from SharpaWaveGraspXLAdjustRotateCfg / SharpaWaveGraspXLSustainedCfg:
      obs_hand_gravity = True, observation_space = 195 (192 + 3 hand-frame gravity)
      pos_diff_penalty_scale = 0.0       (full finger-gaiting freedom; no pose pinning)
      center_reward_bounded = True       (exp(-disp/0.03)*1.0; bounded survival signal)
      object_linvel_penalty_scale = -0.5 (penalise ball drift preceding a drop)
      torque_penalty_scale = -0.1        (damp high-force jitter)
      work_penalty_scale = -0.5          (penalise energy)
      graspxl_init = "replay"            (pinch RESET from the GraspXL trajectory)
      graspxl_orient_rand = True         (SO(3) orientation randomisation, annealed curriculum)
      graspxl_orient_curriculum = True   (start at orient_start=0.3 rad, anneal toward pi)
      graspxl_curr_drop_thresh = 0.01    (gravity advances when drop_rate < 1%)
      force_scale = 0.5                  (random perturbations; dislodge weak holds)
      gravity_curriculum = True          (adaptive: -0.05 ramps toward -10 m/s^2)

    Train FROM SCRATCH (obs dim 195; same architecture as all GraspXL variants).
    """

    # ---- Disable PBRS (proven optimum-preserving; caused v1 cradle and v2 drop failure) ----
    use_readiness_potential: bool = False

    # ---- No rotation reward: pure adjustment objective ----
    rotate_reward_scale: float = 0.0

    # ---- Gravity curriculum: gate on DROP RATE ONLY (rotate_reward=0 -> gate must be -1) ----
    # Without this, the rotation gate (0.15 default) keeps gravity at -0.05 forever because
    # rotate_reward stays ~0 with rotate_reward_scale=0.
    graspxl_orient_rotate_gate: float = -1.0

    # ---- Persistent HOLD reward ----
    # Per-step hold survival bonus. Bounded in [0, w_hold].
    # Changes the optimum from "any strategy" to "hold the ball". Gating on held also means the
    # policy is penalised (via shorter episode length) for strategies that drop reliably.
    w_hold: float = 1.0

    # ---- Persistent COVERAGE reward ----
    # Per-step finger-coverage bonus. Bounded in [0, w_cov]. Gated on held.
    # clip(n_engaged / cov_K, 0, 1): n=1->0.33, n=2->0.67, n=3->1.0, n=4,5->1.0.
    # Strictly orders multi-finger enclosing > 2-finger cradle > palm-rest per step.
    w_cov: float = 0.5

    # Fingers required for full coverage score.
    # K=3: n=3+ earns full 0.5/step; n=2 earns 0.33/step; n=0 earns 0/step.
    cov_K: float = 3.0

    # Displacement gate (metres): ball must be within this of object_default_pose to earn
    # r_hold and r_cov. Default 0.10 m = graspxl_drop_disp (the episode termination criterion).
    # Using the same threshold means held=1 iff the episode is still running.
    # A STRICTER value (e.g. 0.06) creates additional gradient toward tighter holding, but may
    # suppress r_hold during legitimate regrasping transients; 0.10 is the safe default.
    hold_disp_gate: float = 0.10


class SharpaWaveGraspXLAdjustHoldPersistentEnv(SharpaWaveGraspXLAdjustRotateEnv):
    """Adds persistent HOLD + COVERAGE reward on top of the inherited base reward.

    The parent _get_rewards chain (called via super()) provides:
      - rotate (scale=0; off)
      - centering: exp(-disp/0.03)*1.0  (bounded; inherited from SharpaWaveGraspXLSustainedCfg)
      - linvel penalty (-0.5)
      - torque penalty (-0.1)
      - work penalty (-0.5)
      - PBRS: SKIPPED because use_readiness_potential=False (SharpaWaveGraspXLAdjustRotateEnv
              checks the flag at the top of its _get_rewards; sharpa_wave_graspxl_adjustrotate.py:77)

    This class adds r_hold and r_cov (non-potential, per-step, bounded).

    Signals referenced (all live in the base API; cite + line):
      self.object_pos              (N,3) env-relative ball position, updated each step
                                   sharpa_wave_env.py:428
      self.object_default_pose     (N,7) settled held-pose (env-relative), set at reset
                                   sharpa_wave_graspxl_env.py:316 (_reset_idx)
      self.last_contacts           (N,5) per-finger EMA contact force (N), updated each call
                                   to compute_observations -> contact sensor pipeline
                                   sharpa_wave_env.py:128 (buffer), 466-468 (update)
      self.cfg.phi_contact_thresh  0.1 N engagement threshold
                                   sharpa_wave_graspxl_adjustrotate.py:34
      self.cfg.hold_disp_gate      0.10 m (default = graspxl_drop_disp) -- new field
      self.cfg.w_hold              1.0 -- new field
      self.cfg.w_cov               0.5 -- new field
      self.cfg.cov_K               3.0 -- new field
    """

    def _get_rewards(self) -> torch.Tensor:
        # Call parent chain. Because use_readiness_potential=False, AdjustRotateEnv's _get_rewards
        # returns immediately after super()._get_rewards() (sharpa_wave_graspxl_adjustrotate.py:77-78).
        # What comes back: rotate(0) + linvel_pen + centering(bounded) + torque_pen + work_pen.
        total = super()._get_rewards()

        # ------------------------------------------------------------------
        # Displacement from the settled held-pose (orientation-invariant L2 distance).
        # Both tensors are env-relative world positions (scene.env_origins subtracted).
        # Source: sharpa_wave_env.py:428 (object_pos), sharpa_wave_graspxl_env.py:316 (default)
        # ------------------------------------------------------------------
        disp = torch.norm(
            self.object_pos - self.object_default_pose[:, :3], dim=-1
        )  # (N,)

        # Hard held gate: 1 if ball is within hold_disp_gate of the target, 0 otherwise.
        # Default hold_disp_gate = 0.10 m = graspxl_drop_disp, so held=1 iff the episode is
        # still active. Episode terminates when disp > graspxl_drop_disp (_orient_dones).
        gate = float(getattr(self.cfg, "hold_disp_gate", 0.10))
        held = (disp < gate).float()  # (N,) in {0.0, 1.0}

        # ------------------------------------------------------------------
        # r_hold: per-step hold/survival bonus.
        # Bounded: r_hold ∈ [0, w_hold].
        # Effect: policies that HOLD LONGER accumulate more r_hold across the episode.
        # At full gravity, palm-rest drops early (short episode); enclosing holds (long episode).
        # ------------------------------------------------------------------
        w_hold = float(getattr(self.cfg, "w_hold", 1.0))
        r_hold = w_hold * held  # (N,)

        # ------------------------------------------------------------------
        # r_cov: per-step finger-coverage bonus, gated on holding.
        # n_engaged: number of elastomers (thumb/index/middle/ring/pinky) with contact force
        #            strictly above phi_contact_thresh (0.1 N).
        # last_contacts (N,5): per-finger EMA-smoothed contact force from the contact sensor
        #   pipeline, updated in compute_observations (sharpa_wave_env.py:466-468).
        #   Threshold 0.1 N requires real pressing force; light grazing does not count.
        #
        # Anti-hacking:
        #   (1) GATE: held = 0 when ball drops → r_cov = 0. Tapping fingers without supporting
        #       the ball → ball drifts → disp > gate → held = 0 → no reward.
        #   (2) CAP: clip(n/K, 0, 1) ≤ 1 → r_cov ≤ w_cov. Squeezing harder gives no bonus
        #       beyond K engaged fingers (reward is COUNT-based, not force-magnitude-based).
        #   (3) Torque + work penalties (inherited, -0.1 / -0.5): penalise excessive squeezing
        #       force. The policy balances coverage bonus vs energy cost.
        #   (4) force_scale=0.5 random perturbations (inherited, _pre_physics_step): dislodge
        #       strategies that are not load-bearing; genuine multi-finger enclosing is stable.
        # ------------------------------------------------------------------
        thresh = float(getattr(self.cfg, "phi_contact_thresh", 0.1))
        n_engaged = (self.last_contacts > thresh).float().sum(dim=-1)  # (N,)
        K = float(getattr(self.cfg, "cov_K", 3.0))
        cov = (n_engaged / K).clamp(0.0, 1.0)  # (N,) in [0, 1]
        w_cov = float(getattr(self.cfg, "w_cov", 0.5))
        r_cov = w_cov * cov * held  # (N,); bounded in [0, w_cov]

        # Diagnostics logged to wandb via extras
        self.extras["r_hold"] = r_hold.mean()
        self.extras["r_cov"] = r_cov.mean()
        self.extras["n_engaged"] = n_engaged.mean()
        self.extras["held_frac"] = held.mean()
        self.extras["disp_mean"] = disp.mean()

        return total + r_hold + r_cov
