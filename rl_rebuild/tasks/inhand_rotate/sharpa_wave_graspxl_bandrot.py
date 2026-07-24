# ADDITIVE anti-jitter BAND rotation reward (no baseline cfg/env is edited).
#
# WHY (the "rewarded jitter attractor"):
#   The baseline rotate term is  clip(omega_obj . rot_axis, -0.5, 0.5) * rotate_reward_scale
#   (sharpa_wave_env.py:276-277). With rotate_reward_scale>0 and a POSITIVE clip floor of 0, a tiny signed
#   jitter of omega~0.037 rad/s already earns positive reward. Every prior rotation run therefore converged
#   to a shaking/hold basin (signed yaw stuck ~0.037 rad/s) instead of a real sustained spin.
#
# FIX: replace the base clip rotate (rotate_reward_scale=0) with a BAND reward that is
#   (1) EXACTLY ZERO below a speed floor (jitter earns nothing),
#   (2) EMA-smoothed so a +-0.5 rad/s oscillation averages out to ~0 net signed speed (band ~0),
#   (3) adaptively up-weighted as the policy accumulates net revolutions (running mean of signed revs/ep),
#   (4) topped up by a SPARSE per-revolution bonus that can only be earned by NET signed progress
#       (integrated from the smoothed signed speed), never by oscillation.
#   The inherited anti-drop / anti-translate / energy terms (object_linvel penalty, torque, work, bounded
#   centering) are KEPT unchanged (HORA: without object_linvel the policy translates instead of rotating).
#
# Offline validation: rl_rebuild/scripts/bandrot_reward_unit_test.py (pure numpy; no sim) proves band==0 for
# constant 0.037 jitter, band>0 & monotonically increasing on omega in [0.15,3.0], +-0.5 square-wave -> band~0
# through the EMA, and the sparse bonus fires only on net cum-angle progress (not oscillation).
#
# Two registered tasks (see __init__.py), both reuse the SAME band logic via _BandRotMixin:
#   Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-Cylinder-v1  -> cylinder object + cylinder cache/init
#   Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-SphereEnc-v1 -> GraspXL ball 002aa185, enclosing-cache init
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul, axis_angle_from_quat

from .sharpa_wave_env import SharpaWaveInhandRotateEnv
from .sharpa_wave_env_cfg import SharpaWaveEnvCfg
from .sharpa_wave_graspxl_adjustrotate_diverse import (
    SharpaWaveGraspXLAdjustRotateDiverseCfg,
    SharpaWaveGraspXLAdjustRotateDiverseEnv,
)

_TWO_PI = 2.0 * math.pi


# ============================================================================
# Band-reward MIXIN (cooperative super(); works on top of ANY SharpaWave env).
# ============================================================================
class _BandRotMixin:
    """Adds the anti-jitter BAND rotation reward on top of the inherited per-step reward.

    Buffers are created AFTER super().__init__() (the base env allocates self.num_envs / self.device /
    self.hand first). _reset_idx and _get_rewards both use a hasattr guard because DirectRLEnv calls
    _reset_idx once DURING super().__init__() (before these buffers exist)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N = self.num_envs
        # per-env band state
        self.omega_ema = torch.zeros(N, device=self.device)            # EMA of signed omega.axis
        self.cum_angle = torch.zeros(N, device=self.device)            # integral of smoothed signed omega
        self._sparse_level = torch.zeros(N, device=self.device)        # ratcheted sparse-bonus level (floor(cum/theta))
        self._phys_cum = torch.zeros(N, device=self.device)            # PHYSICAL net rotation (all steps, NOT held-gated)
        self._phys_hw = torch.zeros(N, device=self.device)            # high-water mark of physical net rotation
        # adaptive-weight running mean of per-episode SIGNED revolutions
        self._mean_signed_revs_per_ep = torch.zeros((), device=self.device)
        self._rev_accum_sum = torch.zeros((), device=self.device)      # sum of finalized episode revs since last commit
        self._rev_accum_n = torch.zeros((), device=self.device)        # count of finalized episodes since last commit
        self._bandrot_step_count = 0

    def _get_rewards(self) -> torch.Tensor:
        # Parent chain: base rotate term is 0 here (rotate_reward_scale=0) + object_linvel penalty + torque
        # + work + bounded centering (+ Phi is skipped when use_readiness_potential=False).
        total = super()._get_rewards()
        if not hasattr(self, "omega_ema"):
            return total  # called before buffers exist (during base __init__)
        cfg = self.cfg

        # --- REUSE the base omega.axis computation verbatim (sharpa_wave_env.py:276-277) ---
        object_angvel = axis_angle_from_quat(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))
        ) / self.step_dt                                               # (N,3) rad/s
        omega_signed = (object_angvel * self.rot_axis).sum(-1)         # (N,) signed speed about the commanded axis

        # --- EMA smoothing (damps jitter; a +-0.5 square wave shrinks to |ema| ~ (1-a)/(1+a)*0.5) ---
        a = float(getattr(cfg, "ema_alpha", 0.9))
        self.omega_ema = a * self.omega_ema + (1.0 - a) * omega_signed
        ema = self.omega_ema

        # --- BAND: ZERO below floor, linear ramp to 1 at band_omega_max, signed to reward ONLY commanded dir ---
        floor = float(getattr(cfg, "band_omega_floor", 0.15))
        omax = float(getattr(cfg, "band_omega_max", 3.0))
        bscale = float(getattr(cfg, "band_scale", 3.0))
        # SCAFFOLD ANNEAL (2026-07-08 consensus): the band is the early rotation-GRADIENT scaffold
        # (pays any forward rotation -> cold-start gradient). Anneal it toward band_scale_end over
        # band_scaffold_anneal env-steps so the CONVERGED objective is net-progress + drop penalty
        # (object-agnostic). Keep end>0 (a floor), not true zero, or the dense gradient vanishes
        # exactly when net-progress goes sparse near the per-grasp limit.
        if getattr(cfg, "band_scale_end", None) is not None:
            _asf = min(float(getattr(self, "common_step_counter", 0)) /
                       max(float(getattr(cfg, "band_scaffold_anneal", 1.0)), 1.0), 1.0)
            bscale = bscale + (float(cfg.band_scale_end) - bscale) * _asf
            self.extras["band_scale_now"] = torch.tensor(bscale, device=ema.device)
        mag = ((ema.abs() - floor) / max(omax - floor, 1e-6)).clamp(0.0, 1.0)
        # BAND-PASS CEILING (2026-07-03 night, default OFF; guarded): the saturating ramp pays MAX
        # for arbitrarily violent spin — loop6c learned a fling machine (wind to 3.7 rad/s in-grasp,
        # displacement-terminate every ~50 steps, repeat). Tapering pay to zero above band_omega_ceil
        # makes the controllable regime (~1 rad/s, the demonstrated cylinder/burst speed) the optimum.
        ceil0 = getattr(cfg, "band_omega_ceil", None)
        if ceil0 is not None:
            c_end = float(ceil0)
            # IN-TRAINING ANNEAL (2026-07-04, two-strike lesson from 6e/6f: the basin tolerates no
            # price JUMP): ceiling walks band_omega_ceil_start -> band_omega_ceil over
            # band_omega_ceil_anneal env-steps, paying the current operating point first and
            # dragging it into the controllable band gradually.
            c_start = float(getattr(cfg, "band_omega_ceil_start", c_end))
            ann = float(getattr(cfg, "band_omega_ceil_anneal", 0.0))
            if ann > 0.0 and c_start != c_end:
                frac = min(float(getattr(self, "common_step_counter", 0)) / ann, 1.0)
                c0 = c_start + (c_end - c_start) * frac
            else:
                c0 = c_end
            self._band_c0_now = c0
            c1 = c0 + float(getattr(cfg, "band_omega_ceil_taper", 0.8))
            over = ((ema.abs() - c0) / max(c1 - c0, 1e-6)).clamp(0.0, 1.0)
            mag = mag * (1.0 - over)
            self.extras["band_ceil_now"] = torch.tensor(c0, device=ema.device)
        band = bscale * mag * torch.sign(ema)                         # +: commanded spin; 0: |ema|<=floor; -: reverse

        # --- adaptive up-weight from running mean of signed revs/episode (>=1, capped) ---
        gain = float(getattr(cfg, "adaptive_wrot_gain", 0.5))
        cap = float(getattr(cfg, "adaptive_wrot_cap", 3.0))
        adaptive_mult = (1.0 + gain * self._mean_signed_revs_per_ep.clamp_min(0.0)).clamp(1.0, cap)
        r_rot = band * adaptive_mult
        # HELD GATE (closed-loop 2026-07-03, default OFF; guarded — legacy behavior unchanged):
        # the escape-spin loophole — a falling/rolling ball accrues omega without being held, and
        # ungated band/sparse rewards pay it (observed: loop1 pose2 farmed zero-g spin). With
        # cfg.band_held_gate=True, the band AND the effective-angle integration (hence the sparse
        # bonus) count only while held: n_engaged >= band_held_min_fingers AND object within
        # band_held_disp of its episode seat.
        held_gate = torch.ones_like(ema)
        if bool(getattr(cfg, "band_held_gate", False)):
            n_eng_h = (self.last_contacts > float(getattr(cfg, "band_held_thresh", 0.15))).sum(-1)
            disp_h = torch.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
            held_gate = ((n_eng_h >= int(getattr(cfg, "band_held_min_fingers", 2)))
                         & (disp_h < float(getattr(cfg, "band_held_disp", 0.05)))).float()
            r_rot = r_rot * held_gate
            # ASYMMETRIC BACKSLIP COST (2026-07-06, default OFF; guarded): the held gate zeroes the
            # NEGATIVE band during unheld moments — exactly when large objects unwind (braking
            # curves: n=0-2 bins −0.55..−0.72) — so backslip was FREE. With band_backslip_w set,
            # backward rotation costs ALWAYS (held or not); forward still pays only while held.
            bw = getattr(cfg, "band_backslip_w", None)
            if bw is not None:
                r_rot = r_rot + float(bw) * torch.clamp(band, max=0.0) * (1.0 - held_gate)
            self.extras["band_held_frac"] = held_gate.mean()
        # N4 BOOST (2026-07-03 PM, default OFF; guarded): the pose3 braking curve measured rotation
        # is FASTEST and only strongly directional at n=4 contacts (+0.39 rad/s vs ~0 at n=2/3), but
        # the policy visits that mode ~3% of steps. Tiered DENSE income — band scaled band_lown_scale
        # below 4 engaged fingers, band_n4_boost at >=4 — pays rotation everywhere (no cold-start,
        # unlike a hard n>=4 gate) while making the n>=4 drive mode the profitable one. The SPARSE
        # ratchet path is intentionally left n-agnostic (held-gate only) as the base gradient.
        n4b = getattr(cfg, "band_n4_boost", None)
        if n4b is not None:
            n_eng_b = (self.last_contacts > float(getattr(cfg, "band_held_thresh", 0.15))).sum(-1)
            nmult = torch.where(n_eng_b >= 4,
                                torch.full_like(ema, float(n4b)),
                                torch.full_like(ema, float(getattr(cfg, "band_lown_scale", 0.3))))
            r_rot = r_rot * nmult
            self.extras["band_n4_frac"] = (n_eng_b >= 4).float().mean()
        # CONTINUOUS N-SCALING (2026-07-03 eve, default OFF; guarded): the N4-boost run parked at
        # n=3 — one finger under the tier — and farmed flickers (threshold camping). Continuous
        # multiplier n/div has no ledge: every additional engaged finger raises rotation pay
        # (div=3 -> n=2:0.67, 3:1.0, 4:1.33, 5:1.67) and n=0 pays 0 (escape-spin hardened).
        ndiv = getattr(cfg, "band_n_scale_div", None)
        if ndiv is not None:
            n_eng_c = (self.last_contacts > float(getattr(cfg, "band_held_thresh", 0.15))).sum(-1).float()
            nmult = (n_eng_c / float(ndiv)).clamp(0.0, 5.0 / float(ndiv))
            # SCAFFOLD ANNEAL (2026-07-08): n-scaling is OBJECT-SPECIFIC support shaping (more fingers
            # pay more -> wrong for a pen/book). Anneal toward FLAT (no n preference) so the endpoint
            # lets the policy discover each object's own support. Kept in reserve if it proves needed.
            if getattr(cfg, "band_scale_end", None) is not None:
                _asf2 = min(float(getattr(self, "common_step_counter", 0)) /
                            max(float(getattr(cfg, "band_scaffold_anneal", 1.0)), 1.0), 1.0)
                nmult = nmult * (1.0 - _asf2) + _asf2
            r_rot = r_rot * nmult
            self.extras["band_n3plus_frac"] = (n_eng_c >= 3).float().mean()
            self.extras["band_n4plus_frac"] = (n_eng_c >= 4).float().mean()
        # HOLD-PERSISTENCE INCOME (2026-07-03 eve, default OFF; guarded): pure n-scaled drive income
        # (lane 4) bought +2.2 rad/s bursts at held-frac 0.07 — rotation and holding are separately
        # purchasable; nothing yet pays their CONJUNCTION over time. Small per-step held income
        # (~25% of avg step income at 0.2) makes extending the held window itself profitable, so
        # burst → sustained is the gradient direction. Requires band_held_gate for a live held_gate.
        hinc = getattr(cfg, "band_hold_income", None)
        if hinc is not None:
            r_rot = r_rot + float(hinc) * held_gate
        # ROTATION-STREAK BONUS (2026-07-03 night, default OFF; guarded): pays for CONSECUTIVE steps
        # of (held AND supra-floor rotation) — the literal definition of sustained rotation. Ramps
        # linearly to full over band_rotstreak_cap steps. Static holds get ZERO (unlike hold income,
        # cannot re-create the camp); flings get little (streak resets when the hold breaks). This is
        # the retention lever for the 6d overshoot-coast gap.
        stk = getattr(cfg, "band_rotstreak_scale", None)
        if stk is not None:
            if not hasattr(self, "_rot_streak"):
                self._rot_streak = torch.zeros_like(ema)
            rotating_held = (held_gate > 0.0) & (ema.abs() > floor)
            self._rot_streak = torch.where(rotating_held, self._rot_streak + 1.0,
                                           torch.zeros_like(self._rot_streak))
            cap_s = float(getattr(cfg, "band_rotstreak_cap", 40.0))
            r_rot = r_rot + float(stk) * (self._rot_streak / cap_s).clamp(0.0, 1.0) * rotating_held.float()
            self.extras["band_rotstreak_mean"] = self._rot_streak.mean()

        # --- integrate SUPRA-FLOOR SMOOTHED signed angle; SPARSE bonus only on NET commanded progress ---
        # NOTE (deviation from the literal spec `cum += omega_ema*dt`, justified by the offline unit test):
        # integrating the RAW ema lets a constant sub-floor drift (the 0.037 rad/s jitter attractor) slowly
        # accumulate angle and LEAK sparse bonuses -- re-creating a weak version of the very attractor this
        # reward kills. Integrating only the supra-floor signed speed (the SAME |ema|-floor quantity the band
        # uses) makes both the band AND the sparse term exactly 0 below the floor. cum_angle is thus "net
        # EFFECTIVE (above-floor) angle"; revs = cum_angle/(2*pi) then measures genuine rotation, not drift.
        omega_eff = torch.sign(ema) * (ema.abs() - floor).clamp_min(0.0) * held_gate
        if getattr(cfg, "band_omega_ceil", None) is not None:
            # fling must not farm the sparse ratchet either: revolutions accrue at most at the
            # payable speed (clamp, not zero — net progress still counts, just capped). Tracks the
            # ANNEALED ceiling when active.
            _c0 = float(getattr(self, "_band_c0_now", cfg.band_omega_ceil))
            omega_eff = omega_eff.clamp(-_c0, _c0)
        self.cum_angle = self.cum_angle + omega_eff * self.step_dt
        theta = float(getattr(cfg, "sparse_theta", 0.3))
        bonus = float(getattr(cfg, "sparse_bonus", 5.0))
        new_level = torch.floor(self.cum_angle / theta)              # signed; oscillation stays ~0 -> level 0
        gained = (new_level - self._sparse_level).clamp_min(0.0)     # only NET +progress pays; retreat pays nothing
        sparse_reward = bonus * gained
        self._sparse_level = torch.maximum(self._sparse_level, new_level)  # ratchet: no re-award on return to a level

        # HIGH-WATER NET-PROGRESS INCOME (2026-07-07, default OFF; guarded). The diagnostic proved
        # the treadmill policy is closed-loop & cyclic but UNPRODUCTIVE (forward-while-held earns the
        # held-gated band, unheld slip-back is free -> net ~0). And penalizing backslip DESTROYS
        # recovery (recovery needs transient backward motion). Fix that is BOTH treadmill-proof AND
        # recovery-safe: accumulate PHYSICAL rotation (all steps, NOT held-gated so slip-back really
        # subtracts) while the object is in-play, and pay ONLY for advancing its per-episode HIGH-WATER
        # mark. Treadmill (5->3->5) re-covers old ground => pays 0. Genuine new progress pays. Recovery
        # (drop, re-climb, then exceed) isn't punished — it just doesn't earn until it beats the record.
        npw = getattr(cfg, "band_netprog_w", None)
        netprog_reward = torch.zeros_like(r_rot)
        if npw is not None:
            disp_np = torch.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
            in_play = (disp_np < float(getattr(cfg, "band_netprog_margin", 0.10))).float()
            self._phys_cum = self._phys_cum + omega_signed * self.step_dt * in_play
            netprog_reward = float(npw) * (self._phys_cum - self._phys_hw).clamp_min(0.0)
            self._phys_hw = torch.maximum(self._phys_hw, self._phys_cum)
            self.extras["netprog_mean"] = netprog_reward.mean()
            self.extras["phys_cum_mean"] = self._phys_cum.mean()

        # --- commit the adaptive running mean every adaptive_update_steps env-steps ---
        self._bandrot_step_count += 1
        upd = int(getattr(cfg, "adaptive_update_steps", 2000))
        if upd > 0 and (self._bandrot_step_count % upd == 0) and float(self._rev_accum_n) > 0.0:
            self._mean_signed_revs_per_ep = self._rev_accum_sum / self._rev_accum_n.clamp_min(1.0)
            self._rev_accum_sum = torch.zeros_like(self._rev_accum_sum)
            self._rev_accum_n = torch.zeros_like(self._rev_accum_n)

        # --- diagnostics ---
        self.extras["omega_signed_mean"] = omega_signed.mean()
        self.extras["omega_ema_mean"] = ema.mean()
        self.extras["band_mean"] = band.mean()
        self.extras["adaptive_mult"] = adaptive_mult
        self.extras["cum_angle_mean"] = self.cum_angle.mean()
        self.extras["r_rot_mean"] = r_rot.mean()
        self.extras["sparse_mean"] = sparse_reward.mean()
        self.extras["mean_signed_revs_per_ep"] = self._mean_signed_revs_per_ep
        # DROP PENALTY (2026-07-08 consensus, default OFF; guarded). Point-3 finding: dropping only
        # cost "lost future reward" (weak, far-off). An explicit terminal cost is the minimal
        # feasibility signal — propagates "don't get here" strongly & locally. Calibrate so productive
        # rotation still out-values the drop risk, or it re-creates the timid clamp.
        drop_reward = torch.zeros_like(r_rot)
        drp = getattr(cfg, "drop_penalty", None)
        if drp is not None:
            _dd = float(getattr(self, "_gx_drop_disp", getattr(cfg, "graspxl_drop_disp", 0.10)))
            _settling = (self._gx_settle > 0) if hasattr(self, "_gx_settle") \
                else torch.zeros_like(r_rot, dtype=torch.bool)
            _disp_d = torch.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
            dropping = (_disp_d > _dd) & (~_settling)
            drop_reward = -float(drp) * dropping.float()
            self.extras["drop_frac"] = dropping.float().mean()
        return total + r_rot + sparse_reward + netprog_reward + drop_reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "omega_ema"):
            return  # called during base __init__ before buffers exist
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        # finalize each completing env's SIGNED revolutions into the adaptive running-mean accumulator
        revs = self.cum_angle[ids] / _TWO_PI
        self._rev_accum_sum = self._rev_accum_sum + revs.sum()
        self._rev_accum_n = self._rev_accum_n + float(revs.numel())
        # zero per-env band state
        self.omega_ema[ids] = 0.0
        self.cum_angle[ids] = 0.0
        self._sparse_level[ids] = 0.0
        self._phys_cum[ids] = 0.0
        self._phys_hw[ids] = 0.0


# ============================================================================
# SPHERE-ENCLOSING variant: GraspXL ball 002aa185, reset from the enclosing cache (palm-up).
# Subclasses the Diverse cfg/env so we REUSE the validated enclosing-cache init
# (sharpa_wave_graspxl_adjustrotate_diverse.py:_write_enclosing_state) with diverse_enclosing_frac=1.0
# (all envs start from a settled enclosing grasp), and DISABLE the readiness-Phi shaping.
# ============================================================================
@configclass
class SharpaWaveGraspXLBandRotCfg(SharpaWaveGraspXLAdjustRotateDiverseCfg):
    """Anti-jitter BAND rotation reward on the GraspXL ball, initialized entirely from the ENCLOSING cache.

    Inherited from Diverse -> AdjustRotate -> Sustained (unchanged): torque control, obs_hand_gravity
    (observation_space=195), object_linvel_penalty_scale=-0.5 (MANDATORY anti-translate), torque -0.1,
    work -0.5, bounded centering (object_pos_reward_scale=1.0, exp(-disp/0.03)), pos_diff 0.0, friction
    combine='max' on the object, SE(3) displacement-drop termination, adaptive gravity curriculum.
    Train FROM SCRATCH (obs dim 195)."""
    # turn OFF the base clip rotate term; the BAND term replaces it
    rotate_reward_scale = 0.0
    # disable readiness-Phi shaping (pure band + inherited anti-drop/energy terms)
    use_readiness_potential = False
    # ALL reset envs start from a settled ENCLOSING grasp (palm-up, no SE(3) qrand), not the pinch
    diverse_enclosing_frac = 1.0
    # --- BAND reward ---
    band_scale = 3.0
    band_omega_floor = 0.15
    band_omega_max = 3.0
    ema_alpha = 0.9
    adaptive_wrot_gain = 0.5
    adaptive_wrot_cap = 3.0
    adaptive_update_steps = 2000
    sparse_bonus = 5.0
    sparse_theta = 0.3            # rad
    # --- curriculum gates ---
    graspxl_orient_rotate_gate = -1.0   # gate gravity curriculum on DROP RATE ONLY (base clip rotate is off)
    graspxl_curr_drop_thresh = 0.01     # LOOSE gate (not 5e-4): let gravity advance while the policy rotates


class SharpaWaveGraspXLBandRotEnv(_BandRotMixin, SharpaWaveGraspXLAdjustRotateDiverseEnv):
    """GraspXL enclosing-cache env + anti-jitter band reward (Phi disabled via use_readiness_potential=False)."""
    pass


# ============================================================================
# CYLINDER variant: same object / cache / init as the working cylinder rotation task
# (Isaac-Inhand-Rotate-Sharpa-Wave-v0 -> SharpaWaveEnvCfg / SharpaWaveInhandRotateEnv, cache
# sharpa_grasp_linspace_0.5-0.5-1). Only the reward changes (base clip rotate off -> band on).
# ============================================================================
@configclass
class SharpaWaveBandRotCylinderCfg(SharpaWaveEnvCfg):
    """Anti-jitter BAND rotation reward on the CYLINDER. Everything inherited from SharpaWaveEnvCfg is
    unchanged (cylinder object, grasp_cache_path='cache/sharpa_grasp_linspace', scale_range [0.5,0.5,1],
    observation_space=192, object_linvel_penalty_scale=-0.3 anti-translate, torque -0.1, work -0.5,
    object_pos_reward_scale=0.003 centering, adaptive gravity curriculum) EXCEPT the rotate reward, which
    switches from the base clip term to the band term. Train FROM SCRATCH."""
    # turn OFF the base clip rotate term; the BAND term replaces it
    rotate_reward_scale = 0.0
    # --- BAND reward ---
    band_scale = 3.0
    band_omega_floor = 0.15
    band_omega_max = 3.0
    ema_alpha = 0.9
    adaptive_wrot_gain = 0.5
    adaptive_wrot_cap = 3.0
    adaptive_update_steps = 2000
    sparse_bonus = 5.0
    sparse_theta = 0.3            # rad


class SharpaWaveBandRotCylinderEnv(_BandRotMixin, SharpaWaveInhandRotateEnv):
    """Cylinder rotation env + anti-jitter band reward (same object/cache/init as the baseline cylinder task)."""
    pass
