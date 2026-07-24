# ADDITIVE finger-GAITING reward sweep for the CYLINDER (no baseline cfg/env is edited).
#
# WHY (the "2-finger rolling" attractor):
#   The BandRot cylinder policy (sharpa_wave_graspxl_bandrot.py: SharpaWaveBandRotCylinderCfg/Env) learns to
#   NET-rotate the cylinder, but the contact-switch probe (scripts/contact_switch_probe.py) shows it does so
#   by a FIXED-CONTACT roll: two fingers (typically thumb+middle) stay planted the whole episode and drive the
#   object like a wheel; the other fingers stay off; joints drift toward their limits. That is rolling, not
#   GAITING (release -> reposition -> re-contact, with the SET of engaged fingers cycling).
#
# GOAL: induce true finger gaiting on the cylinder. Two additive pieces:
#   (A) a MULTI-KEYFRAME reset cache harvested from the BandRot policy across rotation phases
#       (scripts/harvest_gait_keyframes.py -> cache/cylinder_gait_keyframes_0.5-0.5-1.npy), so resets span the
#       whole rotation cycle instead of one canonical pinch. Loaded via the base grasp_cache_path mechanism.
#   (B) ONE shared gait-reward env exposing 5 reward-term VARIANTS (G1..G5), each on the SAME backbone
#       (band + sparse + inherited anti-drop/energy terms), selected by cfg.gait_mode. Exactly ONE gait term
#       is added per run so the 5 variants are directly comparable.
#
# Offline validation (pure numpy, no sim): scripts/gait_reward_unit_test.py mirrors the per-step math of ALL
# five terms and proves each scores a synthetic GAITING trajectory > a ROLLING trajectory, and that a FLUTTER
# trajectory (fingers toggling with no rotation, object dropped -> held=0) earns ~0 for every term.
#
# TEMPLATE: this reuses the _BandRotMixin cooperative-super() pattern verbatim
# (sharpa_wave_graspxl_bandrot.py:47-146): buffers created AFTER super().__init__(); _reset_idx and
# _get_rewards both use a hasattr guard because DirectRLEnv calls _reset_idx once DURING super().__init__()
# (before the mixin buffers exist).
#
# Contact / rotation fields reused from the base env (sharpa_wave_env.py):
#   last_contacts (N,5) :128  [thumb,index,middle,ring,pinky]  ->  engaged := last_contacts > gait_contact_thresh
#   rot_axis      (N,3) :123  ->  omega_signed = (object_angvel . rot_axis)  ;  rotating := |omega_signed|>thr
#   object_rot / object_rot_prev :62-64  ;  fingertip_pos (N,5,3, env-local) :418-420
#   hand_dof_pos :423 , hand_dof_lower/upper_limits :92-93 , actuated_dof_indices :79-82
#   object_pos (env-local) :428 , object_default_pose :65 (set in _reset_idx :383-389)
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul, axis_angle_from_quat

from .sharpa_wave_graspxl_bandrot import (
    _BandRotMixin,
    SharpaWaveBandRotCylinderCfg,
    SharpaWaveBandRotCylinderEnv,
    SharpaWaveGraspXLBandRotCfg,
    SharpaWaveGraspXLBandRotEnv,
)


# ============================================================================
# GAIT-reward MIXIN (cooperative super() on top of _BandRotMixin + cylinder env).
# ADDS exactly ONE gait term (selected by cfg.gait_mode) to the band+sparse+penalty reward.
# ============================================================================
class _BandGaitMixin:
    """Adds ONE finger-gaiting reward term on top of the inherited band+sparse+penalty reward.

    All 5 terms are:
      * GATED on 'held' (object within cfg.gait_held_disp of object_default_pose) so nothing pays after a drop,
      * CAPPED (per-step magnitude <= cfg.gait_*_cap, kept <= the band term ~0.5-1.0),
      * logged to self.extras (0-dim scalars only; PPO logs scalars).

    Buffers are created AFTER super().__init__() and both _reset_idx / _get_rewards use a hasattr guard, exactly
    like _BandRotMixin (DirectRLEnv calls _reset_idx during super().__init__() before these buffers exist)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N = self.num_envs
        n_f = self.last_contacts.shape[1]  # 5 fingers
        dev = self.device
        # --- G1 (re-contact) per-finger streak/latch buffers ---
        self._g1_off_count = torch.zeros((N, n_f), device=dev)   # consecutive DISENGAGED steps
        self._g1_on_count = torch.zeros((N, n_f), device=dev)    # consecutive ENGAGED steps
        self._g1_pending = torch.zeros((N, n_f), device=dev)     # latch: qualified re-contact awaiting sustain
        # --- G3 (contact-age) per-finger age buffer ---
        self._g3_age = torch.zeros((N, n_f), device=dev)         # consecutive ENGAGED steps (load age)
        # --- G5 (recovery) buffers ---
        self._g5_neng_ema = torch.zeros(N, device=dev)           # short EMA of n_engaged (release detector)
        self._g5_window = torch.zeros(N, device=dev)             # countdown: steps left in a post-release window

    # ---- shared per-step quantities ----------------------------------------
    def _gait_common(self):
        cfg = self.cfg
        # signed speed about the commanded axis (same computation as the base rotate term / band term)
        object_angvel = axis_angle_from_quat(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))
        ) / self.step_dt                                                     # (N,3)
        omega_signed = (object_angvel * self.rot_axis).sum(-1)               # (N,)
        rotating = (omega_signed.abs() > float(getattr(cfg, "gait_rot_thresh", 0.15))).float()
        disp = torch.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
        held = (disp < float(getattr(cfg, "gait_held_disp", 0.04))).float()
        eng = (self.last_contacts > float(getattr(cfg, "gait_contact_thresh", 0.1))).float()  # (N,5)
        n_eng = eng.sum(-1)                                                  # (N,)
        return eng, n_eng, rotating, held

    # ---- G1: reward true release -> re-contact (sustained) -----------------
    def _term_g1_recontact(self, eng, n_eng, rotating, held):
        cfg = self.cfg
        k_off = float(getattr(cfg, "gait_g1_k_off", 5))
        m_sus = float(getattr(cfg, "gait_g1_m_sustain", 5))
        w1 = float(getattr(cfg, "gait_g1_w", 1.0))
        cap = float(getattr(cfg, "gait_g1_cap", 1.0))
        # a finger 'qualifies' when it RE-engages after being off for >= k_off steps
        reengage = eng * (self._g1_off_count >= k_off).float()
        self._g1_pending = torch.maximum(self._g1_pending, reengage)
        # update streak counters
        self._g1_on_count = torch.where(eng > 0.5, self._g1_on_count + 1.0, torch.zeros_like(self._g1_on_count))
        self._g1_off_count = torch.where(eng > 0.5, torch.zeros_like(self._g1_off_count), self._g1_off_count + 1.0)
        # a pending finger that goes disengaged loses its latch
        self._g1_pending = self._g1_pending * eng
        # award when a qualified re-contact has now been SUSTAINED m steps; then consume the latch
        fire = self._g1_pending * (self._g1_on_count >= m_sus).float()
        self._g1_pending = self._g1_pending * (1.0 - fire)
        r = (w1 * fire.sum(-1)).clamp(0.0, cap) * held * rotating
        self.extras["gait_g1_fire_mean"] = fire.sum(-1).mean()
        return r

    # ---- G2: reward >=3 fingers sharing load (2-finger roll earns 0) --------
    def _term_g2_loadshare(self, eng, n_eng, rotating, held):
        cfg = self.cfg
        w2 = float(getattr(cfg, "gait_g2_w", 0.5))
        cap = float(getattr(cfg, "gait_g2_cap", 1.0))
        share = ((n_eng - 2.0) / 2.0).clamp(0.0, 1.0)   # 0 at <=2 fingers, 1 at >=4
        r = (w2 * share).clamp(0.0, cap) * held
        self.extras["gait_g2_share_mean"] = share.mean()
        return r

    # ---- G3: penalize fingers loaded longer than T (encourage periodic release) ----
    def _term_g3_contactage(self, eng, n_eng, rotating, held):
        cfg = self.cfg
        T = float(getattr(cfg, "gait_g3_T", 25.0))
        w3 = float(getattr(cfg, "gait_g3_w", 0.5))
        cap = float(getattr(cfg, "gait_g3_cap", 1.0))
        self._g3_age = torch.where(eng > 0.5, self._g3_age + 1.0, torch.zeros_like(self._g3_age))
        over = ((self._g3_age - T) / T).clamp(0.0, 1.0)  # 0 until age>T, ramps to 1 at age=2T
        pen = (w3 * over.sum(-1)).clamp(0.0, cap)         # magnitude of the (negative) penalty
        r = -pen * held                                   # small: encourages release without forcing a drop
        self.extras["gait_g3_over_mean"] = over.sum(-1).mean()
        return r

    # ---- G4: reward rotating while joints stay MID-range (forces reposition) ----
    def _term_g4_jointheadroom(self, eng, n_eng, rotating, held):
        cfg = self.cfg
        w4 = float(getattr(cfg, "gait_g4_w", 0.5))
        cap = float(getattr(cfg, "gait_g4_cap", 1.0))
        idx = self.actuated_dof_indices
        q = self.hand_dof_pos[:, idx]
        qlo = self.hand_dof_lower_limits[:, idx]
        qhi = self.hand_dof_upper_limits[:, idx]
        qmid = 0.5 * (qlo + qhi)
        rng = (qhi - qlo).clamp_min(1e-6)
        norm = 2.0 * (q - qmid) / rng                     # ~[-1,1] within limits (0 at mid, +-1 at a limit)
        headroom = (1.0 - norm * norm).clamp(0.0, 1.0)    # 1 at mid-range, 0 at a limit
        r = (w4 * headroom.mean(-1)).clamp(0.0, cap) * rotating * held
        self.extras["gait_g4_headroom_mean"] = headroom.mean()
        return r

    # ---- G5: reward re-establishing a grasp within a window AFTER a release ----
    def _term_g5_recovery(self, eng, n_eng, rotating, held):
        cfg = self.cfg
        w5 = float(getattr(cfg, "gait_g5_w", 1.0))
        cap = float(getattr(cfg, "gait_g5_cap", 1.0))
        beta = float(getattr(cfg, "gait_g5_beta", 0.8))
        window = float(getattr(cfg, "gait_g5_window", 15))
        drop = float(getattr(cfg, "gait_g5_drop", 1.0))
        # release event: n_engaged drops by >= `drop` vs the short EMA (a finger just let go)
        release = ((self._g5_neng_ema - n_eng) >= drop).float()
        self._g5_window = torch.where(release > 0.5, torch.full_like(self._g5_window, window),
                                      (self._g5_window - 1.0).clamp_min(0.0))
        self._g5_neng_ema = beta * self._g5_neng_ema + (1.0 - beta) * n_eng
        # force-closure-ish grasp quality: fraction of fingers engaged * opposition of their contact dirs
        d = self.fingertip_pos - self.object_pos.unsqueeze(1)               # (N,5,3) dir object->fingertip
        dn = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-6)
        wsum = eng / (n_eng.unsqueeze(-1) + 1e-6)                            # weights over engaged fingers
        mean_dir = (dn * wsum.unsqueeze(-1)).sum(1)                          # (N,3)
        opposition = (1.0 - torch.norm(mean_dir, dim=-1)).clamp(0.0, 1.0)    # ~1 if engaged dirs cancel
        grasp_q = (n_eng / 5.0) * opposition
        in_window = (self._g5_window > 0.0).float()
        r = (w5 * grasp_q * in_window).clamp(0.0, cap) * held
        self.extras["gait_g5_graspq_mean"] = grasp_q.mean()
        self.extras["gait_g5_inwindow_mean"] = in_window.mean()
        return r

    _GAIT_TERMS = {
        "g1_recontact": _term_g1_recontact,
        "g2_loadshare": _term_g2_loadshare,
        "g3_contactage": _term_g3_contactage,
        "g4_jointheadroom": _term_g4_jointheadroom,
        "g5_recovery": _term_g5_recovery,
    }

    def _get_rewards(self) -> torch.Tensor:
        total = super()._get_rewards()                    # base penalties + band + sparse (via _BandRotMixin)
        if not hasattr(self, "_g1_off_count"):
            return total                                  # called before buffers exist (during base __init__)
        mode = str(getattr(self.cfg, "gait_mode", "g1_recontact"))
        term_fn = self._GAIT_TERMS.get(mode, None)
        if term_fn is None:
            raise ValueError(f"unknown gait_mode={mode!r}; expected one of {sorted(self._GAIT_TERMS)}")
        eng, n_eng, rotating, held = self._gait_common()
        r_gait = term_fn(self, eng, n_eng, rotating, held)
        # diagnostics
        self.extras["gait_reward_mean"] = r_gait.mean()
        self.extras["gait_n_engaged_mean"] = n_eng.mean()
        self.extras["gait_held_frac"] = held.mean()
        self.extras["gait_rotating_frac"] = rotating.mean()
        return total + r_gait

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)                       # -> _BandRotMixin._reset_idx -> base reset
        if not hasattr(self, "_g1_off_count"):
            return                                        # called during base __init__ before buffers exist
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._g1_off_count[ids] = 0.0
        self._g1_on_count[ids] = 0.0
        self._g1_pending[ids] = 0.0
        self._g3_age[ids] = 0.0
        self._g5_neng_ema[ids] = 0.0
        self._g5_window[ids] = 0.0


# ============================================================================
# CYLINDER gait cfg: BandRot cylinder backbone + multi-keyframe reset cache + gait-term params.
# ============================================================================
@configclass
class SharpaWaveBandGaitCylinderCfg(SharpaWaveBandRotCylinderCfg):
    """Cylinder finger-gaiting sweep. Inherits the WHOLE BandRot cylinder backbone unchanged
    (rotate_reward_scale=0 -> band on; sparse bonus; object_linvel/torque/work/centering penalties;
    adaptive gravity curriculum) and ONLY:
      (1) swaps the reset cache to the harvested MULTI-KEYFRAME cache (grasp_cache_path below; the base
          loader appends '_{scale0}-{scale1}-{scaleN}.npy' -> cache/cylinder_gait_keyframes_0.5-0.5-1.npy,
          MATCHING the cylinder cache format exactly, sharpa_wave_env.py:117), and
      (2) adds ONE finger-gaiting reward term selected by gait_mode (default g1; the 5 registered tasks each
          set gait_mode via a subclass below).
    Train FROM SCRATCH."""
    # (1) multi-keyframe reset cache (base loader appends the '_a-b-c.npy' scale suffix; NO '.npy' here)
    grasp_cache_path = "cache/cylinder_gait_keyframes"

    # (2) gait reward selector + shared gates
    gait_mode = "g1_recontact"
    gait_contact_thresh = 0.1     # per-finger force (N) counted as engaged (== contact_switch_probe default)
    gait_rot_thresh = 0.15        # |omega_signed| above which 'rotating' is True (== band_omega_floor)
    gait_held_disp = 0.04         # object within this displacement of default pose == still 'held' (not dropped)

    # G1 re-contact (release->reposition->re-contact, sustained): per-step magnitude capped at gait_g1_cap
    gait_g1_w = 1.0
    gait_g1_k_off = 5             # a finger must be OFF this many steps to qualify a re-contact
    gait_g1_m_sustain = 5         # ...and STAY on this many steps after re-contact to be paid
    gait_g1_cap = 1.0
    # G2 load-share (>=3 fingers): 2-finger roll earns 0
    gait_g2_w = 0.5
    gait_g2_cap = 1.0
    # G3 contact-age (small penalty for fingers loaded longer than T)
    gait_g3_w = 0.5
    gait_g3_T = 25.0
    gait_g3_cap = 1.0
    # G4 joint-headroom (rotate while joints stay mid-range)
    gait_g4_w = 0.5
    gait_g4_cap = 1.0
    # G5 recovery (re-grip within a window after a release)
    gait_g5_w = 1.0
    gait_g5_window = 15
    gait_g5_beta = 0.8
    gait_g5_drop = 1.0
    gait_g5_cap = 1.0


@configclass
class SharpaWaveBandGaitCylinderG1Cfg(SharpaWaveBandGaitCylinderCfg):
    """G1: reward genuine release -> reposition -> re-contact (per-finger, sustained)."""
    gait_mode = "g1_recontact"


@configclass
class SharpaWaveBandGaitCylinderG2Cfg(SharpaWaveBandGaitCylinderCfg):
    """G2: reward >=3 fingers sharing the load (a 2-finger roll earns 0)."""
    gait_mode = "g2_loadshare"


@configclass
class SharpaWaveBandGaitCylinderG3Cfg(SharpaWaveBandGaitCylinderCfg):
    """G3: small penalty for any finger loaded longer than T steps (periodic release)."""
    gait_mode = "g3_contactage"


@configclass
class SharpaWaveBandGaitCylinderG4Cfg(SharpaWaveBandGaitCylinderCfg):
    """G4: reward rotating WHILE joints stay mid-range (forces reposition, not driving to limits)."""
    gait_mode = "g4_jointheadroom"


@configclass
class SharpaWaveBandGaitCylinderG5Cfg(SharpaWaveBandGaitCylinderCfg):
    """G5: reward re-establishing a good grasp within a window AFTER a release event."""
    gait_mode = "g5_recovery"


# ============================================================================
# ONE shared env; the gait term is chosen at runtime by cfg.gait_mode.
# MRO: _BandGaitMixin -> _BandRotMixin -> SharpaWaveInhandRotateEnv -> DirectRLEnv.
#   _get_rewards : super() -> band+sparse+penalties, then + one gait term.
#   __init__/_reset_idx : cooperative super() with buffers-after-super() + hasattr guard.
# ============================================================================
class SharpaWaveBandGaitCylinderEnv(_BandGaitMixin, SharpaWaveBandRotCylinderEnv):
    """Cylinder band-rotation env + one finger-gaiting reward term (selected by cfg.gait_mode)."""
    pass


# ============================================================================
# SPHERE (GraspXL ball) gait variants — apply the SAME gait terms to the ball.
# Backbone = SharpaWaveGraspXLBandRotCfg/Env (band reward + enclosing-cache reset, diverse_enclosing_frac=1.0).
# NOTE: unlike the cylinder, the ball has NO rotating policy to harvest gait-phase keyframes from
# (chicken-and-egg), so it resets from the validated ENCLOSING cache (2807 diverse good grasps) rather than a
# multi-keyframe gait cache — the gait REWARD supplies the incentive. This is the honest best-available setup;
# a negative result carries the caveat that the keyframe scaffold (which helped the cylinder) is absent here.
# ============================================================================
@configclass
class SharpaWaveGraspXLBandGaitSphereCfg(SharpaWaveGraspXLBandRotCfg):
    """Ball finger-gaiting sweep. Inherits the BandRot SPHERE backbone unchanged (band reward on;
    use_readiness_potential=False; enclosing-cache reset with diverse_enclosing_frac=1.0; inherited
    anti-drop/energy terms; adaptive gravity curriculum) and adds ONE gait term selected by gait_mode.
    Train FROM SCRATCH (obs dim 195)."""
    # gait reward selector + shared gates (same defaults as the cylinder gait cfg)
    gait_mode = "g1_recontact"
    gait_contact_thresh = 0.1
    gait_rot_thresh = 0.15
    gait_held_disp = 0.04
    # G1 re-contact
    gait_g1_w = 1.0
    gait_g1_k_off = 5
    gait_g1_m_sustain = 5
    gait_g1_cap = 1.0
    # G2 load-share
    gait_g2_w = 0.5
    gait_g2_cap = 1.0
    # G3 contact-age
    gait_g3_w = 0.5
    gait_g3_T = 25.0
    gait_g3_cap = 1.0
    # G4/G5 params (unused in the G1-G3 ball sweep; kept for completeness)
    gait_g4_w = 0.5
    gait_g4_cap = 1.0
    gait_g5_w = 1.0
    gait_g5_window = 15
    gait_g5_beta = 0.8
    gait_g5_drop = 1.0
    gait_g5_cap = 1.0


@configclass
class SharpaWaveGraspXLBandGaitSphereG1Cfg(SharpaWaveGraspXLBandGaitSphereCfg):
    """G1 on the ball: reward genuine release -> reposition -> re-contact."""
    gait_mode = "g1_recontact"


@configclass
class SharpaWaveGraspXLBandGaitSphereG2Cfg(SharpaWaveGraspXLBandGaitSphereCfg):
    """G2 on the ball: reward >=3 fingers sharing the load."""
    gait_mode = "g2_loadshare"


@configclass
class SharpaWaveGraspXLBandGaitSphereG3Cfg(SharpaWaveGraspXLBandGaitSphereCfg):
    """G3 on the ball: small penalty for fingers loaded longer than T (periodic release)."""
    gait_mode = "g3_contactage"


class SharpaWaveGraspXLBandGaitSphereEnv(_BandGaitMixin, SharpaWaveGraspXLBandRotEnv):
    """Ball band-rotation env (enclosing-cache reset) + one finger-gaiting reward term (cfg.gait_mode)."""
    pass
