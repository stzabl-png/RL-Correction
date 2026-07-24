# REGRIP sweep R-G1/R-G2/R-G3: the ORIGINAL gait reward terms (bandgait G1 re-contact, G2 load-share,
# G3 contact-age), RE-BASED from the rotation task onto the pinch-reset REGRIP task. ADDITIVE ONLY.
#
# Design doc: robotics-rl-expert/notes/experiment_playbooks/regrip_contact_switch_designs_2026-07-01.md
# Sweep context: the ball gait sweep (band ROTATION reward + gait terms, sharpa_wave_graspxl_bandgait.py)
# failed on the smooth ball. This file isolates REGRIP: same three gait mechanisms, rotation removed.
#
# RE-BASING RULES (vs sharpa_wave_graspxl_bandgait.py):
#   * Backbone = AdjustHold-Perturb (pinch replay reset, v3a hold+coverage reward, n_engaged bonus,
#     gravity-linked adversarial perturbation, rotate_reward_scale=0). NO band/sparse rotation terms.
#   * G1's `* rotating` payoff gate is REMOVED (documented reason the event never fired on the ball:
#     results/gait_sweep_ball/INDEX.md). Here G1 pays on (held AND active) — the regrip objective.
#     G2/G3 never used `rotating`; they are ported unchanged apart from the gates below.
#   * All counters are FROZEN during the GraspXL replay settle (_gx_settle > 0) and every term is
#     gated on active = (_gx_settle == 0): the scripted pinch replay can neither charge off-streaks
#     nor earn reward.
#   * held gate uses gait_held_disp = 0.10 m (== the backbone hold_disp_gate / episode drop threshold),
#     NOT bandgait's 0.04 m: the regrip maneuver itself moves the ball 3-6 cm from the pinch seat
#     (pinch z~0.618 -> in-hand z~0.582), so a 0.04 m gate would zero the reward during the exact
#     transition this sweep is trying to elicit. Documented deviation from the faithful port.
#
# Self-collision arm: every cfg in the regrip sweep reads SHARPA_SELF_COLLISION ('1' default / '0')
# and overrides robot_cfg.spawn.articulation_props.enabled_self_collisions (baseline setting is True;
# all prior results were trained with it ON). The sweep trains each variant once per setting.
#
# Obs space UNCHANGED (195): contact_switch_probe.py / d0a_pinch_regrasp_probe.py /
# extract_adjusted_poses.py / orient_diag_eval.py run on these tasks unmodified. Eval per variant:
#   contact_switch_probe.py --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-G1-Sphere-v1 \
#     --load_path <ckpt> --num_envs 128 --eval_steps 600 --gravity_z -9.81 --contact_thresh 0.1 --headless
#   d0a_pinch_regrasp_probe.py --task ... --pinch_max 2 --enclose_min 3 --sustain_steps 10 \
#     --num_envs 200 --eval_steps 1200 --gravity_z -9.81 --orient_cap 0 --headless
from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjusthold_perturb import (
    SharpaWaveGraspXLAdjustHoldPerturbCfg,
    SharpaWaveGraspXLAdjustHoldPerturbEnv,
)


def apply_self_collision_override(cfg) -> None:
    """Regrip-sweep arm toggle: SHARPA_SELF_COLLISION env var -> articulation self-collision flag.

    Called from each sweep cfg's __post_init__ (cfg.robot_cfg is already a per-instance deepcopy at
    that point — same guarantee the FixedG cfg relies on for cfg.sim, sharpa_wave_graspxl_
    adjustrotate_diverse.py:202). Shared by all regrip-sweep variant files."""
    sc = os.environ.get("SHARPA_SELF_COLLISION", "1").strip().lower() not in ("0", "false", "off")
    cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = sc
    print(f"[regrip-sweep] enabled_self_collisions={sc}")


class _RegripSelfCollisionPostInit:
    """Mixin for sweep cfgs: chain any parent __post_init__ (none exists on the AdjustHold-Perturb
    chain today; guarded for future-proofing) then apply the self-collision arm override."""

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        apply_self_collision_override(self)


# ============================================================================
# REGRIP-gait mixin: ONE de-rotationed gait term (cfg.gait_mode) on the perturb backbone.
# Idiom identical to _BandGaitMixin (sharpa_wave_graspxl_bandgait.py:54): buffers AFTER
# super().__init__(); hasattr guards in _reset_idx/_get_rewards (DirectRLEnv calls _reset_idx
# during super().__init__(), before the mixin buffers exist).
# ============================================================================
class _RegripGaitMixin:
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, n_f = self.num_envs, self.last_contacts.shape[1]  # 5 fingers [thumb,index,middle,ring,pinky]
        dev = self.device
        # G1 (re-contact) per-finger streak/latch buffers — semantics of bandgait.py:71-73
        self._rg1_off_count = torch.zeros((N, n_f), device=dev)
        self._rg1_on_count = torch.zeros((N, n_f), device=dev)
        self._rg1_pending = torch.zeros((N, n_f), device=dev)
        # G3 (contact-age) per-finger age buffer — semantics of bandgait.py:75
        self._rg3_age = torch.zeros((N, n_f), device=dev)

    # ---- shared per-step quantities (bandgait._gait_common minus `rotating`) ----
    def _regrip_common(self):
        cfg = self.cfg
        disp = torch.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
        held = (disp < float(getattr(cfg, "gait_held_disp", 0.10))).float()
        eng = (self.last_contacts > float(getattr(cfg, "gait_contact_thresh", 0.1))).float()  # (N,5)
        n_eng = eng.sum(-1)
        if hasattr(self, "_gx_settle"):
            active = (self._gx_settle == 0).float()
        else:
            active = torch.ones(self.num_envs, device=self.device)
        return eng, n_eng, held, active

    # ---- R-G1: release -> re-contact EVENT, sustained; pays on held (NOT rotating) ----
    def _term_g1_recontact(self, eng, n_eng, held, active):
        cfg = self.cfg
        k_off = float(getattr(cfg, "gait_g1_k_off", 5))
        m_sus = float(getattr(cfg, "gait_g1_m_sustain", 5))
        w1 = float(getattr(cfg, "gait_g1_w", 1.0))
        cap = float(getattr(cfg, "gait_g1_cap", 1.0))
        act = active.unsqueeze(-1)  # (N,1); freeze counters during settle
        reengage = eng * (self._rg1_off_count >= k_off).float() * act
        self._rg1_pending = torch.maximum(self._rg1_pending, reengage)
        on_new = torch.where(eng > 0.5, self._rg1_on_count + 1.0, torch.zeros_like(self._rg1_on_count))
        off_new = torch.where(eng > 0.5, torch.zeros_like(self._rg1_off_count), self._rg1_off_count + 1.0)
        self._rg1_on_count = act * on_new + (1.0 - act) * self._rg1_on_count
        self._rg1_off_count = act * off_new + (1.0 - act) * self._rg1_off_count
        self._rg1_pending = self._rg1_pending * eng          # released pending finger loses its latch
        fire = self._rg1_pending * (self._rg1_on_count >= m_sus).float() * act
        self._rg1_pending = self._rg1_pending * (1.0 - fire)
        r = (w1 * fire.sum(-1)).clamp(0.0, cap) * held * active
        self.extras["regrip_g1_fire_mean"] = fire.sum(-1).mean()
        return r

    # ---- R-G2: >=3 fingers sharing load (ported unchanged; 2-finger cradle earns 0) ----
    def _term_g2_loadshare(self, eng, n_eng, held, active):
        cfg = self.cfg
        w2 = float(getattr(cfg, "gait_g2_w", 0.5))
        cap = float(getattr(cfg, "gait_g2_cap", 1.0))
        share = ((n_eng - 2.0) / 2.0).clamp(0.0, 1.0)   # 0 at <=2 fingers, 1 at >=4
        r = (w2 * share).clamp(0.0, cap) * held * active
        self.extras["regrip_g2_share_mean"] = share.mean()
        return r

    # ---- R-G3: small penalty for fingers loaded longer than T (periodic release) ----
    def _term_g3_contactage(self, eng, n_eng, held, active):
        cfg = self.cfg
        T = float(getattr(cfg, "gait_g3_T", 25.0))
        w3 = float(getattr(cfg, "gait_g3_w", 0.5))
        cap = float(getattr(cfg, "gait_g3_cap", 1.0))
        act = active.unsqueeze(-1)
        age_new = torch.where(eng > 0.5, self._rg3_age + 1.0, torch.zeros_like(self._rg3_age))
        self._rg3_age = act * age_new + (1.0 - act) * self._rg3_age
        over = ((self._rg3_age - T) / T).clamp(0.0, 1.0)
        pen = (w3 * over.sum(-1)).clamp(0.0, cap)
        r = -pen * held * active
        self.extras["regrip_g3_over_mean"] = over.sum(-1).mean()
        return r

    _REGRIP_TERMS = {
        "g1_recontact": _term_g1_recontact,
        "g2_loadshare": _term_g2_loadshare,
        "g3_contactage": _term_g3_contactage,
    }

    def _get_rewards(self) -> torch.Tensor:
        total = super()._get_rewards()      # perturb backbone: hold+coverage+centering+penalties+n_engaged
        if not hasattr(self, "_rg1_off_count"):
            return total                    # called during base __init__ before mixin buffers exist
        mode = str(getattr(self.cfg, "gait_mode", "g1_recontact"))
        term_fn = self._REGRIP_TERMS.get(mode, None)
        if term_fn is None:
            raise ValueError(f"unknown gait_mode={mode!r}; expected one of {sorted(self._REGRIP_TERMS)}")
        eng, n_eng, held, active = self._regrip_common()
        r = term_fn(self, eng, n_eng, held, active)
        self.extras["regrip_reward_mean"] = r.mean()
        self.extras["regrip_held_frac"] = held.mean()
        self.extras["regrip_active_frac"] = active.mean()
        return total + r

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_rg1_off_count"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._rg1_off_count[ids] = 0.0
        self._rg1_on_count[ids] = 0.0
        self._rg1_pending[ids] = 0.0
        self._rg3_age[ids] = 0.0


# ============================================================================
# Cfgs: perturb backbone + one de-rotationed gait term + self-collision arm hook.
# ============================================================================
@configclass
class SharpaWaveGraspXLRegripGaitSphereCfg(_RegripSelfCollisionPostInit, SharpaWaveGraspXLAdjustHoldPerturbCfg):
    """Regrip sweep shared cfg (ball). Perturb backbone unchanged; ONE gait term via gait_mode.
    Term magnitudes/caps are the bandgait cylinder-sweep values (comparability with the original G1-G3)."""
    gait_mode = "g1_recontact"
    gait_contact_thresh = 0.1     # per-finger force (N) counted as engaged (== contact_switch_probe default)
    gait_held_disp = 0.10         # == backbone hold_disp_gate; see header for why NOT bandgait's 0.04
    # G1 re-contact (release->reposition->re-contact, sustained)
    gait_g1_w = 1.0
    gait_g1_k_off = 5
    gait_g1_m_sustain = 5
    gait_g1_cap = 1.0
    # G2 load-share (>=3 fingers)
    gait_g2_w = 0.5
    gait_g2_cap = 1.0
    # G3 contact-age
    gait_g3_w = 0.5
    gait_g3_T = 25.0
    gait_g3_cap = 1.0


@configclass
class SharpaWaveGraspXLRegripG1SphereCfg(SharpaWaveGraspXLRegripGaitSphereCfg):
    """R-G1: reward genuine release -> reposition -> re-contact (held-gated, rotation-free)."""
    gait_mode = "g1_recontact"


@configclass
class SharpaWaveGraspXLRegripG2SphereCfg(SharpaWaveGraspXLRegripGaitSphereCfg):
    """R-G2: reward >=3 fingers sharing the load (held-gated)."""
    gait_mode = "g2_loadshare"


@configclass
class SharpaWaveGraspXLRegripG3SphereCfg(SharpaWaveGraspXLRegripGaitSphereCfg):
    """R-G3: small penalty for any finger loaded longer than T steps (periodic release)."""
    gait_mode = "g3_contactage"


class SharpaWaveGraspXLRegripGaitSphereEnv(_RegripGaitMixin, SharpaWaveGraspXLAdjustHoldPerturbEnv):
    """Perturb regrip backbone + one de-rotationed gait term (cfg.gait_mode)."""
    pass
