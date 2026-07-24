# REGRIP sweep R-D5 "RGTB": release-token-gated recruitment ratchet. ADDITIVE ONLY.
#
# Design doc: robotics-rl-expert/notes/experiment_playbooks/regrip_contact_switch_designs_2026-07-01.md (D5).
# Skeptic-mandated construction (from the design review):
#   * G1-style per-finger release->re-contact hysteresis latch (k_off=5 off, m_sustain=5 on), but with
#     DUAL-THRESHOLD debounced engagement (on > 0.15 N, off < 0.05 N, hold state in between) so 0.1 N
#     threshold flicker can neither charge off-streaks nor sustain on-streaks.
#   * A recruitment EVENT pays w_event ONLY if ALL of:
#       (a) RELEASE TOKEN — at least one finger of the ORIGINAL pinch set (debounced engaged set at the
#           first active step) has genuinely released (>= k_off debounced off steps) earlier this episode;
#       (b) RATCHET — debounced n_engaged exceeds the episode's best-so-far, which starts at the FIXED
#           baseline rgtb_baseline_n = 2 (hardcoded from the known pinch reset; NEVER a policy-set
#           snapshot — blocks baseline sandbagging);
#       (c) held AND active (settle over), and at most rgtb_event_cap_per_ep payments per episode.
#   * One-time milestone w_milestone when debounced n_engaged >= 3 sustained rgtb_sus3_need active steps.
#   * FORENSIC BIT logged per fired finger: had it been engaged EARLIER this episode (true release->
#     re-place) or is this its first touch (accretion)? extras rt_replace_frac reports the ratio —
#     an accretion-dominated positive is reported as such, never re-interpreted as regrip.
#
# Backbone: AdjustHold-Perturb (pinch replay reset, hold rewards, perturbation) — rotation-free.
# Counters are FROZEN during settle; the debounced contact STATE itself is tracked through settle so the
# pinch set is measured physically. Obs space unchanged (195). Self-collision arm via SHARPA_SELF_COLLISION.
#
# Eval:
#   d0a_pinch_regrasp_probe.py --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-D5-RGTB-Sphere-v1 \
#     --pinch_max 2 --enclose_min 3 --sustain_steps 10 --num_envs 200 --eval_steps 1200 \
#     --gravity_z -9.81 --orient_cap 0 --headless
#   contact_switch_probe.py --task ... --num_envs 128 --eval_steps 600 --gravity_z -9.81 --headless
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjusthold_perturb import (
    SharpaWaveGraspXLAdjustHoldPerturbCfg,
    SharpaWaveGraspXLAdjustHoldPerturbEnv,
)
from .sharpa_wave_graspxl_regrip_gait import _RegripSelfCollisionPostInit


@configclass
class SharpaWaveGraspXLRegripD5RGTBSphereCfg(_RegripSelfCollisionPostInit, SharpaWaveGraspXLAdjustHoldPerturbCfg):
    """R-D5 RGTB cfg: perturb backbone + release-token recruitment ratchet parameters."""
    # debounced engagement (dual threshold + min durations)
    rgtb_eng_on: float = 0.15         # N; force above this switches a finger's debounced state ON
    rgtb_eng_off: float = 0.05        # N; force below this switches it OFF (in between: hold state)
    rgtb_k_off: float = 5             # debounced OFF steps for a release to count / qualify a re-contact
    rgtb_m_sustain: float = 5         # debounced ON steps a qualified re-contact must hold before paying
    # payoff
    rgtb_w_event: float = 1.0         # per recruitment event (per-step magnitude <= 1.0 by construction)
    rgtb_event_cap_per_ep: float = 3  # max paid events per episode
    rgtb_baseline_n: float = 2.0      # FIXED ratchet baseline (known pinch reset); never policy-set
    rgtb_w_milestone: float = 5.0     # one-time sustained n>=3 milestone
    rgtb_sus3_need: float = 10        # active steps of debounced n>=3 required for the milestone


class SharpaWaveGraspXLRegripD5RGTBSphereEnv(SharpaWaveGraspXLAdjustHoldPerturbEnv):
    """Perturb regrip backbone + release-token recruitment ratchet (see module header)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, n_f = self.num_envs, self.last_contacts.shape[1]
        dev = self.device
        self._rt_eng = torch.zeros((N, n_f), device=dev)          # debounced engagement state {0,1}
        self._rt_off = torch.zeros((N, n_f), device=dev)          # consecutive debounced-OFF active steps
        self._rt_on = torch.zeros((N, n_f), device=dev)           # consecutive debounced-ON active steps
        self._rt_pending = torch.zeros((N, n_f), device=dev)      # qualified re-contact awaiting sustain
        self._rt_pending_prior = torch.zeros((N, n_f), device=dev)  # was the finger engaged before it qualified?
        self._rt_ever_eng = torch.zeros((N, n_f), device=dev)     # engaged at any point this episode
        self._rt_started = torch.zeros(N, dtype=torch.bool, device=dev)  # first-active-step latch done
        self._rt_pinchset = torch.zeros((N, n_f), device=dev)     # debounced engaged set at first active step
        self._rt_token = torch.zeros(N, device=dev)               # release-token latch {0,1}
        self._rt_level = torch.full((N,), float(cfg.rgtb_baseline_n), device=dev)  # ratchet best-so-far
        self._rt_paid = torch.zeros(N, device=dev)                # events paid this episode
        self._rt_sus3 = torch.zeros(N, device=dev)                # consecutive active steps with n_deb>=3
        self._rt_milestone = torch.zeros(N, device=dev)           # milestone paid {0,1}
        self._rt_ev_replace = torch.zeros(N, device=dev)          # forensic: paid re-place events
        self._rt_ev_accrete = torch.zeros(N, device=dev)          # forensic: paid first-touch events

    def _get_rewards(self) -> torch.Tensor:
        total = super()._get_rewards()
        if not hasattr(self, "_rt_eng"):
            return total
        cfg = self.cfg

        # ---- debounced engagement STATE (tracked through settle: the pinch set is physical) ----
        f = self.last_contacts                                     # (N,5)
        on_thr = float(cfg.rgtb_eng_on)
        off_thr = float(cfg.rgtb_eng_off)
        self._rt_eng = torch.where(f > on_thr, torch.ones_like(self._rt_eng),
                                   torch.where(f < off_thr, torch.zeros_like(self._rt_eng), self._rt_eng))
        eng = self._rt_eng
        n_deb = eng.sum(-1)                                        # (N,)

        if hasattr(self, "_gx_settle"):
            active = (self._gx_settle == 0).float()
        else:
            active = torch.ones(self.num_envs, device=self.device)
        act = active.unsqueeze(-1)

        # ---- first active step: snapshot the pinch set ----
        newly = (active > 0.5) & (~self._rt_started)
        if bool(newly.any()):
            self._rt_pinchset[newly] = eng[newly]
            self._rt_started = self._rt_started | (active > 0.5)

        # ---- G1-style latch on the debounced state (counters frozen during settle) ----
        k_off = float(cfg.rgtb_k_off)
        m_sus = float(cfg.rgtb_m_sustain)
        qualify = eng * (self._rt_off >= k_off).float() * act      # re-engaged after a genuine off period
        new_pend = (qualify > 0.5) & (self._rt_pending < 0.5)
        # Forensic snapshot at qualification, BEFORE ever_eng absorbs the current step: a finger counts
        # as re-place iff it was in the pinch set OR engaged on a PREVIOUS streak this episode; a pure
        # first-touch finger fails both -> accretion.
        prior = torch.maximum(self._rt_ever_eng, self._rt_pinchset).clamp(0.0, 1.0)
        self._rt_pending_prior = torch.where(new_pend, prior, self._rt_pending_prior)
        # only now absorb the current step's engagement into the episode history
        self._rt_ever_eng = torch.maximum(self._rt_ever_eng, eng * act)
        self._rt_pending = torch.maximum(self._rt_pending, qualify)
        on_new = torch.where(eng > 0.5, self._rt_on + 1.0, torch.zeros_like(self._rt_on))
        off_new = torch.where(eng > 0.5, torch.zeros_like(self._rt_off), self._rt_off + 1.0)
        self._rt_on = act * on_new + (1.0 - act) * self._rt_on
        self._rt_off = act * off_new + (1.0 - act) * self._rt_off
        self._rt_pending = self._rt_pending * eng
        fire = self._rt_pending * (self._rt_on >= m_sus).float() * act   # (N,5)
        self._rt_pending = self._rt_pending * (1.0 - fire)

        # ---- release token: an ORIGINAL pinch finger has genuinely released ----
        released_pinch = (self._rt_pinchset * (self._rt_off >= k_off).float()).amax(dim=-1)
        self._rt_token = torch.maximum(self._rt_token, released_pinch * active)

        # ---- held gate (backbone convention) ----
        disp = torch.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
        held = (disp < float(getattr(cfg, "hold_disp_gate", 0.10))).float()

        # ---- recruitment event payoff: token AND ratchet AND cap AND held AND active ----
        fired_any = (fire.amax(dim=-1) > 0.5)
        ratchet_up = n_deb > self._rt_level
        can_pay = (self._rt_token > 0.5) & ratchet_up & (self._rt_paid < float(cfg.rgtb_event_cap_per_ep)) \
                  & (held > 0.5) & (active > 0.5) & fired_any
        r_event = can_pay.float() * float(cfg.rgtb_w_event)
        if bool(can_pay.any()):
            self._rt_level = torch.where(can_pay, n_deb, self._rt_level)
            self._rt_paid = self._rt_paid + can_pay.float()
            # forensic classification of the paid event: any firing finger with prior engagement?
            fired_prior = (fire * self._rt_pending_prior).amax(dim=-1)
            self._rt_ev_replace = self._rt_ev_replace + (can_pay.float() * (fired_prior > 0.5).float())
            self._rt_ev_accrete = self._rt_ev_accrete + (can_pay.float() * (fired_prior <= 0.5).float())

        # ---- one-time sustained n>=3 milestone ----
        sus_new = torch.where((n_deb >= 3.0) & (active > 0.5) & (held > 0.5),
                              self._rt_sus3 + 1.0, torch.zeros_like(self._rt_sus3))
        self._rt_sus3 = torch.where(active > 0.5, sus_new, self._rt_sus3)
        hit = (self._rt_sus3 >= float(cfg.rgtb_sus3_need)).float() * (1.0 - self._rt_milestone)
        r_milestone = hit * float(cfg.rgtb_w_milestone)
        self._rt_milestone = torch.maximum(self._rt_milestone, hit)

        # ---- diagnostics ----
        tot_ev = self._rt_ev_replace + self._rt_ev_accrete
        self.extras["rt_fire_mean"] = fire.sum(-1).mean()
        self.extras["rt_token_frac"] = self._rt_token.mean()
        self.extras["rt_level_mean"] = self._rt_level.mean()
        self.extras["rt_paid_mean"] = self._rt_paid.mean()
        self.extras["rt_milestone_frac"] = self._rt_milestone.mean()
        self.extras["rt_replace_frac"] = (self._rt_ev_replace.sum() / tot_ev.sum().clamp_min(1.0))
        self.extras["rt_n_deb_mean"] = n_deb.mean()

        return total + r_event + r_milestone

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_rt_eng"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._rt_eng[ids] = 0.0
        self._rt_off[ids] = 0.0
        self._rt_on[ids] = 0.0
        self._rt_pending[ids] = 0.0
        self._rt_pending_prior[ids] = 0.0
        self._rt_ever_eng[ids] = 0.0
        self._rt_started[ids] = False
        self._rt_pinchset[ids] = 0.0
        self._rt_token[ids] = 0.0
        self._rt_level[ids] = float(self.cfg.rgtb_baseline_n)
        self._rt_paid[ids] = 0.0
        self._rt_sus3[ids] = 0.0
        self._rt_milestone[ids] = 0.0
        self._rt_ev_replace[ids] = 0.0
        self._rt_ev_accrete[ids] = 0.0
