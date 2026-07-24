# REGRIP v2 — five load-bearing-centric designs x two objects (ball, cylinder). ADDITIVE ONLY.
# Design doc: robotics-rl-expert/notes/reward_design/regrip_v2_load_bearing_designs_2026-07-02.md
# Chassis (sink termination, gravity curriculum, wrench ledger, handover FSM): sharpa_wave_regripv2_common.py
#
# Structure: each design is an object-agnostic REWARD MIXIN reading the chassis' _v2_common(); the
# mechanism mixins (_ShedMixin forced dropout, _RotLoadMixin rotating lateral load) are also
# object-agnostic. Ball envs mount them on the AdjustHold-Perturb backbone (pinch replay reset);
# cylinder envs mount them on _RegripV2CylBaseEnv (plain cylinder task, rotation zeroed, grasp-cache
# reset, same gravity-linked perturbation block). Count-based reward terms are zeroed everywhere.
#
# Designs (weights per the finalized note):
#   R1 LEDGER  : 1.0*(5*min cumulative load share)*clear*balanced + 1.0 ledger-gated re-contact event
#                (quota 5) + 0.5 transfer bonus (quota 5)
#   R2 CHAIN   : 0.3*clear*balanced + 1.0 per completed handover (cap 6) + 5.0 once when >=4 distinct
#                fingers completed one; decaying forced-shed seed (frequent at low g, off at full g)
#   R3 MARGIN  : 1.5*leave-one-out wrench margin*clear (n_eng>=2) + rotating load (half strength)
#                + 3.0*(5*min share) on TIMEOUT only
#   R4 SHED    : survival-first: 0.3*clear*balanced only + forced dropout (uniform among engaged,
#                skeptic-preferred) + 4.0 once when all 5 fingers survived >=1 shed by 30 steps
#   R5 ENTROPY : 2.0*new-highwater support entropy (clear-gated ratchet) + 0.3*H*clear*balanced
#                + 5.0 once at H>=0.9 + 0.5 transfer bonus (quota 5)
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from .sharpa_wave_env import SharpaWaveInhandRotateEnv
from .sharpa_wave_env_cfg import SharpaWaveEnvCfg
from .sharpa_wave_graspxl_adjusthold_perturb import (
    SharpaWaveGraspXLAdjustHoldPerturbCfg,
    SharpaWaveGraspXLAdjustHoldPerturbEnv,
)
from .sharpa_wave_regripv2_common import _RegripV2CfgHooks, _RegripV2Mixin

_EPS = 1e-9


# ================================ mechanism mixins ================================
class _ShedMixin:
    """Forced per-finger actuation dropout (chassis-based, object-agnostic). Bernoulli(1/interval)
    per step; target uniform among debounced-engaged, non-cooldown fingers; zero that finger's
    software-PD gains for shed_window steps, then restore + cooldown. interval is gravity-linked:
    shed_interval_t0 at |g|~0 -> shed_interval_t1 at |g|=10 (R2 uses t0=60, t1=1e9: a DECAYING seed;
    R4 uses t0=240, t1=80: pressure grows with gravity). Gains restored before any mid-shed reset."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        n_f = self.last_contacts.shape[1]
        act_idx = self.actuated_dof_indices
        act_list = act_idx.tolist() if torch.is_tensor(act_idx) else list(act_idx)
        names = [self.hand.joint_names[i] for i in act_list]
        mask = torch.zeros(n_f, len(names), dtype=torch.bool, device=dev)
        for fi, key in enumerate(("thumb", "index", "middle", "ring", "pinky")):
            mask[fi] = torch.tensor([key in n for n in names], dtype=torch.bool, device=dev)
        self._sh_mask = mask
        self._sh_nom_p = self.p_gain.clone()
        self._sh_nom_d = self.d_gain.clone()
        self._sh_finger = torch.full((N,), -1, dtype=torch.long, device=dev)
        self._sh_timer = torch.zeros(N, device=dev)
        self._sh_cd = torch.zeros((N, n_f), device=dev)
        self._sh_count = torch.zeros((N, n_f), device=dev)      # sheds started, per finger
        self._sh_prove = torch.zeros((N, n_f), device=dev)      # 30-step survive countdown post-shed
        self._sh_proven = torch.zeros((N, n_f), device=dev)     # survived >=1 shed

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if not hasattr(self, "_sh_timer"):
            return
        cfg, N, dev = self.cfg, self.num_envs, self.device
        active = (self._gx_settle == 0) if hasattr(self, "_gx_settle") else \
            torch.ones(N, dtype=torch.bool, device=dev)
        g = self.physics_sim_view.get_gravity()
        t = min(((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5) / 10.0, 1.0)
        interval = float(cfg.shed_interval_t0) + (float(cfg.shed_interval_t1) - float(cfg.shed_interval_t0)) * t
        window = float(getattr(cfg, "shed_window", 12.0))
        eng = self._v2_eng if hasattr(self, "_v2_eng") else (self.last_contacts > 0.15).float()
        # shed gravity gate (2026-07-03 PM): at near-zero g a shed impulse launches the weightless
        # ball into the base displacement termination -> curriculum stall (loop4 v1/v2 diagnosis).
        if t * 10.0 < float(getattr(cfg, "shed_min_g", 0.0)):
            return
        self._sh_cd = (self._sh_cd - active.float().unsqueeze(-1)).clamp_min(0.0)
        # advance timers; restore on end
        shedding = self._sh_timer > 0.0
        self._sh_timer = (self._sh_timer - (active & shedding).float()).clamp_min(0.0)
        ended = shedding & (self._sh_timer <= 0.0)
        if bool(ended.any()):
            ids = ended.nonzero(as_tuple=False).squeeze(-1)
            fm = self._sh_mask[self._sh_finger[ids].clamp_min(0)]
            self.p_gain[ids] = torch.where(fm, self._sh_nom_p[ids], self.p_gain[ids])
            self.d_gain[ids] = torch.where(fm, self._sh_nom_d[ids], self.d_gain[ids])
            self._sh_cd[ids, self._sh_finger[ids].clamp_min(0)] = 40.0
            self._sh_prove[ids, self._sh_finger[ids].clamp_min(0)] = 30.0
            self._sh_finger[ids] = -1
        # prove countdown: survive 30 steps after a shed ends -> proven
        prove = self._sh_prove > 0.0
        self._sh_prove = (self._sh_prove - (active.unsqueeze(-1) & prove).float()).clamp_min(0.0)
        proven_now = prove & (self._sh_prove <= 0.0)
        self._sh_proven = torch.maximum(self._sh_proven, proven_now.float())
        # trigger
        eligible = (eng > 0.5) & (self._sh_cd <= 0.0)
        trig = active & (~(self._sh_timer > 0.0)) & \
               (torch.rand(N, device=dev) < 1.0 / max(interval, 1.0)) & eligible.any(-1)
        if bool(trig.any()):
            r = torch.rand(N, eng.shape[1], device=dev) * eligible.float()
            fin = r.argmax(-1)
            ids = trig.nonzero(as_tuple=False).squeeze(-1)
            self._sh_finger[ids] = fin[ids]
            self._sh_timer[ids] = window
            self._sh_count[ids, fin[ids]] += 1.0
            fm = self._sh_mask[fin[ids]]
            self.p_gain[ids] = self.p_gain[ids] * (~fm).float()
            self.d_gain[ids] = self.d_gain[ids] * (~fm).float()

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if hasattr(self, "_sh_timer"):
            ids = self.hand._ALL_INDICES if env_ids is None else env_ids
            ids_t = ids if torch.is_tensor(ids) else torch.as_tensor(ids, device=self.device, dtype=torch.long)
            mid = ids_t[self._sh_timer[ids_t] > 0.0]
            if mid.numel() > 0:
                fm = self._sh_mask[self._sh_finger[mid].clamp_min(0)]
                self.p_gain[mid] = torch.where(fm, self._sh_nom_p[mid], self.p_gain[mid])
                self.d_gain[mid] = torch.where(fm, self._sh_nom_d[mid], self.d_gain[mid])
        super()._reset_idx(env_ids)
        if not hasattr(self, "_sh_timer"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._sh_nom_p[ids] = self.p_gain[ids]
        self._sh_nom_d[ids] = self.d_gain[ids]
        self._sh_finger[ids] = -1
        self._sh_timer[ids] = 0.0
        self._sh_cd[ids] = 0.0
        self._sh_count[ids] = 0.0
        self._sh_prove[ids] = 0.0
        self._sh_proven[ids] = 0.0


class _RotLoadMixin:
    """Rotating lateral load (object-agnostic, half-strength D4 mechanism). Overwrites rb_forces
    AFTER the backbone's perturbation block (set perturb_prob=0 in cfgs using this)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._rl_phi = torch.rand(self.num_envs, device=self.device) * 2.0 * math.pi

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if not hasattr(self, "_rl_phi"):
            return
        cfg, N, dev = self.cfg, self.num_envs, self.device
        settling = (self._gx_settle > 0) if hasattr(self, "_gx_settle") else \
            torch.zeros(N, dtype=torch.bool, device=dev)
        g = self.physics_sim_view.get_gravity()
        t = min(((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5) / 10.0, 1.0)
        omega = float(cfg.rv_omega_start) + (float(cfg.rv_omega_end) - float(cfg.rv_omega_start)) * t
        g_extra = float(cfg.rv_g_base) + (float(cfg.rv_g_max) - float(cfg.rv_g_base)) * t
        self._rl_phi = torch.remainder(self._rl_phi + 2.0 * math.pi * omega * self.step_dt, 2.0 * math.pi)
        m = self.object.root_physx_view.get_masses().reshape(N).to(dev)
        fmag = m * g_extra
        self.rb_forces[:, 0] = fmag * torch.cos(self._rl_phi)
        self.rb_forces[:, 1] = fmag * torch.sin(self._rl_phi)
        self.rb_forces[:, 2] = 0.0
        if bool(settling.any()):
            self.rb_forces[settling] = 0.0
        tq = self._perturb_torques if hasattr(self, "_perturb_torques") else torch.zeros(N, 3, device=dev)
        tq[:] = 0.0
        self.object.set_external_force_and_torque(
            forces=self.rb_forces.reshape(N, 1, 3), torques=tq.reshape(N, 1, 3))
        self.extras["rv_omega"] = torch.tensor(omega, device=dev)
        self.extras["rv_gextra"] = torch.tensor(g_extra, device=dev)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_rl_phi"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        ids_t = ids if torch.is_tensor(ids) else torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self._rl_phi[ids_t] = torch.rand(ids_t.numel(), device=self.device) * 2.0 * math.pi


# ================================ design reward mixins ================================
class _R1LedgerMixin:
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        n_f = self.last_contacts.shape[1]
        self._r1_on = torch.zeros((N, n_f), device=dev)
        self._r1_pending = torch.zeros((N, n_f), device=dev)
        self._r1_events = torch.zeros(N, device=dev)
        self._r1_min_at_event = torch.zeros(N, device=dev)
        self._r1_prev_transfers = torch.zeros(N, device=dev)
        self._r1_transfer_paid = torch.zeros(N, device=dev)

    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_r1_on"):
            return total
        com = self._v2_common()
        eng, active, clear, balanced = com["eng"], com["active"], com["clear"], com["balanced"]
        act = active.float().unsqueeze(-1)
        # maximin cumulative support
        r_max = 1.0 * (5.0 * com["min_share"]).clamp(0.0, 1.0) * clear.float()
        # re-contact event (chassis off_cnt >= 5 -> re-engage -> sustain 5) gated on ledger growth
        qualify = eng * (self._v2_off_cnt >= 5.0).float() * act
        self._r1_pending = torch.maximum(self._r1_pending, qualify)
        on_new = torch.where(eng > 0.5, self._r1_on + 1.0, torch.zeros_like(self._r1_on))
        self._r1_on = act * on_new + (1.0 - act) * self._r1_on
        self._r1_pending = self._r1_pending * eng
        fire = (self._r1_pending * (self._r1_on >= 5.0).float() * act).amax(-1)
        self._r1_pending = self._r1_pending * (1.0 - (self._r1_on >= 5.0).float() * act)
        grew = com["min_share"] > self._r1_min_at_event + 0.005
        pay = (fire > 0.5) & grew & clear & (self._r1_events < 5.0)
        r_ev = pay.float() * 1.0
        self._r1_events += pay.float()
        self._r1_min_at_event = torch.where(pay, com["min_share"], self._r1_min_at_event)
        # transfer bonus (chassis arm events)
        dt_tr = (self._v2_transfer_events - self._r1_prev_transfers).clamp(0.0, 2.0)
        self._r1_prev_transfers = self._v2_transfer_events.clone()
        can = (self._r1_transfer_paid < 5.0) & clear
        r_tr = 0.5 * dt_tr * can.float()
        self._r1_transfer_paid += dt_tr * can.float()
        self.extras["r1_maximin"] = r_max.mean()
        self.extras["r1_events"] = self._r1_events.mean()
        return total + r_max + r_ev + r_tr

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_r1_on"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        for b in (self._r1_on, self._r1_pending):
            b[ids] = 0.0
        for b in (self._r1_events, self._r1_min_at_event, self._r1_prev_transfers, self._r1_transfer_paid):
            b[ids] = 0.0


class _R2ChainMixin:
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        self._r2_paid = torch.zeros(N, device=dev)
        self._r2_quota_paid = torch.zeros(N, device=dev)

    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_r2_paid"):
            return total
        com = self._v2_common()
        r_dense = 0.3 * com["clear"].float() * com["balanced"].float()
        ho = com["completed_handover"].sum(-1)                       # (N,) completed this step
        can = self._r2_paid < 6.0
        r_ho = 1.0 * torch.minimum(ho, (6.0 - self._r2_paid).clamp_min(0.0)) * can.float()
        self._r2_paid += r_ho
        distinct = (self._v2_handover_done > 0.5).float().sum(-1)
        hit = (distinct >= 4.0) & (self._r2_quota_paid < 0.5)
        r_all = hit.float() * 5.0
        self._r2_quota_paid = torch.maximum(self._r2_quota_paid, hit.float())
        self.extras["r2_handovers_paid"] = self._r2_paid.mean()
        self.extras["r2_distinct"] = distinct.mean()
        return total + r_dense + r_ho + r_all

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_r2_paid"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._r2_paid[ids] = 0.0
        self._r2_quota_paid[ids] = 0.0


class _R3MarginMixin:
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_v2_eng"):
            return total
        com = self._v2_common()
        F, eng = com["F"], com["eng"]
        n_eng = eng.sum(-1)
        base = com["Fsum"] + com["m"].unsqueeze(-1) * com["gvec"] + com["F_ext"]      # (N,3)
        rho = (base.unsqueeze(1) - F).norm(dim=-1) / (com["m"] * com["gmag"]).clamp_min(_EPS).unsqueeze(-1)
        rho = torch.where(eng > 0.5, rho, torch.full_like(rho, 9.0))
        M = (1.0 - rho.min(dim=-1).values.clamp(0.0, 1.0)) * (n_eng >= 2.0).float()
        r_m = 1.5 * M * com["clear"].float()
        # timeout-only coverage bonus
        tout = self.reset_time_outs & (~self._v2_fail_now)
        r_end = 3.0 * (5.0 * com["min_share"]).clamp(0.0, 1.0) * tout.float()
        self.extras["r3_margin"] = M.mean()
        return total + r_m + r_end


class _R4ShedRewardMixin:
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._r4_all_paid = torch.zeros(self.num_envs, device=self.device)

    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_r4_all_paid"):
            return total
        com = self._v2_common()
        r_dense = 0.3 * com["clear"].float() * com["balanced"].float()
        proven_all = (self._sh_proven.sum(-1) >= 5.0) if hasattr(self, "_sh_proven") else \
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        hit = proven_all & (self._r4_all_paid < 0.5)
        r_all = hit.float() * 4.0
        self._r4_all_paid = torch.maximum(self._r4_all_paid, hit.float())
        if hasattr(self, "_sh_proven"):
            self.extras["r4_proven_mean"] = self._sh_proven.sum(-1).mean()
        return total + r_dense + r_all

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_r4_all_paid"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._r4_all_paid[ids] = 0.0


class _R5EntropyMixin:
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        self._r5_hmax = torch.zeros(N, device=dev)
        self._r5_big_paid = torch.zeros(N, device=dev)
        self._r5_prev_transfers = torch.zeros(N, device=dev)
        self._r5_transfer_paid = torch.zeros(N, device=dev)

    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_r5_hmax"):
            return total
        com = self._v2_common()
        p = com["share"].clamp_min(_EPS)
        H = (-(p * p.log()).sum(-1) / math.log(5.0)).clamp(0.0, 1.0)
        H = torch.where(com["Lsum"] > _EPS, H, torch.zeros_like(H))
        grow = (H - self._r5_hmax).clamp_min(0.0) * com["clear"].float()
        self._r5_hmax = torch.maximum(self._r5_hmax, H * com["clear"].float())
        r_ratchet = 2.0 * grow
        r_level = 0.3 * H * com["clear"].float() * com["balanced"].float()
        hit = (H >= 0.9) & com["clear"] & (self._r5_big_paid < 0.5)
        r_big = hit.float() * 5.0
        self._r5_big_paid = torch.maximum(self._r5_big_paid, hit.float())
        dt_tr = (self._v2_transfer_events - self._r5_prev_transfers).clamp(0.0, 2.0)
        self._r5_prev_transfers = self._v2_transfer_events.clone()
        can = (self._r5_transfer_paid < 5.0) & com["clear"]
        r_tr = 0.5 * dt_tr * can.float()
        self._r5_transfer_paid += dt_tr * can.float()
        self.extras["r5_H"] = H.mean()
        self.extras["r5_hmax"] = self._r5_hmax.mean()
        return total + r_ratchet + r_level + r_big + r_tr

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_r5_hmax"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        for b in (self._r5_hmax, self._r5_big_paid, self._r5_prev_transfers, self._r5_transfer_paid):
            b[ids] = 0.0


# ================================ BALL cfgs/envs (perturb backbone) ================================
@configclass
class _RegripV2BallCfgBase(_RegripV2CfgHooks, SharpaWaveGraspXLAdjustHoldPerturbCfg):
    """v2 ball backbone: perturb chain with contact-COUNT terms zeroed; chassis owns gravity."""
    w_cov = 0.0
    n_engaged_reward_scale = 0.0
    n_engaged_episode_scale = 0.0
    gravity_curriculum = False           # base/graspxl auto-advance OFF (chassis owns the curriculum)
    rgv2_gravity_curriculum = True
    rgv2_curr_term_thresh = 2.5e-3   # raised from 1e-3 (2026-07-02 hotfix): the ball's transit-dip
    # failures pinned fail_ema at ~1.1e-3 and froze gravity at 0.05 for a whole run; ~60% episode-
    # failure ceiling still halts on catastrophe but lets the curriculum move while learning.
    rgv2_drop_below = 0.03
    rgv2_drop_debounce = 3
    rgv2_terminal_penalty = 15.0
    rgv2_clear_margin = 0.015
    rgv2_balance_tol = 0.3
    rgv2_eng_on = 0.15
    rgv2_eng_off = 0.05


@configclass
class RegripV2R1BallCfg(_RegripV2BallCfgBase):
    """R1 LEDGER (ball)."""


@configclass
class RegripV2R2BallCfg(_RegripV2BallCfgBase):
    """R2 CHAIN (ball) + decaying forced-shed seed (frequent at low g, off at full g)."""
    shed_interval_t0 = 60.0
    shed_interval_t1 = 1e9
    shed_window = 12.0
    chain_seed_prob_start = 1.0          # marker field for the eval-clean hook
    chain_seed_prob_end = 0.0


@configclass
class RegripV2R3BallCfg(_RegripV2BallCfgBase):
    """R3 MARGIN (ball): rotating load replaces the stochastic wrench."""
    perturb_prob = 0.0
    rv_omega_start = 0.0625              # 1 rev / 16 s
    rv_omega_end = 0.125                 # 1 rev / 8 s
    rv_g_base = 0.3
    rv_g_max = 5.0


@configclass
class RegripV2R4BallCfg(_RegripV2BallCfgBase):
    """R4 SHED (ball): dropout pressure grows with gravity."""
    shed_interval_t0 = 240.0
    shed_interval_t1 = 80.0
    shed_window = 12.0


@configclass
class RegripV2R5BallCfg(_RegripV2BallCfgBase):
    """R5 ENTROPY (ball)."""


_BallBase = SharpaWaveGraspXLAdjustHoldPerturbEnv


class RegripV2R1BallEnv(_R1LedgerMixin, _RegripV2Mixin, _BallBase):
    pass


class RegripV2R2BallEnv(_R2ChainMixin, _ShedMixin, _RegripV2Mixin, _BallBase):
    pass


class RegripV2R3BallEnv(_R3MarginMixin, _RotLoadMixin, _RegripV2Mixin, _BallBase):
    pass


class RegripV2R4BallEnv(_R4ShedRewardMixin, _ShedMixin, _RegripV2Mixin, _BallBase):
    pass


class RegripV2R5BallEnv(_R5EntropyMixin, _RegripV2Mixin, _BallBase):
    pass


# ================================ CYLINDER backbone + cfgs/envs ================================
class _RegripV2CylBaseEnv(SharpaWaveInhandRotateEnv):
    """Plain cylinder task + the gravity-linked perturbation block (ported from the perturb env,
    object-agnostic). rotate_reward_scale=0 in the cfg makes the base reward hold/penalties only."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._perturb_torques = torch.zeros(self.num_envs, 3, device=self.device)

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if not hasattr(self, "_perturb_torques"):
            return
        cfg, N, dev = self.cfg, self.num_envs, self.device
        g = self.physics_sim_view.get_gravity()
        t = min(((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5) /
                max(float(getattr(cfg, "perturb_curr_g_full", 10.0)), 1e-6), 1.0)
        f_scale = float(cfg.perturb_force_base) + (float(cfg.perturb_force_max) - float(cfg.perturb_force_base)) * t
        t_scale = float(cfg.perturb_torque_base) + (float(cfg.perturb_torque_max) - float(cfg.perturb_torque_base)) * t
        decay = float(cfg.perturb_decay) ** (self.physics_dt / float(cfg.perturb_decay_interval))
        self.rb_forces *= decay
        self._perturb_torques *= decay
        hit = torch.rand(N, device=dev) < float(cfg.perturb_prob)
        ids = hit.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            m = self.object.root_physx_view.get_masses().reshape(N).to(dev)[ids, None]
            self.rb_forces[ids] = torch.randn(ids.numel(), 3, device=dev) * m * f_scale
            self._perturb_torques[ids] = torch.randn(ids.numel(), 3, device=dev) * m * t_scale
        self.object.set_external_force_and_torque(
            forces=self.rb_forces.reshape(N, 1, 3), torques=self._perturb_torques.reshape(N, 1, 3))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_perturb_torques"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._perturb_torques[ids] = 0.0


@configclass
class _RegripV2CylCfgBase(_RegripV2CfgHooks, SharpaWaveEnvCfg):
    """v2 cylinder backbone: plain cylinder task, ROTATION ZEROED, grasp-cache reset, gravity-linked
    perturbation (same fields as the ball backbone), chassis-owned gravity curriculum."""
    rotate_reward_scale = 0.0
    force_scale = 0.0                    # base random-force block OFF (we run our own)
    random_force_prob_scalar = 0.0
    gravity_curriculum = False
    rgv2_gravity_curriculum = True
    rgv2_curr_term_thresh = 2.5e-3   # raised from 1e-3 (2026-07-02 hotfix): the ball's transit-dip
    # failures pinned fail_ema at ~1.1e-3 and froze gravity at 0.05 for a whole run; ~60% episode-
    # failure ceiling still halts on catastrophe but lets the curriculum move while learning.
    rgv2_drop_below = 0.03
    rgv2_drop_debounce = 3
    rgv2_terminal_penalty = 15.0
    rgv2_clear_margin = 0.015
    rgv2_balance_tol = 0.3
    rgv2_eng_on = 0.15
    rgv2_eng_off = 0.05
    perturb_force_base = 0.5
    perturb_force_max = 10.0
    perturb_torque_base = 0.02
    perturb_torque_max = 0.10
    perturb_curr_g_full = 10.0
    perturb_prob = 0.25
    perturb_decay = 0.9
    perturb_decay_interval = 0.08


@configclass
class RegripV2R1CylCfg(_RegripV2CylCfgBase):
    """R1 LEDGER (cylinder)."""


@configclass
class RegripV2R2CylCfg(_RegripV2CylCfgBase):
    """R2 CHAIN (cylinder)."""
    shed_interval_t0 = 60.0
    shed_interval_t1 = 1e9
    shed_window = 12.0
    chain_seed_prob_start = 1.0
    chain_seed_prob_end = 0.0


@configclass
class RegripV2R3CylCfg(_RegripV2CylCfgBase):
    """R3 MARGIN (cylinder)."""
    perturb_prob = 0.0
    rv_omega_start = 0.0625
    rv_omega_end = 0.125
    rv_g_base = 0.3
    rv_g_max = 5.0


@configclass
class RegripV2R4CylCfg(_RegripV2CylCfgBase):
    """R4 SHED (cylinder)."""
    shed_interval_t0 = 240.0
    shed_interval_t1 = 80.0
    shed_window = 12.0


@configclass
class RegripV2R5CylCfg(_RegripV2CylCfgBase):
    """R5 ENTROPY (cylinder)."""


class RegripV2R1CylEnv(_R1LedgerMixin, _RegripV2Mixin, _RegripV2CylBaseEnv):
    pass


class RegripV2R2CylEnv(_R2ChainMixin, _ShedMixin, _RegripV2Mixin, _RegripV2CylBaseEnv):
    pass


class RegripV2R3CylEnv(_R3MarginMixin, _RotLoadMixin, _RegripV2Mixin, _RegripV2CylBaseEnv):
    pass


class RegripV2R4CylEnv(_R4ShedRewardMixin, _ShedMixin, _RegripV2Mixin, _RegripV2CylBaseEnv):
    pass


class RegripV2R5CylEnv(_R5EntropyMixin, _RegripV2Mixin, _RegripV2CylBaseEnv):
    pass


# ================================ v3net A/B variants (2026-07-02) ================================
# Same envs/rewards as R1/R4 ball; ONLY the priv buffer widens to 23 (chassis fills dims 8:23 with
# object quat/linvel/angvel + debounced contacts — critic-side only via ppo_cfg_v3net.yaml).
@configclass
class RegripV2R1BallNetCfg(RegripV2R1BallCfg):
    """R1 LEDGER (ball) + v3net extended critic priv."""
    priv_info_dim = 23


@configclass
class RegripV2R4BallNetCfg(RegripV2R4BallCfg):
    """R4 SHED (ball) + v3net extended critic priv."""
    priv_info_dim = 23


# ============================ R4-family sweep (2026-07-02 PM): R4L / R4R / R4S ============================
# User-approved 4x2 matrix (R4 control + three R4-based designs, ball + cylinder, self-collision default).
# All three keep R4's dropout mechanism (_ShedMixin) and use SELF-CONTAINED clear-only gates (the shared
# dense terms above are left untouched per the user's file state). Judged against the dual criterion:
# (a) sustained per-finger contact switching, or (b) adjustment to a stable spread grasp.
class _R4LedgerMixin:
    """R4L 'Shed-Ledger': dropout bootstraps R1's maximin ledger + transfer bonus (target (b)).
    r_spread = w_spread*(5*min per-finger cumulative load share)*clear   [continuous maximin]
    r_transfer = w_transfer per chassis smooth-transfer event, quota, clear-gated."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        self._r4l_prev_tr = torch.zeros(N, device=dev)
        self._r4l_tr_paid = torch.zeros(N, device=dev)

    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_r4l_prev_tr"):
            return total
        cfg = self.cfg
        com = self._v2_common()
        clear = com["clear"].float()
        r_spread = float(cfg.r4l_w_spread) * (5.0 * com["min_share"]).clamp(0.0, 1.0) * clear
        dt_tr = (self._v2_transfer_events - self._r4l_prev_tr).clamp(0.0, 2.0)
        self._r4l_prev_tr = self._v2_transfer_events.clone()
        can = (self._r4l_tr_paid < float(cfg.r4l_transfer_quota)) & com["clear"]
        r_tr = float(cfg.r4l_w_transfer) * dt_tr * can.float()
        self._r4l_tr_paid += dt_tr * can.float()
        self.extras["r4l_spread"] = r_spread.mean()
        self.extras["r4l_min_share"] = com["min_share"].mean()
        return total + r_spread + r_tr

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_r4l_prev_tr"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._r4l_prev_tr[ids] = 0.0
        self._r4l_tr_paid[ids] = 0.0


class _R4RecruitMixin:
    """R4R 'Shed-Recruit': monotone per-finger recruitment (target (a) participation).
    +r4r_tier one-time per DISTINCT finger on its first SUSTAINED (>=r4r_streak steps) LOAD-BEARING
    (share > r4r_load_share_min) contact while clear; persistent r_cov = w_cov*(distinct/5)*clear.
    Monotone-distinct by construction — the instantaneous-count cage of v1 cannot pay here."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        n_f = self.last_contacts.shape[1]
        self._r4r_streak = torch.zeros((N, n_f), device=dev)
        self._r4r_recruited = torch.zeros((N, n_f), device=dev)

    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_r4r_streak"):
            return total
        cfg = self.cfg
        com = self._v2_common()
        clear, active = com["clear"], com["active"]
        loaded = (com["eng"] > 0.5) & (com["lam"] > float(cfg.r4r_load_share_min))
        ok = loaded & (clear & active).unsqueeze(-1)
        self._r4r_streak = torch.where(ok, self._r4r_streak + 1.0,
                                       torch.where(active.unsqueeze(-1), torch.zeros_like(self._r4r_streak),
                                                   self._r4r_streak))
        newly = (self._r4r_streak >= float(cfg.r4r_streak)) & (self._r4r_recruited < 0.5)
        r_tier = float(cfg.r4r_tier) * newly.float().sum(-1)
        self._r4r_recruited = torch.maximum(self._r4r_recruited, newly.float())
        # contact-gate the latched coverage income (escape-toss guard: airborne must not pay)
        any_contact = (com["eng"].sum(-1) >= 1.0).float()
        r_cov = float(cfg.r4r_w_cov) * (self._r4r_recruited.sum(-1) / 5.0) * clear.float() * any_contact
        self.extras["r4r_recruited"] = self._r4r_recruited.sum(-1).mean()
        return total + r_tier + r_cov

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_r4r_streak"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._r4r_streak[ids] = 0.0
        self._r4r_recruited[ids] = 0.0


class _R4SwitchMixin:
    """R4S 'Shed-AllSwitch': ACTIVE per-finger contact switches; LARGE bonus when all five have
    switched (target (a), repeatable). A switch = debounced release (>=k_off active steps off) ->
    re-contact sustained >= m_on steps, while clear — EXCLUDING releases caused by the finger's own
    shed window (taint flag set while a finger is the shed target, cleared on its next re-contact):
    the +8 pays only policy-initiated switching. First switch per finger +r4s_first; when all five
    fingers have >=1 active switch: +r4s_all, flags reset, up to r4s_rounds_cap rounds/episode."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        n_f = self.last_contacts.shape[1]
        self._r4s_on = torch.zeros((N, n_f), device=dev)
        self._r4s_pending = torch.zeros((N, n_f), device=dev)
        self._r4s_taint = torch.zeros((N, n_f), device=dev)
        self._r4s_switched = torch.zeros((N, n_f), device=dev)
        self._r4s_first_paid = torch.zeros((N, n_f), device=dev)
        self._r4s_rounds = torch.zeros(N, device=dev)

    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_r4s_on"):
            return total
        cfg = self.cfg
        com = self._v2_common()
        eng, clear, active = com["eng"], com["clear"], com["active"]
        act = active.float().unsqueeze(-1)
        # taint: finger is currently the shed target -> its release is forced, not active
        if hasattr(self, "_sh_finger"):
            shed_now = torch.zeros_like(self._r4s_taint)
            has = self._sh_finger >= 0
            if bool(has.any()):
                ids = has.nonzero(as_tuple=False).squeeze(-1)
                shed_now[ids, self._sh_finger[ids]] = 1.0
            self._r4s_taint = torch.maximum(self._r4s_taint, shed_now)
        # qualify on re-contact after >=k_off debounced off steps (chassis counter), untainted only
        qualify = eng * (self._v2_off_cnt >= float(cfg.r4s_k_off)).float() * act \
            * (1.0 - self._r4s_taint)
        self._r4s_pending = torch.maximum(self._r4s_pending, qualify)
        # taint clears once the finger re-engages (its next cycle can qualify again)
        self._r4s_taint = self._r4s_taint * (1.0 - eng)
        on_new = torch.where(eng > 0.5, self._r4s_on + 1.0, torch.zeros_like(self._r4s_on))
        self._r4s_on = act * on_new + (1.0 - act) * self._r4s_on
        self._r4s_pending = self._r4s_pending * eng
        fire = self._r4s_pending * (self._r4s_on >= float(cfg.r4s_m_on)).float() * act \
            * clear.float().unsqueeze(-1)
        self._r4s_pending = self._r4s_pending * (1.0 - fire)
        # first-switch tier per finger
        newly_first = (fire > 0.5) & (self._r4s_first_paid < 0.5)
        r_first = float(cfg.r4s_first) * newly_first.float().sum(-1)
        self._r4s_first_paid = torch.maximum(self._r4s_first_paid, newly_first.float())
        # round bonus: all five switched -> +r4s_all, reset flags, cap rounds
        self._r4s_switched = torch.maximum(self._r4s_switched, (fire > 0.5).float())
        all5 = (self._r4s_switched.sum(-1) >= 5.0) & (self._r4s_rounds < float(cfg.r4s_rounds_cap)) \
            & clear & active
        r_all = all5.float() * float(cfg.r4s_all)
        if bool(all5.any()):
            self._r4s_switched[all5] = 0.0
            self._r4s_rounds += all5.float()
        self.extras["r4s_fire_mean"] = fire.sum(-1).mean()
        self.extras["r4s_switched_mean"] = self._r4s_switched.sum(-1).mean()
        self.extras["r4s_rounds_mean"] = self._r4s_rounds.mean()
        return total + r_first + r_all

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_r4s_on"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        for b in (self._r4s_on, self._r4s_pending, self._r4s_taint, self._r4s_switched,
                  self._r4s_first_paid):
            b[ids] = 0.0
        self._r4s_rounds[ids] = 0.0


# ---- cfgs: ball + cylinder per design, priv 23 (v3net), shed params per design ----
@configclass
class RegripV2R4CylNetCfg(RegripV2R4CylCfg):
    """R4 control (cylinder) + v3net critic priv."""
    priv_info_dim = 23


@configclass
class RegripV2R4LBallCfg(RegripV2R4BallCfg):
    """R4L Shed-Ledger (ball)."""
    priv_info_dim = 23
    r4l_w_spread = 1.0
    r4l_w_transfer = 0.5
    r4l_transfer_quota = 5.0


@configclass
class RegripV2R4LCylCfg(RegripV2R4CylCfg):
    """R4L Shed-Ledger (cylinder)."""
    priv_info_dim = 23
    r4l_w_spread = 1.0
    r4l_w_transfer = 0.5
    r4l_transfer_quota = 5.0


@configclass
class RegripV2R4RBallCfg(RegripV2R4BallCfg):
    """R4R Shed-Recruit (ball): sheds never fully anneal."""
    priv_info_dim = 23
    shed_interval_t1 = 60.0
    r4r_tier = 0.5
    r4r_w_cov = 0.5
    r4r_load_share_min = 0.05
    r4r_streak = 5.0


@configclass
class RegripV2R4RCylCfg(RegripV2R4CylCfg):
    """R4R Shed-Recruit (cylinder)."""
    priv_info_dim = 23
    shed_interval_t1 = 60.0
    r4r_tier = 0.5
    r4r_w_cov = 0.5
    r4r_load_share_min = 0.05
    r4r_streak = 5.0


@configclass
class RegripV2R4SBallCfg(RegripV2R4BallCfg):
    """R4S Shed-AllSwitch (ball)."""
    priv_info_dim = 23
    r4s_k_off = 5.0
    r4s_m_on = 5.0
    r4s_first = 0.5
    r4s_all = 8.0
    r4s_rounds_cap = 2.0


@configclass
class RegripV2R4SCylCfg(RegripV2R4CylCfg):
    """R4S Shed-AllSwitch (cylinder)."""
    priv_info_dim = 23
    r4s_k_off = 5.0
    r4s_m_on = 5.0
    r4s_first = 0.5
    r4s_all = 8.0
    r4s_rounds_cap = 2.0


class RegripV2R4LBallEnv(_R4LedgerMixin, _R4ShedRewardMixin, _ShedMixin, _RegripV2Mixin, _BallBase):
    pass


class RegripV2R4LCylEnv(_R4LedgerMixin, _R4ShedRewardMixin, _ShedMixin, _RegripV2Mixin, _RegripV2CylBaseEnv):
    pass


class RegripV2R4RBallEnv(_R4RecruitMixin, _R4ShedRewardMixin, _ShedMixin, _RegripV2Mixin, _BallBase):
    pass


class RegripV2R4RCylEnv(_R4RecruitMixin, _R4ShedRewardMixin, _ShedMixin, _RegripV2Mixin, _RegripV2CylBaseEnv):
    pass


class RegripV2R4SBallEnv(_R4SwitchMixin, _R4ShedRewardMixin, _ShedMixin, _RegripV2Mixin, _BallBase):
    pass


class RegripV2R4SCylEnv(_R4SwitchMixin, _R4ShedRewardMixin, _ShedMixin, _RegripV2Mixin, _RegripV2CylBaseEnv):
    pass


# ============================ CLOSED-LOOP iter-2: R4RL (2026-07-03) ============================
# R4R recruit tiers + R4L clear-gated maximin/transfer on the R4 shed core, with the dropout WEANED
# as gravity grows (teach early, wean late) — on the ORIGINAL network config (priv 8, entropy 0,
# H=8: the replication-confirmed spread-capable regime). Weights are cfg fields so the closed loop
# can adjust them from iteration-1 rotation evidence without new code.
@configclass
class RegripV2R4RLBallCfg(_RegripV2BallCfgBase):
    """R4RL closed-loop regrip (ball, old-net agent cfg via registration)."""
    # shed: teach hard at low gravity, wean by full gravity
    shed_interval_t0 = 60.0
    shed_interval_t1 = 600.0
    shed_window = 12.0
    # R4L terms
    r4l_w_spread = 1.0
    r4l_w_transfer = 0.5
    r4l_transfer_quota = 5.0
    # R4R terms
    r4r_tier = 0.5
    r4r_w_cov = 0.5
    r4r_load_share_min = 0.05
    r4r_streak = 5.0
    # CLOSED-LOOP iter-1 finding (2026-07-03): 14 N symmetric squeezes rotation-lock the ball while
    # holding fine — penalize total fingertip force above a cap (anti-squeeze; keeps grips firm but
    # not crushing, aiming at rotatable initializations).
    r4rl_force_cap = 6.0     # N, total debounced fingertip force allowed penalty-free
    r4rl_w_squeeze = 0.3
    rgv2_rise_above = 0.05   # m; terminate sustained upward escape (anti-toss, closed-loop finding 3)


class RegripV2R4RLBallEnv(_R4LedgerMixin, _R4RecruitMixin, _R4ShedRewardMixin, _ShedMixin,
                          _RegripV2Mixin, _BallBase):
    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_v2_eng"):
            return total
        com = self._v2_common()
        over = ((com["sup"] - float(self.cfg.r4rl_force_cap)) / float(self.cfg.r4rl_force_cap)).clamp(min=0.0)
        r_squeeze = -float(self.cfg.r4rl_w_squeeze) * over.clamp(max=2.0)
        self.extras["r4rl_squeeze_pen"] = (-r_squeeze).mean()
        self.extras["r4rl_sup_mean"] = com["sup"].mean()
        return total + r_squeeze


# ============================ COMBINED regrip+rotation (2026-07-03 AM) ============================
from .sharpa_wave_graspxl_bandrot import _BandRotMixin


@configclass
class RegripRotateCombinedBallCfg(RegripV2R4RLBallCfg):
    """R4RL regrip terms + HELD-GATED band rotation, from the pinch replay reset: the first joint
    adjust+rotate training. User's watch-item: regrip income may dominate ('adjust too much') —
    telemetry r_rot_mean vs regrip terms + band_held_frac tells the story."""
    band_held_gate = True
    band_held_min_fingers = 2
    band_held_disp = 0.05
    band_held_thresh = 0.15


class RegripRotateCombinedBallEnv(_BandRotMixin, _R4LedgerMixin, _R4RecruitMixin, _R4ShedRewardMixin,
                                  _ShedMixin, _RegripV2Mixin, _BallBase):
    """MRO: band mixin wraps the full R4RL regrip stack; all cooperative-super."""
    pass


@configclass
class RegripRotateMixedBallCfg(RegripRotateCombinedBallCfg):
    """Track A (2026-07-03): combined reward + MIXED resets — diverse_enclosing_frac of episodes start
    from the banked regrip poses (SHARPA_POSE_CACHE overrides the path), the rest from the pinch
    replay. The Q1 reset-mixing architecture, small-scale."""
    enclosing_cache_path = "results/closedloop_2026-07-03/mixed_regrip_poses.npy"
    diverse_enclosing_frac = 0.6
    shed_min_g = 1.0     # same low-g shed stall fix as R4RLN4 (loop4 diagnosis)

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        import os as _os
        c = _os.environ.get("SHARPA_POSE_CACHE", "").strip()
        if c:
            self.enclosing_cache_path = c


class RegripRotateMixedBallEnv(_BandRotMixin, _R4LedgerMixin, _R4RecruitMixin, _R4ShedRewardMixin,
                               _ShedMixin, _RegripV2Mixin, _BallBase):
    pass


# ============================ Track B escalation: R4RL-N4 (2026-07-03) ============================
# User: ">=4 supporting fingers must work in physics; the network just doesn't know how to get there."
# Show it: resets mixed 50% pinch / 50% VALIDATED 4-5-contact enclosing rows (from the annotated
# cache), + tiered n-income (clear- and contact-gated; cage guards: sink+rise terminations,
# anti-squeeze, monotone terms elsewhere).
from .sharpa_wave_graspxl_adjustrotate_diverse import SharpaWaveGraspXLAdjustRotateDiverseEnv


class _RN4IncomeMixin:
    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_v2_eng"):
            return total
        com = self._v2_common()
        n_deb = com["eng"].sum(-1)
        gate = com["clear"].float() * (n_deb >= 1.0).float()
        r_n = (float(self.cfg.r4n_w3) * (n_deb >= 3.0).float()
               + float(self.cfg.r4n_w4) * (n_deb >= 4.0).float()) * gate
        self.extras["r4n_income"] = r_n.mean()
        self.extras["r4n_n_deb"] = n_deb.mean()
        return total + r_n


@configclass
class RegripV2R4RLN4BallCfg(RegripV2R4RLBallCfg):
    """R4RL + tiered n-income + 50/50 pinch/enclosing-n45 resets."""
    enclosing_cache_path = "results/closedloop_2026-07-03/enclosing_n45.npy"
    diverse_enclosing_frac = 0.5
    r4n_w3 = 0.3
    r4n_w4 = 0.6
    shed_min_g = 1.0     # sheds only once |g| >= 1 (see _ShedMixin gravity gate)


class RegripV2R4RLN4BallEnv(_RN4IncomeMixin, _R4LedgerMixin, _R4RecruitMixin, _R4ShedRewardMixin,
                            _ShedMixin, _RegripV2Mixin, SharpaWaveGraspXLAdjustRotateDiverseEnv,
                            SharpaWaveGraspXLAdjustHoldPerturbEnv):
    """Diverse cache-reset (D2-precedent MRO) under the full R4RL regrip stack + n-income."""
    pass
