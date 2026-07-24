# REGRIP v2 shared chassis — load-bearing-centric regrip sweep (2026-07-02). ADDITIVE ONLY.
#
# Design doc: robotics-rl-expert/notes/reward_design/regrip_v2_load_bearing_designs_2026-07-02.md
# Implements the pieces every v2 design shares:
#   * SINK TERMINATION (user-fixed): terminate with a one-time −rgv2_terminal_penalty when object z
#     stays below z_start − rgv2_drop_below (3 cm) for rgv2_drop_debounce (3) consecutive control
#     steps. z_start = object z at the episode's FIRST ACTIVE step (post-settle on GraspXL replay;
#     first step otherwise). The whole fall-into-palm class is excluded from the MDP, per
#     reward-design-principles 1.3/2.5 (terminate, never per-frame-price).
#     The −penalty is applied to EVERY failure termination (sink, height band, displacement drop) —
#     uniform drop penalty. Timeouts are never penalized; end-of-episode bonuses pay on TIMEOUT ONLY.
#   * GRAVITY CURRICULUM (owned here; base/graspxl auto-advance MUST be off via gravity_curriculum=False):
#     +0.05 m/s^2 per step while the failure-termination EMA < rgv2_curr_term_thresh, up to 10.
#     Uniform for ball and cylinder backbones; disabled when rgv2_gravity_curriculum=False or under
#     SHARPA_EVAL_CLEAN=1.
#   * WRENCH LEDGER from per-finger elastomer force VECTORS (contact sensors 0-4, force_matrix_w
#     object-filtered): debounced engagement (on >0.15 N / off <0.05 N), load shares, cumulative
#     per-finger support ledger, wrench-balance flag ||sum F + m g + F_ext|| < tol * m|g|.
#   * HANDOVER STATE MACHINE (shared by R2's reward, R1/R5's transfer bonus, and the probe):
#     transfer (debounced release while total support holds >=80%) -> off >=5 steps -> re-contact
#     displaced >= 8 mm -> 10-step clear+balanced window => completed handover for that finger.
#   * SHARPA_EVAL_CLEAN=1 (env var, read at cfg time): zeroes perturbation and design mechanisms
#     (shed / rotating load / seed) and the gravity curriculum — probes and renders compare the
#     LEARNED GRASP under identical clean physics.
#
# Obs space unchanged. Self-collision arm via SHARPA_SELF_COLLISION (hook reused from regrip_gait).
from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from .sharpa_wave_graspxl_regrip_gait import apply_self_collision_override

_EPS = 1e-9


def apply_regripv2_eval_clean(cfg) -> None:
    """SHARPA_EVAL_CLEAN=1 -> disable perturbation + design mechanisms + gravity curriculum on this cfg.
    Field writes are hasattr-guarded so the same hook serves every backbone/design."""
    if os.environ.get("SHARPA_EVAL_CLEAN", "0").strip().lower() not in ("1", "true", "on"):
        return
    for f, v in [("perturb_prob", 0.0), ("perturb_force_base", 0.0), ("perturb_force_max", 0.0),
                 ("perturb_torque_base", 0.0), ("perturb_torque_max", 0.0),
                 ("force_scale", 0.0), ("random_force_prob_scalar", 0.0),
                 ("ls_interval_start", 1e9), ("ls_interval_end", 1e9),          # shed off
                 ("rv_g_base", 0.0), ("rv_g_max", 0.0),                          # rotating load off
                 ("chain_seed_prob_start", 0.0), ("chain_seed_prob_end", 0.0),   # R2 seed off
                 ("rgv2_gravity_curriculum", False)]:
        if hasattr(cfg, f):
            setattr(cfg, f, v)
    print("[regripv2] SHARPA_EVAL_CLEAN=1 -> perturbation/mechanisms/curriculum disabled for eval")


class _RegripV2CfgHooks:
    """__post_init__ mixin for every v2 cfg: self-collision arm + eval-clean overrides."""

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        apply_self_collision_override(self)
        apply_regripv2_eval_clean(self)


# Cfg fields every v2 cfg must define (values are the finalized defaults; declared per-cfg because
# the two backbones have different bases): rgv2_drop_below=0.03, rgv2_drop_debounce=3,
# rgv2_terminal_penalty=15.0, rgv2_clear_margin=0.015, rgv2_balance_tol=0.3,
# rgv2_eng_on=0.15, rgv2_eng_off=0.05, rgv2_gravity_curriculum=True, rgv2_curr_term_thresh=1e-3,
# gravity_curriculum=False (base auto-advance OFF — the mixin owns it),
# w_cov=0 n_engaged_reward_scale=0 n_engaged_episode_scale=0 (contact-COUNT terms are the cage vector).


class _RegripV2Mixin:
    """Chassis mixin. MRO position: FIRST (before the backbone env). Buffers created after
    super().__init__(); every hook hasattr-guards (DirectRLEnv calls _reset_idx during init)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        n_f = self.last_contacts.shape[1]
        self._v2_zstart = torch.zeros(N, device=dev)
        self._v2_started = torch.zeros(N, dtype=torch.bool, device=dev)
        self._v2_below_cnt = torch.zeros(N, device=dev)
        self._v2_sink_now = torch.zeros(N, dtype=torch.bool, device=dev)
        self._v2_fail_now = torch.zeros(N, dtype=torch.bool, device=dev)
        # debounced engagement + ledger
        self._v2_eng = torch.zeros((N, n_f), device=dev)
        self._v2_L = torch.zeros((N, n_f), device=dev)        # cumulative debounced |F_i| (N*s units)
        self._v2_act_steps = torch.zeros(N, device=dev)
        # handover state machine
        self._v2_sup_hist = torch.zeros((N, 6), device=dev)   # ring buffer of total support |F| sum
        self._v2_hist_ptr = 0
        self._v2_ho_state = torch.zeros((N, n_f), device=dev)         # 0=idle, 1=armed, 2=verify
        self._v2_off_cnt = torch.zeros((N, n_f), device=dev)
        self._v2_rel_pos = torch.zeros((N, n_f, 3), device=dev)       # fingertip pos at release
        self._v2_window_cnt = torch.zeros((N, n_f), device=dev)       # post-re-contact verify window
        self._v2_handover_done = torch.zeros((N, n_f), device=dev)    # completed handovers per finger
        self._v2_transfer_events = torch.zeros(N, device=dev)
        # curriculum bookkeeping
        self._v2_term_ema = torch.tensor(1.0, device=dev)             # start pessimistic
        self._v2_step_stamp = -1
        self._v2_cache = {}

    # ---------------- per-step shared computation (once per control step) ----------------
    def _v2_forces(self):
        F = torch.stack([self._contact_sensor[i].data.force_matrix_w[:, 0, 0, :] for i in range(5)], dim=1)
        return F                                                        # (N,5,3) world frame

    def _v2_common(self):
        step = int(self.common_step_counter)
        if self._v2_step_stamp == step and self._v2_cache:
            return self._v2_cache
        cfg = self.cfg
        N, dev = self.num_envs, self.device
        F = self._v2_forces()                                           # (N,5,3)
        mag = F.norm(dim=-1)                                            # (N,5)
        # debounced engagement (dual threshold)
        on_t, off_t = float(cfg.rgv2_eng_on), float(cfg.rgv2_eng_off)
        self._v2_eng = torch.where(mag > on_t, torch.ones_like(self._v2_eng),
                                   torch.where(mag < off_t, torch.zeros_like(self._v2_eng), self._v2_eng))
        eng = self._v2_eng
        dmag = mag * eng                                                # debounced magnitudes
        sup = dmag.sum(-1)                                              # total fingertip support (N,)
        lam = dmag / sup.clamp_min(_EPS).unsqueeze(-1)                  # load shares (N,5)
        if hasattr(self, "_gx_settle"):
            active = self._gx_settle == 0
        else:
            active = torch.ones(N, dtype=torch.bool, device=dev)
        alive = active & (~self._v2_sink_now)
        # z / clearance
        z = self.object_pos[:, 2]
        clear = (z >= self._v2_zstart - float(cfg.rgv2_clear_margin)) & self._v2_started
        # wrench balance: sum F + m g + F_ext(applied perturb force) ~ 0
        m = self.object.root_physx_view.get_masses().reshape(N).to(dev)
        g = self.physics_sim_view.get_gravity()
        gvec = torch.tensor([g[0], g[1], g[2]], device=dev).view(1, 3)
        gmag = float(max((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5, 1e-3))
        F_ext = self.rb_forces if hasattr(self, "rb_forces") else torch.zeros(N, 3, device=dev)
        Fsum = F.sum(1)                                                  # (N,3)
        resid = (Fsum + m.unsqueeze(-1) * gvec + F_ext).norm(dim=-1) / (m * gmag).clamp_min(_EPS)
        balanced = (resid < float(cfg.rgv2_balance_tol)) & (sup > 0.05)
        # cumulative ledger (active, palm-clear only — sunk states must not accrue support credit)
        upd = (active & clear).float().unsqueeze(-1)
        self._v2_L = self._v2_L + dmag * upd
        self._v2_act_steps = self._v2_act_steps + (active & clear).float()
        Lsum = self._v2_L.sum(-1)
        share = self._v2_L / Lsum.clamp_min(_EPS).unsqueeze(-1)          # cumulative load shares (N,5)
        min_share = torch.where(Lsum > _EPS, share.min(dim=-1).values, torch.zeros(N, device=dev))
        # ---------------- handover state machine (per finger: 0=idle, 1=armed, 2=verify) ----------------
        # transfer: debounced release while total support holds >=80% of its value ~5 steps ago
        # armed -> verify: re-contact after >=5 off steps, fingertip displaced >= 8 mm from release
        # verify: 10 consecutive clear+balanced+engaged steps -> COMPLETED; any violation -> idle
        k = self._v2_hist_ptr % 6
        sup_prev = self._v2_sup_hist[:, (k + 1) % 6].clone()             # ~5 steps ago
        self._v2_sup_hist[:, k] = sup
        self._v2_hist_ptr += 1
        off_prev = self._v2_off_cnt.clone()
        off_new = torch.where(eng > 0.5, torch.zeros_like(self._v2_off_cnt), self._v2_off_cnt + 1.0)
        self._v2_off_cnt = torch.where(active.unsqueeze(-1), off_new, self._v2_off_cnt)
        just_released = (off_prev == 0.0) & (self._v2_off_cnt == 1.0)
        just_recontacted = (off_prev > 0.0) & (eng > 0.5)
        support_held = ((sup > 0.8 * sup_prev) & (sup_prev > 0.05)).unsqueeze(-1)
        gate_nc = (clear & active).unsqueeze(-1)
        st = self._v2_ho_state
        # idle -> armed (genuine load transfer)
        arm = (st == 0.0) & just_released & support_held & gate_nc
        if bool(arm.any()):
            self._v2_rel_pos = torch.where(arm.unsqueeze(-1), self.fingertip_pos, self._v2_rel_pos)
            self._v2_transfer_events = self._v2_transfer_events + arm.float().sum(-1)
        st = torch.where(arm, torch.ones_like(st), st)
        # armed -> verify (qualified re-contact) or -> idle (short-hop re-contact)
        rec = (st == 1.0) & just_recontacted
        if bool(rec.any()):
            disp = (self.fingertip_pos - self._v2_rel_pos).norm(dim=-1)
            good = rec & (off_prev >= 5.0) & (disp >= 0.008)
            st = torch.where(good, torch.full_like(st, 2.0), torch.where(rec, torch.zeros_like(st), st))
            self._v2_window_cnt = torch.where(good, torch.full_like(self._v2_window_cnt, 10.0),
                                              self._v2_window_cnt)
        # verify countdown
        ver = st == 2.0
        # HOTFIX 2026-07-02: 'balanced' removed from the verify gate — fingertip-only sensing cannot
        # balance gravity when mid-phalanx contacts carry load (resid median ~1.85 observed); the
        # residual stays as telemetry only.
        w_ok = (clear & active).unsqueeze(-1) & (eng > 0.5)
        self._v2_window_cnt = torch.where(ver & w_ok, self._v2_window_cnt - 1.0, self._v2_window_cnt)
        completed_now = (ver & w_ok & (self._v2_window_cnt <= 0.0)).float()
        aborted = ver & (~w_ok)
        st = torch.where((completed_now > 0.5) | aborted, torch.zeros_like(st), st)
        self._v2_window_cnt = torch.where(st != 2.0, torch.zeros_like(self._v2_window_cnt),
                                          self._v2_window_cnt)
        self._v2_ho_state = st
        self._v2_handover_done = self._v2_handover_done + completed_now

        out = dict(F=F, mag=mag, eng=eng, sup=sup, lam=lam, active=active, clear=clear,
                   balanced=balanced, resid=resid, z=z, share=share, min_share=min_share,
                   Lsum=Lsum, m=m, gvec=gvec, gmag=gmag, Fsum=Fsum, F_ext=F_ext,
                   completed_handover=completed_now)
        self._v2_step_stamp = step
        self._v2_cache = out
        return out

    # ---------------- termination + curriculum ----------------
    def _get_dones(self):
        reset, timeout = super()._get_dones()
        if not hasattr(self, "_v2_below_cnt"):
            return reset, timeout
        cfg = self.cfg
        z = self.object_pos[:, 2]
        if hasattr(self, "_gx_settle"):
            active = self._gx_settle == 0
        else:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        newly = active & (~self._v2_started)
        if bool(newly.any()):
            self._v2_zstart[newly] = z[newly]
            self._v2_started |= active
        below = self._v2_started & (z < self._v2_zstart - float(cfg.rgv2_drop_below))
        # ESCAPE-TOSS guard (closed-loop 2026-07-03, default OFF): the sink rule only bounds downward
        # escape; a policy learned to keep the ball AIRBORNE (dz +0.29 m) farming latched clear income
        # with zero contact. rgv2_rise_above (m) terminates sustained upward escape symmetrically.
        rise_above = getattr(cfg, "rgv2_rise_above", None)
        if rise_above is not None:
            # gravity-gated (2026-07-03 PM): at near-zero g, contact impulses float the ball upward
            # NATURALLY — terminating that froze the curriculum at -0.05 (loop3_mixed, loop4 v1). The
            # toss exploit only pays under real gravity; guard activates once |g| >= rgv2_rise_min_g.
            g_ = self.physics_sim_view.get_gravity()
            gmag_ = (g_[0] ** 2 + g_[1] ** 2 + g_[2] ** 2) ** 0.5
            if gmag_ >= float(getattr(cfg, "rgv2_rise_min_g", 3.0)):
                below = below | (self._v2_started & (z > self._v2_zstart + float(rise_above)))
        self._v2_below_cnt = torch.where(below, self._v2_below_cnt + 1.0,
                                         torch.zeros_like(self._v2_below_cnt))
        sink = self._v2_below_cnt >= float(cfg.rgv2_drop_debounce)
        self._v2_sink_now = sink
        new_reset = reset | sink
        self._v2_fail_now = new_reset & (~timeout)
        self.extras["v2_sink_rate"] = sink.float().mean()
        # ---- gravity curriculum (owned here; base advancers are OFF via gravity_curriculum=False) ----
        fail_rate = self._v2_fail_now.float().mean()
        self._v2_term_ema = 0.995 * self._v2_term_ema + 0.005 * fail_rate
        self.extras["v2_fail_ema"] = self._v2_term_ema
        if bool(getattr(cfg, "rgv2_gravity_curriculum", False)) and self.common_step_counter > 1000 \
                and float(self._v2_term_ema) < float(cfg.rgv2_curr_term_thresh):
            import carb
            g = self.physics_sim_view.get_gravity()
            amp = (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5
            if amp < 10.0:
                self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -float(amp) - 0.05))
                print(f"[regripv2] gravity -> {-(float(amp) + 0.05):.2f} (fail_ema={float(self._v2_term_ema):.4f})")
        return new_reset, timeout

    # ---------------- terminal penalty + shared extras ----------------
    def _get_rewards(self):
        total = super()._get_rewards()
        if not hasattr(self, "_v2_below_cnt"):
            return total
        com = self._v2_common()
        # v3net critic priv (task cfgs with priv_info_dim=23): [0:8] legacy (live rel-pos, friction,
        # mass, COM — base env), [8:12] object quat, [12:15] linvel, [15:18] angvel, [18:23] debounced
        # per-finger contacts. Raw state only — no task-derived quantities (rotation-reusable).
        if self.priv_info_buf.shape[1] >= 23:
            rs = self.object.data.root_state_w
            self.priv_info_buf[:, 8:12] = rs[:, 3:7]
            self.priv_info_buf[:, 12:15] = rs[:, 7:10]
            self.priv_info_buf[:, 15:18] = rs[:, 10:13]
            self.priv_info_buf[:, 18:23] = self._v2_eng
        pen = float(self.cfg.rgv2_terminal_penalty)
        total = total - pen * self._v2_fail_now.float()
        self.extras["v2_clear_frac"] = com["clear"].float().mean()
        self.extras["v2_balanced_frac"] = com["balanced"].float().mean()
        self.extras["v2_min_share"] = com["min_share"].mean()
        self.extras["v2_resid_mean"] = com["resid"].clamp(0, 3).mean()
        self.extras["v2_sup_mean"] = com["sup"].mean()
        self.extras["v2_dz_mean"] = (com["z"] - self._v2_zstart).mean()
        self.extras["v2_handovers_mean"] = self._v2_handover_done.sum(-1).mean()
        self.extras["v2_transfers_mean"] = self._v2_transfer_events.mean()
        return total

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_v2_below_cnt"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._v2_zstart[ids] = 0.0
        self._v2_started[ids] = False
        self._v2_below_cnt[ids] = 0.0
        self._v2_sink_now[ids] = False
        self._v2_fail_now[ids] = False
        self._v2_eng[ids] = 0.0
        self._v2_L[ids] = 0.0
        self._v2_act_steps[ids] = 0.0
        self._v2_sup_hist[ids] = 0.0
        self._v2_ho_state[ids] = 0.0
        self._v2_off_cnt[ids] = 0.0
        self._v2_rel_pos[ids] = 0.0
        self._v2_window_cnt[ids] = 0.0
        self._v2_handover_done[ids] = 0.0
        self._v2_transfer_events[ids] = 0.0
        self._v2_step_stamp = -1
        self._v2_cache = {}
