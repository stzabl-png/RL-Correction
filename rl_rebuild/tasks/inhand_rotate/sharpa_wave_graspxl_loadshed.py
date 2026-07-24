# REGRIP sweep R-D1 "LOADSHED": forced-release per-finger actuation dropout. ADDITIVE ONLY.
#
# Design doc: robotics-rl-expert/notes/experiment_playbooks/regrip_contact_switch_designs_2026-07-01.md (D1).
# Mechanism: perturb-v1 proved DYNAMICS changes (not shaping) move the grasp optimum, but object-level
# wrenches are agnostic to WHICH fingers hold — the policy planted the same 1-2. LOADSHED periodically
# disables the ACTUATION of one currently-engaged finger (software-PD gains -> 0 for a short window), so
# keeping that finger planted is physically impossible: release + substitution by a DIFFERENT finger is
# the only survival strategy. NO new reward term (survival + the inherited hold/coverage/n_engaged terms
# carry the incentive).
#
# Skeptic-mandated construction (from the design review):
#   * Shed target = UNIFORM-RANDOM among currently-engaged, non-cooldown, eligibility-streak fingers
#     (NOT argmax-load: a deterministic rule invites a sacrificial decoy finger).
#   * Trigger jittered: Bernoulli(1/interval) per control step (mirrors the perturb impulse pattern).
#   * A finger is shed-eligible only after >= ls_eligible_streak consecutive engaged steps post-settle.
#   * Shed state is NOT observed (obs stays 195; the policy experiences dropout as a disturbance).
#     Documented limitation: preemptive redistribution is impossible without an obs; v2 follow-up.
#
# Implementation note: the hand runs SOFTWARE PD torque control (sharpa_wave_env.py:101-107, 250-254 —
# torques = p_gain*(cur_targets - q) - d_gain*qdot, applied via set_joint_effort_target), and
# self.p_gain/self.d_gain are live (N,22) tensors over the ACTUATED dof order. Shedding therefore is
# pure tensor masking of those gain rows — no sim-API gain writes needed. Nominal gains are snapshotted
# per env AFTER each reset (so PD-gain DR, when enabled, is respected) and restored when a shed window
# ends AND before any mid-shed env reset (so zeroed gains can never leak across episodes).
#
# Backbone: AdjustHold-Perturb unchanged (pinch reset, hold rewards, gravity-linked perturbation).
# Curriculum: shed interval 200->60 steps and window 10->20 steps, both driven by the SAME
# gravity-linked phase t = |g|/perturb_curr_g_full the perturbation magnitude uses.
# Self-collision arm via SHARPA_SELF_COLLISION. Eval probes as in sharpa_wave_graspxl_regrip_gait.py.
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjusthold_perturb import (
    SharpaWaveGraspXLAdjustHoldPerturbCfg,
    SharpaWaveGraspXLAdjustHoldPerturbEnv,
)
from .sharpa_wave_graspxl_regrip_gait import _RegripSelfCollisionPostInit

_FINGERS = ("thumb", "index", "middle", "ring", "pinky")   # == last_contacts column order


@configclass
class SharpaWaveGraspXLRegripD1LoadShedSphereCfg(_RegripSelfCollisionPostInit, SharpaWaveGraspXLAdjustHoldPerturbCfg):
    """R-D1 LOADSHED cfg: perturb backbone + per-finger actuation-dropout schedule."""
    ls_interval_start: float = 200.0   # expected control steps between sheds at curriculum phase t=0
    ls_interval_end: float = 60.0      # ...at t=1 (full gravity)
    ls_window_start: float = 10.0      # shed duration (control steps ~0.5 s) at t=0
    ls_window_end: float = 20.0        # ...at t=1
    ls_cooldown: float = 40.0          # per-finger cooldown after a shed (control steps)
    ls_eligible_streak: float = 10.0   # consecutive engaged steps before a finger is shed-eligible
    ls_contact_thresh: float = 0.1     # N; engagement threshold for streak/eligibility


class SharpaWaveGraspXLRegripD1LoadShedSphereEnv(SharpaWaveGraspXLAdjustHoldPerturbEnv):
    """Perturb regrip backbone + LOADSHED actuation dropout (see module header)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        N = self.num_envs
        dev = self.device
        n_f = self.last_contacts.shape[1]
        # per-finger -> actuated-dof-column boolean masks, by JOINT NAME (never contiguous slices)
        act_idx = self.actuated_dof_indices
        act_list = act_idx.tolist() if torch.is_tensor(act_idx) else list(act_idx)
        names = [self.hand.joint_names[i] for i in act_list]
        mask = torch.zeros(n_f, len(names), dtype=torch.bool, device=dev)
        for fi, key in enumerate(_FINGERS):
            mask[fi] = torch.tensor([key in n for n in names], dtype=torch.bool, device=dev)
            assert bool(mask[fi].any()), f"[loadshed] no actuated joints matched finger '{key}' in {names}"
        self._ls_joint_mask = mask                                  # (5, 22)
        print("[loadshed] finger->joint columns: " +
              "; ".join(f"{k}:{mask[i].nonzero().squeeze(-1).tolist()}" for i, k in enumerate(_FINGERS)))
        self._ls_nom_p = self.p_gain.clone()                        # nominal gains (refreshed per reset)
        self._ls_nom_d = self.d_gain.clone()
        self._ls_finger = torch.full((N,), -1, dtype=torch.long, device=dev)  # currently shed finger
        self._ls_timer = torch.zeros(N, device=dev)                 # steps left in the shed window
        self._ls_cd = torch.zeros((N, n_f), device=dev)             # per-finger cooldown
        self._ls_streak = torch.zeros((N, n_f), device=dev)         # consecutive engaged steps
        self._ls_shed_count = torch.zeros(N, device=dev)            # sheds this episode (diagnostic)

    # ------------------------------------------------------------------ dropout schedule
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)                          # perturb wrench + base targets
        if not hasattr(self, "_ls_timer"):
            return
        cfg = self.cfg
        N = self.num_envs
        dev = self.device
        if hasattr(self, "_gx_settle"):
            active = self._gx_settle == 0
        else:
            active = torch.ones(N, dtype=torch.bool, device=dev)
        actf = active.float().unsqueeze(-1)

        # curriculum phase from gravity magnitude (same convention as the perturb magnitude)
        g = self.physics_sim_view.get_gravity()
        g_mag = float((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5)
        t = min(g_mag / max(float(getattr(cfg, "perturb_curr_g_full", 10.0)), 1e-6), 1.0)
        interval = float(cfg.ls_interval_start) + (float(cfg.ls_interval_end) - float(cfg.ls_interval_start)) * t
        window = float(cfg.ls_window_start) + (float(cfg.ls_window_end) - float(cfg.ls_window_start)) * t
        self._ls_t_phase, self._ls_interval_cur, self._ls_window_cur = t, interval, window

        # engagement streaks (active-gated) + cooldown decrement
        eng = (self.last_contacts > float(cfg.ls_contact_thresh)).float()
        streak_new = torch.where(eng > 0.5, self._ls_streak + 1.0, torch.zeros_like(self._ls_streak))
        self._ls_streak = actf * streak_new + (1.0 - actf) * self._ls_streak
        self._ls_cd = (self._ls_cd - actf).clamp_min(0.0)

        # advance shed timers; restore gains when a window ends
        shedding = self._ls_timer > 0.0
        self._ls_timer = (self._ls_timer - (active & shedding).float()).clamp_min(0.0)
        ended = shedding & (self._ls_timer <= 0.0)
        if bool(ended.any()):
            ids = ended.nonzero(as_tuple=False).squeeze(-1)
            fm = self._ls_joint_mask[self._ls_finger[ids].clamp_min(0)]         # (K,22)
            self.p_gain[ids] = torch.where(fm, self._ls_nom_p[ids], self.p_gain[ids])
            self.d_gain[ids] = torch.where(fm, self._ls_nom_d[ids], self.d_gain[ids])
            self._ls_cd[ids, self._ls_finger[ids].clamp_min(0)] = float(cfg.ls_cooldown)
            self._ls_finger[ids] = -1

        # trigger new sheds: Bernoulli(1/interval), active, not already shedding, >=1 eligible finger
        eligible = (eng > 0.5) & (self._ls_streak >= float(cfg.ls_eligible_streak)) & (self._ls_cd <= 0.0)
        trig = active & (~(self._ls_timer > 0.0)) & \
               (torch.rand(N, device=dev) < (1.0 / max(interval, 1.0)))
        trig = trig & eligible.any(dim=-1)
        if bool(trig.any()):
            r = torch.rand(N, self.last_contacts.shape[1], device=dev) * eligible.float()
            finger = r.argmax(dim=-1)                                            # uniform among eligible
            ids = trig.nonzero(as_tuple=False).squeeze(-1)
            self._ls_finger[ids] = finger[ids]
            self._ls_timer[ids] = window
            self._ls_shed_count[ids] += 1.0
            fm = self._ls_joint_mask[finger[ids]]                                # (K,22)
            self.p_gain[ids] = self.p_gain[ids] * (~fm).float()
            self.d_gain[ids] = self.d_gain[ids] * (~fm).float()

    # ------------------------------------------------------------------ diagnostics
    def _get_rewards(self) -> torch.Tensor:
        total = super()._get_rewards()
        if hasattr(self, "_ls_timer"):
            self.extras["ls_shedding_frac"] = (self._ls_timer > 0.0).float().mean()
            self.extras["ls_sheds_per_ep_mean"] = self._ls_shed_count.mean()
            self.extras["ls_interval"] = torch.tensor(getattr(self, "_ls_interval_cur", 0.0), device=self.device)
            self.extras["ls_window"] = torch.tensor(getattr(self, "_ls_window_cur", 0.0), device=self.device)
        return total

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # restore any in-flight shed BEFORE the base reset so zeroed gains never leak across episodes
        if hasattr(self, "_ls_timer"):
            ids = self.hand._ALL_INDICES if env_ids is None else env_ids
            ids_t = ids if torch.is_tensor(ids) else torch.as_tensor(ids, device=self.device, dtype=torch.long)
            mid = ids_t[self._ls_timer[ids_t] > 0.0]
            if mid.numel() > 0:
                fm = self._ls_joint_mask[self._ls_finger[mid].clamp_min(0)]
                self.p_gain[mid] = torch.where(fm, self._ls_nom_p[mid], self.p_gain[mid])
                self.d_gain[mid] = torch.where(fm, self._ls_nom_d[mid], self.d_gain[mid])
        super()._reset_idx(env_ids)                                  # may re-randomize p_gain/d_gain (DR)
        if not hasattr(self, "_ls_timer"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        self._ls_nom_p[ids] = self.p_gain[ids]                       # snapshot this episode's nominal
        self._ls_nom_d[ids] = self.d_gain[ids]
        self._ls_finger[ids] = -1
        self._ls_timer[ids] = 0.0
        self._ls_cd[ids] = 0.0
        self._ls_streak[ids] = 0.0
        self._ls_shed_count[ids] = 0.0
