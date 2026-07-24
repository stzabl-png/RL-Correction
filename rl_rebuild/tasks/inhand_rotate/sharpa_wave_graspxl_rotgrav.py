# REGRIP sweep R-D4 "ROTGRAV": rotating low-point sustained lateral load. ADDITIVE ONLY.
#
# Design doc: robotics-rl-expert/notes/experiment_playbooks/regrip_contact_switch_designs_2026-07-01.md (D4).
# Mechanism: perturb-v1's stochastic impulses are absorbable by reflexive stiffening of a fixed contact
# set. A SUSTAINED lateral force whose azimuth slowly cycles moves the ball's mechanical low point AROUND
# the hand, so any fixed contact set that resists straight-down gravity becomes progressively insufficient
# within one episode — load-bearing must migrate. NO new reward term.
#
# Construction:
#   * REPLACES the stochastic wrench: cfg.perturb_prob = 0.0 disables the parent's impulse resampling
#     (its decay-only block still runs, harmlessly, on zeroed buffers); this env then overwrites
#     rb_forces with F = m_obj * g_extra(t) * [cos(phi_i), sin(phi_i), 0] and re-applies the wrench
#     (the second set_external_force_and_torque call wins at write_data_to_sim).
#   * phi_i(t) = phi_i0 + 2*pi*omega*t_step, phi_i0 ~ U(0, 2pi) per env at reset (azimuth diversity
#     despite the shared omega); phase advances once per CONTROL step by 2*pi*omega*step_dt.
#   * Curricula (both driven by the gravity-linked phase t = |g|/perturb_curr_g_full, but logged as
#     INDEPENDENT extras so their effects can be disentangled post-hoc):
#       omega:   rv_omega_start (1 rev / 20 s) -> rv_omega_end (1 rev / 4 s)   [rev/s]
#       g_extra: rv_g_base 0.5 -> rv_g_max 10.0                               [N/kg of object mass]
#   * VET FIX 2 mirrored: wrench hard-zeroed on settling envs (no load during the scripted pinch replay).
#   * rv_omega_zero=True turns the rotation OFF (fixed off-axis load at the same magnitude) — the
#     control arm that separates "rotation of the load" from "off-vertical loading" (skeptic fix).
#
# Backbone: AdjustHold-Perturb rewards unchanged. Obs unchanged (195). Self-collision arm via
# SHARPA_SELF_COLLISION. Eval probes as in sharpa_wave_graspxl_regrip_gait.py.
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjusthold_perturb import (
    SharpaWaveGraspXLAdjustHoldPerturbCfg,
    SharpaWaveGraspXLAdjustHoldPerturbEnv,
)
from .sharpa_wave_graspxl_regrip_gait import _RegripSelfCollisionPostInit


@configclass
class SharpaWaveGraspXLRegripD4RotGravSphereCfg(_RegripSelfCollisionPostInit, SharpaWaveGraspXLAdjustHoldPerturbCfg):
    """R-D4 ROTGRAV cfg: perturb backbone with the stochastic wrench replaced by a rotating lateral load."""
    perturb_prob = 0.0                 # disable the parent's stochastic impulse resampling
    rv_omega_start: float = 0.05       # rev/s at curriculum phase t=0  (1 revolution / 20 s)
    rv_omega_end: float = 0.25         # rev/s at t=1                   (1 revolution / 4 s)
    rv_g_base: float = 0.5             # N/kg lateral load at t=0
    rv_g_max: float = 10.0             # N/kg at t=1 (matches perturb_force_max magnitude convention)
    rv_omega_zero: bool = False        # True = fixed off-axis load control arm (no rotation of the load)


class SharpaWaveGraspXLRegripD4RotGravSphereEnv(SharpaWaveGraspXLAdjustHoldPerturbEnv):
    """Perturb regrip backbone + rotating low-point lateral load (see module header)."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._rv_phi = torch.rand(self.num_envs, device=self.device) * 2.0 * math.pi

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # parent chain: targets + (prob=0 -> decay-only) stochastic block + its wrench application
        super()._pre_physics_step(actions)
        if not hasattr(self, "_rv_phi"):
            return
        cfg = self.cfg
        if hasattr(self, "_gx_settle"):
            settling = self._gx_settle > 0
        else:
            settling = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # gravity-linked curriculum phase (same convention as the perturb magnitude)
        g = self.physics_sim_view.get_gravity()
        g_mag = float((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5)
        t = min(g_mag / max(float(getattr(cfg, "perturb_curr_g_full", 10.0)), 1e-6), 1.0)
        omega = float(cfg.rv_omega_start) + (float(cfg.rv_omega_end) - float(cfg.rv_omega_start)) * t
        if bool(getattr(cfg, "rv_omega_zero", False)):
            omega = 0.0
        g_extra = float(cfg.rv_g_base) + (float(cfg.rv_g_max) - float(cfg.rv_g_base)) * t

        # advance the per-env azimuth once per control step
        self._rv_phi = torch.remainder(self._rv_phi + 2.0 * math.pi * omega * self.step_dt, 2.0 * math.pi)

        # sustained lateral force, overwriting the (decayed/zero) stochastic wrench
        obj_mass = self.object.root_physx_view.get_masses().reshape(self.num_envs).to(self.device)
        fmag = obj_mass * g_extra                                            # (N,)
        self.rb_forces[:, 0] = fmag * torch.cos(self._rv_phi)
        self.rb_forces[:, 1] = fmag * torch.sin(self._rv_phi)
        self.rb_forces[:, 2] = 0.0
        self._perturb_torques[:] = 0.0
        if bool(settling.any()):                                             # VET FIX 2 mirrored
            self.rb_forces[settling] = 0.0
        self.object.set_external_force_and_torque(
            forces=self.rb_forces.reshape(self.num_envs, 1, 3),
            torques=self._perturb_torques.reshape(self.num_envs, 1, 3),
        )
        self._rv_omega_cur, self._rv_gextra_cur, self._rv_t_cur = omega, g_extra, t

    def _get_rewards(self) -> torch.Tensor:
        total = super()._get_rewards()
        if hasattr(self, "_rv_phi"):
            self.extras["rv_omega"] = torch.tensor(getattr(self, "_rv_omega_cur", 0.0), device=self.device)
            self.extras["rv_gextra"] = torch.tensor(getattr(self, "_rv_gextra_cur", 0.0), device=self.device)
            self.extras["rv_curr_t"] = torch.tensor(getattr(self, "_rv_t_cur", 0.0), device=self.device)
        return total

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_rv_phi"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        ids_t = ids if torch.is_tensor(ids) else torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self._rv_phi[ids_t] = torch.rand(ids_t.numel(), device=self.device) * 2.0 * math.pi
