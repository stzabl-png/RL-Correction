# VARIANT: adjusthold-perturb-v1 -- REGRIP-HOLD via an ADVERSARIAL PERTURBATION curriculum.
#
# Root diagnosis (why all prior regrip variants failed):
#   v1 (linear PBRS)  -> 2-finger opposition cradle (the true cheapest optimum of survive+center+energy).
#   v2 (mult PBRS)    -> collapsed to open-palm-rest + drops.
#   v3a (persistent hold+coverage, this file's PARENT) -> still tends toward a PALM-SUPPORTED hold of the
#       smooth ball, because resting the ball on the palm is the low-effort optimum. Shaping rewards
#       (PBRS or persistent bonuses) cannot by themselves force a fingertip grasp: the palm-rest survives.
#
# FIX (change the MDP DYNAMICS, not the shaping function):
#   Apply RANDOM EXTERNAL perturbation forces (+ small torques) to the object every physics step, with a
#   magnitude on a CURRICULUM that GROWS WITH GRAVITY (weak at g~0, strong at full g). A weak palm-rest or
#   2-finger cradle is knocked off in a few steps; only a secure multi-finger fingertip grasp survives a
#   full episode. SURVIVAL (episode length) -- not a shaping hack -- then demands the grasp. A bounded
#   per-step + per-episode n_engaged bonus provides an exploration gradient toward fingertip contact
#   BEFORE the perturbation is strong enough to discriminate grasps. This is Ng-1999-safe: perturbation is
#   a dynamics change (not a potential), so it CAN move the optimum; the n_engaged bonus is bounded and
#   gated so it cannot be farmed by flailing.
#
# ADDITIVE ONLY: a new @configclass (subclass of SharpaWaveGraspXLAdjustHoldPersistentCfg -> inherits the
# working NON-PBRS hold+coverage reward, rotate_reward_scale=0, graspxl_orient_rotate_gate=-1, drop-gated
# gravity curriculum, pinch/replay reset) + a new env subclass that overrides ONLY _pre_physics_step,
# _get_rewards, __init__, _reset_idx. No baseline cfg/env is edited.
#
# VET FIXES APPLIED (mandatory, from regrip_variant_specs.md :: adjusthold-perturb-v1):
#   (1) All new buffers (self._perturb_torques, episode accumulators) are created AFTER super().__init__();
#       _reset_idx uses a hasattr guard for the early DirectRLEnv.__init__-time reset call.
#   (2) Perturbation application AND buffer zeroing are gated with ~(self._gx_settle > 0): during the
#       GraspXL replay/settle window NO perturbation is applied and the wrench buffers are held at zero.
#   (3) perturb_force_base = 0.5 (NOT 2.0) for the first run.
#   (4) A per-step AND a per-episode n_engaged reward pay for engaged fingertips (on top of the inherited
#       hold/centering/coverage) so the surviving behavior is a genuine fingertip grasp.
#
# GROUNDING (real IsaacLab API, cited file:line in the code below):
#   - self.object.set_external_force_and_torque(forces=(N,1,3), torques=(N,1,3))  sharpa_wave_env.py:248
#   - self.rb_forces (N,3) buffer                                                  sharpa_wave_env.py:66
#   - force decay pattern (0.9 ** (physics_dt / interval))                         sharpa_wave_env.py:242
#   - obj mass: self.object.root_physx_view.get_masses().reshape(N).to(device)     sharpa_wave_env.py:244
#   - self._gx_settle (settle countdown), settling = _gx_settle > 0                sharpa_wave_graspxl_env.py:131,200
#   - self.last_contacts (N,5) per-finger contact force                            sharpa_wave_env.py:128
#   - self.physics_sim_view.get_gravity()[0..2]                                    sharpa_wave_env.py:310-312
#   - reset_terminated/reset_time_outs set BEFORE _get_rewards                     direct_rl_env.py:391-393
#
# Train FROM SCRATCH (obs dim 195, same architecture as all GraspXL variants).
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjusthold import (
    SharpaWaveGraspXLAdjustHoldPersistentCfg,
    SharpaWaveGraspXLAdjustHoldPersistentEnv,
)


@configclass
class SharpaWaveGraspXLAdjustHoldPerturbCfg(SharpaWaveGraspXLAdjustHoldPersistentCfg):
    """adjusthold-perturb-v1 config.

    Inherits the WORKING non-PBRS hold+coverage reward from SharpaWaveGraspXLAdjustHoldPersistentCfg
    (r_hold=1.0, r_cov<=0.5 gated on held, centering exp(-disp/0.03), linvel -0.5, torque -0.1, work -0.5),
    plus rotate_reward_scale=0.0, use_readiness_potential=False (PBRS off), graspxl_orient_rotate_gate=-1.0
    (gravity curriculum gated on DROP RATE only), graspxl_init="replay" (pinch reset), and the adaptive
    gravity curriculum. Adds an adversarial perturbation curriculum + an n_engaged fingertip bonus.

    Train FROM SCRATCH (obs dim 195).
    """

    # ---- Disable the BASE random-force mechanism (SharpaWaveInhandRotateEnv._pre_physics_step,
    #      sharpa_wave_env.py:241). Our env applies its own gravity-linked perturbation instead, so the
    #      base wrench code must be fully skipped. (Overrides inherited force_scale=0.5.) ----
    force_scale = 0.0
    # Belt-and-suspenders: ensure the base force block is skipped even if the force_scale guard changed.
    random_force_prob_scalar = 0.0

    # ---- PBRS stays OFF (inherited from the persistent parent; re-assert for clarity). ----
    use_readiness_potential = False

    # ---- Perturbation MAGNITUDE curriculum (grows with gravity magnitude). ----------------------------
    # Scales are per-kg of object mass (F = randn * obj_mass * f_scale), matching the base random-force
    # convention (sharpa_wave_env.py:247). t = clip(|g| / perturb_curr_g_full, 0, 1);
    #   f_scale = force_base  + (force_max  - force_base ) * t   in [force_base,  force_max ]
    #   t_scale = torque_base + (torque_max - torque_base) * t   in [torque_base, torque_max]
    # BOUNDED by construction: t in [0,1] -> scales stay within [base, max].
    perturb_force_base: float = 0.5     # N/kg at |g|~0. VET FIX 3: 0.5 (NOT 2.0) for the first run.
    perturb_force_max: float = 10.0     # N/kg at full gravity. At obj_mass~0.05 kg -> F_rms ~ 0.40 N.
    perturb_torque_base: float = 0.02   # N*m/kg at |g|~0.
    perturb_torque_max: float = 0.10    # N*m/kg at full gravity.
    perturb_curr_g_full: float = 10.0   # |g| (m/s^2) at which the perturbation reaches its max scale.

    # ---- Perturbation impulse dynamics (mirror the base random-force pattern, sharpa_wave_env.py:242-247).
    perturb_prob: float = 0.25          # prob of resampling a new perturbation impulse each physics step.
    perturb_decay: float = 0.9          # exp decay of the wrench each physics step: 0.9 ** (dt / interval).
    perturb_decay_interval: float = 0.08

    # ---- n_engaged fingertip bonus (per-step + per-episode). Both bounded. ----------------------------
    # Provides an exploration gradient toward fingertip contact BEFORE perturbation is strong enough to
    # discriminate grasps. On TOP of the inherited hold/coverage/centering.
    n_engaged_thresh: float = 0.1       # N; min last_contacts to count a finger engaged (== phi_contact_thresh).
    # Per-step: r = scale * clip(n_engaged / 5, 0, 1)  in [0, scale].
    n_engaged_reward_scale: float = 1.5
    # Per-episode: paid ONCE at episode termination = scale * mean_over_episode(clip(n_engaged/5,0,1)),
    # in [0, scale]. Rewards SUSTAINED multi-finger engagement over the whole trajectory. Cannot be gamed
    # by early termination: a dropper accrues a LOW episode-mean engagement (and a short episode).
    n_engaged_episode_scale: float = 5.0


class SharpaWaveGraspXLAdjustHoldPerturbEnv(SharpaWaveGraspXLAdjustHoldPersistentEnv):
    """AdjustHold-Persistent env + adversarial perturbation curriculum + n_engaged fingertip bonus.

    MRO: SharpaWaveGraspXLAdjustHoldPerturbEnv -> SharpaWaveGraspXLAdjustHoldPersistentEnv
         -> SharpaWaveGraspXLAdjustRotateEnv -> SharpaWaveGraspXLEnv -> SharpaWaveInhandRotateEnv.

    _pre_physics_step chain: this override calls super()._pre_physics_step(actions), which reaches
    SharpaWaveGraspXLEnv._pre_physics_step (sharpa_wave_graspxl_env.py:196 -> replay settle targets)
    then SharpaWaveInhandRotateEnv._pre_physics_step (sharpa_wave_env.py:229; the base random-force block
    at :241 is SKIPPED because force_scale=0.0). We then apply our own gravity-linked wrench.

    _get_rewards chain: super() -> AdjustHoldPersistentEnv (r_hold + r_cov, sharpa_wave_graspxl_adjusthold.py)
    -> AdjustRotateEnv (returns early: use_readiness_potential=False -> no PBRS,
    sharpa_wave_graspxl_adjustrotate.py:77) -> base compute_rewards (rotate*0 + centering + linvel + torque
    + work). We ADD the per-step + per-episode n_engaged bonus.
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        # VET FIX 1: super().__init__ FIRST (allocates num_envs / device / object / rb_forces / _gx_settle /
        # last_contacts). New buffers are created AFTER, so the framework's initial _reset_idx(None) call
        # (during DirectRLEnv.__init__) is handled by the hasattr guard in _reset_idx below.
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # New torque wrench buffer (self.rb_forces (N,3) already exists, sharpa_wave_env.py:66; reused for
        # translational perturbation). Only torques need a new buffer.
        self._perturb_torques = torch.zeros(self.num_envs, 3, device=self.device)

        # Per-episode engagement accumulators (mean engaged-finger fraction over the running episode).
        self._ep_engaged_sum = torch.zeros(self.num_envs, device=self.device)
        self._ep_engaged_count = torch.zeros(self.num_envs, device=self.device)

        # Diagnostics stash (set in _pre_physics_step, logged in _get_rewards).
        self._last_perturb_f_scale = 0.0
        self._last_perturb_t_scale = 0.0
        self._last_perturb_g_mag = 0.0

    # ---------------------------------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Base chain: replay settle targets (GraspXL) + action targets; base random-force SKIPPED (fs=0).
        super()._pre_physics_step(actions)

        # VET FIX 2: gate on the GraspXL settle window. During replay/settle NO perturbation is applied and
        # the wrench buffers are held at zero. (_gx_settle exists in replay mode; guard defensively.)
        if hasattr(self, "_gx_settle"):
            settling = self._gx_settle > 0
        else:
            settling = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- MAGNITUDE curriculum from the current gravity magnitude (sharpa_wave_env.py:310-312 read g). ---
        g = self.physics_sim_view.get_gravity()                      # carb.Float3
        g_mag = float((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5)    # ~0.05 (start) -> 10.0 (full)
        g_full = float(getattr(self.cfg, "perturb_curr_g_full", 10.0))
        t = min(g_mag / max(g_full, 1e-6), 1.0)                      # curriculum phase in [0,1]
        f_base = float(self.cfg.perturb_force_base)
        f_max = float(self.cfg.perturb_force_max)
        tq_base = float(self.cfg.perturb_torque_base)
        tq_max = float(self.cfg.perturb_torque_max)
        f_scale = f_base + (f_max - f_base) * t                      # in [f_base,  f_max ]
        t_scale = tq_base + (tq_max - tq_base) * t                   # in [tq_base, tq_max]
        self._last_perturb_f_scale = f_scale
        self._last_perturb_t_scale = t_scale
        self._last_perturb_g_mag = g_mag

        # --- decay existing wrench (mirror base pattern, sharpa_wave_env.py:242) ---
        decay = float(self.cfg.perturb_decay) ** (self.physics_dt / float(self.cfg.perturb_decay_interval))
        self.rb_forces *= decay
        self._perturb_torques *= decay

        # --- resample a new impulse with prob perturb_prob, ONLY on non-settling envs (VET FIX 2) ---
        hit = (torch.rand(self.num_envs, device=self.device) < float(self.cfg.perturb_prob)) & (~settling)
        hit_ids = hit.nonzero(as_tuple=False).squeeze(-1)
        if hit_ids.numel() > 0:
            # obj mass per env (sharpa_wave_env.py:244 pattern)
            obj_mass = self.object.root_physx_view.get_masses().reshape(self.num_envs).to(self.device)
            m = obj_mass[hit_ids, None]                              # (K,1)
            self.rb_forces[hit_ids] = torch.randn(hit_ids.numel(), 3, device=self.device) * m * f_scale
            self._perturb_torques[hit_ids] = torch.randn(hit_ids.numel(), 3, device=self.device) * m * t_scale

        # VET FIX 2: hard-zero the wrench on settling envs (extra guard against any residual/decayed force).
        if bool(settling.any()):
            self.rb_forces[settling] = 0.0
            self._perturb_torques[settling] = 0.0

        # --- apply the external wrench to the object rigid body (sharpa_wave_env.py:248 API) ---
        self.object.set_external_force_and_torque(
            forces=self.rb_forces.reshape(self.num_envs, 1, 3),
            torques=self._perturb_torques.reshape(self.num_envs, 1, 3),
        )

    # ---------------------------------------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # super(): AdjustHoldPersistent hold+coverage + base centering/linvel/torque/work (PBRS off).
        total = super()._get_rewards()

        thresh = float(getattr(self.cfg, "n_engaged_thresh", 0.1))
        # last_contacts (N,5) per-finger contact force (sharpa_wave_env.py:128, updated :466-468).
        n_eng = (self.last_contacts > thresh).float().sum(-1)        # (N,)
        eng_frac = (n_eng / 5.0).clamp(0.0, 1.0)                     # (N,) in [0,1]

        # --- per-step bonus (bounded [0, n_engaged_reward_scale]) ---
        eng_r = eng_frac * float(getattr(self.cfg, "n_engaged_reward_scale", 1.5))

        # --- per-episode bonus: accumulate the engaged-finger fraction, pay ONCE at episode termination. ---
        # reset_terminated / reset_time_outs are already set for THIS step (direct_rl_env.py:391-393),
        # before _get_rewards is called (:393). done envs are reset afterwards (:398) -> _reset_idx zeros
        # the accumulators. Guard in case the buffers do not yet exist (paranoia; __init__ creates them).
        ep_bonus = torch.zeros_like(eng_r)
        if hasattr(self, "_ep_engaged_sum"):
            self._ep_engaged_sum += eng_frac
            self._ep_engaged_count += 1.0
            done = (self.reset_terminated | self.reset_time_outs).float()          # (N,)
            ep_mean = self._ep_engaged_sum / self._ep_engaged_count.clamp_min(1.0)  # (N,) in [0,1]
            ep_bonus = ep_mean * float(getattr(self.cfg, "n_engaged_episode_scale", 5.0)) * done

        # --- diagnostics ---
        self.extras["n_engaged"] = n_eng.mean()
        self.extras["eng_reward"] = eng_r.mean()
        self.extras["ep_engaged_bonus"] = ep_bonus.sum() / self.num_envs
        self.extras["perturb_f_scale"] = torch.tensor(self._last_perturb_f_scale, device=self.device)
        self.extras["perturb_t_scale"] = torch.tensor(self._last_perturb_t_scale, device=self.device)
        self.extras["perturb_g_mag"] = torch.tensor(self._last_perturb_g_mag, device=self.device)

        return total + eng_r + ep_bonus

    # ---------------------------------------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # super() chain resets phi buffers (inert), and zeros rb_forces[env_ids] (sharpa_wave_env.py:390).
        super()._reset_idx(env_ids)
        # VET FIX 1: hasattr guard for the DirectRLEnv.__init__-time reset (before our buffers exist).
        if not hasattr(self, "_perturb_torques"):
            return
        if env_ids is None:
            self._perturb_torques[:] = 0.0
            self._ep_engaged_sum[:] = 0.0
            self._ep_engaged_count[:] = 0.0
        else:
            self._perturb_torques[env_ids] = 0.0
            self._ep_engaged_sum[env_ids] = 0.0
            self._ep_engaged_count[env_ids] = 0.0
