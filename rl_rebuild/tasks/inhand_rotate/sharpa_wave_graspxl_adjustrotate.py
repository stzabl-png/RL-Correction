# Adjust-then-Rotate variant: minimal reward that adds a potential-based rotation-READINESS term so a
# single rotation objective can drive the policy to convert a non-rotation-ready (pinch) grasp into an
# enclosing grasp and THEN rotate -- no phase switch, no reference trajectory.
#
# Design + offline validation: robotics-rl-expert notes/experiment_playbooks/learn_adjust_then_rotate_program.md
#   ("Reward design -- minimal, validated offline"); the Phi formula + weights passed the offline reward
#   unit test (rl_rebuild/scripts/d0_reward_unit_test.py: Phi-sanity + ranks the target behavior the
#   unique argmax above squeeze/translate/one-time/do-nothing).
#
# This file is SELF-CONTAINED: a new cfg variant (no baseline cfg edits) + a new env subclass that
# overrides ONLY _get_rewards (calls super(), then adds the potential-based shaping). The pose-deviation
# penalty is removed via a config VALUE (pos_diff_penalty_scale = 0.0), not a code edit.
import torch
from isaaclab.utils import configclass

from .sharpa_wave_env_cfg import SharpaWaveGraspXLSustainedCfg
from .sharpa_wave_graspxl_env import SharpaWaveGraspXLEnv


@configclass
class SharpaWaveGraspXLAdjustRotateCfg(SharpaWaveGraspXLSustainedCfg):
    """Sustained-SO(3) config + rotation-readiness potential, minus the residual pose penalty.

    Changes vs SharpaWaveGraspXLSustainedCfg (everything else inherited: torque control, obs_hand_gravity,
    bounded centering, rotate 3.0 / angvel +-0.5, linvel -0.5, torque -0.1, work -0.5, loosened gravity/
    orient gates):
      - pos_diff_penalty_scale 0.0  -> DELETE the residual pose penalty so large finger excursions (regrasp/
        gaiting) are not suppressed (Sustained already lowered it to -0.1 'allow finger gaiting'; we go to 0).
      - add potential-based readiness shaping w_phi*(gamma*Phi(s') - Phi(s)), Ng-1999 optimum-preserving.
    Train FROM SCRATCH (obs dim 195, same as Sustained)."""
    pos_diff_penalty_scale = 0.0          # full freedom for adjustment; readiness comes from Phi, not pose-pinning
    use_readiness_potential = True
    w_phi = 3.0                            # readiness-potential gain (validated weight)
    phi_contact_thresh = 0.1              # N; per-finger contact counted as engaged (matches env 0.1 cutoff)
    phi_smooth = 0.5                      # EMA on Phi to damp runtime contact-sensor noise (0 = off)
    gamma_phi = 0.99                      # MUST match the agent discount for potential-based invariance


class SharpaWaveGraspXLAdjustRotateEnv(SharpaWaveGraspXLEnv):
    """GraspXL env + potential-based rotation-readiness shaping on top of the inherited reward."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Phi(s) of the previous step and the EMA-smoothed Phi, per env.
        self._phi_prev = torch.zeros(self.num_envs, device=self.device)
        self._phi_ema = torch.zeros(self.num_envs, device=self.device)

    def _readiness_phi(self) -> torch.Tensor:
        """Axis-conditioned rotation-readiness in [0,1], FORCE-MAGNITUDE-INDEPENDENT (squeeze can't game it):
        Phi = 0.5*spread + 0.5*coverage, computed exactly as the offline unit test.
          spread   = 1 - ||mean(engaged fingertip dirs projected _|_ rot_axis)||   (<2 engaged -> 0)
          coverage = clip((n_engaged - 1)/4, 0, 1)
        Engaged = per-finger contact force (last_contacts) > phi_contact_thresh."""
        thresh = float(getattr(self.cfg, "phi_contact_thresh", 0.1))
        center = self.object.data.root_pos_w.unsqueeze(1)                              # (N,1,3) world
        tips = self.hand.data.body_link_state_w[:, self.elastomer_ids, :3]            # (N,5,3) world
        rel = tips - center                                                           # (N,5,3)
        k = self.rot_axis
        k = k.view(1, 1, 3) if k.dim() == 1 else k.unsqueeze(1)                       # (N or 1,1,3)
        rel_perp = rel - (rel * k).sum(-1, keepdim=True) * k                          # project out axis
        d = rel_perp / rel_perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)            # (N,5,3) unit dirs
        eng = (self.last_contacts > thresh).float()                                   # (N,5)
        n = eng.sum(-1)                                                               # (N,)
        coverage = ((n - 1.0) / 4.0).clamp(0.0, 1.0)
        mean_d = (d * eng.unsqueeze(-1)).sum(1) / n.clamp_min(1.0).unsqueeze(-1)      # (N,3)
        resultant = mean_d.norm(dim=-1)                                               # (N,)
        spread = torch.where(n >= 2.0, 1.0 - resultant, torch.zeros_like(resultant))
        if getattr(self.cfg, "phi_mode", "linear") == "mult":
            # MULTIPLICATIVE: coverage GATES spread, so a 2-finger opposition grasp cannot score high
            # (n=2 -> coverage 0.25 -> Phi <= ~0.22). Forces the policy to recruit more fingers into
            # support instead of parking the spare fingers lifted (the 2-finger-cradle failure mode).
            return spread * coverage
        return 0.5 * spread + 0.5 * coverage

    def _get_rewards(self) -> torch.Tensor:
        total = super()._get_rewards()                                                # rotation + linvel + (pos_diff*0) + torque + work + centering
        if not getattr(self.cfg, "use_readiness_potential", False) or not hasattr(self, "_phi_prev"):
            return total
        phi_raw = self._readiness_phi()
        a = float(getattr(self.cfg, "phi_smooth", 0.5))
        phi = a * self._phi_ema + (1.0 - a) * phi_raw                                 # EMA smooth
        self._phi_ema = phi.detach()
        gamma = float(getattr(self.cfg, "gamma_phi", 0.99))
        w_phi = float(getattr(self.cfg, "w_phi", 3.0))
        shaping = w_phi * (gamma * phi - self._phi_prev)                              # F = w*(gamma*Phi(s') - Phi(s))
        self._phi_prev = phi.detach()
        self.extras["readiness_phi"] = phi.mean()
        self.extras["phi_shaping"] = shaping.mean()
        return total + shaping

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_phi_prev"):
            return                                                                    # called during base __init__, before buffers exist
        if env_ids is None:
            self._phi_prev[:] = 0.0
            self._phi_ema[:] = 0.0
        else:
            self._phi_prev[env_ids] = 0.0                                             # contacts are 0 right after reset -> Phi(reset)=0; avoids a spurious first-step spike
            self._phi_ema[env_ids] = 0.0
