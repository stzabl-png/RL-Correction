# REGRIP sweep R-D2 "HANDOFF": maintain the harvested >=3-fingertip states. ADDITIVE ONLY.
#
# Design doc: robotics-rl-expert/notes/experiment_playbooks/regrip_contact_switch_designs_2026-07-01.md (D2).
# Stage 0 (done 2026-07-01, scripts/harvest_handoff_states.py): perturb-v1 reaches a sustained
# >=3-fingertip ball grasp in 1.88% of pinch episodes; 105 such states were harvested to
# cache/graspxl_handoff_002aa1853c974f3a9565e85f5e09a515.npy (29-vec rows, fingertip height z~0.6235).
# d0c verdict on the frozen poses: ~86% stay within the 0.10 m drop gate under zero action but SAG ~7 cm
# into the palm (fingertip contact does not survive passively). This env therefore asks the LEARNING
# question: can a policy actively STABILIZE these states (make the fingertip grasp the equilibrium)
# when every episode STARTS there? NO grace period — the policy must grip from step 0.
#
# Construction (pure composition; no new mechanism code):
#   * Reset = SharpaWaveGraspXLAdjustRotateDiverseEnv._write_enclosing_state on 100% of resets
#     (diverse_enclosing_frac=1.0), pointed at the HANDOFF cache instead of the enclosing cache.
#     That proven path writes object pose + joints, sets object_default_pose (drop/centering reference)
#     to the cached pose, zeroes rb_forces, and disables the replay settle (_gx_settle=0) so the policy
#     is in control immediately. MRO note: Perturb._reset_idx runs FIRST (zeroing _perturb_torques and
#     the episode accumulators for the full batch) and Diverse then overwrites the reset states, so no
#     perturbation-buffer leakage is possible.
#   * Rewards + perturbation dynamics + curricula = AdjustHold-Perturb, unchanged. With _gx_settle=0
#     from step 0, the perturbation is live immediately (weak at low gravity, per its curriculum).
#   * DR ranges unchanged for this sweep (comparability across the 16 runs); widened-DR is the v2 axis.
#
# Obs unchanged (195). Self-collision arm via SHARPA_SELF_COLLISION.
# Eval: contact_switch_probe / extract_adjusted_poses as usual. d0a_pinch_regrasp_probe still runs but
# its "migration" number is meaningless here (episodes do not start at a pinch — t=0 histogram will
# show n>=3 by construction); read its t=5s/t=10s histograms as RETENTION instead.
from __future__ import annotations

from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjusthold_perturb import (
    SharpaWaveGraspXLAdjustHoldPerturbCfg,
    SharpaWaveGraspXLAdjustHoldPerturbEnv,
)
from .sharpa_wave_graspxl_adjustrotate_diverse import SharpaWaveGraspXLAdjustRotateDiverseEnv
from .sharpa_wave_graspxl_regrip_gait import _RegripSelfCollisionPostInit

_GX_OBJ = "002aa1853c974f3a9565e85f5e09a515"


@configclass
class SharpaWaveGraspXLRegripD2HandoffSphereCfg(_RegripSelfCollisionPostInit, SharpaWaveGraspXLAdjustHoldPerturbCfg):
    """R-D2 HANDOFF cfg: perturb backbone; 100% of resets from the harvested handoff-state cache."""
    enclosing_cache_path = f"cache/graspxl_handoff_{_GX_OBJ}.npy"   # (105,29) harvested sustained >=3-finger states
    diverse_enclosing_frac = 1.0                                     # every reset uses the cache


class SharpaWaveGraspXLRegripD2HandoffSphereEnv(SharpaWaveGraspXLAdjustRotateDiverseEnv,
                                                SharpaWaveGraspXLAdjustHoldPerturbEnv):
    """Diverse cache-reset (100% handoff states) on the perturb regrip backbone.

    MRO: this -> Diverse -> Perturb -> AdjustHoldPersistent -> AdjustRotate -> GraspXL -> base.
    Diverse.__init__/_reset_idx call super() cooperatively, so the perturb machinery (wrench, episode
    accumulators, n_engaged bonus) is fully active; Diverse then overwrites reset states from the cache
    (and zeroes rb_forces for those envs; _perturb_torques was already zeroed by Perturb._reset_idx)."""
    pass
