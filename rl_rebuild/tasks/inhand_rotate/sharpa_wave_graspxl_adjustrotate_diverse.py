# Step 3B of the adjust-then-rotate program: DIVERSE-RESET adjust-rotate.
#
# 3A (pinch-only reset) was confounded: the adaptive gravity curriculum stalled at gravity_z ~= -0.40
# (~4% of full g) because a fingertip PINCH cannot hold as gravity rises, so the drop-rate gate never
# opened. 3B fixes that with TWO changes (the reward is unchanged -- still the validated readiness-Phi
# potential + pos_diff=0 inherited from AdjustRotate):
#   (1) FRICTION COMBINE = "max" on the GraspXL object (set in the cfg via make_graspxl_object_cfg's
#       physics_material arg) so object<->finger contact grips with max(mu).
#   (2) DIVERSE RESETS: ~50% of envs reset to the PINCH init (the existing replay behavior, via super())
#       and ~50% reset directly to a settled ENCLOSING grasp drawn from the validated enclosing cache.
#       The enclosing envs can hold as gravity rises, so the curriculum's drop-rate gate can advance.
#
# Phi is an ACTIVE metric: a freshly-reset enclosing state has Phi~=0 until the policy grips; expect
# readiness_phi ~0.6 under active grip (about the cylinder level), not ~0.9.
#
# Self-contained: new cfg + new env subclass that overrides ONLY _reset_idx (calls super() for the full
# reset batch -- which runs the proven replay/pinch reset + the AdjustRotate Phi reset -- then OVERWRITES a
# random ~frac subset with the enclosing grasp). Additive registration; no baseline task is edited.
from __future__ import annotations

import os
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjustrotate import (
    SharpaWaveGraspXLAdjustRotateCfg,
    SharpaWaveGraspXLAdjustRotateEnv,
)
from .sharpa_wave_env_cfg import _GX_OBJ
from ...graspxl.object_cfg import make_graspxl_object_cfg, object_usd_for
from ...graspxl.sharpa_dataset import DEFAULT_SHARPA_DATASET_ROOT


@configclass
class SharpaWaveGraspXLAdjustRotateDiverseCfg(SharpaWaveGraspXLAdjustRotateCfg):
    """AdjustRotate (readiness-Phi reward, pos_diff=0) + friction combine="max" + diverse pinch/enclosing
    resets. Everything else inherited from AdjustRotate -> Sustained -> OrientCurr -> ... (torque control,
    obs_hand_gravity obs 195, replay pinch init, SE(3) orient curriculum, adaptive gravity curriculum).
    Train FROM SCRATCH (obs dim 195, same as AdjustRotate)."""
    # ---- diverse reset ----
    enclosing_cache_path = f"cache/graspxl_enclosing_{_GX_OBJ}.npy"  # (N,29): 22 dof + 3 obj pos(env-rel) + 4 quat
    diverse_enclosing_frac = 0.5                                     # ~fraction of reset envs that use ENCLOSING init
    # ---- friction combine = max on the object (user-approved) ----
    # object<->finger contact uses max(mu_object, mu_finger). The env's set_friction DR overwrites the
    # static/dynamic coefficients each reset but NOT the combine mode, so combine="max" persists.
    object_cfg = make_graspxl_object_cfg(
        object_usd_for(_GX_OBJ, DEFAULT_SHARPA_DATASET_ROOT),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,            # = object_base_friction; DR re-randomizes this each reset
            dynamic_friction=0.5,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="max",
        ),
    )


class SharpaWaveGraspXLAdjustRotateDiverseEnv(SharpaWaveGraspXLAdjustRotateEnv):
    """AdjustRotate env + diverse pinch/enclosing reset distribution.

    On reset: super() resets the FULL batch with the proven replay (pinch) reset + Phi reset; then a random
    ~diverse_enclosing_frac subset is OVERWRITTEN with a settled enclosing grasp from the cache (canonical
    palm-up frame, no SE(3) qrand), mirroring how the cache-mode reset writes object/joint state."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._diverse_frac = float(getattr(cfg, "diverse_enclosing_frac", 0.5))
        self._enclosing_cache = None
        self._diverse_logged = False
        path = getattr(cfg, "enclosing_cache_path", None)
        if path:
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            arr = np.load(path)
            assert arr.ndim == 2 and arr.shape[1] == 29, \
                f"enclosing cache must be (N,29) [22 dof + 3 obj pos + 4 quat], got {arr.shape}"
            self._enclosing_cache = torch.from_numpy(arr).float().to(self.device)
            print(f"[Diverse] enclosing cache loaded: {self._enclosing_cache.shape[0]} states from "
                  f"{path}; diverse_enclosing_frac={self._diverse_frac:.2f}")
        else:
            print("[Diverse] no enclosing_cache_path -> behaves as pure pinch (AdjustRotate)")

    def _reset_idx(self, env_ids):
        # Full-batch reset via the proven path: AdjustRotate._reset_idx -> GraspXL replay reset (settle,
        # SE(3) qrand, object/joint writes) + Phi reset. We then clobber the enclosing subset.
        super()._reset_idx(env_ids)
        # guards: called during base __init__ before cache/buffers exist; or diverse disabled.
        if getattr(self, "_enclosing_cache", None) is None or not hasattr(self, "_phi_prev"):
            return
        if self._diverse_frac <= 0.0:
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        ids = ids.to(self.device)
        if ids.numel() == 0:
            return
        enc = ids[torch.rand(ids.shape[0], device=self.device) < self._diverse_frac]
        if enc.numel() == 0:
            return
        self._write_enclosing_state(enc)
        if not self._diverse_logged:
            print(f"[Diverse] first diverse reset: {int(enc.numel())}/{int(ids.numel())} envs -> ENCLOSING "
                  f"init, rest -> pinch/replay (friction combine=max)")
            self._diverse_logged = True

    def _write_enclosing_state(self, enc: torch.Tensor):
        """Overwrite the `enc` envs with a settled enclosing grasp (canonical palm-up, no qrand). Mirrors
        the cache-mode reset (sharpa_wave_env._reset_idx) on the enclosing-cache rows, and disables the
        replay-settle so these envs are under RL control immediately."""
        m = int(enc.numel())
        rows = self._enclosing_cache[torch.randint(0, self._enclosing_cache.shape[0], (m,), device=self.device)]
        dof = rows[:, :22]
        obj_pos_rel = rows[:, 22:25]
        obj_quat = rows[:, 25:29]
        origins = self.scene.env_origins[enc]

        # --- object pose (env-relative -> world) + zero velocity ---
        ostate = torch.zeros(m, 7, device=self.device)
        ostate[:, :3] = obj_pos_rel + origins
        ostate[:, 3:7] = obj_quat
        self.object.write_root_pose_to_sim(ostate, enc)
        self.object.write_root_velocity_to_sim(torch.zeros(m, 6, device=self.device), enc)
        # centering / drop reference = the enclosing object pose (env-relative)
        self.object_default_pose[enc, :3] = obj_pos_rel
        self.object_default_pose[enc, 3:7] = obj_quat
        self.rb_forces[enc, :] = 0.0

        # --- hand: canonical (palm-up) root, overriding super's SE(3) qrand for these envs ---
        hand_state = self.hand.data.default_root_state.clone()[enc]
        hand_state[:, 0:3] += origins
        self.hand.write_root_state_to_sim(hand_state, enc)
        # enclosing joints as both state and target
        self.prev_targets[enc] = dof
        self.cur_targets[enc] = dof
        self.hand.set_joint_position_target(dof, env_ids=enc)
        self.hand.write_joint_state_to_sim(dof, torch.zeros_like(dof), env_ids=enc)

        # --- canonical (un-rotated) rotation axis + identity qrand + NO replay settle ---
        if hasattr(self, "_gx_settle"):
            self._gx_settle[enc] = 0
        if hasattr(self, "_gx_qrand"):
            self._gx_qrand[enc, 0] = 1.0
            self._gx_qrand[enc, 1:] = 0.0
        base_axis = getattr(self, "_gx_base_axis", None)
        if base_axis is None:
            base_axis = torch.tensor(self.cfg.rot_axis, device=self.device, dtype=torch.float32)
        self.rot_axis[enc] = base_axis.view(1, 3).expand(m, 3)
        # drop bands unused in orient mode (displacement drop) -> set generous, centered on the enclosing z
        self.reset_height_upper[enc] = obj_pos_rel[:, 2] + 0.30
        self.reset_height_lower[enc] = obj_pos_rel[:, 2] - 0.30

        # --- refresh + reset per-env step buffers (mirror cache-mode reset tail) ---
        self._refresh_lab()
        self.object_pos_prev[enc] = self.object_pos[enc]
        self.object_rot_prev[enc] = self.object_rot[enc]
        self.last_contacts[enc] = 0
        self.proprio_hist_buf[enc] = 0
        self.at_reset_buf[enc] = 1
        # contacts are 0 right after reset -> Phi(reset)=0 (active metric); avoid a spurious first-step spike
        self._phi_prev[enc] = 0.0
        self._phi_ema[enc] = 0.0


@configclass
class SharpaWaveGraspXLAdjustRotateDiverseFixedGCfg(SharpaWaveGraspXLAdjustRotateDiverseCfg):
    """FIXED-GRAVITY variant of the 3B diverse adjust-rotate cfg (additive; reuses the Diverse env).

    WHY: the ADAPTIVE gravity curriculum stalled near-zero (3A reached gravity_z ~= -0.40, 3B stalled at
    ~= -0.20 = ~2-4% of full g). Every adjust-rotate run therefore trained at NEAR-ZERO gravity, where a
    fingertip PINCH trivially holds the object, so there is no pressure to form an enclosing grasp -- which
    confounds the adjust-then-rotate test. Fix: train at a FIXED, meaningful gravity (-4.9 = half g, with the
    curriculum OFF) so a good (enclosing) grasp actually matters, and probe later at the SAME gravity
    (matched). Everything else inherited from Diverse: friction combine="max", 50/50 pinch/enclosing resets,
    the validated readiness-Phi potential reward, torque control, obs dim 195. Train FROM SCRATCH.
    """
    # NO adaptive ramp: with this False, both curriculum blocks (sharpa_wave_env._reset_idx and
    # sharpa_wave_graspxl_env._get_dones) are skipped, so gravity stays at the fixed sim value below.
    gravity_curriculum = False
    # Override the inherited SimulationCfg gravity (base = -0.05) to a FIXED -4.9 (half g). Same dt /
    # render_interval / physx as the base; ONLY gravity_z changes. configclass converts this class-level
    # value into a per-instance default_factory deepcopy, so it overrides the parent without contaminating
    # it. The __post_init__ below re-asserts the same value as a belt-and-suspenders guarantee.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -4.9),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608,   # 2**23
            gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )

    def __post_init__(self):
        # Guaranteed override: even if the class-level redefine above were ever a no-op, force fixed gravity.
        # self.sim is already a per-instance deepcopy here, so this does not touch any base/shared default.
        self.sim.gravity = (0.0, 0.0, -4.9)
        self.gravity_curriculum = False
