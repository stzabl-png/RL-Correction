# GraspXL ENCLOSING grasp generation (drop-and-settle), applied to the GraspXL ball object.
#
# This is the official `gen_grasp` recipe (see docs/OFFICIAL_BASELINE.md step 1 + the cylinder grasp env
# rl_rebuild/tasks/inhand_rotate/sharpa_wave_grasp_env.py): place the object in the hand's pre-shaped
# ENCLOSING pose (robot_cfg default joint_pos), hold that pose with zero policy action, cycle gravity
# through 6 directions, and keep only the (hand-joints + object-pose) states that survive a full episode
# (fingertips near object + >=3 contacts + object rotation < reset_angle_diff, within a tight z-band).
# The cylinder produces cache/sharpa_grasp_linspace_*.npy this way; this file does the same for the
# GraspXL ball so it can be used as a ROTATABLE reset state (the imported GraspXL grasp is a fingertip
# PINCH that only holds -- see docs/CYLINDER_VS_BALL_CHECKLIST.md D4).
#
# SELF-CONTAINED: a NEW cfg subclass of the grasp-gen cfg + a NEW env subclass of the grasp-gen env, plus
# an additive task registration (rl_rebuild/tasks/inhand_rotate/__init__.py). No baseline files are edited.
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
from collections.abc import Sequence

import carb
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import saturate

from .sharpa_wave_grasp_env_cfg import SharpaWaveEnvCfg as _GraspGenCfg
from .sharpa_wave_grasp_env import SharpaWaveInhandRotateGraspEnv
from .sharpa_wave_env_cfg import _GX_OBJ
from rl_rebuild.graspxl.object_cfg import make_graspxl_object_cfg, object_usd_for, OBJECT_MASS
from rl_rebuild.graspxl.sharpa_dataset import DEFAULT_SHARPA_DATASET_ROOT

# Where the GraspXL ball seats in the (fixed) hand-init frame: measured from the trajectory's settled
# object center (env-relative), pose 0 -- it is within ~3 mm of the cylinder's enclosing seat
# (-0.09559, -0.00517, 0.61906), so the same enclosing hand pose wraps the (larger) ball.
_BALL_SEAT = (-0.0954, 0.0017, 0.6218)
_BALL_BAND_HALF = 0.015  # +-1.5 cm z-band quality filter (looser than the cylinder's +-0.5 cm: heavier ball)


@configclass
class SharpaWaveGraspXLGenGraspCfg(_GraspGenCfg):
    """Drop-and-settle ENCLOSING-grasp generation cfg for the GraspXL ball.

    Inherits the cylinder grasp-gen cfg VERBATIM (position control `torque_control=False`, the enclosing
    `robot_cfg` default hand pose, the 6-direction gravity stability filter via the grasp env's reward,
    `gravity_curriculum=False`, scale_range [1,1,1]). The ONLY changes are the OBJECT (GraspXL ball at its
    real size + mass) and the seat/z-band/mass/friction adapted to the ball.
    """
    # --- swap the cylinder for the GraspXL ball (real-size mesh, OBJECT_MASS=0.10 kg, convexDecomposition)
    object_cfg = make_graspxl_object_cfg(object_usd_for(_GX_OBJ, DEFAULT_SHARPA_DATASET_ROOT), mass=OBJECT_MASS)
    object_cfg.init_state = RigidObjectCfg.InitialStateCfg(pos=_BALL_SEAT, rot=(1.0, 0.0, 0.0, 0.0))

    # --- z-band centered on the ball seat (absolute world z; the grasp env writes these on reset)
    reset_height_lower = _BALL_SEAT[2] - _BALL_BAND_HALF
    reset_height_upper = _BALL_SEAT[2] + _BALL_BAND_HALF
    # The object is a SPHERE -> its orientation is physically irrelevant to grasp quality, yet the grasp
    # env's reward resets any env whose object rotated past reset_angle_diff (default 30 deg). That rejects
    # rolling-but-held spheres and crushes the yield. Relax it to 90 deg (still catches gross instability;
    # cond1 fingertips-near + cond2 >=3-contacts + the z-band remain the real hold filters).
    reset_angle_diff = math.pi / 2

    # keep the GraspXL ball at its true mass (do NOT let mass DR override 0.10 -> ~0.05)
    randomize_mass = False
    # deterministically set the training friction (object 0.5, elastomer 0.8, metal 0.1) so the heavier
    # sphere has a realistic grip (scale 1.0 = no randomization, just sets the base values).
    randomize_friction = True
    randomize_friction_scale_lower = 1.0
    randomize_friction_scale_upper = 1.0
    # NOTE: the grasp-gen cfg already has scene.replicate_physics=False (required by the GraspXL custom
    # per-env prim spawn), so it is inherited unchanged.

    # --- generation controls (read by the env below) -------------------------------------------------
    # Target settled-grasp count PER scale bucket (scale_range[2]=1 -> total). 50000 (cylinder default)
    # won't finish under a ~600 s sim cap; a few thousand enclosing grasps is plenty for a reset dist.
    gen_target_per_bucket = 8000
    gen_save_interval = 200  # incrementally overwrite the cache every +N states so a timeout still yields one
    gen_save_path = f"cache/graspxl_enclosing_{_GX_OBJ}.npy"
    # Generate under CONSTANT -z gravity (the palm-up rotation condition) instead of the cylinder's
    # 6-direction schedule. Rationale: the cylinder schedule captures the episode-END state, which lands on
    # the -x (sideways) phase; reloading that -x equilibrium under -z lets the (bigger, heavier) ball sink
    # ~6 cm out of the fingertip cage (held by the palm, ~0 fingertip contact -> Phi~0). Holding the ball at
    # the seat for the FULL 12 s under -z forces survivors to be genuine -z-stable FINGERTIP grasps whose
    # captured state reproduces the grip + contacts on reload. rot_axis=(0,0,1) -> -z is the operative g.
    gen_gravity_z = -9.81


class SharpaWaveGraspXLGenGraspEnv(SharpaWaveInhandRotateGraspEnv):
    """Grasp-gen env for the GraspXL ball: identical drop-and-settle + 6-direction-gravity filter as the
    cylinder grasp env; only the cache size cap + save path are parameterized (via cfg) and the cache is
    saved INCREMENTALLY (the baseline saves once at 50000 then exits -> nothing on a timeout)."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._last_saved_total = 0
        # constant -z gravity (override the inherited 6-direction schedule); the grasp env's _get_rewards
        # still cycles gravity_id through this list every 40 steps, but every entry is -z -> g stays -z.
        gz = float(getattr(self.cfg, "gen_gravity_z", -9.81))
        self.gravity_all_directions = [carb.Float3(0.0, 0.0, gz) for _ in range(6)]

    def _save_cache(self, final: bool):
        save_data = torch.zeros((0, 29), dtype=torch.float32, device=self.device)
        for sgs in self.saved_grasping_states:
            save_data = torch.cat([save_data, sgs], dim=0)
        os.makedirs("cache", exist_ok=True)
        path = self.cfg.gen_save_path
        np.save(path, save_data.cpu().numpy())
        print(f"[GraspXL-gen] saved {'FINAL' if final else 'checkpoint'} cache -> {path}  "
              f"shape={tuple(save_data.shape)}")

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        self._refresh_lab()
        target = int(self.cfg.gen_target_per_bucket)
        success = self.episode_length_buf == self.max_episode_length - 1
        all_states = torch.cat([self.hand_dof_pos, self.object_pos, self.object_rot], dim=1)[success]
        saved_scale_ids = self.scale_ids[success]
        for i, saved_scale_id in enumerate(saved_scale_ids):
            k = int(saved_scale_id)
            if self.saved_grasping_states[k].shape[0] < target:
                self.saved_grasping_states[k] = torch.cat(
                    [self.saved_grasping_states[k], all_states[i].reshape(-1, 29)], dim=0)
        sum_total = 0
        finish_scale = 0
        for sgs in self.saved_grasping_states:
            if sgs.shape[0] >= target:
                finish_scale += 1
            sum_total += sgs.shape[0]
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] graspxl enclosing cache: {sum_total}/'
              f'{target * self.cfg.scale_range[2]}, finished buckets: {finish_scale}/{self.cfg.scale_range[2]}')

        if sum_total > 0 and sum_total - self._last_saved_total >= int(self.cfg.gen_save_interval):
            self._save_cache(final=False)
            self._last_saved_total = sum_total
        if finish_scale == self.cfg.scale_range[2]:
            self._save_cache(final=True)
            exit()

        # ---- standard grasp-env reset (verbatim from SharpaWaveInhandRotateGraspEnv._reset_idx tail) ----
        self.scene.reset(env_ids)
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        rand_floats = 2.0 * torch.rand((len(env_ids), self.num_hand_dofs), device=self.device) - 1.0

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, :3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        self.reset_height_lower[env_ids] = self.cfg.reset_height_lower
        self.reset_height_upper[env_ids] = self.cfg.reset_height_upper

        dof_pos = self.hand.data.default_joint_pos[env_ids] + 0.15 * rand_floats
        dof_pos = saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids])
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._refresh_lab()
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
