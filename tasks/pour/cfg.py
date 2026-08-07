"""IsaacLab configuration for the isolated Task-5 pouring environment."""
from __future__ import annotations

import copy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg
from tasks.pour.core import AblationMode, LiquidProxyConfig, RewardWeights, SuccessThresholds
from tasks.pour.scene import PourSceneManifest


@configclass
class PourTaskCfg(DexmateCorrectionEnvCfg):
    """Two-arm task config; existing single-hand configs remain untouched."""

    action_space = 26
    observation_space = 280
    state_space = 0
    priv_info_dim = 23
    enable_pointcloud = False
    n_obj_points = 0

    scene_manifest = ""
    # Screening mode is only used by tasks.pour.screen_prior before human
    # approval.  Normal training/evaluation always require a ready manifest.
    screening_mode = False
    screening_left_prior = ""
    screening_right_prior = ""
    ablation = AblationMode.FULL.value
    curriculum_stage = 4
    stance_prefix_frames = 60
    start_jitter_position_m = 0.03
    start_jitter_rotation_deg = 15.0
    start_pool_size = 256
    phase_timeout = (20, 100, 120, 120, 120, 100, 20)

    arm_step_scale = 0.25
    arm_tube_rad = 0.12
    closure_ref_rate = 0.025
    closure_rate_max = 0.025
    closure_max = 1.25
    finger_delta_rate = 0.03
    finger_delta_max = 0.30
    grasp_min_pads = 4
    grasp_hold_steps = 8
    contact_force_thresh = 0.1
    contact_loss_steps = 5
    approach_pos_tol_m = 0.025
    cup_tip_fail_deg = 45.0
    max_spill_fraction = 0.20

    liquid = LiquidProxyConfig()
    success = SuccessThresholds()
    reward = RewardWeights()

    # The main rigid object is the right-hand bottle.
    object_cfg = copy.deepcopy(DexmateCorrectionEnvCfg.object_cfg)
    object_cfg.prim_path = "/World/envs/env_.*/Bottle"
    cup_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cup",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DexmateCorrectionEnvCfg.object_cfg.spawn.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.97)),
    )

    # Contact sensors are authored dynamically because the two sides have
    # different object filters.  The base list must not create right-only
    # sensors against the old /Object path.
    contact_sensors = []
    fingertip_bodies = []

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 26
        self.observation_space = 280
        self.priv_info_dim = 23
        self.episode_length_s = max(
            12.0,
            (sum(self.phase_timeout) + self.stance_prefix_frames) / 20.0 + 1.0,
        )


def configure_from_manifest(cfg: PourTaskCfg, path: str) -> PourSceneManifest:
    """Validate an approved manifest and write its assets into the cfg."""

    manifest = PourSceneManifest.load(path)
    manifest.validate(require_assets=True)
    cfg.scene_manifest = str(path)

    cfg.object_cfg.spawn.usd_path = manifest.bottle.usd
    cfg.object_cfg.spawn.mass_props.mass = float(manifest.bottle.mass_kg)
    cfg.object_cfg.init_state.pos = tuple(manifest.bottle.initial_pose_wxyz[:3])
    cfg.object_cfg.init_state.rot = tuple(manifest.bottle.initial_pose_wxyz[3:7])

    cfg.cup_cfg.spawn.usd_path = manifest.cup.usd
    cfg.cup_cfg.spawn.mass_props.mass = float(manifest.cup.mass_kg)
    cfg.cup_cfg.init_state.pos = tuple(manifest.cup.initial_pose_wxyz[:3])
    cfg.cup_cfg.init_state.rot = tuple(manifest.cup.initial_pose_wxyz[3:7])
    cfg.liquid = LiquidProxyConfig(
        initial_mass=cfg.liquid.initial_mass,
        flow_rate_per_second=cfg.liquid.flow_rate_per_second,
        onset_tilt_deg=cfg.liquid.onset_tilt_deg,
        full_flow_tilt_deg=cfg.liquid.full_flow_tilt_deg,
        cup_radius_m=float(manifest.cup.opening_radius_m),
        min_mouth_height_m=cfg.liquid.min_mouth_height_m,
        max_mouth_height_m=cfg.liquid.max_mouth_height_m,
        stream_margin_m=cfg.liquid.stream_margin_m,
    )
    return manifest


def configure_for_screening(
    cfg: PourTaskCfg,
    path: str,
    *,
    left_prior: str,
    right_prior: str,
) -> PourSceneManifest:
    """Configure reconstructed assets without bypassing the training gate."""

    manifest = PourSceneManifest.load(path)
    if manifest.status != "pending_grasp_approval":
        raise RuntimeError(
            "grasp screening requires a pending_grasp_approval scene, "
            f"got {manifest.status}"
        )
    manifest.cup.validate(require_assets=True, require_geometry=True)
    manifest.bottle.validate(require_assets=True, require_geometry=True)
    for label, value in (
        ("reference_npz", manifest.reference_npz),
        ("left_prior", left_prior),
        ("right_prior", right_prior),
    ):
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"{label} not found: {value}")

    cfg.screening_mode = True
    cfg.screening_left_prior = str(Path(left_prior).resolve())
    cfg.screening_right_prior = str(Path(right_prior).resolve())
    # Asset/geometry configuration is identical to approved training scenes.
    cfg.scene_manifest = str(path)
    cfg.object_cfg.spawn.usd_path = manifest.bottle.usd
    cfg.object_cfg.spawn.mass_props.mass = float(manifest.bottle.mass_kg)
    cfg.object_cfg.init_state.pos = tuple(manifest.bottle.initial_pose_wxyz[:3])
    cfg.object_cfg.init_state.rot = tuple(manifest.bottle.initial_pose_wxyz[3:7])
    cfg.cup_cfg.spawn.usd_path = manifest.cup.usd
    cfg.cup_cfg.spawn.mass_props.mass = float(manifest.cup.mass_kg)
    cfg.cup_cfg.init_state.pos = tuple(manifest.cup.initial_pose_wxyz[:3])
    cfg.cup_cfg.init_state.rot = tuple(manifest.cup.initial_pose_wxyz[3:7])
    return manifest
