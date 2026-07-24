# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from rl_rebuild.utils.modified_events import randomize_rigid_body_scale
from rl_rebuild.graspxl.object_cfg import make_graspxl_object_cfg, make_graspxl_multi_object_cfg, object_usd_for
from rl_rebuild.graspxl.sharpa_dataset import DEFAULT_SHARPA_DATASET_ROOT

_GX_OBJ = os.environ.get("SHARPA_GRASPXL_OBJECT", "002aa1853c974f3a9565e85f5e09a515")
# 4 convex-ish CRADLE objects spanning size/shape (ball + others), all with valid grasp trajectories
_GX_MULTI = [
    "002aa1853c974f3a9565e85f5e09a515",
    "f71f454a9d574e1eb4ea3e832e24ff0f",
    "ac7385946a4849c4924e2017fa51e24c",
    "20b591de51eb4fc3a4c5a4d40c6011d5",
]


@configclass
class EventCfg:
    def rand_params(self, scale_range: list[float, float, int]):
        '''
        Randomize the scale of the object.
        '''
        self.randomize_scale = EventTermCfg(
            func=randomize_rigid_body_scale,
            mode="prestartup",
            params={
                "scale_range": scale_range,
                "asset_cfg": SceneEntityCfg("object"),
            },
        )


@configclass
class SharpaWaveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0 # Episode length in seconds
    action_space = 22
    observation_space = 192
    prop_hist_len = 30      # Proprioception hist frames used in policy
    priv_info_dim = 8
    state_space = 0
    asymmetric_obs = False
    # control
    decimation = 12
    clip_obs = 5.0
    clip_actions = 1.0
    action_scale = 1 / 24
    torque_control = True
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -0.05),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608, # 2**23
            gpu_max_rigid_patch_count=5*2**18
        ),
    )
    # robot
    hand_init_pose = ((0.0, 0.0, 0.5), (0.819152, 0.0, -0.5735764, 0.0))
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"../../../assets/SharpaWave/right_sharpa_wave.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=hand_init_pose[0],
            rot=hand_init_pose[1],
            joint_pos={
                "right_thumb_CMC_FE": math.pi/180 * 95.12771,
                "right_thumb_CMC_AA": math.pi/180 * -3.11244,
                "right_thumb_MCP_FE": math.pi/180 * 14.81626,
                "right_thumb_MCP_AA": math.pi/180 * -1.03493,
                "right_thumb_IP": math.pi/180 * 12.23986,
                "right_index_MCP_FE": math.pi/180 * 65.21091, 
                "right_index_MCP_AA": math.pi/180 * 6.1133,
                "right_index_PIP": math.pi/180 * 15.58495,
                "right_index_DIP": math.pi/180 * 5.90325,
                "right_middle_MCP_FE": math.pi/180 * 31.74149,
                "right_middle_MCP_AA": math.pi/180 * -0.95812,
                "right_middle_PIP": math.pi/180 * 41.88173,
                "right_middle_DIP": math.pi/180 * 12.844,
                "right_ring_MCP_FE": math.pi/180 * 31.72383,
                "right_ring_MCP_AA": math.pi/180 * 9.84458,
                "right_ring_PIP": math.pi/180 * 35.22366,
                "right_ring_DIP": math.pi/180 * 18.02839,
                "right_pinky_CMC": math.pi/180 * 10.9712,
                "right_pinky_MCP_FE": math.pi/180 * 68.30895,
                "right_pinky_MCP_AA": math.pi/180 * 7.99151,
                "right_pinky_PIP": math.pi/180 * 5.89626,
                "right_pinky_DIP": math.pi/180 * 5.89875,
            },
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )

    contact_sensor = [
        # elastomer
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_thumb_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_index_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_ring_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_pinky_elastomer",
            history_length=3,
            track_contact_points=True,
            max_contact_data_count_per_prim=10,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        # DP
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_thumb_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_index_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_middle_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_ring_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        ),
        ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/right_pinky_DP",
            history_length=3,
            filter_prim_paths_expr=["/World/envs/env_.*/object"],
        )
    ]

    actuated_joint_names = [
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
        "right_index_MCP_FE",
        "right_index_MCP_AA",
        "right_index_PIP",
        "right_index_DIP",
        "right_middle_MCP_FE",
        "right_middle_MCP_AA",
        "right_middle_PIP",
        "right_middle_DIP",
        "right_ring_MCP_FE",
        "right_ring_MCP_AA",
        "right_ring_PIP",
        "right_ring_DIP",
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ]
    fingertip_body_names = [
        "right_thumb_fingertip",
        "right_index_fingertip",
        "right_middle_fingertip",
        "right_ring_fingertip",
        "right_pinky_fingertip",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"../../../assets/cylinder/cylinder.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002, 
                rest_offset=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            scale=(1., 1., 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.09559, -0.00517, 0.61906), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16384, env_spacing=0.75, replicate_physics=False)
    # event
    events: EventCfg = EventCfg()
    # reset
    reset_height_lower = 0.59906
    reset_height_upper = 0.63906
    reset_angle_diff = 45 / 180 * math.pi # Not used.
    reset_random_quat = False             # If True, the quaternion of the hand and object is randomized.
    # reward
    rot_axis = (0, 0, 1)                  # Rotation axis used in reward function.
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5                 
    rotate_reward_scale = 2.5            
    object_linvel_penalty_scale = -0.3
    pos_diff_penalty_scale = -0.4
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    object_pos_reward_scale = 0.003
    # grasp cache
    grasp_cache_path = 'cache/sharpa_grasp_linspace' # Grasp cache used in training.
    # noise
    joint_noise_scale = 0.02
    # contact
    enable_tactile = True       # If True, the tactile sensor is enabled.
    binary_contact = False      # If True, the output tactile force will be binarized according to the contact_threshold.
    enable_contact_pos = False  # Not tested yet. If True, the tactile sensor will output the contact position.
    disable_tactile_ids = []    # Set 0 to according tactile ids.
                                # 0, 1, 2, 3, 4 are thumb, index, middle, ring, pinky finger, respectively.
    contact_smooth = 0.5        # Smoothing factor for tactile force.
    contact_threshold = 0.05    # Binary contact force threshold, only used when binary_contact is True.
    contact_latency = 0.005     # Contact latency.
    contact_sensor_noise = 0.01 # Contact sensor noise, only used when binary_contact is True.
    # align real
    dof_limits_scale = 0.9      # Multiply a scale to the URDF joint limits.
    # randomize
    scale_range = [0.5, 0.5, 1] # Scale size of the object, [lower, upper, num].
    events.rand_params(scale_range)
    randomize_pd_gains = True   # Randomize PD gains.
    randomize_p_gain_scale_lower = 0.5
    randomize_p_gain_scale_upper = 2
    randomize_d_gain_scale_lower = 0.5
    randomize_d_gain_scale_upper = 2
    randomize_friction = True   # Randomize friction.
    randomize_friction_scale_lower = 0.5
    randomize_friction_scale_upper = 2.0
    elastomer_base_friction = 0.8
    metal_base_friction = 0.1
    object_base_friction = 0.5
    randomize_com = True        # Randomize center of mass.
    randomize_com_lower = -0.01
    randomize_com_upper = 0.01
    randomize_mass = True       # Randomize object mass.
    randomize_mass_lower = 0.01
    randomize_mass_upper = 0.25
    # random forces applied to the object
    force_scale = 2
    random_force_prob_scalar = 0.25
    force_decay = 0.9
    force_decay_interval = 0.08
    # curriculum
    gravity_curriculum = True # If True, gravity is gradually increased during training, upper limits is 10m/s^2.
    # debug
    debug_show_axes = False   # If True, visualize the coordinate axes of the object.

    # --- REBUILD STAGE 2: hand/object point cloud + masks (egocentric, wrist frame) ----------------
    # When enabled, the env emits obs["pointcloud"] of shape (num_envs, P, 5) = xyz(3) + mask(2 one-hot:
    # [1,0]=hand, [0,1]=object), in the WRIST frame. Object points are sampled on the cylinder surface
    # (transformed by the object pose each step); hand points are the hand body link positions (FK).
    # A lightweight PointNet branch in the policy consumes it. Gated -> default OFF keeps the baseline.
    enable_pointcloud = False
    pc_num_object = 128       # object surface points sampled on the cylinder
    pc_obj_radius = 0.04      # cylinder radius (URDF) BEFORE the object scale is applied
    pc_obj_half_height = 0.032  # cylinder half-height (URDF length 0.064 / 2) BEFORE scale

    # --- REBUILD STAGE 3: world-model auxiliary loss ----------------------------------------------
    # When enabled, the policy net gets a head that predicts the object pose (pos[3] + quat[4]) from the
    # actor latent; the env exposes obs["obj_pose"] (env-relative, wrist frame) as the target, and PPO
    # adds an MSE auxiliary loss. Gated -> default OFF.
    enable_world_model = False
    world_model_coef = 1.0


@configclass
class SharpaWaveEnvCfgPC(SharpaWaveEnvCfg):
    """STAGE 2: baseline + hand/object point cloud + masks (tactile obs kept)."""
    enable_pointcloud = True


@configclass
class SharpaWaveEnvCfgWM(SharpaWaveEnvCfg):
    """STAGE 3: point cloud + world-model auxiliary loss (predict object pose)."""
    enable_pointcloud = True
    enable_world_model = True


@configclass
class SharpaWaveGraspXLCfg(SharpaWaveEnvCfg):
    """GraspXL object (your dataset objects) with trajectory-derived grasp init. Keeps the rebuild's
    proprio+tactile core + torque control + adaptive gravity curriculum + HORA reward. Two init modes
    (graspxl_init: 'cache' | 'replay'). Point cloud / world model OFF for now (the PC sampler is
    cylinder-specific; a per-object mesh sampler is a follow-up)."""
    graspxl_object_id = _GX_OBJ
    graspxl_dataset_root = str(DEFAULT_SHARPA_DATASET_ROOT)
    graspxl_init = "cache"
    graspxl_num_poses = 3
    graspxl_cache_frames = 30
    graspxl_hold_steps = 20
    graspxl_drop_margin = 0.08
    graspxl_up_margin = 0.05
    # STEP 3 orientation randomization (off by default)
    graspxl_orient_rand = False
    graspxl_orient_max_angle = math.pi
    # STEP 3b annealed orientation-range curriculum (off by default): start near palm-up so the adaptive
    # gravity curriculum reaches full g, then expand the sampled-orientation cap toward orient_max_angle.
    graspxl_orient_curriculum = False
    graspxl_orient_start = 0.3       # rad initial tilt cap (~17 deg about palm-up)
    graspxl_orient_step = 0.15       # rad expand per (gravity-full + drop~0) event
    graspxl_drop_disp = 0.10
    # GRASP-SQUEEZE: continue the finger-closing past the settled pinch into a tighter, more enclosing grip
    # (fraction of the net approach closing; 0 = off). Fixes the pinch's one-time-rotation-then-drop.
    graspxl_squeeze = 0.0
    graspxl_squeeze_frames = 15
    object_cfg = make_graspxl_object_cfg(object_usd_for(_GX_OBJ, DEFAULT_SHARPA_DATASET_ROOT))
    grasp_cache_path = None
    scale_range = [1.0, 1.0, 1]
    events: EventCfg = EventCfg()
    events.rand_params([1.0, 1.0, 1])
    # cache-mode drop band uses (upper-lower) as the band width centered on the cached object z
    reset_height_lower = 0.0
    reset_height_upper = 0.16
    # DR: keep friction/com/pd; fix mass (grasp-specific); gentler force perturbation for a pinch grasp
    randomize_mass = False
    force_scale = 0.5
    # PC/WM off for GraspXL (cylinder PC sampler doesn't match arbitrary objects)
    enable_pointcloud = False
    enable_world_model = False


@configclass
class SharpaWaveGraspXLCacheCfg(SharpaWaveGraspXLCfg):
    graspxl_init = "cache"


@configclass
class SharpaWaveGraspXLReplayCfg(SharpaWaveGraspXLCfg):
    graspxl_init = "replay"


@configclass
class SharpaWaveGraspXLReplayPCWMCfg(SharpaWaveGraspXLReplayCfg):
    """STAGE-4 design on GraspXL (single object, replay init, fixed orientation): proprio+tactile PLUS the
    point-cloud + 2-ch mask branch AND the world-model pose head. The object point cloud is sampled from the
    GraspXL object MESH (object.obj), not the cylinder (see SharpaWaveGraspXLEnv.__init__). This is the
    standing 'all models follow stage-4' design — the policy sees real object geometry and predicts object
    pose. Object scale is 1.0 (mesh is real-size)."""
    enable_pointcloud = True
    enable_world_model = True
    world_model_coef = 1.0
    # CLEAN INHERITANCE of the WORKING cylinder design: match the cylinder's domain randomization exactly
    # (the GraspXL base had softened these to coddle the fragile pinch — that was meaningless carry-over).
    # The ONLY intended differences from the cylinder world-model task are the OBJECT (GraspXL ball) and the
    # INITIAL GRASP POSE (trajectory replay). Everything else (network, reward, DR) is identical.
    randomize_mass = True            # cylinder: True (0.01-0.25); GraspXL base had turned it OFF
    force_scale = 2                  # cylinder: 2; GraspXL base had softened to 0.5
    graspxl_squeeze = 0.0            # NO squeeze (that was an old-work hack)


@configclass
class SharpaWaveGraspXLReplayPCWMSqueezeCfg(SharpaWaveGraspXLReplayPCWMCfg):
    """STAGE-4 (PC+world-model) on GraspXL + GRASP-SQUEEZE. The plain PC+WM run did one-time-rotation
    (rotate-then-drop) because the imported pinch can't sustain rotation; squeeze the grasp into a tighter,
    more enclosing grip so rotation either sustains (like the cylinder) or settles into a stable hold —
    not a one-time drop. Single object, fixed orientation."""
    graspxl_squeeze = 0.2
    graspxl_squeeze_frames = 15


@configclass
class SharpaWaveGraspXLOrientCfg(SharpaWaveGraspXLReplayCfg):
    """STEP 3: SE(3) orientation randomization (single object) — the whole hand+object is rotated by a
    random quaternion each episode so the palm faces different directions; the grasp (SE3 relation) is
    preserved. Drop is displacement-based (orientation-invariant). Starts with full SO(3)."""
    graspxl_orient_rand = True
    graspxl_orient_max_angle = math.pi
    graspxl_drop_disp = 0.10


@configclass
class SharpaWaveGraspXLOrientPCWMCfg(SharpaWaveGraspXLOrientCfg):
    """STAGE-4 design on GraspXL + SE(3) orientation as initial state (single object): the 'multiple
    orientation initial state' step, with the point-cloud + world-model design. Object PC sampled from the
    GraspXL mesh. Uses the ANNEALED orientation-range curriculum — start near palm-up so the adaptive
    gravity curriculum reaches full g, then expand the sampled-orientation cap toward full SO(3) (full SO(3)
    from step 1 stalls the gravity curriculum at ~52%)."""
    graspxl_orient_curriculum = True
    graspxl_orient_start = 0.3
    graspxl_orient_step = 0.15
    graspxl_orient_max_angle = math.pi
    enable_pointcloud = True
    enable_world_model = True
    world_model_coef = 1.0


@configclass
class SharpaWaveGraspXLOrientCurrCfg(SharpaWaveGraspXLOrientCfg):
    """STEP 3b: SE(3) orientation randomization with an ANNEALED orientation-range curriculum. Starts
    near palm-up (small tilt cap) so the adaptive gravity curriculum can reach FULL g, THEN expands the
    sampled-orientation cap toward full SO(3) (drop-rate-gated, after gravity is full). Fixes the
    baseline Orient stall at ~52% g (full SO(3) from step 1 gated the gravity curriculum)."""
    graspxl_orient_curriculum = True
    graspxl_orient_start = 0.3
    graspxl_orient_step = 0.15
    graspxl_orient_max_angle = math.pi
    # NOTE: replicate_physics stays False (inherited) — the env's USD-level domain randomization requires
    # it; the slow CPU-clone build is the cost of DR. Launch via the harness background runner (setsid
    # boot-crashes on the carb mutex; long foreground commands tear down the bg task).


@configclass
class SharpaWaveGraspXLOrientGravCfg(SharpaWaveGraspXLOrientCurrCfg):
    """SO(3) sustained rotation, the principled way (use the OFFICIAL reward, not hora-hacks).
    The proprio+tactile obs is EGOCENTRIC -> identical at every palm orientation except gravity pulls in a
    different (unobserved) direction; that's why the egocentric policy couldn't adapt and collapsed to
    shaking/one-time rotation. FIX: append the GRAVITY VECTOR IN THE HAND FRAME (3-d) to the policy obs so
    the same gait can be made SO(3)-equivariant. Keeps the official SharpaWave reward (rotate 2.5, clip
    ±0.5, pos_diff -0.4, work -0.5, object_pos 0.003 — all inherited, NOT hacked) + the annealed
    orientation curriculum + adaptive gravity curriculum. Trains FROM SCRATCH (obs dim changed)."""
    obs_hand_gravity = True
    proprio_frame_dim = 64               # keep the proprio lag-history at 64/frame (3*64=192)...
    observation_space = 192 + 3          # ...and declare the policy obs as 192 + hand-gravity(3)
    # mild loosening of the curriculum gate (official 5e-4 demands near-perfect holding and STALLS once the
    # policy rotates); 3e-3 still << the 10% held target, lets rotation+curriculum coexist.
    graspxl_curr_drop_thresh = 0.003
    # expand the orientation cap only while rotation is MAINTAINED (mean rotate_reward > gate) — prevents
    # the cap from racing ahead of the gait into a holding/shaking collapse. Slower step for fine adaptation.
    graspxl_orient_rotate_gate = 0.20
    graspxl_orient_step = 0.10


@configclass
class SharpaWaveGraspXLOrientScratchCfg(SharpaWaveGraspXLOrientCurrCfg):
    """STEP 3d: FROM-SCRATCH (do NOT --resume) run of the orientation curriculum with a ROTATION-DOMINANT
    reward that KEEPS the jitter-damping penalties. Rationale: warm-starting either checkpoint converged to
    a jitter/hold basin; learning rotation from scratch during the EASY stage of the curriculum (near-zero
    g, near-palm-up) — when rotation-while-held is achievable — can establish the rotation skill before the
    curriculum hardens, instead of inheriting a bad basin. Reward:
      - rotate_reward_scale 2.5 -> 5.0 (rotation dominant)
      - angvel_clip [-0.5,0.5] -> [-1.0,1.0] SYMMETRIC (jitter cancels to ~0 net; faster rotation pays up
        to 1 rad/s)
      - pos_diff -0.4 -> -0.15 (allow gaiting), object_linvel -0.3 -> -0.2 (mild)
      - work -0.5 -> -0.3, torque -0.1 KEPT (DAMP the high-power back-and-forth jitter; relaxing these in
        v1 was a mistake), object_pos 0.003 -> 0.002 (keep a mild centering/hold term for the curriculum)."""
    rotate_reward_scale = 6.0
    angvel_clip_min = -1.0
    angvel_clip_max = 1.0
    pos_diff_penalty_scale = -0.1
    object_linvel_penalty_scale = -0.2
    torque_penalty_scale = -0.05
    work_penalty_scale = -0.2
    # object_pos_reward REMOVED: the term 1/(disp+eps) EXPLODES (~400) when the object is held dead-still,
    # so it dominated the rotate reward and the policy just held. The displacement-drop termination
    # (disp>0.10 ends the episode) + object_linvel are enough to enforce holding without a hold *reward*.
    object_pos_reward_scale = 0.0
    # the drop-gated curriculum STALLS under rotation (rotation-induced drops keep the rate > 5e-4);
    # loosen the gate so gravity/orient advance while the policy rotates (2% drop << the 10% held target).
    graspxl_curr_drop_thresh = 0.02


@configclass
class SharpaWaveGraspXLOrientCurrFTCfg(SharpaWaveGraspXLOrientCurrCfg):
    """STEP 3c: warm-start FINE-TUNE of the OrientCurr checkpoint for SUSTAINED DIRECTIONAL rotation.
    The OrientCurr policy reaches full g + full SO(3) but HOLDS-AND-SHAKES (signed yaw ~0). Rebalance the
    HORA reward so directional rotation pays far more than holding/jitter:
    MODERATE reward (v2): warm-start from the ROTATING single-object Replay policy (logs/debug/
    2026-06-28_02-02-07 — yaw 0.79) into the orientation curriculum, so the rotation skill is present at
    the start. v1 (warm-start the SHAKING OrientCurr ckpt + heavily-relaxed penalties + rotate 5) FAILED:
    it shook FASTER (eval |yaw| 1.66, signed 0.06) — relaxing work/torque freed jitter and PPO stayed in
    the jitter basin. v2 keeps the JITTER-suppressing penalties and only mildly boosts rotation:
      - rotate_reward_scale 2.5 -> 3.5 (mild rotation boost)
      - angvel_clip [-0.5,0.5] -> [-1.0,1.0] (SYMMETRIC, wider so sustained ~0.8 rad/s isn't clipped and
        jitter still cancels to ~0 net; never asymmetric — that rewards jitter)
      - pos_diff_penalty -0.4 -> -0.2 (allow finger gaiting away from the grasp pose)
      - work_penalty -0.5 -> -0.3, torque -0.1 (KEEP: penalize the high-power back-and-forth jitter)
      - object_linvel -0.3, object_pos 0.003 (KEEP holds; the single-object policy rotates fine with them)
    Train: --resume --load_path logs/debug/2026-06-28_02-02-07/stage1_nn/last.pth"""
    rotate_reward_scale = 3.5
    angvel_clip_min = -1.0
    angvel_clip_max = 1.0
    pos_diff_penalty_scale = -0.2
    object_linvel_penalty_scale = -0.3
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.3
    object_pos_reward_scale = 0.003


@configclass
class SharpaWaveGraspXLSustainedCfg(SharpaWaveGraspXLOrientCurrCfg):
    """SUSTAINED SO(3) rotation (solve the one-time-rotation drop). Built on the validated two-regime
    finding: the official/survival-dominant reward HOLDS (signed yaw ~0.03-0.09 rad/s -- the single-object
    'rotation' was that, mislabeled by the old |yaw| metric), while a rotation-dominant reward (OrientScratch,
    rotate 6 / clip +-1 / no centering) ROTATES (signed ~0.6) but DROPS in ~3.6 s (one-time). The drop is the
    real blocker, so the design rewards rotating WHILE staying in grasp, with a CAPPED rotation speed so the
    policy is not paid to shear the ball out:
      - rotate_reward_scale 3.0 (rotation dominant but moderate), angvel_clip +-0.5 SYMMETRIC (official cap;
        0.5 rad/s IS a good sustained speed -- do NOT reward faster spins, which shear a fingertip pinch out)
      - BOUNDED centering reward (exp(-disp/0.03) in [0,1], scale 1.0): the per-step still-in-grasp /
        survival-while-rotating signal that the 1/(disp+eps) term (explodes to ~400) could never provide
        without dominating. This is the key new ingredient vs OrientScratch.
      - object_linvel -0.5 (STRONG: penalize the object drift that precedes a drop), pos_diff -0.1 (allow
        finger gaiting), work -0.5 / torque -0.1 (KEEP: damp the high-power back-and-forth jitter)
      - obs_hand_gravity True (SO(3)-equivariance fix: hand-frame gravity, obs 192+3) -> train FROM SCRATCH
      - rotation-gated curriculum (gravity + orientation advance only while rotation is maintained)
    Train FROM SCRATCH (obs dim changed; do NOT --resume)."""
    obs_hand_gravity = True
    proprio_frame_dim = 64
    observation_space = 192 + 3
    rotate_reward_scale = 3.0
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5
    object_linvel_penalty_scale = -0.5
    pos_diff_penalty_scale = -0.1
    torque_penalty_scale = -0.1
    work_penalty_scale = -0.5
    center_reward_bounded = True          # use exp(-disp/sigma) in [0,1] instead of 1/(disp+eps)
    center_reward_sigma = 0.03
    object_pos_reward_scale = 1.0         # bounded -> safe to weight at 1.0 (max 1.0/step, ~ rotate's 1.5)
    graspxl_curr_drop_thresh = 0.01       # let gravity/orient advance while rotating (1% drop << 10% target)
    graspxl_orient_rotate_gate = 0.15     # expand orientation cap only while rotation is maintained
    graspxl_orient_step = 0.10


@configclass
class SharpaWaveGraspXLMultiCfg(SharpaWaveGraspXLReplayCfg):
    """STEP 2: multiple convex GraspXL objects (per-env different), trajectory-replay init. Requires
    replicate_physics=False (inherited) and num_envs divisible by the object count for clean cycling."""
    graspxl_object_ids = _GX_MULTI
    object_cfg = make_graspxl_multi_object_cfg(
        [object_usd_for(o, DEFAULT_SHARPA_DATASET_ROOT) for o in _GX_MULTI])


@configclass
class SharpaWaveEnvCfgWMMulti(SharpaWaveEnvCfg):
    """STAGE 4: pc + world model + MULTI-SCALE objects (8 cylinder sizes 0.4-0.6) -> shape/size
    generalization. Uses the provided cache/sharpa_grasp_linspace_0.4-0.6-8.npy. The per-env object
    geometric scale (scale_ids) also scales the object point cloud, so the visual branch sees the
    correct size per env."""
    enable_pointcloud = True
    enable_world_model = True
    scale_range = [0.4, 0.6, 8]
    events: EventCfg = EventCfg()
    events.rand_params([0.4, 0.6, 8])
