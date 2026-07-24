"""Isaac Lab scene for Dexmate Vega-1P + dual SharpaWave hands.

Asset + drive parameters come from MagicSim (the user's sim framework):
  USD:    ~/luhr/magicsim/MagicSim/Assets/Robots/vega_1p_sharpa.usd
  Source: MagicSim/src/magicsim/Env/Robot/Cfg/MobileManip/vega1psharpa.py
          (VEGA_1P_SHARPA_CFG - gains mirror the USD's authored drive props;
          hand gains are MagicSim's stability override kp=20/kd=2/effort=300)
We copy only the ArticulationCfg here instead of importing magicsim, which
would drag in curobo/pink deps this env does not have.

IMPORTANT: import only after AppLauncher has created the Kit app.

Articulation layout (67 joints):
  dummy_base_prismatic_x/y_joint, dummy_base_revolute_z_joint  (holonomic base)
  torso_j1..3, head_j1..3, L/R_arm_j1..7, 44 sharpa finger joints
EEF body links: R_ee / L_ee; base link: vega_1p_base.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

MAGICSIM_ASSETS = os.environ.get(
    "MAGICSIM_ASSETS", os.path.expanduser("~/luhr/magicsim/MagicSim/Assets")
)
VEGA_SHARPA_USD = f"{MAGICSIM_ASSETS}/Robots/vega_1p_sharpa.usd"

R_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
L_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
EE_BODY = {"right": "R_ee", "left": "L_ee"}

# -- drive parameters copied from MagicSim vega1psharpa.py (USD-aligned) -------
_TORSO_DEFAULT_POS = {"torso_j1": 1.0472, "torso_j2": 1.5708, "torso_j3": -0.4363}
_TORSO_EFFORT = {"torso_j1": 700000.0, "torso_j2": 380000.0, "torso_j3": 380000.0}
_HEAD_KP = {"head_j1": 45.22, "head_j2": 53.39, "head_j3": 30.43}
_HEAD_KD = {"head_j1": 0.01809, "head_j2": 0.02136, "head_j3": 0.01217}
_HEAD_EFFORT = {"head_j1": 6.0, "head_j2": 2.5, "head_j3": 6.0}
_ARM_KP = {
    "L_arm_j1": 2503.5105, "L_arm_j2": 2174.9553, "L_arm_j3": 1889.6868,
    "L_arm_j4": 1059.5746, "L_arm_j5": 659.2767, "L_arm_j6": 210.2241,
    "L_arm_j7": 113.6074,
    "R_arm_j1": 2448.9192, "R_arm_j2": 2174.8486, "R_arm_j3": 1885.5143,
    "R_arm_j4": 1056.1354, "R_arm_j5": 661.4667, "R_arm_j6": 210.5952,
    "R_arm_j7": 113.3994,
}
_ARM_EFFORT = {"L_arm_j[1-7]": 1500.0, "R_arm_j[1-6]": 1500.0, "R_arm_j7": 250.0}

VEGA_1P_SHARPA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=VEGA_SHARPA_USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # torso/L_arm match the USD drive targets so high-kp joints don't yank
        # at t=0. The RIGHT arm gets an elbow-bent teleop home that keeps every
        # joint AWAY from its limits (j2 in [-1.55,0.45], j4 in [-3.07,0.24],
        # j6 in [-1.40,1.40] - measured): straight-arm j4=0 sits at the j4
        # bound and lets the IK drift onto a branch with j6 limit-pinned,
        # which blocks local-x (roll) wrist rotation.
        joint_pos={
            **_TORSO_DEFAULT_POS,
            "L_arm_j1": -0.7854,
            "R_arm_j1": 0.7854,
            "R_arm_j2": -0.40,
            "R_arm_j4": -1.40,
            "R_arm_j7": 0.10,
            "^(?!torso_j[123]$|[LR]_arm_j1$|R_arm_j[247]$).*$": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            effort_limit_sim=4800.0, stiffness=0.0, damping=1e5,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_j.*"],
            effort_limit_sim=_TORSO_EFFORT, stiffness=800000.0, damping=36000.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_j.*"],
            effort_limit_sim=_HEAD_EFFORT, stiffness=_HEAD_KP, damping=_HEAD_KD,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["L_arm_j.*", "R_arm_j.*"],
            effort_limit_sim=_ARM_EFFORT, stiffness=_ARM_KP, damping=150.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_(thumb|index|middle|ring|pinky)_.*",
                "right_(thumb|index|middle|ring|pinky)_.*",
            ],
            effort_limit_sim=300.0, stiffness=20.0, damping=2.0,
        ),
    },
    articulation_root_prim_path=None,  # keep the USD's /root_joint fixed base
)


def make_snap_camera_cfg(width: int = 1280, height: int = 800) -> CameraCfg:
    """RGB camera for self-inspection screenshots (teleop_vega_sharpa --snap-dir)."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/snap_cam",
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 30.0)),
    )


@configclass
class VegaSharpaSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )
    robot: ArticulationCfg = VEGA_1P_SHARPA_CFG
    # set to make_snap_camera_cfg() by the runner when screenshots are requested
    camera: CameraCfg | None = None
