"""Isaac Lab scene for the bare Dexmate Vega-1P (no hands): PICO arm teleop.

USD: MagicSim/Assets/Robots/vega_1p.usd - 26 revolute joints (3x2 swerve
wheels, torso 3, arms 2x7, head 3) with L_ee / R_ee fixed end-effector links
(verified by USD traversal 2026-07-19). The articulation root sits on
/vega_1p/base as a FLOATING body, so we pin it with fix_root_link=True -
phase 1 is upper-body-only, the wheeled base stays parked (wheel joints held
by a zero-stiffness / high-damping actuator, same recipe as the sharpa
scene's dummy base).

Drive parameters: torso/head/arm gains reuse vega_sharpa_scene.py's values
(measured from MagicSim's vega1psharpa cfg). Gravity is disabled on the
robot, matching the sharpa scene - this is a kinematic-mapping sandbox, not
a dynamics validation rig.

Init pose: torso upright (same as sharpa scene); BOTH arms get the
elbow-bent teleop home that keeps every joint away from its limits (the
sharpa scene derived it for the right arm; the left is its mirror - limits
are mirrored per the USD: R_j2 [-89,26]deg vs L_j2 [-26,89]deg,
R_j7 [-64,79]deg vs L_j7 [-79,64]deg).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

# USD 查找顺序:环境变量 > 仓库自带 assets/isaac(打包发布带 USD 时存在)>
# 本机 MagicSim 检出。这样克隆下来的独立仓库不用配任何环境变量就能跑仿真。
_REPO_ASSETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "isaac")
MAGICSIM_ASSETS = os.environ.get(
    "MAGICSIM_ASSETS",
    _REPO_ASSETS if os.path.isdir(_REPO_ASSETS)
    else os.path.expanduser("~/magicsim/MagicSim/Assets")
)
VEGA_1P_USD = f"{MAGICSIM_ASSETS}/Robots/vega_1p.usd"
# Weld wheels: wheel joints retyped to fixed (vega_1p_weldwheels.usd).
# 2026-07-28 forensics: the numerically-unstable parked wheels corrupt the
# whole articulation solve and drag the low-impedance wrist joints (j6/j7)
# around - crawling response, parking off-target. Welding them removes the
# corruption at the source. DEFAULT ON since 2026-07-29 (welded demo: track
# err mean 0.9mm vs 76mm free-wheel); set VEGA_WELD_WHEELS=0 to revert.
WELD_WHEELS = os.environ.get("VEGA_WELD_WHEELS", "1").lower() not in (
    "0", "", "no", "off")
_WELD_USD = f"{MAGICSIM_ASSETS}/Robots/vega_1p_weldwheels.usd"
if WELD_WHEELS and not os.path.isfile(_WELD_USD):
    print(f"[vega_scene] WARNING: {_WELD_USD} missing - falling back to the "
          "free-wheel vega_1p.usd (wheel instability will degrade wrist "
          "tracking; see workflow_fj.md 2026-07-29)")
    WELD_WHEELS = False
if WELD_WHEELS:
    VEGA_1P_USD = _WELD_USD
# URDF the USD was generated from (kinematically identical; used by the Pink IK
# solver, which needs a pinocchio model). Base frame differs from the USD by a
# fixed rigid transform - teleop_vega_pico.py calibrates it at startup.
VEGA_1P_URDF = os.environ.get(
    "VEGA_URDF",
    os.path.expanduser("~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf"))

R_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
L_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
ARM_JOINTS = {"right": R_ARM_JOINTS, "left": L_ARM_JOINTS}
EE_BODY = {"right": "R_ee", "left": "L_ee"}

# -- gains copied from vega_sharpa_scene.py (MagicSim-measured) ----------------
# Imported, not re-declared: this constant lived in five files and every
# copy said 55 deg of forward lean while all thirteen authored poses were
# at 10.3 deg. See magicdexmate/head_waist_map.py.
from magicdexmate.head_waist_map import TORSO_HOME as _TORSO_DEFAULT_POS
_TORSO_EFFORT = {"torso_j1": 700000.0, "torso_j2": 380000.0, "torso_j3": 380000.0}
# VEGA_HEAD_KP_SCALE: the head is the one group whose gains were never scaled
# (torso 1000x the URDF effort, arms 10x, head 1x), and it does not hold its
# commanded angle - at the authored torso home, head_j1 commanded 0 settles
# short. A P controller under a constant load always does; the only lever in an
# ImplicitActuator is kp. Kept as a knob rather than a hardcoded number so the
# choice is made by the [home]/[settle] readouts, not by taste.
# 10.0 chosen by measurement, 2026-08-02, on the [home] readout: commanded 0
# deg, settled - 1x: +9.6 / -0.8 / -0.1;  4x: -2.3 / +0.1 / +5.5;
# 10x: -0.1 / +0.0 / +0.2;  30x: no further gain. It is the same 10x the arms
# already get over their URDF values, so this brings the head into line rather
# than singling it out. (Most of the earlier misbehaviour - head_j3 settling
# 62.8 deg off - was GPU physics, not gains: on CPU physics the same build
# settles j3 to -0.1 deg at 1x. Both fixes were needed.)
_HEAD_KP_SCALE = float(os.environ.get("VEGA_HEAD_KP_SCALE", "10.0"))
_HEAD_KP = {"head_j1": 45.22 * _HEAD_KP_SCALE,
            "head_j2": 53.39 * _HEAD_KP_SCALE,
            "head_j3": 30.43 * _HEAD_KP_SCALE}
# kd ~ kp/16 (same recipe as the arms): the original 0.012-0.021 values gave
# damping ratio ~0.001 - an undamped oscillator that wobbled forever off arm
# reaction torques (2026-07-29 user observation in the replay session)
# VEGA_HEAD_KD_RATIO: kd = kp * ratio. 1/16 is the arms' recipe, but the head
# is not an arm - it carries no load and the operator looks at it, so a wobble
# costs more than a few ms of lag. Measured 2026-08-03: at 1/16, head_j3 is
# commanded to hold at 0 all run and still swings 7.15 deg peak.
# 1/8 picked by measurement, 2026-08-03, on |commanded - achieved| over
# logs/clip_headwaist.msgpack. head_j3 is commanded to hold at 0 the whole run,
# so its deviation is pure wobble; head_j1 carries the pitch, so its deviation
# is tracking lag. More damping trades the first for the second:
#     kd/kp     j3 wobble max     j1 lag max
#     1/16          7.28 deg        5.10 deg   (was here)
#     1/8           3.53            6.85       <- picked
#     1/4           2.29            9.18
#     1/2           1.17           11.86
# Past 1/8 the joint that actually does the looking gets mushy, which is the
# opposite of the "make up/down smoother" this was for.
_HEAD_KD_RATIO = float(os.environ.get("VEGA_HEAD_KD_RATIO", str(1.0 / 8.0)))
_HEAD_KD = {n: kp * _HEAD_KD_RATIO for n, kp in _HEAD_KP.items()}
# 10x the URDF values, the same headroom the arms get (URDF 150/80/25 ->
# 1500/250). The head was the ONE group still at 1x while the torso runs at
# 1000x, and it could not hold its own commanded angle: at the authored torso
# home, head_j1 commanded 0 deg settled at +20.4 and head_j3 at -19.1, and the
# settle loop never went quiet. That is saturation, not tuning - kp is 45.22,
# so 20.4 deg of error asks for 16.1 Nm against a 6 Nm ceiling. It is also the
# "the head swings freely 109-158 deg with no command" observation from
# 2026-07-31, which was logged as a damping question.
_HEAD_EFFORT = {"head_j1": 60.0, "head_j2": 25.0, "head_j3": 60.0}
_ARM_KP = {
    "L_arm_j1": 2503.5105, "L_arm_j2": 2174.9553, "L_arm_j3": 1889.6868,
    "L_arm_j4": 1059.5746, "L_arm_j5": 659.2767, "L_arm_j6": 210.2241,
    "L_arm_j7": 113.6074,
    "R_arm_j1": 2448.9192, "R_arm_j2": 2174.8486, "R_arm_j3": 1885.5143,
    "R_arm_j4": 1056.1354, "R_arm_j5": 661.4667, "R_arm_j6": 210.5952,
    "R_arm_j7": 113.3994,
}
_ARM_EFFORT = {"L_arm_j[1-6]": 1500.0, "L_arm_j7": 250.0,
               "R_arm_j[1-6]": 1500.0, "R_arm_j7": 250.0}
# Damping scaled to each joint's kp (tau = kd/kp ~ 60ms). The old uniform
# kd=150 gave the low-kp distal joints tracking time constants of 0.7-1.3s -
# they lagged every motion (EE completed only 26-75% of commanded excursions,
# the 2026-07-27 demo-scorecard root cause) while holding perfectly at rest.
_ARM_KD = {n: max(8.0, kp / 16.0) for n, kp in _ARM_KP.items()}

VEGA_1P_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=VEGA_1P_USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            **_TORSO_DEFAULT_POS,
            # "tuck" home (2026-07-27): hands pulled toward the chest so every
            # demo move (fwd 25cm / lat 30cm / up 30cm, ori held) is reachable
            # with 0mm standalone-Pink error - the old extended home had only
            # ~15cm of physical-forward headroom and cost 86-106mm on every
            # excursion. MUST stay in sync with pink_vega_ik.HOME_ARM.
            "L_arm_j1": -0.5,
            "L_arm_j2": 0.30,
            "L_arm_j4": -2.2,
            "L_arm_j5": -0.4,
            "R_arm_j1": 0.5,
            "R_arm_j2": -0.30,
            "R_arm_j4": -2.2,
            "R_arm_j5": 0.4,
            "^(?!torso_j[123]$|[LR]_arm_j[1245]$).*$": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Swerve base is parked (phase-1 is upper-body only). These heavy,
        # offset wheel joints are numerically unstable in this USD and spin up
        # under no drive config we tried (stiffness/damping/armature/limit/
        # velocity-clamp all fail). teleop_vega_pico.py additionally pins the
        # wheel joints to the straight pose each step; this drive + a big
        # armature is the backstop. NOT merely cosmetic (2026-07-28): the
        # instability bleeds into the articulation solve and degrades the
        # wrist joints - welding the wheels (the default) removes it; this
        # free-wheel drive only exists for the VEGA_WELD_WHEELS=0 fallback.
        **({} if WELD_WHEELS else {"wheels": ImplicitActuatorCfg(
            joint_names_expr=["[BLR]_wheel_j[12]"],
            effort_limit_sim=4800.0, stiffness=1.0e4, damping=100.0,
            armature=1.0e3,
        )}),
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
            effort_limit_sim=_ARM_EFFORT, stiffness=_ARM_KP, damping=_ARM_KD,
        ),
    },
)


def make_snap_camera_cfg(width: int = 1280, height: int = 800,
                         update_period: float = 0.2) -> CameraCfg:
    """RGB camera for self-inspection screenshots (teleop_vega_pico --snap-dir).

    update_period 0.2 s: snapshots only need a recent frame, and rendering
    the camera every step was the main real-time-factor killer (0.15x rtf
    measured 2026-07-20 with update_period=0).
    """
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/snap_cam",
        update_period=update_period,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 30.0)),
    )


@configclass
class VegaSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )
    robot: ArticulationCfg = VEGA_1P_CFG
    camera: CameraCfg | None = None
