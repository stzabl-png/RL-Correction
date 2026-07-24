"""Shared Isaac Lab scene definition for the SharpaWave teleop scripts.

IMPORTANT: this module imports isaaclab.* at import time, so it must only be
imported AFTER `AppLauncher(args).app` has been created in the entry script.

Articulation config mirrors sharpa-rl-lab's SharpaWaveEnvCfg.robot_cfg:
same USD (pre-tuned gains), IdealPDActuatorCfg(stiffness=None, damping=None),
same rigid/articulation solver properties.
"""

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Default asset: the rl-lab pre-tuned USD (keep its sibling configuration/ dir!)
RL_LAB_USD = os.path.expanduser(
    "~/luhr/magicsim/Sharpa/sharpa-rl-lab/assets/SharpaWave/right_sharpa_wave.usda"
)


def sharpa_sdk_joint_names(hand: str) -> list:
    """SharpaWave SDK joint order == URDF order (docs/references/02 §2).

    Must match magicdexmate/retarget/mapping.py (ZMQ hello crc guards drift).
    """
    f = []
    f += [f"{hand}_thumb_{j}" for j in ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP")]
    for finger in ("index", "middle", "ring"):
        f += [f"{hand}_{finger}_{j}" for j in ("MCP_FE", "MCP_AA", "PIP", "DIP")]
    f += [f"{hand}_pinky_{j}" for j in ("CMC", "MCP_FE", "MCP_AA", "PIP", "DIP")]
    return f


@configclass
class SharpaSceneCfg(InteractiveSceneCfg):
    """Minimal teleop scene: ground + light + floating fixed-base Sharpa hand."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95))
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RL_LAB_USD,  # override from CLI via scene_cfg.robot.spawn.usd_path
            activate_contact_sensors=False,
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
        ),
        # Palm tilted above the ground (same rotation as rl-lab), fingers free.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.819152, 0.0, -0.5735764, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,  # use gains pre-tuned in the USD (vendor recommendation)
                damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def sine_targets(t: float, joint_names: list, limits: torch.Tensor) -> torch.Tensor:
    """Per-joint staggered raised-cosine flexion within scaled limits (demo motion)."""
    finger_phase = {"thumb": 0.0, "index": 0.1, "middle": 0.2, "ring": 0.3, "pinky": 0.4}
    targets = torch.zeros(len(joint_names), device=limits.device)
    for i, name in enumerate(joint_names):
        lo, hi = limits[i, 0].item(), limits[i, 1].item()
        phase = next((p for f, p in finger_phase.items() if f in name), 0.0)
        s = 0.5 - 0.5 * math.cos(2.0 * math.pi * (t / 4.0 - phase))
        if name.endswith("_AA"):
            center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            targets[i] = center + 0.4 * half * math.sin(2.0 * math.pi * t / 4.0)
        else:  # flexion-type joints: _FE, _PIP, _DIP, _IP, pinky_CMC
            targets[i] = lo + 0.75 * (hi - lo) * s
    return targets
