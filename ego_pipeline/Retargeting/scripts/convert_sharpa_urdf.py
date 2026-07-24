"""Convert the committed SharpaWave URDF -> USD so the Isaac sim scripts can run
without the vendor-only `*.usda` (absent on this machine).

Usage:
  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/convert_sharpa_urdf.py --side right
Output: assets/robots/hands/sharpa_wave/<side>/<side>_sharpa_wave.usd
"""
import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--side", default="right", choices=["right", "left"])
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
side = args.side
urdf = REPO / "assets" / "robots" / "hands" / "sharpa_wave" / side / f"{side}_sharpa_wave.urdf"
out_dir = urdf.parent
assert urdf.exists(), urdf

print("UrdfConverterCfg fields:", list(UrdfConverterCfg.__dataclass_fields__.keys()))

# Gentle PD drive so IdealPDActuator(stiffness=None) finds gains baked in the USD.
cfg_kwargs = dict(
    asset_path=str(urdf),
    usd_dir=str(out_dir),
    usd_file_name=f"{side}_sharpa_wave.usd",
    fix_base=True,
    merge_fixed_joints=True,
    force_usd_conversion=True,
)
# joint drive config name differs across versions; set if present.
fields = UrdfConverterCfg.__dataclass_fields__
if "joint_drive" in fields:
    from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg as U
    JD = U.JointDriveCfg
    cfg_kwargs["joint_drive"] = JD(
        gains=JD.PDGainsCfg(stiffness=20.0, damping=2.0),
        target_type="position",
    )

cfg = UrdfConverterCfg(**cfg_kwargs)
conv = UrdfConverter(cfg)
print("USD written:", conv.usd_path)

simulation_app.close()
