"""Build one configured Sweep Part-4 reference with the validated Sweep2 builder."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--take", required=True, choices=("32", "80", "128", "180"))
    parser.add_argument("--geometry_only", action="store_true")
    parser.add_argument("--audit_side", choices=("left", "right"))
    parser.add_argument("--audit_row", type=int)
    parser.add_argument("--audit_seeds", type=int, default=96)
    parser.add_argument("--scene_yaw_deg", type=float)
    parser.add_argument("--broom_start_x", type=float)
    parser.add_argument("--broom_start_y", type=float)
    parser.add_argument("--broom_prior")
    args = parser.parse_args()
    config = json.loads((CONFIG_DIR / f"take_{args.take}.json").read_text())
    broom = args.broom_prior or config["grasppose"]["broom"]["prior"]
    dustpan = config["grasppose"]["dustpan"]["prior"]
    command = [
        sys.executable,
        str(ROOT / "tasks/Sweep/2/A_Design/L2_Reference/build_reference.py"),
        "--data", config["data_dir"],
        "--task_name", config["task_name"],
        "--rts_stem", config["rts_stem"],
        "--cube_half_m", str(config["cube_half_m"]),
        "--max_nominal_brush_distance_m", str(config["max_nominal_brush_distance_m"]),
        "--broom_prior", broom,
        "--pan_prior", dustpan,
        "--output", config["reference"],
        "--scene_yaw_deg", str(args.scene_yaw_deg if args.scene_yaw_deg is not None
                                else config.get("scene_yaw_deg", -14.0)),
        "--broom_start_x", str(args.broom_start_x if args.broom_start_x is not None
                                else config.get("broom_start_x", -0.116)),
        "--broom_start_y", str(args.broom_start_y if args.broom_start_y is not None
                                else config.get("broom_start_y", -0.177)),
    ]
    if config.get("pan_semantic_frame", False):
        command.append("--pan_semantic_frame")
    if args.geometry_only:
        command.append("--geometry_only")
    if args.audit_side:
        command.extend(("--audit_side", args.audit_side))
    if args.audit_row is not None:
        command.extend(("--audit_row", str(args.audit_row)))
        command.extend(("--audit_seeds", str(args.audit_seeds)))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
