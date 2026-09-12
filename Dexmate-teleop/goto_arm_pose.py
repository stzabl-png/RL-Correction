#!/usr/bin/env python
"""Move a Vega component (arm/torso/head) to a named pose saved in arm_poses.json,
or save the current pose under a new name.

Run from the repo root with the dexcontrol venv (~/Dexmate/dexcontrol/.venv, see SETUP.md)
and ROBOT_NAME set:

    # 列出已存位姿
    ROBOT_NAME=dm/vgd1262ab823-1p ~/Dexmate/dexcontrol/.venv/bin/python \
        goto_arm_pose.py --list

    # 回到已存位姿（会先打印当前/目标角度，回车确认后才动；--scale 越小越慢）
    ROBOT_NAME=dm/vgd1262ab823-1p ~/Dexmate/dexcontrol/.venv/bin/python \
        goto_arm_pose.py right_arm_teleop

    # 把当前位置存成新位姿（只读，不动机器人）
    ROBOT_NAME=dm/vgd1262ab823-1p ~/Dexmate/dexcontrol/.venv/bin/python \
        goto_arm_pose.py --save my_pose --component right_arm

Motion is executed by the robot-side motion plugin (move_to_joint_pos: smoothing,
gravity compensation, collision avoidance). Ctrl+C during motion cancels it.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

POSES_FILE = Path(__file__).resolve().parent / "arm_poses.json"
COMPONENTS = ("left_arm", "right_arm", "torso", "head")


def load_poses() -> dict:
    if POSES_FILE.exists():
        return json.loads(POSES_FILE.read_text())
    return {}


def fmt(vals) -> str:
    return np.array2string(np.asarray(vals, dtype=float), precision=2, separator=", ", max_line_width=200)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pose", nargs="?", help="pose name in arm_poses.json to move to")
    ap.add_argument("--list", action="store_true", help="list saved poses and exit")
    ap.add_argument("--save", metavar="NAME", help="save current pose under NAME (read-only, no motion)")
    ap.add_argument("--component", choices=COMPONENTS, help="component for --save")
    ap.add_argument("--note", default="", help="note stored with --save")
    ap.add_argument("--scale", type=float, default=0.2, help="motion speed scale (0,1], default 0.2 = slow")
    ap.add_argument("--timeout", type=float, default=60.0, help="max seconds to wait for motion")
    ap.add_argument("--yes", action="store_true", help="skip confirmation prompt")
    args = ap.parse_args()

    poses = load_poses()

    if args.list:
        if not poses:
            print(f"no poses in {POSES_FILE}")
        for name, p in poses.items():
            print(f"{name:24s} {p['component']:10s} saved {p.get('saved','?')}  deg={fmt(p['deg'])}  {p.get('note','')}")
        return 0

    if args.save:
        if not args.component:
            ap.error("--save requires --component")
    elif not args.pose:
        ap.error("give a pose name to move to, or --list / --save (see --help)")
    elif args.pose not in poses:
        print(f"pose '{args.pose}' not found in {POSES_FILE}; saved poses: {', '.join(poses) or '(none)'}")
        return 1

    from dexcontrol.robot import Robot  # slow import; after arg validation

    bot = Robot()
    try:
        if args.save:
            comp = getattr(bot, args.component)
            cur = np.array(comp.get_joint_pos())
            poses[args.save] = {
                "component": args.component,
                "rad": [round(float(v), 6) for v in cur],
                "deg": [round(float(v), 2) for v in np.rad2deg(cur)],
                "saved": date.today().isoformat(),
                "note": args.note,
            }
            POSES_FILE.write_text(json.dumps(poses, ensure_ascii=False, indent=2) + "\n")
            print(f"saved '{args.save}' ({args.component}) deg={fmt(np.rad2deg(cur))} -> {POSES_FILE}")
            return 0

        entry = poses[args.pose]
        comp = getattr(bot, entry["component"])
        target = np.array(entry["rad"], dtype=float)
        cur = np.array(comp.get_joint_pos())
        print(f"component : {entry['component']}")
        print(f"current  deg: {fmt(np.rad2deg(cur))}")
        print(f"target   deg: {fmt(entry['deg'])}")
        print(f"max delta deg: {np.max(np.abs(np.rad2deg(cur) - np.array(entry['deg']))):.1f}   speed scale: {args.scale}")
        if not args.yes:
            print("确认运动范围无障碍、e-stop 在手边。回车开始运动，Ctrl+C 取消 > ", end="", flush=True)
            try:
                input()
            except (KeyboardInterrupt, EOFError):
                print("\naborted, no motion sent")
                return 1

        # dexcontrol >=0.5.0: move_to_joint_pos (tracked, returns MotionHandle).
        # (0.5.0rc1 was set_joint_target(..., tracked=True); no alias in 0.5.0.)
        handle = comp.move_to_joint_pos(target, velocity_scale=args.scale)
        try:
            state = handle.wait(timeout=args.timeout)
            print(f"motion finished: {state}")
        except KeyboardInterrupt:
            handle.cancel()
            print("\nCtrl+C -> motion cancelled")
            return 1
        final = np.array(comp.get_joint_pos())
        print(f"final    deg: {fmt(np.rad2deg(final))}  (max err {np.max(np.abs(np.rad2deg(final) - np.array(entry['deg']))):.2f}°)")
        return 0
    finally:
        bot.shutdown()


if __name__ == "__main__":
    sys.exit(main())
