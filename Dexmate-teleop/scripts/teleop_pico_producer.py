#!/usr/bin/env python
"""PICO -> ZMQ producer: publishes RAW headset/tracker poses (+ controller
channels when controllers happen to be paired - the final rig has NONE, and
nothing downstream may depend on them; they ride along for legacy replays).

Runs in a **py3.10** env (xrobotoolkit_sdk ships a cp310 .so). Two options:
  - GR00T's ready-made venv (zero setup):
      ~/magicsim/GR00T-WholeBodyControl/.venv_teleop/bin/python scripts/teleop_pico_producer.py
  - local .venv-pico (see scripts/setup_pico_env.sh)

Prerequisites: PICO PC Service installed (XRoboToolkit deb), headset + PC on
the same network, XRoboToolkit app running on the headset. Pass --run-service
to have this script start /opt/apps/roboticsservice/runService.sh itself.

No-hardware modes:
  --fake        synthesize frames (MockPicoSource trajectory, wall-clock
                driven, --fake-speed to slow it down) - no SDK import, no PC
                service. Validates the ZMQ wire path end-to-end.
  --record F    append every published msgpack frame to F (raw concatenation
                of the exact bytes sent over ZMQ), replayable offline by
                magicdexmate.sources.pico_source.PicoLogSource.

All frame math (yaw compensation, zeroing, scaling) happens on the CONSUMER
side (sim/teleop_vega_pico.py) - this producer stays dumb on purpose: raw
poses in, raw poses out, so recordings stay reusable and tunables live in one
place.

Message (msgpack dict, ZMQ PUB, no topic frame):
  t_us: int                      producer wall-clock, microseconds
  device_ts_ns: int              SDK device timestamp
  head/left/right: [x,y,z,qx,qy,qz,qw]  raw SDK frames (y-up)
  left_trigger/right_trigger/left_grip/right_grip: float 0..1
  buttons: {A,B,X,Y,left_menu,right_menu,left_axis_click,right_axis_click}: bool
  left_joystick/right_joystick: [x, y]
  trackers: {serial: [x,y,z,qx,qy,qz,qw]}   Swift motion trackers ({} if none)
"""

import argparse
import random
import subprocess
import sys
import time

import msgpack
import zmq

import pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # repo root for magicdexmate.*

BUTTON_NAMES = {
    "A": "A", "B": "B", "X": "X", "Y": "Y",
    "left_menu": "left_menu_button", "right_menu": "right_menu_button",
    "left_axis_click": "left_axis_click", "right_axis_click": "right_axis_click",
}


# PICO full-body 24-joint SMPL indices. The PICO motion trackers stream as this
# fused full-body estimate ("Full Body" mode in the app); we republish the
# joints the consumer uses under fixed names so it needs no serials.
#
# The names are the SMPL joint names, deliberately. The old set (LWRIST /
# RWRIST / WAIST / LELBOW / RELBOW) misled in three separate ways and cost
# real debugging time:
#   - "WAIST" was joint 9 = Spine3, an UPPER-CHEST joint, not the waist. A
#     shoulder fit against it produced "shoulder sits 0.45 m above the waist",
#     which is anatomically absurd and took a while to trace back to the name.
#   - "LELBOW"/"RELBOW" suggest a worn tracker; there is none on the elbow.
#     Those joints are inferred by the body model.
#   - The operator's ankle trackers appear nowhere, so the five published names
#     read like the five worn devices while being a different five things.
# So: joint names describe the JOINT, never the hardware.
SMPL_JOINT = {
    "PELVIS": 0,        # official gear_sonic uses this as the body root
    "SPINE3": 9,        # upper chest - what used to be called WAIST
    "NECK": 12,
    "HEAD": 15,
    "L_SHOULDER": 16, "R_SHOULDER": 17,   # the mapping's arm origin: measured
    "L_ELBOW": 18, "R_ELBOW": 19,         # model-inferred, no tracker there
    "L_WRIST": 20, "R_WRIST": 21,         # anatomical wrist
    "L_HAND": 22, "R_HAND": 23,           # what the official G1 reader uses
    "L_ANKLE": 7, "R_ANKLE": 8,           # where the ankle trackers ride
}
# Both wrist candidates ship every frame now rather than being chosen at record
# time: 20/21 sit where a wrist-worn tracker is, 22/23 are what a hand-mounted
# controller constrains, and which is better is an empirical question the
# recording itself should answer. --wrist-joint only picks the legacy alias.
BODY_JOINT_BY_MODE = {
    "wrist": {"LWRIST": 20, "RWRIST": 21},
    "hand": {"LWRIST": 22, "RWRIST": 23},
}
# Legacy aliases. The consumer defaults, scripts/trackers.env and every
# recording made before 2026-07-31 use these, so they keep being published -
# same values, honest names alongside.
LEGACY_ALIAS = {"WAIST": "SPINE3", "LELBOW": "L_ELBOW", "RELBOW": "R_ELBOW"}
BODY_JOINT = BODY_JOINT_BY_MODE["wrist"]  # overridden by --wrist-joint in main()


def _safe(fn, default):
    # Controller channels only: the final rig runs WITHOUT controllers, and a
    # controller-side SDK hiccup must never kill the tracker stream.
    try:
        return fn()
    except Exception:
        return default


def read_sdk_msg(client, body_full: bool = False) -> dict:
    ts = client.get_timestamp_ns()
    body24 = None
    tracker_motion = {}
    try:  # Swift motion trackers (individual, serial-keyed) - empty in Full Body
        raw = client.get_motion_tracker_data()
        trackers = {sn: list(map(float, d["pose"])) for sn, d in raw.items()}
        # Velocity/acceleration were being dropped on the floor (2026-07-30).
        # They matter: these trackers lose POSITION whenever the headset
        # cameras cannot see them while orientation keeps streaming, and a
        # reported velocity is a far better bridge across that gap than any
        # arc inferred from orientation alone (see pico/arm_fusion.py).
        for sn, d in raw.items():
            vel, acc = d.get("velocity"), d.get("acceleration")
            if vel is None and acc is None:
                continue
            tracker_motion[sn] = {
                "v": list(map(float, vel)) if vel is not None else None,
                "a": list(map(float, acc)) if acc is not None else None,
            }
    except Exception:
        trackers = {}
    try:  # full-body 24-joint estimate -> fixed-name wrist/waist trackers
        body = client.get_body_tracking_data()
        if body is not None and body.get("poses") is not None:
            bp = body["poses"]
            for name, idx in SMPL_JOINT.items():
                if idx < len(bp):
                    trackers[name] = list(map(float, bp[idx]))
            for name, idx in BODY_JOINT.items():      # legacy LWRIST/RWRIST
                trackers[name] = list(map(float, bp[idx]))
            for alias, real in LEGACY_ALIAS.items():
                if real in trackers:
                    trackers[alias] = trackers[real]
            if body_full:  # all 24 joints for the skeleton viewer / debugging
                body24 = [list(map(float, bp[i])) for i in range(len(bp))]
    except Exception:
        pass
    zero7 = [0.0] * 7
    msg = {
        "trackers": trackers,
        # raw-channel velocity/acceleration, keyed by the same serials; absent
        # for body-model entries, which have no such reading
        "tracker_motion": tracker_motion,
        "t_us": time.time_ns() // 1000,
        "device_ts_ns": ts,
        "head": list(map(float, client.get_pose_by_name("headset"))),
        "left": list(map(float, _safe(
            lambda: client.get_pose_by_name("left_controller"), zero7))),
        "right": list(map(float, _safe(
            lambda: client.get_pose_by_name("right_controller"), zero7))),
        "left_trigger": _safe(lambda: client.get_key_value_by_name("left_trigger"), 0.0),
        "right_trigger": _safe(lambda: client.get_key_value_by_name("right_trigger"), 0.0),
        "left_grip": _safe(lambda: client.get_key_value_by_name("left_grip"), 0.0),
        "right_grip": _safe(lambda: client.get_key_value_by_name("right_grip"), 0.0),
        "buttons": {k: _safe(lambda n=v: client.get_button_state_by_name(n), False)
                    for k, v in BUTTON_NAMES.items()},
        "left_joystick": list(map(float, _safe(
            lambda: client.get_joystick_state("left"), [0.0, 0.0]))),
        "right_joystick": list(map(float, _safe(
            lambda: client.get_joystick_state("right"), [0.0, 0.0]))),
    }
    if body24 is not None:
        msg["body24"] = body24
    return msg


def _fake_body24(f) -> list:
    """Crude stick-figure 24-joint skeleton around the mock wrists (y-up raw
    frame) so the skeleton viewer can be exercised without hardware."""
    def j(x, y, z, quat=(0.0, 0.0, 0.0, 1.0)):
        return [float(x), float(y), float(z), *map(float, quat)]

    lw, rw = f.left, f.right
    lsh, rsh = (-0.18, 1.40, 0.0), (0.18, 1.40, 0.0)
    le = [(lsh[i] + lw[i]) / 2 for i in range(3)]
    re = [(rsh[i] + rw[i]) / 2 for i in range(3)]
    return [
        j(0, 0.95, 0), j(-0.09, 0.90, 0), j(0.09, 0.90, 0), j(0, 1.05, 0),
        j(-0.09, 0.50, 0), j(0.09, 0.50, 0), j(0, 1.15, 0),
        j(-0.09, 0.10, 0), j(0.09, 0.10, 0), j(0, 1.25, 0),
        j(-0.09, 0.05, -0.12), j(0.09, 0.05, -0.12), j(0, 1.45, 0),
        j(-0.05, 1.40, 0), j(0.05, 1.40, 0), j(*f.head[:3]),
        j(*lsh), j(*rsh), j(*le), j(*re),
        j(*lw[:3], quat=lw[3:7]), j(*rw[:3], quat=rw[3:7]),
        j(lw[0], lw[1] - 0.05, lw[2] - 0.05),
        j(rw[0], rw[1] - 0.05, rw[2] - 0.05),
    ]


FAKE_NOISE_M = 3e-4  # freeze detection assumes real sensors never repeat
                     # values bit-for-bit; noiseless demo holds would trip it


def _jitter3(pose):
    return [float(pose[i] + random.gauss(0.0, FAKE_NOISE_M)) for i in range(3)] \
        + [float(v) for v in pose[3:]]


def read_fake_msg(mock, t: float, body_full: bool = False) -> dict:
    f = mock.sample_at(t)
    now_ns = time.time_ns()
    msg_body = {"body24": [_jitter3(j) for j in _fake_body24(f)]} if body_full else {}
    return {
        **msg_body,
        "t_us": now_ns // 1000,
        "device_ts_ns": now_ns,
        "head": _jitter3(f.head),
        "left": [float(v) for v in f.left],
        "right": [float(v) for v in f.right],
        "left_trigger": float(f.left_trigger),
        "right_trigger": float(f.right_trigger),
        "left_grip": float(f.left_grip),
        "right_grip": float(f.right_grip),
        "buttons": {k: False for k in BUTTON_NAMES},
        "left_joystick": [0.0, 0.0],
        "right_joystick": [0.0, 0.0],
        "trackers": {sn: _jitter3(p) for sn, p in f.trackers.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pub", default="tcp://*:5581", help="ZMQ PUB bind address")
    ap.add_argument("--hz", type=float, default=100.0, help="publish rate")
    ap.add_argument("--run-service", action="store_true",
                    help="spawn /opt/apps/roboticsservice/runService.sh")
    ap.add_argument("--fake", action="store_true",
                    help="synthesize frames instead of reading the SDK")
    ap.add_argument("--fake-speed", type=float, default=1.0,
                    help="time scale for the fake trajectory (<1 = slower; "
                         "use ~0.05 against a slower-than-realtime sim)")
    ap.add_argument("--fake-motion", choices=["circle", "demo"], default="circle",
                    help="fake trajectory: circle = the classic orbit+roll; "
                         "demo = legible pose-hold sequence (forward push / "
                         "lateral raise / overhead, 3s holds, no rotation) "
                         "for eyeballing skeleton<->robot mapping consistency")
    ap.add_argument("--control", default="",
                    help="开关频道地址,例如 tcp://127.0.0.1:5584。给了就订阅它,\n                          按页面的录制开关把 PICO 数据分段写进 data/sessions/")
    ap.add_argument("--record", default=None, metavar="FILE",
                    help="append published msgpack frames to FILE (PicoLogSource replays it)")
    ap.add_argument("--wrist-joint", choices=["wrist", "hand"], default="wrist",
                    help="which body joints become LWRIST/RWRIST: wrist=20/21 "
                         "(anatomical, where the worn trackers sit - default "
                         "for the no-controller rig), hand=22/23 (official G1 "
                         "choice, only right when controllers are held)")
    ap.add_argument("--body-full", action="store_true",
                    help="also publish all 24 body joints as 'body24' (for "
                         "scripts/view_skeleton.py - see what the PICO body "
                         "estimate thinks you are doing; ~1.5KB/frame extra)")
    ap.add_argument("--print-every", type=float, default=2.0,
                    help="status print period [s], 0 = quiet")
    args = ap.parse_args()

    global BODY_JOINT
    BODY_JOINT = BODY_JOINT_BY_MODE[args.wrist_joint]

    service = None
    if args.run_service:
        service = subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
        print(f"[pico] PC service spawned (pid {service.pid}); waiting 2 s")
        time.sleep(2.0)

    if args.fake:
        from magicdexmate.sources.pico_source import MockPicoSource

        client = None
        mock = MockPicoSource.demo() if args.fake_motion == "demo" \
            else MockPicoSource()
        print(f"[pico] FAKE mode ({args.fake_motion}, speed x{args.fake_speed:g})"
              " - no SDK, no headset")
    else:
        from magicdexmate.pico.xr_client import XrClient

        client, mock = XrClient(), None

    seg = None
    if args.control:
        import types
        from magicdexmate.control_link import ControlSubscriber
        seg = types.SimpleNamespace(
            ctl=ControlSubscriber(args.control), fh=None, n=0)
        print(f"[pico] 订阅开关频道 {args.control},按段落盘")
    rec = open(args.record, "ab") if args.record else None
    if rec is not None:
        print(f"[pico] recording frames to {args.record}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(args.pub)
    print(f"[pico] publishing on {args.pub} at {args.hz:.0f} Hz - Ctrl-C to stop")

    period = 1.0 / args.hz
    n_sent, last_print, last_ts = 0, time.time(), -1
    t_start = time.time()
    try:
        while True:
            t0 = time.time()
            if args.fake:
                msg = read_fake_msg(mock, (t0 - t_start) * args.fake_speed,
                                    body_full=args.body_full)
            else:
                msg = read_sdk_msg(client, body_full=args.body_full)
            packed = msgpack.packb(msg)
            sock.send(packed)
            if rec is not None:
                rec.write(packed)
            # 按录制分段落盘。--record 那一份是整场一个文件(留着不动,它是
            # 排障用的全程记录),这一份跟着页面的录制开关走,和手臂指令、
            # 手部数据、真机关节写进同一个目录。
            if seg is not None:
                st = seg.ctl.latest(time.time()) or {}
                on = bool(st.get("recording") and st.get("session"))
                if on and seg.fh is None:
                    d = ROOT / "data" / "sessions" / str(st["session"])
                    d.mkdir(parents=True, exist_ok=True)
                    seg.fh = open(d / "pico.msgpack", "ab")
                    seg.n = 0
                    print(f"\n[pico] 开始按段录制 -> {d.name}")
                elif not on and seg.fh is not None:
                    seg.fh.close()
                    seg.fh = None
                    print(f"\n[pico] 本段结束,共 {seg.n} 帧")
                if seg.fh is not None:
                    seg.fh.write(packed)
                    seg.n += 1
            n_sent += 1

            if args.print_every > 0 and time.time() - last_print > args.print_every:
                ts = msg["device_ts_ns"]
                fresh = "fresh" if ts != last_ts else "STALE (headset app running?)"
                h = msg["head"][:3]
                print(f"\r[pico] sent={n_sent} device_ts={fresh} "
                      f"head=[{h[0]:+.2f} {h[1]:+.2f} {h[2]:+.2f}] "
                      f"grips L{msg['left_grip']:.1f}/R{msg['right_grip']:.1f} "
                      f"trk={len(msg.get('trackers', {}))}",
                      end="", flush=True)
                last_print, last_ts = time.time(), ts

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[pico] stopped")
    finally:
        if rec is not None:
            rec.close()
            print(f"[pico] recorded {n_sent} frames -> {args.record}")
        if client is not None:
            client.close()
        if service is not None:
            subprocess.Popen(["kill", "-9", str(service.pid)])


if __name__ == "__main__":
    main()
