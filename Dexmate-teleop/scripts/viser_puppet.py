#!/usr/bin/env python
"""Viser puppet: drag the human model's wrist/waist gizmos in the browser and
the poses stream onto the teleop wire (same msgpack format as
teleop_pico_producer) so the Isaac consumer follows live.

Doubles as the MANUAL CALIBRATION bench: the axis buttons apply exact 10 cm
single-axis displacements in the raw wire frame (x=right, y=up, -z=front), so
direction/scale/orientation of the mapping can be verified move by move
against the robot's measured response.

Run (viser lives in the main py3.11 venv - no PICO SDK needed):
    .venv/bin/python scripts/viser_puppet.py          # browser: :8080
Consumer:
    .venv-isaac: sim/teleop_vega_pico.py --source zmq --ori-mode hold --anchor home
"""
import argparse
import random
import time

import msgpack
import numpy as np
import viser
import zmq

BUTTON_NAMES = {
    "A": "A", "B": "B", "X": "X", "Y": "Y",
    "left_menu": "left_menu_button", "right_menu": "right_menu_button",
    "left_axis_click": "left_axis_click", "right_axis_click": "right_axis_click",
}
NOISE_M = 3e-4  # freeze detection assumes real sensors always jitter

# wire frame: x=right, y=up, -z=front  |  viser frame: z-up, x=front, y=left
# rows of C are the viser axes expressed in wire coords: viser = C @ wire
C = np.array([[0.0, 0.0, -1.0],
              [-1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0]])

NEUTRAL_WIRE = {  # matches MockPicoSource's neutral so anchors look familiar
    "LWRIST": np.array([-0.25, 1.20, -0.35]),
    "RWRIST": np.array([0.25, 1.20, -0.35]),
    "WAIST": np.array([0.0, 1.25, 0.0]),   # Spine3 (joint 9), same as producer
}
SHOULDER_WIRE = {"L": np.array([-0.18, 1.40, 0.0]),
                 "R": np.array([0.18, 1.40, 0.0])}
HEAD_WIRE = np.array([0.0, 1.60, 0.0])

# SMPL-24 kinematic tree - IDENTICAL to PICO's body estimate / view_skeleton
# (0 Pelvis .. 9 Spine3 .. 18/19 elbows, 20/21 wrists, 22/23 hands)
SMPL_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
               16, 17, 18, 19, 20, 21]
NEUTRAL_BODY24 = np.array([  # wire frame, matches teleop_pico_producer._fake_body24
    [0, 0.95, 0], [-0.09, 0.90, 0], [0.09, 0.90, 0], [0, 1.05, 0],
    [-0.09, 0.50, 0], [0.09, 0.50, 0], [0, 1.15, 0],
    [-0.09, 0.10, 0], [0.09, 0.10, 0], [0, 1.25, 0],
    [-0.09, 0.05, -0.12], [0.09, 0.05, -0.12], [0, 1.45, 0],
    [-0.05, 1.40, 0], [0.05, 1.40, 0], [0, 1.60, 0],
    [-0.18, 1.40, 0], [0.18, 1.40, 0],
    [-0.21, 1.20, -0.17], [0.21, 1.20, -0.17],   # elbows (recomputed live)
    [-0.25, 1.20, -0.35], [0.25, 1.20, -0.35],   # wrists (gizmos)
    [-0.25, 1.15, -0.40], [0.25, 1.15, -0.40],   # hands
])
CONSUMED = [9, 18, 19, 20, 21]  # joints the teleop mapping actually reads
UPPER_ARM_M, FOREARM_M = 0.28, 0.26


def elbow_2bone(shoulder, wrist, side_sign):
    """Anatomical elbow via two-bone IK: fixed bone lengths pin the elbow to a
    circle; the pole vector (down, slightly out/back) picks the natural swivel.
    A midpoint 'elbow' is always collinear -> the arm renders as one rod."""
    sw = wrist - shoulder
    d = np.linalg.norm(sw)
    d = np.clip(d, abs(UPPER_ARM_M - FOREARM_M) + 1e-3,
                UPPER_ARM_M + FOREARM_M - 1e-3)
    u = sw / (np.linalg.norm(sw) + 1e-9)
    a = (UPPER_ARM_M ** 2 - FOREARM_M ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(UPPER_ARM_M ** 2 - a ** 2, 1e-6))
    pole = np.array([0.25 * side_sign, -1.0, 0.15])
    p = pole - (pole @ u) * u
    if np.linalg.norm(p) < 1e-6:
        p = np.array([0.0, 0.0, 1.0]) - u * u[2]
    return shoulder + a * u + h * p / np.linalg.norm(p)


def R_from_qxyzw(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def qxyzw_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4
    return np.array([x, y, z, w])


def wire_pose_from_gizmo(tc):
    """viser TransformControls state -> wire pose7 [x,y,z,qx,qy,qz,qw]."""
    p_wire = C.T @ np.asarray(tc.position)
    w, x, y, z = tc.wxyz
    R_viser = R_from_qxyzw(np.array([x, y, z, w]))
    R_wire = C.T @ R_viser @ C
    return np.concatenate([p_wire, qxyzw_from_R(R_wire)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub", default="tcp://*:5581")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--viser-port", type=int, default=8080)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.viser_port)
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    gizmos = {}
    for name, p_wire in NEUTRAL_WIRE.items():
        gizmos[name] = server.scene.add_transform_controls(
            f"/puppet/{name}", scale=0.18, position=tuple(C @ p_wire),
            wxyz=(1.0, 0.0, 0.0, 0.0))

    def body24_wire():
        """Live SMPL-24 joint positions (wire frame): gizmos drive wrists (20/
        21) + Spine3 (9); elbows (18/19) via two-bone IK; hands (22/23) extend
        along the forearm. Same joint table/topology as the PICO estimate."""
        j = NEUTRAL_BODY24.copy()
        lw = C.T @ np.asarray(gizmos["LWRIST"].position)
        rw = C.T @ np.asarray(gizmos["RWRIST"].position)
        j[9] = C.T @ np.asarray(gizmos["WAIST"].position)
        j[20], j[21] = lw, rw
        j[18] = elbow_2bone(SHOULDER_WIRE["L"], lw, side_sign=-1.0)
        j[19] = elbow_2bone(SHOULDER_WIRE["R"], rw, side_sign=+1.0)
        for h, w, e in ((22, lw, j[18]), (23, rw, j[19])):
            d = w - e
            n = np.linalg.norm(d)
            j[h] = w + (d / n * 0.08 if n > 1e-6 else np.array([0, -0.08, 0]))
        return j

    def draw_skeleton():
        j = body24_wire()
        pts = np.array([[C @ j[i], C @ j[p]]
                        for i, p in enumerate(SMPL_PARENT) if p >= 0])
        server.scene.add_line_segments("/puppet/bones", points=pts,
                                       colors=(230, 180, 60), line_width=4.0)
        server.scene.add_point_cloud(
            "/puppet/consumed", points=np.array([C @ j[i] for i in CONSUMED]),
            colors=(255, 60, 60), point_size=0.025)

    # manual-calibration axis buttons: exact 10 cm steps in the WIRE frame
    with server.gui.add_folder("标定步进(右腕,线框系)"):
        status = server.gui.add_markdown("offset: 0,0,0")
        for label, d in [("右 +10cm", [0.1, 0, 0]), ("左 +10cm", [-0.1, 0, 0]),
                         ("上 +10cm", [0, 0.1, 0]), ("下 +10cm", [0, -0.1, 0]),
                         ("前 +10cm", [0, 0, -0.1]), ("后 +10cm", [0, 0, 0.1])]:
            btn = server.gui.add_button(label)

            def _cb(_, d=np.array(d)):
                tc = gizmos["RWRIST"]
                tc.position = tuple(np.asarray(tc.position) + C @ d)

            btn.on_click(_cb)
        reset = server.gui.add_button("全部回中")

        def _reset(_):
            for n, p in NEUTRAL_WIRE.items():
                gizmos[n].position = tuple(C @ p)
                gizmos[n].wxyz = (1.0, 0.0, 0.0, 0.0)

        reset.on_click(_reset)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(args.pub)
    print(f"[puppet] viser on http://localhost:{args.viser_port}  "
          f"publishing {args.pub} at {args.hz:.0f} Hz - Ctrl-C to stop")

    period = 1.0 / args.hz
    n = 0
    try:
        while True:
            now_ns = time.time_ns()
            trackers = {}
            for name, tc in gizmos.items():
                pose = wire_pose_from_gizmo(tc)
                pose[:3] += np.random.normal(0.0, NOISE_M, 3)
                trackers[name] = [float(v) for v in pose]
            # anatomical synthetic elbows on the wire too, so the consumer's
            # elbow task (--elbow-weight) tracks a plausible elbow, not the
            # collinear midpoint the mock uses
            for tname, gname, sgn in (("LELBOW", "LWRIST", -1.0),
                                      ("RELBOW", "RWRIST", +1.0)):
                w = C.T @ np.asarray(gizmos[gname].position)
                e = elbow_2bone(SHOULDER_WIRE[tname[0]], w, sgn)
                e += np.random.normal(0.0, NOISE_M, 3)
                trackers[tname] = [float(v) for v in e] + [0.0, 0.0, 0.0, 1.0]
            # full SMPL-24 skeleton on the wire, same as producer --body-full
            b24 = body24_wire()
            body24 = [[float(v) for v in b24[i]] + [0.0, 0.0, 0.0, 1.0]
                      for i in range(24)]
            for gi, gname in ((20, "LWRIST"), (21, "RWRIST")):
                body24[gi][3:] = trackers[gname][3:]   # wrist quats from gizmos
            head = list(HEAD_WIRE + np.random.normal(0.0, NOISE_M, 3)) + [0.0, 0.0, 0.0, 1.0]
            msg = {
                "body24": body24,
                "t_us": now_ns // 1000, "device_ts_ns": now_ns,
                "head": [float(v) for v in head],
                "left": [0.0] * 3 + [0.0, 0.0, 0.0, 1.0],
                "right": [0.0] * 3 + [0.0, 0.0, 0.0, 1.0],
                "left_trigger": 0.0, "right_trigger": 0.0,
                "left_grip": 0.0, "right_grip": 0.0,
                "buttons": {k: False for k in BUTTON_NAMES},
                "left_joystick": [0.0, 0.0], "right_joystick": [0.0, 0.0],
                "trackers": trackers,
            }
            sock.send(msgpack.packb(msg, use_single_float=True))
            if n % 7 == 0:
                draw_skeleton()
            if n % 20 == 0:
                off = (C.T @ np.asarray(gizmos["RWRIST"].position)) - NEUTRAL_WIRE["RWRIST"]
                status.content = (f"右腕偏移(线框系 右/上/前): "
                                  f"{off[0]:+.3f} / {off[1]:+.3f} / {-off[2]:+.3f} m")
            n += 1
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[puppet] bye")


if __name__ == "__main__":
    main()
