#!/usr/bin/env python
"""Live PICO body-skeleton viewer - see what the body estimate THINKS you do.

The 2026-07-26 session showed the full-body estimate freezing solid and
hallucinating arm motion while everything looked "connected". This viewer
renders the estimate itself so those failures are visible at a glance:

  - green skeleton: values updating normally
  - RED title + frozen skeleton: values bit-identical > 0.5s (estimator dead)
  - orange wrist markers: the LWRIST/RWRIST/WAIST joints the teleop consumes

Run alongside the stack (producer must publish the joints):
  producer:  ... teleop_pico_producer.py --body-full
  viewer :   .venv-pico/bin/python scripts/view_skeleton.py

Replay a recording made with --body-full:
  .venv-pico/bin/python scripts/view_skeleton.py --file logs/xxx.msgpack

Falls back to drawing just the 3 synthesized trackers when 'body24' is not
in the stream (producer started without --body-full).
"""
import argparse
import sys
import time

import msgpack
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

# SMPL kinematic tree (24 joints; -1 = root). Index table: see PICO_teleop.md.
PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
          18, 19, 20, 21]
NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
         "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
         "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
         "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"]
USED = {20: "LWRIST", 21: "RWRIST", 9: "WAIST"}   # joints the teleop consumes
FREEZE_S = 0.5


def zmq_frames(sub_addr):
    import zmq

    ctx = zmq.Context()
    s = ctx.socket(zmq.SUB)
    s.connect(sub_addr)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVTIMEO, 100)
    while True:
        try:
            raw = s.recv()
            # drain to the freshest frame: matplotlib renders ~15 fps while
            # the producer sends 100 Hz - without dropping the backlog the
            # displayed pose lags further behind every second (10+ s seen)
            while True:
                try:
                    raw = s.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break
            yield msgpack.unpackb(raw)
        except Exception:
            yield None


def file_frames(path, rate):
    unp = msgpack.Unpacker(open(path, "rb"))
    frames = list(unp)
    if not frames:
        raise SystemExit(f"empty recording: {path}")
    t0_msg = frames[0]["t_us"]
    t0 = time.time()
    for d in frames:
        target = (d["t_us"] - t0_msg) / 1e6 / rate
        dt = t0 + target - time.time()
        if dt > 0:
            time.sleep(min(dt, 0.5))
        yield d
    print("[viewer] recording finished")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sub", default="tcp://127.0.0.1:5581")
    ap.add_argument("--file", default=None, help="replay a --record file")
    ap.add_argument("--rate", type=float, default=1.0, help="replay speed")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.ion()
    plt.show(block=False)

    src = file_frames(args.file, args.rate) if args.file else zmq_frames(args.sub)
    last_key, last_change, msg_t = None, time.time(), None
    for d in src:
        if d is None:
            plt.pause(0.05)
            continue
        body = d.get("body24")
        trk = d.get("trackers") or {}
        key = msgpack.packb(body if body is not None else trk)
        now = time.time()
        if key != last_key:
            last_key, last_change = key, now
        frozen = (now - last_change) > FREEZE_S
        msg_t = d.get("t_us")

        ax.cla()
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2)
        ax.set_xlabel("x"); ax.set_ylabel("z(fwd raw -z)"); ax.set_zlabel("y up")
        col = "red" if frozen else "tab:green"
        if body is not None:
            P = np.asarray(body, dtype=float)[:, :3]
            # raw y-up frame -> plot as (x, -z, y) so up is up, forward is +y
            X, Y, Z = P[:, 0], -P[:, 2], P[:, 1] + 1.0
            for i, par in enumerate(PARENT):
                if par >= 0:
                    ax.plot([X[i], X[par]], [Y[i], Y[par]], [Z[i], Z[par]],
                            c=col, lw=2)
            ax.scatter(X, Y, Z, c=col, s=10)
            for i, name in USED.items():
                ax.scatter([X[i]], [Y[i]], [Z[i]], c="orange", s=60)
                ax.text(X[i], Y[i], Z[i], name, fontsize=8)
        else:
            for name, p in trk.items():
                p = np.asarray(p, dtype=float)
                ax.scatter([p[0]], [-p[2]], [p[1] + 1.0], c="orange", s=60)
                ax.text(p[0], -p[2], p[1] + 1.0, name, fontsize=9)
        state = "FROZEN - estimator dead (shake trackers / look at wrists)" \
            if frozen else "live"
        ax.set_title(f"PICO body estimate: {state}\n"
                     f"{'24-joint skeleton' if body is not None else 'trackers only (producer without --body-full)'}",
                     color=col)
        plt.pause(0.001)


if __name__ == "__main__":
    main()
