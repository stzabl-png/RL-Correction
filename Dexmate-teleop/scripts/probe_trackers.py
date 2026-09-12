"""Live tracker identifier: subscribe to the running producer and show which
serial moves when you wave a limb. Run it, then move ONE tracker at a time.

  .venv-pico/bin/python <this> [--sub tcp://127.0.0.1:5581]

For each tracker serial it prints raw position + how far it moved since start;
the one you're waving lights up as MOVING. Note the three serials (left wrist,
right wrist, waist). CPU only, no Isaac.
"""
import argparse
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from magicdexmate.sources.pico_source import PicoZmqSource  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sub", default="tcp://127.0.0.1:5581")
args = ap.parse_args()

src = PicoZmqSource(sub=args.sub)
src.start()
print("waiting for producer frames with trackers...")

origin = {}          # serial -> first-seen position
last_print = 0.0
t0 = time.time()
while True:
    f = src.get_latest()
    if f is not None and f.trackers:
        for sn, p in f.trackers.items():
            if sn not in origin:
                origin[sn] = p[:3].copy()
                print(f"\n[new tracker] {sn}")
        if time.time() - last_print > 0.3:
            last_print = time.time()
            rows = []
            for sn in sorted(f.trackers):
                p = f.trackers[sn][:3]
                moved = float(np.linalg.norm(p - origin[sn]))
                tag = "  <<< MOVING" if moved > 0.08 else ""
                rows.append(f"  {sn}: xyz=[{p[0]:+.2f} {p[1]:+.2f} {p[2]:+.2f}] "
                            f"moved={moved:.2f}m{tag}")
            print("\033[2J\033[H"  # clear screen
                  f"[probe] {len(f.trackers)} tracker(s), {time.time() - t0:.0f}s "
                  "- wave ONE limb to identify it:\n" + "\n".join(rows))
    elif f is not None and time.time() - t0 > 3.0 and not origin:
        if time.time() - last_print > 2.0:
            last_print = time.time()
            print("frames arriving but trackers={} - is Motion Tracker ticked "
                  "in the headset app, and the tracker awake?")
    time.sleep(0.02)
