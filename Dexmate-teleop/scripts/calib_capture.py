"""Side-tap the producer ZMQ stream and record wrist positions in the SAME
mapped frame the consumer uses (process_xr_pose vs the live waist), for
calibration analysis. Run from MagicDexMate root with .venv-pico python."""
import sys
import time

sys.path.insert(0, ".")
import msgpack  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402

from magicdexmate.pico.xr_pose import mat_to_pos_quat_wxyz, process_xr_pose  # noqa: E402

OUT = sys.argv[1]
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect("tcp://127.0.0.1:5581")
s.setsockopt(zmq.SUBSCRIBE, b"")

t0 = time.time()
rows = []
while time.time() - t0 < DUR:
    try:
        d = msgpack.unpackb(s.recv(flags=zmq.NOBLOCK))
    except zmq.Again:
        time.sleep(0.002)
        continue
    trk = d.get("trackers") or {}
    if all(k in trk for k in ("LWRIST", "RWRIST", "WAIST")):
        w = np.asarray(trk["WAIST"], dtype=float)
        pl, _ = mat_to_pos_quat_wxyz(process_xr_pose(np.asarray(trk["LWRIST"], dtype=float), w))
        pr, _ = mat_to_pos_quat_wxyz(process_xr_pose(np.asarray(trk["RWRIST"], dtype=float), w))
        rows.append([time.time() - t0, *pl, *pr])
arr = np.array(rows)
np.save(OUT, arr)
print(f"captured {len(rows)} frames over {DUR}s -> {OUT}")
