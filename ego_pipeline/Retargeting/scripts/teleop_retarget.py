#!/usr/bin/env python
"""Realtime retarget loop: glove (wuji|mock) -> dex-retargeting -> ZMQ [-> SAPIEN viz].

Examples (teleop venv):
  # mock glove, publish on :5556, print stats
  .venv/bin/python scripts/teleop_retarget.py --source mock --motion cycle

  # mock glove + SAPIEN window (needs display)
  .venv/bin/python scripts/teleop_retarget.py --source mock --viz

  # real Wuji glove (auto-discovers by handedness; or pass --sn / --address)
  .venv/bin/python scripts/teleop_retarget.py --source wuji --hand right

  # timed headless smoke run
  .venv/bin/python scripts/teleop_retarget.py --source mock --duration 5 --no-pub
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value  # noqa: E402
from magicdexmate.retarget.frames import to_mano  # noqa: E402
from magicdexmate.retarget.mapping import JointMapper  # noqa: E402
from magicdexmate.sources import make_source  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["mock", "wuji"], default="mock")
    p.add_argument("--hand", choices=["right", "left"], default="right")
    p.add_argument("--mode", choices=["vector", "dexpilot"], default="vector")
    p.add_argument("--rate", type=float, default=60.0, help="retarget loop rate [Hz]")
    p.add_argument("--pub", default="tcp://*:5556", help="ZMQ PUB bind address")
    p.add_argument("--no-pub", action="store_true")
    p.add_argument("--viz", action="store_true", help="SAPIEN viewer (needs display)")
    p.add_argument("--duration", type=float, default=0.0, help="exit after N seconds (0 = run forever)")
    p.add_argument("--stale-ms", type=float, default=150.0, help="glove staleness watchdog")
    p.add_argument("--conf-min", type=float, default=0.3, help="min fingertip confidence")
    # mock options
    p.add_argument("--motion", default="cycle", choices=["open", "fist", "wave", "pinch", "cycle"])
    # wuji options
    p.add_argument("--sn", default=None)
    p.add_argument("--address", default=None, help="e.g. 192.168.1.101:50000")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[setup] building retargeting: sharpa_wave {args.hand} {args.mode}")
    retargeting = build_sharpa_retargeting(args.hand, args.mode)
    mapper = JointMapper(retargeting, args.hand)

    src_kwargs = {}
    if args.source == "mock":
        src_kwargs = {"motion": args.motion}
    else:
        src_kwargs = {"sn": args.sn, "address": args.address}
    source = make_source(args.source, args.hand, **src_kwargs)

    publisher = None
    if not args.no_pub:
        from magicdexmate.sinks.qpos_publisher import QposPublisher

        publisher = QposPublisher(args.pub, args.hand, mapper.sdk_names)
        print(f"[setup] publishing qpos on {args.pub} (crc {publisher.crc})")

    viz = None
    if args.viz:
        from magicdexmate.retarget.builder import ASSETS_DIR
        from magicdexmate.sinks.sapien_viz import SapienViz

        urdf = ASSETS_DIR / "sharpa_wave" / args.hand / f"{args.hand}_sharpa_wave.urdf"
        viz = SapienViz(urdf, retargeting.joint_names)

    period = 1.0 / args.rate
    stale_us = args.stale_ms * 1000.0
    rt_ms, lat_ms = [], []
    n_loops = n_retargets = 0
    last_t_us = -1
    last_print = time.time()
    q_sdk = mapper.to_sdk(retargeting.get_qpos())
    kp_mano = None

    source.start()
    print(f"[run] source={args.source} rate={args.rate}Hz  Ctrl-C to stop")
    t_start = time.time()
    try:
        while True:
            loop_t0 = time.perf_counter()
            now_us = time.time_ns() // 1000
            frame = source.get_latest()

            valid = frame is not None and (now_us - frame.t_us) < stale_us \
                and frame.min_fingertip_conf() >= args.conf_min
            if valid and frame.t_us != last_t_us:
                last_t_us = frame.t_us
                kp_mano = to_mano(frame.kp, args.hand)
                ref_value = compute_ref_value(retargeting, kp_mano)
                t0 = time.perf_counter()
                qpos = retargeting.retarget(ref_value)
                rt_ms.append((time.perf_counter() - t0) * 1e3)
                q_sdk = mapper.to_sdk(qpos)
                lat_ms.append((time.time_ns() // 1000 - frame.t_us) / 1e3)
                n_retargets += 1
                if viz is not None and not viz.update(qpos, kp_mano):
                    break

            if publisher is not None:
                publisher.send(
                    q_sdk,
                    t_capture_us=frame.t_us if frame is not None else now_us,
                    valid=bool(valid),
                    wrist_quat=frame.wrist_quat if frame is not None else None,
                )

            n_loops += 1
            if time.time() - last_print > 2.0:
                if rt_ms:
                    a = np.array(rt_ms[-200:])
                    l = np.array(lat_ms[-200:])
                    print(
                        f"\r[stats] loops {n_loops}  retargets {n_retargets}  "
                        f"retarget p50/p95 {np.percentile(a, 50):.2f}/{np.percentile(a, 95):.2f} ms  "
                        f"capture->qpos {np.percentile(l, 50):.1f} ms  valid={valid}",
                        end="", flush=True,
                    )
                else:
                    print(f"\r[stats] loops {n_loops}  waiting for glove frames... valid={valid}", end="", flush=True)
                last_print = time.time()

            if args.duration > 0 and time.time() - t_start > args.duration:
                break
            sleep = period - (time.perf_counter() - loop_t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        source.stop()
        if publisher is not None:
            publisher.close()
        if rt_ms:
            a, l = np.array(rt_ms), np.array(lat_ms)
            print(
                f"[done] {n_retargets} retargets in {time.time() - t_start:.1f}s | "
                f"retarget p50 {np.percentile(a, 50):.2f} ms p95 {np.percentile(a, 95):.2f} ms | "
                f"capture->qpos p50 {np.percentile(l, 50):.1f} ms"
            )
            print(f"[done] last qpos (SDK order, rad): {np.array2string(q_sdk, precision=3, suppress_small=True)}")


if __name__ == "__main__":
    main()
