#!/usr/bin/env python
"""Dual-hand retarget producer: BOTH gloves -> retarget -> two ZMQ streams, one process.

Same Mode-A bridge as scripts/teleop_retarget.py, but runs the right and left
pipelines side by side in a single process and publishes each on its own port
(right -> :5556, left -> :5557, matching sim/teleop_dual_sharpa.py's defaults).
Lets you drive the dual-hand sim/real consumer from TWO terminals instead of three:

  # terminal 1 (this dual producer; both gloves)
  PYTHONPATH= .venv/bin/python scripts/teleop_retarget_dual.py --source wuji
  # terminal 2 (dual-hand Isaac consumer)
  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_dual_sharpa.py

Self-test without gloves (both hands cycle):
  .venv/bin/python scripts/teleop_retarget_dual.py --source mock --motion cycle --duration 5

Real dual-hand SharpaWave (consumer side, terminal 2):
  LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib PYTHONPATH= .venv/bin/python \
      scripts/sharpa_real_runner.py --hand both --sub tcp://127.0.0.1:5556 --sub-left tcp://127.0.0.1:5557

Each per-hand pipeline is byte-for-byte the single-hand path; the only addition
is a distinct wuji device_name per hand so two gloves share one SdkManager singleton.
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

HANDS = ("right", "left")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["mock", "wuji"], default="mock")
    # Default flipped to vector (fj, 2026-07-03), matches teleop_retarget.py.
    p.add_argument("--mode", choices=["vector", "dexpilot"], default="vector")
    p.add_argument("--abduction-weight", type=float, default=1.0,
                   help="vector mode: per-finger abduction vector weight (Step 4.1); 0 == baseline.")
    p.add_argument("--pinch-weight", type=float, default=0.0,
                   help="vector mode: thumb->finger pinch vector weight (Step 5); 0 == baseline, higher = crisper 对指.")
    p.add_argument(
        "--relax-distal", default=None, action=argparse.BooleanOptionalAction,
        help="relax DexPilot's redundant thumb_IP/pinky_DIP over-curl on the open hand "
             "(pinch opposition unchanged); default on for dexpilot, off for vector",
    )
    p.add_argument("--thumb-ip-margin", type=float, default=0.16,
                   help="Step 5c: cap robot thumb_IP at operator flexion + margin (rad), even "
                        "during a pinch, so the thumb tip never over-bends. Default 0.16 (tuned "
                        "on real glove). 0 = exact match.")
    p.add_argument("--rate", type=float, default=60.0, help="retarget loop rate [Hz] (both hands per loop)")
    p.add_argument("--pub-right", default="tcp://*:5556", help="right-hand ZMQ PUB bind")
    p.add_argument("--pub-left", default="tcp://*:5557", help="left-hand ZMQ PUB bind")
    p.add_argument("--no-pub", action="store_true")
    p.add_argument("--duration", type=float, default=0.0, help="exit after N seconds (0 = run forever)")
    p.add_argument("--stale-ms", type=float, default=150.0, help="glove staleness watchdog")
    p.add_argument("--conf-min", type=float, default=0.3, help="min fingertip confidence")
    # mock options (both hands use the same motion)
    p.add_argument("--motion", default="cycle", choices=["open", "fist", "wave", "pinch", "cycle"])
    # wuji options (per hand; default = auto-connect by handedness)
    p.add_argument("--sn-right", default=None)
    p.add_argument("--sn-left", default=None)
    p.add_argument("--address-right", default=None, help="e.g. 192.168.1.101:50000")
    p.add_argument("--address-left", default=None, help="e.g. 192.168.1.100:50000")
    return p.parse_args()


class HandWorker:
    """One hand's full pipeline: source -> retarget -> SDK mapping -> ZMQ publish.

    Mirrors scripts/teleop_retarget.py's per-step logic exactly, one instance per hand.
    """

    def __init__(self, hand: str, args):
        self.hand = hand
        self.do_relax = args.relax_distal if args.relax_distal is not None else (args.mode == "dexpilot")
        self.thumb_ip_margin = args.thumb_ip_margin

        self.retargeting = build_sharpa_retargeting(
            hand, args.mode, abduction_weight=args.abduction_weight, pinch_weight=args.pinch_weight
        )
        self.mapper = JointMapper(self.retargeting, hand)

        if args.source == "mock":
            src_kwargs = {"motion": args.motion}
        else:
            src_kwargs = {
                "sn": getattr(args, f"sn_{hand}"),
                "address": getattr(args, f"address_{hand}"),
                "device_name": f"glove_{hand}",  # distinct handle: two gloves, one SdkManager
            }
        self.source = make_source(args.source, hand, **src_kwargs)

        self.publisher = None
        if not args.no_pub:
            from magicdexmate.sinks.qpos_publisher import QposPublisher

            bind = getattr(args, f"pub_{hand}")
            self.publisher = QposPublisher(bind, hand, self.mapper.sdk_names)
            print(f"[setup] {hand}: publishing qpos on {bind} (crc {self.publisher.crc})")

        self.q_sdk = self.mapper.to_sdk(self.retargeting.get_qpos())
        self.last_t_us = -1
        self.valid = False
        self.rt_ms: list[float] = []
        self.n_retargets = 0

    def start(self):
        self.source.start()

    def step(self, now_us: int, stale_us: float, conf_min: float):
        frame = self.source.get_latest()
        self.valid = frame is not None and (now_us - frame.t_us) < stale_us \
            and frame.min_fingertip_conf() >= conf_min
        if self.valid and frame.t_us != self.last_t_us:
            self.last_t_us = frame.t_us
            kp_mano = to_mano(frame.kp, self.hand)
            ref_value = compute_ref_value(self.retargeting, kp_mano)
            t0 = time.perf_counter()
            qpos = self.retargeting.retarget(ref_value)
            self.rt_ms.append((time.perf_counter() - t0) * 1e3)
            q_sdk = self.mapper.to_sdk(qpos)
            if self.do_relax:
                q_sdk = self.mapper.relax_distal(q_sdk, frame.kp, thumb_ip_margin=self.thumb_ip_margin)
            self.q_sdk = q_sdk
            self.n_retargets += 1

        if self.publisher is not None:
            self.publisher.send(
                self.q_sdk,
                t_capture_us=frame.t_us if frame is not None else now_us,
                valid=bool(self.valid),
                wrist_quat=frame.wrist_quat if frame is not None else None,
            )

    def stop(self):
        self.source.stop()
        if self.publisher is not None:
            self.publisher.close()


def main():
    args = parse_args()

    print(f"[setup] building dual retargeting: sharpa_wave {args.mode} (right + left)")
    workers = [HandWorker(h, args) for h in HANDS]
    if workers[0].do_relax:
        print("[setup] distal relax ON (thumb_IP/pinky_DIP follow the hand when not pinching)")
    if args.source == "wuji":
        print("[setup] wuji: need BOTH gloves online (left 192.168.1.100, right 192.168.1.101); "
              "a missing glove just leaves its sim hand at home (valid=False)")

    period = 1.0 / args.rate
    stale_us = args.stale_ms * 1000.0
    n_loops = 0
    last_print = time.time()

    for w in workers:
        w.start()
    print(f"[run] source={args.source} rate={args.rate}Hz  Ctrl-C to stop")
    t_start = time.time()
    try:
        while True:
            loop_t0 = time.perf_counter()
            now_us = time.time_ns() // 1000
            for w in workers:
                w.step(now_us, stale_us, args.conf_min)

            n_loops += 1
            if time.time() - last_print > 2.0:
                parts = []
                for w in workers:
                    if w.rt_ms:
                        a = np.array(w.rt_ms[-200:])
                        parts.append(
                            f"{w.hand[0].upper()} valid={int(w.valid)} "
                            f"rt p50/p95 {np.percentile(a, 50):.2f}/{np.percentile(a, 95):.2f}ms (n{w.n_retargets})"
                        )
                    else:
                        parts.append(f"{w.hand[0].upper()} valid={int(w.valid)} waiting...")
                print(f"\r[stats] loops {n_loops}  " + "  |  ".join(parts), end="", flush=True)
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
        for w in workers:
            w.stop()
        for w in workers:
            if w.rt_ms:
                a = np.array(w.rt_ms)
                print(
                    f"[done] {w.hand}: {w.n_retargets} retargets in {time.time() - t_start:.1f}s | "
                    f"retarget p50 {np.percentile(a, 50):.2f} ms p95 {np.percentile(a, 95):.2f} ms | "
                    f"last qpos (SDK, rad): {np.array2string(w.q_sdk, precision=3, suppress_small=True)}"
                )


if __name__ == "__main__":
    main()
