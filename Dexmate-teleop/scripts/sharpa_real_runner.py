#!/usr/bin/env python
"""Distributed real-hand receiver: ZMQ qpos stream -> real SharpaWave hand(s).

The receiving half of bridged teleop (the producer is scripts/teleop_retarget.py
--source wuji, which publishes 22-DoF qpos over ZMQ). One or both hands.

Single hand:
  # terminal 1 (teleop .venv): producer
  PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand right
  # terminal 2: this receiver (SDK libs on LD_LIBRARY_PATH)
  LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib PYTHONPATH= .venv/bin/python \
      scripts/sharpa_real_runner.py --hand right --sub tcp://127.0.0.1:5556

Both hands (two producers on two ports + one receiver):
  PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand right --pub tcp://*:5556
  PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand left  --pub tcp://*:5557
  LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib PYTHONPATH= .venv/bin/python \
      scripts/sharpa_real_runner.py --hand both --sub tcp://127.0.0.1:5556 --sub-left tcp://127.0.0.1:5557

SAFETY: starts DISENGAGED. Keys while running: [e] engage  [w] freeze  [q] home+freeze
[x] stop & quit  [r] record start/stop (only with --record). Watchdog freezes a hand
if its qpos stream goes stale. Joint targets are limit-clipped and per-step
rate-limited inside SharpaRealHand.

Episode recording (--record DIR): per episode and hand one HDF5 with the joint
stream (target + measured qpos each control tick) and the fingertip tactile
stream (f6/deform/raw at 30 Hz); replay with scripts/replay_hand_viser.py.
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from magicdexmate.sinks.sharpa_real import SharpaRealHand  # noqa: E402


class QposSub:
    """Non-blocking SUB of one teleop qpos stream; keeps the newest valid frame."""

    def __init__(self, addr: str, hand: str):
        import zmq

        self._zmq = zmq
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.connect(addr)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.hand = hand
        self.crc = None
        self.qpos = None
        self.valid = False
        self.t_recv = 0.0  # local monotonic time of last fresh valid frame
        self.t_capture_us = 0  # producer-side glove capture time of the last frame

    def poll(self):
        while True:
            try:
                msg = json.loads(self.sock.recv_string(flags=self._zmq.NOBLOCK))
            except self._zmq.Again:
                return
            if msg.get("hand") != self.hand:
                continue
            if msg["type"] == "hello":
                self.crc = msg["crc"]
            elif msg["type"] == "qpos":
                if self.crc is not None and msg["crc"] != self.crc:
                    continue  # producer restarted with a different joint order
                self.qpos = np.asarray(msg["qpos"], dtype=float)
                self.valid = bool(msg["valid"])
                self.t_capture_us = int(msg.get("t_capture_us", 0))
                if self.valid:
                    self.t_recv = time.monotonic()


class EpisodeRecorder:
    """Optional episode recording (--record DIR): joints every control tick +
    a 30 Hz tactile fetch thread, written per hand to HDF5 by HandDataWriter.

    Inactive (or --record not given) it costs the control loop exactly one
    attribute check per tick. Press 'r' to start/stop an episode.
    """

    def __init__(self, out_dir, hands, subs, tactile_types, tactile_hz, skip_calib):
        from magicdexmate.recording import HandDataWriter  # noqa: F401 - fail fast if h5py missing

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.hands = hands
        self.subs = subs
        self.tactile_types = tactile_types
        self.tactile_hz = tactile_hz
        self.skip_calib = skip_calib
        self.session = ""      # 页面广播来的本次录制目录名;空 = 自己编号
        self.tactile_delta = False
        self.active = False
        self.writers = {}      # hand -> HandDataWriter
        self.tactile_on = {}   # hand -> bool (device reported has_fingertip_tactile)
        self.last_f6 = {}      # hand -> (5,) |F| for the status line
        self._calibrated = set()
        self._thread = None
        self._stop = None

    def _next_episode_dir(self):
        # 页面给了本次录制的目录名就用它,手臂那一路写的是同一个目录。此前
        # 两边各自扫自己的目录编号(logs/episodes/ 与 data/teleop/),互不知道
        # 对方录到第几段,已经错开了一位 —— 而合并录制的前提是同一段数据在
        # 同一个地方。页面没给名字(单独跑 runner)才退回自己编号。
        if self.session:
            ep = self.out_dir / self.session
            ep.mkdir(parents=True, exist_ok=True)
            return self.session, ep
        idx = -1
        for d in self.out_dir.glob("episode_*"):
            try:
                idx = max(idx, int(d.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        name = f"episode_{idx + 1:04d}"
        ep = self.out_dir / name
        ep.mkdir()
        return name, ep

    def toggle(self):
        if self.active:
            self.stop_episode()
        else:
            self.start_episode()

    def start_episode(self):
        import threading

        from magicdexmate.recording import HandDataWriter

        name, ep_dir = self._next_episode_dir()
        for rh in self.hands:
            self.tactile_on[rh.hand] = bool(self.tactile_types) and rh.has_tactile()
            if self.tactile_types and not self.tactile_on[rh.hand]:
                print(f"\n[record] {rh.hand}: firmware reports NO fingertip tactile -> joints only")
            if self.tactile_on[rh.hand] and not self.skip_calib and rh.hand not in self._calibrated:
                print(f"\n[record] {rh.hand}: tactile zero-calibration - keep fingertips CONTACT-FREE...")
                if rh.calib_tactile():
                    self._calibrated.add(rh.hand)
            self.writers[rh.hand] = HandDataWriter(
                ep_dir / f"{name}_{rh.hand}.h5", rh.hand,
                joint_names=_sdk_joint_names(rh.hand),
                tactile_types=self.tactile_types if self.tactile_on[rh.hand] else (),
                extra_attrs={"source": "sharpa_real_runner"},
                tactile_delta=self.tactile_delta,
            ).start()
        self._stop = threading.Event()
        if any(self.tactile_on.values()):
            self._thread = threading.Thread(target=self._tactile_loop, daemon=True)
            self._thread.start()
        self.active = True
        print(f"\n[record] episode {name} STARTED (press 'r' to stop)")

    def log_joint(self, rh, sub):
        w = self.writers.get(rh.hand)
        if w is None or sub.qpos is None:
            return
        try:
            measured = rh.current_angles()
        except Exception:  # noqa: BLE001 - a failed read must not kill the loop
            measured = np.full(22, np.nan)
        w.append_joint(time.time(), sub.qpos, measured, sub.valid, rh._engaged, sub.t_capture_us)

    def _tactile_loop(self):
        period = 1.0 / self.tactile_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            for rh in self.hands:
                if not self.tactile_on.get(rh.hand):
                    continue
                try:
                    tac = rh.fetch_tactile(self.tactile_types)
                except Exception as e:  # noqa: BLE001 - keep the thread alive
                    print(f"\n[record] {rh.hand} tactile fetch error: {e}")
                    continue
                if "f6" in tac:
                    self.last_f6[rh.hand] = np.linalg.norm(tac["f6"][:, :3], axis=1)
                w = self.writers.get(rh.hand)
                if w is not None:
                    w.append_tactile(tac)
            sleep = period - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

    def stop_episode(self):
        self.active = False
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for hand, w in self.writers.items():
            w.stop()
        self.writers.clear()
        print("[record] episode STOPPED")

    def status(self):
        if not self.active:
            return "rec:off"
        peaks = "  ".join(
            f"{h}|F|max {np.max(f):.2f}N" for h, f in self.last_f6.items()) or "no tactile"
        return f"rec:ON {peaks}"


def _sdk_joint_names(hand):
    from magicdexmate.retarget.mapping import sharpa_sdk_joint_names

    return sharpa_sdk_joint_names(hand)


class KeyPoller:
    """Non-blocking single-key reader from a terminal (no-op if stdin isn't a tty)."""

    def __enter__(self):
        self.tty = sys.stdin.isatty()
        if self.tty:
            import termios
            import tty as _tty

            self.fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self.fd)
            _tty.setcbreak(self.fd)
        return self

    def get(self):
        if self.tty and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *exc):
        if self.tty:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hand", choices=["right", "left", "both"], default="right")
    p.add_argument("--sub", default="tcp://127.0.0.1:5556", help="qpos stream for right/single hand")
    p.add_argument("--sub-left", default="tcp://127.0.0.1:5557", help="qpos stream for the left hand (--hand both)")
    p.add_argument("--rate", type=float, default=20.0, help="control loop rate [Hz] (ramp up after it's stable)")
    p.add_argument("--stale-ms", type=float, default=150.0, help="freeze a hand if its stream is older than this")
    p.add_argument("--speed-coeff", type=float, default=0.3)
    p.add_argument("--current-coeff", type=float, default=0.3)
    p.add_argument("--tactile-delta", action="store_true",
                   help="触觉大图存帧间差分(每 30 帧一张完整图)+gzip1,实测真实数据"
                        "省 49%% 磁盘,逐位可还原。读取一律走 "
                        "magicdexmate.recording.read_tactile")
    p.add_argument("--max-step-rad", type=float, default=0.05, help="per-joint per-step cap (engage ramp + glitch guard)")
    p.add_argument("--auto-engage", action="store_true", help="engage at start without pressing 'e' (use with care)")
    p.add_argument("--record", metavar="DIR", default=None,
                   help="enable episode recording into DIR; press 'r' to start/stop an episode "
                        "(joints every tick + fingertip tactile at --tactile-hz, HDF5 per hand)")
    p.add_argument("--tactile", default="f6,deform,raw",
                   help="tactile streams to record: comma list of f6,deform,raw, or 'none' (joints only)")
    p.add_argument("--tactile-hz", type=float, default=30.0, help="tactile fetch rate [Hz] (sensor native 30)")
    p.add_argument("--skip-tactile-calib", action="store_true",
                   help="skip the contact-free tactile zero-calibration at first episode start")
    p.add_argument("--control", default=None, metavar="ADDR",
                   help="订阅浏览器页面的开关频道(如 tcp://127.0.0.1:5584)。"
                        "页面上「驱动真手」打开即跟随、关闭即冻结;录制开关同理。"
                        "不传则完全保持原有键盘操作,行为不变。频道 1.5 秒无消息"
                        "视为关闭,关掉页面等于冻结。")
    return p.parse_args()


def main():
    args = parse_args()
    pairs = [("right", args.sub)] if args.hand == "right" else \
            [("left", args.sub)] if args.hand == "left" else \
            [("right", args.sub), ("left", args.sub_left)]

    hands, subs = [], []
    for hand, addr in pairs:
        rh = SharpaRealHand(hand, speed_coeff=args.speed_coeff, current_coeff=args.current_coeff,
                            max_step_rad=args.max_step_rad)
        rh.connect().configure()
        hands.append(rh)
        subs.append(QposSub(addr, hand))
        print(f"[runner] {hand} hand <- {addr}")

    recorder = None
    if args.record:
        tactile_types = tuple(
            t for t in args.tactile.replace(" ", "").split(",") if t in ("f6", "deform", "raw"))
        recorder = EpisodeRecorder(args.record, hands, subs, tactile_types,
                                   args.tactile_hz, args.skip_tactile_calib)
        recorder.tactile_delta = bool(args.tactile_delta)
        if recorder.tactile_delta:
            print("[runner] 触觉存差分+gzip1(省约一半磁盘),读取走 "
                  "magicdexmate.recording.read_tactile")
        print(f"[runner] recording armed -> {args.record} (tactile: {','.join(tactile_types) or 'none'})")

    if args.auto_engage:
        for rh in hands:
            rh.engage()
    keys_help = "[e]ngage  [w] freeze  [q] home+freeze  [x] stop&quit"
    if recorder is not None:
        keys_help += "  [r] record start/stop"
    print(f"[runner] keys: {keys_help}   (starts DISENGAGED)")

    ctl = None
    ctl_prev = {"hand": False, "rec": False}
    if args.control:
        from magicdexmate.control_link import ControlSubscriber
        ctl = ControlSubscriber(args.control)
        print(f"[runner] 已订阅开关频道 {args.control}(页面控制跟随与录制)")

    period = 1.0 / args.rate
    stale_s = args.stale_ms / 1000.0
    last_print = time.time()
    try:
        with KeyPoller() as keys:
            while True:
                t0 = time.perf_counter()
                key = keys.get()
                if key in ("e", "E"):
                    for rh in hands:
                        rh.engage()
                elif key in ("w", "W"):
                    for rh in hands:
                        rh.freeze()
                elif key in ("q", "Q"):
                    for rh in hands:
                        rh.freeze()
                        rh.go_home()
                elif key in ("r", "R"):
                    if recorder is not None:
                        recorder.toggle()
                elif key in ("x", "X"):
                    break

                # switches from the browser page, edge-triggered so the
                # keyboard keys keep working alongside
                if ctl is not None:
                    st = ctl.latest(time.time()) or {}
                    h_on = bool(st.get("hand_enabled"))
                    if h_on != ctl_prev["hand"]:
                        ctl_prev["hand"] = h_on
                        for rh in hands:
                            (rh.engage if h_on else rh.freeze)()
                    r_on = bool(st.get("recording"))
                    if recorder is not None and r_on != ctl_prev["rec"]:
                        ctl_prev["rec"] = r_on
                        if r_on:
                            # 目录名必须在 toggle 之前设好:start_episode 立刻
                            # 就要用它建目录。
                            recorder.session = str(st.get("session") or "")
                        if r_on != recorder.active:
                            recorder.toggle()

                now = time.monotonic()
                for rh, sub in zip(hands, subs):
                    sub.poll()
                    fault = rh.fault_code()
                    if fault != 0:
                        rh.freeze()
                        print(f"\n[runner] {rh.hand} FAULT 0x{fault:x} -> frozen")
                    fresh = sub.valid and (now - sub.t_recv) < stale_s
                    if fresh and sub.qpos is not None:
                        rh.command(sub.qpos)
                    elif rh._engaged and sub.t_recv > 0 and (now - sub.t_recv) >= stale_s:
                        rh.freeze()  # watchdog: stream went stale
                        print(f"\n[runner] {rh.hand} stream stale -> frozen (press 'e' to re-engage)")
                    if recorder is not None and recorder.active:
                        recorder.log_joint(rh, sub)

                if time.time() - last_print > 2.0:
                    st = "  ".join(f"{rh.hand}:{'ENG' if rh._engaged else 'off'}" for rh in hands)
                    if recorder is not None:
                        st += "  " + recorder.status()
                    print(f"\r[runner] {st}", end="", flush=True)
                    last_print = time.time()

                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[runner] stopping")
        if recorder is not None and recorder.active:
            recorder.stop_episode()
        for rh in hands:
            rh.stop()


if __name__ == "__main__":
    main()
