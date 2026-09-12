"""The operator's switches, from the browser to the processes that obey them.

Three processes, one direction of control:

    viser_isaac_mirror.py   (browser: arm/disarm, mode)   --PUB-->  :5584
    teleop_vega_pico.py     (applies the mode)            <--SUB
    dexmate_bridge.py       (applies armed/disarmed)      <--SUB

Why a channel and not a flag: arming real hardware and changing what it follows
are decisions made WHILE WEARING the headset, and the operator cannot restart a
tmux pane with their hands in the air. The mirror is already the window they
watch, so the switches belong there.

Deliberately one-way and stateless. The publisher re-sends its whole state
every tick, so a subscriber that starts late, or restarts, picks the current
switch positions up within one period - no handshake, no session, nothing to
get out of sync. CONFLATE on the subscriber means it only ever sees the latest.

FAIL-SAFE, and this is the part that matters: a subscriber that has never heard
from the publisher gets `None`, and every consumer of this module treats `None`
as "not armed". Losing the browser therefore disarms the robot rather than
leaving it running on the last thing it was told.
"""
from __future__ import annotations

import msgpack
import zmq

DEFAULT_ADDR = "tcp://127.0.0.1:5584"


class ControlPublisher:
    """Browser side. Binds; re-sends the full switch state every tick."""

    def __init__(self, addr: str = DEFAULT_ADDR):
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 2)
        self.sock.bind(addr)
        self.addr = addr

    def send(self, state: dict) -> None:
        try:
            self.sock.send(msgpack.packb(state, use_bin_type=True), zmq.NOBLOCK)
        except Exception:                                  # noqa: BLE001
            pass


class ControlSubscriber:
    """Consumer / bridge side. Never blocks; `latest()` is None until heard."""

    def __init__(self, addr: str = DEFAULT_ADDR, stale_s: float = 1.5):
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.connect(addr)
        self.stale_s = stale_s
        self._last = None
        self._t = 0.0

    def latest(self, now: float) -> dict | None:
        while True:
            try:
                self._last = msgpack.unpackb(self.sock.recv(zmq.NOBLOCK),
                                             raw=False)
                self._t = now
            except zmq.Again:
                break
        if self._last is None or now - self._t > self.stale_s:
            # stale is the same as never: the browser tab was closed, the
            # machine went to sleep, the process died. None means disarm.
            return None
        return self._last
