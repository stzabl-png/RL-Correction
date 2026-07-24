"""ZMQ publisher for retargeted qpos: one stream feeding sim and/or real hand.

Message schema (JSON, one line per message):
  hello: {"type": "hello", "hand", "joint_names": [...22 SDK order...], "crc"}
         re-sent every 2 s so late subscribers always get the joint order.
  qpos:  {"type": "qpos", "seq", "hand", "t_capture_us", "t_pub_us", "valid",
          "crc", "qpos": [...22 floats, SDK order, rad...], "wrist_quat": [w,x,y,z]|null}

`crc` is crc32 over ",".join(joint_names); consumers must check it matches the
hello they used to build their joint-index mapping.
"""

from __future__ import annotations

import json
import time
import zlib

import numpy as np
import zmq

HELLO_INTERVAL_S = 2.0


def names_crc(joint_names: list[str]) -> int:
    return zlib.crc32(",".join(joint_names).encode())


class QposPublisher:
    def __init__(self, bind: str, hand: str, joint_names: list[str]):
        self.hand = hand
        self.joint_names = list(joint_names)
        self.crc = names_crc(self.joint_names)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(bind)
        self._seq = 0
        self._last_hello = 0.0

    def send(
        self,
        qpos_sdk: np.ndarray,
        t_capture_us: int,
        valid: bool = True,
        wrist_quat: np.ndarray | None = None,
    ):
        now = time.time()
        if now - self._last_hello > HELLO_INTERVAL_S:
            self._sock.send_string(
                json.dumps(
                    {"type": "hello", "hand": self.hand, "joint_names": self.joint_names, "crc": self.crc}
                )
            )
            self._last_hello = now
        msg = {
            "type": "qpos",
            "seq": self._seq,
            "hand": self.hand,
            "t_capture_us": int(t_capture_us),
            "t_pub_us": time.time_ns() // 1000,
            "valid": bool(valid),
            "crc": self.crc,
            "qpos": np.asarray(qpos_sdk, dtype=float).tolist(),
            "wrist_quat": None if wrist_quat is None else np.asarray(wrist_quat, dtype=float).tolist(),
        }
        self._sock.send_string(json.dumps(msg))
        self._seq += 1

    def close(self):
        self._sock.close(linger=200)
