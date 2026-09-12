"""Threaded HDF5 episode writer for one SharpaWave hand (joints + fingertip tactile).

Modeled on V2AP-demo teleop/data_writer.py (background thread + queue so the
control loop never blocks on disk), trimmed to the hand-only setup and changed
in one deliberate way: the joint stream and the tactile stream are recorded as
TWO independent-length streams with their own timestamps (control loop runs at
20-60 Hz, tactile arrives at 30 Hz; the player aligns them by nearest-neighbor
on a common timeline instead of duplicating tactile rows per control tick).

HDF5 layout (side = "left" | "right"):
  joint stream, one row per control tick:
    hand_timestamp                        (N,)    f64  host wall-clock [s]
    {side}_hand_target_joint_positions    (N,22)  f64  commanded qpos, SDK order [rad]
    {side}_hand_joint_positions           (N,22)  f64  measured qpos (NaN row if read failed)
    hand_valid                            (N,)    bool producer 'valid' flag
    hand_engaged                          (N,)    bool hand was engaged this tick
    glove_capture_us                      (N,)    i64  producer t_capture_us (0 if unknown)
  tactile stream, one row per fetch (30 Hz), finger order thumb..pinky:
    tactile_timestamp                     (M,)    f64  host wall-clock at fetch [s]
    tactile_device_timestamp              (M,5)   f64  sensor-side ts per finger [s]
    tactile_fresh                         (M,5)   bool frame arrived for that finger
    {side}_hand_tactile_f6                (M,5,6) f32  [Fx Fy Fz Mx My Mz]
    {side}_hand_tactile_deform            (M,5,240,240) u8  (lzf-compressed)
    {side}_hand_tactile_raw               (M,5,240,320) u8  (lzf-compressed)
File attrs: side, joint_names (json), tactile_fingers/channels/types, counts,
layout_version, created_unix.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import h5py
import numpy as np

from magicdexmate.sinks.sharpa_real import TACTILE_CHANNELS, TACTILE_FINGERS, TACTILE_SHAPES

LAYOUT_VERSION = 1
HAND_JOINT_COUNT = 22
TACTILE_KEYFRAME = 30   # 每秒一张完整图(触觉 30 Hz)
QUEUE_MAX = 256  # ~8 s of tactile at 30 Hz; on overflow we DROP (never block control)


class HandDataWriter:
    """Async HDF5 writer for one hand's episode. start() -> append_*() -> stop()."""

    def __init__(
        self,
        path: str | Path,
        side: str,
        joint_names: list[str],
        tactile_types: tuple[str, ...] = ("f6", "deform", "raw"),
        extra_attrs: dict | None = None,
        tactile_delta: bool = False,
    ):
        self.tactile_delta = bool(tactile_delta)
        # 差分只作用在 deform/raw 这两张大图上;f6 是 (5,6) 的小数组,没有意义
        self._delta_keys: set[str] = set()
        self._prev: dict = {}
        assert side in ("left", "right")
        assert len(joint_names) == HAND_JOINT_COUNT
        self.path = Path(path)
        self.side = side
        self.joint_names = list(joint_names)
        self.tactile_types = tuple(t for t in tactile_types if t in TACTILE_SHAPES)
        self.extra_attrs = dict(extra_attrs or {})

        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._file: h5py.File | None = None
        self._ds: dict[str, h5py.Dataset] = {}
        self.joint_rows = 0
        self.tactile_rows = 0
        self.dropped = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        f, s = self._file, self.side

        def ds(name, shape, dtype, **kw):
            self._ds[name] = f.create_dataset(name, (0, *shape), maxshape=(None, *shape),
                                              dtype=dtype, chunks=True, **kw)

        ds("hand_timestamp", (), np.float64)
        ds(f"{s}_hand_target_joint_positions", (HAND_JOINT_COUNT,), np.float64)
        ds(f"{s}_hand_joint_positions", (HAND_JOINT_COUNT,), np.float64)
        ds("hand_valid", (), np.bool_)
        ds("hand_engaged", (), np.bool_)
        ds("glove_capture_us", (), np.int64)

        if self.tactile_types:
            ds("tactile_timestamp", (), np.float64)
            ds("tactile_device_timestamp", (5,), np.float64)
            ds("tactile_fresh", (5,), np.bool_)
            if "f6" in self.tactile_types:
                ds(f"{s}_hand_tactile_f6", TACTILE_SHAPES["f6"], np.float32)
            if "deform" in self.tactile_types:
                self._ds[f"{s}_hand_tactile_deform"] = f.create_dataset(
                    f"{s}_hand_tactile_deform", (0, *TACTILE_SHAPES["deform"]),
                    maxshape=(None, *TACTILE_SHAPES["deform"]), dtype=np.uint8,
                    chunks=(1, *TACTILE_SHAPES["deform"]), **self._tac_comp(
                        f"{s}_hand_tactile_deform"))
            if "raw" in self.tactile_types:
                self._ds[f"{s}_hand_tactile_raw"] = f.create_dataset(
                    f"{s}_hand_tactile_raw", (0, *TACTILE_SHAPES["raw"]),
                    maxshape=(None, *TACTILE_SHAPES["raw"]), dtype=np.uint8,
                    chunks=(1, *TACTILE_SHAPES["raw"]), **self._tac_comp(
                        f"{s}_hand_tactile_raw"))

        f.attrs["layout_version"] = LAYOUT_VERSION
        f.attrs["side"] = s
        f.attrs["joint_names"] = json.dumps(self.joint_names)
        f.attrs["tactile_fingers"] = json.dumps(list(TACTILE_FINGERS))
        f.attrs["tactile_channels"] = json.dumps(list(TACTILE_CHANNELS[s]))
        f.attrs["tactile_types"] = json.dumps(list(self.tactile_types))
        # 编码方式必须写进文件。读的人不看这个属性就会拿到差分而不是原图,
        # 而差分看起来也是一张合法的图 —— 静默错到底。read_tactile() 按它解码。
        f.attrs["tactile_encoding"] = f"delta{TACTILE_KEYFRAME}+gzip1" if self.tactile_delta else "raw+lzf"
        f.attrs["created_unix"] = time.time()
        for k, v in self.extra_attrs.items():
            f.attrs[k] = v

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        print(f"[writer:{s}] recording -> {self.path}  (tactile: {','.join(self.tactile_types) or 'none'})")
        return self

    # -- producers (called from control loop / tactile thread) ---------------

    def append_joint(self, t_host: float, target, measured, valid: bool, engaged: bool,
                     capture_us: int = 0):
        self._put(("joint", {
            "hand_timestamp": t_host,
            f"{self.side}_hand_target_joint_positions": np.asarray(target, dtype=np.float64),
            f"{self.side}_hand_joint_positions": np.asarray(measured, dtype=np.float64),
            "hand_valid": bool(valid),
            "hand_engaged": bool(engaged),
            "glove_capture_us": int(capture_us),
        }))

    def _tac_comp(self, key: str) -> dict:
        """触觉大图的压缩设置。开了差分就登记这个 key 并换 gzip1。"""
        if not self.tactile_delta:
            return {"compression": "lzf"}
        self._delta_keys.add(key)
        return {"compression": "gzip", "compression_opts": 1}

    def append_tactile(self, tac: dict):
        """tac = SharpaRealHand.fetch_tactile() result."""
        row = {
            "tactile_timestamp": float(tac["t_host"]),
            "tactile_device_timestamp": np.asarray(tac["dev_ts"], dtype=np.float64),
            "tactile_fresh": np.asarray(tac["fresh"], dtype=bool),
        }
        for t in self.tactile_types:
            if t in tac:
                row[f"{self.side}_hand_tactile_{t}"] = tac[t]
        self._put(("tactile", row))

    def _put(self, item):
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 30 == 1:
                print(f"[writer:{self.side}] disk can't keep up, dropped {self.dropped} rows")

    # -- background thread ----------------------------------------------------

    def _writer_loop(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                kind, row = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write_row(kind, row)
            except Exception as e:  # noqa: BLE001 - a bad row must not kill the writer
                print(f"[writer:{self.side}] error writing {kind} row: {e}")

    def _write_row(self, kind: str, row: dict):
        if kind == "joint":
            self.joint_rows += 1
            idx = self.joint_rows
        else:
            self.tactile_rows += 1
            idx = self.tactile_rows
        for key, value in row.items():
            d = self._ds.get(key)
            if d is None:
                continue
            if key in self._delta_keys:
                # 每 TACTILE_KEYFRAME 行存一张完整图(视频里的 I 帧)。纯差分
                # 的随机访问是 O(k):要看第 3000 行得先解前 2999 行,而回放台
                # 正是按行随机读的。有了关键帧,最多解 29 行。
                # 存**帧间差分**而不是原图。触觉图的时间相关性极强,而 lzf 是
                # 逐块的通用压缩,看不到这一点 —— 实测真实数据上 lzf 只有
                # 1.14 倍,差分之后配 gzip1 是 2.27 倍(与无损视频的 2.34 倍
                # 相当),而且不引入外部进程和新文件格式。
                # 模 256 的加减是可逆的(已逐位验证),所以这仍然是无损的。
                cur = np.asarray(value, dtype=np.uint8)
                prev = self._prev.get(key)
                key_row = (idx - 1) % TACTILE_KEYFRAME == 0
                store = cur if (prev is None or key_row) else (
                    cur.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
                self._prev[key] = cur
                value = store
            d.resize(idx, axis=0)
            d[idx - 1] = value
        if (self.joint_rows + self.tactile_rows) % 100 == 0:
            self._file.flush()

    # -- shutdown -------------------------------------------------------------

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        if self._file is not None:
            self._file.attrs["joint_rows"] = self.joint_rows
            self._file.attrs["tactile_rows"] = self.tactile_rows
            self._file.attrs["dropped_rows"] = self.dropped
            self._file.close()
            self._file = None
        counts = {"joint": self.joint_rows, "tactile": self.tactile_rows, "dropped": self.dropped}
        print(f"[writer:{self.side}] closed {self.path.name}: {counts}")
        return counts
