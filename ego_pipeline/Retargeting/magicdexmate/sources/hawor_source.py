"""HaWoR replay source: feed reconstructed hand joints into the retarget pipeline.

HaWoR (CVPR'25, World-Space Hand Motion Reconstruction) reconstructs MANO hand
motion from egocentric video. Its joint output is 21 keypoints in OpenPose hand
order (lib/models/mano_wrapper.py: mano_to_openpose), which is identical to the
MediaPipe-21 order this pipeline expects (skeleton.py: MEDIAPIPE_JOINT_NAMES):
    0=wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
So HaWoR joints feed straight into retarget/frames.to_mano with NO reindexing.
to_mano re-estimates the operator frame per-frame, so HaWoR's world frame and
units (meters) need no alignment - only the joint geometry + order, which match.

This source consumes a lightweight joints .npz produced by
scripts/hawor_to_joints.py (run in the HaWoR env), so the magicdexmate side
never needs MANO/smplx. Expected keys:
    joints_right : (T, 21, 3) float, world meters, OpenPose/MediaPipe order
    joints_left  : (T, 21, 3) float            (either hand may be absent)
    valid        : (2, T)     float, [0]=left [1]=right   (optional, default 1)
    fps          : scalar capture rate                    (optional, default 30)

Two consumption modes, both supported:
- sim-time driven (Isaac single-process loop): source.sample_at(sim_t) - one
  HaWoR frame per (t*fps), deterministic, robust to sub-realtime sim speed.
- wall-clock driven (bridged ZMQ / live-like): source.get_latest().
"""

from __future__ import annotations

import time

import numpy as np

from magicdexmate.skeleton import HandFrame
from magicdexmate.sources.base import GloveSource


class HaWoRSource(GloveSource):
    def __init__(
        self,
        hand: str = "right",
        npz: str | None = None,
        fps: float | None = None,
        loop: bool = False,
    ):
        assert hand in ("right", "left")
        if npz is None:
            raise ValueError("HaWoRSource needs npz=<path to hawor joints .npz> "
                             "(make one with scripts/hawor_to_joints.py)")
        self.hand = hand
        self.loop = loop
        self.ended = False

        data = np.load(npz)
        key = f"joints_{hand}"
        if key not in data:
            raise KeyError(f"{npz} has no '{key}' (keys present: {list(data.keys())})")
        self.joints = np.asarray(data[key], dtype=np.float64)
        if self.joints.ndim != 3 or self.joints.shape[1:] != (21, 3):
            raise ValueError(f"'{key}' must be (T,21,3), got {self.joints.shape}")
        self.n_frames = int(self.joints.shape[0])

        if "valid" in data:
            v = np.asarray(data["valid"])
            self.valid = v[0 if hand == "left" else 1].astype(bool)
        else:
            self.valid = np.ones(self.n_frames, dtype=bool)

        self.fps = float(fps if fps is not None else
                         (data["fps"] if "fps" in data else 30.0))

        self._last_idx = -1
        self._last_t_us = 0
        self._t0: float | None = None

    # -- core: index -> HandFrame --------------------------------------------

    def _frame_at_index(self, idx: int) -> HandFrame:
        valid = bool(self.valid[idx])
        # Reuse one timestamp per distinct frame so the consumer's
        # `frame.t_us != last_t_us` dedup skips re-retargeting an unchanged
        # frame, yet the stamp stays recent (wall clock) so the staleness
        # watchdog never trips on replay.
        if idx != self._last_idx:
            self._last_idx = idx
            self._last_t_us = time.time_ns() // 1000
        return HandFrame(
            t_us=self._last_t_us,
            hand=self.hand,
            kp=self.joints[idx],
            conf=np.full(21, 1.0 if valid else 0.0),
            wrist_quat=None,
        )

    def _index_for_time(self, t: float) -> int:
        raw = int(t * self.fps)
        if self.loop:
            return raw % self.n_frames
        if raw >= self.n_frames - 1:
            self.ended = True
        return min(raw, self.n_frames - 1)

    def sample_at(self, t: float) -> HandFrame:
        """Frame at playback-time t seconds (sim-time driven path)."""
        return self._frame_at_index(self._index_for_time(t))

    # -- GloveSource interface (wall-clock path) -----------------------------

    def start(self) -> None:
        self._t0 = time.monotonic()

    def stop(self) -> None:
        self._t0 = None

    def get_latest(self) -> HandFrame | None:
        if self._t0 is None:
            return None
        return self.sample_at(time.monotonic() - self._t0)
