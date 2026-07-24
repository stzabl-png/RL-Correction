"""Live Wuji glove source via wuji-sdk (`pip install wuji-sdk`).

Verified against wuji-sdk 2026.6.2:
  SdkManager.instance() / .scan() / .connect(sn=|address=|handedness=, device_name=) / .auto_connect(device_name)
  WujiGlove.hand_skeleton() -> HandSkeletonResource -> .subscribe() -> Subscription.recv()  (sync, blocking)
  HandSkeleton: .header (FrameHeader: seq, timestamp_us, frame_id), .joints: list[SkeletonJoint]
  SkeletonJoint: .name, .pose (.position .x/.y/.z, .orientation .w/.x/.y/.z), .confidence
  WujiGlove.imu_palm() -> ImuData stream (.orientation quaternion) at 800 Hz

Network bring-up (docs/references/01_wuji_glove.md): host NIC on 192.168.1.x/24,
left glove 192.168.1.100, right 192.168.1.101, UDP 50000/50001.
"""

from __future__ import annotations

import threading

import numpy as np

from magicdexmate.skeleton import HandFrame, joint_name_to_index
from magicdexmate.sources.base import GloveSource, LatestSlot


class WujiGloveSource(GloveSource):
    def __init__(
        self,
        hand: str = "right",
        sn: str | None = None,
        address: str | None = None,
        device_name: str = "glove0",
        with_imu: bool = True,
    ):
        assert hand in ("right", "left")
        self.hand = hand
        self.sn = sn
        self.address = address
        self.device_name = device_name
        self.with_imu = with_imu

        self._slot = LatestSlot()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._glove = None
        self._skel_sub = None
        self._imu_sub = None
        self._wrist_quat: np.ndarray | None = None
        self._wrist_lock = threading.Lock()
        self._warned_names = False

    # -- connection -----------------------------------------------------------

    def start(self):
        import wuji_sdk

        manager = wuji_sdk.SdkManager.instance()
        if self.sn is not None:
            self._glove = manager.connect(sn=self.sn, device_name=self.device_name)
        elif self.address is not None:
            self._glove = manager.connect(address=self.address, device_name=self.device_name)
        else:
            handedness = wuji_sdk.Handedness.Right if self.hand == "right" else wuji_sdk.Handedness.Left
            self._glove = manager.connect(handedness=handedness, device_name=self.device_name)
        print(f"[wuji] connected: sn={self._glove.sn()} hand={self.hand}")

        self._skel_sub = self._glove.hand_skeleton().subscribe()
        t = threading.Thread(target=self._skeleton_loop, name="wuji-skeleton", daemon=True)
        t.start()
        self._threads.append(t)

        if self.with_imu:
            self._imu_sub = self._glove.imu_palm().subscribe()
            t = threading.Thread(target=self._imu_loop, name="wuji-imu", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for sub in (self._skel_sub, self._imu_sub):
            if sub is not None:
                try:
                    sub.close()
                except Exception:
                    pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        if self._glove is not None:
            try:
                self._glove.disconnect()
            except Exception:
                pass
            self._glove = None

    # -- streaming ------------------------------------------------------------

    def _skeleton_loop(self):
        while not self._stop.is_set():
            try:
                skel = self._skel_sub.recv()
            except Exception:
                if self._stop.is_set():
                    return
                raise
            frame = self._parse_skeleton(skel)
            if frame is not None:
                self._slot.put(frame)

    def _imu_loop(self):
        while not self._stop.is_set():
            try:
                imu = self._imu_sub.recv()
            except Exception:
                if self._stop.is_set():
                    return
                raise
            q = imu.orientation
            with self._wrist_lock:
                self._wrist_quat = np.array([q.w, q.x, q.y, q.z])

    def _parse_skeleton(self, skel) -> HandFrame | None:
        kp = np.zeros((21, 3))
        conf = np.full(21, -1.0)
        for j in skel.joints:
            try:
                idx = joint_name_to_index(j.name)
            except KeyError:
                if not self._warned_names:
                    self._warned_names = True
                    names = [jj.name for jj in skel.joints]
                    print(f"[wuji] WARNING: unmapped joint name {j.name!r}; frame has names: {names}")
                continue
            p = j.pose.position
            kp[idx] = (p.x, p.y, p.z)
            conf[idx] = j.confidence

        if (conf < 0).any():
            if not self._warned_names:
                self._warned_names = True
                missing = [i for i in range(21) if conf[i] < 0]
                print(f"[wuji] WARNING: skeleton frame missing keypoint indices {missing}")
            return None

        with self._wrist_lock:
            wrist_quat = None if self._wrist_quat is None else self._wrist_quat.copy()
        return HandFrame(
            t_us=int(skel.header.timestamp_us),
            hand=self.hand,
            kp=kp,
            conf=conf,
            wrist_quat=wrist_quat,
        )

    def get_latest(self) -> HandFrame | None:
        return self._slot.get()
