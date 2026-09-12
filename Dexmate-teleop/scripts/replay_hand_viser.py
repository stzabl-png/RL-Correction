#!/usr/bin/env python
"""viser 3D replay of a recorded SharpaWave hand episode (joints + fingertip tactile).

Reads one HDF5 written by magicdexmate.recording.HandDataWriter (see
scripts/sharpa_real_runner.py --record) and serves an interactive browser scene:

  - the hand (URDF) replayed from measured (default) or commanded joint positions
  - per-fingertip contact: elastomer overlay colored by |F| (grey -> yellow -> red)
    plus a force vector line (F6 force rotated into the world by the elastomer pose)
  - fingertip 3D trails (trail length adjustable)
  - 5-finger DEFORM image panel, synced to playback
  - timeline: slider + play/pause + speed, tactile aligned to the joint timeline
    by nearest-neighbor timestamp matching (the two streams are independent rates)

Usage:
  PYTHONPATH= .venv/bin/python scripts/replay_hand_viser.py DATA.h5 [--port 8080]
  # then open http://localhost:8080 in a browser

--smoke builds the whole scene headless, steps 10 frames and exits (offline CI).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TRAIL_COLORS = {  # per-finger trail color (RGB 0-255)
    "thumb": (230, 90, 40), "index": (60, 160, 240), "middle": (110, 210, 90),
    "ring": (200, 120, 230), "pinky": (240, 200, 60),
}
FORCE_THRESH_N = 0.15  # below this the elastomer stays grey (sensor noise floor)
REST_COLOR = (0.72, 0.72, 0.75)


# -- data loading -------------------------------------------------------------

class Episode:
    def __init__(self, path: Path):
        self.path = path
        with h5py.File(path, "r") as f:
            self.side = f.attrs["side"]
            self.joint_names = json.loads(f.attrs["joint_names"])
            s = self.side
            self.t = np.asarray(f["hand_timestamp"])
            self.target = np.asarray(f[f"{s}_hand_target_joint_positions"])
            self.measured = np.asarray(f[f"{s}_hand_joint_positions"])
            self.tac_t = np.asarray(f["tactile_timestamp"]) if "tactile_timestamp" in f else np.zeros(0)
            self.f6 = np.asarray(f[f"{s}_hand_tactile_f6"]) if f"{s}_hand_tactile_f6" in f else None
            # deform stays on disk (can be GBs); read rows lazily
            self._h5 = None
            self.has_deform = f"{s}_hand_tactile_deform" in f
        if len(self.t) == 0:
            raise SystemExit(f"{path}: empty joint stream")
        # forward-fill measured rows where the SDK read failed (NaN) so FK never sees NaN
        bad = np.isnan(self.measured).any(axis=1)
        if bad.all():
            self.measured = None  # e.g. synthetic files without a measured stream
        elif bad.any():
            for i in np.flatnonzero(bad):
                self.measured[i] = self.measured[i - 1] if i > 0 else self.target[i]
        self.t0 = self.t[0]
        self.duration = float(self.t[-1] - self.t0)

    def deform_row(self, k: int) -> np.ndarray | None:
        if not self.has_deform:
            return None
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        # 必须走 read_tactile:开了差分录制的文件里存的是帧间差,直接读那一行
        # 会得到一张看起来正常、其实是差分的图,不会报任何错。
        from magicdexmate.recording import read_tactile
        return read_tactile(self._h5, f"{self.side}_hand_tactile_deform", index=k)

    def tac_index(self, t_abs: float) -> int | None:
        """Nearest tactile row for an absolute joint-stream timestamp."""
        if len(self.tac_t) == 0:
            return None
        k = int(np.searchsorted(self.tac_t, t_abs))
        if k <= 0:
            return 0
        if k >= len(self.tac_t):
            return len(self.tac_t) - 1
        return k if abs(self.tac_t[k] - t_abs) < abs(self.tac_t[k - 1] - t_abs) else k - 1


# -- kinematics ---------------------------------------------------------------

class HandFK:
    """pinocchio FK: SDK-order qpos -> world poses of fingertip + elastomer links."""

    def __init__(self, urdf_path: Path, side: str, joint_names: list[str]):
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.q_idx = []
        for n in joint_names:
            jid = self.model.getJointId(n)
            assert jid < self.model.njoints, f"joint {n} not in URDF"
            self.q_idx.append(self.model.joints[jid].idx_q)
        self.tip_fid = [self.model.getFrameId(f"{side}_{f}_fingertip") for f in FINGERS]
        self.ela_fid = [self.model.getFrameId(f"{side}_{f}_elastomer") for f in FINGERS]

    def compute(self, qpos_sdk: np.ndarray):
        """-> (tips (5,3), ela_R (5,3,3), ela_p (5,3)) for one frame."""
        pin = self._pin
        q = np.zeros(self.model.nq)
        q[self.q_idx] = qpos_sdk
        pin.framesForwardKinematics(self.model, self.data, q)
        tips = np.array([self.data.oMf[fid].translation for fid in self.tip_fid])
        ela_R = np.array([self.data.oMf[fid].rotation for fid in self.ela_fid])
        ela_p = np.array([self.data.oMf[fid].translation for fid in self.ela_fid])
        return tips, ela_R, ela_p

    def precompute(self, qpos_all: np.ndarray):
        n = len(qpos_all)
        tips = np.zeros((n, 5, 3))
        ela_R = np.zeros((n, 5, 3, 3))
        ela_p = np.zeros((n, 5, 3))
        for i, q in enumerate(qpos_all):
            tips[i], ela_R[i], ela_p[i] = self.compute(q)
        return tips, ela_R, ela_p


def mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    import pinocchio as pin

    xyzw = pin.Quaternion(np.ascontiguousarray(R)).coeffs()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def force_color(f_n: float, f_max: float) -> tuple[float, float, float]:
    """|F| [N] -> grey (no contact) -> yellow -> red (saturated at f_max)."""
    if f_n < FORCE_THRESH_N:
        return REST_COLOR
    t = min(1.0, (f_n - FORCE_THRESH_N) / max(1e-6, f_max - FORCE_THRESH_N))
    y, r = np.array([1.0, 0.85, 0.10]), np.array([0.90, 0.12, 0.05])
    return tuple((1 - t) * y + t * r)


# -- player -------------------------------------------------------------------

class Player:
    def __init__(self, ep: Episode, port: int, show_gui: bool = True):
        import trimesh
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf

        self.ep = ep
        urdf_path = REPO / "assets/robots/hands/sharpa_wave" / ep.side / f"{ep.side}_sharpa_wave.urdf"
        self.urdf = yourdfpy.URDF.load(str(urdf_path))
        self.fk = HandFK(urdf_path, ep.side, ep.joint_names)

        self.server = viser.ViserServer(port=port, verbose=False)
        self.vurdf = ViserUrdf(self.server, self.urdf, root_node_name="/hand")
        # ViserUrdf expects cfg in yourdfpy actuated-joint order; map from SDK order
        act = list(self.urdf.actuated_joint_names)
        self.cfg_map = np.array([ep.joint_names.index(n) for n in act])

        # pose sources + precomputed FK per source
        self.sources = {}
        if ep.measured is not None:
            self.sources["measured"] = ep.measured
        self.sources["target"] = ep.target
        self.source = "measured" if "measured" in self.sources else "target"
        print(f"[replay] precomputing FK for {len(ep.t)} frames x {len(self.sources)} source(s)...")
        self.fk_cache = {k: self.fk.precompute(v) for k, v in self.sources.items()}

        # elastomer overlay meshes (slightly inflated copies we can recolor per frame)
        self.ela_handles = []
        for f in FINGERS:
            link = self.urdf.link_map[f"{ep.side}_{f}_elastomer"]
            mesh_file = link.visuals[0].geometry.mesh.filename
            m = trimesh.load(str(urdf_path.parent / mesh_file), force="mesh")
            verts = np.asarray(m.vertices) * 1.03  # avoid z-fighting with the URDF mesh
            self.ela_handles.append(self.server.scene.add_mesh_simple(
                f"/overlay/ela_{f}", verts.astype(np.float32),
                np.asarray(m.faces, dtype=np.uint32), color=REST_COLOR))

        self.arrow_handle = self.server.scene.add_line_segments(
            "/overlay/forces", points=np.zeros((5, 2, 3), dtype=np.float32),
            colors=np.full((5, 2, 3), 255, dtype=np.uint8), line_width=4.0)
        self.trail_handles = [
            self.server.scene.add_line_segments(
                f"/overlay/trail_{f}", points=np.zeros((1, 2, 3), dtype=np.float32),
                colors=np.tile(np.array(TRAIL_COLORS[f], dtype=np.uint8), (1, 2, 1)),
                line_width=2.5)
            for f in FINGERS
        ]

        self._build_gui(show_gui)
        self.frame = -1
        self.playing = False
        self.set_frame(0)
        threading.Thread(target=self._play_loop, daemon=True).start()

    # -- GUI ------------------------------------------------------------------

    def _build_gui(self, show):
        g = self.server.gui
        n = len(self.ep.t)
        with g.add_folder("Playback"):
            self.ui_slider = g.add_slider("frame", min=0, max=n - 1, step=1, initial_value=0)
            self.ui_time = g.add_text("t [s]", initial_value="0.00", disabled=True)
            self.ui_play = g.add_checkbox("play", initial_value=False)
            self.ui_speed = g.add_dropdown("speed", options=("0.25", "0.5", "1", "2", "4"),
                                           initial_value="1")
            self.ui_loop = g.add_checkbox("loop", initial_value=True)
            self.ui_source = g.add_dropdown("pose source", options=tuple(self.sources),
                                            initial_value=self.source)
        with g.add_folder("Contact / force"):
            self.ui_fmax = g.add_slider("F max [N] (color/arrow saturation)", min=0.5, max=30.0,
                                        step=0.5, initial_value=5.0)
            self.ui_arrow = g.add_slider("arrow scale [m/N]", min=0.0, max=0.05, step=0.002,
                                         initial_value=0.02)
            self.ui_f6 = g.add_markdown("`|F|` waiting...")
        with g.add_folder("Trails"):
            self.ui_trail = g.add_slider("trail length [frames]", min=0, max=600, step=10,
                                         initial_value=150)
        self.ui_deform = []
        if self.ep.has_deform:
            with g.add_folder("DEFORM (fingertip depth)"):
                blank = np.zeros((120, 120, 3), dtype=np.uint8)
                for f in FINGERS:
                    self.ui_deform.append(g.add_image(blank, label=f, format="jpeg"))

        self.ui_slider.on_update(lambda _: self.set_frame(int(self.ui_slider.value)))
        self.ui_source.on_update(lambda _: self._set_source(self.ui_source.value))

    def _set_source(self, src):
        if src in self.sources:
            self.source = src
            self.set_frame(self.frame, force=True)

    # -- frame update ---------------------------------------------------------

    def set_frame(self, i: int, force: bool = False):
        i = int(np.clip(i, 0, len(self.ep.t) - 1))
        if i == self.frame and not force:
            return
        prev_tac = self.ep.tac_index(self.ep.t[self.frame]) if self.frame >= 0 else None
        self.frame = i
        qpos = self.sources[self.source][i]
        tips_all, ela_R_all, ela_p_all = self.fk_cache[self.source]
        tips, ela_R, ela_p = tips_all[i], ela_R_all[i], ela_p_all[i]

        with self.server.atomic():
            self.vurdf.update_cfg(qpos[self.cfg_map])

            # elastomer pose + color, force arrows
            f6 = self.ep.f6[self.ep.tac_index(self.ep.t[i])] if self.ep.f6 is not None else None
            fmax = float(self.ui_fmax.value)
            arrow_pts = np.zeros((5, 2, 3), dtype=np.float32)
            arrow_col = np.zeros((5, 2, 3), dtype=np.uint8)
            lines = []
            for k, f in enumerate(FINGERS):
                h = self.ela_handles[k]
                h.position = ela_p[k]
                h.wxyz = mat_to_wxyz(ela_R[k])
                fvec = ela_R[k] @ f6[k, :3] if f6 is not None else np.zeros(3)
                fN = float(np.linalg.norm(fvec))
                h.color = force_color(fN, fmax)
                arrow_pts[k, 0] = tips[k]
                arrow_pts[k, 1] = tips[k] + fvec / max(fN, 1e-9) * min(fN, fmax) * self.ui_arrow.value
                c = np.array(force_color(fN, fmax)) * 255
                arrow_col[k] = c.astype(np.uint8)
                lines.append(f"{f[:2]} {fN:5.2f}N")
            self.arrow_handle.points = arrow_pts
            self.arrow_handle.colors = arrow_col
            self.ui_f6.content = "`|F|` " + "  ".join(lines)

            # trails
            K = int(self.ui_trail.value)
            for k in range(5):
                lo = max(0, i - K)
                seg = tips_all[lo:i + 1, k]
                if len(seg) >= 2 and K > 0:
                    pts = np.stack([seg[:-1], seg[1:]], axis=1).astype(np.float32)
                else:
                    pts = np.zeros((1, 2, 3), dtype=np.float32)
                self.trail_handles[k].points = pts
                self.trail_handles[k].colors = np.tile(
                    np.array(TRAIL_COLORS[FINGERS[k]], dtype=np.uint8), (len(pts), 2, 1))

            # DEFORM panel (only when the tactile row actually changed)
            tac = self.ep.tac_index(self.ep.t[i])
            if self.ui_deform and tac is not None and tac != prev_tac:
                row = self.ep.deform_row(tac)
                if row is not None:
                    import cv2

                    for k in range(5):
                        small = cv2.resize(row[k], (120, 120))
                        self.ui_deform[k].image = np.stack([small] * 3, axis=-1)

            self.ui_time.value = f"{self.ep.t[i] - self.ep.t0:.2f} / {self.ep.duration:.2f}"
            if int(self.ui_slider.value) != i:
                self.ui_slider.value = i

    # -- playback loop --------------------------------------------------------

    def _play_loop(self):
        last = time.perf_counter()
        while True:
            time.sleep(0.02)
            now = time.perf_counter()
            dt, last = now - last, now
            if not self.ui_play.value:
                continue
            t_rel = self.ep.t[self.frame] - self.ep.t0 + dt * float(self.ui_speed.value)
            if t_rel >= self.ep.duration:
                if self.ui_loop.value:
                    self.set_frame(0)
                else:
                    self.ui_play.value = False
                continue
            self.set_frame(int(np.searchsorted(self.ep.t - self.ep.t0, t_rel)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", type=Path, help="episode .h5 (from sharpa_real_runner.py --record)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--smoke", action="store_true", help="headless self-test: build scene, step 10 frames, exit")
    args = ap.parse_args()

    ep = Episode(args.h5)
    print(f"[replay] {args.h5.name}: side={ep.side} joints={len(ep.t)} rows "
          f"tactile={len(ep.tac_t)} rows dur={ep.duration:.1f}s")
    player = Player(ep, args.port)

    if args.smoke:
        for i in range(0, min(10, len(ep.t))):
            player.set_frame(i, force=True)
        print("[replay] smoke OK (scene built, 10 frames stepped)")
        return

    print(f"[replay] open http://localhost:{args.port}  (Ctrl-C to quit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
