#!/usr/bin/env python3
"""Viser capture inspector: watch a pose take in 3D before trusting it.

Every tracker and the headset are drawn as 6D frames (axes show orientation)
with a scrubbable timeline, so a take can be judged the moment it is recorded
instead of weeks later in an analysis.

It exists because two capture batches were recorded and thrown away. The
failures are obvious in 3D and invisible in a progress bar:

  FROZEN   a tracker leaves the headset FOV and its value stops changing
           bit-for-bit. Drawn in RED with a "FROZEN" tag - if a wrist is red
           while you were moving it, that take is dead.
  脑补     the body model keeps streaming a plausible-looking pose that is not
           where your arm is. Visible as a collapsed skeleton: on the
           2026-07-29 batch "arms high" and "arms forward" ended up 0.168 m
           apart when they are ~1 m apart in reality.

Both channels are shown when both are streaming (the 2026-07-29 hybrid rig):
raw per-tracker serials and the body-model joints, side by side, so they can be
compared directly.

Poses are converted to the z-up world frame (x forward, y left, z up) - the
frame the mapping speaks - so what you see is what the consumer sees.

Usage:
    .venv/bin/python scripts/viser_capture_view.py logs/pose_capture_20260729
    .venv/bin/python scripts/viser_capture_view.py logs/pose_capture_20260729_raw
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import msgpack
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.pico.xr_pose import R_HEADSET_TO_WORLD  # noqa: E402

# fixed colours per role so a take always looks the same
COLORS = {
    "head": (240, 240, 240),
    "LWRIST": (80, 200, 255), "RWRIST": (255, 170, 60),
    "WAIST": (170, 255, 120), "LELBOW": (60, 130, 200), "RELBOW": (200, 120, 40),
}
RAW_COLORS = [(255, 90, 200), (140, 230, 140), (255, 230, 90)]
FROZEN = (255, 40, 40)
LINKS = [("WAIST", "head"), ("WAIST", "LELBOW"), ("LELBOW", "LWRIST"),
         ("WAIST", "RELBOW"), ("RELBOW", "RWRIST")]


def load_take(path: pathlib.Path):
    """-> (t[s], {name: (N,3) pos_world}, {name: (N,4) wxyz_world}, {name: (N,) frozen})"""
    ts, poses = [], {}
    for m in msgpack.Unpacker(open(path, "rb"), raw=False):
        entry = {}
        if m.get("head") is not None:
            entry["head"] = m["head"]
        for k, v in (m.get("trackers") or {}).items():
            entry[k] = v
        if not entry:
            continue
        ts.append(m.get("t_us", len(ts) * 10_000) / 1e6)
        for k, v in entry.items():
            poses.setdefault(k, {})[len(ts) - 1] = v

    n = len(ts)
    t = np.array(ts)
    t -= t[0] if n else 0
    P, Q, F = {}, {}, {}
    M = R_HEADSET_TO_WORLD
    for name, by_i in poses.items():
        raw = np.full((n, 7), np.nan)
        for i, v in by_i.items():
            raw[i] = v
        # carry the last known sample through gaps so the frame never teleports
        for i in range(1, n):
            if np.isnan(raw[i, 0]):
                raw[i] = raw[i - 1]
        pos = (M @ raw[:, :3].T).T
        quat = np.zeros((n, 4))
        for i in range(n):
            q = raw[i, 3:7]
            if np.any(np.isnan(q)) or np.allclose(q, 0):
                quat[i] = [1, 0, 0, 0]
                continue
            rot = M @ R.from_quat(q).as_matrix() @ M.T
            xyzw = R.from_matrix(rot).as_quat()
            quat[i] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]     # viser wants wxyz
        frozen = np.zeros(n, dtype=bool)
        for i in range(1, n):
            frozen[i] = np.array_equal(raw[i], raw[i - 1])
        P[name], Q[name], F[name] = pos, quat, frozen
    return t, P, Q, F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=pathlib.Path,
                    help="capture directory, or a single .msgpack take")
    ap.add_argument("--port", type=int, default=8082,
                    help="8082: 8080 is the trace viewer, 8081 the joint bench")
    args = ap.parse_args()

    if args.capture.is_dir():
        files = sorted(p for p in args.capture.glob("*.msgpack")
                       if "format_test" not in p.name)
    else:
        files = [args.capture]
    if not files:
        raise SystemExit(f"no takes found in {args.capture}")
    print(f"[view] {len(files)} takes from {args.capture}")

    cache: dict[str, tuple] = {}

    def get(name: str):
        if name not in cache:
            f = next(p for p in files if p.stem == name)
            cache[name] = load_take(f)
            t, P, _, F = cache[name]
            frozen_pct = {k: 100.0 * v.mean() for k, v in F.items()}
            worst = sorted(frozen_pct.items(), key=lambda kv: -kv[1])[:3]
            print(f"[view] {name}: {len(t)} frames, {t[-1]:.1f}s, "
                  + ", ".join(f"{k} {v:.0f}% frozen" for k, v in worst))
        return cache[name]

    names = [p.stem for p in files]
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/origin", axes_length=0.3, axes_radius=0.005)

    with server.gui.add_folder("take"):
        g_take = server.gui.add_dropdown("段", names, initial_value=names[0])
        g_play = server.gui.add_checkbox("play", True)
        g_speed = server.gui.add_slider("speed", 0.1, 3.0, 0.1, 1.0)
        g_t = server.gui.add_slider("t [s]", 0.0, 1.0, 0.01, 0.0)
    with server.gui.add_folder("显示"):
        g_trails = server.gui.add_checkbox("轨迹", True)
        g_axes = server.gui.add_checkbox("朝向坐标轴 (6D)", True)
        g_links = server.gui.add_checkbox("骨架连线", True)
    with server.gui.add_folder("健康"):
        g_health = server.gui.add_text("冻结比例", "-")
        g_now = server.gui.add_text("此刻冻结", "-")

    state = {"name": names[0], "t0": time.time()}

    def color_of(name: str, i: int) -> tuple[int, int, int]:
        if name in COLORS:
            return COLORS[name]
        return RAW_COLORS[i % len(RAW_COLORS)]

    def rebuild(name: str) -> None:
        server.scene.reset()
        server.scene.add_frame("/origin", axes_length=0.3, axes_radius=0.005)
        t, P, Q, F = get(name)
        g_t.min, g_t.max, g_t.value = 0.0, float(t[-1]), 0.0
        g_health.value = "  ".join(
            f"{k} {100.0 * v.mean():.0f}%" for k, v in sorted(F.items()))
        state["name"], state["t0"] = name, time.time()

    rebuild(names[0])

    @g_take.on_update
    def _(_) -> None:
        rebuild(g_take.value)

    while True:
        name = state["name"]
        t, P, Q, F = get(name)
        if g_play.value and t[-1] > 0:
            el = (time.time() - state["t0"]) * g_speed.value
            g_t.value = float(el % t[-1])
        i = int(np.clip(np.searchsorted(t, g_t.value), 0, len(t) - 1))

        frozen_now = []
        for k, (name_, pos) in enumerate(P.items()):
            col = color_of(name_, k)
            is_frozen = bool(F[name_][i])
            if is_frozen:
                frozen_now.append(name_)
            if g_axes.value:
                server.scene.add_frame(f"/trk/{name_}", axes_length=0.12,
                                       axes_radius=0.006, position=pos[i],
                                       wxyz=Q[name_][i])
            server.scene.add_icosphere(f"/trk/{name_}/dot", radius=0.035,
                                       color=FROZEN if is_frozen else col,
                                       position=pos[i])
            server.scene.add_label(f"/trk/{name_}/tag",
                                   f"{name_}{' FROZEN' if is_frozen else ''}",
                                   position=pos[i] + np.array([0, 0, 0.08]))
            if g_trails.value and len(pos) > 3:
                server.scene.add_spline_catmull_rom(
                    f"/trail/{name_}", pos, color=col, line_width=1.5)
            elif not g_trails.value:
                server.scene.add_spline_catmull_rom(
                    f"/trail/{name_}", pos[i:i + 1].repeat(2, axis=0),
                    color=col, line_width=0.1)

        if g_links.value:
            for a, b in LINKS:
                if a in P and b in P:
                    server.scene.add_spline_catmull_rom(
                        f"/link/{a}_{b}", np.stack([P[a][i], P[b][i]]),
                        color=(150, 150, 150), line_width=3.0)

        g_now.value = ", ".join(frozen_now) if frozen_now else "无(全部在动)"
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    raise SystemExit(main())
