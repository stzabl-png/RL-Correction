#!/usr/bin/env python
"""Viser review bench: the operator's wrist path and the robot's EE path,
overlaid, for the whole run - no headset, no hardware.

What it shows
-------------
--map waist-abs is a 1:1 law: robot(waist -> EE) = human(waist -> wrist).
So if the mapping is right, the two curves are the SAME curve. This viewer
draws them in one frame and lets you scrub time:

  green   human wrist path       (wrist relative to the waist, from the capture)
  orange  robot EE path          (EE relative to the robot's waist bind point)
  red     the gap between them at the current instant

That makes the mapping judgeable the way the operator judges it - over the
whole trajectory and on arm shape - instead of by a single hold-endpoint
number, which is the reporting habit the user rejected on 2026-07-28.

Input is the CSV from `sim/teleop_vega_pico.py --trace-csv`, which already
contains all three stages (human input, commanded target, measured EE).

Run (viser lives in the main py3.11 venv):
    .venv/bin/python scripts/viser_trace_view.py TRACE.csv [TRACE2.csv ...]
then open the printed URL.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
import viser

HAND_COLORS = {
    # (human path, robot path) - hues split by hand so both arms stay readable
    "right": ((51, 217, 89), (255, 140, 26)),
    "left": ((26, 179, 153), (242, 89, 140)),
}
GAP_COLOR = (255, 26, 26)


def load_trace(path: pathlib.Path) -> dict[str, dict[str, np.ndarray]]:
    """CSV -> {hand: {t, human, cmd, ee, err_mm}} as arrays."""
    rows: dict[str, list] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["hand"], []).append(r)
    out = {}
    for hand, rs in rows.items():
        out[hand] = {
            "t": np.array([float(r["t"]) for r in rs]),
            "human": np.array([[float(r["human_x"]), float(r["human_y"]),
                                float(r["human_z"])] for r in rs]),
            "cmd": np.array([[float(r["cmd_x"]), float(r["cmd_y"]),
                              float(r["cmd_z"])] for r in rs]),
            "ee": np.array([[float(r["ee_x"]), float(r["ee_y"]),
                             float(r["ee_z"])] for r in rs]),
            "err_mm": np.array([float(r["err_mm"]) for r in rs]),
        }
    return out


def bind_point(d: dict[str, np.ndarray]) -> np.ndarray:
    """Robot-side waist bind point, recovered from the run itself.

    waist-abs commands cmd = bind + human*scale, so the offset is the median of
    (cmd - human). Taken from the data rather than re-derived from flags, so the
    viewer can never disagree with the run it is showing. The engage ramp (1.5 s)
    blends from the home pose, so early samples are excluded by the median.
    """
    return np.median(d["cmd"] - d["human"], axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", type=pathlib.Path)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tail", type=float, default=1.5,
                    help="seconds of trailing path drawn around the cursor "
                         "(0 = draw the whole path at once)")
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    traces = {p.stem: load_trace(p) for p in args.traces}
    names = list(traces)
    print(f"[trace] loaded {len(names)} run(s): {', '.join(names)}")

    with server.gui.add_folder("run"):
        gui_run = server.gui.add_dropdown("segment", names, initial_value=names[0])
        gui_play = server.gui.add_checkbox("play", True)
        gui_speed = server.gui.add_slider("speed", 0.1, 2.0, 0.1, 1.0)
        gui_whole = server.gui.add_checkbox("draw whole path", False)
    with server.gui.add_folder("readout"):
        gui_t = server.gui.add_slider("t [s]", 0.0, 1.0, 0.01, 0.0)
        gui_err = server.gui.add_text("gap now", "-")
        gui_stat = server.gui.add_text("whole run", "-")

    state = {"name": names[0], "t0": time.time()}

    def rebuild(name: str) -> None:
        """Draw the static geometry for a run: full paths + the robot origin."""
        server.scene.reset()
        d = traces[name]
        for hand, dd in d.items():
            c_h, c_r = HAND_COLORS.get(hand, ((128, 128, 128), (230, 230, 230)))
            b = bind_point(dd)
            server.scene.add_spline_catmull_rom(
                f"/{hand}/human_full", dd["human"], color=c_h, line_width=1.0)
            server.scene.add_spline_catmull_rom(
                f"/{hand}/robot_full", dd["ee"] - b, color=c_r, line_width=1.0)
        server.scene.add_frame("/origin", axes_length=0.1, axes_radius=0.003)
        stats = "  ".join(
            f"{h}: mean {dd['err_mm'].mean():.1f} / max {dd['err_mm'].max():.1f} mm"
            for h, dd in d.items())
        gui_stat.value = stats
        t = next(iter(d.values()))["t"]
        gui_t.min, gui_t.max = float(t[0]), float(t[-1])
        gui_t.value = float(t[0])
        state["name"] = name
        state["t0"] = time.time()

    rebuild(names[0])

    @gui_run.on_update
    def _(_) -> None:
        rebuild(gui_run.value)

    while True:
        d = traces[state["name"]]
        t_all = next(iter(d.values()))["t"]
        if gui_play.value:
            span = float(t_all[-1] - t_all[0])
            elapsed = (time.time() - state["t0"]) * gui_speed.value
            gui_t.value = float(t_all[0]) + (elapsed % span if span > 0 else 0.0)
        t_now = gui_t.value

        parts = []
        for hand, dd in d.items():
            c_h, c_r = HAND_COLORS.get(hand, ((128, 128, 128), (230, 230, 230)))
            b = bind_point(dd)
            i = int(np.searchsorted(dd["t"], t_now))
            i = min(max(i, 0), len(dd["t"]) - 1)
            hp, rp = dd["human"][i], dd["ee"][i] - b
            server.scene.add_icosphere(f"/{hand}/human_now", radius=0.018,
                                       color=c_h, position=hp)
            server.scene.add_icosphere(f"/{hand}/robot_now", radius=0.018,
                                       color=c_r, position=rp)
            # the gap IS the tracking error, drawn to scale: invisible = good
            server.scene.add_spline_catmull_rom(
                f"/{hand}/gap", np.stack([hp, rp]), color=GAP_COLOR,
                line_width=4.0)
            if not gui_whole.value and args.tail > 0:
                j = int(np.searchsorted(dd["t"], t_now - args.tail))
                seg_h, seg_r = dd["human"][j:i + 1], dd["ee"][j:i + 1] - b
                if len(seg_h) > 2:
                    server.scene.add_spline_catmull_rom(
                        f"/{hand}/human_tail", seg_h, color=c_h, line_width=4.0)
                    server.scene.add_spline_catmull_rom(
                        f"/{hand}/robot_tail", seg_r, color=c_r, line_width=4.0)
            parts.append(f"{hand} {dd['err_mm'][i]:.1f}mm")
        gui_err.value = "  ".join(parts)
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    raise SystemExit(main())
