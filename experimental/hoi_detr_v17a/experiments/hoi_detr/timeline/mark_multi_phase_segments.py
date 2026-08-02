"""Baseline multi-cycle approach-interact-leave segmentation per hand.

Relies ONLY on HOI-DETR hf-link frames (as recorded in per_hand_links.json by
render_per_hand_interaction_video): no pose, no co-motion, no Qwen.

Rule (deliberately minimal):
  1. Per hand, linked frames -> interaction segments, bridging gaps
     <= --bridge-gap frames and dropping segments shorter than --min-seg.
  2. Every remaining frame belongs to exactly one cycle: the gap between two
     consecutive interaction segments is split at its midpoint -- first half is
     the LEAVE of the previous cycle, second half the APPROACH of the next.
     Frames before the first segment are APPROACH of cycle 1; frames after the
     last segment are LEAVE of the last cycle.

Outputs (under --output-dir):
  {side}_hand_multi_phase.mp4   full-length visualization (H.264+faststart):
                                banner + border colored by phase, cycle index,
                                this hand's link boxes, bottom phase timeline
  {side}_hand_multi_phase.json  cycle/phase boundaries + parameters
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .dual_hand_frame_split import build_hand_tracks
from .qwen_interaction_object_filter import HAND_COLOR, LINK_COLOR, OBJECT_COLOR

# BGR phase colors: approach=blue, interact=green, leave=orange
PHASE_COLOR = {"approach": (246, 130, 50), "interact": (80, 220, 80),
               "leave": (30, 150, 255)}
PHASE_CN = {"approach": "APPROACH", "interact": "INTERACT", "leave": "LEAVE"}

# progress-bar geometry (bottom overlay)
BAR_BG_H = 96      # opaque backdrop height
BAR_H = 44         # colored bar height
BAR_MARGIN_X = 18  # left/right margin of the bar


def link_segments(linked: list[int], bridge_gap: int, min_seg: int) -> list[tuple[int, int]]:
    if not linked:
        return []
    linked = sorted(set(linked))
    segs = [[linked[0], linked[0]]]
    for f in linked[1:]:
        if f - segs[-1][1] <= bridge_gap + 1:
            segs[-1][1] = f
        else:
            segs.append([f, f])
    return [(a, b) for a, b in segs if b - a + 1 >= min_seg]


def phase_of_frames(segs: list[tuple[int, int]], num_frames: int) -> list[tuple[str, int]]:
    """Per-frame (phase, cycle_idx). Cycle boundaries at gap midpoints."""
    out: list[tuple[str, int]] = [("approach", 0)] * num_frames
    if not segs:
        return [("approach", 0)] * num_frames
    bounds = [0]
    for (_, e_prev), (s_next, _) in zip(segs, segs[1:]):
        bounds.append((e_prev + 1 + s_next) // 2)
    bounds.append(num_frames)
    for k, (s, e) in enumerate(segs):
        lo, hi = bounds[k], bounds[k + 1]
        for f in range(lo, min(s, num_frames)):
            out[f] = ("approach", k)
        for f in range(s, min(e + 1, num_frames)):
            out[f] = ("interact", k)
        for f in range(e + 1, hi):
            out[f] = ("leave", k)
    return out


def draw_timeline_strip(img, phases, segs, idx, width, height):
    """Prominent bottom progress bar: opaque backdrop, bright per-phase colors,
    white phase separators, frame ticks, legend and a triangle cursor."""
    n = len(phases)
    bar_x0, bar_x1 = BAR_MARGIN_X, width - BAR_MARGIN_X
    bar_w = bar_x1 - bar_x0
    bg_y0 = height - BAR_BG_H
    bar_y0 = bg_y0 + 34
    bar_y1 = bar_y0 + BAR_H

    def x_of(f):
        return bar_x0 + int(f / max(n - 1, 1) * (bar_w - 1))

    # opaque dark backdrop so the bar reads on any footage
    cv2.rectangle(img, (0, bg_y0), (width, height), (28, 24, 20), -1)

    # phase runs -> colored blocks
    runs = []
    for f in range(n):
        p = phases[f][0]
        if runs and runs[-1][0] == p:
            runs[-1][2] = f
        else:
            runs.append([p, f, f])
    for p, a, b in runs:
        cv2.rectangle(img, (x_of(a), bar_y0), (x_of(b + 1) if b + 1 < n else bar_x1, bar_y1),
                      PHASE_COLOR[p], -1)
    # white separators between phases + interaction frame ticks
    for p, a, b in runs[1:]:
        cv2.line(img, (x_of(a), bar_y0 - 4), (x_of(a), bar_y1 + 4), (255, 255, 255), 2)
    for s, e in segs:
        for f in (s, e):
            cv2.putText(img, str(f), (max(x_of(f) - 16, bar_x0), bar_y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(img, (bar_x0, bar_y0), (bar_x1, bar_y1), (255, 255, 255), 2)

    # cursor: white triangle above + line through the bar
    cx = x_of(min(idx, n - 1))
    cv2.line(img, (cx, bar_y0 - 2), (cx, bar_y1 + 2), (255, 255, 255), 3)
    pts = ((cx - 9, bar_y0 - 16), (cx + 9, bar_y0 - 16), (cx, bar_y0 - 3))
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], (255, 255, 255))

    # legend, left-aligned above the bar
    lx = bar_x0
    for p in ("approach", "interact", "leave"):
        cv2.rectangle(img, (lx, bg_y0 + 8), (lx + 22, bg_y0 + 26), PHASE_COLOR[p], -1)
        cv2.putText(img, PHASE_CN[p], (lx + 28, bg_y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
        lx += 200


def render(video, side, hand_tracks, track_id, per_frame, phases, segs,
           out_path, fps, width, height):
    tmp = str(out_path.with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    cap = cv2.VideoCapture(video)
    idx = 0
    n_cycles = len(segs)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        phase, cyc = phases[idx] if idx < len(phases) else ("leave", n_cycles - 1)
        color = PHASE_COLOR[phase]
        hand = hand_tracks.get(idx, {}).get(track_id)
        if hand is not None:
            hx1, hy1, hx2, hy2 = (int(round(v)) for v in hand["box_xyxy"])
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), HAND_COLOR, 4)
        rec = per_frame.get(str(idx))
        if rec and hand is not None:
            hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
            for link in rec["links"]:
                ox1, oy1, ox2, oy2 = (int(round(v)) for v in link["object_box"])
                cv2.rectangle(img, (ox1, oy1), (ox2, oy2), OBJECT_COLOR, 5)
                ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                cv2.line(img, (hcx, hcy), (ocx, ocy), (0, 0, 0), 11, cv2.LINE_AA)
                cv2.line(img, (hcx, hcy), (ocx, ocy), LINK_COLOR, 5, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), color, 12)
        cv2.rectangle(img, (0, 0), (width, 64), color, -1)
        cv2.putText(img, f"{side.upper()} HAND  cycle {cyc + 1}/{n_cycles}  "
                         f"{PHASE_CN[phase]}  frame {idx}",
                    (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                    cv2.LINE_AA)
        draw_timeline_strip(img, phases, segs, idx, width, height)
        writer.write(img)
        idx += 1
    cap.release()
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
         str(out_path)],
        check=True,
    )
    Path(tmp).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--per-hand-links", required=True,
                        help="per_hand_links.json from render_per_hand_interaction_video")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bridge-gap", type=int, default=4)
    parser.add_argument("--min-seg", type=int, default=6)
    args = parser.parse_args()

    detections = json.load(open(args.detections))
    links = json.load(open(args.per_hand_links))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    hand_tracks = build_hand_tracks(detections)
    report = {}
    for side in ("left", "right"):
        info = links["sides"].get(side, {})
        if info.get("status") != "ok":
            report[side] = {"status": info.get("status", "missing")}
            continue
        segs = link_segments(info["linked_frames"], args.bridge_gap, args.min_seg)
        if not segs:
            report[side] = {"status": "no_segments_after_filter"}
            continue
        phases = phase_of_frames(segs, num_frames)
        out_path = output_dir / f"{side}_hand_multi_phase.mp4"
        render(args.video, side, hand_tracks, info["track_id"],
               info["per_frame"], phases, segs, out_path, fps, width, height)
        cycles = []
        for k, (s, e) in enumerate(segs):
            frames_k = [f for f, (p, c) in enumerate(phases) if c == k]
            cycles.append({
                "cycle": k + 1,
                "approach": [min(frames_k), s - 1] if min(frames_k) < s else None,
                "interact": [s, e],
                "leave": [e + 1, max(frames_k)] if max(frames_k) > e else None,
            })
        report[side] = {"status": "ok", "num_cycles": len(segs),
                        "cycles": cycles, "video": str(out_path)}
        json.dump({"bridge_gap": args.bridge_gap, "min_seg": args.min_seg,
                   "num_frames": num_frames, "cycles": cycles},
                  open(output_dir / f"{side}_hand_multi_phase.json", "w"),
                  ensure_ascii=False, indent=1)
    print(json.dumps({s: {"cycles": r.get("num_cycles"), "status": r["status"]}
                      for s, r in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
