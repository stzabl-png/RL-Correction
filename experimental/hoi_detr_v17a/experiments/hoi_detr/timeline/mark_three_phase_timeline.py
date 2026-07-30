"""Collapse per-hand interaction segments into a 3-phase timeline.

approach -- interaction -- leave

The interaction phase spans from the START of the EARLIEST interaction
segment to the END of the LATEST one (gaps between segments are treated as
part of the overall interaction process).  Approach is everything before,
leave is everything after; either may be empty when the hand is already /
still interacting at the video boundary.

Inputs are the artifacts of the existing pipeline (no model inference):
  --segments        interaction_segments.json  (from mark_interaction_segments)
  --filtered-track  filtered_track.json        (for per-frame overlay drawing)

Outputs (under --output-dir):
  three_phase.json
  three_phase_timeline.mp4   H.264+faststart preview: phase banner + border,
                             bottom bar blue=approach green=interaction
                             orange=leave, white pointer
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .qwen_interaction_object_filter import HAND_COLOR, LINK_COLOR, OBJECT_COLOR

PHASE_COLORS = {          # BGR
    "approach": (235, 170, 70),
    "interaction": (90, 190, 90),
    "leave": (60, 140, 250),
}
PHASE_TEXT = {"approach": "APPROACH", "interaction": "INTERACTION", "leave": "LEAVE"}
COLOR_POINTER = (255, 255, 255)


def compute_phases(segments: list[dict], total_frames: int) -> list[dict]:
    if not segments:
        raise SystemExit("no interaction segments; cannot build 3-phase timeline")
    start = min(s["start_frame"] for s in segments)
    end = max(s["end_frame"] for s in segments)
    phases = []
    if start > 0:
        phases.append({"phase": "approach", "start_frame": 0, "end_frame": start - 1})
    phases.append({"phase": "interaction", "start_frame": start, "end_frame": end})
    if end < total_frames - 1:
        phases.append({"phase": "leave", "start_frame": end + 1,
                       "end_frame": total_frames - 1})
    for p in phases:
        p["num_frames"] = p["end_frame"] - p["start_frame"] + 1
    return phases


def render(video: str, phases: list[dict], per_frame: dict, object_name: str,
           output_dir: Path) -> None:
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    phase_of_frame = {}
    for p in phases:
        for f in range(p["start_frame"], p["end_frame"] + 1):
            phase_of_frame[f] = p["phase"]

    bar_h, pad, banner_h = 110, 14, 88

    def _x_of(frame: int) -> int:
        return min(width - 1, int(frame * width / max(1, total)))

    bar = np.full((bar_h, width, 3), 25, dtype=np.uint8)
    for x in range(width):
        f = min(total - 1, int(x * total / width))
        bar[pad: bar_h - pad, x] = PHASE_COLORS.get(phase_of_frame.get(f, "approach"),
                                                    (110, 110, 110))
    inter = next(p for p in phases if p["phase"] == "interaction")
    for frame, align_right in ((inter["start_frame"], False),
                               (inter["end_frame"], True)):
        x = _x_of(frame)
        cv2.line(bar, (x, 2), (x, bar_h - 2), (255, 255, 255), 2)
        text = str(frame)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        tx = max(4, x - tw - 6) if align_right else min(width - tw - 4, x + 6)
        cv2.putText(bar, text, (tx, bar_h - pad - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)

    tmp = str(output_dir / "_tmp.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height + banner_h + bar_h))
    idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        phase = phase_of_frame.get(idx, "approach")
        color = PHASE_COLORS[phase]
        info = per_frame.get(str(idx))
        if info is not None:
            fallback = info.get("object_box")
            for link in info["links"]:
                object_box = link.get("object_box", fallback)
                if object_box is None:
                    continue
                ox1, oy1, ox2, oy2 = (int(round(v)) for v in object_box)
                cv2.rectangle(img, (ox1, oy1), (ox2, oy2), OBJECT_COLOR, 6)
                ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                hx1, hy1, hx2, hy2 = (int(round(v)) for v in link["hand_box"])
                cv2.rectangle(img, (hx1, hy1), (hx2, hy2), HAND_COLOR, 6)
                hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                cv2.line(img, (hcx, hcy), (ocx, ocy), (0, 0, 0), 13, cv2.LINE_AA)
                cv2.line(img, (hcx, hcy), (ocx, ocy), LINK_COLOR, 6, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), color, 18)

        banner = np.full((banner_h, width, 3), color, dtype=np.uint8)
        cv2.putText(banner, f"{PHASE_TEXT[phase]}  [{object_name}]   frame {idx}",
                    (24, banner_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (25, 35, 25), 4, cv2.LINE_AA)

        strip = bar.copy()
        x = _x_of(idx)
        cv2.line(strip, (x, 0), (x, bar_h), COLOR_POINTER, 5)
        cv2.fillPoly(strip, [np.array([[x - 14, 0], [x + 14, 0], [x, 24]],
                                      dtype=np.int32)], COLOR_POINTER)
        writer.write(np.vstack([banner, img, strip]))
        idx += 1
    cap.release()
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
         str(output_dir / "three_phase_timeline.mp4")],
        check=True,
    )
    Path(tmp).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--segments", required=True,
                        help="interaction_segments.json from mark_interaction_segments")
    parser.add_argument("--filtered-track", required=True,
                        help="filtered_track.json for per-frame overlays")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    seg_data = json.load(open(args.segments))
    track = json.load(open(args.filtered_track))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    phases = compute_phases(seg_data["segments"], total)
    object_name = str(seg_data.get("task_object", "object"))
    json.dump(
        {
            "task_object": object_name,
            "hand_mode": seg_data.get("hand_mode"),
            "active_hand": seg_data.get("active_hand"),
            "total_frames": total,
            "phases": phases,
            "merged_from_segments": [[s["start_frame"], s["end_frame"]]
                                     for s in seg_data["segments"]],
        },
        open(output_dir / "three_phase.json", "w"), ensure_ascii=False, indent=1,
    )
    render(args.video, phases, track.get("per_frame", {}), object_name, output_dir)
    print(json.dumps({
        "task_object": object_name,
        "phases": {p["phase"]: [p["start_frame"], p["end_frame"]] for p in phases},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
