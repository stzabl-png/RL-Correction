"""Mark interaction segments from qwen-filtered task-object link frames.

Consumes filtered_track.json produced by qwen_interaction_object_filter and
groups the kept (hand<->task-object linked) frames into interaction segments:

  gap <= --max-bridge-gap            bridge silently (detector flicker)
  gap in (bridge, --max-arbitration-gap]
                                     ask Qwen whether the hand kept holding
                                     the object through the gap (occlusion)
  gap >  --max-arbitration-gap       split into separate segments

Segments spanning fewer than --min-segment-frames frames are dropped.

Outputs (under --output-dir):
  interaction_segments.json
  segments_timeline.mp4   H.264 preview: per-frame overlay + timeline bar
                          (green = segment, yellow = bridged gap, orange =
                          qwen-bridged gap, gray = outside segments)
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .qwen_client import build_user_content, call_qwen, make_client

SYSTEM_PROMPT = (
    "You are a precise vision assistant for robot-manipulation video analysis. "
    "You answer strictly in JSON with no extra text."
)

COLOR_SEGMENT = (90, 190, 90)
COLOR_BRIDGE = (60, 210, 230)
COLOR_QWEN_BRIDGE = (60, 140, 250)
COLOR_IDLE = (110, 110, 110)
COLOR_POINTER = (255, 255, 255)
HAND_COLOR = (60, 60, 230)
OBJECT_COLOR = (0, 200, 255)
LINK_COLOR = (255, 255, 255)


def _parse_json(content: str) -> dict:
    text = content.strip().strip("`")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON in Qwen reply: {content[:200]!r}")
    return json.loads(text[start : end + 1])


def _read_frames(video_path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    out, idx = {}, 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if idx in wanted:
            out[idx] = img
        idx += 1
    cap.release()
    return out


def _crop_region(track: dict, before: int, after: int,
                 width: int, height: int) -> tuple[int, int, int, int]:
    """Union of hand+object boxes at the gap boundaries, with margin."""
    boxes = []
    for idx in (before, after):
        info = track["per_frame"][str(idx)]
        fallback = info.get("object_box")
        if fallback:
            boxes.append(fallback)
        for link in info["links"]:
            boxes.append(link["hand_box"])
            object_box = link.get("object_box", fallback)
            if object_box:
                boxes.append(object_box)
    xs1, ys1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    xs2, ys2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    mx, my = 0.25 * (xs2 - xs1), 0.25 * (ys2 - ys1)
    return (max(0, int(xs1 - mx)), max(0, int(ys1 - my)),
            min(width, int(xs2 + mx)), min(height, int(ys2 + my)))


def arbitrate_gap_with_qwen(
    video: str, track: dict, object_name: str,
    before: int, after: int, width: int, height: int, scratch: Path,
) -> dict:
    """Ask Qwen whether the hand kept holding the object through [before+1, after-1]."""
    gap_frames = list(range(before + 1, after))
    if len(gap_frames) > 3:
        pick = [gap_frames[0], gap_frames[len(gap_frames) // 2], gap_frames[-1]]
    else:
        pick = gap_frames
    wanted = [before] + pick + [after]
    images = _read_frames(video, set(wanted))
    x1, y1, x2, y2 = _crop_region(track, before, after, width, height)
    scratch.mkdir(parents=True, exist_ok=True)
    paths, lines = [], [
        f"The person manipulates one object: {object_name}.",
        "Chronological crops from consecutive video moments around a detection gap:",
    ]
    for order, idx in enumerate(wanted):
        crop = images[idx][y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if w < 480:
            crop = cv2.resize(crop, (480, int(h * 480 / w)))
        path = scratch / f"gap_{before:04d}_{after:04d}_{order}_f{idx:04d}.jpg"
        cv2.imwrite(str(path), crop)
        paths.append(path)
        role = ("holding CONFIRMED by detector" if idx in (before, after)
                else "detection gap frame")
        lines.append(f"IMAGE {order + 1} = frame {idx} ({role})")
    lines += [
        "",
        "In the gap frames, is the hand still continuously holding the object",
        "(e.g. the object is occluded by the hand or motion blur), or did the",
        "hand actually release the object during the gap?",
        'Answer strict JSON: {"holding": true|false, "reason": "<short>"}',
    ]
    response = call_qwen(SYSTEM_PROMPT, build_user_content("\n".join(lines), paths),
                         client=make_client())
    parsed = _parse_json(response.content)
    parsed["gap"] = [before + 1, after - 1]
    return parsed


def build_segments(track: dict, args: argparse.Namespace, video: str,
                   width: int, height: int, output_dir: Path) -> tuple[list, list]:
    kept = sorted(int(f) for f in track["kept_frames"])
    object_name = str(track.get("task_object", "object"))
    segments, arbitrations = [], []
    current = {"start": kept[0], "end": kept[0], "frames": [kept[0]],
               "bridged_gaps": [], "qwen_bridged_gaps": []}
    for frame in kept[1:]:
        gap = frame - current["end"] - 1
        decision = "continue"
        if gap == 0:
            pass
        elif gap <= args.max_bridge_gap:
            current["bridged_gaps"].append([current["end"] + 1, frame - 1])
        elif gap <= args.max_arbitration_gap and args.qwen_arbitration:
            verdict = arbitrate_gap_with_qwen(
                video, track, object_name, current["end"], frame,
                width, height, output_dir / "qwen_gap_queries")
            arbitrations.append(verdict)
            if verdict.get("holding") is True:
                current["qwen_bridged_gaps"].append([current["end"] + 1, frame - 1])
            else:
                decision = "split"
        else:
            decision = "split"
        if decision == "split":
            segments.append(current)
            current = {"start": frame, "end": frame, "frames": [],
                       "bridged_gaps": [], "qwen_bridged_gaps": []}
        current["end"] = frame
        current["frames"].append(frame)
    segments.append(current)

    final = []
    for seg in segments:
        span = seg["end"] - seg["start"] + 1
        if span < args.min_segment_frames:
            continue
        probs = []
        for f in seg["frames"]:
            info = track["per_frame"][str(f)]
            probs.extend(l["prob"] for l in info["links"])
        final.append({
            "segment_idx": len(final),
            "start_frame": seg["start"],
            "end_frame": seg["end"],
            "span_frames": span,
            "num_linked_frames": len(seg["frames"]),
            "mean_link_prob": round(float(np.mean(probs)), 4) if probs else None,
            "bridged_gaps": seg["bridged_gaps"],
            "qwen_bridged_gaps": seg["qwen_bridged_gaps"],
        })
    return final, arbitrations


def render_timeline(video: str, track: dict, segments: list,
                    output_dir: Path, object_name: str) -> None:
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bar_h, pad = 110, 14
    banner_h = 88
    frame_color = np.full(total, 0, dtype=np.int32)  # 0 idle 1 seg 2 bridge 3 qwen
    seg_of_frame = np.full(total, -1, dtype=np.int32)
    for seg in segments:
        frame_color[seg["start_frame"]: seg["end_frame"] + 1] = 1
        seg_of_frame[seg["start_frame"]: seg["end_frame"] + 1] = seg["segment_idx"]
        for a, b in seg["bridged_gaps"]:
            frame_color[a: b + 1] = 2
        for a, b in seg["qwen_bridged_gaps"]:
            frame_color[a: b + 1] = 3
    palette = {0: COLOR_IDLE, 1: COLOR_SEGMENT, 2: COLOR_BRIDGE, 3: COLOR_QWEN_BRIDGE}

    def _x_of(frame: int) -> int:
        return min(width - 1, int(frame * width / max(1, total)))

    bar = np.full((bar_h, width, 3), 25, dtype=np.uint8)
    for x in range(width):
        f = min(total - 1, int(x * total / width))
        bar[pad: bar_h - pad, x] = palette[int(frame_color[f])]
    # segment boundary labels on the bar
    for seg in segments:
        for frame, align_right in ((seg["start_frame"], False), (seg["end_frame"], True)):
            x = _x_of(frame)
            cv2.line(bar, (x, 2), (x, bar_h - 2), (255, 255, 255), 2)
            text = str(frame)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            tx = max(4, x - tw - 6) if align_right else min(width - tw - 4, x + 6)
            cv2.putText(bar, text, (tx, bar_h - pad - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)

    tmp = str(output_dir / "_timeline_tmp.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height + banner_h + bar_h))
    idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        info = track["per_frame"].get(str(idx))
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
        seg_idx = int(seg_of_frame[idx]) if idx < total else -1
        if seg_idx >= 0:
            # thick full-frame border while interacting
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), COLOR_SEGMENT, 18)

        # solid banner above the frame
        banner_color = COLOR_SEGMENT if seg_idx >= 0 else (55, 55, 55)
        banner = np.full((banner_h, width, 3), banner_color, dtype=np.uint8)
        if seg_idx >= 0:
            text = f"INTERACTION  SEGMENT {seg_idx}  [{object_name}]   frame {idx}"
            text_color = (20, 40, 20)
        else:
            text = f"no interaction   frame {idx}"
            text_color = (170, 170, 170)
        cv2.putText(banner, text, (24, banner_h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, text_color, 4, cv2.LINE_AA)

        strip = bar.copy()
        x = _x_of(idx)
        cv2.line(strip, (x, 0), (x, bar_h), COLOR_POINTER, 5)
        pts = np.array([[x - 14, 0], [x + 14, 0], [x, 24]], dtype=np.int32)
        cv2.fillPoly(strip, [pts], COLOR_POINTER)
        writer.write(np.vstack([banner, img, strip]))
        idx += 1
    cap.release()
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
         str(output_dir / "segments_timeline.mp4")],
        check=True,
    )
    Path(tmp).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--filtered-track", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-bridge-gap", type=int, default=4)
    parser.add_argument("--max-arbitration-gap", type=int, default=15)
    parser.add_argument("--min-segment-frames", type=int, default=6)
    parser.add_argument("--qwen-arbitration", action="store_true",
                        help="ask Qwen whether medium gaps (bridge..arbitration) are "
                             "occlusions; default OFF -- such gaps simply split")
    args = parser.parse_args()

    track = json.load(open(args.filtered_track))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    segments, arbitrations = build_segments(track, args, args.video,
                                            width, height, output_dir)
    object_name = str(track.get("task_object", "object"))
    json.dump(
        {
            "task_object": object_name,
            "hand_mode": track.get("hand_mode"),
            "active_hand": track.get("active_hand"),
            "config": {
                "max_bridge_gap": args.max_bridge_gap,
                "max_arbitration_gap": args.max_arbitration_gap,
                "min_segment_frames": args.min_segment_frames,
                "qwen_arbitration": args.qwen_arbitration,
            },
            "segments": segments,
            "qwen_gap_arbitrations": arbitrations,
        },
        open(output_dir / "interaction_segments.json", "w"),
        ensure_ascii=False, indent=1,
    )
    render_timeline(args.video, track, segments, output_dir, object_name)
    print(json.dumps({
        "task_object": object_name,
        "segments": [[s["start_frame"], s["end_frame"]] for s in segments],
        "qwen_arbitrations": len(arbitrations),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
