"""Split a video into per-hand interaction frames (any-object link rule).

For each hand, a frame counts as interacting when that hand has an hf link
(prob >= --min-link-prob) to ANY plausible object box.  Only obviously
unreasonable background boxes are filtered out (area above
--max-frame-area-fraction of the frame, or a box hugging 3+ image borders).
There is no per-hand task-object anchoring or tracking any more.

Qwen is used for two things only:
  1. deciding which hand track is LEFT / RIGHT (vote on sampled frames,
     average-x fallback), and
  2. metadata: task_type (single/dual) and each hand's primary object name
     -- recorded and displayed, never used for filtering.

Output is data-driven per hand: a hand with zero linked frames simply gets
no filtered_track.json / no downstream timeline.

Outputs (under --output-dir):
  dual_hand_analysis.json
  qwen_hand_queries/ , qwen_object_queries/
  {left,right}_hand/frames/*.jpg , kept_frames.mp4 , filtered_track.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..qwen_client import build_user_content, call_qwen, make_client
from .qwen_interaction_object_filter import (
    HAND_COLOR,
    LINK_COLOR,
    OBJECT_COLOR,
    SYSTEM_PROMPT,
    _box_area,
    _iou,
    _parse_json,
    _read_frames,
    _resize_width,
)

TRACK_COLORS = {"A": (60, 60, 230), "B": (230, 130, 40)}  # BGR
MIN_HAND_SCORE = 0.3
BORDER_MARGIN_PX = 12


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _reasonable_objects(frame: dict, width: int, height: int,
                        max_area_fraction: float) -> list[dict]:
    """Non-hand boxes that are not obvious full-background boxes."""
    frame_area = float(width * height)
    out = []
    for det in frame.get("detections", []):
        if det["class_name"] == "hand":
            continue
        x1, y1, x2, y2 = det["box_xyxy"]
        if _box_area(det["box_xyxy"]) > max_area_fraction * frame_area:
            continue
        borders = ((x1 <= BORDER_MARGIN_PX) + (y1 <= BORDER_MARGIN_PX)
                   + (x2 >= width - BORDER_MARGIN_PX)
                   + (y2 >= height - BORDER_MARGIN_PX))
        if borders >= 3:
            continue
        out.append(det)
    return out


def _hands_of_frame(frame: dict) -> list[dict]:
    hands = [d for d in frame.get("detections", [])
             if d["class_name"] == "hand" and d["score"] >= MIN_HAND_SCORE]
    hands.sort(key=lambda d: -d["score"])
    return hands[:2]


def build_hand_tracks(detections: dict) -> dict[int, dict[str, dict]]:
    """Assign per-frame hand detections to two stable tracks A / B."""
    frames = [f for f in detections["frames"] if f.get("processed")]
    assignment: dict[int, dict[str, dict]] = {}
    prev: dict[str, list[float]] = {}
    for frame in frames:
        idx = frame["frame_idx"]
        hands = _hands_of_frame(frame)
        if not hands:
            continue
        current: dict[str, dict] = {}
        if not prev:
            if len(hands) == 2:
                ordered = sorted(hands, key=lambda d: _center(d["box_xyxy"])[0])
                current["A"], current["B"] = ordered[0], ordered[1]
            else:
                current["A"] = hands[0]
        elif len(hands) == 2 and len(prev) == 2:
            straight = (_iou(hands[0]["box_xyxy"], prev["A"])
                        + _iou(hands[1]["box_xyxy"], prev["B"]))
            crossed = (_iou(hands[0]["box_xyxy"], prev["B"])
                       + _iou(hands[1]["box_xyxy"], prev["A"]))
            if straight >= crossed:
                current["A"], current["B"] = hands[0], hands[1]
            else:
                current["A"], current["B"] = hands[1], hands[0]
        else:
            for hand in hands:
                best_track, best_iou = None, 0.0
                for track, box in prev.items():
                    if track in current:
                        continue
                    iou = _iou(hand["box_xyxy"], box)
                    if iou > best_iou:
                        best_track, best_iou = track, iou
                if best_track is None:
                    for track in ("A", "B"):
                        if track not in current and track not in prev:
                            best_track = track
                            break
                if best_track is not None:
                    current[best_track] = hand
        for track, det in current.items():
            prev[track] = det["box_xyxy"]
        assignment[idx] = current
    return assignment


def qwen_label_hand_sides(
    video: str, hand_tracks: dict[int, dict[str, dict]], query_dir: Path,
) -> dict:
    """Vote with Qwen which track is the left / right hand (viewer perspective)."""
    two_hand_frames = [i for i, h in sorted(hand_tracks.items()) if len(h) == 2]
    votes = {"A": {"left": 0, "right": 0}, "B": {"left": 0, "right": 0}}
    qwen_answers = []
    if two_hand_frames:
        step = max(1, len(two_hand_frames) // 4)
        sample = two_hand_frames[::step][:4]
        images = _read_frames(video, set(sample))
        query_dir.mkdir(parents=True, exist_ok=True)
        paths, lines = [], [
            "Frames from one egocentric manipulation video.",
            "Two hand boxes are drawn: box A (red) and box B (orange).",
        ]
        for order, idx in enumerate(sample):
            img = images[idx].copy()
            for track, det in hand_tracks[idx].items():
                x1, y1, x2, y2 = (int(round(v)) for v in det["box_xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), TRACK_COLORS[track], 4)
                cv2.putText(img, track, (x1 + 6, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, TRACK_COLORS[track], 3,
                            cv2.LINE_AA)
            path = query_dir / f"hands_{order:02d}_frame_{idx:04d}.jpg"
            cv2.imwrite(str(path), _resize_width(img, 960))
            paths.append(path)
            lines.append(f"IMAGE {order + 1} = frame {idx}")
        lines += [
            "",
            "From the CAMERA VIEWER's perspective (left side of the image = left),",
            "which box is the person's left hand and which is the right hand,",
            "per image? Answer strict JSON:",
            '{"frames": [ {"frame_idx": <int>, "left": "A"|"B", "right": "A"|"B"}, ... ]}',
        ]
        try:
            reply = call_qwen(SYSTEM_PROMPT, build_user_content("\n".join(lines), paths),
                              client=make_client())
            qwen_answers = _parse_json(reply.content).get("frames", [])
            for item in qwen_answers:
                for side in ("left", "right"):
                    track = item.get(side)
                    if track in votes:
                        votes[track][side] += 1
        except Exception as exc:  # Qwen failure falls back to geometry below
            qwen_answers = [{"error": str(exc)}]

    avg_x = {}
    for track in ("A", "B"):
        xs = [_center(h[track]["box_xyxy"])[0]
              for h in hand_tracks.values() if track in h]
        avg_x[track] = float(np.mean(xs)) if xs else None
    if votes["A"]["left"] + votes["A"]["right"] + votes["B"]["left"] + votes["B"]["right"]:
        a_left = votes["A"]["left"] + votes["B"]["right"]
        a_right = votes["A"]["right"] + votes["B"]["left"]
        mapping = {"A": "left", "B": "right"} if a_left >= a_right else \
                  {"A": "right", "B": "left"}
        source = "qwen_vote"
    else:
        if avg_x["A"] is not None and avg_x["B"] is not None:
            mapping = ({"A": "left", "B": "right"} if avg_x["A"] <= avg_x["B"]
                       else {"A": "right", "B": "left"})
        else:
            mapping = {"A": "left", "B": "right"}
        source = "average_x_fallback"
    return {"mapping": mapping, "source": source, "votes": votes,
            "average_x": avg_x, "qwen_frames": qwen_answers}


def qwen_task_metadata(
    video: str,
    detections: dict,
    hand_tracks: dict[int, dict[str, dict]],
    side_of_track: dict[str, str],
    query_dir: Path,
    width: int,
    height: int,
    max_area_fraction: float,
    min_link_prob: float,
    max_query_frames: int,
) -> dict:
    """Metadata-only Qwen call: task type + each hand's primary object name.

    The answer is recorded and shown in overlays but never used to filter
    links or frames.
    """
    frames_by_idx = {f["frame_idx"]: f for f in detections["frames"]}
    linked = []
    for frame in detections["frames"]:
        if any(l["prob"] >= min_link_prob for l in frame.get("links", {}).get("hf", [])):
            if frame["frame_idx"] in hand_tracks:
                linked.append(frame["frame_idx"])
    if not linked:
        return {"task_type": None, "left_object": None, "right_object": None,
                "note": "no linked frames"}
    step = max(1, len(linked) // max_query_frames)
    sample = linked[::step][:max_query_frames]
    images = _read_frames(video, set(sample))
    query_dir.mkdir(parents=True, exist_ok=True)

    paths, lines = [], [
        "Frames from ONE egocentric manipulation video, in temporal order.",
        "Hand boxes are drawn with side labels L (left hand) and R (right hand).",
        "Candidate object boxes are drawn in yellow.",
        "",
    ]
    for order, idx in enumerate(sample):
        img = images[idx].copy()
        for det in _reasonable_objects(frames_by_idx[idx], width, height,
                                       max_area_fraction):
            x1, y1, x2, y2 = (int(round(v)) for v in det["box_xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), OBJECT_COLOR, 3)
        for track, det in hand_tracks[idx].items():
            side = side_of_track[track].upper()[0]
            x1, y1, x2, y2 = (int(round(v)) for v in det["box_xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), HAND_COLOR, 4)
            cv2.putText(img, side, (x1 + 6, min(int(y2) - 10, y1 + 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, HAND_COLOR, 4, cv2.LINE_AA)
        path = query_dir / f"objects_{order:02d}_frame_{idx:04d}.jpg"
        cv2.imwrite(str(path), _resize_width(img, 960))
        paths.append(path)
        lines.append(f"IMAGE {order + 1} = frame {idx}")
    lines += [
        "",
        "Definition: task_type is \"dual\" if BOTH hands actively manipulate,",
        "hold or stabilize objects at any point in the video -- even when the",
        "two hands act on the SAME object (e.g. one hand steadies a hub while",
        "the other inserts a plug into it). Use \"single\" ONLY when one hand",
        "does all the interaction and the other hand never touches any object.",
        "",
        "Answer strict JSON only:",
        "{",
        '  "task_type": "single" | "dual",',
        '  "left_object": "<short name of the object the LEFT hand mainly acts on, or null>",',
        '  "right_object": "<short name of the object the RIGHT hand mainly acts on, or null>"',
        "}",
    ]
    reply = call_qwen(SYSTEM_PROMPT, build_user_content("\n".join(lines), paths),
                      client=make_client())
    parsed = _parse_json(reply.content)
    parsed["_raw_reply"] = reply.content
    return parsed


def keep_frames_any_object(
    detections: dict,
    hand_tracks: dict[int, dict[str, dict]],
    track_id: str,
    width: int,
    height: int,
    max_area_fraction: float,
    min_link_prob: float,
) -> dict[int, dict]:
    """Frames where THIS hand links to ANY reasonable object box."""
    kept: dict[int, dict] = {}
    for frame in detections["frames"]:
        idx = frame["frame_idx"]
        hand = hand_tracks.get(idx, {}).get(track_id)
        if hand is None:
            continue
        dets = frame.get("detections", [])
        ok_targets = {d["detection_index"]
                      for d in _reasonable_objects(frame, width, height,
                                                   max_area_fraction)}
        links = []
        for link in frame.get("links", {}).get("hf", []):
            if link["prob"] < min_link_prob:
                continue
            if link["source_detection_index"] != hand["detection_index"]:
                continue
            if link["target_detection_index"] not in ok_targets:
                continue
            links.append({
                "prob": link["prob"],
                "hand_box": hand["box_xyxy"],
                "object_box": dets[link["target_detection_index"]]["box_xyxy"],
            })
        if links:
            kept[idx] = {"links": links}
    return kept


def render_hand_outputs(video: str, kept: dict[int, dict], side: str,
                        object_name: str, out_dir: Path, fps: float,
                        width: int, height: int) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    tmp = str(out_dir / "_tmp.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    cap = cv2.VideoCapture(video)
    idx = 0
    tag = f"{side.upper()} HAND  [{object_name}]"
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if idx in kept:
            for link in kept[idx]["links"]:
                ox1, oy1, ox2, oy2 = (int(round(v)) for v in link["object_box"])
                cv2.rectangle(img, (ox1, oy1), (ox2, oy2), OBJECT_COLOR, 6)
                hx1, hy1, hx2, hy2 = (int(round(v)) for v in link["hand_box"])
                cv2.rectangle(img, (hx1, hy1), (hx2, hy2), HAND_COLOR, 6)
                hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                cv2.line(img, (hcx, hcy), (ocx, ocy), (0, 0, 0), 13, cv2.LINE_AA)
                cv2.line(img, (hcx, hcy), (ocx, ocy), LINK_COLOR, 6, cv2.LINE_AA)
                cv2.putText(img, f"HF {link['prob']:.2f}",
                            ((hcx + ocx) // 2, (hcy + ocy) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINK_COLOR, 2, cv2.LINE_AA)
            cv2.putText(img, f"{tag}  frame {idx}", (18, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imwrite(str(frames_dir / f"frame_{idx:04d}.jpg"), img)
            writer.write(img)
        idx += 1
    cap.release()
    writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
         str(out_dir / "kept_frames.mp4")],
        check=True,
    )
    Path(tmp).unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-query-frames", type=int, default=8)
    parser.add_argument("--min-link-prob", type=float, default=0.5)
    parser.add_argument("--max-frame-area-fraction", type=float, default=0.7)
    args = parser.parse_args()

    detections = json.load(open(args.detections))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    hand_tracks = build_hand_tracks(detections)
    sides = qwen_label_hand_sides(args.video, hand_tracks,
                                  output_dir / "qwen_hand_queries")
    side_of_track = sides["mapping"]

    metadata = qwen_task_metadata(
        args.video, detections, hand_tracks, side_of_track,
        output_dir / "qwen_object_queries", width, height,
        args.max_frame_area_fraction, args.min_link_prob, args.max_query_frames)

    summary = {
        "link_rule": "any_reasonable_object",
        "max_frame_area_fraction": args.max_frame_area_fraction,
        "task_type_qwen": metadata.get("task_type"),
        "left_object": metadata.get("left_object"),
        "right_object": metadata.get("right_object"),
        "hand_side_assignment": sides,
        "hands": {},
    }
    for side in ("left", "right"):
        track_id = next((t for t, s in side_of_track.items() if s == side), None)
        if track_id is None:
            summary["hands"][side] = {"status": "no_hand_track"}
            continue
        kept = keep_frames_any_object(
            detections, hand_tracks, track_id, width, height,
            args.max_frame_area_fraction, args.min_link_prob)
        if not kept:
            summary["hands"][side] = {"track_id": track_id,
                                      "status": "no_linked_frames"}
            continue
        object_name = str(metadata.get(f"{side}_object") or "object")
        out_dir = output_dir / f"{side}_hand"
        render_hand_outputs(args.video, kept, side, object_name,
                            out_dir, fps, width, height)
        json.dump(
            {
                "task_object": f"{side.upper()} {object_name}",
                "hand_mode": "per_hand_any_object",
                "active_hand": side,
                "kept_frames": sorted(kept),
                "dropped_linked_frames": [],
                "per_frame": {str(k): kept[k] for k in sorted(kept)},
            },
            open(out_dir / "filtered_track.json", "w"),
            ensure_ascii=False, indent=1,
        )
        summary["hands"][side] = {"track_id": track_id, "status": "ok",
                                  "num_kept_frames": len(kept),
                                  "kept_frames": sorted(kept)}
    json.dump(summary, open(output_dir / "dual_hand_analysis.json", "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps({
        "task_type_qwen": summary["task_type_qwen"],
        "left": summary["hands"].get("left", {}).get("num_kept_frames", 0),
        "right": summary["hands"].get("right", {}).get("num_kept_frames", 0),
        "side_source": sides["source"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
