"""Qwen-guided interaction-object link filtering.

Step 1: Qwen reads sampled annotated frames and decides
  - what the true task/interaction object is (name + per-frame box label),
  - whether the task is single- or dual-hand, and which hand is active
    (hand info is recorded for later use but not consumed yet).
Step 2: the chosen boxes are linked into a per-frame track by IoU
  association, then hf links are filtered so only hand<->interaction-object
  lines survive; frames whose interaction object has no link are dropped.

Outputs (per video, under --output-dir):
  qwen_query_frames/     annotated frames sent to Qwen
  qwen_task_analysis.json
  filtered_track.json    per-frame track box / kept links
  filtered_frames/       kept frames drawn with only the task-object link
  kept_frames.mp4        H.264 video of kept frames only
  full_overlay.mp4       all frames; dropped frames are dimmed and tagged
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from .qwen_client import build_user_content, call_qwen, make_client

HAND_COLOR = (60, 60, 230)      # BGR red-ish
OBJECT_COLOR = (0, 200, 255)    # amber
LINK_COLOR = (255, 255, 255)
LABEL_BG = (30, 30, 30)

SYSTEM_PROMPT = (
    "You are a precise vision assistant for robot-manipulation video analysis. "
    "You answer strictly in JSON with no extra text."
)


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _candidate_objects(frame: dict, frame_area: float, max_area_fraction: float) -> list[dict]:
    """Non-hand boxes small enough to be real objects (drops giant background boxes)."""
    out = []
    for det in frame.get("detections", []):
        if det["class_name"] == "hand":
            continue
        if _box_area(det["box_xyxy"]) > max_area_fraction * frame_area:
            continue
        out.append(det)
    return out


def _linked_frames(detections: dict, min_link_prob: float, max_area_fraction: float,
                   frame_area: float) -> list[int]:
    """Frames with at least one valid hand->object link to a plausible object box."""
    frames = []
    for frame in detections["frames"]:
        if not frame.get("processed"):
            continue
        dets = frame.get("detections", [])
        ok = False
        for link in frame.get("links", {}).get("hf", []):
            if link["prob"] < min_link_prob:
                continue
            target = dets[link["target_detection_index"]]
            if target["class_name"] == "hand":
                continue
            if _box_area(target["box_xyxy"]) > max_area_fraction * frame_area:
                continue
            ok = True
            break
        if ok:
            frames.append(frame["frame_idx"])
    return frames


def _read_frames(video_path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    out: dict[int, np.ndarray] = {}
    idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if idx in wanted:
            out[idx] = img
        idx += 1
    cap.release()
    return out


def _annotate_query_frame(img: np.ndarray, candidates: list[dict]) -> np.ndarray:
    vis = img.copy()
    for label, det in enumerate(candidates):
        x1, y1, x2, y2 = (int(round(v)) for v in det["box_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), OBJECT_COLOR, 3)
        text = str(label)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.rectangle(vis, (x1, y1 - th - 14), (x1 + tw + 12, y1), LABEL_BG, -1)
        cv2.putText(vis, text, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    OBJECT_COLOR, 3, cv2.LINE_AA)
    return vis


def _resize_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)))


def _parse_json(content: str) -> dict:
    text = content.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in Qwen reply: {content[:200]!r}")
    return json.loads(text[start : end + 1])


def ask_qwen_task_object(
    query_frames: list[tuple[int, np.ndarray, list[dict]]],
    query_dir: Path,
) -> dict:
    """One multi-image Qwen call: task object per frame, hand mode, active hand."""
    query_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    lines = [
        "These images are sampled frames from ONE manipulation video, in temporal order.",
        "In each image, candidate object boxes are drawn with numeric labels "
        "(labels restart from 0 in every image; they are per-image, not tracks).",
        "The person interacts with exactly one task object in this video.",
        "",
    ]
    for order, (frame_idx, img, candidates) in enumerate(query_frames):
        path = query_dir / f"query_{order:02d}_frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(path), _resize_width(_annotate_query_frame(img, candidates), 960))
        image_paths.append(path)
        lines.append(f"IMAGE {order + 1} = frame {frame_idx}, labels 0..{len(candidates) - 1}")
    lines += [
        "",
        "Answer in strict JSON only:",
        "{",
        '  "task_object": "<short name of the manipulated object>",',
        '  "hand_mode": "single" | "dual",',
        '  "active_hand": "left" | "right" | "both",',
        '  "frames": [ {"frame_idx": <int>, "box_label": <int or null>}, ... one per image ]',
        "}",
        'Use box_label null when the task object has no labeled box in that image.',
        '"active_hand" is from the camera viewer\'s perspective (left side of image = left).',
    ]
    response = call_qwen(
        SYSTEM_PROMPT,
        build_user_content("\n".join(lines), image_paths),
        client=make_client(),
    )
    parsed = _parse_json(response.content)
    parsed["_raw_reply"] = response.content
    return parsed


def build_track(
    detections: dict,
    anchors: dict[int, list[float]],
    frame_area: float,
    max_area_fraction: float,
    min_track_iou: float,
) -> dict[int, dict]:
    """Associate the anchored object across all frames by IoU (forward + backward)."""
    frames = {f["frame_idx"]: f for f in detections["frames"] if f.get("processed")}
    order = sorted(frames)
    track: dict[int, dict] = {}
    for idx in sorted(anchors):
        frame = frames.get(idx)
        if frame is None:
            continue
        best, best_iou = None, 0.0
        for det in _candidate_objects(frame, frame_area, max_area_fraction):
            iou = _iou(det["box_xyxy"], anchors[idx])
            if iou > best_iou:
                best, best_iou = det, iou
        if best is not None and best_iou >= 0.5:
            track[idx] = {"detection_index": best["detection_index"],
                          "box_xyxy": best["box_xyxy"], "source": "anchor"}

    if not track:
        return {}

    def _propagate(sequence: list[int]) -> None:
        current = None
        for idx in sequence:
            if idx in track and track[idx]["source"] == "anchor":
                current = track[idx]["box_xyxy"]
                continue
            if current is None:
                continue
            best, best_iou = None, 0.0
            for det in _candidate_objects(frames[idx], frame_area, max_area_fraction):
                iou = _iou(det["box_xyxy"], current)
                if iou > best_iou:
                    best, best_iou = det, iou
            if best is not None and best_iou >= min_track_iou:
                candidate = {"detection_index": best["detection_index"],
                             "box_xyxy": best["box_xyxy"], "source": "propagated",
                             "iou": round(best_iou, 3)}
                prev = track.get(idx)
                if prev is None or prev["source"] != "anchor" and best_iou > prev.get("iou", 0.0):
                    track[idx] = candidate
                current = best["box_xyxy"]
            # else: object undetected this frame; keep coasting on last box

    _propagate(order)
    _propagate(list(reversed(order)))
    return track


def filter_links(
    detections: dict,
    track: dict[int, dict],
    min_link_prob: float,
) -> dict[int, dict]:
    """Per frame: hf links whose target IS the tracked task object (drop the rest)."""
    result: dict[int, dict] = {}
    for frame in detections["frames"]:
        idx = frame["frame_idx"]
        entry = track.get(idx)
        if entry is None:
            continue
        dets = frame.get("detections", [])
        kept = []
        for link in frame.get("links", {}).get("hf", []):
            if link["prob"] < min_link_prob:
                continue
            if link["target_detection_index"] != entry["detection_index"]:
                continue
            hand = dets[link["source_detection_index"]]
            kept.append({
                "prob": link["prob"],
                "hand_box": hand["box_xyxy"],
                "hand_detection_index": hand["detection_index"],
            })
        if kept:
            result[idx] = {"object_box": entry["box_xyxy"],
                           "track_source": entry["source"], "links": kept}
    return result


def _draw_kept(img: np.ndarray, info: dict, frame_idx: int, object_name: str) -> np.ndarray:
    vis = img.copy()
    ox1, oy1, ox2, oy2 = (int(round(v)) for v in info["object_box"])
    cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), OBJECT_COLOR, 3)
    cv2.putText(vis, object_name, (ox1, max(30, oy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, OBJECT_COLOR, 2, cv2.LINE_AA)
    ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
    for link in info["links"]:
        hx1, hy1, hx2, hy2 = (int(round(v)) for v in link["hand_box"])
        cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), HAND_COLOR, 3)
        hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
        cv2.line(vis, (hcx, hcy), (ocx, ocy), (0, 0, 0), 7, cv2.LINE_AA)
        cv2.line(vis, (hcx, hcy), (ocx, ocy), LINK_COLOR, 3, cv2.LINE_AA)
        cv2.putText(vis, f"HF {link['prob']:.2f}", ((hcx + ocx) // 2, (hcy + ocy) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINK_COLOR, 2, cv2.LINE_AA)
    cv2.putText(vis, f"frame {frame_idx}", (18, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def run(args: argparse.Namespace) -> dict:
    detections = json.load(open(args.detections))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    frame_area = float(width * height)

    linked = _linked_frames(detections, args.min_link_prob,
                            args.max_frame_area_fraction, frame_area)
    if not linked:
        raise SystemExit("no linked frames; nothing to filter")
    step = max(1, len(linked) // args.max_query_frames)
    sample = linked[::step][: args.max_query_frames]

    frames_by_idx = {f["frame_idx"]: f for f in detections["frames"]}
    images = _read_frames(args.video, set(sample))
    query_frames = []
    candidates_by_frame: dict[int, list[dict]] = {}
    for idx in sample:
        cands = _candidate_objects(frames_by_idx[idx], frame_area,
                                   args.max_frame_area_fraction)
        candidates_by_frame[idx] = cands
        query_frames.append((idx, images[idx], cands))

    analysis = ask_qwen_task_object(query_frames, output_dir / "qwen_query_frames")
    json.dump(analysis, open(output_dir / "qwen_task_analysis.json", "w"),
              ensure_ascii=False, indent=1)

    anchors: dict[int, list[float]] = {}
    for item in analysis.get("frames", []):
        idx, label = item.get("frame_idx"), item.get("box_label")
        if idx in candidates_by_frame and label is not None:
            cands = candidates_by_frame[idx]
            if 0 <= int(label) < len(cands):
                anchors[idx] = cands[int(label)]["box_xyxy"]
    if not anchors:
        raise SystemExit("Qwen returned no usable box anchors")

    track = build_track(detections, anchors, frame_area,
                        args.max_frame_area_fraction, args.min_track_iou)
    kept = filter_links(detections, track, args.min_link_prob)

    json.dump(
        {
            "task_object": analysis.get("task_object"),
            "hand_mode": analysis.get("hand_mode"),
            "active_hand": analysis.get("active_hand"),
            "anchor_frames": sorted(anchors),
            "kept_frames": sorted(kept),
            "dropped_linked_frames": sorted(set(linked) - set(kept)),
            "per_frame": {str(k): kept[k] for k in sorted(kept)},
        },
        open(output_dir / "filtered_track.json", "w"), ensure_ascii=False, indent=1,
    )

    # Render kept frames + full overlay.
    frames_dir = output_dir / "filtered_frames"
    frames_dir.mkdir(exist_ok=True)
    object_name = str(analysis.get("task_object", "object"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_kept = str(output_dir / "_kept_tmp.mp4")
    tmp_full = str(output_dir / "_full_tmp.mp4")
    kept_writer = cv2.VideoWriter(tmp_kept, fourcc, fps, (width, height))
    full_writer = cv2.VideoWriter(tmp_full, fourcc, fps, (width, height))
    cap = cv2.VideoCapture(args.video)
    idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        if idx in kept:
            vis = _draw_kept(img, kept[idx], idx, object_name)
            kept_writer.write(vis)
            cv2.imwrite(str(frames_dir / f"frame_{idx:04d}.jpg"), vis)
            full_writer.write(vis)
        else:
            dim = (img * 0.35).astype(np.uint8)
            cv2.putText(dim, f"frame {idx}  DROPPED", (18, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 120, 120), 2, cv2.LINE_AA)
            full_writer.write(dim)
        idx += 1
    cap.release()
    kept_writer.release()
    full_writer.release()

    import subprocess
    for tmp, final in ((tmp_kept, "kept_frames.mp4"), (tmp_full, "full_overlay.mp4")):
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", tmp, "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "18", str(output_dir / final)],
            check=True,
        )
        Path(tmp).unlink()

    return {
        "task_object": object_name,
        "hand_mode": analysis.get("hand_mode"),
        "active_hand": analysis.get("active_hand"),
        "linked_frames": len(linked),
        "kept_frames": len(kept),
        "anchors": len(anchors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-query-frames", type=int, default=8)
    parser.add_argument("--min-link-prob", type=float, default=0.5)
    parser.add_argument("--max-frame-area-fraction", type=float, default=0.35)
    parser.add_argument("--min-track-iou", type=float, default=0.2)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
