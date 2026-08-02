"""Render full-length per-hand interaction videos from HOI-DETR detections.

Unlike dual_hand_frame_split (which keeps only linked frames), this renders the
WHOLE video twice: one output shows only the RIGHT hand's interactions (its hand
box, its hf-linked object boxes and link lines), the other only the LEFT hand's.
The other hand's boxes/links are filtered out of the rendering entirely.

Hand identity (left/right) comes from the same Qwen vote used by
dual_hand_frame_split, with average-x geometry fallback (--no-qwen forces the
fallback and avoids any API call).

Outputs (under --output-dir):
  right_hand_interactions.mp4 / left_hand_interactions.mp4   (H.264+faststart)
  per_hand_links.json   per-side, per-frame link records + hand side assignment
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2

from .dual_hand_frame_split import (
    build_hand_tracks,
    keep_frames_any_object,
    qwen_label_hand_sides,
)
from .qwen_interaction_object_filter import HAND_COLOR, LINK_COLOR, OBJECT_COLOR

SIDE_BANNER_COLOR = {"left": (230, 130, 40), "right": (60, 200, 60)}  # BGR


def _render_side_video(
    video: str,
    hand_tracks: dict[int, dict[str, dict]],
    track_id: str,
    kept: dict[int, dict],
    side: str,
    out_path: Path,
    fps: float,
    width: int,
    height: int,
) -> None:
    tmp = str(out_path.with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    cap = cv2.VideoCapture(video)
    banner = SIDE_BANNER_COLOR[side]
    idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        hand = hand_tracks.get(idx, {}).get(track_id)
        if hand is not None:
            hx1, hy1, hx2, hy2 = (int(round(v)) for v in hand["box_xyxy"])
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), HAND_COLOR, 4)
            cv2.putText(img, side.upper()[0], (hx1 + 6, max(36, hy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, HAND_COLOR, 3, cv2.LINE_AA)
        if idx in kept:
            hcx, hcy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
            for link in kept[idx]["links"]:
                ox1, oy1, ox2, oy2 = (int(round(v)) for v in link["object_box"])
                cv2.rectangle(img, (ox1, oy1), (ox2, oy2), OBJECT_COLOR, 6)
                ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                cv2.line(img, (hcx, hcy), (ocx, ocy), (0, 0, 0), 13, cv2.LINE_AA)
                cv2.line(img, (hcx, hcy), (ocx, ocy), LINK_COLOR, 6, cv2.LINE_AA)
                cv2.putText(img, f"HF {link['prob']:.2f}",
                            ((hcx + ocx) // 2, (hcy + ocy) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, LINK_COLOR, 2,
                            cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), banner, 14)
        cv2.putText(img, f"{side.upper()} HAND interactions  frame {idx}",
                    (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                    cv2.LINE_AA)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-link-prob", type=float, default=0.5)
    parser.add_argument("--max-frame-area-fraction", type=float, default=0.7)
    parser.add_argument("--no-qwen", action="store_true",
                        help="skip the Qwen hand-side vote, use average-x only")
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
    if args.no_qwen:
        # average-x geometry only: reuse the same fallback rule without any API call
        import numpy as np

        avg_x = {}
        for track in ("A", "B"):
            xs = [(h[track]["box_xyxy"][0] + h[track]["box_xyxy"][2]) / 2.0
                  for h in hand_tracks.values() if track in h]
            avg_x[track] = float(np.mean(xs)) if xs else None
        if avg_x["A"] is not None and avg_x["B"] is not None:
            mapping = ({"A": "left", "B": "right"} if avg_x["A"] <= avg_x["B"]
                       else {"A": "right", "B": "left"})
        else:
            mapping = {"A": "left", "B": "right"}
        sides = {"mapping": mapping, "source": "average_x_no_qwen",
                 "votes": None, "average_x": avg_x, "qwen_frames": []}
    else:
        sides = qwen_label_hand_sides(args.video, hand_tracks,
                                      output_dir / "qwen_hand_queries")
    side_of_track = sides["mapping"]

    summary = {"hand_side_assignment": sides, "min_link_prob": args.min_link_prob,
               "sides": {}}
    for side in ("left", "right"):
        track_id = next((t for t, s in side_of_track.items() if s == side), None)
        if track_id is None:
            summary["sides"][side] = {"status": "no_hand_track"}
            continue
        kept = keep_frames_any_object(
            detections, hand_tracks, track_id, width, height,
            args.max_frame_area_fraction, args.min_link_prob)
        out_path = output_dir / f"{side}_hand_interactions.mp4"
        _render_side_video(args.video, hand_tracks, track_id, kept, side,
                           out_path, fps, width, height)
        summary["sides"][side] = {
            "track_id": track_id,
            "status": "ok",
            "num_linked_frames": len(kept),
            "linked_frames": sorted(kept),
            "video": str(out_path),
            "per_frame": {str(k): kept[k] for k in sorted(kept)},
        }
    json.dump(summary, open(output_dir / "per_hand_links.json", "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps({
        side: summary["sides"].get(side, {}).get("num_linked_frames", 0)
        for side in ("left", "right")
    } | {"side_source": sides["source"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
