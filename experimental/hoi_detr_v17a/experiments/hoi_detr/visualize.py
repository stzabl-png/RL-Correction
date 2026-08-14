"""Compact visualization for HOI-DETR candidate boxes and relation links.

Both schemas produced by :mod:`run_sequence` are accepted:

* the preserved upstream payload (``box`` plus top-level ``hf``/``fs``); and
* normalized ``detections.json`` (``box_xyxy`` plus nested ``links``).

The command-line interface makes the renderer reusable without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CLASS_COLORS_BGR = {
    0: (32, 50, 220),       # hand
    1: (10, 194, 255),      # first-order object
    2: (210, 180, 20),      # second-order object
}


def _center(box: list[float]) -> tuple[int, int]:
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)


def _box(detection: dict[str, Any]) -> list[float]:
    """Return an XYXY box from either supported detection schema."""
    box = detection.get("box_xyxy", detection.get("box"))
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("detection must contain a four-value box or box_xyxy")
    return box


def _links(record: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    nested = record.get("links")
    if isinstance(nested, dict):
        return list(nested.get(kind, []))
    return list(record.get(kind, []))


def _link_indices(link: dict[str, Any]) -> tuple[int, int]:
    """Return endpoint indices from upstream or normalized link fields."""
    if "source_detection_index" in link and "target_detection_index" in link:
        return int(link["source_detection_index"]), int(link["target_detection_index"])
    return int(link["a"]), int(link["b"])


def _draw_frame(frame: Any, record: dict[str, Any]) -> Any:
    import cv2

    detections = record.get("detections", [])
    for link_key, color in (("hf", (255, 255, 255)), ("fs", (0, 255, 255))):
        for link in _links(record, link_key):
            source_index, target_index = _link_indices(link)
            source = detections[source_index]
            target = detections[target_index]
            start, end = _center(_box(source)), _center(_box(target))
            cv2.line(frame, start, end, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.line(frame, start, end, color, 3, cv2.LINE_AA)
            midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(
                frame,
                f"{link_key.upper()} {float(link['prob']):.2f}",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

    for detection in detections:
        class_id = int(detection["class_id"])
        color = CLASS_COLORS_BGR[class_id]
        x1, y1, x2, y2 = (int(round(v)) for v in _box(detection))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        label = f"{detection['class_name']} {float(detection['score']):.2f}"
        cv2.putText(
            frame,
            label,
            (max(0, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def render_predictions_video(
    *,
    video_path: Path,
    frames: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Render the frame-aligned prediction records to one MP4 artifact."""
    import cv2

    records = {int(record["frame_idx"]): record for record in frames}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for visualization: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open visualization writer: {output_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            record = records.get(frame_idx)
            if record is not None and bool(record.get("processed", True)):
                frame = _draw_frame(frame, record)
            cv2.putText(
                frame,
                f"frame {frame_idx}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render HOI-DETR boxes and hand-object relation links."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="upstream_predictions.json or normalized detections.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.predictions.read_text())
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise SystemExit(f"predictions file has no frames list: {args.predictions}")
    render_predictions_video(
        video_path=args.video,
        frames=frames,
        output_path=args.output,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
