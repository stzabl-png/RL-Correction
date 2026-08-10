"""Baseline multi-cycle approach-interact-leave segmentation per hand.

Relies ONLY on HOI-DETR hf-link frames (as recorded in per_hand_links.json by
render_per_hand_interaction_video): no pose, no co-motion, no Qwen.

Rule:
  1. Per hand, linked frames -> interaction segments, bridging gaps
     <= --bridge-gap frames and dropping segments shorter than --min-seg.
  1b. Qwen occlusion arbitration (on by default): a gap whose length is in
     [--qwen-gap-min, --qwen-gap-max] is sent to Qwen with frames sampled
     around it and two strict yes/no questions -- (q1) is the hand still
     holding/manipulating the object with the object occluded, (q2) are the
     pre-gap and post-gap objects the same object or part/whole of one
     assembly.  Both "yes" -> the whole gap is merged into the interaction.
     Gaps above --qwen-gap-max split unconditionally; API errors never bridge.
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

from ..qwen_client import build_user_content, call_qwen, make_client
from .dual_hand_frame_split import build_hand_tracks
from .qwen_interaction_object_filter import HAND_COLOR, LINK_COLOR, OBJECT_COLOR

ARBITRATION_SYSTEM_PROMPT = (
    "You are a precise vision assistant for egocentric manipulation video analysis. "
    "You will be shown frames sampled around a temporal GAP in which a hand-object "
    "interaction detector lost its link to an object. Your job is to answer exactly "
    "two yes/no questions. Answer STRICTLY in JSON with no extra text, no reasoning, "
    'no markdown: {"q1": "yes" | "no", "q2": "yes" | "no"}'
)

ARBITRATION_USER_TMPL = """Frames from ONE egocentric manipulation video, in temporal order:
- IMAGE 1: the LAST frame BEFORE the gap where the detector still linked the {side} hand (red box) to an object (yellow box).
- IMAGE 2..{k1}: frames sampled INSIDE the gap (no boxes; detector lost the object).
- IMAGE {k}: the FIRST frame AFTER the gap where the link returned (red box = hand, yellow box = object).

Q1: During the gap frames (middle images), is the {side} hand still holding or directly manipulating the object seen in IMAGE 1, with the object partially or fully hidden - occluded by the hand itself, by another object, or fused into another object? Answer "no" if the hand has visibly released the object (e.g., the object lies on the table away from the hand, or the hand is doing something else empty-handed).

Q2: Is the object in IMAGE 1 and the object in IMAGE {k} the same physical object, OR is one of them a part/component of the other (or of the same assembly)? Answer "yes" also when before and after simply show the same object.

Examples of correct judgments:
- A hand sweeps with a broom; mid-gap the broom is hidden behind the hand and the debris pile; after the gap the same broom reappears in the same hand. -> q1: yes, q2: yes (same broom throughout).
- A hand holds a key; mid-gap the key is inserted into a padlock so the detector now boxes the whole "padlock with key" assembly; the key is a component of that assembly. -> q1: yes, q2: yes.
- A hand puts a dustpan down on the table and starts picking up beads with the now-empty fingers; the dustpan lies visible on the table away from the hand. -> q1: no (hand released the object; not an occlusion), q2 irrelevant.
- Before the gap the hand held an apple; after the gap it holds a banana. -> q2: no (different objects, not part of one assembly).

Answer strict JSON only: {{"q1": "yes"|"no", "q2": "yes"|"no"}}"""


def _parse_yesno_json(content: str) -> dict:
    text = content.strip().strip("`")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON in Qwen reply: {content[:200]!r}")
    return json.loads(text[start:end + 1])


def _read_video_frames(video: str, wanted: set[int]) -> dict[int, "np.ndarray"]:
    cap = cv2.VideoCapture(video)
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


def _resize_width(img, width=960):
    h = int(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h))


def _draw_frame_links(img, links):
    for lk in links:
        hx1, hy1, hx2, hy2 = (int(round(v)) for v in lk["hand_box"])
        ox1, oy1, ox2, oy2 = (int(round(v)) for v in lk["object_box"])
        cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (60, 60, 230), 5)
        cv2.rectangle(img, (ox1, oy1), (ox2, oy2), (0, 220, 255), 5)
    return img


def qwen_arbitrate_gap(video: str, side: str, a: int, b: int,
                       per_frame: dict, query_dir: Path, client) -> dict:
    """Ask Qwen the two occlusion questions for the gap (a, b). Returns the
    record dict; 'bridge' is True only when both answers are yes."""
    mids = sorted({a + (b - a) // 3, a + (b - a) // 2, a + 2 * (b - a) // 3} - {a, b})[:3]
    frames = _read_video_frames(video, {a, b, *mids})
    query_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{side}_{a}-{b}"
    paths = []
    img = _draw_frame_links(frames[a].copy(), per_frame[str(a)]["links"])
    p = query_dir / f"{tag}_1_before_f{a}.jpg"
    cv2.imwrite(str(p), _resize_width(img)); paths.append(p)
    for j, m in enumerate(mids):
        p = query_dir / f"{tag}_{2 + j}_mid_f{m}.jpg"
        cv2.imwrite(str(p), _resize_width(frames[m])); paths.append(p)
    img = _draw_frame_links(frames[b].copy(), per_frame[str(b)]["links"])
    p = query_dir / f"{tag}_{2 + len(mids)}_after_f{b}.jpg"
    cv2.imwrite(str(p), _resize_width(img)); paths.append(p)

    record = {"gap_start": a, "gap_end": b, "gap_len": b - a - 1}
    try:
        prompt = ARBITRATION_USER_TMPL.format(side=side, k=len(paths), k1=len(paths) - 1)
        reply = call_qwen(ARBITRATION_SYSTEM_PROMPT, build_user_content(prompt, paths),
                          client=client)
        ans = _parse_yesno_json(reply.content)
        record["q1"], record["q2"] = ans.get("q1"), ans.get("q2")
        record["bridge"] = record["q1"] == "yes" and record["q2"] == "yes"
    except Exception as exc:  # API failure never bridges
        record["error"] = str(exc)
        record["bridge"] = False
    return record


def arbitrate_and_merge(segs: list[tuple[int, int]], video: str, side: str,
                        per_frame: dict, query_dir: Path,
                        gap_min: int, gap_max: int, max_calls: int) -> tuple[list, list]:
    """Merge consecutive segments across Qwen-approved occlusion gaps."""
    if len(segs) < 2:
        return segs, []
    client = None
    records, merged = [], [list(segs[0])]
    calls = 0
    for s, e in segs[1:]:
        prev_end = merged[-1][1]
        gap = s - prev_end - 1
        bridge = False
        if gap_min <= gap <= gap_max and calls < max_calls:
            if client is None:
                client = make_client()
            rec = qwen_arbitrate_gap(video, side, prev_end, s, per_frame,
                                     query_dir, client)
            calls += 1
            records.append(rec)
            bridge = rec["bridge"]
        if bridge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged], records

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
    parser.add_argument("--no-qwen-arbitration", action="store_true",
                        help="disable Qwen occlusion arbitration of medium gaps")
    parser.add_argument("--qwen-gap-min", type=int, default=5)
    parser.add_argument("--qwen-gap-max", type=int, default=30)
    parser.add_argument("--qwen-max-calls", type=int, default=8,
                        help="max Qwen calls per hand per video")
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
        segs = link_segments(info["linked_frames"], args.bridge_gap, 1)
        arb_records = []
        if not args.no_qwen_arbitration and segs:
            segs, arb_records = arbitrate_and_merge(
                segs, args.video, side, info["per_frame"],
                output_dir / "qwen_gap_queries" / side,
                args.qwen_gap_min, args.qwen_gap_max, args.qwen_max_calls)
        segs = [(a, b) for a, b in segs if b - a + 1 >= args.min_seg]
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
                        "cycles": cycles, "video": str(out_path),
                        "qwen_bridged_gaps": sum(1 for r in arb_records if r.get("bridge"))}
        json.dump({"bridge_gap": args.bridge_gap, "min_seg": args.min_seg,
                   "qwen_arbitration": not args.no_qwen_arbitration,
                   "qwen_gap_range": [args.qwen_gap_min, args.qwen_gap_max],
                   "qwen_gap_records": arb_records,
                   "num_frames": num_frames, "cycles": cycles},
                  open(output_dir / f"{side}_hand_multi_phase.json", "w"),
                  ensure_ascii=False, indent=1)
    print(json.dumps({s: {"cycles": r.get("num_cycles"), "status": r["status"]}
                      for s, r in report.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
