"""Detect interaction-active intervals from stable hand-object links."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InteractionEpisodeConfig:
    """Temporal confirmation and public keyframe delays for one interaction."""

    min_link_probability: float = 0.5
    min_positive_frames: int = 3
    confirmation_window: int = 5
    max_gap_frames: int = 1
    keyframe_delay_seconds: float = 0.1

    def validate(self) -> None:
        if not 0.0 <= self.min_link_probability <= 1.0:
            raise ValueError("min_link_probability must be in [0, 1]")
        if self.min_positive_frames < 1:
            raise ValueError("min_positive_frames must be positive")
        if self.confirmation_window < self.min_positive_frames:
            raise ValueError("confirmation_window must be at least min_positive_frames")
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be non-negative")
        if self.keyframe_delay_seconds < 0.0:
            raise ValueError("keyframe_delay_seconds must be non-negative")


def _positive_links(
    frames: list[dict[str, Any]], config: InteractionEpisodeConfig
) -> list[tuple[int, float]]:
    positives = []
    for expected_idx, frame in enumerate(frames):
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx != expected_idx:
            raise ValueError("interaction frames must be continuous and zero-indexed")
        probability = frame.get("hand_link_probability")
        if (
            frame.get("status") == "accepted"
            and frame.get("selection_source") == "hf_link"
            and probability is not None
            and float(probability) >= config.min_link_probability
        ):
            positives.append((frame_idx, float(probability)))
    return positives


def detect_interaction_episodes(
    frames: list[dict[str, Any]],
    *,
    fps: float,
    config: InteractionEpisodeConfig | None = None,
) -> list[dict[str, Any]]:
    """Return stable start/end keyframes; internal confirmation frames stay hidden."""

    config = config or InteractionEpisodeConfig()
    config.validate()
    if not frames:
        raise ValueError("interaction frames cannot be empty")
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    positives = _positive_links(frames, config)
    runs: list[list[tuple[int, float]]] = []
    for item in positives:
        if not runs or item[0] - runs[-1][-1][0] > config.max_gap_frames + 1:
            runs.append([item])
        else:
            runs[-1].append(item)

    stable_runs = []
    for run in runs:
        confirmed = False
        for start_idx in range(len(run)):
            evidence = [run[start_idx]]
            for candidate in run[start_idx + 1 :]:
                if candidate[0] - evidence[0][0] >= config.confirmation_window:
                    break
                evidence.append(candidate)
                if len(evidence) >= config.min_positive_frames:
                    confirmed = True
                    break
            if confirmed:
                break
        if confirmed:
            stable_runs.append((run, evidence[: config.min_positive_frames]))

    offset = int(round(config.keyframe_delay_seconds * fps))
    video_end = len(frames) - 1
    episodes = []
    for episode_idx, (run, evidence) in enumerate(stable_runs):
        raw_start = run[0][0]
        raw_end = run[-1][0]
        next_raw_start = (
            stable_runs[episode_idx + 1][0][0][0]
            if episode_idx + 1 < len(stable_runs)
            else None
        )
        start_frame = min(video_end, raw_start + offset)
        end_frame = min(video_end, raw_end + offset)
        if next_raw_start is not None:
            end_frame = min(end_frame, next_raw_start - 1)
        truncated = raw_end == video_end or (
            next_raw_start is None and raw_end + offset >= video_end
        )
        if end_frame < start_frame:
            continue
        episodes.append(
            {
                "episode_idx": episode_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "raw_start_frame": raw_start,
                "raw_end_frame": raw_end,
                "start_source": "stable_hand_object_link_plus_delay",
                "end_source": (
                    "video_end_truncated"
                    if truncated
                    else "stable_hand_object_link_falling_edge_plus_delay"
                ),
                "end_truncated": truncated,
                "confidence": statistics.median(item[1] for item in evidence),
                "keyframe_offset_frames": offset,
                "config": asdict(config),
            }
        )
    return episodes
