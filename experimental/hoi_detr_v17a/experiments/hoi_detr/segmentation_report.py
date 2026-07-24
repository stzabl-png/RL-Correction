"""Write the compact human-review report for a video mask sequence."""

from __future__ import annotations

from pathlib import Path


def summarize_instances_by_episode(manifest: dict) -> dict:
    """Count output-mask frames for every global instance in each episode."""

    frames = {
        int(frame["frame_idx"]): frame.get("objects", {})
        for frame in manifest.get("frames", [])
    }
    episodes = []
    for fallback_idx, keyframe in enumerate(manifest.get("interaction_keyframes", [])):
        episode_idx = int(keyframe.get("cycle_idx", fallback_idx))
        start_frame = int(keyframe.get("start_frame", keyframe["frame_idx"]))
        end_frame = int(keyframe["end_frame"])
        instance_ids = [str(value) for value in keyframe.get("object_ids", [])]
        instances = []
        for instance_id in instance_ids:
            duration_frames = sum(
                bool(frames.get(frame_idx, {}).get(instance_id, {}).get("mask"))
                for frame_idx in range(start_frame, end_frame + 1)
            )
            instances.append(
                {
                    "instance_id": instance_id,
                    "duration_frames": int(duration_frames),
                }
            )
        episodes.append(
            {
                "episode_idx": episode_idx,
                "instance_count": len(instance_ids),
                "instances": instances,
            }
        )
    return {"episode_count": len(episodes), "episodes": episodes}


def render_segmentation_report(summary: dict) -> str:
    """Render only the three user-requested segmentation statistics."""

    lines = [
        "# Segmentation Report",
        "",
        f"Total episodes: {summary['episode_count']}",
        "",
        "| Episode | Instance count | Instance ID | Duration (frames) |",
        "|---|---:|---|---:|",
    ]
    for episode in summary["episodes"]:
        label = f"episode_{episode['episode_idx']:02d}"
        if not episode["instances"]:
            lines.append(f"| {label} | 0 | — | 0 |")
            continue
        for instance in episode["instances"]:
            lines.append(
                f"| {label} | {episode['instance_count']} | "
                f"{instance['instance_id']} | {instance['duration_frames']} |"
            )
    return "\n".join(lines) + "\n"


def write_segmentation_report(manifest: dict, output_path: Path) -> dict:
    summary = summarize_instances_by_episode(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_segmentation_report(summary), encoding="utf-8")
    return summary
