"""Combine verified per-cycle registries into one immutable video registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import write_json_atomic
from .video_registry import combine_cycle_registries


def run(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    registries = []
    failures = []
    for cycle_idx, path in enumerate(args.cycle_registry):
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("status") != "ready":
            failures.append(
                {"cycle_idx": cycle_idx, "reason": "registry_not_ready", "path": str(path)}
            )
            continue
        missing_masks = [
            object_id
            for object_id, entry in registry.get("objects", {}).items()
            if not Path(entry["mask"]).is_file()
        ]
        if missing_masks:
            failures.append(
                {
                    "cycle_idx": cycle_idx,
                    "reason": "registered_mask_missing",
                    "object_ids": missing_masks,
                    "path": str(path),
                }
            )
            continue
        registries.append(registry)

    if failures or len(registries) != len(args.cycle_registry):
        summary = {
            "schema_version": "video_registry_combine_v1",
            "status": "failed",
            "video": str(args.video.resolve()),
            "failures": failures,
            "final_registry": None,
        }
        write_json_atomic(output_dir / "summary.json", summary)
        return summary

    cycle_metadata = getattr(args, "cycle_metadata", None)
    if isinstance(cycle_metadata, Path):
        cycle_metadata = json.loads(cycle_metadata.read_text(encoding="utf-8"))["cycles"]
    identity_map = getattr(args, "identity_map", None)
    if isinstance(identity_map, Path):
        identity_map = json.loads(identity_map.read_text(encoding="utf-8"))[
            "cycle_object_id_maps"
        ]
    combined = combine_cycle_registries(
        registries,
        video=str(args.video.resolve()),
        cycle_metadata=cycle_metadata,
        cycle_object_id_maps=identity_map,
    )
    registry_path = output_dir / "reconstruction_registry.json"
    write_json_atomic(registry_path, combined)
    summary = {
        "schema_version": "video_registry_combine_v1",
        "status": "success",
        "video": str(args.video.resolve()),
        "cycle_count": len(registries),
        "object_count": len(combined["objects"]),
        "failures": [],
        "final_registry": str(registry_path.resolve()),
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cycle-registry", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycle-metadata", type=Path)
    parser.add_argument(
        "--identity-map",
        type=Path,
        help="Pure-visual cross-cycle local-to-global identity mapping",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    summary = run(_parser().parse_args(argv))
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
