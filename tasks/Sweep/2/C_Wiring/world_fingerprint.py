"""Sweep2 checkpoint/world compatibility gate (Pour 4f2f857 pattern)."""
from __future__ import annotations

import json
import os


CRITICAL = (
    "task", "policy_io", "time", "cube_start_world_m", "success",
    "mouth_floor", "reference.sha256", "dustpan_asset.sha256", "method",
    "cube_variants",
)


def _flat(value, prefix=""):
    out = {}
    for key, item in value.items():
        path = f"{prefix}{key}"
        if isinstance(item, dict):
            out.update(_flat(item, path + "."))
        else:
            out[path] = item
    return out


def compare(recorded: dict, expected: dict):
    left, right = _flat(recorded), _flat(expected)
    mismatches, unverifiable = [], []
    for key in CRITICAL:
        if (key not in recorded and key not in expected
                and not any(k.startswith(key + ".") for k in left)
                and not any(k.startswith(key + ".") for k in right)):
            continue
        if key in recorded and isinstance(recorded[key], dict):
            keys = [k for k in right if k.startswith(key + ".")]
        else:
            keys = [key]
        for leaf in keys:
            if leaf not in left or leaf not in right:
                unverifiable.append((leaf, left.get(leaf), right.get(leaf)))
            elif left[leaf] != right[leaf]:
                mismatches.append((leaf, left[leaf], right[leaf]))
    return mismatches, unverifiable


def assert_match(path: str, expected: dict, allow_legacy_full: bool = False):
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"checkpoint world fingerprint not found: {path}")
    with open(path) as handle:
        recorded = json.load(handle)
    if allow_legacy_full and "method" not in recorded:
        recorded["method"] = expected["method"]
    mismatches, unverifiable = compare(recorded, expected)
    if mismatches or unverifiable:
        detail = "\n".join(
            f"  {key}: recorded={old!r} current={new!r}"
            for key, old, new in mismatches + unverifiable)
        raise RuntimeError(f"checkpoint world mismatch/unverifiable:\n{detail}")
    print(f"[world] matched {path}", flush=True)
    return recorded
