#!/usr/bin/env python3
"""Parallel SAM3D mesh reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RECON_ROOT))

from _common.parallel import main_parallel_cli

if __name__ == "__main__":
    raise SystemExit(main_parallel_cli("sam3d", Path(__file__).with_name("run_sequence.py")))
