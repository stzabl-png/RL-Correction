#!/usr/bin/env python3
"""Parallel ViPE over dataset videos."""

from __future__ import annotations

import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.parallel import main_parallel_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main_parallel_cli("vipe", Path(__file__).with_name("run_sequence.py")))
