#!/usr/bin/env python3
"""Run OCIR Sharpa grasp synthesis with the BODex algorithm ported to official cuRobo v2."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ocir.grasp_synthesis.synthesize_sharpa_bodex_curobo_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
