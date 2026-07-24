#!/usr/bin/env python3
"""Affordance-seeded Sharpa grasp synthesis (thin conda-env wrapper).

Predicts the object's expected grasp area with the AffordanceModel, then runs
BODex-on-cuRobo-v2 synthesis with seeds restricted to that region.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ocir.grasp_synthesis.synthesize_affordance_seeded import main


if __name__ == "__main__":
    raise SystemExit(main())
