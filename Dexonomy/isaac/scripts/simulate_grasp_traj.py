#!/usr/bin/env python3
"""Task-module wrapper: registers grasp_traj_simulation from the vendored dexisaac package."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_module = importlib.import_module("dexisaac.isaac.simulate_grasp_traj")
_module = importlib.reload(_module)
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_module.main())
