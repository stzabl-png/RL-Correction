#!/usr/bin/env python3
"""Start the vendored Isaac simulation server (dexisaac.sim.start_isaacsim_server)."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_module = importlib.import_module("dexisaac.sim.start_isaacsim_server")
_module = importlib.reload(_module)
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_module.main())
