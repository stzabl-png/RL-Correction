#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`ocir.grasp_synthesis.anchored_bodex.synthesize_sharpa_anchored_bodex`."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_module = importlib.import_module("ocir.grasp_synthesis.anchored_bodex.synthesize_sharpa_anchored_bodex")
_module = importlib.reload(_module)
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})


if __name__ == "__main__":
    raise SystemExit(_module.main())
