"""Compatibility wrapper for :mod:`ocir.dexycb.mano_model`."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_module = importlib.import_module("ocir.dexycb.mano_model")
_module = importlib.reload(_module)
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("__")})
