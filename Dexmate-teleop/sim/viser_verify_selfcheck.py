#!/usr/bin/env python3
"""对比台的自检包装:跑 viser_verify.py --self-check,给 check_all.py 用。"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
r = subprocess.run([str(ROOT / ".venv/bin/python"), "sim/viser_verify.py"
                    if (ROOT / "sim/viser_verify.py").exists()
                    else "scripts/viser_verify.py", "--self-check", "400"],
                   cwd=ROOT)
sys.exit(r.returncode)
