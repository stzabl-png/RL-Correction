"""CPU-only behavior checks for the locked Sweep2 main ablations."""
from __future__ import annotations

from ablation_settings import resolve
import world_fingerprint as WF


def effective(reference_conf, method):
    setting = resolve(method)
    conf = [list(row) for row in reference_conf]
    if setting.confidence_mode == "ones":
        conf = [[1.0 for _ in row] for row in conf]
    floor = [min(row) for row in conf]
    hand = [0.0 if value >= 0.70 else 0.50 if value >= 0.40 else 0.80
            for value in floor]
    if not setting.human_shape:
        hand = [0.0 for _ in hand]
    return conf, hand


sample = [[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]]
full_c, full_h = effective(sample, "full")
human_c, human_h = effective(sample, "wo_human")
flat_c, flat_h = effective(sample, "wo_conf")
assert full_c == sample
assert full_h == [0.8, 0.5, 0.0]
assert human_c == sample and human_h == [0.0, 0.0, 0.0]
assert flat_c == [[1.0, 1.0]] * 3 and flat_h == [0.0, 0.0, 0.0]
assert resolve("full").warmup_role == "none"
assert resolve("wo_human").warmup_role == "none"
assert resolve("wo_conf").warmup_role == "none"
base_world = {"task": "sweep", "method": {"name": "full"}}
same_world = {"task": "sweep", "method": {"name": "full"}}
wrong_world = {"task": "sweep", "method": {"name": "wo_conf"}}
old_critical = WF.CRITICAL
WF.CRITICAL = ("task", "method")
assert WF.compare(base_world, same_world) == ([], [])
assert WF.compare(base_world, wrong_world)[0]
WF.CRITICAL = old_critical
try:
    resolve("typo")
except ValueError:
    pass
else:
    raise AssertionError("unknown method must fail closed")
print("[selftest_ablation] PASS")
