"""CPU tests for the Sweep2 physical success contract."""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from progress_batch import SweepGeometry, SweepProgressBatch, sweep_signals  # noqa: E402


def sig(z: float, speed: float = 0.0):
    # The 25 mm cube centre rests around +22.5 mm on the measured +10 mm
    # load-bearing basin surface.  A centre near zero is underneath the pan.
    cube = torch.tensor([[0.0, 0.0225, z]])
    return sweep_signals(
        cube, torch.tensor([[0.0, 0.0, speed]]), torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 3),
        torch.tensor([[0.0, 0.0, 0.160]]), SweepGeometry())


def main():
    g = SweepGeometry()
    p = SweepProgressBatch(1, "cpu", g)
    ids = torch.tensor([0])
    p.reset(ids)

    # Stationary outside cube cannot farm positive reward or advance a task gate.
    for _ in range(20):
        out = p.step(sig(0.125), torch.tensor([True]), torch.tensor([False]))
    assert out["gates"].tolist() == [[True, False, False, False]]
    assert float(out["task_reward"]) == 0.0

    # A geometrically valid centre crossing is immediate operational success.
    rewards = []
    entered_step = None
    for i, z in enumerate(torch.linspace(0.119, 0.094, 12).tolist()):
        out = p.step(sig(z, -0.08), torch.tensor([True]), torch.tensor([True]))
        rewards.append(float(out["task_reward"]))
        if bool(out["gates"][0, 2]):
            entered_step = i
            break
    assert entered_step is not None
    assert out["gates"][0].tolist() == [True, True, True, True]
    assert bool(out["success"][0])
    assert bool(out["entered"][0]) and not bool(out["fully_inside"][0])
    assert max(rewards) > 0.0

    # Deep containment is diagnostic only and is not required for success.
    s = sig(0.080)
    assert bool(s["entered"][0]) and bool(s["fully_inside"][0])
    assert not bool(s["deep_inside"][0])

    # Deep progress is diagnostics-only under the restored contract.
    deep = sig(0.0625, -0.02)
    assert bool(deep["deep_inside"][0])
    assert abs(float(deep["deep_margin"][0])) < 1.0e-6

    # Centre inside but cube footprint crossing a side wall is not entry.
    cube = torch.tensor([[g.pan_half_width, 0.0225, 0.080]])
    edge = sweep_signals(cube, torch.zeros(1, 3), torch.zeros(1, 3),
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 3),
                         torch.tensor([[0.0, 0.0, 0.160]]), g)
    assert not bool(edge["entered"][0]) and not bool(edge["fully_inside"][0])
    under_pan = sweep_signals(
        torch.tensor([[0.0, 0.002, 0.080]]), torch.zeros(1, 3),
        torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 0.160]]), g)
    assert not bool(under_pan["entered"][0]) and not bool(under_pan["fully_inside"][0])
    assert not bool(sig(0.096)["entered"][0])
    print("Sweep2 progress self-test: PASS")


if __name__ == "__main__":
    main()
