"""CPU tests for the Sweep2 physical success contract."""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from progress_batch import SweepGeometry, SweepProgressBatch, sweep_signals  # noqa: E402


def sig(z: float, speed: float = 0.0):
    cube = torch.tensor([[0.0, 0.0, z]])
    return sweep_signals(
        cube, torch.tensor([[0.0, 0.0, speed]]), torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 3),
        torch.tensor([[0.0, 0.0, 0.125]]), SweepGeometry())


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

    # A monotonic sweep crosses the lip; stability, not entry alone, terminates.
    rewards = []
    for z in torch.linspace(0.119, 0.080, 12).tolist():
        out = p.step(sig(z, -0.08), torch.tensor([True]), torch.tensor([True]))
        rewards.append(float(out["task_reward"]))
    assert out["gates"][0, :3].all() and not bool(out["success"][0])
    assert max(rewards) > 0.0
    for _ in range(g.stable_steps - 1):
        out = p.step(sig(0.080), torch.tensor([True]), torch.tensor([True]))
        assert not bool(out["success"][0])
    out = p.step(sig(0.080), torch.tensor([True]), torch.tensor([True]))
    assert bool(out["success"][0])

    # Centre inside but cube footprint crossing a side wall is not success.
    s = sig(0.080)
    cube = torch.tensor([[g.pan_half_width, 0.0, 0.080]])
    edge = sweep_signals(cube, torch.zeros(1, 3), torch.zeros(1, 3),
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.zeros(1, 3),
                         torch.tensor([[0.0, 0.0, 0.125]]), g)
    assert bool(s["fully_inside"][0]) and not bool(edge["fully_inside"][0])
    print("Sweep2 progress self-test: PASS")


if __name__ == "__main__":
    main()
