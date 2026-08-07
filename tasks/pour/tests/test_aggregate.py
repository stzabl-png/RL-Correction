from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tasks.pour.aggregate import REQUIRED_ABLATIONS, REQUIRED_SEEDS, aggregate


def _write_summary(
    root: Path,
    *,
    demo_id: str,
    ablation: str,
    seed: int,
    episodes: int = 1000,
) -> None:
    destination = root / f"demo{demo_id}" / ablation / f"seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    value = {
        "run_name": f"demo{demo_id}-{ablation}-seed{seed}",
        "demo_id": demo_id,
        "seed": seed,
        "ablation": ablation,
        "episodes": episodes,
        "success_rate": 0.90 if demo_id == "11" else 0.70,
        "mean_cup_fraction": 0.85,
        "mean_spill_fraction": 0.10,
        "failure_counts": {"spill": 1},
    }
    (destination / "summary.json").write_text(
        json.dumps(value), encoding="utf-8"
    )


class AggregateTest(unittest.TestCase):
    def test_acceptance_requires_complete_fixed_budget_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluations = root / "evaluations"
            for ablation in REQUIRED_ABLATIONS:
                for seed in REQUIRED_SEEDS:
                    _write_summary(
                        evaluations,
                        demo_id="11",
                        ablation=ablation,
                        seed=seed,
                    )
            for seed in REQUIRED_SEEDS:
                _write_summary(
                    evaluations,
                    demo_id="9",
                    ablation="full",
                    seed=seed,
                )

            report = aggregate(evaluations, root / "report")
            self.assertTrue(all(report["acceptance"].values()))

            incomplete = (
                evaluations / "demo11" / "pure_rl" / "seed44" / "summary.json"
            )
            value = json.loads(incomplete.read_text(encoding="utf-8"))
            value["episodes"] = 999
            incomplete.write_text(json.dumps(value), encoding="utf-8")
            report = aggregate(evaluations, root / "report-incomplete")
            self.assertFalse(
                report["acceptance"]["demo11_ablation_matrix_complete"]
            )


if __name__ == "__main__":
    unittest.main()
