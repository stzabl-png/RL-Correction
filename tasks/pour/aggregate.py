"""Aggregate deterministic evaluations and 60%-success milestones."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from tasks.pour.enums import AblationMode


REQUIRED_SEEDS = {42, 43, 44}
REQUIRED_EVAL_EPISODES = 1000
REQUIRED_ABLATIONS = {mode.value for mode in AblationMode}


def _json_files(root: Path, name: str) -> list[Path]:
    return sorted(path for path in root.rglob(name) if path.is_file())


def aggregate(
    evaluation_root: str | Path,
    output_dir: str | Path,
    *,
    milestone_root: str | Path | None = None,
) -> dict:
    eval_root = Path(evaluation_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in _json_files(eval_root, "summary.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "run_name",
            "demo_id",
            "seed",
            "ablation",
            "episodes",
            "success_rate",
            "mean_cup_fraction",
            "mean_spill_fraction",
            "failure_counts",
        }
        if not required.issubset(value):
            continue
        rows.append({**value, "summary_path": str(path.resolve())})
    if not rows:
        raise RuntimeError(f"no evaluation summary.json files found under {eval_root}")

    milestones = {}
    if milestone_root is not None:
        for path in _json_files(Path(milestone_root), "milestone_60.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("stage") != "full":
                continue
            key = (str(value["ablation"]), int(value["seed"]))
            steps = int(value["agent_steps"])
            milestones[key] = min(steps, milestones.get(key, steps))

    with (destination / "evaluation_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "run_name",
            "demo_id",
            "seed",
            "ablation",
            "episodes",
            "success_rate",
            "mean_cup_fraction",
            "mean_spill_fraction",
            "steps_to_60",
            "summary_path",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: row[name] for name in fields if name in row},
                    "steps_to_60": milestones.get(
                        (str(row["ablation"]), int(row["seed"])), ""
                    ),
                }
            )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["ablation"]), str(row["demo_id"]))].append(row)
    groups = []
    for (ablation, demo_id), values in sorted(grouped.items()):
        failures = Counter()
        for value in values:
            failures.update(value["failure_counts"])
        step_values = [
            milestones[(ablation, int(value["seed"]))]
            for value in values
            if (ablation, int(value["seed"])) in milestones
        ]
        groups.append(
            {
                "ablation": ablation,
                "demo_id": demo_id,
                "seeds": sorted(int(value["seed"]) for value in values),
                "mean_success_rate": mean(float(value["success_rate"]) for value in values),
                "mean_cup_fraction": mean(
                    float(value["mean_cup_fraction"]) for value in values
                ),
                "mean_spill_fraction": mean(
                    float(value["mean_spill_fraction"]) for value in values
                ),
                "mean_steps_to_60": mean(step_values) if step_values else None,
                "failure_counts": dict(sorted(failures.items())),
            }
        )

    full_11 = [
        row for row in rows if row["ablation"] == "full" and str(row["demo_id"]) == "11"
    ]
    full_9 = [
        row for row in rows if row["ablation"] == "full" and str(row["demo_id"]) == "9"
    ]
    demo11_by_ablation = {
        mode: [
            row
            for row in rows
            if str(row["demo_id"]) == "11" and str(row["ablation"]) == mode
        ]
        for mode in REQUIRED_ABLATIONS
    }
    ablation_matrix_complete = all(
        {int(row["seed"]) for row in values} == REQUIRED_SEEDS
        and len(values) == len(REQUIRED_SEEDS)
        and all(int(row["episodes"]) == REQUIRED_EVAL_EPISODES for row in values)
        for values in demo11_by_ablation.values()
    )
    holdout_complete = (
        {int(row["seed"]) for row in full_9} == REQUIRED_SEEDS
        and len(full_9) == len(REQUIRED_SEEDS)
        and all(int(row["episodes"]) == REQUIRED_EVAL_EPISODES for row in full_9)
    )
    acceptance = {
        "demo11_three_seeds_each_ge_80": len(full_11) == 3
        and {int(row["seed"]) for row in full_11} == REQUIRED_SEEDS
        and all(float(row["success_rate"]) >= 0.80 for row in full_11),
        "demo9_three_seed_mean_ge_60": holdout_complete
        and mean(float(row["success_rate"]) for row in full_9) >= 0.60,
        "demo11_ablation_matrix_complete": ablation_matrix_complete,
        "demo9_full_evaluation_complete": holdout_complete,
    }
    report = {"runs": rows, "groups": groups, "acceptance": acceptance}
    (destination / "aggregate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--milestone-root")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = aggregate(
        args.evaluation_root,
        args.output_dir,
        milestone_root=args.milestone_root,
    )
    print(
        f"[pour-aggregate] runs={len(report['runs'])} "
        f"groups={len(report['groups'])} acceptance={report['acceptance']}"
    )


if __name__ == "__main__":
    main()
