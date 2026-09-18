"""
Run this anytime you've added new papers to see whether the
integrity-score weights still make sense.

Usage:
    python metric_variance_report.py [evaluation_dir]

Defaults to ./storage/evaluation (settings.evaluation_dir) if no
argument is given. Reads every *_quality.json in that directory.
"""

import json
import statistics
import sys
from pathlib import Path

# Current weights, kept in sync with
# extraction_quality_metrics.py::_weighted_integrity_score
CURRENT_WEIGHTS = {
    "coordinate_integrity": 0.03,
    "reference_integrity": 0.03,
    "abstract_integrity": 0.06,
    "duplication_integrity": 0.06,
    "contamination_integrity": 0.06,
    "numeric_consistency": 0.11,
    "structure_integrity": 0.13,
    "metadata_integrity": 0.14,
    "truncation_integrity": 0.16,
    "cross_parser_consistency": 0.22,
}

# Metrics currently treated as "near-zero variance, low weight".
# If their observed std climbs meaningfully above this, it's worth
# reconsidering their weight.
WATCH_THRESHOLD = 0.02


def main() -> None:

    directory = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "./storage/evaluation"
    )

    files = sorted(
        directory.glob("*_quality.json")
    )

    if not files:
        print(
            f"No *_quality.json files found in {directory}"
        )
        return

    per_metric: dict[str, list[float]] = {
        name: [] for name in CURRENT_WEIGHTS
    }

    paper_ids = []

    for file_path in files:

        data = json.loads(
            file_path.read_text(
                encoding="utf-8"
            )
        )

        metrics = data.get(
            "metrics",
            {},
        )

        paper_ids.append(
            file_path.stem.replace(
                "_quality",
                "",
            )
        )

        for name in CURRENT_WEIGHTS:

            if name in metrics:

                per_metric[name].append(
                    metrics[name]
                )

    n = len(files)

    print(
        f"Loaded {n} quality report(s) "
        f"from {directory}\n"
    )

    if n < 15:
        print(
            "Note: fewer than 15 papers -- single outliers can "
            "still dominate std. Treat flags below as leads to "
            "investigate, not final verdicts.\n"
        )

    header = (
        f"{'metric':28s}{'weight':>8s}"
        f"{'mean':>8s}{'std':>8s}{'min':>8s}  worst paper"
    )
    print(header)
    print("-" * len(header))

    # Sort by weight ascending so the low-weight ("assumed boring")
    # metrics are grouped together at the top for easy scanning.
    for name, weight in sorted(
        CURRENT_WEIGHTS.items(),
        key=lambda item: item[1],
    ):

        values = per_metric[name]

        if not values:
            continue

        mean = statistics.mean(values)
        std = (
            statistics.pstdev(values)
            if len(values) > 1
            else 0.0
        )

        min_value = min(values)
        min_index = values.index(min_value)
        worst_paper = paper_ids[min_index]

        flag = ""

        if (
            weight <= 0.06
            and std > WATCH_THRESHOLD
        ):
            flag = (
                "  <-- more variance than expected for "
                "its current low weight"
            )

        print(
            f"{name:28s}{weight:8.2f}"
            f"{mean:8.4f}{std:8.4f}{min_value:8.4f}"
            f"  {worst_paper}{flag}"
        )


if __name__ == "__main__":
    main()