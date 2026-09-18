"""
Evaluate the observed variance of integrity metrics across papers.

Usage:
    python extraction_metric_variance.py [evaluation_dir]

Defaults to ./storage/evaluation.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


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

WATCH_THRESHOLD = 0.02


def main() -> None:

    directory = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "./storage/evaluation"
    )

    files = sorted(
        directory.glob(
            "*_quality.json"
        )
    )

    if not files:
        print(
            f"No *_quality.json files found in {directory}"
        )
        return

    per_metric: dict[
        str,
        list[
            tuple[str, float]
        ],
    ] = {
        name: []
        for name in CURRENT_WEIGHTS
    }

    for file_path in files:

        try:
            data = json.loads(
                file_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:

            print(
                f"[SKIP] {file_path.name}: "
                f"{exc}"
            )

            continue

        metrics = data.get(
            "metrics",
            {},
        )

        if not isinstance(
            metrics,
            dict,
        ):
            continue

        paper_id = file_path.stem.replace(
            "_quality",
            "",
        )

        for name in CURRENT_WEIGHTS:

            value = metrics.get(
                name
            )

            if isinstance(
                value,
                (int, float),
            ):

                per_metric[
                    name
                ].append(
                    (
                        paper_id,
                        float(value),
                    )
                )

    print()
    print(
        f"Loaded {len(files)} quality report(s) "
        f"from {directory}"
    )
    print()

    if len(files) < 15:
        print(
            "Note: fewer than 15 papers -- single "
            "outliers can dominate standard deviation."
        )
        print()

    header = (
        f"{'metric':28s}"
        f"{'weight':>8s}"
        f"{'n':>6s}"
        f"{'mean':>9s}"
        f"{'std':>9s}"
        f"{'min':>9s}"
        f"{'max':>9s}"
        f"  worst paper"
    )

    print(header)
    print("-" * len(header))

    for name, weight in sorted(
        CURRENT_WEIGHTS.items(),
        key=lambda item: item[1],
    ):

        pairs = per_metric[
            name
        ]

        if not pairs:
            continue

        values = [
            value
            for _, value in pairs
        ]

        mean = statistics.mean(
            values
        )

        std = (
            statistics.pstdev(
                values
            )
            if len(values) > 1
            else 0.0
        )

        worst_paper, min_value = min(
            pairs,
            key=lambda item: item[1],
        )

        max_value = max(
            values
        )

        flag = ""

        if (
            weight <= 0.06
            and std > WATCH_THRESHOLD
        ):

            flag = (
                "  <-- variance worth reviewing"
            )

        print(
            f"{name:28s}"
            f"{weight:8.2f}"
            f"{len(values):6d}"
            f"{mean:9.4f}"
            f"{std:9.4f}"
            f"{min_value:9.4f}"
            f"{max_value:9.4f}"
            f"  {worst_paper}"
            f"{flag}"
        )


if __name__ == "__main__":
    main()
