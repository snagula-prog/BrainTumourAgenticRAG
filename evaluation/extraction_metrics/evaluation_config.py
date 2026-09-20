"""Central evaluation policy for the research-paper extraction pipeline.

Weights are intentionally kept outside metric implementations so scoring policy
can be changed without rewriting metric logic.

Integrity score:
    Measures extraction health/internal consistency before ground-truth data exists.

Ground-truth accuracy score:
    Measures extraction against manually verified annotations.
"""

from __future__ import annotations

import math
from typing import Mapping


# ---------------------------------------------------------------------------
# Phase-level ingestion integrity
# ---------------------------------------------------------------------------
#
# Primary metrics are metrics that materially affect whether extracted content
# can be trusted downstream. The remaining metrics are retained as diagnostics
# rather than allowed to dominate the primary score.
#
# IMPORTANT: These are initial, defensible defaults. Re-tune only after a
# sufficiently broad evaluation corpus is available (recommended >= 15 papers).
PRIMARY_INTEGRITY_WEIGHTS = {
    "metadata_integrity": 0.15,
    "structure_integrity": 0.20,
    "truncation_integrity": 0.20,
    "cross_parser_consistency": 0.15,
    "contamination_integrity": 0.15,
    "duplication_integrity": 0.10,
    "numeric_consistency": 0.05,
}

DIAGNOSTIC_INTEGRITY_METRICS = {
    "abstract_integrity",
    "reference_integrity",
    "coordinate_integrity",
}


# ---------------------------------------------------------------------------
# Canonical artifact integrity (post-reconstruction)
# ---------------------------------------------------------------------------
#
# This score validates that logical tables/figures/formulas are represented
# with usable provenance and unique IDs. It does not claim visual/content
# correctness without ground truth.
CANONICAL_INTEGRITY_WEIGHTS = {
    "table_provenance_coverage": 0.15,
    "table_status_validity": 0.10,
    "figure_provenance_coverage": 0.15,
    "figure_caption_status_validity": 0.10,
    "formula_provenance_coverage": 0.10,
    "logical_id_uniqueness": 0.20,
    "source_reference_validity": 0.20,
}


# ---------------------------------------------------------------------------
# Ground-truth extraction accuracy
# ---------------------------------------------------------------------------
#
# Accuracy is only produced when manually verified ground truth is supplied.
# Paragraphs get the largest share because they carry the majority of the
# paper's searchable scientific content. Tables/figures are currently scored
# by count; content-level matching can be added once the GT schema supports it.
ACCURACY_WEIGHTS = {
    "metadata": 0.15,
    "abstract_similarity": 0.10,
    "sections": 0.10,
    "paragraphs": 0.25,
    "reference_score": 0.10,
    "formulas": 0.10,
    "table_count_score": 0.10,
    "figure_count_score": 0.10,
}


# ---------------------------------------------------------------------------
# Diagnostic policy
# ---------------------------------------------------------------------------
LOW_VARIANCE_WATCH_THRESHOLD = 0.02

INTEGRITY_ISSUE_THRESHOLDS = {
    "metadata_integrity": 0.75,
    "abstract_integrity": 0.75,
    "structure_integrity": 0.80,
    "duplication_integrity": 0.90,
    "reference_integrity": 0.80,
    "truncation_integrity": 0.85,
    "numeric_consistency": 0.80,
    "cross_parser_consistency": 0.70,
    "contamination_integrity": 0.85,
    "coordinate_integrity": 0.90,
}


def weighted_mean(
    metrics: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> float | None:
    """Compute a normalized weighted mean over available metrics only."""
    numerator = 0.0
    denominator = 0.0

    for name, weight in weights.items():
        value = metrics.get(name)

        if value is None:
            continue

        numerator += float(value) * float(weight)
        denominator += float(weight)

    if denominator == 0.0:
        return None

    return numerator / denominator


def validate_weights(
    name: str,
    weights: Mapping[str, float],
) -> None:
    """Fail fast if a scoring policy does not sum to one."""
    total = sum(float(value) for value in weights.values())

    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{name} weights must sum to 1.0; got {total:.10f}"
        )


validate_weights(
    "PRIMARY_INTEGRITY_WEIGHTS",
    PRIMARY_INTEGRITY_WEIGHTS,
)
validate_weights(
    "CANONICAL_INTEGRITY_WEIGHTS",
    CANONICAL_INTEGRITY_WEIGHTS,
)
validate_weights(
    "ACCURACY_WEIGHTS",
    ACCURACY_WEIGHTS,
)
