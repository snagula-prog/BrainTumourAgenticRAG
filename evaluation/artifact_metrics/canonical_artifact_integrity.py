from __future__ import annotations

from collections import Counter
from typing import Any

from evaluation.extraction_metrics.evaluation_config import (
    CANONICAL_INTEGRITY_WEIGHTS,
    weighted_mean,
)


class CanonicalArtifactIntegrity:
    """
    Evaluate canonical artifact integrity.

    This measures whether the canonical representation is internally
    usable and provenance-preserving.

    IMPORTANT:
    This is NOT ground-truth extraction accuracy.
    Accuracy requires manually annotated ground truth.
    """

    VALID_TABLE_STATUSES = {
        "structured",
        "text_only",
        "detected_no_content",
        "unavailable",
    }

    VALID_VISUAL_CLASSIFICATIONS = {
        "figure",
        "table",
        "equation",
        "diagram",
        "page_furniture",
        "image_fragment",
        "unknown",
    }

    def evaluate(
        self,
        canonical: dict[str, Any],
    ) -> dict[str, Any]:
        tables = self._safe_list(
            canonical.get("tables")
        )

        figures = self._safe_list(
            canonical.get("figures")
        )

        formulas = self._safe_list(
            canonical.get("formulas")
        )

        visual_artifacts = self._safe_list(
            canonical.get("visual_artifacts")
        )

        metrics = {
            "table_provenance_coverage": self._coverage(
                tables,
                lambda item:
                    bool(item.get("source_refs"))
                    and bool(
                        item.get("regions")
                        or item.get("coords")
                    ),
            ),

            "table_status_validity": self._coverage(
                tables,
                lambda item:
                    item.get("structure_status")
                    in self.VALID_TABLE_STATUSES,
            ),

            "figure_provenance_coverage": self._coverage(
                figures,
                lambda item:
                    bool(item.get("source_refs"))
                    and bool(
                        item.get("regions")
                        or item.get("coords")
                    ),
            ),

            "figure_caption_status_validity": self._coverage(
                figures,
                lambda item:
                    item.get("caption_status")
                    in {
                        "available",
                        "missing",
                    },
            ),

            "formula_provenance_coverage": self._coverage(
                formulas,
                lambda item:
                    bool(
                        item.get("source_refs")
                        or item.get("source")
                    ),
            ),

            # Diagnostic only for now: table structure is not included in the
            # weighted score until a broader benchmark corpus is available.
            "table_structure_representation_validity": (
                self._coverage(
                    tables,
                    lambda item:
                        self._table_structure_is_valid(item),
                )
            ),

            # Diagnostic only: measures whether tables produced by the
            # targeted fallback carry a method and confidence record.
            "table_fallback_metadata_validity": (
                self._coverage(
                    tables,
                    lambda item:
                        self._table_fallback_metadata_is_valid(item),
                )
            ),

            "formula_content_validity": self._coverage(
                formulas,
                lambda item:
                    bool(
                        str(
                            item.get("text")
                            or ""
                        ).strip()
                    ),
            ),

            "visual_classification_validity": self._coverage(
                visual_artifacts,
                lambda item:
                    item.get("classification")
                    in self.VALID_VISUAL_CLASSIFICATIONS,
            ),

            "logical_id_uniqueness": self._id_uniqueness(
                tables,
                figures,
                formulas,
            ),

            "source_reference_validity": (
                self._source_reference_validity(
                    tables,
                    figures,
                    formulas,
                )
            ),
        }

        score = weighted_mean(
            metrics,
            CANONICAL_INTEGRITY_WEIGHTS,
        )

        return {
            "evaluation_type": (
                "canonical_artifact_integrity"
            ),

            "integrity_score": (
                round(score, 4)
                if score is not None
                else None
            ),

            "status": self._status(
                score
            ),

            "scoring": {
                "weights": dict(
                    CANONICAL_INTEGRITY_WEIGHTS
                ),
            },

            "metrics": {
                key: (
                    round(value, 4)
                    if value is not None
                    else None
                )
                for key, value in metrics.items()
            },

            "counts": {
                "logical_tables": len(
                    tables
                ),
                "structured_tables": sum(
                    1
                    for item in tables
                    if item.get("structure_status")
                    == "structured"
                    and self._table_structure_is_valid(item)
                ),
                "fallback_tables": sum(
                    1
                    for item in tables
                    if isinstance(item.get("table_fallback"), dict)
                    and item.get("table_fallback", {}).get("used") is True
                ),
                "fallback_methods": dict(
                    Counter(
                        item.get("table_fallback", {}).get(
                            "method",
                            "unknown",
                        )
                        for item in tables
                        if isinstance(item.get("table_fallback"), dict)
                        and item.get("table_fallback", {}).get("used") is True
                    )
                ),
                "fallback_mean_confidence": self._mean_fallback_confidence(
                    tables
                ),
                "logical_figures": len(
                    figures
                ),
                "formulas": len(
                    formulas
                ),
                "visual_artifacts": len(
                    visual_artifacts
                ),
                "visual_classifications": dict(
                    Counter(
                        item.get(
                            "classification",
                            "unknown",
                        )
                        for item in visual_artifacts
                    )
                ),
                "unclassified_visuals": len(
                    self._safe_list(
                        canonical.get(
                            "unclassified_visuals"
                        )
                    )
                ),
            },

            "notes": [
                (
                    "This evaluator measures canonical "
                    "representation integrity only."
                ),
                (
                    "No metric is a ground-truth "
                    "extraction accuracy claim."
                ),
                (
                    "Ground-truth accuracy requires "
                    "manually annotated reference data."
                ),
            ],
        }

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _safe_list(
        value: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(
            value,
            list,
        ):
            return []

        return [
            item
            for item in value
            if isinstance(
                item,
                dict,
            )
        ]

    @classmethod
    def _table_structure_is_valid(
        cls,
        item: dict[str, Any],
    ) -> bool:

        status = item.get(
            "structure_status"
        )

        if status != "structured":
            # Non-structured states are valid when they accurately describe
            # the available evidence.
            return status in cls.VALID_TABLE_STATUSES

        structured = item.get(
            "structured_data"
        )

        if not isinstance(
            structured,
            (dict, list),
        ):
            return False

        if isinstance(
            structured,
            list,
        ):
            return bool(
                structured
            )

        cells = structured.get(
            "table_cells"
        )
        grid = structured.get(
            "grid"
        )
        rows = structured.get(
            "rows"
        )
        num_rows = structured.get(
            "num_rows"
        )
        num_cols = structured.get(
            "num_cols"
        )

        if isinstance(
            cells,
            list,
        ) and cells:
            return True

        if isinstance(
            grid,
            list,
        ) and grid and any(grid):
            return True

        if isinstance(
            rows,
            list,
        ) and rows:
            return True

        return (
            isinstance(
                num_rows,
                int,
            )
            and num_rows > 0
            and isinstance(
                num_cols,
                int,
            )
            and num_cols > 0
        )

    @staticmethod
    def _table_fallback_metadata_is_valid(
        item: dict[str, Any],
    ) -> bool:
        fallback = item.get(
            "table_fallback"
        )
        if not isinstance(fallback, dict):
            return True

        if not fallback.get("used"):
            return True

        method = fallback.get("method")
        confidence = fallback.get("confidence")
        return (
            isinstance(method, str)
            and bool(method.strip())
            and isinstance(confidence, (int, float))
            and 0.0 <= float(confidence) <= 1.0
        )

    @staticmethod
    def _mean_fallback_confidence(
        tables: list[dict[str, Any]],
    ) -> float | None:
        values = []
        for item in tables:
            fallback = item.get("table_fallback")
            if not isinstance(fallback, dict):
                continue
            if fallback.get("used") is not True:
                continue
            value = fallback.get("confidence")
            if isinstance(value, (int, float)):
                values.append(float(value))

        if not values:
            return None

        return round(
            sum(values) / len(values),
            4,
        )

    @staticmethod
    def _coverage(
        items: list[dict[str, Any]],
        predicate,
    ) -> float | None:
        if not items:
            return None

        valid = sum(
            1
            for item in items
            if predicate(item)
        )

        return valid / len(items)

    @staticmethod
    def _id_uniqueness(
        tables: list[dict[str, Any]],
        figures: list[dict[str, Any]],
        formulas: list[dict[str, Any]],
    ) -> float | None:

        ids = []

        for item in tables:
            value = item.get(
                "table_id"
            )

            if value:
                ids.append(value)

        for item in figures:
            value = item.get(
                "figure_id"
            )

            if value:
                ids.append(value)

        for item in formulas:
            value = item.get(
                "formula_id"
            )

            if value:
                ids.append(value)

        if not ids:
            return None

        counts = Counter(
            ids
        )

        unique = sum(
            1
            for count in counts.values()
            if count == 1
        )

        return unique / len(ids)

    @staticmethod
    def _source_reference_validity(
        tables: list[dict[str, Any]],
        figures: list[dict[str, Any]],
        formulas: list[dict[str, Any]],
    ) -> float | None:

        artifacts = (
            tables
            + figures
            + formulas
        )

        if not artifacts:
            return None

        valid = 0

        for item in artifacts:

            refs = item.get(
                "source_refs"
            )

            source = item.get(
                "source"
            )

            has_refs = (
                isinstance(
                    refs,
                    list,
                )
                and any(
                    isinstance(
                        ref,
                        dict,
                    )
                    for ref in refs
                )
            )

            has_source = (
                isinstance(
                    source,
                    str,
                )
                and bool(
                    source.strip()
                )
            )

            if has_refs or has_source:
                valid += 1

        return valid / len(
            artifacts
        )

    @staticmethod
    def _status(
        score: float | None,
    ) -> str:

        if score is None:
            return "not_applicable"

        if score >= 0.90:
            return "healthy"

        if score >= 0.75:
            return "review"

        return "poor"