# evaluation/extraction_benchmark.py

from __future__ import annotations

import math
import re
from typing import Any

from .evaluation_config import ACCURACY_WEIGHTS, weighted_mean


class ExtractionBenchmark:

    """
    Compares canonical extraction against manually verified
    ground-truth annotations.

    This is the ONLY evaluator allowed to report an
    extraction accuracy score.
    """

    def evaluate(
        self,
        *,
        canonical: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:

        metadata = self._metadata_score(
            canonical.get(
                "metadata",
                {},
            ),
            ground_truth.get(
                "metadata",
                {},
            ),
        )

        sections = self._list_f1(
            [
                item.get("heading", "")
                for item in canonical.get(
                    "sections",
                    [],
                )
            ],
            ground_truth.get(
                "sections",
                [],
            ),
        )

        paragraphs = self._paragraph_coverage(
            canonical.get(
                "paragraphs",
                [],
            ),
            ground_truth.get(
                "paragraphs",
                [],
            ),
        )

        formulas = self._list_f1(
            [
                item.get("text", "")
                for item in canonical.get(
                    "formulas",
                    [],
                )
            ],
            ground_truth.get(
                "formulas",
                [],
            ),
            formula=True,
        )

        references = self._reference_score(
            canonical.get(
                "references",
                [],
            ),
            ground_truth.get(
                "references",
                [],
            ),
        )

        tables = self._count_score(
            len(
                canonical.get(
                    "tables",
                    [],
                )
            ),
            int(
                ground_truth.get(
                    "table_count",
                    0,
                )
            ),
        )

        figures = self._count_score(
            len(
                canonical.get(
                    "figures",
                    [],
                )
            ),
            int(
                ground_truth.get(
                    "figure_count",
                    0,
                )
            ),
        )

        abstract_similarity = self._abstract_score(
            canonical.get(
                "metadata",
                {},
            ).get(
                "abstract"
            ),
            ground_truth.get(
                "metadata",
                {},
            ).get(
                "abstract"
            ),
        )

        component_scores = {
            "metadata": metadata["score"],
            "abstract_similarity": abstract_similarity,
            "sections": sections["f1"],
            "paragraphs": paragraphs["f1"],
            "reference_score": references,
            "formulas": formulas["f1"],
            "table_count_score": tables,
            "figure_count_score": figures,
        }

        overall = weighted_mean(
            component_scores,
            ACCURACY_WEIGHTS,
        )

        return {
            "evaluation_type": "ground_truth_accuracy",
            "accuracy_score": round(
                overall or 0.0,
                4,
            ),
            "scoring": {
                "weights": dict(
                    ACCURACY_WEIGHTS
                ),
            },
            "metrics": {
                "metadata": metadata,
                "sections": sections,
                "paragraphs": {
                    "precision": round(
                        paragraphs["precision"],
                        4,
                    ),
                    "recall": round(
                        paragraphs["recall"],
                        4,
                    ),
                    "f1": round(
                        paragraphs["f1"],
                        4,
                    ),
                    "coverage": round(
                        paragraphs["coverage"],
                        4,
                    ),
                },
                "formulas": formulas,
                "reference_score": round(
                    references,
                    4,
                ),
                "table_count_score": round(
                    tables,
                    4,
                ),
                "figure_count_score": round(
                    figures,
                    4,
                ),
                "abstract_similarity": round(
                    abstract_similarity,
                    4,
                ),
            },
        }

    # =========================================================
    # METADATA
    # =========================================================

    def _metadata_score(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
    ) -> dict[str, Any]:

        scores = {}

        scores["title"] = self._similarity(
            actual.get("title"),
            expected.get("title"),
        )

        scores["year"] = (
            1.0
            if actual.get("year")
            == expected.get("year")
            else 0.0
        )

        scores["doi"] = (
            1.0
            if self._normalize_doi(
                actual.get("doi")
            )
            == self._normalize_doi(
                expected.get("doi")
            )
            and expected.get("doi")
            else 0.0
        )

        scores["authors"] = self._list_f1(
            actual.get(
                "authors",
                [],
            ),
            expected.get(
                "authors",
                [],
            ),
        )

        return {
            "title_similarity": round(
                scores["title"],
                4,
            ),
            "year_accuracy": round(
                scores["year"],
                4,
            ),
            "doi_accuracy": round(
                scores["doi"],
                4,
            ),
            "author_f1": round(
                scores["authors"]["f1"],
                4,
            ),
            "score": sum(
                scores.values()
            ) / 4,
        }

    # =========================================================
    # ABSTRACT
    # =========================================================

    def _abstract_score(
        self,
        actual: str | None,
        expected: str | None,
    ) -> float:

        if not actual or not expected:
            return 0.0

        return self._similarity(
            actual,
            expected,
        )

    # =========================================================
    # LIST F1
    # =========================================================

    def _list_f1(
        self,
        actual: list[Any],
        expected: list[Any],
        *,
        formula: bool = False,
    ) -> dict[str, Any]:

        actual_norm = [
            self._normalize_formula(
                value
            )
            if formula
            else self._normalize(
                str(value)
            )
            for value in actual
            if str(value).strip()
        ]

        expected_norm = [
            self._normalize_formula(
                value
            )
            if formula
            else self._normalize(
                str(value)
            )
            for value in expected
            if str(value).strip()
        ]

        if not expected_norm:
            if not actual_norm:
                return {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                }

            return {
                "precision": 0.0,
                "recall": 1.0,
                "f1": 0.0,
            }

        matched_actual = set()
        matched_expected = set()

        for i, actual_value in enumerate(
            actual_norm
        ):

            best_index = None
            best_score = 0.0

            for j, expected_value in enumerate(
                expected_norm
            ):

                if j in matched_expected:
                    continue

                score = self._similarity(
                    actual_value,
                    expected_value,
                )

                threshold = (
                    0.75
                    if formula
                    else 0.85
                )

                if (
                    score >= threshold
                    and score > best_score
                ):
                    best_index = j
                    best_score = score

            if best_index is not None:
                matched_actual.add(i)
                matched_expected.add(
                    best_index
                )

        precision = (
            len(matched_actual)
            / len(actual_norm)
            if actual_norm
            else 0.0
        )

        recall = (
            len(matched_expected)
            / len(expected_norm)
        )

        f1 = self._f1(
            precision,
            recall,
        )

        return {
            "precision": round(
                precision,
                4,
            ),
            "recall": round(
                recall,
                4,
            ),
            "f1": round(
                f1,
                4,
            ),
        }

    # =========================================================
    # PARAGRAPH COVERAGE
    # =========================================================

    def _paragraph_coverage(
        self,
        actual: list[dict[str, Any]],
        expected: list[Any],
    ) -> dict[str, Any]:

        expected_texts = []

        for item in expected:
            if isinstance(
                item,
                dict,
            ):
                value = item.get(
                    "text",
                    "",
                )
            else:
                value = str(item)

            if value.strip():
                expected_texts.append(
                    value
                )

        actual_texts = [
            item.get(
                "text",
                "",
            )
            for item in actual
            if isinstance(item, dict)
        ]
        actual_texts = [
            text
            for text in actual_texts
            if str(text).strip()
        ]

        if not expected_texts:
            if not actual_texts:
                return {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "coverage": 1.0,
                }

            return {
                "precision": 0.0,
                "recall": 1.0,
                "f1": 0.0,
                "coverage": 1.0,
            }

        if not actual_texts:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "coverage": 0.0,
            }

        expected_matched = 0
        actual_matched = 0
        threshold = 0.80

        for expected_text in expected_texts:
            best = max(
                (
                    self._similarity(
                        actual_text,
                        expected_text,
                    )
                    for actual_text in actual_texts
                ),
                default=0.0,
            )

            if best >= threshold:
                expected_matched += 1

        for actual_text in actual_texts:
            best = max(
                (
                    self._similarity(
                        actual_text,
                        expected_text,
                    )
                    for expected_text in expected_texts
                ),
                default=0.0,
            )

            if best >= threshold:
                actual_matched += 1

        recall = (
            expected_matched
            / len(expected_texts)
        )

        precision = (
            actual_matched
            / len(actual_texts)
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": self._f1(
                precision,
                recall,
            ),
            "coverage": recall,
        }

    # =========================================================
    # REFERENCES
    # =========================================================

    def _reference_score(
        self,
        actual: list[dict[str, Any]],
        expected: list[Any],
    ) -> float:

        actual_texts = [
            self._reference_to_text(
                item
            )
            for item in actual
        ]

        expected_texts = [
            (
                self._reference_to_text(item)
                if isinstance(
                    item,
                    dict,
                )
                else str(item)
            )
            for item in expected
        ]

        return self._list_f1(
            actual_texts,
            expected_texts,
        )["f1"]

    # =========================================================
    # COUNT SCORE
    # =========================================================

    @staticmethod
    def _count_score(
        actual: int,
        expected: int,
    ) -> float:

        if expected <= 0:
            return 0.0

        difference = abs(
            actual - expected
        )

        return max(
            0.0,
            1.0
            - (
                difference
                / expected
            ),
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _f1(
        precision: float,
        recall: float,
    ) -> float:

        if (
            precision
            + recall
            == 0
        ):
            return 0.0

        return (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    @staticmethod
    def _normalize(
        value: Any,
    ) -> str:

        text = str(
            value or ""
        ).lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @staticmethod
    def _normalize_formula(
        value: Any,
    ) -> str:

        text = str(
            value or ""
        ).lower()

        text = text.replace(
            "\x08",
            "",
        )

        text = re.sub(
            r"\s+",
            "",
            text,
        )

        return text

    @staticmethod
    def _normalize_doi(
        value: Any,
    ) -> str:

        text = str(
            value or ""
        ).strip().lower()

        text = re.sub(
            r"^https?://doi.org/",
            "",
            text,
        )

        text = re.sub(
            r"^doi:",
            "",
            text,
        )

        return text

    @staticmethod
    def _similarity(
        a: Any,
        b: Any,
    ) -> float:

        a = ExtractionBenchmark._normalize(
            a
        )

        b = ExtractionBenchmark._normalize(
            b
        )

        if not a or not b:
            return 0.0

        from difflib import SequenceMatcher

        return SequenceMatcher(
            None,
            a,
            b,
        ).ratio()

    @staticmethod
    def _reference_to_text(
        reference: dict[str, Any],
    ) -> str:

        parts = [
            reference.get(
                "title",
                "",
            ),
            " ".join(
                reference.get(
                    "authors",
                    [],
                )
            ),
            reference.get(
                "year",
                "",
            ),
            reference.get(
                "doi",
                "",
            ),
        ]

        return " ".join(
            str(part)
            for part in parts
            if part
        )