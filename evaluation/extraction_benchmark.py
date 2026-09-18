from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


class ExtractionBenchmark:
    """
    Compare canonical extraction against manually verified
    ground-truth annotations.

    This is the ONLY evaluator allowed to report extraction
    accuracy.
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
                item.get(
                    "heading",
                    "",
                )
                for item in self._records(
                    canonical.get(
                        "sections"
                    )
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
                item.get(
                    "text",
                    "",
                )
                for item in self._records(
                    canonical.get(
                        "formulas"
                    )
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
                self._records(
                    canonical.get(
                        "tables"
                    )
                )
            ),
            self._safe_int(
                ground_truth.get(
                    "table_count",
                    0,
                )
            ),
        )

        figures = self._count_score(
            len(
                self._records(
                    canonical.get(
                        "figures"
                    )
                )
            ),
            self._safe_int(
                ground_truth.get(
                    "figure_count",
                    0,
                )
            ),
        )

        # Canonical schema stores abstract at top level.
        # Keep a metadata fallback for older ground-truth files.
        actual_abstract = (
            canonical.get(
                "abstract"
            )
            or canonical.get(
                "metadata",
                {},
            ).get(
                "abstract"
            )
        )

        expected_metadata = ground_truth.get(
            "metadata",
            {},
        )

        expected_abstract = (
            ground_truth.get(
                "abstract"
            )
            or expected_metadata.get(
                "abstract"
            )
        )

        abstract_similarity = self._abstract_score(
            actual_abstract,
            expected_abstract,
        )

        table_structure = self._table_structure_score(
            canonical,
            ground_truth,
        )

        figure_provenance = self._artifact_provenance_score(
            canonical.get(
                "figures",
                [],
            ),
            ground_truth.get(
                "figures",
                [],
            ),
            count_key="figure_count",
        )

        formula_provenance = self._artifact_provenance_score(
            canonical.get(
                "formulas",
                [],
            ),
            ground_truth.get(
                "formulas",
                [],
            ),
            count_key=None,
        )

        overall = (
            metadata["score"] * 0.18
            + sections["f1"] * 0.14
            + paragraphs["coverage"] * 0.18
            + formulas["f1"] * 0.14
            + references * 0.08
            + tables * 0.05
            + figures * 0.05
            + abstract_similarity * 0.10
            + table_structure * 0.04
            + figure_provenance * 0.02
            + formula_provenance * 0.02
        )

        return {
            "evaluation_type": (
                "ground_truth_accuracy"
            ),
            "accuracy_score": round(
                overall,
                4,
            ),
            "metrics": {
                "metadata": metadata,
                "sections": sections,
                "paragraph_coverage": round(
                    paragraphs["coverage"],
                    4,
                ),
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
                "table_structure_score": round(
                    table_structure,
                    4,
                ),
                "figure_provenance_score": round(
                    figure_provenance,
                    4,
                ),
                "formula_provenance_score": round(
                    formula_provenance,
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

        title_score = self._similarity(
            actual.get("title"),
            expected.get("title"),
        )

        year_score = (
            1.0
            if actual.get("year")
            == expected.get("year")
            and expected.get("year") is not None
            else 0.0
        )

        expected_doi = expected.get(
            "doi"
        )

        doi_score = (
            1.0
            if (
                expected_doi
                and self._normalize_doi(
                    actual.get("doi")
                )
                == self._normalize_doi(
                    expected_doi
                )
            )
            else 0.0
        )

        author_result = self._list_f1(
            actual.get(
                "authors",
                [],
            ),
            expected.get(
                "authors",
                [],
            ),
        )

        values = [
            title_score,
            year_score,
            doi_score,
            author_result["f1"],
        ]

        return {
            "title_similarity": round(
                title_score,
                4,
            ),
            "year_accuracy": round(
                year_score,
                4,
            ),
            "doi_accuracy": round(
                doi_score,
                4,
            ),
            "author_precision": round(
                author_result["precision"],
                4,
            ),
            "author_recall": round(
                author_result["recall"],
                4,
            ),
            "author_f1": round(
                author_result["f1"],
                4,
            ),
            "score": round(
                sum(values)
                / len(values),
                4,
            ),
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
            (
                self._normalize_formula(value)
                if formula
                else self._normalize(
                    value
                )
            )
            for value in actual
            if str(
                value or ""
            ).strip()
        ]

        expected_norm = [
            (
                self._normalize_formula(value)
                if formula
                else self._normalize(
                    value
                )
            )
            for value in expected
            if str(
                value or ""
            ).strip()
        ]

        if not expected_norm:
            return {
                "precision": 0.0,
                "recall": 0.0,
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
                    normalized=True,
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
                self._f1(
                    precision,
                    recall,
                ),
                4,
            ),
        }

    # =========================================================
    # PARAGRAPHS
    # =========================================================

    def _paragraph_coverage(
        self,
        actual: list[dict[str, Any]],
        expected: list[Any],
    ) -> dict[str, Any]:

        expected_texts = []

        for item in expected:

            value = (
                item.get(
                    "text",
                    "",
                )
                if isinstance(
                    item,
                    dict,
                )
                else str(item)
            )

            if value.strip():
                expected_texts.append(
                    value
                )

        if not expected_texts:
            return {
                "coverage": 0.0
            }

        actual_texts = [
            item.get(
                "text",
                "",
            )
            for item in self._records(
                actual
            )
        ]

        matched = 0

        for expected_text in expected_texts:

            best = max(
                (
                    self._similarity(
                        actual_text,
                        expected_text,
                    )
                    for actual_text in actual_texts
                    if actual_text
                ),
                default=0.0,
            )

            if best >= 0.80:
                matched += 1

        return {
            "coverage": (
                matched
                / len(
                    expected_texts
                )
            )
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
            for item in self._records(
                actual
            )
        ]

        expected_texts = [
            (
                self._reference_to_text(
                    item
                )
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
    # TABLE STRUCTURE
    # =========================================================

    def _table_structure_score(
        self,
        canonical: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> float:

        expected_tables = ground_truth.get(
            "tables"
        )

        actual_tables = self._records(
            canonical.get(
                "tables"
            )
        )

        if not isinstance(
            expected_tables,
            list,
        ):

            # No structural annotations means this metric
            # is not evaluable from ground truth.
            return 1.0

        if not expected_tables:
            return 1.0 if not actual_tables else 0.0

        matched = 0

        for expected in expected_tables:

            expected_rows = (
                expected.get(
                    "rows"
                )
                if isinstance(
                    expected,
                    dict,
                )
                else None
            )

            expected_cells = (
                expected.get(
                    "cells"
                )
                if isinstance(
                    expected,
                    dict,
                )
                else None
            )

            best = 0.0

            for actual in actual_tables:

                actual_rows = actual.get(
                    "rows",
                    [],
                )

                actual_cells = actual.get(
                    "cells",
                    [],
                )

                row_score = (
                    self._structure_similarity(
                        actual_rows,
                        expected_rows,
                    )
                    if expected_rows is not None
                    else 1.0
                )

                cell_score = (
                    self._structure_similarity(
                        actual_cells,
                        expected_cells,
                    )
                    if expected_cells is not None
                    else 1.0
                )

                best = max(
                    best,
                    (
                        row_score
                        + cell_score
                    )
                    / 2,
                )

            if best >= 0.80:
                matched += 1

        return matched / len(
            expected_tables
        )

    def _structure_similarity(
        self,
        actual: Any,
        expected: Any,
    ) -> float:

        if expected is None:
            return 1.0

        actual_text = self._normalize(
            self._flatten_structure(
                actual
            )
        )

        expected_text = self._normalize(
            self._flatten_structure(
                expected
            )
        )

        if not actual_text or not expected_text:
            return 0.0

        return self._similarity(
            actual_text,
            expected_text,
            normalized=True,
        )

    @classmethod
    def _flatten_structure(
        cls,
        value: Any,
    ) -> str:

        if isinstance(
            value,
            dict,
        ):

            parts = []

            for key in (
                "text",
                "value",
                "content",
            ):

                if key in value:
                    parts.append(
                        str(
                            value[key]
                            or ""
                        )
                    )

            if not parts:

                for nested in value.values():
                    parts.append(
                        cls._flatten_structure(
                            nested
                        )
                    )

            return " ".join(parts)

        if isinstance(
            value,
            list,
        ):

            return " ".join(
                cls._flatten_structure(
                    item
                )
                for item in value
            )

        return str(
            value or ""
        )

    # =========================================================
    # ARTIFACT PROVENANCE
    # =========================================================

    def _artifact_provenance_score(
        self,
        actual: list[Any],
        expected: list[Any],
        *,
        count_key: str | None,
    ) -> float:

        actual_records = self._records(
            actual
        )

        if count_key:
            expected_count = (
                self._safe_int(
                    (
                        expected.get(
                            count_key,
                            0,
                        )
                        if isinstance(
                            expected,
                            dict,
                        )
                        else 0
                    )
                )
            )

            if expected_count <= 0:
                return 1.0

        expected_list = (
            expected
            if isinstance(
                expected,
                list,
            )
            else []
        )

        if not expected_list:
            return 1.0

        expected_provenance = sum(
            1
            for item in expected_list
            if isinstance(
                item,
                dict,
            )
            and (
                item.get(
                    "page"
                ) is not None
                or item.get(
                    "coords"
                )
            )
        )

        actual_provenance = sum(
            1
            for item in actual_records
            if (
                item.get(
                    "page"
                ) is not None
                or item.get(
                    "coords"
                )
            )
        )

        if expected_provenance == 0:
            return 1.0

        return min(
            1.0,
            actual_provenance
            / expected_provenance,
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _records(
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

    @staticmethod
    def _count_score(
        actual: int,
        expected: int,
    ) -> float:

        if expected <= 0:
            return (
                1.0
                if actual == 0
                else 0.0
            )

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
            r"^https?://doi\.org/",
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
        *,
        normalized: bool = False,
    ) -> float:

        if normalized:
            left = str(
                a or ""
            )
            right = str(
                b or ""
            )
        else:
            left = ExtractionBenchmark._normalize(
                a
            )
            right = ExtractionBenchmark._normalize(
                b
            )

        if not left or not right:
            return 0.0

        return SequenceMatcher(
            None,
            left,
            right,
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
                if isinstance(
                    reference.get(
                        "authors",
                        [],
                    ),
                    list,
                )
                else [
                    str(
                        reference.get(
                            "authors"
                        )
                    )
                ]
            ),
            reference.get(
                "year",
                "",
            ),
            reference.get(
                "doi",
                "",
            ),
            reference.get(
                "raw",
                "",
            ),
        ]

        return " ".join(
            str(part)
            for part in parts
            if part
        )

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> int:

        try:
            return int(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0
