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
        """Evaluate only dimensions that are actually present in the GT.

        This supports the current targeted GT format:
            metadata, abstract, figures, tables, counts, source_exceptions

        It also remains backward-compatible with the older full GT format
        containing sections, paragraphs, formulas and references.
        """
        gt_metadata = ground_truth.get("metadata") or {}

        metadata = self._metadata_score(
            canonical.get("metadata") or {},
            gt_metadata,
        )

        component_scores: dict[str, float] = {}
        metrics: dict[str, Any] = {"metadata": metadata}
        evaluated_dimensions: list[str] = ["metadata"]

        component_scores["metadata"] = metadata["score"]

        # Abstract lives at the canonical top level. Accept the old
        # metadata.abstract location as a fallback.
        if "abstract" in ground_truth or "abstract" in gt_metadata:
            actual_abstract = (
                canonical.get("abstract")
                or (canonical.get("metadata") or {}).get("abstract")
            )
            expected_abstract = (
                ground_truth.get("abstract")
                or gt_metadata.get("abstract")
            )
            abstract_similarity = self._abstract_score(
                actual_abstract,
                expected_abstract,
            )
            component_scores["abstract_similarity"] = abstract_similarity
            metrics["abstract_similarity"] = round(abstract_similarity, 4)
            evaluated_dimensions.append("abstract")

        # Full GT compatibility.
        if "sections" in ground_truth:
            sections = self._list_f1(
                [
                    item.get("heading", "")
                    for item in canonical.get("sections", [])
                ],
                ground_truth.get("sections", []),
            )
            component_scores["sections"] = sections["f1"]
            metrics["sections"] = sections
            evaluated_dimensions.append("sections")

        if "paragraphs" in ground_truth:
            paragraphs = self._paragraph_coverage(
                canonical.get("paragraphs", []),
                ground_truth.get("paragraphs", []),
            )
            component_scores["paragraphs"] = paragraphs["f1"]
            metrics["paragraphs"] = {
                key: round(paragraphs[key], 4)
                for key in ("precision", "recall", "f1", "coverage")
            }
            evaluated_dimensions.append("paragraphs")

        if "formulas" in ground_truth:
            formulas = self._list_f1(
                [
                    item.get("text", "")
                    for item in canonical.get("formulas", [])
                ],
                ground_truth.get("formulas", []),
                formula=True,
            )
            component_scores["formulas"] = formulas["f1"]
            metrics["formulas"] = formulas
            evaluated_dimensions.append("formulas")

        if "references" in ground_truth:
            references = self._reference_score(
                canonical.get("references", []),
                ground_truth.get("references", []),
            )
            component_scores["reference_score"] = references
            metrics["reference_score"] = round(references, 4)
            evaluated_dimensions.append("references")

        # Targeted artifact GT: compare count + label + caption. Do not use
        # parser-region counts as ground truth.
        if "tables" in ground_truth or "counts" in ground_truth:
            expected_tables = ground_truth.get("tables")
            if isinstance(expected_tables, list):
                table_result = self._artifact_score(
                    canonical.get("tables", []),
                    expected_tables,
                    artifact_type="table",
                )
            else:
                expected_count = self._expected_count(
                    ground_truth,
                    "tables",
                    "numbered_tables",
                )
                table_result = {
                    "score": self._count_score(
                        len(canonical.get("tables", [])),
                        expected_count,
                    ),
                    "count_score": self._count_score(
                        len(canonical.get("tables", [])),
                        expected_count,
                    ),
                }
            component_scores["table_count_score"] = table_result["score"]
            metrics["tables"] = table_result
            evaluated_dimensions.append("tables")

        if "figures" in ground_truth or "counts" in ground_truth:
            expected_figures = ground_truth.get("figures")
            if isinstance(expected_figures, list):
                figure_result = self._artifact_score(
                    canonical.get("figures", []),
                    expected_figures,
                    artifact_type="figure",
                )
            else:
                expected_count = self._expected_count(
                    ground_truth,
                    "figures",
                    "numbered_figures",
                )
                figure_result = {
                    "score": self._count_score(
                        len(canonical.get("figures", [])),
                        expected_count,
                    ),
                    "count_score": self._count_score(
                        len(canonical.get("figures", [])),
                        expected_count,
                    ),
                }
            component_scores["figure_count_score"] = figure_result["score"]
            metrics["figures"] = figure_result
            evaluated_dimensions.append("figures")

        overall = weighted_mean(component_scores, ACCURACY_WEIGHTS)

        return {
            "evaluation_type": "ground_truth_accuracy",
            "accuracy_score": round(overall or 0.0, 4),
            "scoring": {
                "weights": {
                    name: ACCURACY_WEIGHTS[name]
                    for name in component_scores
                    if name in ACCURACY_WEIGHTS
                },
                "evaluated_dimensions": evaluated_dimensions,
                "note": "Only GT fields explicitly supplied are scored.",
            },
            "metrics": metrics,
            "source_exceptions": ground_truth.get("source_exceptions", []),
        }

    def _metadata_score(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
    ) -> dict[str, Any]:
        """Score only metadata fields explicitly verified by the GT."""
        scores: dict[str, float] = {}

        if "title" in expected:
            scores["title"] = self._similarity(
                actual.get("title"),
                expected.get("title"),
            )

        if "year" in expected and expected.get("year") is not None:
            scores["year"] = (
                1.0 if actual.get("year") == expected.get("year") else 0.0
            )

        # A null GT DOI means "not verified/present", not a failed DOI match.
        if "doi" in expected and expected.get("doi") is not None:
            expected_doi = self._normalize_doi(expected.get("doi"))
            actual_doi = self._normalize_doi(actual.get("doi"))
            scores["doi"] = (
                1.0 if expected_doi and actual_doi == expected_doi else 0.0
            )

        if "authors" in expected:
            scores["authors"] = self._list_f1(
                actual.get("authors", []),
                expected.get("authors", []),
            )["f1"]

        score = sum(scores.values()) / len(scores) if scores else 0.0

        return {
            "title_similarity": round(scores.get("title", 0.0), 4) if "title" in scores else None,
            "year_accuracy": round(scores.get("year", 0.0), 4) if "year" in scores else None,
            "doi_accuracy": round(scores.get("doi", 0.0), 4) if "doi" in scores else None,
            "author_f1": round(scores.get("authors", 0.0), 4) if "authors" in scores else None,
            "score": round(score, 4),
            "evaluated_fields": list(scores),
        }

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
    # TARGETED ARTIFACT GT
    # =========================================================

    @staticmethod
    def _expected_count(
        ground_truth: dict[str, Any],
        records_key: str,
        count_key: str,
    ) -> int:
        counts = ground_truth.get("counts") or {}
        if count_key in counts:
            return int(counts[count_key])
        records = ground_truth.get(records_key)
        return len(records) if isinstance(records, list) else 0

    def _artifact_score(
        self,
        actual: list[dict[str, Any]],
        expected: list[dict[str, Any]],
        *,
        artifact_type: str,
    ) -> dict[str, Any]:
        actual = [item for item in actual if isinstance(item, dict)]
        expected = [item for item in expected if isinstance(item, dict)]

        count_score = self._count_score(len(actual), len(expected))

        if not expected:
            return {
                "artifact_type": artifact_type,
                "score": round(count_score, 4),
                "count_score": round(count_score, 4),
                "label_f1": 1.0 if not actual else 0.0,
                "caption_similarity": 1.0 if not actual else 0.0,
                "page_accuracy": 1.0 if not actual else None,
                "actual_count": len(actual),
                "expected_count": 0,
                "matched": 0,
                "missing_labels": [],
                "unexpected_labels": [item.get("label") for item in actual],
            }

        # First pass: match on normalized labels, which are the strongest
        # identity signal in the current GT.
        remaining_actual = set(range(len(actual)))
        remaining_expected = set(range(len(expected)))
        matches: list[tuple[int, int]] = []

        actual_labels = [self._normalize_label(x.get("label")) for x in actual]
        expected_labels = [self._normalize_label(x.get("label")) for x in expected]

        for j, expected_label in enumerate(expected_labels):
            if not expected_label:
                continue
            candidates = [
                i for i in remaining_actual
                if actual_labels[i] and actual_labels[i] == expected_label
            ]
            if candidates:
                i = candidates[0]
                matches.append((i, j))
                remaining_actual.remove(i)
                remaining_expected.remove(j)

        # Second pass: use caption similarity for any unmatched artifacts.
        pairs: list[tuple[float, int, int]] = []
        for i in remaining_actual:
            for j in remaining_expected:
                caption_score = self._similarity(
                    actual[i].get("caption", ""),
                    expected[j].get("caption", ""),
                )
                label_score = self._similarity(
                    actual_labels[i],
                    expected_labels[j],
                )
                pairs.append((0.7 * label_score + 0.3 * caption_score, i, j))

        for _, i, j in sorted(pairs, reverse=True):
            if i not in remaining_actual or j not in remaining_expected:
                continue
            label_score = self._similarity(actual_labels[i], expected_labels[j])
            caption_score = self._similarity(
                actual[i].get("caption", ""),
                expected[j].get("caption", ""),
            )
            if max(label_score, caption_score) >= 0.70:
                matches.append((i, j))
                remaining_actual.remove(i)
                remaining_expected.remove(j)

        matched = len(matches)
        label_precision = matched / len(actual) if actual else (1.0 if not expected else 0.0)
        label_recall = matched / len(expected) if expected else (1.0 if not actual else 0.0)
        label_f1 = self._f1(label_precision, label_recall)

        caption_scores: list[float] = []
        page_matches = 0
        page_evaluable = 0
        for i, j in matches:
            expected_caption = expected[j].get("caption", "")
            actual_caption = actual[i].get("caption", "")
            if str(expected_caption).strip():
                caption_scores.append(self._similarity(actual_caption, expected_caption))

            actual_page = actual[i].get("page")
            if actual_page is None and isinstance(actual[i].get("pages"), list) and actual[i].get("pages"):
                actual_page = actual[i]["pages"][0]
            expected_page = expected[j].get("page")
            if expected_page is not None and actual_page is not None:
                page_evaluable += 1
                if actual_page == expected_page:
                    page_matches += 1

        caption_similarity = (
            sum(caption_scores) / len(caption_scores)
            if caption_scores else 1.0
        )
        page_accuracy = (
            page_matches / page_evaluable
            if page_evaluable else None
        )

        # Count remains important, but label identity gets the strongest weight.
        score = 0.40 * count_score + 0.40 * label_f1 + 0.20 * caption_similarity

        return {
            "artifact_type": artifact_type,
            "score": round(score, 4),
            "count_score": round(count_score, 4),
            "label_f1": round(label_f1, 4),
            "caption_similarity": round(caption_similarity, 4),
            "page_accuracy": round(page_accuracy, 4) if page_accuracy is not None else None,
            "actual_count": len(actual),
            "expected_count": len(expected),
            "matched": matched,
            "missing_labels": [
                expected[j].get("label") for j in sorted(remaining_expected)
            ],
            "unexpected_labels": [
                actual[i].get("label") for i in sorted(remaining_actual)
            ],
        }

    @staticmethod
    def _normalize_label(value: Any) -> str:
        text = str(value or "").lower().strip()
        text = re.sub(r"^fig\.?\s*", "figure ", text)
        text = re.sub(r"^tab\.?\s*", "table ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # =========================================================
    # COUNT SCORE
    # =========================================================

    @staticmethod
    @staticmethod
    def _count_score(
        actual: int,
        expected: int,
    ) -> float:
        """Exact for zero; otherwise linearly penalize count differences."""
        if expected < 0:
            raise ValueError("expected count cannot be negative")

        if expected == 0:
            return 1.0 if actual == 0 else 0.0

        difference = abs(actual - expected)
        return max(0.0, 1.0 - (difference / expected))

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