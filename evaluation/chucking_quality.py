from __future__ import annotations

import re
from typing import Any


class ChunkingQualityEvaluator:
    """
    Evaluates structural quality of generated retrieval units.

    This does NOT measure retrieval accuracy.

    Retrieval metrics such as Hit@K, Recall@K, MRR and nDCG
    belong to the retrieval evaluation stage.
    """

    def evaluate(
        self,
        *,
        canonical: dict[str, Any],
        chunked: dict[str, Any],
    ) -> dict[str, Any]:

        units = self._records(
            chunked.get(
                "units",
                [],
            )
        )

        text_units = [
            unit
            for unit in units
            if unit.get(
                "unit_type"
            ) == "text"
        ]

        chunking_config = chunked.get(
            "chunking",
            {},
        )

        chunk_size = self._positive_int(
            chunking_config.get(
                "chunk_size"
            ),
            default=0,
        )

        metrics = {
            "source_coverage": (
                self.source_coverage(
                    canonical,
                    text_units,
                )
            ),
            "size_compliance": (
                self.size_compliance(
                    text_units,
                    chunk_size,
                )
                if chunk_size
                else 1.0
            ),
            "sentence_boundary_ratio": (
                self.sentence_boundary_ratio(
                    text_units
                )
            ),
            "section_integrity": (
                self.section_integrity(
                    text_units
                )
            ),
            "page_provenance": (
                self.page_provenance(
                    units
                )
            ),
            "metadata_integrity": (
                self.metadata_integrity(
                    units
                )
            ),
            "artifact_preservation": (
                self.artifact_preservation(
                    canonical,
                    units,
                )
            ),
            "duplicate_ratio": (
                self.duplicate_ratio(
                    text_units
                )
            ),
        }

        # duplicate_ratio is a diagnostic, not a positive-quality
        # component. More duplication should lower integrity.
        duplication_quality = max(
            0.0,
            1.0
            - metrics["duplicate_ratio"],
        )

        score = (
            0.24 * metrics[
                "source_coverage"
            ]
            + 0.14 * metrics[
                "size_compliance"
            ]
            + 0.14 * metrics[
                "sentence_boundary_ratio"
            ]
            + 0.14 * metrics[
                "section_integrity"
            ]
            + 0.10 * metrics[
                "page_provenance"
            ]
            + 0.10 * metrics[
                "metadata_integrity"
            ]
            + 0.10 * metrics[
                "artifact_preservation"
            ]
            + 0.04 * duplication_quality
        )

        issues: list[str] = []

        if metrics[
            "source_coverage"
        ] < 0.95:
            issues.append(
                "source_coverage_low"
            )

        if metrics[
            "size_compliance"
        ] < 0.95:
            issues.append(
                "size_compliance_low"
            )

        if metrics[
            "sentence_boundary_ratio"
        ] < 0.90:
            issues.append(
                "many_hard_splits"
            )

        if metrics[
            "section_integrity"
        ] < 1.0:
            issues.append(
                "section_metadata_incomplete"
            )

        if metrics[
            "page_provenance"
        ] < 0.95:
            issues.append(
                "page_provenance_incomplete"
            )

        if metrics[
            "metadata_integrity"
        ] < 0.95:
            issues.append(
                "unit_metadata_incomplete"
            )

        if metrics[
            "artifact_preservation"
        ] < 0.95:
            issues.append(
                "artifact_units_missing"
            )

        if metrics[
            "duplicate_ratio"
        ] > 0.30:
            issues.append(
                "high_text_duplication"
            )

        if score >= 0.90:
            status = "pass"
        elif score >= 0.75:
            status = "review"
        else:
            status = "fail"

        return {
            "evaluation_type": (
                "chunking_integrity"
            ),
            "integrity_score": round(
                score,
                4,
            ),
            "status": status,
            "chunking_config": {
                "chunk_size": (
                    chunk_size
                    or None
                ),
                "overlap": (
                    chunking_config.get(
                        "overlap"
                    )
                ),
                "tokenizer": (
                    chunking_config.get(
                        "tokenizer"
                    )
                ),
            },
            "metrics": {
                key: round(
                    value,
                    4,
                )
                for key, value
                in metrics.items()
            },
            "counts": {
                "total_units": len(units),
                "text_units": len(
                    text_units
                ),
                "figure_units": self._count_type(
                    units,
                    "figure",
                ),
                "table_units": self._count_type(
                    units,
                    "table",
                ),
                "formula_units": self._count_type(
                    units,
                    "formula",
                ),
                "reference_units": self._count_type(
                    units,
                    "reference",
                ),
            },
            "issues": issues,
        }

    # ============================================================
    # SOURCE COVERAGE
    # ============================================================

    @classmethod
    def source_coverage(
        cls,
        canonical: dict[str, Any],
        text_units: list[dict[str, Any]],
    ) -> float:

        paragraphs = [
            str(
                paragraph.get(
                    "text",
                    "",
                )
            ).strip()
            for paragraph in cls._records(
                canonical.get(
                    "paragraphs",
                    [],
                )
            )
            if str(
                paragraph.get(
                    "text",
                    "",
                )
            ).strip()
        ]

        if not paragraphs:
            return 1.0

        covered = 0

        for paragraph in paragraphs:

            normalized_paragraph = cls._normalize(
                paragraph
            )

            if not normalized_paragraph:
                continue

            # Check every chunk independently rather than joining the
            # complete corpus. Joining allows two unrelated chunks to
            # falsely satisfy a substring test.
            found = False

            for unit in text_units:

                unit_text = cls._normalize(
                    unit.get(
                        "text",
                        "",
                    )
                )

                if not unit_text:
                    continue

                if (
                    normalized_paragraph
                    in unit_text
                ):
                    found = True
                    break

                similarity = cls._coverage_similarity(
                    normalized_paragraph,
                    unit_text,
                )

                if similarity >= 0.90:
                    found = True
                    break

            if found:
                covered += 1

        return covered / len(
            paragraphs
        )

    # ============================================================
    # SIZE
    # ============================================================

    @staticmethod
    def size_compliance(
        text_units: list[dict[str, Any]],
        chunk_size: int,
    ) -> float:

        if not text_units:
            return 1.0

        if chunk_size <= 0:
            return 1.0

        valid = sum(
            1
            for unit in text_units
            if (
                unit.get(
                    "boundary_type"
                )
                in {
                    "sentence",
                    "paragraph",
                    "token",
                }
                and (
                    unit.get(
                        "boundary_type"
                    ) == "token"
                    or unit.get(
                        "token_count",
                        0,
                    ) <= chunk_size
                )
            )
        )

        return valid / len(
            text_units
        )

    # ============================================================
    # BOUNDARY
    # ============================================================

    @staticmethod
    def sentence_boundary_ratio(
        text_units: list[dict[str, Any]],
    ) -> float:

        if not text_units:
            return 1.0

        sentence_chunks = sum(
            1
            for unit in text_units
            if unit.get(
                "boundary_type"
            )
            == "sentence"
        )

        return (
            sentence_chunks
            / len(text_units)
        )

    # ============================================================
    # SECTION
    # ============================================================

    @staticmethod
    def section_integrity(
        text_units: list[dict[str, Any]],
    ) -> float:

        if not text_units:
            return 1.0

        valid = 0

        for unit in text_units:

            section = str(
                unit.get(
                    "section",
                    "",
                )
                or ""
            ).strip()

            section_path = unit.get(
                "section_path"
            )

            if (
                section
                and isinstance(
                    section_path,
                    list,
                )
                and section_path
            ):
                valid += 1

        return valid / len(
            text_units
        )

    # ============================================================
    # PAGE
    # ============================================================

    @staticmethod
    def page_provenance(
        units: list[dict[str, Any]],
    ) -> float:

        if not units:
            return 1.0

        valid = 0

        for unit in units:

            pages = unit.get(
                "pages"
            )

            page_start = unit.get(
                "page_start"
            )

            page = unit.get(
                "page"
            )

            coords = unit.get(
                "coords"
            )

            has_pages = (
                isinstance(
                    pages,
                    list,
                )
                and bool(pages)
            )

            has_page = (
                page is not None
                or page_start is not None
            )

            has_coords = (
                isinstance(
                    coords,
                    list,
                )
                and any(
                    isinstance(
                        coord,
                        dict,
                    )
                    and coord.get(
                        "page"
                    ) is not None
                    for coord in coords
                )
            )

            if (
                has_pages
                or has_page
                or has_coords
            ):
                valid += 1

        return valid / len(
            units
        )

    # ============================================================
    # METADATA
    # ============================================================

    @staticmethod
    def metadata_integrity(
        units: list[dict[str, Any]],
    ) -> float:

        if not units:
            return 1.0

        valid = 0

        for unit in units:

            unit_type = unit.get(
                "unit_type"
            )

            text = str(
                unit.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            section_ok = (
                unit_type != "text"
                or bool(
                    str(
                        unit.get(
                            "section",
                            "",
                        )
                        or ""
                    ).strip()
                )
            )

            source_ok = bool(
                unit.get(
                    "paper_id"
                )
            )

            if (
                unit_type
                and text
                and source_ok
                and section_ok
            ):
                valid += 1

        return valid / len(
            units
        )

    # ============================================================
    # ARTIFACT PRESERVATION
    # ============================================================

    @classmethod
    def artifact_preservation(
        cls,
        canonical: dict[str, Any],
        units: list[dict[str, Any]],
    ) -> float:

        expected_types = {
            "figure": len(
                cls._records(
                    canonical.get(
                        "figures"
                    )
                )
            ),
            "table": len(
                cls._records(
                    canonical.get(
                        "tables"
                    )
                )
            ),
            "formula": len(
                cls._records(
                    canonical.get(
                        "formulas"
                    )
                )
            ),
            "reference": len(
                cls._records(
                    canonical.get(
                        "references"
                    )
                )
            ),
        }

        actual_ids = {
            "figure": set(),
            "table": set(),
            "formula": set(),
            "reference": set(),
        }

        for unit in units:

            unit_type = unit.get(
                "unit_type"
            )

            if unit_type not in actual_ids:
                continue

            source_ref = (
                unit.get(
                    "source_ref"
                )
                or unit.get(
                    "artifact_id"
                )
                or unit.get(
                    "chunk_id"
                )
            )

            if source_ref:
                actual_ids[
                    unit_type
                ].add(
                    str(source_ref)
                )

        scores: list[float] = []

        for unit_type, expected_count in (
            expected_types.items()
        ):

            if expected_count == 0:
                continue

            canonical_records = cls._records(
                canonical.get(
                    (
                        "figures"
                        if unit_type == "figure"
                        else "tables"
                        if unit_type == "table"
                        else "formulas"
                        if unit_type == "formula"
                        else "references"
                    )
                )
            )

            canonical_refs = set()

            for record in canonical_records:

                ref = (
                    record.get(
                        "source_ref"
                    )
                    or record.get(
                        f"{unit_type}_id"
                    )
                    or record.get(
                        "id"
                    )
                )

                if ref:
                    canonical_refs.add(
                        str(ref)
                    )

            if canonical_refs:

                matched = len(
                    canonical_refs
                    & actual_ids[
                        unit_type
                    ]
                )

                scores.append(
                    min(
                        1.0,
                        matched
                        / len(
                            canonical_refs
                        ),
                    )
                )

            else:
                # Some canonical record types, especially references,
                # intentionally do not have source IDs. In that case
                # preservation is evaluated by count rather than by ID.
                actual_count = sum(
                    1
                    for unit in units
                    if unit.get(
                        "unit_type"
                    ) == unit_type
                )

                scores.append(
                    min(
                        1.0,
                        actual_count
                        / expected_count,
                    )
                )

        return (
            sum(scores)
            / len(scores)
            if scores
            else 1.0
        )

    # ============================================================
    # DUPLICATION
    # ============================================================

    @classmethod
    def duplicate_ratio(
        cls,
        text_units: list[dict[str, Any]],
    ) -> float:

        if len(text_units) < 2:
            return 0.0

        normalized = [
            cls._normalize(
                unit.get(
                    "text",
                    "",
                )
            )
            for unit in text_units
        ]

        normalized = [
            text
            for text in normalized
            if text
        ]

        if not normalized:
            return 0.0

        seen = set()
        duplicates = 0

        for text in normalized:

            if text in seen:
                duplicates += 1
            else:
                seen.add(text)

        return duplicates / len(
            normalized
        )

    # ============================================================
    # HELPERS
    # ============================================================

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
    def _count_type(
        units: list[dict[str, Any]],
        unit_type: str,
    ) -> int:

        return sum(
            1
            for unit in units
            if unit.get(
                "unit_type"
            ) == unit_type
        )

    @staticmethod
    def _positive_int(
        value: Any,
        default: int = 0,
    ) -> int:

        try:
            value = int(value)
            return (
                value
                if value > 0
                else default
            )
        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _normalize(
        text: Any,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            str(
                text or ""
            ).lower().strip(),
        )

    @classmethod
    def _coverage_similarity(
        cls,
        source: str,
        target: str,
    ) -> float:

        source_tokens = set(
            cls._tokens(source)
        )

        target_tokens = set(
            cls._tokens(target)
        )

        if not source_tokens:
            return 0.0

        return (
            len(
                source_tokens
                & target_tokens
            )
            / len(source_tokens)
        )

    @staticmethod
    def _tokens(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"[a-z0-9]+(?:['’-][a-z0-9]+)*",
            text.lower(),
        )
