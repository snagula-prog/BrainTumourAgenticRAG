from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any


BODY_TYPES = {
    "body",
}

SOURCE_TEXT_TYPES = {
    "body",
    "abstract",
    "author_bio",
    "table_caption",
    "figure_caption",
}

PROVENANCE_TYPES = {
    "body",
    "author_bio",
    "table_caption",
    "figure_caption",
}


class ChunkingEvaluator:
    """
    Evaluate structural integrity of the current chunk representation.

    This evaluator does NOT measure retrieval accuracy.

    Retrieval metrics such as Recall@K, Precision@K, MRR and nDCG
    belong to the later retrieval evaluation stage.
    """

    def evaluate(
        self,
        *,
        paper_id: str,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
        max_words: int = 300,
    ) -> dict[str, Any]:

        if not isinstance(canonical, dict):
            raise TypeError(
                "canonical must be a dictionary"
            )

        if not isinstance(chunks, list):
            raise TypeError(
                "chunks must be a list"
            )

        valid_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict)
        ]

        metrics = {
            "paragraph_preservation": (
                self.paragraph_preservation(
                    canonical,
                    valid_chunks,
                )
            ),
            "source_character_coverage": (
                self.source_character_coverage(
                    canonical,
                    valid_chunks,
                )
            ),
            "section_integrity": (
                self.section_integrity(
                    canonical,
                    valid_chunks,
                )
            ),
            "size_compliance": (
                self.size_compliance(
                    valid_chunks,
                    max_words,
                )
            ),
            "heading_coverage": (
                self.heading_coverage(
                    canonical,
                    valid_chunks,
                )
            ),
            "provenance_completeness": (
                self.provenance_completeness(
                    canonical,
                    valid_chunks,
                )
            ),
            "subsection_id_integrity": (
                self.subsection_id_integrity(
                    paper_id,
                    valid_chunks,
                )
            ),
            "artifact_caption_coverage": (
                self.artifact_caption_coverage(
                    canonical,
                    valid_chunks,
                )
            ),
            "reference_coverage": (
                self.reference_coverage(
                    canonical,
                    valid_chunks,
                )
            ),
        }

        weights = {
            "paragraph_preservation": 0.20,
            "source_character_coverage": 0.20,
            "section_integrity": 0.15,
            "size_compliance": 0.10,
            "heading_coverage": 0.05,
            "provenance_completeness": 0.10,
            "subsection_id_integrity": 0.05,
            "artifact_caption_coverage": 0.10,
            "reference_coverage": 0.05,
        }

        score = self._weighted_mean(
            metrics,
            weights,
        )

        diagnostics = self._diagnostics(
            canonical,
            valid_chunks,
            max_words,
        )

        issues = self._issues(metrics)

        return {
            "evaluation_type": (
                "chunking_integrity"
            ),

            "paper_id": paper_id,

            "integrity_score": round(
                score,
                4,
            ),

            "status": self._status(score),

            "scoring": {
                "weights": weights,
            },

            "metrics": {
                key: round(
                    value,
                    4,
                )
                for key, value in metrics.items()
            },

            "counts": self._counts(
                canonical,
                valid_chunks,
            ),

            "diagnostics": diagnostics,

            "issues": issues,

            "notes": [
                (
                    "Chunking integrity measures preservation "
                    "and structural correctness of retrieval units."
                ),
                (
                    "It does not measure retrieval accuracy."
                ),
                (
                    "Retrieval Recall@K, Precision@K, MRR and nDCG "
                    "belong to the retrieval evaluation stage."
                ),
            ],
        }

    # ============================================================
    # STATUS
    # ============================================================

    @staticmethod
    def _status(score: float) -> str:
        if score >= 0.90:
            return "pass"

        if score >= 0.75:
            return "review"

        return "poor"

    # ============================================================
    # PARAGRAPH PRESERVATION
    # ============================================================

    @classmethod
    def paragraph_preservation(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> float:

        paragraphs = cls._records(
            canonical.get("paragraphs")
        )

        expected_ids = []

        for paragraph in paragraphs:
            section = str(
                paragraph.get("section")
                or ""
            ).strip().casefold()

            # Author biographies are emitted separately.
            if section == "author biographies":
                continue

            paragraph_id = paragraph.get(
                "paragraph_id"
            )

            if paragraph_id:
                expected_ids.append(
                    str(paragraph_id)
                )

        if not expected_ids:
            return 1.0

        occurrences: Counter[str] = Counter()

        for chunk in chunks:
            if chunk.get("chunk_type") != "body":
                continue

            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            ids = metadata.get(
                "paragraph_ids",
                [],
            )

            if not isinstance(ids, list):
                continue

            for paragraph_id in ids:
                if paragraph_id:
                    occurrences[
                        str(paragraph_id)
                    ] += 1

        correct = sum(
            1
            for paragraph_id in expected_ids
            if occurrences[paragraph_id] == 1
        )

        return correct / len(expected_ids)

    # ============================================================
    # CHARACTER COVERAGE
    # ============================================================

    @classmethod
    def source_character_coverage(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> float:

        expected_texts: list[str] = []

        abstract = canonical.get(
            "abstract"
        )

        if abstract:
            expected_texts.append(
                str(abstract)
            )

        biographies = cls._records(
            canonical.get(
                "author_biographies"
            )
        )

        for biography in biographies:
            text = str(
                biography.get("text")
                or ""
            ).strip()

            if text:
                expected_texts.append(
                    text
                )

        paragraphs = cls._records(
            canonical.get("paragraphs")
        )

        for paragraph in paragraphs:
            section = str(
                paragraph.get("section")
                or ""
            ).strip().casefold()

            if section == "author biographies":
                continue

            text = str(
                paragraph.get("text")
                or ""
            ).strip()

            if text:
                expected_texts.append(
                    text
                )

        tables = cls._records(
            canonical.get("tables")
        )

        for table in tables:
            caption = str(
                table.get("caption")
                or ""
            ).strip()

            if caption:
                expected_texts.append(
                    caption
                )

        figures = cls._records(
            canonical.get("figures")
        )

        for figure in figures:
            caption = str(
                figure.get("caption")
                or ""
            ).strip()

            if caption:
                expected_texts.append(
                    caption
                )

        expected_chars = cls._non_ws_chars(
            expected_texts
        )

        actual_texts: list[str] = []

        for chunk in chunks:
            if chunk.get(
                "chunk_type"
            ) not in SOURCE_TEXT_TYPES:
                continue

            text = str(
                chunk.get("text")
                or ""
            )

            if chunk.get(
                "chunk_type"
            ) == "index_terms":
                text = re.sub(
                    r"^\[Index Terms\]\s*",
                    "",
                    text,
                    flags=re.IGNORECASE,
                )

            actual_texts.append(
                text
            )

        actual_chars = cls._non_ws_chars(
            actual_texts
        )

        if expected_chars == 0:
            return 1.0

        return min(
            actual_chars / expected_chars,
            1.0,
        )

    # ============================================================
    # SECTION INTEGRITY
    # ============================================================

    @classmethod
    def section_integrity(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> float:

        paragraphs = cls._records(
            canonical.get("paragraphs")
        )

        paragraph_paths = {}

        for paragraph in paragraphs:
            section = str(
                paragraph.get("section")
                or ""
            ).strip().casefold()

            if section == "author biographies":
                continue

            paragraph_id = paragraph.get(
                "paragraph_id"
            )

            if not paragraph_id:
                continue

            path = cls._section_path(
                paragraph
            )

            paragraph_paths[
                str(paragraph_id)
            ] = path

        checked = 0
        valid = 0

        for chunk in chunks:
            if chunk.get(
                "chunk_type"
            ) != "body":
                continue

            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            chunk_path = tuple(
                cls._clean_path(
                    metadata.get(
                        "section_path"
                    )
                )
            )

            ids = metadata.get(
                "paragraph_ids",
                [],
            )

            if not ids:
                continue

            checked += 1

            paths = {
                tuple(
                    paragraph_paths[
                        str(paragraph_id)
                    ]
                )
                for paragraph_id in ids
                if str(
                    paragraph_id
                ) in paragraph_paths
            }

            if len(paths) == 1 and chunk_path in paths:
                valid += 1

        if checked == 0:
            return 1.0

        return valid / checked

    # ============================================================
    # SIZE COMPLIANCE
    # ============================================================

    @staticmethod
    def size_compliance(
        chunks: list[dict[str, Any]],
        max_words: int,
    ) -> float:

        body_chunks = [
            chunk
            for chunk in chunks
            if chunk.get(
                "chunk_type"
            ) == "body"
        ]

        if not body_chunks:
            return 1.0

        valid = 0

        for chunk in body_chunks:
            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            oversized = bool(
                metadata.get(
                    "oversized"
                )
            )

            word_count = int(
                chunk.get(
                    "word_count",
                    len(
                        str(
                            chunk.get(
                                "text"
                            )
                            or ""
                        ).split()
                    ),
                )
            )

            if (
                oversized
                or word_count <= max_words
            ):
                valid += 1

        return valid / len(body_chunks)

    # ============================================================
    # HEADING COVERAGE
    # ============================================================

    @classmethod
    def heading_coverage(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> float:

        sections = cls._records(
            canonical.get("sections")
        )

        required_paths: set[
            tuple[str, ...]
        ] = set()

        for section in sections:
            path = cls._section_path(section)

            if not path:
                continue

            # The canonical section itself is the
            # source of truth for heading existence.
            required_paths.add(
                tuple(path)
            )

        if not required_paths:
            return 1.0

        actual_paths: set[
            tuple[str, ...]
        ] = set()

        for chunk in chunks:
            if chunk.get(
                "chunk_type"
            ) != "heading":
                continue

            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            path = tuple(
                cls._clean_path(
                    metadata.get(
                        "section_path"
                    )
                )
            )

            if path:
                actual_paths.add(path)

        covered = len(
            required_paths
            & actual_paths
        )

        return covered / len(
            required_paths
        )

    # ============================================================
    # PROVENANCE
    # ============================================================

    @classmethod
    def provenance_completeness(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> float:

        relevant = [
            chunk
            for chunk in chunks
            if chunk.get(
                "chunk_type"
            ) in PROVENANCE_TYPES
        ]

        if not relevant:
            return 1.0

        paragraphs = {
            str(
                item.get("paragraph_id")
            ): item
            for item in cls._records(
                canonical.get("paragraphs")
            )
            if item.get("paragraph_id")
        }

        biographies = cls._records(
            canonical.get(
                "author_biographies"
            )
        )

        tables = {
            str(
                item.get("table_id")
            ): item
            for item in cls._records(
                canonical.get("tables")
            )
            if item.get("table_id")
        }

        figures = {
            str(
                item.get("figure_id")
            ): item
            for item in cls._records(
                canonical.get("figures")
            )
            if item.get("figure_id")
        }

        valid = 0

        for chunk in relevant:

            chunk_type = chunk.get(
                "chunk_type"
            )

            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            # --------------------------------------------------
            # BODY
            # --------------------------------------------------

            if chunk_type == "body":

                paragraph_ids = metadata.get(
                    "paragraph_ids",
                    [],
                )

                if not paragraph_ids:
                    continue

                source_records = [
                    paragraphs.get(
                        str(paragraph_id)
                    )
                    for paragraph_id in paragraph_ids
                ]

                source_records = [
                    item
                    for item in source_records
                    if isinstance(item, dict)
                ]

                if not source_records:
                    continue

                # Identity is always required.
                identity_ok = True

                # Page provenance is required only when the
                # canonical source provides it.
                source_has_page = any(
                    item.get("page") is not None
                    or item.get("pages")
                    for item in source_records
                )

                chunk_has_page = bool(
                    metadata.get("page")
                    or metadata.get("pages")
                )

                page_ok = (
                    chunk_has_page
                    if source_has_page
                    else True
                )

                # Coordinate provenance is similarly conditional.
                source_has_coords = any(
                    item.get("coords")
                    or item.get("coordinates")
                    for item in source_records
                )

                chunk_has_coords = bool(
                    metadata.get("coordinates")
                )

                coords_ok = (
                    chunk_has_coords
                    if source_has_coords
                    else True
                )

                if identity_ok and page_ok and coords_ok:
                    valid += 1

            # --------------------------------------------------
            # AUTHOR BIO
            # --------------------------------------------------

            elif chunk_type == "author_bio":

                author = str(
                    metadata.get("author")
                    or ""
                ).strip()

                if not author:
                    continue

                valid += 1

            # --------------------------------------------------
            # TABLE CAPTION
            # --------------------------------------------------

            elif chunk_type == "table_caption":

                table_id = metadata.get(
                    "table_id"
                )

                source = tables.get(
                    str(table_id)
                )

                if not source:
                    continue

                source_has_page = bool(
                    source.get("page")
                    or source.get("pages")
                )

                chunk_has_page = bool(
                    metadata.get("page")
                    or metadata.get("pages")
                )

                if source_has_page and not chunk_has_page:
                    continue

                valid += 1

            # --------------------------------------------------
            # FIGURE CAPTION
            # --------------------------------------------------

            elif chunk_type == "figure_caption":

                figure_id = metadata.get(
                    "figure_id"
                )

                source = figures.get(
                    str(figure_id)
                )

                if not source:
                    continue

                source_has_page = bool(
                    source.get("page")
                    or source.get("pages")
                )

                chunk_has_page = bool(
                    metadata.get("page")
                    or metadata.get("pages")
                )

                if source_has_page and not chunk_has_page:
                    continue

                valid += 1

        return valid / len(relevant)

    # ============================================================
    # SUBSECTION ID
    # ============================================================

    @staticmethod
    def subsection_id_integrity(
        paper_id: str,
        chunks: list[dict[str, Any]],
    ) -> float:

        relevant = [
            chunk
            for chunk in chunks
            if chunk.get(
                "chunk_type"
            ) in {
                "body",
                "heading",
            }
        ]

        if not relevant:
            return 1.0

        valid = 0

        for chunk in relevant:
            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            path = tuple(
                ChunkingEvaluator._clean_path(
                    metadata.get(
                        "section_path"
                    )
                )
            )

            subsection_id = metadata.get(
                "subsection_id"
            )

            # Sectionless chunks legitimately have
            # no subsection ID.
            if not path:
                if subsection_id is None:
                    valid += 1
                continue

            payload = "\x1f".join(path)

            digest = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()[:12]

            expected = (
                f"{paper_id}_subsection_"
                f"{digest}"
            )

            if subsection_id == expected:
                valid += 1

        return valid / len(relevant)

    # ============================================================
    # ARTIFACT CAPTIONS
    # ============================================================

    @classmethod
    def artifact_caption_coverage(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> float:

        expected_ids: set[str] = set()

        for table in cls._records(
            canonical.get("tables")
        ):
            caption = str(
                table.get("caption")
                or ""
            ).strip()

            table_id = table.get(
                "table_id"
            )

            if caption and table_id:
                expected_ids.add(
                    str(table_id)
                )

        for figure in cls._records(
            canonical.get("figures")
        ):
            caption = str(
                figure.get("caption")
                or ""
            ).strip()

            figure_id = figure.get(
                "figure_id"
            )

            if caption and figure_id:
                expected_ids.add(
                    str(figure_id)
                )

        if not expected_ids:
            return 1.0

        actual_ids: set[str] = set()

        for chunk in chunks:
            chunk_type = chunk.get(
                "chunk_type"
            )

            if chunk_type not in {
                "table_caption",
                "figure_caption",
            }:
                continue

            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            artifact_id = (
                metadata.get(
                    "table_id"
                )
                if chunk_type
                == "table_caption"
                else metadata.get(
                    "figure_id"
                )
            )

            if artifact_id:
                actual_ids.add(
                    str(artifact_id)
                )

        return len(
            expected_ids
            & actual_ids
        ) / len(
            expected_ids
        )

    # ============================================================
    # REFERENCE COVERAGE
    # ============================================================

    @classmethod
    def reference_coverage(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> float:

        references = cls._records(
            canonical.get("references")
        )

        if not references:
            return 1.0

        expected: set[int] = set()

        for position, reference in enumerate(
            references,
            start=1,
        ):
            number = reference.get(
                "number"
            )

            if isinstance(number, int):
                expected.add(number)
            else:
                expected.add(position)

        actual: set[int] = set()

        for chunk in chunks:
            if chunk.get(
                "chunk_type"
            ) != "reference":
                continue

            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            numbers = metadata.get(
                "reference_numbers",
                [],
            )

            if not isinstance(
                numbers,
                list,
            ):
                continue

            for number in numbers:
                if isinstance(
                    number,
                    int,
                ):
                    actual.add(number)

        return len(
            expected & actual
        ) / len(expected)

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    @classmethod
    def _diagnostics(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
        max_words: int,
    ) -> dict[str, Any]:

        body_chunks = [
            chunk
            for chunk in chunks
            if chunk.get(
                "chunk_type"
            ) == "body"
        ]

        sizes = [
            int(
                chunk.get(
                    "word_count",
                    0,
                )
            )
            for chunk in body_chunks
        ]

        normal_sizes = [
            size
            for chunk, size in zip(
                body_chunks,
                sizes,
            )
            if not bool(
                isinstance(
                    chunk.get(
                        "metadata"
                    ),
                    dict,
                )
                and chunk.get(
                    "metadata"
                ).get(
                    "oversized"
                )
            )
        ]

        short_chunks = []

        for index, chunk in enumerate(
            body_chunks
        ):
            metadata = chunk.get(
                "metadata"
            )

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            word_count = int(
                chunk.get(
                    "word_count",
                    0,
                )
            )

            if (
                word_count >= max_words * 0.25
            ):
                continue

            next_body = (
                body_chunks[index + 1]
                if index + 1 < len(body_chunks)
                else None
            )

            if next_body is None:
                reason = "document_end"
            else:
                current_path = tuple(
                    cls._clean_path(
                        metadata.get(
                            "section_path"
                        )
                    )
                )

                next_metadata = (
                    next_body.get(
                        "metadata"
                    )
                )

                next_path = tuple(
                    cls._clean_path(
                        next_metadata.get(
                            "section_path"
                        )
                    )
                ) if isinstance(
                    next_metadata,
                    dict,
                ) else ()

                if current_path != next_path:
                    reason = (
                        "section_boundary"
                    )
                elif (
                    isinstance(
                        next_metadata,
                        dict,
                    )
                    and next_metadata.get(
                        "oversized"
                    )
                ):
                    reason = (
                        "oversized_paragraph_boundary"
                    )
                else:
                    reason = (
                        "normal_flush"
                    )

            short_chunks.append(
                {
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),
                    "word_count": word_count,
                    "reason": reason,
                }
            )

        text_hashes = Counter()

        for chunk in chunks:
            normalized = cls._normalize(
                str(
                    chunk.get(
                        "text"
                    )
                    or ""
                )
            )

            if normalized:
                text_hashes[
                    normalized
                ] += 1

        duplicate_chunks = sum(
            count - 1
            for count in text_hashes.values()
            if count > 1
        )

        return {
            "body_chunk_size": {
                "count": len(
                    body_chunks
                ),
                "mean_words": (
                    round(
                        mean(sizes),
                        2,
                    )
                    if sizes
                    else 0
                ),
                "median_words": (
                    round(
                        median(sizes),
                        2,
                    )
                    if sizes
                    else 0
                ),
                "max_words": (
                    max(sizes)
                    if sizes
                    else 0
                ),
                "min_words": (
                    min(sizes)
                    if sizes
                    else 0
                ),
                "normal_chunk_max_words": (
                    max(normal_sizes)
                    if normal_sizes
                    else 0
                ),
            },

            "short_body_chunks": {
                "count": len(
                    short_chunks
                ),
                "items": short_chunks,
            },

            "duplicate_chunk_text_count": (
                duplicate_chunks
            ),

            "artifact_leakage": (
                cls._artifact_leakage(
                    canonical,
                    chunks,
                )
            ),
        }

    # ============================================================
    # ISSUES
    # ============================================================

    @staticmethod
    def _issues(
        metrics: dict[str, float],
    ) -> list[str]:

        issues: list[str] = []

        if metrics[
            "paragraph_preservation"
        ] < 1.0:
            issues.append(
                "paragraph_preservation_incomplete"
            )

        if metrics[
            "source_character_coverage"
        ] < 0.99:
            issues.append(
                "source_character_coverage_low"
            )

        if metrics[
            "section_integrity"
        ] < 1.0:
            issues.append(
                "section_boundary_violation"
            )

        if metrics[
            "size_compliance"
        ] < 1.0:
            issues.append(
                "body_chunk_size_violation"
            )

        if metrics[
            "heading_coverage"
        ] < 1.0:
            issues.append(
                "heading_coverage_incomplete"
            )

        if metrics[
            "provenance_completeness"
        ] < 0.95:
            issues.append(
                "provenance_incomplete"
            )

        if metrics[
            "subsection_id_integrity"
        ] < 1.0:
            issues.append(
                "subsection_id_invalid"
            )

        if metrics[
            "artifact_caption_coverage"
        ] < 1.0:
            issues.append(
                "artifact_caption_coverage_incomplete"
            )

        if metrics[
            "reference_coverage"
        ] < 1.0:
            issues.append(
                "reference_coverage_incomplete"
            )

        return issues

    # ============================================================
    # ARTIFACT LEAKAGE
    # ============================================================

    @classmethod
    def _artifact_leakage(
        cls,
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:

        non_artifact_text = " ".join(
            cls._normalize(
                str(
                    chunk.get("text")
                    or ""
                )
            )
            for chunk in chunks
            if chunk.get(
                "chunk_type"
            )
            not in {
                "table_caption",
                "figure_caption",
                "reference",
                "formula",
            }
        )

        leaked_tables = []

        for table in cls._records(
            canonical.get("tables")
        ):
            content = cls._normalize(
                str(
                    table.get(
                        "content"
                    )
                    or ""
                )
            )

            if len(content) < 40:
                continue

            if content in non_artifact_text:
                leaked_tables.append(
                    table.get(
                        "table_id"
                    )
                )

        return {
            "table_content_exact_leaks": (
                len(leaked_tables)
            ),
            "table_ids": leaked_tables,
        }

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
    def _clean_path(
        value: Any,
    ) -> list[str]:

        if not isinstance(
            value,
            list,
        ):
            return []

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    @classmethod
    def _section_path(
        cls,
        source: dict[str, Any],
    ) -> list[str]:

        path = cls._clean_path(
            source.get(
                "section_path"
            )
        )

        if path:
            return path

        section = str(
            source.get(
                "section"
            )
            or ""
        ).strip()

        return (
            [section]
            if section
            else []
        )

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip().casefold()

    @staticmethod
    def _non_ws_chars(
        texts: list[str],
    ) -> int:

        return sum(
            len(
                re.sub(
                    r"\s+",
                    "",
                    text,
                )
            )
            for text in texts
        )

    @staticmethod
    def _weighted_mean(
        metrics: dict[str, float],
        weights: dict[str, float],
    ) -> float:

        return sum(
            metrics[name] * weight
            for name, weight in weights.items()
        )

    @staticmethod
    def _counts(
        canonical: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:

        chunk_counts = Counter(
            str(
                chunk.get(
                    "chunk_type"
                )
                or "unknown"
            )
            for chunk in chunks
        )

        return {
            "canonical_paragraphs": len(
                ChunkingEvaluator._records(
                    canonical.get(
                        "paragraphs"
                    )
                )
            ),
            "canonical_tables": len(
                ChunkingEvaluator._records(
                    canonical.get(
                        "tables"
                    )
                )
            ),
            "canonical_figures": len(
                ChunkingEvaluator._records(
                    canonical.get(
                        "figures"
                    )
                )
            ),
            "canonical_references": len(
                ChunkingEvaluator._records(
                    canonical.get(
                        "references"
                    )
                )
            ),
            "canonical_sections": len(
                ChunkingEvaluator._records(
                    canonical.get(
                        "sections"
                    )
                )
            ),
            "total_chunks": len(
                chunks
            ),
            "chunks_by_type": dict(
                chunk_counts
            ),
        }