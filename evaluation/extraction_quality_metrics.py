from __future__ import annotations

import re
from collections import Counter
from typing import Any


class QualityMetrics:
    """
    Extraction integrity evaluator.

    IMPORTANT:
    This class measures extraction health and internal consistency.
    It does NOT measure ground-truth extraction accuracy.

    Ground-truth accuracy is handled separately by:
        evaluation/extraction_benchmark.py

    This evaluator runs BEFORE canonicalization, so its
    extraction_counts describe the parser/artifact layer.
    The ingestion pipeline separately records canonical_counts
    after the final canonical representation is built.
    """

    # =========================================================
    # MAIN
    # =========================================================

    def evaluate(
        self,
        *,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:

        metrics = {
            "metadata_integrity": (
                self.metadata_integrity(
                    grobid
                )
            ),
            "abstract_integrity": (
                self.abstract_integrity(
                    grobid,
                    pymupdf,
                )
            ),
            "structure_integrity": (
                self.structure_integrity(
                    grobid
                )
            ),
            "duplication_integrity": (
                self.duplication_integrity(
                    grobid
                )
            ),
            "reference_integrity": (
                self.reference_integrity(
                    grobid
                )
            ),
            "truncation_integrity": (
                self.truncation_integrity(
                    grobid
                )
            ),
            "numeric_consistency": (
                self.numeric_consistency(
                    grobid,
                    pymupdf,
                    artifacts,
                )
            ),
            "cross_parser_consistency": (
                self.cross_parser_consistency(
                    grobid,
                    pymupdf,
                )
            ),
            "contamination_integrity": (
                self.contamination_integrity(
                    grobid,
                    pymupdf,
                    artifacts,
                )
            ),
            "coordinate_integrity": (
                self.coordinate_integrity(
                    artifacts
                )
            ),
        }

        integrity_score = (
            self._weighted_integrity_score(
                metrics
            )
        )

        issues = self._issues(
            metrics
        )

        return {
            "evaluation_type": "integrity",
            "integrity_score": round(
                integrity_score,
                4,
            ),
            "status": self._status(
                integrity_score
            ),
            "metrics": {
                key: round(
                    value,
                    4,
                )
                for key, value
                in metrics.items()
            },
            "orphan_paragraph_ratio": round(
                self.orphan_paragraph_ratio(
                    grobid
                ),
                4,
            ),
            "artifact_counts": (
                self._artifact_counts(
                    artifacts
                )
            ),
            "extraction_counts": (
                self._parser_counts(
                    grobid,
                    artifacts,
                )
            ),
            "ground_truth": {
                "available": False,
                "accuracy_score": None,
            },
            "issues": issues,
        }

    # =========================================================
    # STATUS
    # =========================================================

    @staticmethod
    def _status(
        score: float,
    ) -> str:

        if score >= 0.85:
            return "healthy"

        if score >= 0.70:
            return "review"

        return "poor"

    # =========================================================
    # METADATA
    # =========================================================

    @staticmethod
    def metadata_integrity(
        grobid: dict[str, Any],
    ) -> float:

        fields = {
            "title": bool(
                str(
                    grobid.get(
                        "title"
                    )
                    or ""
                ).strip()
            ),
            "authors": bool(
                grobid.get(
                    "authors"
                )
            ),
            "year": bool(
                grobid.get(
                    "year"
                )
            ),
            "doi": bool(
                str(
                    grobid.get(
                        "doi"
                    )
                    or ""
                ).strip()
            ),
        }

        # DOI can legitimately be absent in valid papers. Treat
        # title/authors/year as core metadata and DOI as optional.
        core = (
            sum(
                fields[
                    name
                ]
                for name in (
                    "title",
                    "authors",
                    "year",
                )
            )
            / 3
        )

        doi_component = (
            1.0
            if fields["doi"]
            else 0.5
        )

        return (
            0.85 * core
            + 0.15 * doi_component
        )

    # =========================================================
    # ABSTRACT
    # =========================================================

    @classmethod
    def abstract_integrity(
        cls,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
    ) -> float:

        abstract = cls._normalize(
            str(
                grobid.get(
                    "abstract"
                )
                or ""
            )
        )

        if not abstract:
            return 0.0

        pdf_text = cls._normalize(
            str(
                pymupdf.get(
                    "text"
                )
                or ""
            )
        )

        if not pdf_text:
            return cls._length_score(
                abstract
            )

        if abstract in pdf_text:
            return 1.0

        return cls._coverage(
            abstract,
            pdf_text,
        )

    # =========================================================
    # STRUCTURE
    # =========================================================

    @classmethod
    def structure_integrity(
        cls,
        grobid: dict[str, Any],
    ) -> float:

        sections = cls._records(
            grobid.get(
                "sections"
            )
        )

        paragraphs = cls._records(
            grobid.get(
                "paragraphs"
            )
        )

        if not sections:
            return 0.0

        valid_sections = sum(
            1
            for section in sections
            if (
                cls._clean_text(
                    section.get(
                        "heading"
                    )
                )
                and not cls._is_garbage_heading(
                    cls._clean_text(
                        section.get(
                            "heading"
                        )
                    )
                )
            )
        )

        section_validity = (
            valid_sections
            / len(sections)
        )

        if paragraphs:
            assigned = sum(
                1
                for paragraph in paragraphs
                if cls._clean_text(
                    paragraph.get(
                        "section"
                    )
                )
            )

            assignment_ratio = (
                assigned
                / len(paragraphs)
            )
        else:
            assignment_ratio = 0.0

        return (
            0.60 * section_validity
            + 0.40 * assignment_ratio
        )

    # =========================================================
    # DUPLICATION
    # =========================================================

    @classmethod
    def duplication_integrity(
        cls,
        grobid: dict[str, Any],
    ) -> float:

        paragraphs = cls._records(
            grobid.get(
                "paragraphs"
            )
        )

        texts = []

        for paragraph in paragraphs:

            text = cls._normalize(
                str(
                    paragraph.get(
                        "text"
                    )
                    or ""
                )
            )

            if len(text) >= 40:
                texts.append(text)

        if not texts:
            return 1.0

        counts = Counter(
            texts
        )

        duplicate_count = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        return max(
            0.0,
            1.0
            - (
                duplicate_count
                / len(texts)
            ),
        )

    # =========================================================
    # REFERENCES
    # =========================================================

    @classmethod
    def reference_integrity(
        cls,
        grobid: dict[str, Any],
    ) -> float:

        references = cls._records(
            grobid.get(
                "references"
            )
        )

        if not references:
            return 1.0

        valid = sum(
            1
            for reference in references
            if len(
                cls._reference_text(
                    reference
                )
            ) >= 20
        )

        return (
            valid
            / len(references)
        )

    @classmethod
    def _reference_text(
        cls,
        reference: dict[str, Any],
    ) -> str:

        parts = [
            reference.get(
                "title",
                "",
            ),
            reference.get(
                "raw",
                "",
            ),
            reference.get(
                "author",
                "",
            ),
            reference.get(
                "authors",
                "",
            ),
        ]

        return cls._clean_text(
            " ".join(
                str(part)
                for part in parts
                if part
            )
        )

    # =========================================================
    # TRUNCATION
    # =========================================================

    @classmethod
    def truncation_integrity(
        cls,
        grobid: dict[str, Any],
    ) -> float:

        paragraphs = cls._records(
            grobid.get(
                "paragraphs"
            )
        )

        if not paragraphs:
            return 1.0

        suspicious = 0
        checked = 0

        bad_endings = {
            "and",
            "or",
            "the",
            "of",
            "to",
            "for",
            "with",
            "a",
            "an",
            "by",
            "from",
            "in",
            "on",
            "using",
            "that",
        }

        for paragraph in paragraphs:

            text = cls._clean_text(
                paragraph.get(
                    "text",
                    "",
                )
            )

            if len(text) < 50:
                continue

            checked += 1

            words = re.sub(
                r"[^a-zA-Z]+$",
                "",
                text,
            ).split()

            if (
                words
                and words[-1].lower()
                in bad_endings
            ):

                suspicious += 1
                continue

            if text.endswith(
                (
                    ",",
                    ":",
                    ";",
                    "-",
                    "(",
                )
            ):
                suspicious += 1

        if checked == 0:
            return 1.0

        return max(
            0.0,
            1.0
            - (
                suspicious
                / checked
            ),
        )

    # =========================================================
    # NUMERIC CONSISTENCY
    # =========================================================

    @classmethod
    def numeric_consistency(
        cls,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
        artifacts: dict[str, Any] | None = None,
    ) -> float:

        grobid_text = cls._combined_body_text(
            grobid
        )

        if artifacts:
            artifact_text = cls._artifact_numeric_text(
                artifacts
            )
        else:
            artifact_text = ""

        pdf_text = cls._normalize(
            str(
                pymupdf.get(
                    "text"
                )
                or ""
            )
        )

        if not grobid_text or not pdf_text:
            return 0.0

        grobid_numbers = cls._extract_numbers(
            grobid_text
        )

        if artifact_text:
            grobid_numbers.extend(
                cls._extract_numbers(
                    artifact_text
                )
            )

        pdf_numbers = cls._extract_numbers(
            pdf_text
        )

        if not grobid_numbers:
            return 0.5

        g_counts = Counter(
            grobid_numbers
        )

        p_counts = Counter(
            pdf_numbers
        )

        matched = 0
        total = sum(
            g_counts.values()
        )

        for value, count in (
            g_counts.items()
        ):

            matched += min(
                count,
                p_counts.get(
                    value,
                    0,
                ),
            )

        return (
            matched
            / total
        )

    # =========================================================
    # CROSS-PARSER CONSISTENCY
    # =========================================================

    @classmethod
    def cross_parser_consistency(
        cls,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
    ) -> float:

        grobid_text = cls._combined_body_text(
            grobid
        )

        pdf_text = str(
            pymupdf.get(
                "text"
            )
            or ""
        )

        if not grobid_text or not pdf_text:
            return 0.0

        g_tokens = cls._tokenize(
            grobid_text
        )

        p_tokens = cls._tokenize(
            pdf_text
        )

        if not g_tokens or not p_tokens:
            return 0.0

        g_counts = Counter(
            g_tokens
        )

        p_counts = Counter(
            p_tokens
        )

        intersection = sum(
            (
                g_counts
                & p_counts
            ).values()
        )

        g_total = sum(
            g_counts.values()
        )

        p_total = sum(
            p_counts.values()
        )

        recall = (
            intersection
            / g_total
        )

        precision = (
            intersection
            / p_total
        )

        if (
            precision
            + recall
        ) == 0:
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

    # =========================================================
    # CONTAMINATION
    # =========================================================

    @classmethod
    def contamination_integrity(
        cls,
        grobid: dict[str, Any],
        pymupdf: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> float:

        paragraphs = cls._records(
            grobid.get(
                "paragraphs"
            )
        )

        if not paragraphs:
            return 1.0

        candidates = []

        for paragraph in paragraphs:

            text = cls._clean_text(
                paragraph.get(
                    "text",
                    "",
                )
            )

            if len(text) >= 20:
                candidates.append(text)

        if not candidates:
            return 1.0

        suspicious = 0

        normalized = [
            cls._normalize(
                text
            )
            for text in candidates
        ]

        frequency = Counter(
            normalized
        )

        for text, norm in zip(
            candidates,
            normalized,
        ):

            if (
                len(text) < 180
                and frequency[norm] >= 3
            ):

                suspicious += 1
                continue

            if cls._looks_like_header_footer(
                text
            ):
                suspicious += 1
                continue

            if cls._looks_like_email_block(
                text
            ):
                suspicious += 1
                continue

            # Only flag chart-like paragraphs when there is
            # no matching figure/table artifact nearby. This
            # avoids penalizing legitimate discussion of metrics.
            if cls._looks_like_chart_text(
                text
            ) and not cls._has_matching_artifact_text(
                text,
                artifacts,
            ):
                suspicious += 1
                continue

        ratio = (
            suspicious
            / len(candidates)
        )

        return max(
            0.0,
            1.0 - ratio,
        )

    @staticmethod
    def _looks_like_header_footer(
        text: str,
    ) -> bool:

        lowered = text.lower().strip()

        patterns = [
            r"^page\s+\d+$",
            r"^\d+\s+of\s+\d+$",
            r"copyright\s+\d{4}",
            r"©\s*\d{4}",
            r"open access",
            r"all rights reserved",
            r"published by",
            r"springer nature",
            r"scientific reports\s*\|",
            r"mdpi",
            r"proceedings of the .* conference",
        ]

        return any(
            re.search(
                pattern,
                lowered,
            )
            for pattern in patterns
        )

    @staticmethod
    def _looks_like_email_block(
        text: str,
    ) -> bool:

        email_count = len(
            re.findall(
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                text,
                flags=re.IGNORECASE,
            )
        )

        if email_count >= 2:
            return True

        lowered = text.lower()

        correspondence_terms = (
            "correspondence:",
            "corresponding author",
            "email:",
            "e-mail:",
        )

        return (
            email_count >= 1
            and any(
                term in lowered
                for term in correspondence_terms
            )
        )

    @staticmethod
    def _looks_like_chart_text(
        text: str,
    ) -> bool:

        words = text.split()

        if len(words) < 18:
            return False

        numeric_tokens = sum(
            1
            for word in words
            if re.fullmatch(
                r"[-+]?\d+(?:\.\d+)?%?",
                word.strip(
                    ",.;:()[]"
                ),
            )
        )

        numeric_ratio = (
            numeric_tokens
            / len(words)
        )

        chart_terms = {
            "accuracy",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
            "parameters",
            "top-1",
            "top1",
            "f1",
            "auc",
            "epoch",
            "epochs",
        }

        has_chart_term = any(
            term in text.lower()
            for term in chart_terms
        )

        return (
            numeric_ratio >= 0.35
            and has_chart_term
        )

    @classmethod
    def _has_matching_artifact_text(
        cls,
        text: str,
        artifacts: dict[str, Any],
    ) -> bool:

        normalized = cls._normalize(
            text
        )

        for key in (
            "figures",
            "tables",
        ):

            for item in cls._records(
                artifacts.get(
                    key
                )
            ):

                candidate = cls._normalize(
                    " ".join(
                        [
                            str(
                                item.get(
                                    "caption",
                                    ""
                                )
                                or ""
                            ),
                            str(
                                item.get(
                                    "text",
                                    ""
                                )
                                or ""
                            ),
                            str(
                                item.get(
                                    "content",
                                    ""
                                )
                                or ""
                            ),
                        ]
                    )
                )

                if candidate and (
                    normalized[:80]
                    in candidate
                    or candidate[:80]
                    in normalized
                ):
                    return True

        return False

    # =========================================================
    # ORPHAN PARAGRAPHS
    # =========================================================

    @staticmethod
    def orphan_paragraph_ratio(
        grobid: dict[str, Any],
    ) -> float:

        paragraphs = QualityMetrics._records(
            grobid.get(
                "paragraphs"
            )
        )

        if not paragraphs:
            return 0.0

        orphaned = sum(
            1
            for paragraph in paragraphs
            if not str(
                paragraph.get(
                    "section"
                )
                or ""
            ).strip()
        )

        return (
            orphaned
            / len(paragraphs)
        )

    # =========================================================
    # COORDINATE INTEGRITY
    # =========================================================

    @classmethod
    def coordinate_integrity(
        cls,
        artifacts: dict[str, Any],
    ) -> float:

        text_blocks = cls._records(
            artifacts.get(
                "text_blocks"
            )
        )

        if not text_blocks:
            return 0.0

        valid_blocks = 0

        for block in text_blocks:

            coords = block.get(
                "coords",
                [],
            )

            if (
                isinstance(
                    coords,
                    list,
                )
                and any(
                    cls._valid_coord_record(
                        coord
                    )
                    for coord in coords
                )
            ):
                valid_blocks += 1
                continue

            provenance = block.get(
                "prov",
                [],
            )

            if (
                isinstance(
                    provenance,
                    list,
                )
                and any(
                    cls._valid_provenance_record(
                        item
                    )
                    for item in provenance
                )
            ):
                valid_blocks += 1
                continue

            page = block.get(
                "page"
            )

            bbox = block.get(
                "bbox"
            )

            if (
                page is not None
                and isinstance(
                    bbox,
                    dict,
                )
                and cls._valid_bbox(
                    bbox
                )
            ):
                valid_blocks += 1

        return (
            valid_blocks
            / len(text_blocks)
        )

    @staticmethod
    def _valid_coord_record(
        coord: Any,
    ) -> bool:

        if not isinstance(
            coord,
            dict,
        ):
            return False

        page = coord.get(
            "page",
            coord.get(
                "page_no"
            ),
        )

        if page is None:
            return False

        try:
            x = float(
                coord.get(
                    "x"
                )
            )
            y = float(
                coord.get(
                    "y"
                )
            )
            w = float(
                coord.get(
                    "w",
                    coord.get(
                        "width"
                    ),
                )
            )
            h = float(
                coord.get(
                    "h",
                    coord.get(
                        "height"
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

        return (
            x >= 0
            and y >= 0
            and w > 0
            and h > 0
        )

    @classmethod
    def _valid_provenance_record(
        cls,
        item: Any,
    ) -> bool:

        if not isinstance(
            item,
            dict,
        ):
            return False

        page = item.get(
            "page_no",
            item.get(
                "page"
            ),
        )

        bbox = item.get(
            "bbox"
        )

        return (
            page is not None
            and isinstance(
                bbox,
                dict,
            )
            and cls._valid_bbox(
                bbox
            )
        )

    @staticmethod
    def _valid_bbox(
        bbox: dict[str, Any],
    ) -> bool:

        try:
            left = float(
                bbox["l"]
            )
            right = float(
                bbox["r"]
            )
            top = float(
                bbox["t"]
            )
            bottom = float(
                bbox["b"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

        return (
            right > left
            and top != bottom
        )

    # =========================================================
    # WEIGHTING
    # =========================================================

    @staticmethod
    def _weighted_integrity_score(
        metrics: dict[str, float],
    ) -> float:

        weights = {
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

        return sum(
            metrics[name]
            * weight
            for name, weight
            in weights.items()
        )

    # =========================================================
    # ISSUES
    # =========================================================

    @staticmethod
    def _issues(
        metrics: dict[str, float],
    ) -> list[str]:

        issues = []

        thresholds = {
            "metadata_integrity": (
                0.75,
                "metadata_integrity_low",
            ),
            "abstract_integrity": (
                0.75,
                "abstract_integrity_low",
            ),
            "structure_integrity": (
                0.80,
                "structure_integrity_low",
            ),
            "duplication_integrity": (
                0.90,
                "duplicate_text_detected",
            ),
            "reference_integrity": (
                0.80,
                "reference_integrity_low",
            ),
            "truncation_integrity": (
                0.85,
                "possible_truncation",
            ),
            "numeric_consistency": (
                0.80,
                "numeric_consistency_low",
            ),
            "cross_parser_consistency": (
                0.70,
                "cross_parser_disagreement",
            ),
            "contamination_integrity": (
                0.85,
                "possible_contamination",
            ),
            "coordinate_integrity": (
                0.90,
                "coordinate_integrity_low",
            ),
        }

        for metric, (
            threshold,
            issue,
        ) in thresholds.items():

            if metrics.get(
                metric,
                0.0,
            ) < threshold:
                issues.append(issue)

        return issues

    # =========================================================
    # COUNTS
    # =========================================================

    @staticmethod
    def _artifact_counts(
        artifacts: dict[str, Any],
    ) -> dict[str, int]:

        counts = {}

        for key in (
            "figures",
            "tables",
            "formulas",
            "text_blocks",
        ):

            value = artifacts.get(
                key,
                [],
            )

            counts[key] = (
                len(value)
                if isinstance(
                    value,
                    list,
                )
                else 0
            )

        provided = artifacts.get(
            "counts"
        )

        if isinstance(
            provided,
            dict,
        ):
            for key in counts:
                value = provided.get(
                    key
                )

                if isinstance(
                    value,
                    int,
                ):
                    counts[key] = value

        return counts

    @staticmethod
    def _parser_counts(
        grobid: dict[str, Any],
        artifacts: dict[str, Any],
    ) -> dict[str, int]:

        return {
            "sections": len(
                QualityMetrics._records(
                    grobid.get(
                        "sections"
                    )
                )
            ),
            "paragraphs": len(
                QualityMetrics._records(
                    grobid.get(
                        "paragraphs"
                    )
                )
            ),
            "references": len(
                QualityMetrics._records(
                    grobid.get(
                        "references"
                    )
                )
            ),
            # GROBID often has fewer/no figure/table structures in the
            # parser output. These are kept as parser-level counts.
            "figures": len(
                QualityMetrics._records(
                    grobid.get(
                        "figures"
                    )
                )
            ),
            "tables": len(
                QualityMetrics._records(
                    grobid.get(
                        "tables"
                    )
                )
            ),
            "formulas": len(
                QualityMetrics._records(
                    artifacts.get(
                        "formulas"
                    )
                )
            ),
        }

    # =========================================================
    # TEXT HELPERS
    # =========================================================

    @staticmethod
    def _combined_body_text(
        grobid: dict[str, Any],
    ) -> str:

        parts = []

        abstract = str(
            grobid.get(
                "abstract"
            )
            or ""
        ).strip()

        if abstract:
            parts.append(
                abstract
            )

        for paragraph in QualityMetrics._records(
            grobid.get(
                "paragraphs"
            )
        ):

            text = str(
                paragraph.get(
                    "text"
                )
                or ""
            ).strip()

            if text:
                parts.append(
                    text
                )

        return "\n".join(
            parts
        )

    @classmethod
    def _artifact_numeric_text(
        cls,
        artifacts: dict[str, Any],
    ) -> str:

        parts: list[str] = []

        for key in (
            "tables",
            "figures",
            "formulas",
        ):

            for item in cls._records(
                artifacts.get(
                    key
                )
            ):

                for field in (
                    "caption",
                    "text",
                    "content",
                    "orig",
                ):

                    value = item.get(
                        field
                    )

                    if value:
                        parts.append(
                            str(value)
                        )

        return " ".join(
            parts
        )

    @staticmethod
    def _extract_numbers(
        text: str,
    ) -> list[str]:

        pattern = (
            r"(?<![\w])"
            r"\d+(?:,\d{3})*"
            r"(?:\.\d+)?"
            r"(?:[eE][+-]?\d+)?"
            r"%?"
            r"(?![\w])"
        )

        return [
            value.replace(
                ",",
                "",
            )
            for value in re.findall(
                pattern,
                text,
            )
        ]

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"\b[\w%.-]+\b",
            text.lower(),
        )

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return re.sub(
            r"\s+",
            " ",
            str(value)
            .replace(
                "\xa0",
                " ",
            ),
        ).strip()

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            str(text or "")
            .lower(),
        ).strip()

    @classmethod
    def _coverage(
        cls,
        source: str,
        target: str,
    ) -> float:

        source_tokens = cls._tokenize(
            source
        )

        target_counts = Counter(
            cls._tokenize(
                target
            )
        )

        if not source_tokens:
            return 0.0

        matched = sum(
            1
            for token in source_tokens
            if target_counts.get(
                token,
                0,
            ) > 0
        )

        return (
            matched
            / len(
                source_tokens
            )
        )

    @staticmethod
    def _length_score(
        text: str,
    ) -> float:

        length = len(
            text.strip()
        )

        if length >= 500:
            return 1.0

        if length >= 300:
            return 0.95

        if length >= 200:
            return 0.85

        if length >= 100:
            return 0.70

        if length > 0:
            return 0.40

        return 0.0

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
    def _is_garbage_heading(
        heading: str,
    ) -> bool:

        heading = heading.strip()

        if not heading:
            return True

        if len(heading) > 180:
            return True

        if len(
            heading.split()
        ) > 20:
            return True

        if (
            len(
                heading.split()
            ) <= 10
            and re.search(
                r"[=^_{}]",
                heading,
            )
        ):
            return True

        return False
