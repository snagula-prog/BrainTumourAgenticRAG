from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .evaluation_config import (
    DIAGNOSTIC_INTEGRITY_METRICS,
    INTEGRITY_ISSUE_THRESHOLDS,
    PRIMARY_INTEGRITY_WEIGHTS,
    weighted_mean,
)


class QualityMetrics:
    """
    Extraction integrity evaluator.

    IMPORTANT:
    This class measures extraction health and internal consistency.
    It does NOT measure ground-truth extraction accuracy.

    Ground-truth accuracy is handled separately by:
        evaluation/extraction_benchmark.py
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
            "metadata_integrity": self.metadata_integrity(
                grobid
            ),
            "abstract_integrity": self.abstract_integrity(
                grobid,
                pymupdf,
            ),
            "structure_integrity": self.structure_integrity(
                grobid
            ),
            "duplication_integrity": self.duplication_integrity(
                grobid
            ),
            "reference_integrity": self.reference_integrity(
                grobid
            ),
            "truncation_integrity": self.truncation_integrity(
                grobid
            ),
            "numeric_consistency": self.numeric_consistency(
                grobid,
                pymupdf,
            ),
            "cross_parser_consistency": self.cross_parser_consistency(
                grobid,
                pymupdf,
            ),
            "contamination_integrity": self.contamination_integrity(
                grobid,
                pymupdf,
                artifacts,
            ),
            "coordinate_integrity": self.coordinate_integrity(
                artifacts
            ),
        }

        integrity_score = self._weighted_integrity_score(
            metrics
        )

        issues = self._issues(
            metrics
        )

        diagnostics = {
            "orphan_paragraph_ratio": round(
                self.orphan_paragraph_ratio(
                    grobid
                ),
                4,
            ),
            **{
                name: round(
                    metrics[name],
                    4,
                )
                for name in DIAGNOSTIC_INTEGRITY_METRICS
                if name in metrics
            },
        }

        return {
            "evaluation_type": "integrity",
            "integrity_score": round(
                integrity_score,
                4,
            ),
            "status": self._status(
                integrity_score,
                issues,
            ),
            "scoring": {
                "primary_metrics": dict(
                    PRIMARY_INTEGRITY_WEIGHTS
                ),
                "diagnostic_metrics": sorted(
                    DIAGNOSTIC_INTEGRITY_METRICS
                ),
            },
            "metrics": {
                key: round(
                    value,
                    4,
                )
                for key, value in metrics.items()
            },
            "diagnostics": diagnostics,
            "parser_artifact_counts": self._artifact_counts(
                artifacts
            ),
            "raw_extraction_counts": {
                "sections": len(
                    grobid.get(
                        "sections",
                        [],
                    )
                ),
                "paragraphs": len(
                    grobid.get(
                        "paragraphs",
                        [],
                    )
                ),
                "references": len(
                    grobid.get(
                        "references",
                        [],
                    )
                ),
                "figures": len(
                    grobid.get(
                        "figures",
                        [],
                    )
                ),
                "tables": len(
                    grobid.get(
                        "tables",
                        [],
                    )
                ),
                "formulas": len(
                    artifacts.get(
                        "formulas",
                        [],
                    )
                ),
            },
            "ground_truth_available": False,
            "issues": issues,
        }

    # =========================================================
    # STATUS
    # =========================================================

    @staticmethod
    def _status(
        score: float,
        issues: list[str],
    ) -> str:

        if score < 0.70:
            return "poor"

        if score < 0.85:
            return "review"

        if issues:
            return "healthy_with_warnings"

        return "healthy"

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

        return (
            sum(
                fields.values()
            )
            / len(fields)
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

        sections = grobid.get(
            "sections",
            [],
        )

        paragraphs = grobid.get(
            "paragraphs",
            [],
        )

        if not sections:
            return 0.0

        valid_sections = 0

        for section in sections:

            heading = cls._clean_text(
                section.get(
                    "heading"
                )
            )

            if not heading:
                continue

            if cls._is_garbage_heading(
                heading
            ):
                continue

            valid_sections += 1

        section_validity = (
            valid_sections
            / len(sections)
        )

        if paragraphs:
            assigned = sum(
                1
                for paragraph in paragraphs
                if str(
                    paragraph.get(
                        "section"
                    )
                    or ""
                ).strip()
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

        paragraphs = grobid.get(
            "paragraphs",
            [],
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
                texts.append(
                    text
                )

        if not texts:
            return 0.0

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

        references = grobid.get(
            "references",
            [],
        )

        if not references:
            return 0.0

        valid = 0

        for reference in references:

            text = cls._reference_text(
                reference
            )

            if len(text) >= 20:
                valid += 1

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

        paragraphs = grobid.get(
            "paragraphs",
            [],
        )

        if not paragraphs:
            return 0.0

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
            return 0.0

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
    ) -> float:

        grobid_text = (
            cls._combined_body_text(
                grobid
            )
        )

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

        pdf_numbers = cls._extract_numbers(
            pdf_text
        )

        # No numbers means the metric is not applicable.
        #
        # We return None conceptually, but the evaluator keeps
        # numeric metrics numeric. A neutral score is preferable
        # to claiming perfection.
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

        grobid_text = (
            cls._combined_body_text(
                grobid
            )
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
        """
        Detect likely contamination using evidence from:

        1. Explicit parser structure
        2. Repeated short boilerplate text
        3. Known header/footer patterns
        4. Figure/chart-like text

        This does NOT penalize a paragraph merely because its exact
        normalized text differs from PyMuPDF output.
        """

        paragraphs = grobid.get(
            "paragraphs",
            [],
        )

        if not paragraphs:
            return 0.0

        candidates = []

        for paragraph in paragraphs:

            text = cls._clean_text(
                paragraph.get(
                    "text",
                    "",
                )
            )

            if len(text) < 20:
                continue

            candidates.append(
                text
            )

        if not candidates:
            return 0.0

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

            # -------------------------------------------------
            # Repeated boilerplate
            # -------------------------------------------------

            if (
                len(text) < 180
                and frequency[norm] >= 3
            ):
                suspicious += 1
                continue

            # -------------------------------------------------
            # Explicit header/footer patterns
            # -------------------------------------------------

            if cls._looks_like_header_footer(
                text
            ):
                suspicious += 1
                continue

            # -------------------------------------------------
            # Email / correspondence blocks
            # -------------------------------------------------

            if cls._looks_like_email_block(
                text
            ):
                suspicious += 1
                continue

            # -------------------------------------------------
            # Figure/chart text
            # -------------------------------------------------

            if cls._looks_like_chart_text(
                text
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

    # =========================================================
    # HEADER / FOOTER
    # =========================================================

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

    # =========================================================
    # EMAIL BLOCK
    # =========================================================

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

    # =========================================================
    # CHART / FIGURE TEXT
    # =========================================================

    @staticmethod
    def _looks_like_chart_text(
        text: str,
    ) -> bool:

        words = text.split()

        if len(words) < 12:
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
            numeric_ratio >= 0.25
            and has_chart_term
        )

    # =========================================================
    # ORPHAN PARAGRAPHS
    # =========================================================

    @staticmethod
    def orphan_paragraph_ratio(
        grobid: dict[str, Any],
    ) -> float:

        paragraphs = grobid.get(
            "paragraphs",
            [],
        )

        if not paragraphs:
            return 1.0

        orphaned = sum(
            1
            for paragraph
            in paragraphs
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
        """
        Validate Docling text-block coordinate coverage.

        The current DoclingArtifactExtractor / canonical representation
        uses coordinate records like:

        {
            "page": 13,
            "x": 306.14,
            "y": 276.38,
            "w": 30.56,
            "h": 9.69
        }

        We also accept older/raw provenance representations.
        """

        text_blocks = artifacts.get(
            "text_blocks",
            [],
        )

        if not text_blocks:
            return 0.0

        valid_blocks = 0

        for block in text_blocks:

            coords = block.get(
                "coords",
                [],
            )

            if isinstance(
                coords,
                list,
            ) and any(
                cls._valid_coord_record(
                    coord
                )
                for coord in coords
            ):
                valid_blocks += 1
                continue

            # Raw Docling provenance fallback.
            provenance = block.get(
                "prov",
                [],
            )

            if isinstance(
                provenance,
                list,
            ) and any(
                cls._valid_provenance_record(
                    item
                )
                for item in provenance
            ):
                valid_blocks += 1
                continue

            # Flattened page + bbox fallback.
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
            and top > bottom
        )

    # =========================================================
    # WEIGHTING
    # =========================================================

    @staticmethod
    def _weighted_integrity_score(
        metrics: dict[str, float],
    ) -> float:

        score = weighted_mean(
            metrics,
            PRIMARY_INTEGRITY_WEIGHTS,
        )

        return (
            score
            if score is not None
            else 0.0
        )

    # =========================================================
    # ISSUES
    # =========================================================

    @staticmethod
    def _issues(
        metrics: dict[str, float],
    ) -> list[str]:

        issues = []

        for metric_name, threshold in INTEGRITY_ISSUE_THRESHOLDS.items():
            value = metrics.get(metric_name)

            if value is None:
                continue

            if value < threshold:
                issue_name = metric_name

                if metric_name == "duplication_integrity":
                    issue_name = "duplicate_text_detected"
                elif metric_name == "truncation_integrity":
                    issue_name = "possible_truncation"
                elif metric_name == "cross_parser_consistency":
                    issue_name = "cross_parser_disagreement"
                elif metric_name == "contamination_integrity":
                    issue_name = "possible_contamination"
                else:
                    issue_name = f"{metric_name}_low"

                issues.append(issue_name)

        return issues

    # =========================================================
    # ARTIFACT COUNTS
    # =========================================================

    @staticmethod
    def _artifact_counts(
        artifacts: dict[str, Any],
    ) -> dict[str, Any]:

        counts = dict(
            artifacts.get(
                "counts",
                {},
            )
        )

        # Ensure these are always exposed.
        counts.setdefault(
            "figures",
            len(
                artifacts.get(
                    "figures",
                    [],
                )
            ),
        )

        counts.setdefault(
            "tables",
            len(
                artifacts.get(
                    "tables",
                    [],
                )
            ),
        )

        counts.setdefault(
            "formulas",
            len(
                artifacts.get(
                    "formulas",
                    [],
                )
            ),
        )

        counts.setdefault(
            "text_blocks",
            len(
                artifacts.get(
                    "text_blocks",
                    [],
                )
            ),
        )

        return counts

    # =========================================================
    # HELPERS
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

        for paragraph in grobid.get(
            "paragraphs",
            [],
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
            str(text)
            .lower(),
        ).strip()

    @staticmethod
    def _coverage(
        source: str,
        target: str,
    ) -> float:

        source_tokens = QualityMetrics._tokenize(
            source
        )

        target_counts = Counter(
            QualityMetrics._tokenize(
                target
            )
        )

        if not source_tokens:
            return 0.0

        matched = sum(
            1
            for token
            in source_tokens
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
    def _is_garbage_heading(
        heading: str,
    ) -> bool:

        heading = heading.strip()

        if not heading:
            return True

        if len(
            heading
        ) > 180:
            return True

        if len(
            heading.split()
        ) > 20:
            return True

        # Formula-like headings are almost certainly not
        # real section headings.
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